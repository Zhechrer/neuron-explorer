"""Checks experiment runs (several neurons pushed at once) on the live model.

Each case generates from one runtime.steer_multi batch, as /api/steer does for an experiment: row 0 carries every push,
row 1 is the unsteered baseline, row 2 reverses each push. Read-only probes sit on every pushed layer's down_proj input,
one just before the explorer's own hook and one just after, so each forward pass shows every pushed neuron's value with
and without the push. A push counts as applied in a pass when, for every token and row, the value after the hook matches
the value before it moved by exactly that row's push (add: + dose in residual-norm units; mul: x dose), within bf16
rounding; row 1 must come through unchanged. The case passes when every push is applied in every forward pass (so they
are all active in the same passes), the hook state is cleared afterwards, and no weight of the pushed layers changed.

    uv run python scripts/experiment_check.py                       # cases A, B and C
    EXPLORER_QUANT=4bit uv run --extra gpu python scripts/experiment_check.py --cases C --max-new 40

Stop the explorer server first: this loads its own copy of the model.
"""
import argparse
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
from runtime import Runtime  # noqa: E402

DOG = "Tell me a short story about a dog that gets lost."
CASES = {  # name: (pushes [(layer, n, dose, mode)], prompt)
    "A": ([(31, 9036, -0.5, "add")], DOG),
    "B": ([(31, 9036, 0.0, "mul")], DOG),
    "C": ([(14, 8949, 1.0, "add"), (13, 7956, -1.0, "add")], "hey qwen, how's it going?"),
}
RTOL = 2 ** -7  # bf16 keeps 8 significant bits: one rounding of the push and one of the sum, with room to spare
ATOL = 1e-3


def weights_digest(rt, layers):
    """Hash of the stored bytes of every parameter in the pushed layers' MLPs (packed bytes when quantized)."""
    h = hashlib.sha256()
    for l in sorted(layers):
        for p in rt.lm.layers[l].mlp.parameters():
            h.update(p.detach().contiguous().flatten().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class Probe:
    """Each pushed neuron's down_proj input just before and just after the explorer's hook, per forward pass."""

    def __init__(self, rt, pushes):
        self.step, self.records, self.handles = -1, [], []
        self.handles.append(rt.lm.register_forward_pre_hook(self._tick))
        for l in sorted({p[0] for p in pushes}):
            ns = [p[1] for p in pushes if p[0] == l]
            dp = rt.lm.layers[l].mlp.down_proj
            self.handles.append(dp.register_forward_pre_hook(self._before(l, ns), prepend=True))
            self.handles.append(dp.register_forward_pre_hook(self._after(l, ns)))
        self._pre = {}

    def _tick(self, mod, args, kwargs=None):
        self.step += 1

    def _before(self, l, ns):
        def hook(mod, inp):
            self._pre[l] = inp[0][:, :, ns].float().clone()
        return hook

    def _after(self, l, ns):
        def hook(mod, inp):
            self.records.append((self.step, l, ns, self._pre.pop(l), inp[0][:, :, ns].float().clone()))
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def check(rt, name, pushes, prompt, max_new):
    label = ", ".join(f"L{l} N{n} {m} {d:+g}" for l, n, d, m in pushes)
    print(f"\n== case {name}: {label}  |  prompt: {prompt!r}")
    # the per-row values the explorer's hook applies, from its own helper: add in residual-norm units, mul a factor
    vals = {(l, n): rt._push_entry(l, n, d, m, (1, 0, -1))[3].float() for l, n, d, m in pushes}
    modes = {(l, n): m for l, n, _, m in pushes}
    before = weights_digest(rt, {p[0] for p in pushes})

    probe = Probe(rt, pushes)
    texts = ["", "", ""]
    try:
        for row, delta in rt.steer_multi(pushes, [[{"role": "user", "content": prompt}]] * 3, max_new):
            if row is not None:
                texts[row] += delta
    finally:
        probe.remove()

    passes = probe.step + 1
    applied = {k: set() for k in vals}  # forward passes where the push was exact in rows 0 and 2 and row 1 untouched
    worst = {k: 0.0 for k in vals}
    act = {k: [0.0, 0.0] for k in vals}  # the experiment row's largest |activation| before and after the hook
    for step, l, ns, pre, post in probe.records:
        for j, n in enumerate(ns):
            v = vals[(l, n)].to(pre.device)[:, None]
            a, b = pre[:, :, j], post[:, :, j]
            want = a + v if modes[(l, n)] == "add" else a * v
            err = (b - want).abs()
            tol = RTOL * (want.abs() + v.abs()) + ATOL
            worst[(l, n)] = max(worst[(l, n)], float((err / tol).max()))
            act[(l, n)] = [max(act[(l, n)][0], float(a[0].abs().max())), max(act[(l, n)][1], float(b[0].abs().max()))]
            if bool((err <= tol).all()):
                applied[(l, n)].add(step)

    ok = True
    for (l, n), steps in applied.items():
        fmt = (lambda x: f"x{x:.4g}") if modes[(l, n)] == "mul" else (lambda x: f"{x:+.4g}")
        v = [fmt(x) for x in vals[(l, n)].tolist()]
        good = len(steps) == passes
        ok &= good
        print(f"  L{l} N{n}: applied in {len(steps)}/{passes} forward passes {'OK' if good else 'FAIL'}"
              f"  (rows: experiment {v[0]}, baseline {v[1]}, reversed {v[2]}; worst error {worst[(l, n)]:.2f} of tolerance;"
              f" experiment row max |activation| {act[(l, n)][0]:.3g} before the hook -> {act[(l, n)][1]:.3g} after)")
    together = set.intersection(*applied.values())
    good = len(together) == passes
    ok &= good
    print(f"  all {len(pushes)} push(es) active in the same forward passes: {len(together)}/{passes} {'OK' if good else 'FAIL'}")
    cleared = rt._push is None
    ok &= cleared
    print(f"  hook state cleared after generation: {'OK' if cleared else 'FAIL'}")
    after = weights_digest(rt, {p[0] for p in pushes})
    ok &= before == after
    print(f"  pushed layers' MLP weights unchanged: {'OK' if before == after else 'FAIL'} ({before} -> {after})")

    base, alt = rt.next_logprobs(prompt, True, "", pushes)
    kl = float((base.exp() * (base - alt)).sum())
    top = lambda lp: [rt.tok.decode([int(i)]) for i in lp.topk(5).indices]
    print(f"  first reply token, KL(baseline -> experiment) = {kl:.4f} nats; top 5 baseline {top(base)} / experiment {top(alt)}")
    differ = texts[0] != texts[1]
    print(f"  texts differ: {differ}")
    print(f"  BASELINE:   {texts[1][:300]!r}")
    print(f"  EXPERIMENT: {texts[0][:300]!r}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", default="ABC", help="which of A, B, C to run (default ABC)")
    ap.add_argument("--max-new", type=int, default=60, help="tokens generated per row (default 60)")
    a = ap.parse_args()
    rt = Runtime(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stats"))
    rt.ensure_loaded()
    print(f"model on {rt.dev}, EXPLORER_QUANT={os.environ.get('EXPLORER_QUANT', '') or 'off'}, "
          f"tolerance |err| <= {RTOL:.4g} * (|expected| + |push|) + {ATOL:g}")
    results = {c: check(rt, c, *CASES[c], a.max_new) for c in a.cases.upper() if c in CASES}
    print("\n" + "  ".join(f"{c}: {'PASS' if r else 'FAIL'}" for c, r in results.items()))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
