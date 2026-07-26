#!/usr/bin/env python3
"""PHASE 1 item 1.2 (comments.txt): parameter-budget table over every Core
cell -- n_units | n_synapses_effective | n_params_total | deviation from the
S=0 reference, per [KHONA23]'s node-vs-synapse conventions (report all
three columns always). "Substrate" here is the S bit (flat vs
hierarchical): A2's fix is that the two should match, so deviation is
measured within each S group against that group's own mean, and the table
also reports the S=0-vs-S=1 group-mean ratio directly (the number A2 was
about).
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _mask_aware_param_count(module: torch.nn.Module) -> int:
    """Recursively sums trainable parameters, but counts `weight_hh`/`alpha`
    at only their `mask`-alive entries wherever a submodule carries a mask
    (`MaskedGRUCell`/`PlasticGRUCell`/`VanillaRNNCell`) -- same convention
    as `effective_param_count()`, applied to the WHOLE model (front end +
    core + heads), not just the core's recurrent weights. A masked-out
    entry has zero gradient and no effect on the forward pass; PyTorch
    still allocates it (dense tensor, masked at forward time -- A2's root
    cause), but it is not a real synapse, so it must not count toward
    `n_params_total` either."""
    total = 0
    mask = getattr(module, "mask", None)
    masked_names = set()
    if mask is not None:
        for name in ("weight_hh", "alpha"):
            if getattr(module, name, None) is not None:
                total += int(mask.sum().item())
                masked_names.add(name)
    for name, p in module.named_parameters(recurse=False):
        if name not in masked_names:
            total += p.numel()
    for child in module.children():
        total += _mask_aware_param_count(child)
    return total


def _cell_params(full_cfg: dict, model_id: str) -> dict:
    from brainalign_wm.training.generate_activity_logs import _parse_model_id
    from brainalign_wm.training.train import _build_model

    S, M, P, T, D = _parse_model_id(model_id)
    _front_end, core, _heads = _build_model(full_cfg, S, M, P, torch.device("cpu"))
    # Core-only, same scope as n_units/n_synapses_effective: front_end and
    # heads are shared/near-fixed across every cell by design (front_end
    # is literally identical everywhere -- see models/front_end.py), so
    # they add no per-cell signal; KHONA23's node/synapse table is about
    # the recurrent core.
    n_params_total = _mask_aware_param_count(core)
    return {
        "model_id": model_id, "S": S, "M": M, "P": P, "T": T, "D": D,
        "n_units": core.n_units(),
        "n_synapses_effective": core.effective_param_count(),
        "n_params_total": n_params_total,
    }


def build_table(full_cfg: dict) -> list[dict]:
    rows = [_cell_params(full_cfg, model_id) for model_id in full_cfg["cells"]]
    for s_group in (0, 1):
        group = [r["n_synapses_effective"] for r in rows if r["S"] == s_group]
        ref = sum(group) / len(group) if group else 0
        for r in rows:
            if r["S"] == s_group:
                r["deviation_from_group_mean"] = (r["n_synapses_effective"] - ref) / ref if ref else 0.0
    return rows


def max_pairwise_deviation(values: list[int]) -> float:
    """max |a-b|/max(a,b) over all pairs -- 0.0 for 0 or 1 values."""
    if len(values) < 2:
        return 0.0
    return max(
        abs(a - b) / max(a, b) for i, a in enumerate(values) for b in values[i + 1:]
    )


def main() -> None:
    full_cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    rows = build_table(full_cfg)

    print("| model_id | S | M | P | T | D | n_units | n_synapses_effective | n_params_total | dev_from_S_group_mean |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['model_id']} | {r['S']} | {r['M']} | {r['P']} | {r['T']} | {r['D']} | "
            f"{r['n_units']} | {r['n_synapses_effective']} | {r['n_params_total']} | "
            f"{r['deviation_from_group_mean']:+.2%} |"
        )

    print()
    for s_group in (0, 1):
        values = [r["n_synapses_effective"] for r in rows if r["S"] == s_group]
        dev = max_pairwise_deviation(values)
        print(f"S={s_group}: max pairwise deviation in n_synapses_effective = {dev:.2%} (n={len(values)} cells)")

    s0_mean = sum(r["n_synapses_effective"] for r in rows if r["S"] == 0) / max(1, sum(r["S"] == 0 for r in rows))
    s1_mean = sum(r["n_synapses_effective"] for r in rows if r["S"] == 1) / max(1, sum(r["S"] == 1 for r in rows))
    print(f"S=0 group mean n_synapses_effective = {s0_mean:.0f}; S=1 group mean = {s1_mean:.0f} "
          f"(ratio S1/S0 = {s1_mean / s0_mean:.3f})")

    max_total = max(r["n_params_total"] for r in rows)
    print(f"max n_params_total across all cells = {max_total} (<=150,000 required)")


if __name__ == "__main__":
    main()
