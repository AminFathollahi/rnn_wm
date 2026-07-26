#!/usr/bin/env python3
"""PHASE 1 item 1.3 (comments.txt): given a target EFFECTIVE-synapse budget
(weight_ih + masked weight_hh, the same convention as
`MaskedGRUCell.effective_param_count`), solve for the flat (S=0) core's
`flat_units`, and for the hierarchical (S=1) core's (`worker_grid`,
`manager_units`) at a fixed `worker_density` -- `worker_density` is a
pre-registered scientific choice (comments.txt §1.6, B2: mean in-degree
6 -> 14), not something this solver should search over. `worker_grid` is
square, so `worker_units = g*g`.

Prints the solution; does NOT edit config.yaml (comments.txt: "do not
auto-edit config").
"""
from __future__ import annotations

import argparse
import math


def solve_flat_units(target_synapses: float, d_in: int, gates: int) -> int:
    """gates*H*(H+d_in) = target -- solve the quadratic for H, round to the
    nearest integer."""
    a, b, c = float(gates), float(gates * d_in), -float(target_synapses)
    h = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return max(1, round(h))


def _pooled_dim(grid_side: int, pool_block: int) -> int:
    cells_per_side = (grid_side + pool_block - 1) // pool_block
    return cells_per_side * cells_per_side


def solve_hierarchical(
    target_synapses: float, d_in: int, g_dim: int, density: float, gates: int,
    pool_block: int = 2, grid_range: range = range(6, 25), manager_range: range = range(8, 160),
) -> dict:
    """Brute-force search over (grid_side, manager_units): worker_units =
    grid_side**2, worker input = d_in + g_dim (worker sees [z_t; g_prev]).
    Effective synapses = worker's (dense weight_ih + density-masked
    weight_hh) + manager's (dense weight_ih + weight_hh, mask=None always)
    + g_proj (manager_units x g_dim, dense)."""
    best = None
    for grid_side in grid_range:
        worker_units = grid_side * grid_side
        worker_in = d_in + g_dim
        worker_syn = gates * worker_units * worker_in + gates * (worker_units ** 2) * density
        s_dim = _pooled_dim(grid_side, pool_block)
        for manager_units in manager_range:
            manager_syn = gates * manager_units * s_dim + gates * manager_units * manager_units
            gproj_syn = manager_units * g_dim
            total = worker_syn + manager_syn + gproj_syn
            err = abs(total - target_synapses)
            if best is None or err < best["err"]:
                best = {
                    "grid": (grid_side, grid_side), "worker_units": worker_units,
                    "manager_units": manager_units, "total_synapses": int(total), "err": err,
                }
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-vanilla", type=float, default=25000)
    ap.add_argument("--target-gru", type=float, default=74000)
    ap.add_argument("--d-in", type=int, default=64, help="bottleneck width")
    ap.add_argument("--g-dim", type=int, default=32)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--pool-block", type=int, default=2)
    args = ap.parse_args()

    print(f"d_in={args.d_in} g_dim={args.g_dim} density={args.density}\n")

    for label, target, gates in [("vanilla", args.target_vanilla, 1), ("gru", args.target_gru, 3)]:
        flat_units = solve_flat_units(target, args.d_in, gates)
        achieved_flat = gates * flat_units * (flat_units + args.d_in)
        print(f"[{label}] target={target:.0f} gates={gates}")
        print(f"  flat_units (S=0) = {flat_units}  (achieves {achieved_flat} synapses)")

        hier = solve_hierarchical(target, args.d_in, args.g_dim, args.density, gates, args.pool_block)
        print(
            f"  hierarchical (S=1): worker_grid={hier['grid']} worker_units={hier['worker_units']} "
            f"manager_units={hier['manager_units']}  (achieves {hier['total_synapses']} synapses)"
        )
        print()


if __name__ == "__main__":
    main()
