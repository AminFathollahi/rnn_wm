"""Annealed training curriculum: phases defined by ABSOLUTE step
boundaries -- warmup (load 1, short delay, no lures), an optional load2
stage (Phase 12.5 lever 3: loads 1-2 only, full delay, still no lures --
isolates load exposure from the lure ramp that starts in `ramp`; width 0
by default, a no-op for configs that don't set `load2_steps`), ramp (all
loads, full delay, lure fraction linearly increased from 0 to its target
value), and target (the full training distribution).

Phase 3 (comments.txt §5 item 3.5): boundaries are `warmup_steps`/
`ramp_steps` counts, not fractions of `total_steps`. A run's stopping point
now varies (train-to-criterion, §3), so a fraction-of-total boundary would
silently change the curriculum's actual shape from run to run; an absolute
step count means the same schedule (e.g. "loads=[1] for the first 30,000
steps") no matter how long the run ultimately runs.

`CurriculumSchedule.params_for` is a pure function of (step_idx,
total_steps, config), so the sampling distribution at any point in training
is fully reconstructable from the logged (step_idx, total_steps) pair.
`total_steps` is still accepted for that logging/reconstruction contract but
no longer used in the phase-boundary decision itself.
"""
from __future__ import annotations

from dataclasses import dataclass


def phase_at(step_idx: int, warmup_steps: int, ramp_steps: int, load2_steps: int = 0) -> str:
    if step_idx < warmup_steps:
        return "warmup"
    if step_idx < warmup_steps + load2_steps:
        return "load2"
    if step_idx < warmup_steps + load2_steps + ramp_steps:
        return "ramp"
    return "target"


@dataclass
class CurriculumSchedule:
    loads: list[int]
    warmup_steps: int
    ramp_steps: int
    full_lure_fraction: float
    full_maintain_steps: int
    # Phase 12.5 lever 3: width of the load2 stage between warmup and ramp.
    # 0 (default) reproduces the old warmup->ramp->target schedule exactly.
    load2_steps: int = 0
    # The warmup-phase maintenance delay is shortened to 20% of the target
    # length (floor of one step) to keep early trials fast and easy.
    warmup_delay_frac: float = 0.2

    @property
    def warmup_maintain_steps(self) -> int:
        return max(1, round(self.full_maintain_steps * self.warmup_delay_frac))

    def params_for(self, step_idx: int, total_steps: int) -> dict:
        phase = phase_at(step_idx, self.warmup_steps, self.ramp_steps, self.load2_steps)
        if phase == "warmup":
            return {
                "phase": phase,
                "step_idx": step_idx,
                "total_steps": total_steps,
                "loads": [1],
                "load_weights": None,
                "lure_fraction": 0.0,
                "maintain_steps": self.warmup_maintain_steps,
            }
        if phase == "load2":
            return {
                "phase": phase,
                "step_idx": step_idx,
                "total_steps": total_steps,
                "loads": sorted(l for l in self.loads if l <= 2),
                "load_weights": None,
                "lure_fraction": 0.0,
                "maintain_steps": self.full_maintain_steps,
            }
        if phase == "ramp":
            ramp_pos = (step_idx - self.warmup_steps - self.load2_steps) / max(self.ramp_steps, 1e-9)
            ramp_pos = min(max(ramp_pos, 0.0), 1.0)
            return {
                "phase": phase,
                "step_idx": step_idx,
                "total_steps": total_steps,
                "loads": sorted(self.loads),
                "load_weights": None,  # uniform over eligible loads
                "lure_fraction": ramp_pos * self.full_lure_fraction,
                "maintain_steps": self.full_maintain_steps,
            }
        return {
            "phase": "target",
            "step_idx": step_idx,
            "total_steps": total_steps,
            "loads": sorted(self.loads),
            "load_weights": None,
            "lure_fraction": self.full_lure_fraction,
            "maintain_steps": self.full_maintain_steps,
        }

    @classmethod
    def from_config(cls, config: dict) -> "CurriculumSchedule":
        t = config["task"]
        c = t["curriculum"]
        return cls(
            loads=list(t["loads"]),
            warmup_steps=int(c["warmup_steps"]),
            ramp_steps=int(c["ramp_steps"]),
            full_lure_fraction=float(t["lure_fraction"]),
            full_maintain_steps=int(t["maintain_steps"]),
            load2_steps=int(c.get("load2_steps", 0)),
        )
