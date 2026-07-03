"""Annealed curriculum (protocol §7.2): 3 phases by fraction of total training
steps -- warmup (load=1, short delay, no lures) -> ramp (all loads, full delay,
lure_fraction linearly ramped 0 -> target) -> target (full distribution).

`CurriculumSchedule.params_for` is a pure function of (step_idx, total_steps,
config) so the sampling distribution over time is fully reconstructable from
logged (step_idx, total_steps) pairs, per the protocol's logging requirement.
"""
from __future__ import annotations

from dataclasses import dataclass


def phase_at(step_idx: int, total_steps: int, warmup_frac: float, ramp_frac: float) -> str:
    if total_steps <= 1:
        return "target"
    frac = step_idx / (total_steps - 1)
    if frac < warmup_frac:
        return "warmup"
    if frac < warmup_frac + ramp_frac:
        return "ramp"
    return "target"


@dataclass
class CurriculumSchedule:
    loads: list[int]
    warmup_frac: float
    ramp_frac: float
    full_lure_fraction: float
    full_maintain_steps: int
    # Warmup delay is shortened to keep early trials fast/easy (protocol §7.2,
    # "short delay"); 20% of the target maintenance length, floor 1 step --
    # an explicit design choice, not specified numerically by the protocol.
    warmup_delay_frac: float = 0.2

    @property
    def warmup_maintain_steps(self) -> int:
        return max(1, round(self.full_maintain_steps * self.warmup_delay_frac))

    def params_for(self, step_idx: int, total_steps: int) -> dict:
        phase = phase_at(step_idx, total_steps, self.warmup_frac, self.ramp_frac)
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
        if phase == "ramp":
            frac = step_idx / max(total_steps - 1, 1)
            ramp_pos = (frac - self.warmup_frac) / max(self.ramp_frac, 1e-9)
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
            warmup_frac=float(c["warmup_frac"]),
            ramp_frac=float(c["ramp_frac"]),
            full_lure_fraction=float(t["lure_fraction"]),
            full_maintain_steps=int(t["maintain_steps"]),
        )
