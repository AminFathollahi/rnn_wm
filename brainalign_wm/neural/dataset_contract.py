"""The `NeuralDataset` contract: a frozen interface implemented by every
neural data source, simulated (`sim_brain/`) and real (`adapters/`).
Analysis code depends only on this interface, never on a specific
dataset's file format.

Because single units are recorded across sessions/patients (never simultaneously
for the whole population), `response_patterns` builds a *pseudopopulation*: units
pooled by matched condition, not a true simultaneous population. Document this
assumption at every call site that treats it as if it were a real population.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

UnitID = str
SessionID = str
PatientID = str


@dataclass(frozen=True)
class ConditionLabel:
    """Harmonized condition schema shared by model and brain data.

    `item_id`/`category` are present for the image datasets (Tier A/B) and
    absent (None) for the verbal dataset (Tier C), which is aligned only on
    coarse conditions -- it has no image-identity match to the model's
    encoder.
    """

    load: int
    probe_in_set: bool
    correct: bool
    epoch: str  # fixation|encode1|encode2|encode3|maintain|probe|response
    item_id: Optional[int] = None
    category: Optional[str] = None

    def key(self) -> tuple:
        """Hashable key for grouping trials/conditions (category-agnostic
        variant available via `coarse_key` for cross-tier comparisons)."""
        return (self.load, self.probe_in_set, self.correct, self.epoch, self.item_id, self.category)

    def coarse_key(self) -> tuple:
        """Condition key without item identity/category — usable across all
        tiers including the verbal (Tier C) dataset."""
        return (self.load, self.probe_in_set, self.correct, self.epoch)


@runtime_checkable
class NeuralDataset(Protocol):
    """Frozen interface implemented by every neural data source."""

    conditions: list[ConditionLabel]

    def units(self, region: Optional[str] = None) -> list[UnitID]:
        """Sorted single-unit ids, optionally filtered by canonical region."""
        ...

    def spike_times(self, unit: UnitID) -> np.ndarray:
        """Spike times in seconds, sorted ascending."""
        ...

    def trials(self) -> pd.DataFrame:
        """One row per trial: epoch onset times + condition labels + response."""
        ...

    def rates(self, region: Optional[str], bin_ms: int, epochs: list[str]) -> np.ndarray:
        """[n_units, n_conditions|n_trials, n_timebins] firing rates."""
        ...

    def response_patterns(self, region: Optional[str], epoch: str) -> np.ndarray:
        """[n_conditions, n_units] condition-mean rates (pseudopopulation)."""
        ...

    def regions(self) -> list[str]:
        """Canonical region names present, e.g. 'hippocampus', 'amygdala',
        'dACC', 'preSMA', 'vmPFC' (see `normalize_region` below)."""
        ...

    def noise_ceiling(self, region: Optional[str], epoch: str) -> tuple[float, float]:
        """(lower, upper) trial-split / LOSO reliability of the neural RDM."""
        ...

    def sessions(self) -> list[SessionID]:
        ...

    def patient_of(self, session: SessionID) -> PatientID:
        ...


# ---- region normalization, shared by every real-data adapter ----

_RAW_TO_CANONICAL = {
    "hippocampus": "hippocampus",
    "amygdala": "amygdala",
    "entorhinal_cortex": "entorhinal",
    "dorsal_anterior_cingulate_cortex": "dACC",
    "pre_supplementary_motor_area": "preSMA",
    "ventral_medial_prefrontal_cortex": "vmPFC",
}

MTL_REGIONS = {"hippocampus", "amygdala", "entorhinal"}
MFC_REGIONS = {"dACC", "preSMA", "vmPFC"}


def normalize_region(raw: str) -> str:
    """Strip the hemisphere suffix and map to a canonical region name."""
    s = raw.strip().lower()
    for suffix in ("_left", "_right", "-left", "-right", " left", " right"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    s = s.strip("_- ")
    if s in _RAW_TO_CANONICAL:
        return _RAW_TO_CANONICAL[s]
    # fall back: title-case unknown regions rather than silently dropping them
    return s


def region_family(canonical_region: str) -> Optional[str]:
    """canonical region -> 'MTL' | 'MFC' | None (unmapped, e.g. other cortex)."""
    if canonical_region in MTL_REGIONS:
        return "MTL"
    if canonical_region in MFC_REGIONS:
        return "MFC"
    return None


def validate_dataset(ds: NeuralDataset, region: Optional[str] = None) -> None:
    """Boundary check: shapes, regions, and conditions are internally
    consistent. Used by every adapter's test suite and by the
    geometry-recovery gate and real-data smoke tests.
    """
    regs = ds.regions()
    if region is not None and region not in regs:
        raise ValueError(f"region {region!r} not in dataset regions {regs}")
    units = ds.units(region)
    if len(units) == 0:
        raise ValueError(f"no units for region={region!r}")
    trials = ds.trials()
    if len(trials) == 0:
        raise ValueError("empty trials table")
    lo, hi = ds.noise_ceiling(region, epoch="maintain")
    if not (0.0 <= lo <= hi <= 1.0 + 1e-6):
        raise ValueError(f"noise ceiling out of range: lower={lo}, upper={hi}")
