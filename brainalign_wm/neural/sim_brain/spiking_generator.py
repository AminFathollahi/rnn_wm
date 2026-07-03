"""Simulated spiking generator with known, configurable ground-truth WM
geometry (protocol §8.3) -- validates the whole analysis pipeline (rates ->
RDM -> noise ceiling -> dPCA -> persistence -> cross-temporal decoding)
before any real recording is touched. Implements the `NeuralDataset`
Protocol (`brainalign_wm/neural/dataset_contract.py`) so the exact same
analysis code runs on simulated and real (M6) data.

Ground-truth latent geometry per trial, planted independently and additively
into each unit's firing rate (so downstream analyses can be checked against
a known answer):
  - item-identity subspace (`item_dim` items, each belonging to one of
    `category_dim` categories)
  - a graded load axis (`load_levels`)
  - a **persistent** delay-period component (constant across maintenance)
  - a **dynamic/rotational** delay-period component (a slowly rotating 2D
    trajectory whose phase encodes item identity -- Libby & Buschman-style)
    mixed with the persistent component by `persistent_frac`
  - an error/surprise axis (elevated on incorrect trials, at probe/response)
  - per-unit tuning heterogeneity / mixed selectivity: tuning weights on all
    axes are independent random Gaussians per unit, so many units are
    naturally mixed-selective by construction (Rigotti/Fusi).

Multiple synthetic sessions/patients, each with jittered per-unit tuning and
only a subset of units "recorded" per session (never simultaneously) --
this is what creates a genuine cross-unit/trial noise ceiling and forces
pseudopopulation pooling, matching real single-unit datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from brainalign_wm.neural.dataset_contract import ConditionLabel

EPOCHS = ["fixation", "encode", "maintain", "probe", "response"]
EPOCH_DURATIONS_S = {"fixation": 0.3, "encode": 1.0, "maintain": 1.5, "probe": 0.5, "response": 0.5}


@dataclass
class _UnitTuning:
    w_item: np.ndarray  # [item_dim]
    w_category: np.ndarray  # [category_dim]
    w_persistent: float
    w_dyn1: float
    w_dyn2: float
    w_load: float
    w_error: float
    base_rate: float
    phase: np.ndarray  # [item_dim] per-item rotational phase offset


class SimulatedBrain:
    """Implements the `NeuralDataset` Protocol over synthetic spikes."""

    def __init__(self, cfg: dict, seed: int | None = None, scramble: bool = False):
        self.cfg = cfg
        self.seed = cfg.get("seed", 0) if seed is None else seed
        self.scramble = scramble
        self.bin_ms = cfg["bin_ms"]
        self.base_rate_hz = cfg["base_rate_hz"]
        self.max_rate_hz = cfg["max_rate_hz"]
        self.refractory_s = cfg["refractory_ms"] / 1000.0
        self.item_dim = cfg["item_dim"]
        self.category_dim = cfg["category_dim"]
        self.load_levels = list(cfg["load_levels"])
        self.persistent_frac = cfg["persistent_frac"]
        self.n_sessions = cfg["n_sessions"]
        self.n_units_per_session = cfg["n_units_per_session"]
        self.n_patients = cfg["n_patients"]
        self.trials_per_condition = cfg["trials_per_condition"]

        self._rng = np.random.RandomState(self.seed)
        self._item_category = self._rng.randint(0, self.category_dim, size=self.item_dim)
        self._build_sessions()
        self._build_trials()
        self._build_units()
        self._epoch_bins_cache = self._epoch_bins()
        self._bin_edges_cache = self._compute_bin_edges()
        self._spikes: dict[tuple, np.ndarray] = {}
        self._rate_cache: dict[tuple, np.ndarray] = {}  # (uid,trial_id) -> [n_bins] binned rate (Hz), from spikes
        self._generate_all_spikes()

    # ---------------- construction ----------------

    def _build_sessions(self) -> None:
        self._sessions = [f"sim-sess-{i:02d}" for i in range(self.n_sessions)]
        self._patients = [f"sim-pat-{i:02d}" for i in range(self.n_patients)]
        self._session_patient = {
            s: self._patients[i % self.n_patients] for i, s in enumerate(self._sessions)
        }

    def _build_trials(self) -> None:
        rows = []
        trial_id = 0
        for session in self._sessions:
            for load in self.load_levels:
                for in_set in (True, False):
                    for correct in (True, False):
                        for _ in range(self.trials_per_condition):
                            item_id = int(self._rng.randint(0, self.item_dim))
                            category = int(self._item_category[item_id])
                            rows.append(
                                {
                                    "trial_id": trial_id,
                                    "session": session,
                                    "load": load,
                                    "item_id": item_id,
                                    "category": category,
                                    "probe_in_set": in_set,
                                    "correct": correct,
                                }
                            )
                            trial_id += 1
        self._trials_df = pd.DataFrame(rows)
        self.conditions = [
            ConditionLabel(
                load=r.load, probe_in_set=r.probe_in_set, correct=r.correct,
                epoch="maintain", item_id=r.item_id, category=str(r.category),
            )
            for r in self._trials_df.drop_duplicates(["load", "probe_in_set", "correct", "item_id", "category"]).itertuples()
        ]

    def _build_units(self) -> None:
        self._units: dict[str, list[str]] = {}
        self._unit_tuning: dict[str, _UnitTuning] = {}
        for session in self._sessions:
            unit_ids = []
            for u in range(self.n_units_per_session):
                uid = f"{session}#u{u:03d}"
                unit_ids.append(uid)
                self._unit_tuning[uid] = _UnitTuning(
                    w_item=self._rng.randn(self.item_dim) * 3.0,
                    w_category=self._rng.randn(self.category_dim) * 3.0,
                    w_persistent=self._rng.randn() * 4.0,
                    w_dyn1=self._rng.randn() * 4.0,
                    w_dyn2=self._rng.randn() * 4.0,
                    w_load=self._rng.randn() * 8.0,  # strong enough to clearly clear the noise floor at default trial counts (see DECISIONS.md)
                    w_error=abs(self._rng.randn()) * 3.0,
                    base_rate=max(0.5, self.base_rate_hz + self._rng.randn() * 1.0),
                    phase=self._rng.uniform(0, 2 * np.pi, size=self.item_dim),
                )
            self._units[session] = unit_ids
        self._all_units = [u for us in self._units.values() for u in us]
        if self.scramble:
            # scrambled control: permute which trial's latents drive which
            # unit's rate, destroying the planted structure while preserving
            # each unit's marginal tuning-parameter distribution.
            perm_rng = np.random.RandomState(self.seed + 999)
            self._scramble_perm = {
                s: perm_rng.permutation(self._trials_df[self._trials_df.session == s].trial_id.values)
                for s in self._sessions
            }
        else:
            self._scramble_perm = None

    def _epoch_bins(self) -> dict[str, np.ndarray]:
        """epoch -> array of bin-center times (s) relative to trial onset."""
        t0 = 0.0
        out = {}
        for ep in EPOCHS:
            dur = EPOCH_DURATIONS_S[ep]
            n_bins = max(1, int(round(dur * 1000 / self.bin_ms)))
            centers = t0 + (np.arange(n_bins) + 0.5) * (self.bin_ms / 1000.0)
            out[ep] = centers
            t0 += dur
        return out

    def _rate_trace(self, tuning: _UnitTuning, row) -> tuple[np.ndarray, np.ndarray]:
        """Returns (bin_times[s] concatenated across epochs, rate[Hz])."""
        epoch_bins = self._epoch_bins_cache if hasattr(self, "_epoch_bins_cache") else self._epoch_bins()
        all_t, all_rate = [], []
        item, cat, load = int(row.item_id), int(row.category), row.load
        correct = bool(row.correct)
        for ep, centers in epoch_bins.items():
            n = len(centers)
            item_term = tuning.w_item[item] if ep in ("encode", "maintain") else 0.0
            cat_term = tuning.w_category[cat] if ep in ("encode", "maintain") else 0.0
            load_term = tuning.w_load * (load - np.mean(self.load_levels))
            if ep == "maintain":
                persistent = tuning.w_persistent * self.persistent_frac
                t_rel = centers - centers[0]
                omega = 2 * np.pi / max(EPOCH_DURATIONS_S["maintain"], 1e-6)
                dyn = (1 - self.persistent_frac) * (
                    tuning.w_dyn1 * np.cos(omega * t_rel + tuning.phase[item])
                    + tuning.w_dyn2 * np.sin(omega * t_rel + tuning.phase[item])
                )
                delay_term = persistent + dyn
            else:
                delay_term = np.zeros(n)
            error_term = tuning.w_error * (0 if correct else 1) if ep in ("probe", "response") else 0.0
            drive = item_term + cat_term + load_term + error_term
            # Purely additive-then-clip (not softplus-of-sum): each planted axis
            # (item/category/load/persistent/dynamic/error) contributes independently
            # to the rate. An earlier softplus-of-sum version let a large load_term
            # saturate the nonlinearity and crush the item/category axes' effective
            # contribution -- an unwanted interaction for a generator whose whole
            # purpose is independently-recoverable planted axes (see DECISIONS.md).
            rate = tuning.base_rate + drive + delay_term
            rate = np.clip(rate, 0.0, self.max_rate_hz)
            all_t.append(centers)
            all_rate.append(rate if np.ndim(rate) else np.full(n, rate))
        return np.concatenate(all_t), np.concatenate(all_rate)

    def _poisson_spikes(self, t_bins: np.ndarray, rate_hz: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        dt = self.bin_ms / 1000.0
        p_spike = 1 - np.exp(-rate_hz * dt)
        draws = rng.random_sample(len(t_bins)) < p_spike
        spike_times = t_bins[draws]
        if len(spike_times) > 1:  # refractory thinning
            keep = [spike_times[0]]
            for s in spike_times[1:]:
                if s - keep[-1] >= self.refractory_s:
                    keep.append(s)
            spike_times = np.array(keep)
        return spike_times

    def _generate_all_spikes(self) -> None:
        """Generates spikes AND caches their re-binned rate (histogram at
        `bin_ms`) once per (unit, trial) -- `rates()` then does O(1) dict
        lookups instead of re-histogramming ~n_units*n_trials times per call
        (the naive version was ~500k histogram calls per `rates()` call,
        called repeatedly by `noise_ceiling_from_dataset` -- far too slow)."""
        rng = np.random.RandomState(self.seed + 1)
        edges = self._bin_edges_cache
        for session in self._sessions:
            trials = self._trials_df[self._trials_df.session == session]
            trial_id_list = trials.trial_id.values.tolist()
            for uid in self._units[session]:
                tuning = self._unit_tuning[uid]
                for pos, row in enumerate(trials.itertuples()):
                    src_row = row
                    if self.scramble:
                        src_trial_id = self._scramble_perm[session][pos]
                        src_row = self._trials_df[self._trials_df.trial_id == src_trial_id].iloc[0]
                    t_bins, rate = self._rate_trace(tuning, src_row)
                    spikes = self._poisson_spikes(t_bins, rate, rng)
                    self._spikes[(uid, row.trial_id)] = spikes
                    counts, _ = np.histogram(spikes, bins=edges)
                    self._rate_cache[(uid, row.trial_id)] = counts / (self.bin_ms / 1000.0)

    # ---------------- NeuralDataset Protocol ----------------

    def units(self, region: str | None = None) -> list[str]:
        return sorted(self._all_units)

    def spike_times(self, unit: str) -> np.ndarray:
        all_spikes = [v for (u, _tid), v in self._spikes.items() if u == unit]
        return np.sort(np.concatenate(all_spikes)) if all_spikes else np.array([])

    def trials(self) -> pd.DataFrame:
        return self._trials_df.copy()

    def rates(self, region: str | None, bin_ms: int, epochs: list[str]) -> np.ndarray:
        """[n_units, n_trials, n_timebins], firing rates re-binned at
        `self.bin_ms` (the generator's native bin width; `bin_ms` arg is
        accepted for Protocol-signature compatibility but re-binning to an
        arbitrary width is not implemented -- assert equal, don't silently
        ignore), restricted to the requested epochs (EPOCHS order). O(1)
        dict lookup per (unit,trial) via `_rate_cache` (built once at
        construction, see `_generate_all_spikes`) -- NOT re-histogrammed
        here, which would be ~500k redundant histogram calls per call on
        default config sizes."""
        if bin_ms != self.bin_ms:
            raise NotImplementedError(f"SimulatedBrain generated at bin_ms={self.bin_ms}; re-binning to {bin_ms} not supported")
        keep_idx = self._epoch_keep_idx(epochs)
        trial_ids = self._trials_df.trial_id.values
        units = self.units(region)
        zeros = np.zeros(len(keep_idx))
        out = np.zeros((len(units), len(trial_ids), len(keep_idx)))
        for ui, uid in enumerate(units):
            for ti, tid in enumerate(trial_ids):
                rate = self._rate_cache.get((uid, tid))
                out[ui, ti, :] = rate[keep_idx] if rate is not None else zeros
        return out

    def _epoch_keep_idx(self, epochs: list[str]) -> np.ndarray:
        keep_mask = []
        for ep in EPOCHS:
            n = len(self._epoch_bins_cache[ep])
            keep_mask.append(np.ones(n, dtype=bool) if ep in epochs else np.zeros(n, dtype=bool))
        return np.where(np.concatenate(keep_mask))[0]

    def _compute_bin_edges(self) -> np.ndarray:
        epoch_bins = self._epoch_bins_cache if hasattr(self, "_epoch_bins_cache") else self._epoch_bins()
        centers = np.concatenate([epoch_bins[ep] for ep in EPOCHS])
        dt = self.bin_ms / 1000.0
        return np.concatenate([centers - dt / 2, [centers[-1] + dt / 2]])

    def response_patterns(self, region: str | None, epoch: str) -> np.ndarray:
        """[n_conditions, n_units] condition-mean rate (pseudopopulation),
        pooling units across sessions by matched (coarse) condition."""
        units = self.units(region)
        trials = self._trials_df
        rates = self.rates(region, self.bin_ms, [epoch])  # [n_units, n_trials, n_bins]
        mean_rate = rates.mean(axis=2)  # [n_units, n_trials]
        cond_keys = [c.coarse_key() for c in self.conditions]
        out = np.zeros((len(cond_keys), len(units)))
        for ci, c in enumerate(self.conditions):
            mask = (
                (trials.load == c.load) & (trials.probe_in_set == c.probe_in_set)
                & (trials.correct == c.correct)
            ).values
            if mask.sum() == 0:
                continue
            out[ci, :] = mean_rate[:, mask].mean(axis=1)
        return out

    def regions(self) -> list[str]:
        return ["sim"]

    def noise_ceiling(self, region: str | None, epoch: str) -> tuple[float, float]:
        from brainalign_wm.analysis.rsa import noise_ceiling_from_dataset

        return noise_ceiling_from_dataset(self, region, epoch, n_splits=20)

    def sessions(self) -> list[str]:
        return list(self._sessions)

    def patient_of(self, session: str) -> str:
        return self._session_patient[session]


def make_scrambled(cfg: dict, seed: int | None = None) -> SimulatedBrain:
    return SimulatedBrain(cfg, seed=seed, scramble=True)
