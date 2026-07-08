"""Pooled Tier A loader for the human single-neuron picture-Sternberg
datasets 000469 (Kyzar/Kaminski et al.) and 000673 (Daume et al., of
near-identical schema). Reads NWB files directly via `h5py`, without a
`pynwb` dependency. Implements the `NeuralDataset` Protocol defined in
`brainalign_wm/neural/dataset_contract.py`, so the same analysis code
(`analysis/rdm.py`, `analysis/rsa.py`, and related modules) that runs on
`SimulatedBrain` runs unchanged on real data.

Schema verified directly against the on-disk files (2026-07): WM sessions
are identified by presence of a `loads` column on a candidate trials
group (`TRIALS_GROUP_CANDIDATES`: `intervals/trials` for 000469/000673,
`intervals/WM_trials` for 001187, which also has a separate
`intervals/LTM_trials` New/Old-recognition task table that must NOT be
picked up), not by `ses-` numbering, which is not guaranteed ordered.
Column names differ slightly between datasets (harmonized in
`COLUMN_MAPS`; 001187's `WM_trials` matches 000673's schema exactly).
`DandiSternbergTierA` itself only ever loads `datasets_tierA`
(000469+000673) by default -- 001187 (Tier B) is loadable via this same
adapter class but not yet consumed by any analysis path (see
`config.yaml`'s `neural.datasets_tierB`); 000574 (Tier C) uses an
unrelated verbal-task schema and needs its own adapter. There is no
region column on `units` -- region comes from joining `units/electrodes`
-> `general/extracellular_ephys/electrodes/location`, normalized via
`dataset_contract.normalize_region`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from brainalign_wm.neural.dataset_contract import ConditionLabel, normalize_region, region_family

COLUMN_MAPS = {
    "000469": {
        "enc_cols": ["loadsEnc1_PicIDs", "loadsEnc2_PicIDs", "loadsEnc3_PicIDs"],
        "probe_col": "loadsProbe_PicIDs",
    },
    "000673": {
        "enc_cols": ["PicIDs_Encoding1", "PicIDs_Encoding2", "PicIDs_Encoding3"],
        "probe_col": "PicIDs_Probe",
    },
    # 001187 (SBCAT-NO, Tier B): its `intervals/WM_trials` group carries
    # the identical column schema to 000673's `intervals/trials`.
    "001187": {
        "enc_cols": ["PicIDs_Encoding1", "PicIDs_Encoding2", "PicIDs_Encoding3"],
        "probe_col": "PicIDs_Probe",
    },
}

# Fixed post-onset window used to bin spikes per epoch, aligned to epoch
# onset with no hemodynamic-response convolution (electrophysiology, unlike
# fMRI, requires none). Real trial durations vary; a fixed window keeps
# the [n_units, n_trials, n_bins] tensor rectangular. `maintain` (1.5s) is
# shorter than the dataset's actual maintenance period (~2.5s typical) --
# deliberately conservative so the window doesn't run into the next trial's
# probe on short-delay trials; document as a limitation (finer per-trial
# variable-length windows are a straightforward but unimplemented extension).
EPOCH_WINDOWS_S = {"fixation": 0.3, "maintain": 1.5, "probe": 0.5}
EPOCH_ONSET_COL = {"fixation": "t_fixation", "maintain": "t_maintain", "probe": "t_probe"}


def _decode(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def find_wm_sessions(dataset_root: Path) -> list[Path]:
    """Working-memory sessions are identified by the presence of a `loads`
    column on a candidate trials group (`TRIALS_GROUP_CANDIDATES`)."""
    wm = []
    for f in sorted(Path(dataset_root).glob("**/*.nwb")):
        try:
            with h5py.File(f, "r") as h:
                if _trials_group(h) is not None:
                    wm.append(f)
        except OSError:
            continue
    return wm


@dataclass
class _SessionData:
    session_id: str
    patient_id: str
    trials: pd.DataFrame
    unit_ids: list[str]
    unit_region: dict
    spikes: dict
    unit_isolation_distance: dict = field(default_factory=dict)


# Working-memory trials for 000469/000673 live at the top-level
# `intervals/trials`; 001187 (SBCAT-NO, "Sternberg-CAT New-Old") instead
# splits its two tasks into `intervals/WM_trials` (Sternberg, same column
# schema as 000673) and `intervals/LTM_trials` (a separate New/Old
# recognition task, not a WM trial table at all -- must not be picked up
# here). Checked in order; the first present group is used.
TRIALS_GROUP_CANDIDATES = ["intervals/trials", "intervals/WM_trials"]


def _trials_group(h: h5py.File) -> Optional[str]:
    for name in TRIALS_GROUP_CANDIDATES:
        if name in h and "loads" in h[name]:
            return name
    return None


def _load_session(path: Path, dataset: str) -> _SessionData:
    colmap = COLUMN_MAPS[dataset]
    with h5py.File(path, "r") as h:
        identifier = _decode(h["identifier"][()])
        session_id = f"{dataset}-{identifier}"
        patient_id = identifier.split("_")[-1] if "_" in identifier else identifier

        tr = h[_trials_group(h)]
        n_trials = tr["id"].shape[0]
        enc_cols = [tr[c][:] for c in colmap["enc_cols"]]
        rows = []
        for i in range(n_trials):
            held = [int(enc_cols[k][i]) for k in range(3) if enc_cols[k][i] != 0]
            rows.append(
                {
                    "trial_id": f"{session_id}#t{i}",
                    "session": session_id,
                    "load": int(tr["loads"][i]),
                    "held_items": held,
                    "probe_item": int(tr[colmap["probe_col"]][i]),
                    "probe_in_set": bool(tr["probe_in_out"][i]),
                    "correct": bool(tr["response_accuracy"][i]),
                    "t_fixation": float(tr["timestamps_FixationCross"][i]),
                    "t_maintain": float(tr["timestamps_Maintenance"][i]),
                    "t_probe": float(tr["timestamps_Probe"][i]),
                    "t_response": float(tr["timestamps_Response"][i]),
                    "t_start": float(tr["start_time"][i]),
                    "t_stop": float(tr["stop_time"][i]),
                }
            )
        trials_df = pd.DataFrame(rows)

        electrode_regions = [normalize_region(_decode(x)) for x in
                              h["general/extracellular_ephys/electrodes/location"][:]]
        units_electrodes = h["units/electrodes"][:]
        n_units = h["units/id"].shape[0]
        spike_idx = h["units/spike_times_index"][:]
        spike_times_all = h["units/spike_times"][:]
        # Isolation distance (Harris et al. 2001 / Schmitzer-Torbert et al.
        # 2005 cluster-quality metric; reported directly by the Rutishauser
        # lab's own spike-sorting pipeline, Kyzar et al. 2024) -- NaN for a
        # unit where the metric couldn't be computed (e.g. no distinct noise
        # cluster), not present at all for datasets released without it.
        iso_all = h["units/waveforms_isolation_distance"][:] if "units/waveforms_isolation_distance" in h else None

        unit_ids, unit_region, spikes, unit_isolation_distance = [], {}, {}, {}
        for u in range(n_units):
            uid = f"{session_id}#u{u}"
            unit_ids.append(uid)
            elec = int(units_electrodes[u])
            unit_region[uid] = electrode_regions[elec] if elec < len(electrode_regions) else "unknown"
            start = int(spike_idx[u - 1]) if u > 0 else 0
            end = int(spike_idx[u])
            spikes[uid] = spike_times_all[start:end]
            unit_isolation_distance[uid] = float(iso_all[u]) if iso_all is not None else None

    return _SessionData(
        session_id=session_id, patient_id=patient_id, trials=trials_df,
        unit_ids=unit_ids, unit_region=unit_region, spikes=spikes,
        unit_isolation_distance=unit_isolation_distance,
    )


class DandiSternbergTierA:
    """Pooled 000469 + 000673 picture-Sternberg adapter (Tier A)."""

    bin_ms: int

    def __init__(
        self,
        data_root: str | Path,
        datasets: tuple[str, ...] = ("000469", "000673"),
        min_firing_hz: float = 0.2,
        min_isolation_distance: float = 20.0,
        bin_ms: int = 50,
        max_sessions_per_dataset: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.bin_ms = bin_ms
        self.min_firing_hz = min_firing_hz
        self.min_isolation_distance = min_isolation_distance
        self._sessions: dict[str, _SessionData] = {}
        self._trials_cache: Optional[pd.DataFrame] = None
        self._rate_cache: dict[tuple, np.ndarray] = {}  # (uid, epoch) -> [n_all_trials, n_bins], at self.bin_ms only
        for ds in datasets:
            ds_root = self.data_root / ds
            if not ds_root.exists():
                continue
            wm_files = find_wm_sessions(ds_root)
            if max_sessions_per_dataset:
                wm_files = wm_files[:max_sessions_per_dataset]
            for f in wm_files:
                sess = _load_session(f, ds)
                self._apply_firing_qc(sess)
                if sess.unit_ids:
                    self._sessions[sess.session_id] = sess
        if not self._sessions:
            raise FileNotFoundError(
                f"no WM sessions found under {self.data_root} for datasets {datasets} "
                f"(is the external USB mounted at that path?)"
            )
        self._build_conditions()

    def _apply_firing_qc(self, sess: _SessionData) -> None:
        """Firing-rate + single-unit isolation-quality QC. Isolation
        distance (Harris et al. 2001 / Schmitzer-Torbert et al. 2005) is
        NOT applied when the dataset release doesn't carry the field at
        all (`unit_isolation_distance[uid] is None`) -- only when the
        field exists but is NaN for a specific unit (the metric couldn't
        be computed for it, e.g. no distinct noise cluster), which this
        pipeline treats as a QC failure, consistent with standard practice
        for this metric in the single-unit literature."""
        if len(sess.trials) == 0:
            sess.unit_ids = []
            return
        duration = max(sess.trials.t_stop.max() - sess.trials.t_start.min(), 1e-6)

        def _passes_isolation(uid: str) -> bool:
            iso = sess.unit_isolation_distance.get(uid)
            if iso is None:
                return True  # dataset doesn't carry this field; not a criterion here
            return not np.isnan(iso) and iso >= self.min_isolation_distance

        sess.unit_ids = [
            uid for uid in sess.unit_ids
            if len(sess.spikes[uid]) / duration >= self.min_firing_hz and _passes_isolation(uid)
        ]

    def _build_conditions(self) -> None:
        keys = set()
        for sess in self._sessions.values():
            for row in sess.trials.itertuples():
                keys.add((row.load, row.probe_in_set, row.correct))
        self.conditions = [
            ConditionLabel(load=l, probe_in_set=p, correct=c, epoch="maintain")
            for (l, p, c) in sorted(keys)
        ]

    # ---------------- NeuralDataset Protocol ----------------

    def _region_matches(self, canonical_region: str, region: Optional[str]) -> bool:
        """`region` may be a canonical single region (e.g. 'hippocampus'), a
        region FAMILY ('MTL'/'MFC', resolved via
        `dataset_contract.region_family`), or None (no filter)."""
        if region is None:
            return True
        if region in ("MTL", "MFC"):
            return region_family(canonical_region) == region
        return canonical_region == region

    def units(self, region: Optional[str] = None) -> list[str]:
        out = []
        for sess in self._sessions.values():
            for uid in sess.unit_ids:
                if self._region_matches(sess.unit_region[uid], region):
                    out.append(uid)
        return sorted(out)

    def spike_times(self, unit: str) -> np.ndarray:
        session_id = unit.split("#")[0]
        return self._sessions[session_id].spikes[unit]

    def trials(self) -> pd.DataFrame:
        if self._trials_cache is None:
            self._trials_cache = pd.concat([s.trials for s in self._sessions.values()], ignore_index=True)
        return self._trials_cache

    def _epoch_rates_uncached(self, uid: str, epoch: str, all_trials: pd.DataFrame, bin_s: float, n_bins: int, session_id: str) -> np.ndarray:
        spikes = self._sessions[session_id].spikes[uid]
        result = np.zeros((len(all_trials), n_bins))
        onset_col = all_trials[EPOCH_ONSET_COL[epoch]].values
        session_mask = (all_trials["session"].values == session_id)
        for ti in np.where(session_mask)[0]:
            edges = onset_col[ti] + np.arange(n_bins + 1) * bin_s
            counts, _ = np.histogram(spikes, bins=edges)
            result[ti] = counts / bin_s
        return result

    def rates(self, region: Optional[str], bin_ms: int, epochs: list[str]) -> np.ndarray:
        """[n_units, n_trials, n_bins]; per-trial epoch-onset-aligned binned
        rates, concatenated across the requested epochs (fixed windows,
        `EPOCH_WINDOWS_S` -- see module docstring). CAUTION: for a unit whose
        own session != a given trial's session, that entry is left at 0.0 --
        a valid "this unit wasn't recorded on this trial" placeholder ONLY
        if the caller subsequently restricts to session-matched
        (unit, trial) pairs (as `analysis.rsa._session_condition_rdm` does).
        Averaging this tensor's columns directly across trials from
        multiple sessions (as `response_patterns` used to, and as a naive
        pooled analysis might) silently dilutes every condition mean with
        injected zeros -- see `response_patterns` and
        `analysis/pseudopopulation.py` for the valid cross-session
        pooling path.

        Per-(unit, epoch) results are cached at `self.bin_ms` (the
        constructor's resolution -- the only one ever requested in
        practice): the audit-fix alignment pipeline calls `rates()` far more
        often than the original single-pass code did (once per session, per
        region, per noise-ceiling resample), and recomputing every unit's
        spike histogram from scratch each time would be impractically
        slow for anything beyond a small session subset -- mirrors
        `sim_brain`'s own `_rate_cache`."""
        units = self.units(region)
        all_trials = self.trials()
        bin_s = bin_ms / 1000.0
        n_bins_per_epoch = {ep: max(1, int(round(EPOCH_WINDOWS_S[ep] / bin_s))) for ep in epochs}
        total_bins = sum(n_bins_per_epoch.values())
        out = np.zeros((len(units), len(all_trials), total_bins))
        use_cache = bin_ms == self.bin_ms
        for ui, uid in enumerate(units):
            session_id = uid.split("#")[0]
            col = 0
            for ep in epochs:
                n_bins = n_bins_per_epoch[ep]
                if use_cache:
                    key = (uid, ep)
                    if key not in self._rate_cache:
                        self._rate_cache[key] = self._epoch_rates_uncached(uid, ep, all_trials, bin_s, n_bins, session_id)
                    per_unit_epoch = self._rate_cache[key]
                else:
                    per_unit_epoch = self._epoch_rates_uncached(uid, ep, all_trials, bin_s, n_bins, session_id)
                out[ui, :, col : col + n_bins] = per_unit_epoch
                col += n_bins
        return out

    def response_patterns(self, region: Optional[str], epoch: str) -> np.ndarray:
        """[n_conditions, n_units] condition-mean pseudopopulation. Audit fix
        A2d: each unit's condition mean is computed using ONLY that unit's
        own session's matching trials (never diluted by other sessions'
        trials, which the pooled `rates(region, ...)` tensor otherwise
        zero-fills for out-of-session entries)."""
        units = self.units(region)
        trials = self.trials()
        rates = self.rates(region, self.bin_ms, [epoch])
        mean_rate = rates.mean(axis=2)  # [n_units, n_trials]; zero-filled outside each unit's own session
        session_of_trial = trials.session.values
        out = np.zeros((len(self.conditions), len(units)))
        for ci, c in enumerate(self.conditions):
            cond_mask = (
                (trials.load == c.load) & (trials.probe_in_set == c.probe_in_set)
                & (trials.correct == c.correct)
            ).values
            for ui, uid in enumerate(units):
                own_session = uid.split("#")[0]
                unit_mask = cond_mask & (session_of_trial == own_session)
                if unit_mask.sum() > 0:
                    out[ci, ui] = mean_rate[ui, unit_mask].mean()
                # else: this unit's own session has no trials for this
                # condition -- left at 0.0 (documented limitation; rare for
                # the coarse load/in_set/correct schema used here), never
                # averaged in from another unit's session (A2d).
        return out

    def regions(self) -> list[str]:
        regs = set()
        for sess in self._sessions.values():
            regs.update(sess.unit_region[u] for u in sess.unit_ids)
        return sorted(regs)

    def noise_ceiling(self, region: Optional[str], epoch: str) -> tuple[float, float]:
        from brainalign_wm.analysis.rsa import noise_ceiling_from_dataset

        return noise_ceiling_from_dataset(self, region, epoch)

    def sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def patient_of(self, session: str) -> str:
        return self._sessions[session].patient_id


def cache_stimulus_features(cfg: dict) -> None:
    """Precompute and cache ResNet features for each dataset's embedded
    `StimulusTemplates` images, keyed by (dataset, session, PicID), enabling
    exact-image alignment between the model and the recorded sessions.
    Called from `encoders/cache_features.py`.

    The trial table's PicID convention is NOT uniform across Tier A: 000673's
    `PicIDs_Encoding*`/`PicIDs_Probe` values are the real `image_<PicID>` key
    suffix directly (e.g. PicID 201 -> `image_201`). 000469's are NOT --
    they are a small 1-indexed POSITION into `StimulusTemplates`'s own
    `order_of_images` (an `ImageReferences` array of HDF5 object references,
    in presentation order): PicID 1 -> `order_of_images[0]` -> e.g.
    `image_22`, not `image_1` (`image_1` doesn't exist in that session at
    all). Caching by the raw `image_<suffix>` key alone (the previous
    behavior) silently failed to match ANY of 000469's real trial PicIDs for
    most sessions (confirmed directly against the mounted NWB files:
    0/45 real trials matched for 4 of 8 sessions checked, low single digits
    for 3 more, so `generate_activity_logs.py`'s per-trial stimulus-feature
    lookup skipped nearly every 000469 trial system-wide -- closer to total
    data loss for 6 of 8 sessions checked than a partial-coverage gap.
    This alone explains why
    load=2 conditions (000469 is the only Tier A dataset with a load=2
    arm) were nearly absent from every activity log.

    Fixed by resolving BOTH candidate PicID->image mappings per session
    (direct key match, and 1-indexed position into `order_of_images`) and
    picking whichever actually covers more of THAT session's own real trial
    PicIDs (self-validating against `intervals/trials`, not a hardcoded
    per-dataset branch, in case this varies by session rather than by
    dataset)."""
    import numpy as np

    from brainalign_wm.encoders.resnet18_encoder import encode_images

    paths = cfg["paths"]
    neural_cfg = cfg["neural"]
    data_root = Path(paths["data_root"])
    out_dir = Path(paths["feature_cache"]) / "dataset_stimuli"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in neural_cfg["datasets_tierA"]:
        ds_root = data_root / ds
        if not ds_root.exists():
            continue
        colmap = COLUMN_MAPS[ds]
        for f in find_wm_sessions(ds_root):
            with h5py.File(f, "r") as h:
                if "stimulus/templates/StimulusTemplates" not in h:
                    continue
                grp = h["stimulus/templates/StimulusTemplates"]
                identifier = _decode(h["identifier"][()])
                session_id = f"{ds}-{identifier}"

                direct: dict[str, np.ndarray] = {}
                for key in grp:
                    if not key.startswith("image_"):
                        continue
                    direct[key.split("_", 1)[1]] = np.asarray(grp[key][:], dtype=np.uint8)

                positional: dict[str, np.ndarray] = {}
                if "order_of_images" in grp:
                    for i, ref in enumerate(grp["order_of_images"][:]):
                        real_pic_id = h[ref].name.rsplit("_", 1)[-1]
                        if real_pic_id in direct:
                            positional[str(i + 1)] = direct[real_pic_id]

                tr = h["intervals/trials"]
                real_pids = set()
                if "loads" in tr:
                    for c in colmap["enc_cols"] + [colmap["probe_col"]]:
                        real_pids.update(str(int(x)) for x in tr[c][:] if int(x) != 0)
                n_direct_hits = sum(1 for pid in real_pids if pid in direct)
                n_positional_hits = sum(1 for pid in real_pids if pid in positional)
                pic_to_image = positional if n_positional_hits > n_direct_hits else direct

                images, pic_ids = [], []
                for pic_id, img in pic_to_image.items():
                    images.append(img)
                    pic_ids.append(pic_id)
                if not images:
                    continue
                feats = encode_images(images)
                out_path = out_dir / f"{session_id}.npz"
                np.savez(out_path, pic_ids=np.array(pic_ids), features=feats)
