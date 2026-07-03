"""`dandi_nwb` adapter (protocol §8.4, M6): pooled Tier A loader for the
human single-neuron picture-Sternberg datasets **000469** (Kyzar/Kaminski)
and **000673** (Daume et al., near-identical schema). Reads NWB files
directly via h5py (no pynwb needed, §0.1). Implements the `NeuralDataset`
Protocol (`brainalign_wm/neural/dataset_contract.py`) so the exact same
analysis code (`analysis/rdm.py`, `analysis/rsa.py`, ...) that runs on
`SimulatedBrain` (M5) runs unchanged on real data.

Schema verified directly against the on-disk files (2026-07, see
DECISIONS.md): WM sessions are identified by presence of a `loads` trials
column (not by `ses-` numbering, which is not guaranteed ordered). Column
names differ slightly between datasets (harmonized in `COLUMN_MAPS`). There
is no region column on `units` -- region comes from joining
`units/electrodes` -> `general/extracellular_ephys/electrodes/location`,
normalized via `dataset_contract.normalize_region`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from brainalign_wm.neural.dataset_contract import ConditionLabel, normalize_region

COLUMN_MAPS = {
    "000469": {
        "enc_cols": ["loadsEnc1_PicIDs", "loadsEnc2_PicIDs", "loadsEnc3_PicIDs"],
        "probe_col": "loadsProbe_PicIDs",
    },
    "000673": {
        "enc_cols": ["PicIDs_Encoding1", "PicIDs_Encoding2", "PicIDs_Encoding3"],
        "probe_col": "PicIDs_Probe",
    },
}

# Fixed post-onset window used to bin spikes per epoch (protocol §9.3: align
# to epoch onsets, no HRF). Real trial durations vary; a fixed window keeps
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
    """WM sessions = trials table has a `loads` column (protocol §8.4)."""
    wm = []
    for f in sorted(Path(dataset_root).glob("**/*.nwb")):
        try:
            with h5py.File(f, "r") as h:
                if "intervals/trials" in h and "loads" in h["intervals/trials"]:
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


def _load_session(path: Path, dataset: str) -> _SessionData:
    colmap = COLUMN_MAPS[dataset]
    with h5py.File(path, "r") as h:
        identifier = _decode(h["identifier"][()])
        session_id = f"{dataset}-{identifier}"
        patient_id = identifier.split("_")[-1] if "_" in identifier else identifier

        tr = h["intervals/trials"]
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

        unit_ids, unit_region, spikes = [], {}, {}
        for u in range(n_units):
            uid = f"{session_id}#u{u}"
            unit_ids.append(uid)
            elec = int(units_electrodes[u])
            unit_region[uid] = electrode_regions[elec] if elec < len(electrode_regions) else "unknown"
            start = int(spike_idx[u - 1]) if u > 0 else 0
            end = int(spike_idx[u])
            spikes[uid] = spike_times_all[start:end]

    return _SessionData(
        session_id=session_id, patient_id=patient_id, trials=trials_df,
        unit_ids=unit_ids, unit_region=unit_region, spikes=spikes,
    )


class DandiSternbergTierA:
    """Pooled 000469 + 000673 picture-Sternberg adapter (Tier A, protocol §8.4/M6)."""

    bin_ms: int

    def __init__(
        self,
        data_root: str | Path,
        datasets: tuple[str, ...] = ("000469", "000673"),
        min_firing_hz: float = 0.2,
        bin_ms: int = 50,
        max_sessions_per_dataset: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.bin_ms = bin_ms
        self.min_firing_hz = min_firing_hz
        self._sessions: dict[str, _SessionData] = {}
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
        if len(sess.trials) == 0:
            sess.unit_ids = []
            return
        duration = max(sess.trials.t_stop.max() - sess.trials.t_start.min(), 1e-6)
        sess.unit_ids = [
            uid for uid in sess.unit_ids
            if len(sess.spikes[uid]) / duration >= self.min_firing_hz
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

    def units(self, region: Optional[str] = None) -> list[str]:
        out = []
        for sess in self._sessions.values():
            for uid in sess.unit_ids:
                if region is None or sess.unit_region[uid] == region:
                    out.append(uid)
        return sorted(out)

    def spike_times(self, unit: str) -> np.ndarray:
        session_id = unit.split("#")[0]
        return self._sessions[session_id].spikes[unit]

    def trials(self) -> pd.DataFrame:
        return pd.concat([s.trials for s in self._sessions.values()], ignore_index=True)

    def rates(self, region: Optional[str], bin_ms: int, epochs: list[str]) -> np.ndarray:
        """[n_units, n_trials, n_bins]; per-trial epoch-onset-aligned binned
        rates, concatenated across the requested epochs (fixed windows,
        `EPOCH_WINDOWS_S` -- see module docstring)."""
        units = self.units(region)
        all_trials = self.trials()
        bin_s = bin_ms / 1000.0
        n_bins_per_epoch = {ep: max(1, int(round(EPOCH_WINDOWS_S[ep] / bin_s))) for ep in epochs}
        total_bins = sum(n_bins_per_epoch.values())
        out = np.zeros((len(units), len(all_trials), total_bins))
        for ui, uid in enumerate(units):
            session_id = uid.split("#")[0]
            spikes = self._sessions[session_id].spikes[uid]
            for ti, row in enumerate(all_trials.itertuples()):
                if row.session != session_id:
                    continue
                col = 0
                for ep in epochs:
                    onset = getattr(row, EPOCH_ONSET_COL[ep])
                    n_bins = n_bins_per_epoch[ep]
                    edges = onset + np.arange(n_bins + 1) * bin_s
                    counts, _ = np.histogram(spikes, bins=edges)
                    out[ui, ti, col : col + n_bins] = counts / bin_s
                    col += n_bins
        return out

    def response_patterns(self, region: Optional[str], epoch: str) -> np.ndarray:
        units = self.units(region)
        trials = self.trials()
        rates = self.rates(region, self.bin_ms, [epoch])
        mean_rate = rates.mean(axis=2)
        out = np.zeros((len(self.conditions), len(units)))
        for ci, c in enumerate(self.conditions):
            mask = (
                (trials.load == c.load) & (trials.probe_in_set == c.probe_in_set)
                & (trials.correct == c.correct)
            ).values
            if mask.sum() > 0:
                out[ci, :] = mean_rate[:, mask].mean(axis=1)
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
    """Precompute + cache ResNet features for each dataset's embedded
    `StimulusTemplates` images (protocol §0.1 Step 0.7, §7.4's exact-image
    alignment), keyed by (dataset, session, PicID). Called from
    `encoders/cache_features.py` once this adapter is ready."""
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
        for f in find_wm_sessions(ds_root):
            with h5py.File(f, "r") as h:
                if "stimulus/templates/StimulusTemplates" not in h:
                    continue
                grp = h["stimulus/templates/StimulusTemplates"]
                identifier = _decode(h["identifier"][()])
                session_id = f"{ds}-{identifier}"
                images, pic_ids = [], []
                for key in grp:
                    if not key.startswith("image_"):
                        continue
                    pic_id = key.split("_", 1)[1]
                    images.append(np.asarray(grp[key][:], dtype=np.uint8))
                    pic_ids.append(pic_id)
                if not images:
                    continue
                feats = encode_images(images)
                out_path = out_dir / f"{session_id}.npz"
                np.savez(out_path, pic_ids=np.array(pic_ids), features=feats)
