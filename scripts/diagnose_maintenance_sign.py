#!/usr/bin/env python3
"""comments.txt §20.6 (advisor.md D37): diagnose the SIGN of the
maintenance-epoch alignment DV. Every trained model's maintenance RDM is
anti-correlated with the human one (`chance -0.1065 < LEGACY -0.0826 <
RL -0.0647 < SUP -0.0132`) -- monotone in training quality, so the DV
carries signal, but a DV that returns the same sign on every model
including the untrained one is not yet a measuring instrument. Three
diagnoses, cheapest first, all read from a checkpoint/activity log already
on disk -- no training, no GPU.

  (a) EPOCH MAPPING. The reported DV compares whole-epoch-MEAN patterns.
      If the human delay-period code drifts across the ~1.5s window while
      the model's is static (or vice versa), a single mean-vs-mean
      comparison is anti-correlated by construction even when some
      tick<->bin offset is strongly positive. Builds a
      [model_tick x neural_bin] alignment matrix and reports the diagonal
      (matched relative-time position) against the argmax.
  (b) LOAD DOMINANCE. The reported DV is load-stratified
      (`stratified_crossnobis_rdm`) specifically to remove cross-load
      structure from driving the correlation. Checks whether the residual
      WITHIN each load stratum is still negative, by computing the
      whole-epoch alignment separately for load 1, 2, and 3.
  (c) POSITIVE CONTROL. The DV has a chance control (near floor) and a
      noise ceiling, but nothing in the repo has ever been shown to
      return a clearly POSITIVE value from it. Builds one from real
      neural data plus independent per-observation noise (same condition
      labels), which must be genuinely related to the source data, and
      confirms the estimator can return a positive number at all.

Diagnostic only. Uses the plain (non-stratified) `crossnobis_rdm`/
`compare_rdms` machinery already in `analysis/rdm.py`/`analysis/rsa.py` and
the per-tick/per-bin pattern extraction already in `analysis/run_all.py`'s
dPCA path (`_model_dpca_patterns`/`_neural_dpca_patterns`) -- it does not
change the reported DV, its estimator, or any gate (D37: that is the
user's call).

    $ python scripts/diagnose_maintenance_sign.py --run-id FLATGRU_SUP_s0 --n-sessions 15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_cfg():
    import yaml
    return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())


def _dandi_data(cfg):
    from brainalign_wm.neural.adapters.dandi_nwb import DandiSternbergTierA

    return DandiSternbergTierA(
        cfg["paths"]["data_root"], datasets=tuple(cfg["neural"]["datasets_tierA"]),
        min_firing_hz=cfg["neural"]["min_firing_hz"], min_session_accuracy=cfg["neural"]["min_session_accuracy"],
        bin_ms=cfg["neural"]["bin_ms"],
    )


def _model_df(run_id: str, checkpoint: str):
    from brainalign_wm.training.generate_activity_logs import activity_log_path
    from brainalign_wm.training.logging_schema import read_log

    return read_log(activity_log_path(run_id, checkpoint))


# ---------------------------------------------------------------- (a) ----

def _matched_bin_indices(n_ticks: int, n_bins: int) -> list[int]:
    """The "diagonal" of a non-square [n_ticks x n_bins] matrix: tick t's
    matched bin is the nearest bin at the same FRACTIONAL position along
    the maintain epoch, not literal index t (15 model ticks and 30 neural
    bins cover the same nominal 1.5s window at different resolutions)."""
    return [int(round(t * (n_bins - 1) / max(n_ticks - 1, 1))) for t in range(n_ticks)]


def tick_bin_matrix(run_id: str, dandi_data, df, region, n_sessions: int, n_folds: int = 2, seed: int = 0):
    """[n_ticks, n_bins] Spearman alignment matrix, averaged over sessions
    (NaN-omitted). Also returns the diagonal (matched relative-time
    position) trace and the argmax cell."""
    from brainalign_wm.analysis.run_all import _maintenance_condition_fn, _model_dpca_patterns, _neural_dpca_patterns
    from brainalign_wm.analysis.rdm import crossnobis_rdm
    from brainalign_wm.analysis.rsa import compare_rdms

    all_trials = dandi_data.trials()
    model_sessions = sorted(set(df["session"].unique()))[:n_sessions]
    mats = []
    n_ticks = n_bins = None
    for session_id in model_sessions:
        session_trials = all_trials[all_trials.session == session_id]
        if len(session_trials) == 0:
            continue
        condition_fn = _maintenance_condition_fn(session_trials)
        mp, ml = _model_dpca_patterns(df, session_id, condition_fn)
        if len(mp) < 8 or len(set(ml)) < 2:
            continue
        npat, nl = _neural_dpca_patterns(dandi_data, session_id, region, condition_fn)
        if len(npat) < 8 or len(set(nl)) < 2 or ml != nl:
            continue  # `_model_dpca_patterns`/`_neural_dpca_patterns` iterate the same session_trials order
        model_arr = np.stack(mp)   # [n_trials, n_ticks, n_units_model]
        neural_arr = np.stack(npat)  # [n_trials, n_bins, n_units_neural]
        n_ticks, n_bins = model_arr.shape[1], neural_arr.shape[1]

        model_rdms = [crossnobis_rdm(model_arr[:, t, :], ml, n_folds=n_folds, seed=seed)[0] for t in range(n_ticks)]
        neural_rdms = [crossnobis_rdm(neural_arr[:, b, :], nl, n_folds=n_folds, seed=seed)[0] for b in range(n_bins)]
        mat = np.full((n_ticks, n_bins), np.nan)
        for t in range(n_ticks):
            for b in range(n_bins):
                mat[t, b] = compare_rdms(model_rdms[t], neural_rdms[b])
        mats.append(mat)

    if not mats:
        return None, None, None
    stacked = np.stack(mats)  # [n_sessions_ok, n_ticks, n_bins]
    mean_mat = np.nanmean(stacked, axis=0)

    # "Diagonal": matched RELATIVE time position -- the maintain epoch has
    # n_ticks model steps and n_bins neural bins covering the SAME nominal
    # window (`configs/config.yaml`'s `bin_ms` against
    # `EPOCH_WINDOWS_S["maintain"]`), so tick t's matched bin is the
    # nearest bin at the same fractional position, not literal index t.
    diag_idx = _matched_bin_indices(n_ticks, n_bins)
    diag_vals = mean_mat[np.arange(n_ticks), diag_idx]
    argmax_flat = int(np.nanargmax(mean_mat))
    argmax_t, argmax_b = np.unravel_index(argmax_flat, mean_mat.shape)
    return mean_mat, diag_vals, (int(argmax_t), int(argmax_b), float(mean_mat[argmax_t, argmax_b]))


# ---------------------------------------------------------------- (b) ----

def within_load_alignment(run_id: str, dandi_data, df, region, n_sessions: int, n_folds: int = 4, seed: int = 0):
    """Whole-epoch-mean alignment computed SEPARATELY within each load
    stratum (not pooled, not cross-load-stratified) -- the residual
    structure the load-stratified estimator is meant to remove, checked
    directly by never forming a cross-load pair in the first place."""
    from brainalign_wm.analysis.run_all import _model_epoch_patterns, _maintenance_condition_fn
    from brainalign_wm.analysis.rdm import crossnobis_rdm
    from brainalign_wm.analysis.rsa import compare_rdms
    from brainalign_wm.analysis import rsa as rsa_mod

    all_trials = dandi_data.trials()
    model_sessions = sorted(set(df["session"].unique()))[:n_sessions]
    by_load: dict[int, list[float]] = {1: [], 2: [], 3: []}
    for session_id in model_sessions:
        session_trials = all_trials[all_trials.session == session_id]
        if len(session_trials) == 0:
            continue
        condition_fn = _maintenance_condition_fn(session_trials)
        sess_df = df[df["session"] == session_id]
        model_patterns, model_labels = _model_epoch_patterns(sess_df, "maintain", condition_fn)
        if len(model_patterns) == 0:
            continue
        neural_data, neural_session_trials = rsa_mod._session_trial_patterns(
            dandi_data, session_id, region, "maintain", dandi_data.bin_ms
        )
        if neural_data is None:
            continue
        neural_labels = [condition_fn(row) for row in neural_session_trials.itertuples()]

        for load in (1, 2, 3):
            m_idx = [i for i, l in enumerate(model_labels) if l[1] == load]
            n_idx = [i for i, l in enumerate(neural_labels) if l[1] == load]
            if len(m_idx) < 8 or len(n_idx) < 8:
                continue
            m_labels_load = [model_labels[i] for i in m_idx]
            n_labels_load = [neural_labels[i] for i in n_idx]
            if len(set(m_labels_load)) < 2 or len(set(n_labels_load)) < 2:
                continue
            m_rdm, m_conds = crossnobis_rdm(model_patterns[m_idx], m_labels_load, n_folds=n_folds, seed=seed)
            n_rdm, n_conds = crossnobis_rdm(neural_data[n_idx], n_labels_load, n_folds=n_folds, seed=seed)
            shared = sorted(set(m_conds) & set(n_conds), key=str)
            if len(shared) < 3:
                continue
            mi = [m_conds.index(c) for c in shared]
            ni = [n_conds.index(c) for c in shared]
            by_load[load].append(compare_rdms(m_rdm[np.ix_(mi, mi)], n_rdm[np.ix_(ni, ni)]))
    return {load: (float(np.mean(vals)) if vals else None, len(vals)) for load, vals in by_load.items()}


# ---------------------------------------------------------------- (c) ----

def positive_control(dandi_data, region, n_sessions: int, noise_frac: float = 0.3, seed: int = 0,
                      n_folds: int = 4) -> tuple[float | None, int]:
    """Construct a "model" side directly from real neural data plus
    independent per-observation Gaussian noise (same condition labels as
    the real data) -- genuinely related to the neural side by
    construction, so the SAME estimator this project reports from must
    return a clearly positive number here, or the estimator itself (not
    the science) is what needs fixing."""
    from brainalign_wm.analysis.run_all import _maintenance_condition_fn
    from brainalign_wm.analysis.rdm import crossnobis_rdm
    from brainalign_wm.analysis.rsa import compare_rdms
    from brainalign_wm.analysis import rsa as rsa_mod

    rng = np.random.RandomState(seed)
    all_trials = dandi_data.trials()
    sessions = sorted(dandi_data.sessions())[:n_sessions] if hasattr(dandi_data, "sessions") else \
        sorted(all_trials["session"].unique())[:n_sessions]
    vals = []
    for session_id in sessions:
        session_trials = all_trials[all_trials.session == session_id]
        if len(session_trials) == 0:
            continue
        condition_fn = _maintenance_condition_fn(session_trials)
        neural_data, neural_session_trials = rsa_mod._session_trial_patterns(
            dandi_data, session_id, region, "maintain", dandi_data.bin_ms
        )
        if neural_data is None or len(neural_data) < 8:
            continue
        labels = [condition_fn(row) for row in neural_session_trials.itertuples()]
        if len(set(labels)) < 2:
            continue
        noisy = neural_data + rng.randn(*neural_data.shape) * neural_data.std() * noise_frac
        real_rdm, real_conds = crossnobis_rdm(neural_data, labels, n_folds=n_folds, seed=seed)
        noisy_rdm, noisy_conds = crossnobis_rdm(noisy, labels, n_folds=n_folds, seed=seed)
        if real_conds != noisy_conds:
            continue
        vals.append(compare_rdms(real_rdm, noisy_rdm))
    return (float(np.mean(vals)) if vals else None), len(vals)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="FLATGRU_SUP_s0")
    ap.add_argument("--checkpoint", default="ckpt.pt")
    ap.add_argument("--region", default=None, help="None (pooled), 'MTL', or 'MFC'")
    ap.add_argument("--n-sessions", type=int, default=15)
    args = ap.parse_args(argv)

    cfg = _load_cfg()
    dandi_data = _dandi_data(cfg)
    df = _model_df(args.run_id, args.checkpoint)
    region = args.region

    t0 = time.time()
    print(f"[diagnose] (a) tick x bin matrix, run={args.run_id}, region={region!r}, n_sessions={args.n_sessions}")
    mat, diag_vals, argmax = tick_bin_matrix(args.run_id, dandi_data, df, region, args.n_sessions)
    if mat is None:
        print("[diagnose]   no usable sessions.")
    else:
        out_csv = ROOT / "results" / f"tick_bin_alignment_{args.run_id}.csv"
        pd.DataFrame(mat).to_csv(out_csv, index_label="model_tick")
        print(f"[diagnose]   wrote {out_csv} (shape {mat.shape})")
        print(f"[diagnose]   diagonal (matched relative time) mean={np.nanmean(diag_vals):.4f}, trace={np.round(diag_vals, 3).tolist()}")
        print(f"[diagnose]   argmax cell: tick={argmax[0]} bin={argmax[1]} value={argmax[2]:.4f}")
        print(f"[diagnose]   whole-matrix mean={np.nanmean(mat):.4f}, whole-matrix max={np.nanmax(mat):.4f}, min={np.nanmin(mat):.4f}")
    print(f"[diagnose]   elapsed {time.time()-t0:.1f}s\n")

    t0 = time.time()
    print(f"[diagnose] (b) within-load alignment, run={args.run_id}, region={region!r}")
    by_load = within_load_alignment(args.run_id, dandi_data, df, region, args.n_sessions)
    for load, (val, n) in by_load.items():
        print(f"[diagnose]   load {load}: {'n/a' if val is None else f'{val:.4f}'} (n={n} sessions)")
    print(f"[diagnose]   elapsed {time.time()-t0:.1f}s\n")

    t0 = time.time()
    print(f"[diagnose] (c) positive control (real neural data vs. itself + noise), region={region!r}")
    pos_val, pos_n = positive_control(dandi_data, region, args.n_sessions)
    print(f"[diagnose]   {'n/a' if pos_val is None else f'{pos_val:.4f}'} (n={pos_n} sessions)")
    print(f"[diagnose]   elapsed {time.time()-t0:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
