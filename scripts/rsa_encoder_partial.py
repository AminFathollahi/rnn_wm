"""Partial RSA: how much of a trained network's neural alignment survives
controlling for its frozen ResNet-18 front end's own RDM.

The probe-epoch RSA baseline (`results/alignment_probe_by_region.csv`)
correlates a trained model's RDM with each brain region's RDM. Some of
that correlation is inherited from the visual input alone, before any
recurrent computation happens. This script partials the encoder's RDM
out of both sides (`task_structure.semipartial_correlation`, the same
quantile-rank estimator `measurement_validation.py` uses for the
task-structure covariate) and ranks regions by what is left.

Two models are analysed: the best-aligned and the best-performing run,
selected from `results/alignment_results.csv` / `results/performance_by_run.csv`
among runs that cleared the accuracy inclusion criterion (see `select_runs`).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from brainalign_wm.analysis.pseudopopulation import build_condition_fold_means, cv_euclidean_rdm, _unit_session
from brainalign_wm.analysis.rdm import crossnobis_rdm, vectorize_upper
from brainalign_wm.analysis.rsa import compare_rdms, normalized_alignment
from brainalign_wm.analysis.run_all import (
    MIN_SHARED_CONDITIONS, MIN_TRIALS_PER_CONDITION, REGIONS,
    _coarse_condition, _encoder_only_patterns_for_session, _filter_min_trials,
    _probe_alignment_for_run, _shared_conditions_or_none,
)
from brainalign_wm.analysis.task_structure import semipartial_correlation
from brainalign_wm.config import get_path, load_config

RESULTS = get_path("results")
FINEST_REGIONS = [r for r in REGIONS if r not in (None, "MTL", "MFC")]
COLLINEARITY_GUARD = 0.95  # |corr(model, encoder)| beyond this makes the partial numerically unstable
N_BOOT = 500
Z_ALPHA_POWER = 1.96 + 0.84  # two-sided alpha=0.05, power=0.80


def select_runs() -> dict[str, str]:
    """Best-aligned and best-performing run ids, among runs that cleared the
    accuracy inclusion criterion (excludes chance-level runs, whose raw
    alignment can spuriously exceed a noisy ceiling estimate)."""
    align = pd.read_csv(RESULTS / "alignment_results.csv")
    perf = pd.read_csv(RESULTS / "performance_by_run.csv")
    df = align.merge(perf[["run_id", "criterion_met_at_budget"]], on="run_id", how="left")
    ok = df[(df.probe_status == "ok") & (df.criterion_met_at_budget == True)].copy()  # noqa: E712
    best_aligned = ok.sort_values("probe_raw_alignment", ascending=False).iloc[0]["run_id"]
    ok["perf"] = (ok.accuracy_load1 + ok.accuracy_load3) / 2
    best_perf = ok.sort_values("perf", ascending=False).iloc[0]["run_id"]
    if best_perf == best_aligned:
        best_perf = ok.sort_values("perf", ascending=False).iloc[1]["run_id"]
    return {"best_aligned": best_aligned, "best_performing": best_perf}


def _encoder_probe_rdm(model_df: pd.DataFrame, dandi_data) -> tuple[np.ndarray | None, list | None]:
    """Encoder-only RDM over the probe epoch's coarse conditions, pooled
    across sessions the same way the model RDM is (crossnobis over
    per-trial encoder patterns, no working-memory processing)."""
    all_trials = dandi_data.trials()
    patterns, labels = [], []
    for session_id in sorted(model_df.session.unique()):
        session_trials = all_trials[all_trials.session == session_id]
        p, l = _encoder_only_patterns_for_session(session_id, session_trials, _coarse_condition)
        if len(p):
            patterns.append(p)
            labels.extend(l)
    if not patterns:
        return None, None
    patterns = np.concatenate(patterns, axis=0)
    patterns, labels = _filter_min_trials(patterns, labels, MIN_TRIALS_PER_CONDITION)
    if len(set(labels)) < 2:
        return None, None
    n_folds = max(2, min(4, min(labels.count(c) for c in set(labels))))
    return crossnobis_rdm(patterns, labels, n_folds=n_folds)


def _patient_bootstrap(fold_means, all_conditions, condition_index, unit_patients, stat_fns, n_boot=N_BOOT, seed=0):
    """Patient-clustered bootstrap: resample patients on the neural unit
    axis, rebuild the pooled neural RDM (`pseudopopulation.cv_euclidean_rdm`,
    the estimator `raw_alignment` itself is built from), and apply each of
    `stat_fns` to the resampled, shared-condition-restricted RDM. One
    resampling pass serves every statistic (raw, encoder, partial) so the
    intervals share the same draws."""
    unique = np.unique(unit_patients)
    index_by_patient = {p: np.where(unit_patients == p)[0] for p in unique}
    rng = np.random.RandomState(seed)
    values = {name: np.empty(n_boot) for name in stat_fns}
    for b in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        cols = np.concatenate([index_by_patient[p] for p in drawn])
        rdm = cv_euclidean_rdm(fold_means[:, :, cols], all_conditions)
        neural_sub = rdm[np.ix_(condition_index, condition_index)]
        for name, fn in stat_fns.items():
            values[name][b] = fn(neural_sub)
    return values


def _ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def analyse_run(run_id: str, dandi_data) -> list[dict]:
    from brainalign_wm.training.generate_activity_logs import activity_log_path, generate_activity_log
    from brainalign_wm.training.logging_schema import read_log

    log_path = activity_log_path(run_id)
    if not log_path.exists():
        log_path = generate_activity_log(run_id, dandi_data)
    model_df = read_log(log_path)

    encoder_rdm, encoder_conds = _encoder_probe_rdm(model_df, dandi_data)
    rows = []
    for region in FINEST_REGIONS:
        row = {"run_id": run_id, "region": region}
        obs = _probe_alignment_for_run(run_id, model_df, dandi_data, region, return_rdms=True)
        if obs.get("status") != "ok" or encoder_rdm is None:
            rows.append({**row, "status": obs.get("status", "no_encoder_features")})
            continue

        model_rdm, neural_rdm, conds = obs["model_rdm"], obs["neural_rdm"], obs["conditions"]
        shared = _shared_conditions_or_none(encoder_conds, conds, floor=MIN_SHARED_CONDITIONS)
        if shared is None:
            rows.append({**row, "status": "insufficient_shared_conditions"})
            continue
        c_idx = [conds.index(c) for c in shared]
        e_idx = [encoder_conds.index(c) for c in shared]
        model_sub = model_rdm[np.ix_(c_idx, c_idx)]
        neural_sub = neural_rdm[np.ix_(c_idx, c_idx)]
        encoder_sub = encoder_rdm[np.ix_(e_idx, e_idx)]
        model_vec, encoder_vec = vectorize_upper(model_sub), vectorize_upper(encoder_sub)

        raw = compare_rdms(neural_sub, model_sub)
        encoder_corr = compare_rdms(neural_sub, encoder_sub)
        model_encoder_corr = compare_rdms(model_sub, encoder_sub)
        collinear = abs(model_encoder_corr) > COLLINEARITY_GUARD

        row.update({
            "status": "ok", "n_shared_conditions": len(shared),
            "raw_correlation": raw, "encoder_correlation": encoder_corr,
            "model_encoder_correlation": model_encoder_corr,
            "noise_ceiling_upper": obs["noise_ceiling_upper"],
            "normalized_alignment": normalized_alignment(raw, obs["noise_ceiling_upper"]),
        })
        if collinear:
            row["partial_status"] = "unstable_covariate_collinear"
            row["partial_correlation"] = float("nan")
            rows.append(row)
            continue

        fold_means, all_conditions = build_condition_fold_means(
            dandi_data, region, dandi_data.bin_ms, "probe", _coarse_condition, n_folds=4, seed=0
        )
        condition_index = [all_conditions.index(c) for c in shared]
        unit_patients = np.array([dandi_data.patient_of(_unit_session(u)) for u in dandi_data.units(region)])

        stat_fns = {
            "raw": lambda n: compare_rdms(n, model_sub),
            "encoder": lambda n: compare_rdms(n, encoder_sub),
            "partial": lambda n: semipartial_correlation(vectorize_upper(n), model_vec, encoder_vec),
        }
        boot = _patient_bootstrap(fold_means, all_conditions, condition_index, unit_patients, stat_fns)
        raw_ci, enc_ci, partial_ci = _ci(boot["raw"]), _ci(boot["encoder"]), _ci(boot["partial"])

        row.update({
            "partial_status": "ok",
            "partial_correlation": semipartial_correlation(vectorize_upper(neural_sub), model_vec, encoder_vec),
            "raw_ci_lower": raw_ci[0], "raw_ci_upper": raw_ci[1],
            "encoder_ci_lower": enc_ci[0], "encoder_ci_upper": enc_ci[1],
            "partial_ci_lower": partial_ci[0], "partial_ci_upper": partial_ci[1],
            "partial_boot_se": float(np.std(boot["partial"])),
            "n_patients": len(set(unit_patients)),
        })
        rows.append(row)
    return rows


def add_ranks_and_mdd(df: pd.DataFrame) -> pd.DataFrame:
    """Rank regions by partial correlation within each run and report the
    minimum detectable difference (MDD) between each region and the next-
    ranked one: the smallest true gap a two-sided test at alpha=0.05 with
    80% power could reliably detect, given the two regions' bootstrap SEs."""
    df = df.copy()
    df["rank"] = np.nan
    df["mdd_to_next_rank"] = np.nan
    for run_id, g in df.groupby("run_id"):
        ok = g[g.partial_status == "ok"].sort_values("partial_correlation", ascending=False)
        ranks = np.arange(1, len(ok) + 1)
        df.loc[ok.index, "rank"] = ranks
        se = ok["partial_boot_se"].to_numpy()
        for pos, idx in enumerate(ok.index[:-1]):
            df.loc[idx, "mdd_to_next_rank"] = Z_ALPHA_POWER * float(np.hypot(se[pos], se[pos + 1]))
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", default=None, help="override: comma-separated criterion=run_id pairs, e.g. best_aligned=M00100_RL_s2")
    args = ap.parse_args(argv)

    runs = select_runs() if args.runs is None else dict(pair.split("=") for pair in args.runs.split(","))
    print(f"[rsa_encoder_partial] runs selected: {runs}")

    from brainalign_wm.config import DEFAULT_CONFIG_PATH
    from brainalign_wm.neural.adapters.dandi_nwb import DandiSternbergTierA

    cfg = load_config(args.config or DEFAULT_CONFIG_PATH)
    dandi_data = DandiSternbergTierA(
        cfg["paths"]["data_root"], datasets=tuple(cfg["neural"]["datasets_tierA"]),
        min_firing_hz=cfg["neural"]["min_firing_hz"], min_session_accuracy=cfg["neural"]["min_session_accuracy"],
        bin_ms=cfg["neural"]["bin_ms"],
    )

    rows = []
    for criterion, run_id in runs.items():
        print(f"[rsa_encoder_partial] analysing {criterion} = {run_id} ...")
        for row in analyse_run(run_id, dandi_data):
            rows.append({**row, "criterion": criterion})

    df = pd.DataFrame(rows)
    df = add_ranks_and_mdd(df)
    out_path = RESULTS / "rsa_encoder_partial.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[rsa_encoder_partial] wrote {out_path}")

    for run_id, g in df.groupby("run_id"):
        ok = g[g.partial_status == "ok"].sort_values("rank")
        crit = g.criterion.iloc[0]
        print(f"\n=== {crit} ({run_id}) -- region ranking by partial correlation (encoder removed) ===")
        cols = ["rank", "region", "raw_correlation", "encoder_correlation", "partial_correlation",
                "partial_ci_lower", "partial_ci_upper", "mdd_to_next_rank", "n_patients"]
        print(ok[cols].to_string(index=False) if len(ok) else "(no region cleared the shared-condition / collinearity guards)")
        bad = g[g.partial_status != "ok"]
        if len(bad):
            print("skipped:", dict(zip(bad.region, bad.partial_status.fillna(bad.status))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
