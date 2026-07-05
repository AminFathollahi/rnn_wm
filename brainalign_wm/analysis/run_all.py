"""Runs the alignment and statistical-inference pipeline over whatever
training runs have completed so far (reads `results/manifest.jsonl`; does
not require the full grid to have finished).

Audit-fix rewrite (A2a-A2g, B3, C2): two epoch-appropriate condition
schemas, each on the representation appropriate to it (A2c: raw alignment
and its noise ceiling are always computed on the SAME representation):

  * **maintenance** (delay-period) alignment uses a CATEGORY-based
    condition -- the sorted multiset of imputed held-item categories + load
    (A2a/A2b -- NOT the probe/response-time in_set/correct labels, which
    cannot causally structure activity that occurred before the probe).
    Raw item-PicID identity (comments.txt's original A2a ask) turned out to
    have essentially zero repeated conditions on the real datasets (every
    held item/set in a representative session's load-1 trials was unique)
    -- categories are imputed via nearest-neighbor in the same frozen
    encoder's feature space against the model's own training-pool category
    centroids (see `analysis/stimulus_categories.py`), which DO recur.
    Category assignment (and hence the condition set) is session-specific,
    so this is computed PER SESSION: a per-session crossnobis RDM (model
    replay vs. that session's neural trials), against that session's OWN
    within-session reliability ceiling (`rsa.within_session_noise_ceiling`
    -- a cross-session LOSO ceiling is meaningless here since the model
    replay only covers sessions with stimulus-cache coverage). Per-session
    rows carry the session's patient (via `patient_of`), directly feeding
    the LME's patient factor (B3).
  * **probe-epoch** alignment uses the coarse (load, in_set, correct)
    schema, which DOES generalize across sessions, and is pooled --
    correctly, via `analysis/pseudopopulation.py` (A2d: each unit's
    condition mean uses only that unit's own session's trials, never
    diluted by another session's zero-filled entries), with a ceiling
    computed on that SAME pooled representation (A2c).

Both schemas are computed at the pooled-region level (region=None) and at
each region-family level (MTL, MFC -- audit fix C2), feeding a region factor
into the mixed-effects model.

A degenerate-RDM guard (A2f) requires >= `MIN_SHARED_CONDITIONS` shared
conditions before reporting a score; below that, the row is marked
"insufficient_shared_conditions" rather than emitting a number.

Results are written to `results/alignment_results.csv` (one row per
run x epoch x region, aggregated across sessions for the per-session path)
and `results/alignment_by_session.csv` (long format, one row per
run x session x region for the maintenance path -- carries the patient
factor for the LME).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

# crossnobis needs multiple trials per condition per cross-validation fold;
# a condition with too few trials gets a fold with zero members, and
# `crossnobis_rdm` fills those undefined entries with 0.0 rather than NaN
# (see `rdm.py`), which silently injects fake zero-distance entries and
# corrupts the whole-RDM correlation. Filtering to conditions with at least
# this many trials avoids that failure mode.
MIN_TRIALS_PER_CONDITION = 8

# Audit fix A2f: with too few shared conditions, a Spearman correlation over
# the RDM's off-diagonal is close to meaningless (e.g. 4 conditions -> 6
# off-diagonal cells). Raised from the original floor of 3; below this,
# fail loudly (report a status, not a score) rather than emit a number.
MIN_SHARED_CONDITIONS = 8

REGIONS = [None, "MTL", "MFC"]


def _filter_min_trials(patterns: np.ndarray, labels: list[tuple], min_count: int) -> tuple[np.ndarray, list[tuple]]:
    counts: dict[tuple, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    keep = [i for i, l in enumerate(labels) if counts[l] >= min_count]
    return patterns[keep], [labels[i] for i in keep]


def _load_completed_runs(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    latest: dict[str, dict] = {}
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[rec.get("run_id", "?")] = rec
    return [r for r in latest.values() if r.get("status") == "completed"]


def _category_condition_fn(pid_to_cat: dict) -> callable:
    """Maintenance-epoch condition (audit fix A2a/A2b, revised scope -- see
    `analysis/stimulus_categories.py`): the SORTED MULTISET of imputed
    held-item categories + load, NOT the probe/response-time in_set/correct
    labels (which cannot causally structure activity that occurred before
    the probe), and NOT raw item identity (empirically infeasible on this
    real dataset -- see module docstring). `pid_to_cat` is a per-session
    mapping (categories are imputed per-session from that session's own
    cached stimulus features)."""
    def _fn(row) -> tuple:
        cats = tuple(sorted(pid_to_cat.get(str(int(x)), "?") for x in row.held_items))
        return (cats, int(row.load))
    return _fn


def _coarse_condition(row) -> tuple:
    """Probe-epoch condition (audit fix A2b): a property of the
    response/decision itself, legitimately poolable across sessions. The
    model's activity log (`in_set`) and the NWB trial table (`probe_in_set`)
    name the same field differently -- accept either."""
    in_set = row.probe_in_set if hasattr(row, "probe_in_set") else row.in_set
    return (int(row.load), bool(in_set), bool(row.correct))


def _model_epoch_patterns(df: pd.DataFrame, epoch: str, condition_fn) -> tuple[np.ndarray, list[tuple]]:
    """df: one run's activity log (optionally pre-filtered to one session).
    Returns per-trial epoch-mean activity ([n_trials, n_units]) and matching
    condition labels. The model's hidden units are shared across all
    replayed sessions (unlike real neurons), so pooling trials across
    sessions here needs no special pseudopopulation handling.

    Groups by `(session, trial_id)`, NOT `trial_id` alone: `trial_id` is
    assigned per-session by `generate_activity_logs.replay_session`
    (`enumerate(session_trials.itertuples())`, reset to 0 for every
    session), so it is NOT globally unique across the full log. Grouping by
    `trial_id` alone (a real bug found while validating the probe-epoch
    pooled path against real data) silently merged activity from
    DIFFERENT trials in DIFFERENT sessions that happened to share the same
    trial index into a single averaged "trial" -- corrupting every pooled
    condition pattern. The maintenance-epoch path never hit this because it
    pre-filters `model_df` to one session before calling this function; the
    probe-epoch pooled path passes the full multi-session log directly."""
    sub = df[df.epoch == epoch]
    if len(sub) == 0:
        return np.zeros((0, 0)), []
    is_flat = sub["h_flat"].iloc[0] is not None
    group_cols = ["session", "trial_id"] if "session" in sub.columns else ["trial_id"]
    patterns, labels = [], []
    for _key, g in sub.groupby(group_cols):
        if is_flat:
            vec = np.stack(g["h_flat"].to_numpy()).mean(axis=0)
        else:
            vec = np.concatenate(
                [np.stack(g["h_worker"].to_numpy()).mean(axis=0), np.stack(g["h_manager"].to_numpy()).mean(axis=0)]
            )
        row0 = g.iloc[0]
        patterns.append(vec)
        labels.append(condition_fn(row0))
    return np.array(patterns), labels


def _shared_conditions_or_none(a_conds: list, b_conds: list, floor: int = MIN_SHARED_CONDITIONS) -> list:
    shared = sorted(set(a_conds) & set(b_conds), key=str)
    return shared if len(shared) >= floor else None


# Maintenance-epoch conditions are (sorted category multiset, load); with
# `n_categories=4` and `loads<=3`, the achievable condition space is much
# smaller than the probe epoch's -- the blanket MIN_SHARED_CONDITIONS floor
# (tuned for the coarser but higher-cardinality probe schema) would reject
# every maintenance row. Use a schema-appropriate floor instead (still a
# real guard against a degenerate 1-2 condition RDM, A2f).
MIN_SHARED_CONDITIONS_MAINTENANCE = 4


def _maintenance_alignment_for_run(run_id: str, model_df: pd.DataFrame, dandi_data, region, category_centroids) -> list[dict]:
    """Per-session, category-based maintenance-epoch alignment (see
    `_category_condition_fn`). Returns one row per session with usable data
    (both model replay coverage and neural trials for that session)."""
    from brainalign_wm.analysis.rsa import compare_rdms, within_session_noise_ceiling
    from brainalign_wm.analysis import rsa as rsa_mod
    from brainalign_wm.analysis.rdm import crossnobis_rdm
    from brainalign_wm.analysis.stimulus_categories import impute_categories_for_session

    rows = []
    model_sessions = set(model_df["session"].unique()) if "session" in model_df.columns else set()
    for session_id in sorted(model_sessions):
        pid_to_cat = impute_categories_for_session(session_id, category_centroids)
        if not pid_to_cat:
            continue
        condition_fn = _category_condition_fn(pid_to_cat)

        sess_df = model_df[model_df["session"] == session_id]
        model_patterns, model_labels = _model_epoch_patterns(sess_df, "maintain", condition_fn)
        model_patterns, model_labels = _filter_min_trials(model_patterns, model_labels, min_count=2)
        if len(set(model_labels)) < 2:
            continue
        n_folds_model = max(2, min(4, min(model_labels.count(l) for l in set(model_labels))))
        if n_folds_model < 2:
            continue
        model_rdm, model_conds = crossnobis_rdm(model_patterns, model_labels, n_folds=n_folds_model)

        neural_rdm, neural_conds = rsa_mod._session_condition_rdm(
            dandi_data, session_id, region, "maintain", dandi_data.bin_ms, condition_fn=condition_fn,
        )
        if neural_rdm is None:
            continue

        shared = _shared_conditions_or_none(model_conds, neural_conds, floor=MIN_SHARED_CONDITIONS_MAINTENANCE)
        if shared is None:
            rows.append({"run_id": run_id, "session": session_id, "region": region or "pooled",
                          "status": "insufficient_shared_conditions", "n_shared_conditions": len(set(model_conds) & set(neural_conds))})
            continue
        m_idx = [model_conds.index(c) for c in shared]
        n_idx = [neural_conds.index(c) for c in shared]
        raw = compare_rdms(model_rdm[np.ix_(m_idx, m_idx)], neural_rdm[np.ix_(n_idx, n_idx)])
        ceiling_lower, ceiling_upper = within_session_noise_ceiling(
            dandi_data, session_id, region, "maintain", dandi_data.bin_ms, condition_fn=condition_fn,
        )
        rows.append({
            "run_id": run_id, "session": session_id, "region": region or "pooled",
            "patient": dandi_data.patient_of(session_id), "status": "ok",
            "raw_alignment": raw, "noise_ceiling_upper": ceiling_upper, "noise_ceiling_lower": ceiling_lower,
            "normalized_alignment": _norm(raw, ceiling_upper), "n_shared_conditions": len(shared),
        })
    return rows


def _norm(raw: float, ceiling_upper: float) -> float:
    from brainalign_wm.analysis.rsa import normalized_alignment

    return normalized_alignment(raw, ceiling_upper)


def _probe_alignment_for_run(run_id: str, model_df: pd.DataFrame, dandi_data, region) -> dict:
    """Pooled coarse-condition probe-epoch alignment (audit fix A2d/A2c)."""
    from brainalign_wm.analysis.rdm import crossnobis_rdm
    from brainalign_wm.analysis.rsa import compare_rdms
    from brainalign_wm.analysis.pseudopopulation import pooled_condition_rdm, pooled_noise_ceiling

    model_patterns, model_labels = _model_epoch_patterns(model_df, "probe", _coarse_condition)
    model_patterns, model_labels = _filter_min_trials(model_patterns, model_labels, MIN_TRIALS_PER_CONDITION)
    if len(set(model_labels)) < 2:
        return {"run_id": run_id, "region": region or "pooled", "status": "too_few_conditions"}
    n_folds = max(2, min(4, min(model_labels.count(l) for l in set(model_labels))))
    model_rdm, model_conds = crossnobis_rdm(model_patterns, model_labels, n_folds=n_folds)

    neural_rdm, neural_conds = pooled_condition_rdm(dandi_data, region, dandi_data.bin_ms, "probe", _coarse_condition, n_folds=4, seed=0)

    shared = _shared_conditions_or_none(model_conds, neural_conds)
    if shared is None:
        return {"run_id": run_id, "region": region or "pooled", "status": "insufficient_shared_conditions",
                "n_shared_conditions": len(set(model_conds) & set(neural_conds))}
    m_idx = [model_conds.index(c) for c in shared]
    n_idx = [neural_conds.index(c) for c in shared]
    raw = compare_rdms(model_rdm[np.ix_(m_idx, m_idx)], neural_rdm[np.ix_(n_idx, n_idx)])
    ceiling_lower, ceiling_upper = pooled_noise_ceiling(dandi_data, region, dandi_data.bin_ms, "probe", _coarse_condition, n_resamples=10, seed=0)
    return {
        "run_id": run_id, "region": region or "pooled", "status": "ok",
        "raw_alignment": raw, "noise_ceiling_upper": ceiling_upper, "noise_ceiling_lower": ceiling_lower,
        "normalized_alignment": _norm(raw, ceiling_upper), "n_shared_conditions": len(shared),
    }


def align_one_run(run_id: str, dandi_data, category_centroids, force_regenerate: bool = False) -> dict:
    """Generates/loads the run's activity log and computes both the
    per-session maintenance alignment and the pooled probe alignment, at
    each region level. Returns {"maintenance": [rows...], "probe": [rows...]}."""
    from brainalign_wm.training.generate_activity_logs import generate_activity_log
    from brainalign_wm.training.logging_schema import read_log

    log_path = ROOT / "results" / "activity_logs" / f"{run_id}.parquet"
    if force_regenerate or not log_path.exists():
        log_path = generate_activity_log(run_id, dandi_data)
    df = read_log(log_path)

    maintenance_rows, probe_rows = [], []
    for region in REGIONS:
        maintenance_rows.extend(_maintenance_alignment_for_run(run_id, df, dandi_data, region, category_centroids))
        probe_rows.append(_probe_alignment_for_run(run_id, df, dandi_data, region))
    return {"maintenance": maintenance_rows, "probe": probe_rows}


def reflection_shuffle_lesion_for_run(run_id: str, dandi_data, category_centroids) -> dict:
    """C1: H2's causal claim ("reflection-shuffle abolishes the M effect").
    Compares pooled-region maintenance alignment between the normal
    activity log and a version replayed with each trial's own R_t sequence
    time-shuffled within the trial (`generate_activity_logs.
    generate_activity_log_reflection_shuffled`). Only defined for M=1 runs.
    Returns a summary dict; if H2 holds, `shuffled_normalized_alignment`
    should be markedly lower than `normal_normalized_alignment`."""
    from brainalign_wm.training.generate_activity_logs import generate_activity_log_reflection_shuffled
    from brainalign_wm.training.logging_schema import read_log

    normal_log = ROOT / "results" / "activity_logs" / f"{run_id}.parquet"
    if not normal_log.exists():
        return {"run_id": run_id, "status": "normal_log_missing"}
    normal_df = read_log(normal_log)
    normal_rows = _maintenance_alignment_for_run(run_id, normal_df, dandi_data, None, category_centroids)
    normal_ok = [r for r in normal_rows if r.get("status") == "ok"]

    shuffled_path = generate_activity_log_reflection_shuffled(run_id, dandi_data)
    shuffled_df = read_log(shuffled_path)
    shuffled_rows = _maintenance_alignment_for_run(run_id, shuffled_df, dandi_data, None, category_centroids)
    shuffled_ok = [r for r in shuffled_rows if r.get("status") == "ok"]

    if not normal_ok or not shuffled_ok:
        return {"run_id": run_id, "status": "insufficient_sessions",
                "n_normal_sessions_ok": len(normal_ok), "n_shuffled_sessions_ok": len(shuffled_ok)}
    normal_norm = float(np.mean([r["normalized_alignment"] for r in normal_ok]))
    shuffled_norm = float(np.mean([r["normalized_alignment"] for r in shuffled_ok]))
    return {
        "run_id": run_id, "status": "ok",
        "normal_normalized_alignment": normal_norm, "shuffled_normalized_alignment": shuffled_norm,
        "lesion_effect": normal_norm - shuffled_norm,
        "n_normal_sessions_ok": len(normal_ok), "n_shuffled_sessions_ok": len(shuffled_ok),
    }


MAX_SESSIONS_FOR_DYNAMICS = 20


def dynamics_and_persistence_for_run(run_id: str, model_df: pd.DataFrame, dandi_data, max_sessions: int = MAX_SESSIONS_FOR_DYNAMICS) -> dict:
    """H5/H6 (audit fix C3/C4): aggregates per-session stability-index
    (dynamic vs. stable delay coding) and persistent-activity-index
    (memoranda-selective persistence) comparisons across every session with
    usable model+neural data, and compares the model's vs. the brain's
    distributions (`stats.compare_distributions`, a permutation test +
    rank-biserial effect size). Pooled-region only (region=None) -- this
    already replays/loads every session once per run; looping regions here
    would triple an already expensive per-session computation for limited
    additional value at this exploratory-analysis tier.

    `cross_temporal_decoding` fits a fresh classifier per (fold, timebin)
    pair -- ~90 fits per session per side at this task's ~30-bin maintenance
    window -- so this is capped at `max_sessions` (deterministically, the
    first N alphabetically) to keep H5/H6 tractable across a full 8-cell
    grid rather than scaling linearly with however many sessions happen to
    have replay coverage."""
    from brainalign_wm.analysis.dynamics_and_persistence import stability_index_for_session, persistence_index_for_session
    from brainalign_wm.analysis.stats import compare_distributions

    model_sessions = sorted(model_df["session"].unique())[:max_sessions] if "session" in model_df.columns else []
    model_stab, neural_stab = [], []
    model_pers, neural_pers = [], []
    n_sessions_used = 0
    for session_id in model_sessions:
        stab = stability_index_for_session(model_df, dandi_data, session_id, None, dandi_data.bin_ms)
        pers = persistence_index_for_session(model_df, dandi_data, session_id, None, dandi_data.bin_ms)
        used = False
        if stab is not None:
            model_stab.append(stab["model_stability"])
            neural_stab.append(stab["neural_stability"])
            used = True
        if pers is not None:
            model_pers.extend(np.atleast_1d(pers["model_index"]).tolist())
            neural_pers.extend(np.atleast_1d(pers["neural_index"]).tolist())
            used = True
        n_sessions_used += int(used)

    if n_sessions_used == 0:
        return {"run_id": run_id, "status": "insufficient_sessions"}

    out = {"run_id": run_id, "status": "ok", "n_sessions_used": n_sessions_used}
    if len(model_stab) >= 2 and len(neural_stab) >= 2:
        h5 = compare_distributions(np.array(model_stab), np.array(neural_stab))
        out.update({f"h5_stability_{k}": v for k, v in h5.items()})
        out["h5_model_stability_mean"] = float(np.mean(model_stab))
        out["h5_neural_stability_mean"] = float(np.mean(neural_stab))
    if len(model_pers) >= 2 and len(neural_pers) >= 2:
        h6 = compare_distributions(np.array(model_pers), np.array(neural_pers))
        out.update({f"h6_persistence_{k}": v for k, v in h6.items()})
        out["h6_model_persistence_mean"] = float(np.mean(model_pers))
        out["h6_neural_persistence_mean"] = float(np.mean(neural_pers))
    return out


def chance_control_check(model_id: str, seed: int, dandi_data, category_centroids) -> dict:
    """comments.txt acceptance-gate H5: an untrained (chance) model must
    land near the alignment floor -- wired into the standard `run_all.py`
    output (not a one-off spot check), so this is re-verified every time
    the pipeline runs, not just once during validation."""
    from brainalign_wm.training.generate_activity_logs import generate_chance_activity_log
    from brainalign_wm.training.logging_schema import read_log

    log_path = generate_chance_activity_log(model_id, seed, dandi_data)
    df = read_log(log_path)
    run_id = f"{model_id}_s{seed}_chance"

    maintenance_rows = _maintenance_alignment_for_run(run_id, df, dandi_data, None, category_centroids)
    probe_row = _probe_alignment_for_run(run_id, df, dandi_data, None)
    maintenance_ok = [r for r in maintenance_rows if r.get("status") == "ok"]

    out = {"run_id": run_id, "model_id": model_id}
    if maintenance_ok:
        out["maintenance_normalized_alignment"] = float(np.mean([r["normalized_alignment"] for r in maintenance_ok]))
    if probe_row.get("status") == "ok":
        out["probe_normalized_alignment"] = probe_row["normalized_alignment"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--regenerate", action="store_true", help="force regeneration of activity logs")
    ap.add_argument(
        "--max-sessions-per-dataset", type=int, default=None,
        help="bound the number of sessions loaded per dataset; `dandi_nwb.rates()` recomputes "
             "spike histograms with no caching (unlike sim_brain's), so the full Tier A pool "
             "(~1800 units, thousands of trials) is currently impractically slow -- see DECISIONS.md.",
    )
    ap.add_argument(
        "--skip-reflection-shuffle", action="store_true",
        help="skip the C1 reflection-shuffle causal-control lesion (it re-replays every M=1 run a second time).",
    )
    ap.add_argument(
        "--skip-dynamics", action="store_true",
        help="skip the C3/C4 H5/H6 dynamics-and-persistence analyses (dPCA/encoding are not yet wired here; "
             "cross-temporal decoding + persistence are, and add per-session compute on top of the RSA path).",
    )
    ap.add_argument(
        "--skip-chance-control", action="store_true",
        help="skip the H5 chance-model (untrained checkpoint) negative control -- it replays a full session set.",
    )
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    completed = _load_completed_runs(ROOT / "results" / "manifest.jsonl")
    if not completed:
        print("[run_all] no completed runs found in results/manifest.jsonl; run `make run-grid` first.")
        return 1
    print(f"[run_all] {len(completed)} completed run(s) found: {[r['run_id'] for r in completed]}")

    from brainalign_wm.neural.adapters.dandi_nwb import DandiSternbergTierA

    dandi_data = DandiSternbergTierA(
        cfg["paths"]["data_root"], datasets=tuple(cfg["neural"]["datasets_tierA"]),
        min_firing_hz=cfg["neural"]["min_firing_hz"], bin_ms=cfg["neural"]["bin_ms"],
        max_sessions_per_dataset=args.max_sessions_per_dataset,
    )
    print(f"[run_all] Tier A loaded: {len(dandi_data.sessions())} sessions, {len(dandi_data.units())} units, "
          f"regions={dandi_data.regions()}")

    from brainalign_wm.tasks.image_token_bank import ImageTokenBank
    from brainalign_wm.analysis.stimulus_categories import category_centroids as compute_category_centroids

    image_bank = ImageTokenBank(
        stimuli_root=ROOT / cfg["paths"]["stimuli"], categories=cfg["task"]["categories"],
        feature_cache_path=ROOT / cfg["paths"]["feature_cache"] / "image_token_bank.npy", seed=0,
    )
    category_centroids = compute_category_centroids(image_bank)

    maintenance_session_rows, probe_rows, lesion_rows, dynamics_rows = [], [], [], []
    for rec in completed:
        run_id = rec["run_id"]
        print(f"[run_all] aligning {run_id} ...")
        try:
            result = align_one_run(run_id, dandi_data, category_centroids, force_regenerate=args.regenerate)
        except FileNotFoundError as e:
            print(f"[run_all]   skipped ({e})")
            continue
        for row in result["maintenance"]:
            maintenance_session_rows.append({**row, "model_id": rec["model_id"], "S": rec["S"], "M": rec["M"], "L": rec["L"], "seed": rec["seed"]})
        for row in result["probe"]:
            probe_rows.append({**row, "run_id": run_id, "model_id": rec["model_id"], "S": rec["S"], "M": rec["M"], "L": rec["L"], "seed": rec["seed"],
                                "accuracy_load1": rec.get("accuracy", {}).get("load1"), "accuracy_load3": rec.get("accuracy", {}).get("load3")})
        ok_n = sum(1 for r in result["maintenance"] if r.get("status") == "ok")
        print(f"[run_all]   maintenance: {ok_n}/{len(result['maintenance'])} session-rows ok; "
              f"probe: {[r.get('status') for r in result['probe']]}")

        if rec["M"] == 1 and not args.skip_reflection_shuffle:
            lesion = reflection_shuffle_lesion_for_run(run_id, dandi_data, category_centroids)
            lesion_rows.append({**lesion, "model_id": rec["model_id"], "S": rec["S"], "L": rec["L"], "seed": rec["seed"]})
            print(f"[run_all]   C1 reflection-shuffle lesion: {lesion.get('status')} "
                  f"(effect={lesion.get('lesion_effect')})" if lesion.get("status") == "ok" else
                  f"[run_all]   C1 reflection-shuffle lesion: {lesion.get('status')}")

        if not args.skip_dynamics:
            log_path = ROOT / "results" / "activity_logs" / f"{run_id}.parquet"
            if log_path.exists():
                from brainalign_wm.training.logging_schema import read_log

                dyn = dynamics_and_persistence_for_run(run_id, read_log(log_path), dandi_data)
                dynamics_rows.append({**dyn, "model_id": rec["model_id"], "S": rec["S"], "M": rec["M"], "L": rec["L"], "seed": rec["seed"]})
                print(f"[run_all]   H5/H6 dynamics/persistence: {dyn.get('status')}")

    if lesion_rows:
        lesion_df = pd.DataFrame(lesion_rows)
        lesion_out = ROOT / "results" / "reflection_shuffle_lesion.csv"
        lesion_df.to_csv(lesion_out, index=False)
        print(f"\n[run_all] wrote {lesion_out}")

    if dynamics_rows:
        dynamics_df = pd.DataFrame(dynamics_rows)
        dynamics_out = ROOT / "results" / "dynamics_persistence.csv"
        dynamics_df.to_csv(dynamics_out, index=False)
        print(f"[run_all] wrote {dynamics_out}")

    if not maintenance_session_rows and not probe_rows:
        print("[run_all] no alignable runs (insufficient shared conditions or missing stimulus cache coverage).")
        return 1

    session_df = pd.DataFrame(maintenance_session_rows)
    session_out = ROOT / "results" / "alignment_by_session.csv"
    session_df.to_csv(session_out, index=False)
    print(f"\n[run_all] wrote {session_out} ({len(session_df)} rows)\n")

    probe_df = pd.DataFrame(probe_rows)

    # headline per-cell CSV: maintenance aggregated across sessions (mean,
    # pooled-region only) + the pooled-region probe row, joined by run_id.
    headline_rows = []
    for run_id, g in session_df[session_df.get("region") == "pooled"].groupby("run_id") if len(session_df) else []:
        ok = g[g.status == "ok"]
        if len(ok) == 0:
            continue
        headline_rows.append({
            "run_id": run_id, "model_id": ok.model_id.iloc[0], "S": ok.S.iloc[0], "M": ok.M.iloc[0], "L": ok.L.iloc[0], "seed": ok.seed.iloc[0],
            "maintenance_raw_alignment": ok.raw_alignment.mean(), "maintenance_noise_ceiling_upper": ok.noise_ceiling_upper.mean(),
            "maintenance_normalized_alignment": ok.normalized_alignment.mean(), "n_sessions_ok": len(ok),
        })
    headline_df = pd.DataFrame(headline_rows)
    if len(probe_df):
        probe_cols = ["run_id", "status", "raw_alignment", "noise_ceiling_upper", "normalized_alignment",
                      "n_shared_conditions", "accuracy_load1", "accuracy_load3"]
        probe_pooled = probe_df[probe_df.region == "pooled"].reindex(columns=probe_cols)
        probe_pooled = probe_pooled.rename(
            columns={c: f"probe_{c}" for c in ("raw_alignment", "noise_ceiling_upper", "normalized_alignment", "n_shared_conditions", "status")}
        )
        headline_df = headline_df.merge(probe_pooled, on="run_id", how="outer") if len(headline_df) else probe_pooled

    out_path = ROOT / "results" / "alignment_results.csv"
    headline_df.to_csv(out_path, index=False)
    print(f"[run_all] wrote {out_path}\n")
    if len(headline_df):
        print(headline_df.to_string(index=False))

    probe_out = ROOT / "results" / "alignment_probe_by_region.csv"
    probe_df.to_csv(probe_out, index=False)
    print(f"\n[run_all] wrote {probe_out}")

    from brainalign_wm.analysis.stats import mixed_effects_alignment

    # Main S*M*L model: POOLED-region rows only. MTL/MFC rows for the SAME
    # session are highly correlated subsets of the same units/trials (not
    # independent observations) -- mixing all three region levels into one
    # flat model (an earlier version of this code did) is pseudo-
    # replication that `mixed_effects_alignment` has no way to correct for,
    # inflating the effective N and anti-conservatively shrinking standard
    # errors on every fixed effect, found on adversarial review. Region
    # dissociation (H1/C2) gets its OWN separate model below instead.
    ok_sessions = session_df[(session_df.get("region") == "pooled") & (session_df.get("status") == "ok")] if len(session_df) else session_df
    accuracy_lookup = pd.DataFrame([{"run_id": r["run_id"], "accuracy": r.get("accuracy", {}).get("load3")} for r in completed])
    if len(ok_sessions) and ok_sessions["S"].nunique() >= 2 and ok_sessions["M"].nunique() >= 2 and ok_sessions["L"].nunique() >= 2 and len(ok_sessions) >= 6:
        sub = ok_sessions.rename(columns={"normalized_alignment": "align_score"}).merge(accuracy_lookup, on="run_id", how="left")
        res = mixed_effects_alignment(sub, formula="align_score ~ S * M * L + accuracy")
        print(f"\n[run_all] maintenance mixed-effects fit ({res['method']}):")
        for k, v in res["params"].items():
            print(f"    {k}: {v:.4f}")
    else:
        print("\n[run_all] insufficient factor coverage for mixed-effects inference on the maintenance path "
              "(need >=2 levels of S, M, L and >=6 session-rows); re-run once more cells complete.")

    # H1/C2 region-dissociation model: restricted to MTL/MFC rows (excludes
    # pooled, which would double-count each session against its own
    # subset). Every contributing session appears at most twice here (once
    # per region family), so `session` is a valid, non-pseudo-replicated
    # MixedLM group for testing the region factor and its interactions
    # with S (worker<->MTL, manager<->MFC).
    region_rows = session_df[(session_df.get("region").isin(["MTL", "MFC"])) & (session_df.get("status") == "ok")] if len(session_df) else session_df
    if len(region_rows) and region_rows["region"].nunique() >= 2 and region_rows["S"].nunique() >= 2 and len(region_rows) >= 6:
        sub_region = region_rows.rename(columns={"normalized_alignment": "align_score"}).merge(accuracy_lookup, on="run_id", how="left")
        res_region = mixed_effects_alignment(
            sub_region, formula="align_score ~ S * C(region) + accuracy", extra_vc_col="session",
        )
        print(f"\n[run_all] H1/C2 region-dissociation mixed-effects fit ({res_region['method']}):")
        for k, v in res_region["params"].items():
            print(f"    {k}: {v:.4f}")
    else:
        print("\n[run_all] insufficient MTL/MFC coverage for the H1/C2 region-dissociation model "
              "(need both region levels, >=2 levels of S, and >=6 rows).")

    if not args.skip_chance_control and len(headline_df):
        control_model_id = "M111" if "M111" in headline_df.model_id.values else headline_df.model_id.iloc[0]
        control_seed = int(headline_df[headline_df.model_id == control_model_id].seed.iloc[0])
        print(f"\n[run_all] H5 chance-model negative control ({control_model_id}, untrained) ...")
        chance = chance_control_check(control_model_id, control_seed, dandi_data, category_centroids)
        trained_row = headline_df[(headline_df.model_id == control_model_id) & (headline_df.seed == control_seed)].iloc[0]
        pd.DataFrame([{**chance, "role": "chance"}, {**trained_row.to_dict(), "role": "trained"}]).to_csv(
            ROOT / "results" / "chance_control.csv", index=False
        )
        for col in ("maintenance_normalized_alignment", "probe_normalized_alignment"):
            if col in chance and col in trained_row and pd.notna(trained_row[col]):
                verdict = "PASS" if chance[col] < trained_row[col] else "FAIL"
                print(f"    {col}: chance={chance[col]:.3f} trained={trained_row[col]:.3f}  [{verdict}]")
        print(f"[run_all] wrote {ROOT / 'results' / 'chance_control.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
