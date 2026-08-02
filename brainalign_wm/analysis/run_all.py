"""Runs the alignment and statistical-inference pipeline over whatever
training runs have completed so far (reads `results/manifest.jsonl`; does
not require the full grid to have finished).

Two epoch-appropriate condition schemas, each scored against a noise
ceiling built on that same representation:

  * **maintenance** (delay-period) alignment conditions on REAL held-item
    IDENTITY (preferred) or REAL dataset-embedded CATEGORY (fallback) +
    load -- never the probe/response-time in_set/correct labels, which
    can't causally structure activity that occurred before the probe.
    `_maintenance_condition_fn` picks per-session (same choice applied to
    both model replay and neural data): identity if this session's real
    trials have repeat structure, else category, else the session
    contributes nothing ("insufficient_shared_conditions" rather than
    worked around). Scored per session against that session's own
    within-session reliability ceiling (`rsa.within_session_noise_
    ceiling`) -- a cross-session LOSO ceiling is meaningless here since
    conditions are session-specific pools, not shared across sessions.
    Per-session rows carry the session's patient, feeding the LME's
    patient factor.
  * **probe-epoch** alignment uses the coarse (load, in_set, correct)
    schema, which generalizes across sessions, pooled via
    `analysis/pseudopopulation.py` (each unit's condition mean uses only
    its own session's trials) against a ceiling on that same pooled
    representation.

Both schemas are computed at the pooled-region level (region=None) and at
each region-family level (MTL, MFC), feeding a region factor into the
mixed-effects model. A degenerate-RDM guard requires >= `MIN_SHARED_
CONDITIONS` shared conditions before reporting a score; below that, the
row is marked "insufficient_shared_conditions" rather than emitting a
number.

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


def _identity_condition(row) -> tuple:
    """Maintenance-epoch condition, identity variant: the sorted set of
    real held-item PicIDs + load, never the probe/response-time in_set/
    correct labels (which can't causally structure activity that occurred
    before the probe). `(item_tuple, load)`-ordered for `_dpca_for_run`,
    which marginalizes item and load as separate tensor axes -- unlike the
    RSA crossnobis path (`_identity_condition_stratified`), so it doesn't
    need load-first stratification."""
    return (tuple(sorted(int(x) for x in row.held_items)), int(row.load))


def _category_condition(row) -> tuple:
    """Maintenance-epoch condition, real-category variant (dataset 000673,
    which never repeats item identity): the sorted multiset of held-item
    category codes + load, where category = first digit of the PicID
    (`pid // 100`) -- the same convention the dataset's own published
    analysis code (`NWB_calcSelective_SB.m`, github.com/rutishauserlab/
    SBCAT-release-NWB) uses, not a model-derived proxy. `(item_tuple,
    load)`-ordered for dPCA; see `_identity_condition`."""
    return (tuple(sorted(int(x) // 100 for x in row.held_items)), int(row.load))


def _identity_condition_stratified(row) -> tuple:
    """RSA/crossnobis counterpart of `_identity_condition`: `(load,
    item_tuple)`, load FIRST as an explicit stratum, for
    `rsa._session_condition_rdm(..., stratified=True)`/`rdm.
    stratified_crossnobis_rdm`, which builds a separate crossnobis RDM
    within each load stratum and assembles them block-diagonally (NaN
    cross-load entries, never computed) rather than pooling every load
    into one RDM whose off-diagonal is dominated by load."""
    return (int(row.load), tuple(sorted(int(x) for x in row.held_items)))


def _category_condition_stratified(row) -> tuple:
    """Stratified (load-first) counterpart to `_category_condition` -- see
    `_identity_condition_stratified`."""
    return (int(row.load), tuple(sorted(int(x) // 100 for x in row.held_items)))


def _session_has_identity_repeats(session_trials: pd.DataFrame, min_conditions: int = 2, min_count: int = 2) -> bool:
    """True if this session's real held-item-set+load combinations recur
    often enough (>=`min_conditions` distinct combinations with
    >=`min_count` trials each) to support `_identity_condition` directly;
    False means fall back to `_category_condition`. Decided from the
    session's own real trial table so the SAME choice applies identically
    to the model replay and the neural data for that session."""
    counts: dict = {}
    for row in session_trials.itertuples():
        key = (tuple(sorted(int(x) for x in row.held_items)), int(row.load))
        counts[key] = counts.get(key, 0) + 1
    return sum(1 for c in counts.values() if c >= min_count) >= min_conditions


def _maintenance_condition_fn(session_trials: pd.DataFrame) -> callable:
    """Picks `_identity_condition` or `_category_condition` for one session
    based on that session's own real trial data. `(item_tuple, load)`-
    ordered, for dPCA only -- see `_maintenance_condition_fn_stratified`
    for the RSA/crossnobis path."""
    return _identity_condition if _session_has_identity_repeats(session_trials) else _category_condition


def _maintenance_condition_fn_stratified(session_trials: pd.DataFrame) -> callable:
    """Picks the `(load, item_tuple)`-ordered, load-stratified counterpart
    of `_maintenance_condition_fn`, for every RSA/crossnobis maintenance-
    path use (`_maintenance_alignment_for_run`, `_baselines_for_run`'s
    B1/B2, and `_encoding_for_run`'s ceiling). Same session-level identity-
    vs-category decision as `_maintenance_condition_fn`."""
    return _identity_condition_stratified if _session_has_identity_repeats(session_trials) else _category_condition_stratified


def _is_ablation_or_catch_variant(rec: dict) -> bool:
    """True for a bio-plausible-ablation (§4.4: M11111_energy/M11111_noise/
    M11111_pbwm) or identity-catch (§9.4a: M00000_idcatch/M11111_idcatch)
    run -- same S/M/P/T/D bits as a Core ablation-battery cell but a
    materially different trained representation, so it must not be pooled
    into that cell's S x M x P x T x D regression rows (reported via its
    own comparison table instead). Reconstructs the 5-bit model_id (not
    the pre-v6.0 3-bit S/M/P one) so every real 15-cell battery cell is
    correctly recognized as non-variant."""
    return rec["model_id"] != f"M{rec['S']}{rec['M']}{rec['P']}{rec['T']}{rec['D']}"


def _coarse_condition(row) -> tuple:
    """Probe-epoch condition: a property of the response/decision itself,
    legitimately poolable across sessions. The model's activity log
    (`in_set`) and the NWB trial table (`probe_in_set`) name the same
    field differently -- accept either."""
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
    (reset to 0 for every session), so it is not globally unique across
    the full log -- grouping by `trial_id` alone would silently merge
    activity from different trials in different sessions that happen to
    share a trial index. The maintenance-epoch path never hits this
    because it pre-filters `model_df` to one session before calling this
    function; the probe-epoch pooled path passes the full multi-session
    log directly."""
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


# Maintenance-epoch achievable condition space varies by which schema a
# session uses (`_maintenance_condition_fn`): 000469-identity sessions can
# reach up to 25 real item-sets x load; 000673-category sessions are capped
# at 5 categories x load = up to 15. Both are smaller than the probe
# epoch's blanket floor would allow through, so a schema-appropriate floor
# is used here instead (still a real guard against a degenerate 1-2
# condition RDM, A2f).
MIN_SHARED_CONDITIONS_MAINTENANCE = 4


MIN_VALID_PAIRS_MAINTENANCE = 3


def _n_valid_pairs(rdm_a: np.ndarray, rdm_b: np.ndarray) -> int:
    """Number of condition pairs actually usable for `compare_rdms` between
    two RDMs -- i.e. neither side NaN at that pair. Under load
    stratification, `len(shared)` (the shared condition count) doesn't
    guarantee a proportional number of informative pairs, since shared
    conditions can be spread thin across load strata (e.g. 4 shared
    conditions split 2+1+1 across three loads gives only one real
    same-stratum pair, not the ~6 a flat 4-condition RDM would). Guards
    `_maintenance_alignment_for_run`/`_baselines_for_run` against reporting
    a number computed from too few genuine (non-cross-stratum) pairs."""
    from brainalign_wm.analysis.rdm import vectorize_upper

    va, vb = vectorize_upper(rdm_a), vectorize_upper(rdm_b)
    return int(np.sum(~np.isnan(va) & ~np.isnan(vb)))


def _maintenance_alignment_for_run(
    run_id: str, model_df: pd.DataFrame, dandi_data, region, n_permutations: int = 0, permutation_seed: int = 0,
) -> list[dict]:
    """Per-session maintenance-epoch alignment, using real identity or real
    category per session, stratified by load (see
    `_maintenance_condition_fn_stratified`/`rdm.stratified_crossnobis_rdm`
    -- pooling all loads into one crossnobis run lets the off-diagonal be
    dominated by load rather than identity). Returns one row per session
    with usable data (both model replay coverage and neural trials for
    that session).

    `n_permutations > 0` additionally computes, per session, a label-
    permutation null: the neural side's condition labels are shuffled
    within each load stratum (`stratified_crossnobis_rdm(...,
    permute_seed=...)`), the model RDM is left at its true, observed value,
    and `n_permutations` independent draws of the resulting raw alignment
    are stashed under `"_perm_raw_alignments"` -- consumed by `main()` to
    build a null distribution of the run's aggregate maintenance DV via the
    identical (signed-mean) aggregation pipeline the observed statistic
    uses, so any residual rectification bias cancels between the two
    rather than inflating one but not the other. The (expensive) neural
    `ds.rates()` fetch happens once per session regardless of
    `n_permutations` (`rsa._session_trial_patterns`), not once per draw."""
    from brainalign_wm.analysis.rsa import compare_rdms, within_session_noise_ceiling
    from brainalign_wm.analysis import rsa as rsa_mod
    from brainalign_wm.analysis.rdm import stratified_crossnobis_rdm

    rows = []
    model_sessions = set(model_df["session"].unique()) if "session" in model_df.columns else set()
    all_trials = dandi_data.trials()
    for session_id in sorted(model_sessions):
        session_trials = all_trials[all_trials.session == session_id]
        if len(session_trials) == 0:
            continue
        condition_fn = _maintenance_condition_fn_stratified(session_trials)

        sess_df = model_df[model_df["session"] == session_id]
        model_patterns, model_labels = _model_epoch_patterns(sess_df, "maintain", condition_fn)
        model_patterns, model_labels = _filter_min_trials(model_patterns, model_labels, min_count=2)
        if len(set(model_labels)) < 2:
            continue
        model_strata = [l[0] for l in model_labels]
        model_conds_only = [l[1] for l in model_labels]
        model_rdm, model_conds = stratified_crossnobis_rdm(model_patterns, model_conds_only, model_strata, n_folds=4, seed=0)
        if model_rdm is None:
            continue

        neural_data, neural_session_trials = rsa_mod._session_trial_patterns(
            dandi_data, session_id, region, "maintain", dandi_data.bin_ms
        )
        if neural_data is None:
            continue
        neural_labels = [condition_fn(row) for row in neural_session_trials.itertuples()]
        neural_strata = [l[0] for l in neural_labels]
        neural_conds_only = [l[1] for l in neural_labels]
        neural_rdm, neural_conds = stratified_crossnobis_rdm(neural_data, neural_conds_only, neural_strata, n_folds=2, seed=0)
        if neural_rdm is None:
            continue

        shared = _shared_conditions_or_none(model_conds, neural_conds, floor=MIN_SHARED_CONDITIONS_MAINTENANCE)
        if shared is None:
            rows.append({"run_id": run_id, "session": session_id, "region": region or "pooled",
                          "status": "insufficient_shared_conditions", "n_shared_conditions": len(set(model_conds) & set(neural_conds))})
            continue
        m_idx = [model_conds.index(c) for c in shared]
        n_idx = [neural_conds.index(c) for c in shared]
        model_sub = model_rdm[np.ix_(m_idx, m_idx)]
        neural_sub = neural_rdm[np.ix_(n_idx, n_idx)]
        if _n_valid_pairs(model_sub, neural_sub) < MIN_VALID_PAIRS_MAINTENANCE:
            rows.append({"run_id": run_id, "session": session_id, "region": region or "pooled",
                          "status": "insufficient_shared_conditions", "n_shared_conditions": len(shared)})
            continue
        raw = compare_rdms(model_sub, neural_sub)
        ceiling_lower, ceiling_upper = within_session_noise_ceiling(
            dandi_data, session_id, region, "maintain", dandi_data.bin_ms, condition_fn=condition_fn, stratified=True,
        )
        row = {
            "run_id": run_id, "session": session_id, "region": region or "pooled",
            "patient": dandi_data.patient_of(session_id), "status": "ok",
            "raw_alignment": raw, "noise_ceiling_upper": ceiling_upper, "noise_ceiling_lower": ceiling_lower,
            "normalized_alignment": _norm(raw, ceiling_upper), "n_shared_conditions": len(shared),
        }
        if n_permutations > 0:
            perm_raws = []
            for p in range(n_permutations):
                neural_rdm_p, neural_conds_p = stratified_crossnobis_rdm(
                    neural_data, neural_conds_only, neural_strata, n_folds=2, seed=0, permute_seed=permutation_seed + p,
                )
                if neural_rdm_p is None or neural_conds_p != neural_conds:
                    continue
                perm_raws.append(compare_rdms(model_sub, neural_rdm_p[np.ix_(n_idx, n_idx)]))
            row["_perm_raw_alignments"] = perm_raws
        rows.append(row)
    return rows


def _norm(raw: float, ceiling_upper: float) -> float:
    from brainalign_wm.analysis.rsa import normalized_alignment

    return normalized_alignment(raw, ceiling_upper)


def _aggregate_maintenance(rows_ok: list[dict]) -> dict:
    """Audit fix N2: aggregate SIGNED per-session `raw_alignment`/
    `noise_ceiling_upper` via a plain mean FIRST, then normalize ONCE --
    NOT the reverse (each session's raw/ceiling ratio clipped to [0,1]
    THEN averaged). Per-session raw alignment is noisy around a near-zero
    mean (verified: std ~0.09 per session, with 40-56% of sessions
    NEGATIVE for well-trained cells); clipping each session at 0 before
    averaging rectifies that symmetric noise into a strictly positive
    bias -- concretely, a session with signed mean raw alignment of
    -0.0004 (indistinguishable from zero) still reported a clipped
    `normalized_alignment` of 0.041, and this bias is exactly what put an
    untrained chance model at the trained-cell distribution's MEDIAN
    rather than at a floor. The signed aggregate is reported explicitly
    (not just the normalized one), per the fix's own instruction not to
    let this recur silently.

    Used identically by `main()`'s headline computation, `chance_control_
    check`, and `reflection_shuffle_lesion_for_run` -- the three places
    that used to each independently average `normalized_alignment`."""
    raws = np.array([r["raw_alignment"] for r in rows_ok], dtype=float)
    ceilings = np.array([r["noise_ceiling_upper"] for r in rows_ok], dtype=float)
    mean_raw = float(raws.mean())
    mean_ceiling = float(ceilings.mean())
    return {
        "maintenance_signed_raw_alignment": mean_raw,
        "maintenance_raw_alignment": mean_raw,
        "maintenance_noise_ceiling_upper": mean_ceiling,
        "maintenance_normalized_alignment": _norm(mean_raw, mean_ceiling),
        "n_sessions_ok": len(rows_ok),
    }


def _probe_alignment_for_run(run_id: str, model_df: pd.DataFrame, dandi_data, region) -> dict:
    """Pooled coarse-condition probe-epoch alignment."""
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


def _task_model_rdm_per_trial(labels: list) -> np.ndarray:
    """B2 (master protocol §4.2, non-negotiable): the per-trial RDM implied
    by ground-truth task structure alone -- 0 if two trials share the same
    held-item set (within a load stratum), 1 otherwise. Trial-level (not
    condition-level) because under load stratification, every condition
    within a stratum is unique, so a condition-level version would be
    constant (zero variance); per-trial labels have real repeats. Returns
    an [n_trials, n_trials] RDM, compared against per-trial Euclidean
    neural distances by `_baselines_for_run`."""
    n = len(labels)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # labels are (stratum, item_tuple) tuples
            same = (labels[i] == labels[j])
            rdm[i, j] = rdm[j, i] = 0.0 if same else 1.0
    return rdm


def _encoder_only_patterns_for_session(session_id: str, session_trials: pd.DataFrame, condition_fn) -> tuple[np.ndarray, list]:
    """B1 (master protocol §4.2, non-negotiable): patterns built from the
    frozen ResNet encoder's own features alone (mean over held items),
    NO working-memory processing at all -- the "vision without WM" lower
    anchor. Uses the same cached per-session stimulus features the model
    replay itself is driven by."""
    from brainalign_wm.training.generate_activity_logs import _stimulus_features_for_session

    pic_to_feature = _stimulus_features_for_session(session_id)
    if pic_to_feature is None:
        return np.zeros((0, 0)), []
    patterns, labels = [], []
    for row in session_trials.itertuples():
        held = [int(x) for x in row.held_items if int(x) != 0]
        feats = [pic_to_feature.get(str(pid)) for pid in held]
        if not held or any(f is None for f in feats):
            continue
        patterns.append(np.mean(feats, axis=0))
        labels.append(condition_fn(row))
    return np.array(patterns), labels


def _baselines_for_run(run_id: str, model_df: pd.DataFrame, dandi_data, region) -> list[dict]:
    """B1 (encoder-only) and B2 (task-model) baselines, master protocol
    S4.2: computed per session, on the SAME shared-condition set and
    neural RDM the real maintenance alignment uses for that session/region
    -- load-stratified, so B1/B2 rows stay directly
    comparable to the trained-model rows in `alignment_by_session.csv`
    (a non-stratified baseline compared against a stratified trained-model
    row would not be an apples-to-apples baseline).

    B2 is a per-trial task-model RSA: the condition-level task-model RDM
    is degenerate under load-stratification (each
    within-stratum condition is unique, producing a constant RDM with zero
    variance). Instead, B2 builds a per-trial "same held-set = 0,
    different = 1" task-model RDM and correlates it against per-trial
    Euclidean neural distances, which have real trial-level variation."""
    from brainalign_wm.analysis.rsa import compare_rdms, within_session_noise_ceiling
    from brainalign_wm.analysis import rsa as rsa_mod
    from brainalign_wm.analysis.rdm import stratified_crossnobis_rdm

    rows = []
    model_sessions = set(model_df["session"].unique()) if "session" in model_df.columns else set()
    all_trials = dandi_data.trials()
    for session_id in sorted(model_sessions):
        session_trials = all_trials[all_trials.session == session_id]
        if len(session_trials) == 0:
            continue
        condition_fn = _maintenance_condition_fn_stratified(session_trials)

        neural_rdm, neural_conds = rsa_mod._session_condition_rdm(
            dandi_data, session_id, region, "maintain", dandi_data.bin_ms, condition_fn=condition_fn, stratified=True,
        )
        if neural_rdm is None:
            continue
        # Also extract per-trial neural data for B2's per-trial task-model RDM
        neural_data, neural_session_trials = rsa_mod._session_trial_patterns(
            dandi_data, session_id, region, "maintain", dandi_data.bin_ms
        )
        ceiling_lower, ceiling_upper = within_session_noise_ceiling(
            dandi_data, session_id, region, "maintain", dandi_data.bin_ms, condition_fn=condition_fn, stratified=True,
        )

        # B1: encoder-only
        b1_patterns, b1_labels = _encoder_only_patterns_for_session(session_id, session_trials, condition_fn)
        b1_patterns, b1_labels = _filter_min_trials(b1_patterns, b1_labels, min_count=2)
        if len(set(b1_labels)) >= 2:
            b1_strata = [l[0] for l in b1_labels]
            b1_conds_only = [l[1] for l in b1_labels]
            b1_rdm, b1_conds = stratified_crossnobis_rdm(b1_patterns, b1_conds_only, b1_strata, n_folds=4, seed=0)
            if b1_rdm is not None:
                shared_b1 = _shared_conditions_or_none(b1_conds, neural_conds, floor=MIN_SHARED_CONDITIONS_MAINTENANCE)
                if shared_b1 is not None:
                    m_idx = [b1_conds.index(c) for c in shared_b1]
                    n_idx = [neural_conds.index(c) for c in shared_b1]
                    b1_sub, neural_sub_b1 = b1_rdm[np.ix_(m_idx, m_idx)], neural_rdm[np.ix_(n_idx, n_idx)]
                    if _n_valid_pairs(b1_sub, neural_sub_b1) >= MIN_VALID_PAIRS_MAINTENANCE:
                        raw_b1 = compare_rdms(b1_sub, neural_sub_b1)
                        rows.append({
                            "run_id": run_id, "session": session_id, "region": region or "pooled", "baseline": "B1_encoder_only",
                            "patient": dandi_data.patient_of(session_id), "status": "ok", "raw_alignment": raw_b1,
                            "noise_ceiling_upper": ceiling_upper, "normalized_alignment": _norm(raw_b1, ceiling_upper),
                            "n_shared_conditions": len(shared_b1),
                        })

        # B2: task-model (ground-truth condition structure only). A
        # condition-level RDM is degenerate under load-stratification (each
        # within-stratum condition is unique, so the RDM is constant = zero
        # variance); use the per-trial task-model RDM instead, correlated
        # against per-trial Euclidean neural distances (a noise-whitened-
        # free approximation to crossnobis at the trial level, which is the
        # level at which the task-model ground truth has real variation).
        neural_labels = [condition_fn(row) for row in neural_session_trials.itertuples()]
        if len(neural_labels) >= 4:
            from brainalign_wm.analysis.rdm import vectorize_upper as _vu
            b2_task_rdm = _task_model_rdm_per_trial(neural_labels)
            # Per-trial Euclidean distances on the raw neural data
            from scipy.spatial.distance import squareform, pdist
            neural_trial_rdm = squareform(pdist(neural_data, metric="euclidean"))
            raw_b2 = compare_rdms(b2_task_rdm, neural_trial_rdm)
            n_valid_b2 = int(np.sum(~np.isnan(_vu(b2_task_rdm)) & ~np.isnan(_vu(neural_trial_rdm))))
            # B2's raw score lives on the per-trial Euclidean
            # representation, not the condition-level crossnobis one
            # `ceiling_upper` (above) estimates -- normalize against a
            # ceiling built on that same per-trial representation instead.
            b2_ceiling_lower, b2_ceiling_upper = rsa_mod.trial_level_split_half_ceiling(neural_data, neural_labels)
            rows.append({
                "run_id": run_id, "session": session_id, "region": region or "pooled", "baseline": "B2_task_model",
                "patient": dandi_data.patient_of(session_id), "status": "ok", "raw_alignment": raw_b2,
                "b2_noise_ceiling_upper": b2_ceiling_upper, "b2_noise_ceiling_lower": b2_ceiling_lower,
                "normalized_alignment": _norm(raw_b2, b2_ceiling_upper),
                "n_shared_conditions": n_valid_b2,
            })
    return rows


def _paired_trial_patterns(model_df: pd.DataFrame, dandi_data, session_id: str, region, epoch: str = "maintain") -> tuple[np.ndarray, np.ndarray]:
    """Master protocol §9.5 (ridge encoding models): model and neural
    per-TRIAL patterns for one session, paired by REAL trial identity --
    NOT by condition (so this works even on sessions with no repeat
    structure, unlike the RSA path). Valid because the model replay
    (`generate_activity_logs.replay_session`) assigns `trial_id` as the
    tick-independent position within that session's own real trial table
    (`enumerate(session_trials.itertuples())`), so `trial_id=k` in the
    model log and the k-th row of `dandi_data.trials()` filtered to this
    session refer to the SAME real trial."""
    sess_df = model_df[model_df["session"] == session_id]
    sub = sess_df[sess_df.epoch == epoch]
    if len(sub) == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))
    is_flat = sub["h_flat"].iloc[0] is not None
    model_by_trial: dict = {}
    for trial_id, g in sub.groupby("trial_id"):
        if is_flat:
            vec = np.stack(g["h_flat"].to_numpy()).mean(axis=0)
        else:
            vec = np.concatenate([np.stack(g["h_worker"].to_numpy()).mean(axis=0), np.stack(g["h_manager"].to_numpy()).mean(axis=0)])
        model_by_trial[int(trial_id)] = vec

    all_trials = dandi_data.trials()
    # Boolean-masking `all_trials` preserves row order, and each session's
    # rows are contiguous in `trials()` (built by `pd.concat` per session,
    # see `DandiSternbergTierA.trials`) -- so `global_idx`'s order already
    # matches `session_trials.reset_index(drop=True)`'s order, i.e. `pos`
    # here is exactly the tick-independent trial_id `replay_session`
    # assigns via `enumerate(session_trials.itertuples())`.
    global_idx = np.where(all_trials.session.values == session_id)[0]
    if len(global_idx) == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))
    rates = dandi_data.rates(region, dandi_data.bin_ms, [epoch])  # [n_units, n_all_trials, n_bins]
    neural_by_pos = rates[:, global_idx, :].mean(axis=2).T  # [n_session_trials, n_units]

    model_rows, neural_rows = [], []
    for pos in range(len(global_idx)):
        if pos in model_by_trial:
            model_rows.append(model_by_trial[pos])
            neural_rows.append(neural_by_pos[pos])
    if not model_rows:
        return np.zeros((0, 0)), np.zeros((0, 0))
    return np.array(model_rows), np.array(neural_rows)


def _encoding_for_run(run_id: str, model_df: pd.DataFrame, dandi_data, region) -> list[dict]:
    """Master protocol §9.5: nested-CV ridge encoding, model<->neuron, both
    directions, noise-ceiling-normalized -- complements RSA's geometric
    view with a predictive-mapping one. Uses PER-TRIAL patterns (see
    `_paired_trial_patterns`), so unlike the RSA maintenance path this
    works on every session with model replay coverage, regardless of
    whether that session supports identity- or category-based conditions.
    The ceiling used to normalize is this session's own RSA within-session
    ceiling (`_maintenance_condition_fn_stratified`-selected, load-
    stratified schema, kept consistent with every other maintenance-path
    ceiling), reused here as a practical proxy for the
    per-target-neuron reliability ceiling a dedicated single-unit
    split-half estimate would give (not separately computed, for
    tractability)."""
    from brainalign_wm.analysis.encoding import encoding_r2, noise_ceiling_normalized_r2
    from brainalign_wm.analysis.rsa import within_session_noise_ceiling

    rows = []
    model_sessions = set(model_df["session"].unique()) if "session" in model_df.columns else set()
    all_trials = dandi_data.trials()
    for session_id in sorted(model_sessions):
        X_model, Y_neural = _paired_trial_patterns(model_df, dandi_data, session_id, region)
        if X_model.shape[0] < 10:  # need enough trials for a meaningful outer CV split
            continue
        session_trials = all_trials[all_trials.session == session_id]
        condition_fn = _maintenance_condition_fn_stratified(session_trials)
        _, ceiling_upper = within_session_noise_ceiling(
            dandi_data, session_id, region, "maintain", dandi_data.bin_ms, condition_fn=condition_fn, stratified=True,
        )
        r2_model_to_neuron = encoding_r2(X_model, Y_neural, n_folds=5, seed=0)
        r2_neuron_to_model = encoding_r2(Y_neural, X_model, n_folds=5, seed=0)
        rows.append({
            "run_id": run_id, "session": session_id, "region": region or "pooled",
            "patient": dandi_data.patient_of(session_id), "status": "ok", "n_trials": X_model.shape[0],
            "model_to_neuron_r2_mean": float(np.mean(r2_model_to_neuron)),
            "model_to_neuron_r2_normalized_mean": float(np.mean(noise_ceiling_normalized_r2(r2_model_to_neuron, ceiling_upper))),
            "neuron_to_model_r2_mean": float(np.mean(r2_neuron_to_model)),
        })
    return rows


def _dpca_condition_tensor(patterns: list, labels: list) -> tuple:
    """patterns: list of per-trial [n_ticks, n_units] time series (all the
    SAME n_ticks -- both the model replay and the neural rate tensor use a
    fixed per-epoch tick/bin count, so no modal-length filtering is
    needed here, unlike `dynamics_and_persistence.model_epoch_timeseries`).
    labels: list of `(item_or_category_tuple, load)` per trial, matching
    `patterns` (from `_maintenance_condition_fn`). Returns
    (R [n_units, n_item_levels, n_load_levels, n_ticks], item_levels,
    load_levels) -- condition-mean tensor for `dpca_components`."""
    if not patterns:
        return None, [], []
    n_ticks, n_units = patterns[0].shape
    item_levels = sorted(set(l[0] for l in labels), key=str)
    load_levels = sorted(set(l[1] for l in labels))
    R = np.zeros((n_units, len(item_levels), len(load_levels), n_ticks))
    counts = np.zeros((len(item_levels), len(load_levels)))
    for mat, (item, load) in zip(patterns, labels):
        ii, li = item_levels.index(item), load_levels.index(load)
        R[:, ii, li, :] += mat.T
        counts[ii, li] += 1
    for ii in range(len(item_levels)):
        for li in range(len(load_levels)):
            if counts[ii, li] > 0:
                R[:, ii, li, :] /= counts[ii, li]
    return R, item_levels, load_levels


def _model_dpca_patterns(model_df: pd.DataFrame, session_id: str, condition_fn, epoch: str = "maintain") -> tuple[list, list]:
    """Per-trial [n_ticks, n_units] time series + `(item_or_cat, load)`
    labels for one session's model replay, for `_dpca_condition_tensor`."""
    sess_df = model_df[model_df["session"] == session_id]
    sub = sess_df[sess_df.epoch == epoch]
    if len(sub) == 0:
        return [], []
    is_flat = sub["h_flat"].iloc[0] is not None
    patterns, labels = [], []
    for _trial_id, g in sub.groupby("trial_id"):
        g = g.sort_values("t")
        if is_flat:
            mat = np.stack(g["h_flat"].to_numpy())
        else:
            mat = np.concatenate([np.stack(g["h_worker"].to_numpy()), np.stack(g["h_manager"].to_numpy())], axis=1)
        patterns.append(mat)
        labels.append(condition_fn(g.iloc[0]))
    return patterns, labels


def _neural_dpca_patterns(dandi_data, session_id: str, region, condition_fn, epoch: str = "maintain") -> tuple[list, list]:
    """Per-trial [n_ticks, n_units] time series + `(item_or_cat, load)`
    labels for one session's real neural data, for `_dpca_condition_tensor`."""
    trials = dandi_data.trials()
    session_trials = trials[trials.session == session_id]
    if len(session_trials) == 0:
        return [], []
    units = dandi_data.units(region)
    session_units = [u for u in units if u.split("#")[0] == str(session_id)]
    if not session_units:
        return [], []
    unit_idx = [i for i, u in enumerate(units) if u in set(session_units)]
    global_idx = np.where(trials.session.values == session_id)[0]
    rates = dandi_data.rates(region, dandi_data.bin_ms, [epoch])  # [n_units(all), n_trials(all), n_bins]
    sub = rates[np.ix_(unit_idx, global_idx)]  # [n_units_sess, n_trials_sess, n_bins]
    patterns, labels = [], []
    for pos, row in enumerate(session_trials.itertuples()):
        patterns.append(sub[:, pos, :].T)  # [n_bins, n_units_sess]
        labels.append(condition_fn(row))
    return patterns, labels


def _dpca_for_run(run_id: str, model_df: pd.DataFrame, dandi_data, region) -> list[dict]:
    """Master protocol §9.5/Core: dPCA marginalization (item/category, load,
    time) on real per-session data, model vs. brain -- previously only run
    against the simulated-spiking recovery gate. Reports the fraction of
    total variance carried by each marginalization, per session, for both
    sides."""
    from brainalign_wm.analysis.dpca import dpca_components

    rows = []
    model_sessions = set(model_df["session"].unique()) if "session" in model_df.columns else set()
    all_trials = dandi_data.trials()
    for session_id in sorted(model_sessions):
        session_trials = all_trials[all_trials.session == session_id]
        if len(session_trials) == 0:
            continue
        condition_fn = _maintenance_condition_fn(session_trials)

        model_patterns, model_labels = _model_dpca_patterns(model_df, session_id, condition_fn)
        if len(model_patterns) < 4 or len(set(model_labels)) < 2:
            continue
        R_model, item_levels_m, load_levels_m = _dpca_condition_tensor(model_patterns, model_labels)
        if R_model is None or R_model.shape[1] < 2 or R_model.shape[2] < 2:
            continue
        comp_model = dpca_components(R_model, ["item", "load", "time"], n_components=1)

        neural_patterns, neural_labels = _neural_dpca_patterns(dandi_data, session_id, region, condition_fn)
        if len(neural_patterns) < 4 or len(set(neural_labels)) < 2:
            continue
        R_neural, item_levels_n, load_levels_n = _dpca_condition_tensor(neural_patterns, neural_labels)
        if R_neural is None or R_neural.shape[1] < 2 or R_neural.shape[2] < 2:
            continue
        comp_neural = dpca_components(R_neural, ["item", "load", "time"], n_components=1)

        rows.append({
            "run_id": run_id, "session": session_id, "region": region or "pooled",
            "patient": dandi_data.patient_of(session_id), "status": "ok",
            "model_item_frac_var": comp_model[("item",)]["fraction_of_total_variance"],
            "model_load_frac_var": comp_model[("load",)]["fraction_of_total_variance"],
            "neural_item_frac_var": comp_neural[("item",)]["fraction_of_total_variance"],
            "neural_load_frac_var": comp_neural[("load",)]["fraction_of_total_variance"],
        })
    return rows


def align_one_run(
    run_id: str, dandi_data, force_regenerate: bool = False, n_permutations: int = 0, permutation_seed: int = 0,
    checkpoint: str = "ckpt.pt",
) -> dict:
    """Generates/loads the run's activity log and computes both the
    per-session maintenance alignment and the pooled probe alignment, at
    each region level. Returns {"maintenance": [rows...], "probe": [rows...]}.

    `n_permutations` is forwarded only to the pooled-region
    (region=None) maintenance call, matching the pooled-only scope of
    `main()`'s headline DV and chance-control gate -- computing it for
    MTL/MFC too would triple an already-expensive per-session permutation
    loop for numbers nothing downstream currently consumes."""
    from brainalign_wm.training.generate_activity_logs import activity_log_path, generate_activity_log
    from brainalign_wm.training.logging_schema import read_log

    log_path = activity_log_path(run_id, checkpoint)
    if force_regenerate or not log_path.exists():
        log_path = generate_activity_log(run_id, dandi_data, checkpoint_name=checkpoint)
    df = read_log(log_path)

    maintenance_rows, probe_rows = [], []
    for region in REGIONS:
        region_n_permutations = n_permutations if region is None else 0
        maintenance_rows.extend(
            _maintenance_alignment_for_run(
                run_id, df, dandi_data, region, n_permutations=region_n_permutations, permutation_seed=permutation_seed,
            )
        )
        probe_rows.append(_probe_alignment_for_run(run_id, df, dandi_data, region))
    return {"maintenance": maintenance_rows, "probe": probe_rows}


def reflection_shuffle_lesion_for_run(run_id: str, dandi_data, checkpoint: str = "ckpt.pt") -> dict:
    """C1: H2's causal claim ("reflection-shuffle abolishes the M effect").
    Compares pooled-region maintenance alignment between the normal
    activity log and a version replayed with each trial's own R_t sequence
    time-shuffled within the trial (`generate_activity_logs.
    generate_activity_log_reflection_shuffled`). Only defined for M=1 runs.
    Returns a summary dict; if H2 holds, `shuffled_normalized_alignment`
    should be markedly lower than `normal_normalized_alignment`."""
    from brainalign_wm.training.generate_activity_logs import (
        activity_log_path, generate_activity_log_reflection_shuffled,
    )
    from brainalign_wm.training.logging_schema import read_log

    normal_log = activity_log_path(run_id, checkpoint)
    if not normal_log.exists():
        return {"run_id": run_id, "status": "normal_log_missing"}
    normal_df = read_log(normal_log)
    normal_rows = _maintenance_alignment_for_run(run_id, normal_df, dandi_data, None)
    normal_ok = [r for r in normal_rows if r.get("status") == "ok"]

    shuffled_path = generate_activity_log_reflection_shuffled(run_id, dandi_data, checkpoint_name=checkpoint)
    shuffled_df = read_log(shuffled_path)
    shuffled_rows = _maintenance_alignment_for_run(run_id, shuffled_df, dandi_data, None)
    shuffled_ok = [r for r in shuffled_rows if r.get("status") == "ok"]

    if not normal_ok or not shuffled_ok:
        return {"run_id": run_id, "status": "insufficient_sessions",
                "n_normal_sessions_ok": len(normal_ok), "n_shuffled_sessions_ok": len(shuffled_ok)}
    # Audit fix N2: aggregate SIGNED raw alignment/ceiling first, normalize
    # once -- not mean-of-per-session-clipped (see `_aggregate_maintenance`).
    normal_agg = _aggregate_maintenance(normal_ok)
    shuffled_agg = _aggregate_maintenance(shuffled_ok)
    normal_norm = normal_agg["maintenance_normalized_alignment"]
    shuffled_norm = shuffled_agg["maintenance_normalized_alignment"]
    return {
        "run_id": run_id, "status": "ok",
        "normal_normalized_alignment": normal_norm, "shuffled_normalized_alignment": shuffled_norm,
        "normal_signed_raw_alignment": normal_agg["maintenance_signed_raw_alignment"],
        "shuffled_signed_raw_alignment": shuffled_agg["maintenance_signed_raw_alignment"],
        "lesion_effect": normal_norm - shuffled_norm,
        "n_normal_sessions_ok": len(normal_ok), "n_shuffled_sessions_ok": len(shuffled_ok),
    }


MAX_SESSIONS_FOR_DYNAMICS = 20


def dynamics_and_persistence_for_run(run_id: str, model_df: pd.DataFrame, dandi_data, max_sessions: int = MAX_SESSIONS_FOR_DYNAMICS) -> dict:
    """H5/H6: aggregates per-session stability-index
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


def chance_control_check(model_id: str, seed: int, dandi_data) -> dict:
    """H5 acceptance gate: an untrained (chance) model must
    land near the alignment floor -- wired into the standard `run_all.py`
    output (not a one-off spot check), so this is re-verified every time
    the pipeline runs, not just once during validation."""
    from brainalign_wm.training.generate_activity_logs import generate_chance_activity_log
    from brainalign_wm.training.logging_schema import read_log

    log_path = generate_chance_activity_log(model_id, seed, dandi_data)
    df = read_log(log_path)
    run_id = f"{model_id}_s{seed}_chance"

    maintenance_rows = _maintenance_alignment_for_run(run_id, df, dandi_data, None)
    probe_row = _probe_alignment_for_run(run_id, df, dandi_data, None)
    maintenance_ok = [r for r in maintenance_rows if r.get("status") == "ok"]

    out = {"run_id": run_id, "model_id": model_id}
    if maintenance_ok:
        # Audit fix N2: signed-aggregate-then-normalize, not
        # mean-of-per-session-clipped (see `_aggregate_maintenance`).
        out.update(_aggregate_maintenance(maintenance_ok))
    if probe_row.get("status") == "ok":
        out["probe_normalized_alignment"] = probe_row["normalized_alignment"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--regenerate", action="store_true", help="force regeneration of activity logs")
    ap.add_argument(
        "--runs", default=None,
        help="comma-separated run_ids to analyse instead of every completed run in the manifest. "
             "Naming a run explicitly also bypasses the local-learning and ablation/catch-variant "
             "auto-skips below -- those exist to keep variants out of the pooled S x M x P "
             "regression, and an explicitly named run is not a pooling decision. Used by §18.4's "
             "pre-launch pipeline check, whose subjects (FLATGRU_*_s0) are variant-named pilots.",
    )
    ap.add_argument(
        "--checkpoint", default="ckpt.pt",
        help="advisor.md D33 / §12.4: which snapshot to analyse. 'ckpt.pt' is Gate B (max_steps, the "
             "headline equal-duration reading); 'ckpt_at_criterion.pt' is the Gate A snapshot, giving "
             "the equal-PERFORMANCE comparison. Non-default checkpoints read and write namespaced "
             "paths (activity_logs/{run_id}_at_criterion.parquet, results/*_at_criterion.csv) -- the "
             "two passes must never write one file. A run that never cleared Gate A has no Gate A "
             "checkpoint and is skipped with its FileNotFoundError reported, not substituted.",
    )
    ap.add_argument(
        "--max-sessions-per-dataset", type=int, default=None,
        help="bound the number of sessions loaded per dataset; `dandi_nwb.rates()` recomputes "
             "spike histograms with no caching (unlike sim_brain's), so the full Tier A pool "
             "(~1800 units, thousands of trials) can be slow for repeated interactive runs.",
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
    ap.add_argument(
        "--skip-baselines", action="store_true",
        help="skip the B1 (encoder-only) / B2 (task-model) non-negotiable baselines (master protocol §4.2).",
    )
    ap.add_argument(
        "--skip-encoding", action="store_true",
        help="skip the master-protocol §9.5 ridge encoding models (model<->neuron, per-session, pooled region only).",
    )
    ap.add_argument(
        "--skip-dpca", action="store_true",
        help="skip the master-protocol §9.5/Core dPCA marginalization on real per-session data (pooled region only).",
    )
    ap.add_argument(
        "--n-permutations", type=int, default=20,
        help="number of within-load-stratum, within-session label-permutation draws per run, "
             "used to build a null distribution of the run's aggregate maintenance DV (pooled region only). "
             "Pass 0 (or --skip-permutation-null) to disable -- each draw re-runs a per-session crossnobis fit.",
    )
    ap.add_argument(
        "--skip-permutation-null", action="store_true",
        help="alias for --n-permutations 0.",
    )
    ap.add_argument(
        "--skip-dv-relationship", action="store_true",
        help="skip the comments.txt §2.1 DV-relationship analysis (accuracy<->alignment<->organization "
             "correlations across ablation cells/seeds) -- needs results/network_properties.jsonl and "
             "results/dynamics_persistence.csv to already exist for full coverage.",
    )
    args = ap.parse_args(argv)
    if args.skip_permutation_null:
        args.n_permutations = 0

    # D33: a Gate A pass must not overwrite the Gate B pass's results. Same
    # convention as the activity logs and `run_geometry.py::out_csv_for`:
    # ckpt.pt keeps every original filename, anything else gets a suffix.
    from brainalign_wm.training.generate_activity_logs import activity_log_path

    _tag = "" if args.checkpoint == "ckpt.pt" else "_" + Path(args.checkpoint).stem.removeprefix("ckpt_")

    def out_csv(name: str) -> Path:
        return ROOT / "results" / f"{name}{_tag}.csv"

    cfg = yaml.safe_load(Path(args.config).read_text())
    completed = _load_completed_runs(ROOT / "results" / "manifest.jsonl")
    explicit_runs = {r.strip() for r in args.runs.split(",") if r.strip()} if args.runs else None
    if explicit_runs is not None:
        missing = explicit_runs - {r["run_id"] for r in completed}
        if missing:
            print(f"[run_all] --runs names run(s) with no completed manifest row: {sorted(missing)}")
            return 1
        completed = [r for r in completed if r["run_id"] in explicit_runs]
    if not completed:
        print("[run_all] no completed runs found in results/manifest.jsonl; run `make run-grid` first.")
        return 1
    print(f"[run_all] {len(completed)} completed run(s) found: {[r['run_id'] for r in completed]}")

    from brainalign_wm.neural.adapters.dandi_nwb import DandiSternbergTierA

    dandi_data = DandiSternbergTierA(
        cfg["paths"]["data_root"], datasets=tuple(cfg["neural"]["datasets_tierA"]),
        min_firing_hz=cfg["neural"]["min_firing_hz"],
        min_session_accuracy=cfg["neural"]["min_session_accuracy"],
        bin_ms=cfg["neural"]["bin_ms"],
        max_sessions_per_dataset=args.max_sessions_per_dataset,
    )
    print(f"[run_all] Tier A loaded: {len(dandi_data.sessions())} sessions, {len(dandi_data.units())} units, "
          f"regions={dandi_data.regions()}")

    maintenance_session_rows, probe_rows, lesion_rows, dynamics_rows, encoding_rows, dpca_rows = [], [], [], [], [], []
    for rec in completed:
        run_id = rec["run_id"]
        # Both auto-skips keep non-Core runs out of the POOLED S x M x P
        # regression rows. Naming a run with --runs is not a pooling
        # decision, so it overrides them (and says so in the log).
        named = explicit_runs is not None and run_id in explicit_runs
        if "P" not in rec and not named:
            # Extended local-learning cells (M**L, §6.3) carry "L", not "P"
            # -- reported on their own terms (rung reached + accuracy,
            # already in manifest.jsonl) rather than pooled into the Core
            # S x M x P RSA/dPCA/encoding analyses below.
            print(f"[run_all]   skipping {run_id} (local-learning cell, not part of the Core S x M x P grid)")
            continue
        if _is_ablation_or_catch_variant(rec):
            if not named:
                print(f"[run_all]   skipping {run_id} (ablation/identity-catch variant {rec['model_id']!r}, reported separately)")
                continue
            print(f"[run_all]   {run_id}: model_id {rec['model_id']!r} is not a Core cell id; included because "
                  f"--runs named it. Do not pool these rows into the Core S x M x P regression.")
        print(f"[run_all] aligning {run_id} ...")
        try:
            result = align_one_run(run_id, dandi_data, force_regenerate=args.regenerate,
                                   n_permutations=args.n_permutations, checkpoint=args.checkpoint)
        except FileNotFoundError as e:
            print(f"[run_all]   skipped ({e})")
            continue

        for row in result["maintenance"]:
            clean_row = {k: v for k, v in row.items() if k != "_perm_raw_alignments"}
            maintenance_session_rows.append({**clean_row, "model_id": rec["model_id"], "S": rec["S"], "M": rec["M"], "P": rec.get("P"), "T": rec["T"], "D": rec["D"], "seed": rec["seed"]})
        for row in result["probe"]:
            probe_rows.append({**row, "run_id": run_id, "model_id": rec["model_id"], "S": rec["S"], "M": rec["M"], "P": rec.get("P"), "T": rec["T"], "D": rec["D"], "seed": rec["seed"],
                                "accuracy_load1": rec.get("accuracy", {}).get("load1"), "accuracy_load3": rec.get("accuracy", {}).get("load3")})
        ok_n = sum(1 for r in result["maintenance"] if r.get("status") == "ok")
        print(f"[run_all]   maintenance: {ok_n}/{len(result['maintenance'])} session-rows ok; "
              f"probe: {[r.get('status') for r in result['probe']]}")

        if rec["M"] == 1 and not args.skip_reflection_shuffle:
            lesion = reflection_shuffle_lesion_for_run(run_id, dandi_data, checkpoint=args.checkpoint)
            lesion_rows.append({**lesion, "model_id": rec["model_id"], "S": rec["S"], "P": rec.get("P"), "T": rec["T"], "D": rec["D"], "seed": rec["seed"]})
            print(f"[run_all]   C1 reflection-shuffle lesion: {lesion.get('status')} "
                  f"(effect={lesion.get('lesion_effect')})" if lesion.get("status") == "ok" else
                  f"[run_all]   C1 reflection-shuffle lesion: {lesion.get('status')}")

        if not args.skip_dynamics:
            log_path = activity_log_path(run_id, args.checkpoint)
            if log_path.exists():
                from brainalign_wm.training.logging_schema import read_log

                dyn = dynamics_and_persistence_for_run(run_id, read_log(log_path), dandi_data)
                dynamics_rows.append({**dyn, "model_id": rec["model_id"], "S": rec["S"], "M": rec["M"], "P": rec.get("P"), "T": rec["T"], "D": rec["D"], "seed": rec["seed"]})
                print(f"[run_all]   H5/H6 dynamics/persistence: {dyn.get('status')}")

        if not args.skip_encoding:
            log_path = activity_log_path(run_id, args.checkpoint)
            if log_path.exists():
                from brainalign_wm.training.logging_schema import read_log

                enc_df = read_log(log_path)
                enc_rows_this_run = _encoding_for_run(run_id, enc_df, dandi_data, None)
                for row in enc_rows_this_run:
                    encoding_rows.append({**row, "model_id": rec["model_id"], "S": rec["S"], "M": rec["M"], "P": rec.get("P"), "T": rec["T"], "D": rec["D"], "seed": rec["seed"]})
                print(f"[run_all]   encoding models (pooled): {len(enc_rows_this_run)} session-rows")

        if not args.skip_dpca:
            log_path = activity_log_path(run_id, args.checkpoint)
            if log_path.exists():
                from brainalign_wm.training.logging_schema import read_log

                dpca_df = read_log(log_path)
                dpca_rows_this_run = _dpca_for_run(run_id, dpca_df, dandi_data, None)
                for row in dpca_rows_this_run:
                    dpca_rows.append({**row, "model_id": rec["model_id"], "S": rec["S"], "M": rec["M"], "P": rec.get("P"), "T": rec["T"], "D": rec["D"], "seed": rec["seed"]})
                print(f"[run_all]   dPCA (pooled): {len(dpca_rows_this_run)} session-rows")

    if lesion_rows:
        lesion_df = pd.DataFrame(lesion_rows)
        lesion_out = out_csv("reflection_shuffle_lesion")
        lesion_df.to_csv(lesion_out, index=False)
        print(f"\n[run_all] wrote {lesion_out}")

    if not args.skip_baselines and completed:
        # B1/B2 depend only on the SESSION and REGION, not on which trained
        # model we're comparing -- computed once (reusing any one completed
        # run's activity log for the session list), not once per run.
        from brainalign_wm.training.logging_schema import read_log

        baseline_run_id = completed[0]["run_id"]
        baseline_log_path = activity_log_path(baseline_run_id, args.checkpoint)
        if baseline_log_path.exists():
            baseline_df = read_log(baseline_log_path)
            print(f"\n[run_all] B1/B2 baselines (master protocol §4.2, using {baseline_run_id}'s session coverage) ...")
            baseline_rows = []
            for region in REGIONS:
                baseline_rows.extend(_baselines_for_run("baseline", baseline_df, dandi_data, region))
            if baseline_rows:
                baseline_df_out = pd.DataFrame(baseline_rows)
                baseline_out = out_csv("baselines")
                baseline_df_out.to_csv(baseline_out, index=False)
                pooled = baseline_df_out[baseline_df_out.region == "pooled"]
                for b in ("B1_encoder_only", "B2_task_model"):
                    sub = pooled[pooled.baseline == b]
                    if len(sub):
                        print(f"    {b}: mean normalized_alignment={sub.normalized_alignment.mean():.3f} (n={len(sub)} sessions)")
                print(f"[run_all] wrote {baseline_out}")

    if dynamics_rows:
        dynamics_df = pd.DataFrame(dynamics_rows)
        dynamics_out = out_csv("dynamics_persistence")
        dynamics_df.to_csv(dynamics_out, index=False)
        print(f"[run_all] wrote {dynamics_out}")

    if encoding_rows:
        encoding_df = pd.DataFrame(encoding_rows)
        encoding_out = out_csv("encoding_results")
        encoding_df.to_csv(encoding_out, index=False)
        print(f"[run_all] wrote {encoding_out}")

    if dpca_rows:
        dpca_out_df = pd.DataFrame(dpca_rows)
        dpca_out = out_csv("dpca_results")
        dpca_out_df.to_csv(dpca_out, index=False)
        print(f"[run_all] wrote {dpca_out}")

    if not maintenance_session_rows and not probe_rows:
        print("[run_all] no alignable runs (insufficient shared conditions or missing stimulus cache coverage).")
        return 1

    session_df = pd.DataFrame(maintenance_session_rows)
    session_out = out_csv("alignment_by_session")
    session_df.to_csv(session_out, index=False)
    print(f"\n[run_all] wrote {session_out} ({len(session_df)} rows)\n")

    probe_df = pd.DataFrame(probe_rows)

    # headline per-cell CSV: maintenance aggregated across sessions
    # (pooled-region only) + the pooled-region probe row, joined by run_id.
    # Aggregation is signed-mean-then-normalize-once, via
    # `_aggregate_maintenance` -- NOT `ok.normalized_alignment.mean()` (mean
    # of per-session values already clipped to [0,1], which rectifies
    # symmetric per-session noise into a spurious positive floor). Per-cell
    # permutation p-values are intentionally omitted from the headline: the
    # one-sided "greater" test against a null centered well below zero is
    # anti-informative (cells with negative alignment get flagged
    # "significant"), and the distributional chance gate (Mann-Whitney
    # trained vs. chance) is the correct, already-wired statistical test
    # for this DV.
    # Audit fix M10: explicit guard instead of a conditional-as-iterable
    # (`for x in y if cond else []`), which is easy to misread and silently
    # skips the whole loop rather than failing loudly on an empty session_df.
    headline_rows = []
    if len(session_df) == 0:
        print("[run_all] no session rows; skipping headline aggregation.")
    else:
        for run_id, g in session_df[session_df.get("region") == "pooled"].groupby("run_id"):
            ok = g[g.status == "ok"]
            if len(ok) == 0:
                continue
            headline_rows.append({
                "run_id": run_id, "model_id": ok.model_id.iloc[0], "S": ok.S.iloc[0], "M": ok.M.iloc[0], "P": ok.P.iloc[0],
                "T": ok.T.iloc[0], "D": ok.D.iloc[0], "seed": ok.seed.iloc[0],
                **_aggregate_maintenance(ok.to_dict("records")),
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

    out_path = out_csv("alignment_results")
    headline_df.to_csv(out_path, index=False)
    print(f"[run_all] wrote {out_path}\n")
    if len(headline_df):
        print(headline_df.to_string(index=False))

    probe_out = out_csv("alignment_probe_by_region")
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
    if len(ok_sessions) and ok_sessions["S"].nunique() >= 2 and ok_sessions["M"].nunique() >= 2 and ok_sessions["P"].nunique() >= 2 and len(ok_sessions) >= 6:
        sub = ok_sessions.rename(columns={"normalized_alignment": "align_score"}).merge(accuracy_lookup, on="run_id", how="left")
        res = mixed_effects_alignment(sub, formula="align_score ~ S * M * P + accuracy")
        print(f"\n[run_all] maintenance mixed-effects fit ({res['method']}):")
        for k, v in res["params"].items():
            print(f"    {k}: {v:.4f}")
    else:
        print("\n[run_all] insufficient factor coverage for mixed-effects inference on the maintenance path "
              "(need >=2 levels of S, M, P and >=6 session-rows); re-run once more cells complete.")

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
        # H5 needs a genuinely well-trained comparison model, not just
        # "M111 if present": M111 is the flagship scientific target cell,
        # but comparing an untrained model against a cell that itself never
        # behaviorally learned would make this gate compare chance to
        # chance, not chance to trained. Pick the row with the best
        # accuracy_load1 (ties broken by accuracy_load3) among completed
        # runs instead.
        best_idx = (headline_df["accuracy_load1"] + headline_df["accuracy_load3"] * 1e-3).idxmax()
        control_model_id = headline_df.loc[best_idx, "model_id"]
        control_seed = int(headline_df.loc[best_idx, "seed"])
        print(f"\n[run_all] H5 chance-model negative control ({control_model_id}, untrained) ...")
        chance_rows = []

        # Probe epoch: this single-cell comparison is solid and unambiguous
        # (chance=0.000 vs trained=0.337) -- kept as is, unlike the
        # maintenance path below.
        chance_best = chance_control_check(control_model_id, control_seed, dandi_data)
        trained_row = headline_df[(headline_df.model_id == control_model_id) & (headline_df.seed == control_seed)].iloc[0]
        chance_rows.append({**chance_best, "role": "chance_best_accuracy_model"})
        chance_rows.append({**trained_row.to_dict(), "role": "trained_best_accuracy_model"})
        if "probe_normalized_alignment" in chance_best and pd.notna(trained_row.get("probe_normalized_alignment")):
            verdict = "PASS" if chance_best["probe_normalized_alignment"] < trained_row["probe_normalized_alignment"] else "FAIL"
            print(f"    probe_normalized_alignment: chance={chance_best['probe_normalized_alignment']:.3f} "
                  f"trained={trained_row['probe_normalized_alignment']:.3f}  [{verdict}]")

        # Maintenance epoch: a single untrained model vs a single best-
        # accuracy cell is meaningless given how noisy the per-cell DV is
        # -- a single-cell comparison can't distinguish signal from
        # per-cell noise. Instead: one untrained ("chance") instantiation
        # per distinct architecture present in the grid, compared against
        # the full trained-cell distribution via a one-sided Mann-Whitney
        # test (is the trained distribution stochastically greater than
        # chance?), reusing `stats.compare_distributions` (already used for
        # H5/H6's model-vs-brain distribution comparisons). Two
        # pseudoreplication guards: (a) deduplicate chance values to one
        # per distinct architecture -- an untrained forward pass ignores P,
        # so M**0 and M**1 chance rows are byte-identical, effective chance
        # n = 4, not 8; (b) aggregate trained values to per-cell means
        # before the test -- the trained rows are 8 cells x N seeds; seeds
        # within a cell are correlated, not independent.
        from brainalign_wm.analysis.stats import compare_distributions

        chance_by_arch = {}
        for model_id in sorted(headline_df["model_id"].unique()):
            model_row = headline_df[headline_df.model_id == model_id].iloc[0]
            c = chance_control_check(model_id, int(model_row["seed"]), dandi_data)
            chance_rows.append({**c, "role": "chance_maintenance_distribution"})
            if "maintenance_normalized_alignment" in c:
                arch = model_id[:-1]  # M**0 / M**1 share architecture
                if arch not in chance_by_arch:
                    chance_by_arch[arch] = c["maintenance_normalized_alignment"]
        chance_maintenance = list(chance_by_arch.values())

        trained_by_cell = headline_df.groupby("model_id")["maintenance_normalized_alignment"].mean().dropna()
        trained_maintenance = trained_by_cell.to_numpy()
        if len(chance_maintenance) >= 2 and len(trained_maintenance) >= 2:
            dist_test = compare_distributions(trained_maintenance, np.array(chance_maintenance), alternative="greater")
            verdict = "PASS" if dist_test["p_value"] < 0.05 and dist_test["rank_biserial"] > 0 else "FAIL"
            print(f"    maintenance_normalized_alignment (trained-distribution vs chance-distribution): "
                  f"trained n={len(trained_maintenance)} mean={trained_maintenance.mean():.3f}, "
                  f"chance n={len(chance_maintenance)} mean={np.mean(chance_maintenance):.3f}, "
                  f"Mann-Whitney p={dist_test['p_value']:.3f}, rank_biserial={dist_test['rank_biserial']:.3f}  [{verdict}]")
            chance_rows.append({
                "role": "maintenance_distributional_test", **dist_test,
                "trained_n": len(trained_maintenance), "trained_mean": float(trained_maintenance.mean()),
                "chance_n": len(chance_maintenance), "chance_mean": float(np.mean(chance_maintenance)),
            })
        else:
            print("    maintenance_normalized_alignment (trained-distribution vs chance-distribution): "
                  "insufficient completed cells yet for a distributional test.")

        pd.DataFrame(chance_rows).to_csv(out_csv("chance_control"), index=False)
        print(f"[run_all] wrote {ROOT / 'results' / 'chance_control.csv'}")

    if not args.skip_dv_relationship:
        from brainalign_wm.analysis.dv_relationship import run_dv_relationship

        print("\n[run_all] DV-relationship analysis (comments.txt §2.1) ...")
        run_dv_relationship()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
