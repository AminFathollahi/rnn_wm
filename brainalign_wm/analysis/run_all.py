"""Runs the alignment and statistical-inference pipeline over whatever
training runs have completed so far (reads `results/manifest.jsonl`; does
not require the full grid to have finished).

Audit-fix rewrite (A2a-A2g, B3, C2): two epoch-appropriate condition
schemas, each on the representation appropriate to it (A2c: raw alignment
and its noise ceiling are always computed on the SAME representation):

  * **maintenance** (delay-period) alignment uses REAL held-item IDENTITY
    (preferred) or REAL dataset-embedded CATEGORY (fallback) + load as the
    condition (A2a/A2b -- NOT the probe/response-time in_set/correct
    labels, which cannot causally structure activity that occurred before
    the probe). Two prior versions of this code got this wrong or
    compromised, both corrected after checking the real data directly
    (see DECISIONS.md for the full history):
      1. First assumed identity never repeats (checked only dataset
         000673, where every trial genuinely does draw a fresh,
         non-repeating item set) and imputed a category label via
         nearest-neighbor in the model's OWN frozen-encoder feature space
         -- risking a confound where the metric measures "how much of the
         encoder survives training" rather than task-driven
         representational change (plausibly why a chance/untrained model
         scored ABOVE a well-trained one during validation).
      2. Checking ALL 65 sessions found dataset 000469 (21 sessions) uses
         a FIXED, CLOSED pool of exactly 25 item-sets per session, each
         repeated ~8-10 times -- real repeat structure supporting a proper
         crossnobis condition mean on TRUE item identity, no proxy needed.
      3. For dataset 000673 (44 sessions, genuinely non-repeating item
         identity), the raw PicID values themselves decode a REAL,
         dataset-embedded category: `pid // 100` gives exactly 5 groups of
         ~55-57 images each. This is not a guess -- it is EXACTLY the
         convention the original authors' own published analysis code
         uses: `NWB_calcSelective_SB.m` in
         github.com/rutishauserlab/SBCAT-release-NWB (cited by the
         dataset's own `dandiset.yaml`) derives category as
         `str2double(num2str(picID)(1))`, i.e. the first digit of the
         PicID -- arithmetically identical to `pid // 100` for these
         3-digit codes. Independently corroborated against the cached
         ResNet features (mean within-group cosine similarity 0.64 vs.
         0.52 across-group). A genuine experimenter-assigned label, NOT a
         model-derived proxy, so it carries none of (1)'s confound.
    `_maintenance_condition_fn` picks per-session, from the session's own
    REAL trials (identical choice applied to both model replay and neural
    data): real identity if it has repeat structure in this session, else
    the real `pid // 100` category, else the session contributes nothing
    (honestly reported as "insufficient_shared_conditions", A2f, rather
    than worked around). Computed PER SESSION: a per-session crossnobis
    RDM (model replay vs. that session's neural trials), against that
    session's OWN within-session reliability ceiling
    (`rsa.within_session_noise_ceiling` -- a cross-session LOSO ceiling is
    meaningless here since conditions are session-specific pools, not
    shared across sessions). Per-session rows carry the session's patient
    (via `patient_of`), directly feeding the LME's patient factor (B3).
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


def _identity_condition(row) -> tuple:
    """Maintenance-epoch condition, identity variant (audit fix A2a/A2b):
    the SORTED SET of real held-item PicIDs + load, NOT the probe/response-
    time in_set/correct labels (which cannot causally structure activity
    that occurred before the probe). Only sessions with a genuinely
    repeating item-set pool (dataset 000469) produce enough same-condition
    trials to pass the per-session min-trial filter and
    MIN_SHARED_CONDITIONS_MAINTENANCE floor below -- see module
    docstring."""
    return (tuple(sorted(int(x) for x in row.held_items)), int(row.load))


def _category_condition(row) -> tuple:
    """Maintenance-epoch condition, REAL-category variant (audit fix
    A2a/A2b, dataset 000673): the SORTED MULTISET of held-item CATEGORY
    codes + load, where category = first digit of the PicID (`pid // 100`)
    -- the exact convention the original authors' own published analysis
    code uses (`NWB_calcSelective_SB.m`,
    github.com/rutishauserlab/SBCAT-release-NWB: `CAT = str2double(
    num2str(picID)(1))`), NOT a model-derived proxy. See module
    docstring."""
    return (tuple(sorted(int(x) // 100 for x in row.held_items)), int(row.load))


def _session_has_identity_repeats(session_trials: pd.DataFrame, min_conditions: int = 2, min_count: int = 2) -> bool:
    """True if this session's REAL held-item-set+load combinations recur
    often enough (>=`min_conditions` distinct combinations with
    >=`min_count` trials each) to support `_identity_condition` directly;
    False means fall back to `_category_condition` (see module
    docstring). Decided from the session's own real trial table so the
    SAME choice applies identically to the model replay and the neural
    data for that session."""
    counts: dict = {}
    for row in session_trials.itertuples():
        key = (tuple(sorted(int(x) for x in row.held_items)), int(row.load))
        counts[key] = counts.get(key, 0) + 1
    return sum(1 for c in counts.values() if c >= min_count) >= min_conditions


def _maintenance_condition_fn(session_trials: pd.DataFrame) -> callable:
    """Picks `_identity_condition` or `_category_condition` for one session
    based on that session's own real trial data (see module docstring for
    the A2a/A2b design and its history)."""
    return _identity_condition if _session_has_identity_repeats(session_trials) else _category_condition


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


# Maintenance-epoch achievable condition space varies by which schema a
# session uses (`_maintenance_condition_fn`): 000469-identity sessions can
# reach up to 25 real item-sets x load; 000673-category sessions are capped
# at 5 categories x load = up to 15. Both are smaller than the probe
# epoch's blanket floor would allow through, so a schema-appropriate floor
# is used here instead (still a real guard against a degenerate 1-2
# condition RDM, A2f).
MIN_SHARED_CONDITIONS_MAINTENANCE = 4


def _maintenance_alignment_for_run(run_id: str, model_df: pd.DataFrame, dandi_data, region) -> list[dict]:
    """Per-session maintenance-epoch alignment, using real identity or real
    category per session (see `_maintenance_condition_fn` and the module
    docstring). Returns one row per session with usable data (both model
    replay coverage and neural trials for that session)."""
    from brainalign_wm.analysis.rsa import compare_rdms, within_session_noise_ceiling
    from brainalign_wm.analysis import rsa as rsa_mod
    from brainalign_wm.analysis.rdm import crossnobis_rdm

    rows = []
    model_sessions = set(model_df["session"].unique()) if "session" in model_df.columns else set()
    all_trials = dandi_data.trials()
    for session_id in sorted(model_sessions):
        session_trials = all_trials[all_trials.session == session_id]
        if len(session_trials) == 0:
            continue
        condition_fn = _maintenance_condition_fn(session_trials)

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


def _task_model_rdm(conds: list) -> np.ndarray:
    """B2 (master protocol §4.2, non-negotiable): the RDM implied by
    ground-truth task variables alone -- a hypothetical "perfect task
    solver" representation that encodes EXACTLY the condition-defining
    variables (item/category identity, load) and nothing else. Each
    condition here is `(item_or_category_tuple, load)`; distance between
    two conditions = (0 if same item/category else 1) + (0 if same load
    else 1), normalized to [0,1]. Needs no real data at all -- built
    directly from the condition list."""
    n = len(conds)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            item_i, load_i = conds[i]
            item_j, load_j = conds[j]
            d = (0.0 if item_i == item_j else 1.0) + (0.0 if load_i == load_j else 1.0)
            rdm[i, j] = rdm[j, i] = d / 2.0
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
    §4.2: computed per session, on the SAME shared-condition set and
    neural RDM the real maintenance alignment uses for that session/region,
    so B1/B2 rows are directly comparable to the trained-model rows in
    `alignment_by_session.csv`."""
    from brainalign_wm.analysis.rsa import compare_rdms, within_session_noise_ceiling
    from brainalign_wm.analysis import rsa as rsa_mod
    from brainalign_wm.analysis.rdm import crossnobis_rdm

    rows = []
    model_sessions = set(model_df["session"].unique()) if "session" in model_df.columns else set()
    all_trials = dandi_data.trials()
    for session_id in sorted(model_sessions):
        session_trials = all_trials[all_trials.session == session_id]
        if len(session_trials) == 0:
            continue
        condition_fn = _maintenance_condition_fn(session_trials)

        neural_rdm, neural_conds = rsa_mod._session_condition_rdm(
            dandi_data, session_id, region, "maintain", dandi_data.bin_ms, condition_fn=condition_fn,
        )
        if neural_rdm is None:
            continue
        ceiling_lower, ceiling_upper = within_session_noise_ceiling(
            dandi_data, session_id, region, "maintain", dandi_data.bin_ms, condition_fn=condition_fn,
        )

        # B1: encoder-only
        b1_patterns, b1_labels = _encoder_only_patterns_for_session(session_id, session_trials, condition_fn)
        b1_patterns, b1_labels = _filter_min_trials(b1_patterns, b1_labels, min_count=2)
        if len(set(b1_labels)) >= 2:
            n_folds_b1 = max(2, min(4, min(b1_labels.count(l) for l in set(b1_labels))))
            if n_folds_b1 >= 2:
                b1_rdm, b1_conds = crossnobis_rdm(b1_patterns, b1_labels, n_folds=n_folds_b1)
                shared_b1 = _shared_conditions_or_none(b1_conds, neural_conds, floor=MIN_SHARED_CONDITIONS_MAINTENANCE)
                if shared_b1 is not None:
                    m_idx = [b1_conds.index(c) for c in shared_b1]
                    n_idx = [neural_conds.index(c) for c in shared_b1]
                    raw_b1 = compare_rdms(b1_rdm[np.ix_(m_idx, m_idx)], neural_rdm[np.ix_(n_idx, n_idx)])
                    rows.append({
                        "run_id": run_id, "session": session_id, "region": region or "pooled", "baseline": "B1_encoder_only",
                        "patient": dandi_data.patient_of(session_id), "status": "ok", "raw_alignment": raw_b1,
                        "noise_ceiling_upper": ceiling_upper, "normalized_alignment": _norm(raw_b1, ceiling_upper),
                        "n_shared_conditions": len(shared_b1),
                    })

        # B2: task-model (ground-truth condition structure only)
        b2_rdm = _task_model_rdm(neural_conds)
        raw_b2 = compare_rdms(b2_rdm, neural_rdm)
        rows.append({
            "run_id": run_id, "session": session_id, "region": region or "pooled", "baseline": "B2_task_model",
            "patient": dandi_data.patient_of(session_id), "status": "ok", "raw_alignment": raw_b2,
            "noise_ceiling_upper": ceiling_upper, "normalized_alignment": _norm(raw_b2, ceiling_upper),
            "n_shared_conditions": len(neural_conds),
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
    ceiling (`_maintenance_condition_fn`-selected schema), reused here as a
    practical proxy for the per-target-neuron reliability ceiling a
    dedicated single-unit split-half estimate would give (not separately
    computed, for tractability)."""
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
        condition_fn = _maintenance_condition_fn(session_trials)
        _, ceiling_upper = within_session_noise_ceiling(dandi_data, session_id, region, "maintain", dandi_data.bin_ms, condition_fn=condition_fn)
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


def align_one_run(run_id: str, dandi_data, force_regenerate: bool = False) -> dict:
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
        maintenance_rows.extend(_maintenance_alignment_for_run(run_id, df, dandi_data, region))
        probe_rows.append(_probe_alignment_for_run(run_id, df, dandi_data, region))
    return {"maintenance": maintenance_rows, "probe": probe_rows}


def reflection_shuffle_lesion_for_run(run_id: str, dandi_data) -> dict:
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
    normal_rows = _maintenance_alignment_for_run(run_id, normal_df, dandi_data, None)
    normal_ok = [r for r in normal_rows if r.get("status") == "ok"]

    shuffled_path = generate_activity_log_reflection_shuffled(run_id, dandi_data)
    shuffled_df = read_log(shuffled_path)
    shuffled_rows = _maintenance_alignment_for_run(run_id, shuffled_df, dandi_data, None)
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


def chance_control_check(model_id: str, seed: int, dandi_data) -> dict:
    """comments.txt acceptance-gate H5: an untrained (chance) model must
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

    maintenance_session_rows, probe_rows, lesion_rows, dynamics_rows, encoding_rows, dpca_rows = [], [], [], [], [], []
    for rec in completed:
        run_id = rec["run_id"]
        print(f"[run_all] aligning {run_id} ...")
        try:
            result = align_one_run(run_id, dandi_data, force_regenerate=args.regenerate)
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
            lesion = reflection_shuffle_lesion_for_run(run_id, dandi_data)
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

        if not args.skip_encoding:
            log_path = ROOT / "results" / "activity_logs" / f"{run_id}.parquet"
            if log_path.exists():
                from brainalign_wm.training.logging_schema import read_log

                enc_df = read_log(log_path)
                enc_rows_this_run = _encoding_for_run(run_id, enc_df, dandi_data, None)
                for row in enc_rows_this_run:
                    encoding_rows.append({**row, "model_id": rec["model_id"], "S": rec["S"], "M": rec["M"], "L": rec["L"], "seed": rec["seed"]})
                print(f"[run_all]   encoding models (pooled): {len(enc_rows_this_run)} session-rows")

        if not args.skip_dpca:
            log_path = ROOT / "results" / "activity_logs" / f"{run_id}.parquet"
            if log_path.exists():
                from brainalign_wm.training.logging_schema import read_log

                dpca_df = read_log(log_path)
                dpca_rows_this_run = _dpca_for_run(run_id, dpca_df, dandi_data, None)
                for row in dpca_rows_this_run:
                    dpca_rows.append({**row, "model_id": rec["model_id"], "S": rec["S"], "M": rec["M"], "L": rec["L"], "seed": rec["seed"]})
                print(f"[run_all]   dPCA (pooled): {len(dpca_rows_this_run)} session-rows")

    if lesion_rows:
        lesion_df = pd.DataFrame(lesion_rows)
        lesion_out = ROOT / "results" / "reflection_shuffle_lesion.csv"
        lesion_df.to_csv(lesion_out, index=False)
        print(f"\n[run_all] wrote {lesion_out}")

    if not args.skip_baselines and completed:
        # B1/B2 depend only on the SESSION and REGION, not on which trained
        # model we're comparing -- computed once (reusing any one completed
        # run's activity log for the session list), not once per run.
        from brainalign_wm.training.logging_schema import read_log

        baseline_run_id = completed[0]["run_id"]
        baseline_log_path = ROOT / "results" / "activity_logs" / f"{baseline_run_id}.parquet"
        if baseline_log_path.exists():
            baseline_df = read_log(baseline_log_path)
            print(f"\n[run_all] B1/B2 baselines (master protocol §4.2, using {baseline_run_id}'s session coverage) ...")
            baseline_rows = []
            for region in REGIONS:
                baseline_rows.extend(_baselines_for_run("baseline", baseline_df, dandi_data, region))
            if baseline_rows:
                baseline_df_out = pd.DataFrame(baseline_rows)
                baseline_out = ROOT / "results" / "baselines.csv"
                baseline_df_out.to_csv(baseline_out, index=False)
                pooled = baseline_df_out[baseline_df_out.region == "pooled"]
                for b in ("B1_encoder_only", "B2_task_model"):
                    sub = pooled[pooled.baseline == b]
                    if len(sub):
                        print(f"    {b}: mean normalized_alignment={sub.normalized_alignment.mean():.3f} (n={len(sub)} sessions)")
                print(f"[run_all] wrote {baseline_out}")

    if dynamics_rows:
        dynamics_df = pd.DataFrame(dynamics_rows)
        dynamics_out = ROOT / "results" / "dynamics_persistence.csv"
        dynamics_df.to_csv(dynamics_out, index=False)
        print(f"[run_all] wrote {dynamics_out}")

    if encoding_rows:
        encoding_df = pd.DataFrame(encoding_rows)
        encoding_out = ROOT / "results" / "encoding_results.csv"
        encoding_df.to_csv(encoding_out, index=False)
        print(f"[run_all] wrote {encoding_out}")

    if dpca_rows:
        dpca_out_df = pd.DataFrame(dpca_rows)
        dpca_out = ROOT / "results" / "dpca_results.csv"
        dpca_out_df.to_csv(dpca_out, index=False)
        print(f"[run_all] wrote {dpca_out}")

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
        # H5 needs a genuinely well-trained comparison model -- NOT "M111
        # if present" (the previous default): M111 is the flagship
        # SCIENTIFIC target cell, but it's also an L=1 cell, and L=1 has
        # repeatedly failed to train above near-chance behavior (see
        # RESPONSES.md). Comparing an untrained model against a cell that
        # itself never behaviorally learned makes this gate compare chance
        # to chance, not chance to trained -- found via a real post-grid
        # run where this silently produced a FAIL (chance=trained=0.000)
        # that had nothing to do with the chance-control machinery itself.
        # Pick the row with the best accuracy_load1 (ties broken by
        # accuracy_load3) among completed runs instead.
        best_idx = (headline_df["accuracy_load1"] + headline_df["accuracy_load3"] * 1e-3).idxmax()
        control_model_id = headline_df.loc[best_idx, "model_id"]
        control_seed = int(headline_df.loc[best_idx, "seed"])
        print(f"\n[run_all] H5 chance-model negative control ({control_model_id}, untrained) ...")
        chance = chance_control_check(control_model_id, control_seed, dandi_data)
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
