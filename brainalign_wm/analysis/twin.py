"""Digital-twin orchestration (Phase 8b, comments.txt §5 items 8.4c/8.4d,
8.5, 8.6): the pieces of the analysis suite that need to DO something
interventional -- perturb a rolling state and observe the causal effect, or
gate a controller on a live decoded read-out -- rather than analyze a
static logged array. `geometry.py` holds the passive array-in-array-out
functions (DMD fit, dominant direction, LQR gain design); this module
drives them against a step function, which in later phases (item 8.10's
driver script) is a trained model's forward pass and here (this sub-phase's
tests) is a synthetic linear system standing in for one -- the causal/
control logic itself doesn't care which.

CONSTRAINT (item 8.3, C3, inherited from `geometry.py`): every ensemble
here is single-trial; nothing here trial-averages before measuring.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from brainalign_wm.analysis.cross_temporal import cross_temporal_decoding, stability_index
from brainalign_wm.analysis.geometry import (
    axis_rotation_angle,
    content_context_rotation,
    dmd_ensemble_fit,
    lqr_gain,
    mean_speed,
    pca_participation_ratio,
)

StepFn = Callable[[np.ndarray], np.ndarray]  # x_t -> x_{t+1} (one system/model step)


# ---------------- item 8.4c: causal perturbation vs. decodability ----------------

def _fit_content_decoder(X: np.ndarray, labels: np.ndarray) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(Xs, labels)
    return clf, scaler


def perturb_and_measure_decodability(
    step_fn: StepFn,
    x0_batch: np.ndarray,
    content_labels: np.ndarray,
    v_star: np.ndarray,
    v_context: np.ndarray,
    t_perturb: int,
    t_final: int,
    perturb_scale: float,
    n_random_dirs: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Item 8.4c: roll `x0_batch` forward through `step_fn` to `t_perturb`,
    add a perturbation of magnitude `perturb_scale` along {v_star, N random
    unit directions, v_context} (one condition at a time), continue rolling
    to `t_final`, and measure the CAUSAL effect on a content decoder fit on
    the UNPERTURBED trajectory at `t_final` -- "do what cannot be done in
    patients: perturb ... and measure the causal effect on decodability."

    `x0_batch`: `[n_trials, n_units]` initial states (one per single trial,
    C3). `content_labels`: `[n_trials]` binary. `v_star`/`v_context`:
    `[n_units]` unit directions. Returns `{"baseline_acc",
    "vstar_acc", "vstar_drop", "context_acc", "context_drop",
    "random_acc_mean", "random_drop_mean"}` -- `*_drop` = baseline_acc -
    perturbed_acc, so a LARGER drop means that direction hurt decodability
    MORE (the ordering item 8.4c's prediction is about)."""
    if rng is None:
        rng = np.random.default_rng(0)

    def _rollout(x_batch: np.ndarray, t_start: int, t_stop: int) -> np.ndarray:
        x = x_batch.copy()
        for _ in range(t_start, t_stop):
            x = np.stack([step_fn(xi) for xi in x])
        return x

    # Unperturbed control: roll all the way to t_final, fit + score the
    # content decoder there. This IS the baseline both the decoder is
    # fit on and every perturbed condition is compared against.
    x_final_unperturbed = _rollout(x0_batch, 0, t_final)
    clf, scaler = _fit_content_decoder(x_final_unperturbed, content_labels)
    baseline_acc = float(clf.score(scaler.transform(x_final_unperturbed), content_labels))

    x_at_perturb = _rollout(x0_batch, 0, t_perturb)

    def _condition_acc(direction: np.ndarray) -> float:
        x_perturbed = x_at_perturb + perturb_scale * direction[None, :]
        x_final = _rollout(x_perturbed, t_perturb, t_final)
        return float(clf.score(scaler.transform(x_final), content_labels))

    vstar_acc = _condition_acc(v_star)
    context_acc = _condition_acc(v_context)
    random_accs = []
    for _ in range(n_random_dirs):
        d = rng.standard_normal(v_star.shape[0])
        d /= np.linalg.norm(d) + 1e-12
        random_accs.append(_condition_acc(d))
    random_acc_mean = float(np.mean(random_accs))

    return {
        "baseline_acc": baseline_acc,
        "vstar_acc": vstar_acc, "vstar_drop": baseline_acc - vstar_acc,
        "context_acc": context_acc, "context_drop": baseline_acc - context_acc,
        "random_acc_mean": random_acc_mean, "random_drop_mean": baseline_acc - random_acc_mean,
    }


# ---------------- item 8.4d: on-demand LQR controller ----------------

def on_demand_lqr_control(
    A: np.ndarray, B: np.ndarray, x0: np.ndarray, T: int,
    decode_confidence_fn: Callable[[np.ndarray], float], threshold: float,
    x_ref: Optional[np.ndarray] = None, q_state: float = 1.0, r_control: float = 1.0,
    process_noise_sigma: float = 0.0, rng: Optional[np.random.Generator] = None,
) -> dict:
    """Item 8.4: "an LQR/on-demand controller that perturbs only when a
    decoded read-out says the memorandum is slipping." Trigger convention
    (a documented choice, not the companion's `stimulation_trigger_window`
    flow-divergence trigger): `decode_confidence_fn(x)` is a decoded
    read-out of how well the memorandum is currently held (e.g. a fitted
    decoder's class-margin/probability); the controller fires
    `u = -K(x - x_ref)` (gain `K` from `geometry.lqr_gain`, same design as
    the companion's `lqr_design`) only on ticks where confidence drops below
    `threshold`, else `u = 0` -- chosen over a flow-divergence trigger
    because "decoded read-out" is the literal spec wording and a decoder
    confidence is what item 8.4c already builds upstream, not a new signal.

    Simulates a NO-CONTROL baseline (`u=0` always) over the same `T` steps
    from the same `x0` for comparison. Reports item 8.4's explicit asks:
    `drift_reduction` (baseline mean state-cost `||x-x_ref||^2` minus
    controlled mean state-cost -- the companion's `Q=q_state*I` state-cost
    convention, so positive = improvement), `decodability_lift` (controlled
    mean confidence minus baseline mean confidence), `duty_cycle` (fraction
    of ticks the controller actually fired -- must be strictly between 0
    and 1 for the gating to be doing anything, not always-on or inert).

    `process_noise_sigma` (default 0, deterministic): with an unstable `A`
    and NO noise, a single correction fully re-stabilizes a linear system
    and the controller never needs to fire again (`duty_cycle` collapses
    toward 0 despite genuinely working) -- nonzero noise gives repeated,
    on-demand re-triggering, the realistic regime this is meant to
    demonstrate against (a real RNN's rolling hidden state is never
    noise-free). The SAME noise draw is replayed for the gated and
    ungated simulations so the comparison isn't confounded by which one
    got luckier noise."""
    n = A.shape[0]
    if x_ref is None:
        x_ref = np.zeros(n)
    if rng is None:
        rng = np.random.default_rng(0)
    design = lqr_gain(A, B, q_state=q_state, r_control=r_control)
    K = design["K"]
    # Same noise draw replayed for both simulations (rng re-seeded to the
    # same state before each) so the gated-vs-ungated comparison isn't
    # confounded by which one happened to get luckier noise.
    noise = rng.standard_normal((T, n)) * process_noise_sigma

    def _simulate(gated: bool) -> tuple[np.ndarray, int]:
        x = np.zeros((T + 1, n))
        x[0] = x0
        n_fired = 0
        for k in range(T):
            u = np.zeros(B.shape[1])
            if gated and decode_confidence_fn(x[k]) < threshold:
                u = -K @ (x[k] - x_ref)
                n_fired += 1
            x[k + 1] = A @ x[k] + B @ u + noise[k]
        return x, n_fired

    x_baseline, _ = _simulate(gated=False)
    x_controlled, n_fired = _simulate(gated=True)

    def _state_cost(x_traj: np.ndarray) -> float:
        return float(np.mean(np.sum((x_traj - x_ref) ** 2, axis=-1)))

    drift_baseline = _state_cost(x_baseline)
    drift_controlled = _state_cost(x_controlled)
    conf_baseline = float(np.mean([decode_confidence_fn(x) for x in x_baseline]))
    conf_controlled = float(np.mean([decode_confidence_fn(x) for x in x_controlled]))

    return {
        "drift_reduction": drift_baseline - drift_controlled,
        "decodability_lift": conf_controlled - conf_baseline,
        "duty_cycle": n_fired / T,
        "drift_baseline": drift_baseline, "drift_controlled": drift_controlled,
    }


# ---------------- item 8.5: correct vs. incorrect trial split ----------------

def correct_vs_incorrect_geometry(
    Z: np.ndarray, correct: np.ndarray, content_labels: np.ndarray, context_labels: np.ndarray,
    t_start: int, t_end: int,
) -> dict:
    """Item 8.5: re-run the drift (`mean_speed`), rotation
    (`content_context_rotation`), and manifold-spread (`pca_participation_ratio`)
    measures SEPARATELY on the `correct=True` vs. `correct=False` trial
    subsets of the SAME `Z`/label arrays -- "does the manifold differ on
    error trials: more drift, lower decodability, larger rotation?" `Z`:
    `[n_trials, n_timebins, n_units]` (single-trial, C3). `correct`:
    `[n_trials]` bool."""
    correct = np.asarray(correct, dtype=bool)
    out = {}
    for label, mask in (("correct", correct), ("incorrect", ~correct)):
        Zm = Z[mask]
        if Zm.shape[0] < 4:
            out[label] = None
            continue
        rot = content_context_rotation(
            Zm, content_labels[mask], context_labels[mask], t_start=t_start, t_end=t_end,
        )
        out[label] = {
            "drift_speed": mean_speed(Zm),
            "content_rotation_deg": rot["content_rotation_deg"],
            "manifold_pr": pca_participation_ratio(Zm[:, t_start:t_end, :].reshape(-1, Zm.shape[-1])),
            "n_trials": int(mask.sum()),
        }
    if out["correct"] is not None and out["incorrect"] is not None:
        out["incorrect_minus_correct"] = {
            "drift_speed": out["incorrect"]["drift_speed"] - out["correct"]["drift_speed"],
            "content_rotation_deg": out["incorrect"]["content_rotation_deg"] - out["correct"]["content_rotation_deg"],
            "manifold_pr": out["incorrect"]["manifold_pr"] - out["correct"]["manifold_pr"],
        }
    return out


# ---------------- item 8.6: cross-temporal generalization of memorandum identity ----------------

def cross_temporal_by_position(
    Z: np.ndarray, position_labels: dict[int, np.ndarray], n_folds: int = 4, seed: int = 0,
) -> dict:
    """Item 8.6: "full T x T decoder-generalization matrix per serial
    position, plus stability_index -- mostly wiring over existing code."
    Reuses `analysis/cross_temporal.py::cross_temporal_decoding`/
    `stability_index` directly (this repo's own convention, distinct from
    `geometry.cross_temporal_generalization_auc`'s companion-mirroring
    LinearSVC+AUC convention used for item 8.1). `position_labels`: e.g.
    `{1: y_pos1, 2: y_pos2, 3: y_pos3}`, one binary "was this position's
    item the memorandum tested" (or equivalent per-position memorandum-
    identity label, position-specific) label array per serial position,
    each `[n_trials]`. `Z`: `[n_trials, n_timebins, n_units]`. Returns
    `{position: {"matrix": [T,T], "stability_index": float}}`."""
    out = {}
    for position, labels in position_labels.items():
        mat = cross_temporal_decoding(Z, labels, n_folds=n_folds, seed=seed)
        out[position] = {"matrix": mat, "stability_index": stability_index(mat)}
    return out
