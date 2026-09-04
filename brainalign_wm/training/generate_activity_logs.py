"""Generates post-training model-activity logs for alignment analysis.

`train_one` persists final weights and summary accuracy but does not log
per-tick activity (the `LogRecord` schema exists precisely so model
representations can be compared against the neural recordings, but nothing
populates it during training -- logging every training tick would produce
an enormous, mostly-uninformative Parquet file, since what alignment
analysis needs is the *trained* model's representations, not a training-time
trace).

This module instead replays each session's *exact* recorded trials (their
`loads`, held item identities, and probe identity) through a trained,
frozen checkpoint, driving the visual front end with the dataset's own
cached stimulus-image features (see `neural/adapters/dandi_nwb.py::
cache_stimulus_features`) rather than the broad training pool -- the
"exact-image alignment" scheme: the model is trained on a broad naturalistic
pool but *evaluated*, for alignment purposes, on the identical images the
recorded patients viewed. Every tick of the replay is logged via
`training.logging_schema.LogRecord`, tagged with the source `session` so
downstream alignment can restrict to session-matched (model, neural) pairs
(see `analysis/run_all.py`).

The M=1 reflective gate's R_t is driven by the replay's own evolving
per-tick surprise (reward-prediction error at feedback, self-generated
surprise -- 1 - p(chosen action) -- elsewhere), computed exactly as
`training/train.py::_run_trial` does, using the replay's own policy/value
output and the trial's real recorded outcome as the feedback reward --
not a static `delta_t = 0`, which would leave R_t constant throughout
replay and erase M's dynamic signature from the logs the alignment DV is
computed on.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from brainalign_wm.config import get_path
from brainalign_wm.training.logging_schema import LogRecord, ParquetLogWriter
from brainalign_wm.training.train import ROOT, _build_model, _gate_width, _init_state, _load_full_config, _step_core
from brainalign_wm.tasks.sternberg import context_vector

# 5-arm bio-plausibility ablation battery (v6.0, arm order fixed as
# S,M,P,T,D): model IDs are "M" + 5 binary digits, e.g. M11111 (full
# reference), M00000 (baseline). Matched with re.match (not fullmatch) so
# a run_id suffix (bio-plausible-ablation-battery/identity-catch/perf-
# matched-baseline arms, e.g. "M11111_energy", "M00000_idcatch") stays
# attached to `model_id` without breaking the digit match.
_RE_5BIT = re.compile(r"^M([01])([01])([01])([01])([01])")
# Extended local-learning study (§6.3): a SEPARATE, unchanged 2-bit + "L"
# cell family (M00L/M01L/M10L/M11L) -- dispatched to its own branch below,
# never migrated to the 5-bit scheme.
_RE_LOCAL = re.compile(r"^M([01])([01])L")
RESULTS = get_path("results")
ACTIVITY_LOGS = get_path("activity_logs")


def _parse_model_id(model_id: str) -> tuple[int, int, int, int, int]:
    """'M11111' -> (S=1,M=1,P=1,T=1,D=1); 'M10111' -> (1,0,1,1,1).
    'M10L' -> (S=1,M=0,P=0,T=0,D=0) -- a local-learning cell (§6.3) is
    architecturally identical to the P=0 Core cell with the same S/M
    (plasticity is orthogonal to which algorithm trained the weights), so
    P=0 is the right replay architecture for it, and T/D never apply to
    the local-learning study (kept separate from the 5-arm ablation), so
    both are 0. This module only ever replays a FROZEN checkpoint
    (forward pass only), which never depends on L, T, or D (T/D are
    training-time LOSS penalties only -- see training/train.py -- they
    change no parameter shape, so a frozen checkpoint's forward pass is
    identical regardless of what T/D were set to at train time)."""
    m5 = _RE_5BIT.match(model_id)
    if m5:
        S, M, P, T, D = (int(x) for x in m5.groups())
        return S, M, P, T, D
    mL = _RE_LOCAL.match(model_id)
    if mL:
        S, M = int(mL.group(1)), int(mL.group(2))
        return S, M, 0, 0, 0
    raise ValueError(f"model_id {model_id!r} matches neither M[01]{{5}} nor M[01]{{2}}L")


def _parse_run_id(run_id: str) -> tuple[str, int, int, int, int, int, int]:
    """'M11111_s3' -> (model_id='M11111', S=1, M=1, P=1, T=1, D=1, seed=3).
    Also handles the bio-plausible-ablation-battery / identity-catch /
    performance-matched-baseline run_id suffix convention ('M11111_pbwm_s0',
    'M00000_idcatch_s0', §4.4/§9.4a/§4.1): `model_id` keeps the full
    suffixed string (so parquet filenames stay unambiguous) while
    S/M/P/T/D still come from the leading 5 (or 2+L) characters via
    `_parse_model_id`."""
    model_part, seed_part = run_id.split("_s")
    try:
        S, M, P, T, D = _parse_model_id(model_part)
    except ValueError:
        arch = _arch_from_manifest(run_id)
        if arch is None:
            raise
        S, M, P, T, D = arch
    return model_part, S, M, P, T, D, int(seed_part)


def _arch_from_manifest(run_id: str) -> Optional[tuple[int, int, int, int, int]]:
    """S/M/P/T/D as RECORDED for a run whose run_id does not encode them --
    the pilot and diagnostic naming (`FLATGRU_SUP_s0`, `HIERGRU_RL_s0`,
    `VANFLAT_INIT100_SUP_s0`). Returns `None` if there is no such row, so
    `_parse_model_id`'s original error still surfaces for a genuine typo.

    The manifest row is the authority on what actually trained. This project
    has twice had a conclusion invalidated by inferring a run's architecture
    instead of reading the recorded one (F1, and the 2026-08-01 supervision
    finding); a run_id is a filename, not a record."""
    manifest = RESULTS / "manifest.jsonl"
    if not manifest.exists():
        return None
    import json

    found = None
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("run_id") == run_id and all(k in rec for k in "SMPTD"):
            found = tuple(int(rec[k]) for k in "SMPTD")  # last row wins, as elsewhere
    return found


def _run_id_extras(model_id: str) -> tuple[bool, float, int]:
    """(pbwm_gate, identity_catch_fraction, flat_units_mult) implied by the
    run_id suffix convention above -- the S/M/P/T/D bits alone don't
    distinguish M11111_pbwm or M11111_idcatch/M00000_idcatch from the plain
    Core cell, but `_build_model`/`Heads` need to match the checkpoint's
    actual trained architecture or `load_state_dict` fails on a shape/key
    mismatch. `flat_units_mult` (performance-matched baseline family,
    §4.1: M00000_2x): the S=0 core's hidden width was doubled at train time,
    which changes every `weight_ih`/`weight_hh` shape -- `_l1`/`_dropout`
    baselines don't change any parameter shape (pure training-time
    regularizers), so they need no entry here."""
    return (
        model_id.endswith("_pbwm"),
        0.12 if model_id.endswith("_idcatch") else 0.0,
        2 if model_id.endswith("_2x") else 1,
    )


def activity_log_path(run_id: str, checkpoint_name: str = "ckpt.pt", out_dir: Optional[Path] = None) -> Path:
    """Where one run's activity log lives, for a given checkpoint (D33).

    `ckpt.pt` (Gate B, `max_steps`) keeps the original unsuffixed filename,
    so nothing that already reads these logs changes. Any other checkpoint
    -- in practice `ckpt_at_criterion.pt`, the Gate A snapshot -- gets its
    own file: the two logs describe the same network at different training
    durations, which is the whole point of §12.4's equal-performance vs
    equal-duration comparison, and a Gate A log overwriting the Gate B log
    is worse than not having one. Mirrors `scripts/run_geometry.py::
    out_csv_for`, which solved the same problem for the topology CSV."""
    out_dir = out_dir or ACTIVITY_LOGS
    suffix = "" if checkpoint_name == "ckpt.pt" else "_" + Path(checkpoint_name).stem.removeprefix("ckpt_")
    return out_dir / f"{run_id}{suffix}.parquet"


def _load_checkpoint(front_end, core, heads, run_id: str, device, checkpoint_name: str = "ckpt.pt") -> int:
    ckpt_path = RESULTS / "checkpoints" / run_id / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint for {run_id} at {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    front_end.load_state_dict(ck["front_end"])
    core.load_state_dict(ck["core"])
    heads.load_state_dict(ck["heads"])
    return ck["step"]


def _stimulus_features_for_session(session_id: str) -> Optional[dict]:
    """Loads the cached (PicID -> feature) map for one session, if present."""
    cache_dir = get_path("feature_cache") / "dataset_stimuli"
    path = cache_dir / f"{session_id}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {str(pid): feat for pid, feat in zip(data["pic_ids"], data["features"])}


def replay_session(
    front_end, core, heads, reflective_gate, S: int, M: int, P: int,
    session_trials, pic_to_feature: dict, cfg: dict, device, run_id: str, model_id: str, seed: int,
    session_id: str, writer: ParquetLogWriter, t0: int,
    shuffled_R_by_trial: Optional[dict] = None,
) -> int:
    """Replays every trial of one session through the frozen model, logging
    each tick. Returns the next global tick counter (`t`)."""
    m = cfg["model"]
    t_cfg = cfg["task"]
    feature_dim = m["feature_dim"]
    action_dim = m["action_dim"]
    gate_width = _gate_width(S, m)
    t = t0

    for trial_idx, row in enumerate(session_trials.itertuples()):
        held_items = [int(x) for x in row.held_items if int(x) != 0]
        probe_item = int(row.probe_item)
        load = len(held_items) or int(row.load)
        in_set = bool(row.probe_in_set)
        correct = bool(row.correct)

        def _feat(pic_id: int) -> Optional[np.ndarray]:
            return pic_to_feature.get(str(pic_id))

        if any(_feat(pid) is None for pid in held_items) or _feat(probe_item) is None:
            continue  # incomplete stimulus cache coverage for this trial; skip rather than fabricate

        state = _init_state(core, S, P, 1, device)
        R_prev = reflective_gate.init_state(1, device) if reflective_gate is not None else None
        # Live per-tick surprise stream: tracked exactly as
        # `training/train.py::_run_trial`'s eval path does, reset per trial.
        prev_policy = torch.full((1, action_dim), 1.0 / action_dim, device=device)
        prev_value = torch.zeros(1, 1, device=device)
        prev_action_logp = torch.log(prev_policy[:, 0].clamp_min(1e-8)).unsqueeze(-1)
        prev_is_feedback = torch.zeros(1, 1, device=device)
        prev_reward = torch.zeros(1, 1, device=device)

        schedule: list[tuple[str, Optional[np.ndarray], int]] = []
        schedule += [("fixation", None, 0)] * t_cfg["fixation_steps"]
        for item_num, pid in enumerate(held_items, start=1):
            schedule += [("encode", _feat(pid), item_num)] * t_cfg["encode_steps"]
        schedule += [("maintain", None, 0)] * t_cfg["maintain_steps"]
        schedule += [("probe", _feat(probe_item), 0)] * t_cfg["probe_steps"]
        schedule += [("feedback", None, 0)] * t_cfg.get("feedback_steps", 1)
        schedule += [("iti", None, 0)] * t_cfg.get("iti_steps", 1)

        for i, (epoch, feat, encoded_count) in enumerate(schedule):
            v_t = (
                torch.as_tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
                if feat is not None
                else torch.zeros(1, feature_dim, device=device)
            )
            c_t = torch.tensor([context_vector(epoch, encoded_count=encoded_count)], dtype=torch.float32, device=device)

            gate_bias = None
            delta_t = None
            if reflective_gate is not None:
                delta_t = reflective_gate.surprise(prev_is_feedback, prev_reward, prev_value, prev_action_logp)
                if shuffled_R_by_trial is not None and trial_idx in shuffled_R_by_trial:
                    # Reflection-shuffle causal control (H2):
                    # bypass the natural surprise->R_t chain and inject this
                    # trial's OWN R_t sequence but time-shuffled within the
                    # trial (temporal alignment with epoch/load/lure
                    # destroyed, marginal distribution preserved).
                    R_t = shuffled_R_by_trial[trial_idx][i].to(device).view(1, 1)
                else:
                    R_t = reflective_gate.step(delta_t, R_prev)
                gate_bias = reflective_gate.gate_bias(R_t).expand(1, gate_width)
                R_prev = R_t

            with torch.no_grad():
                z_t = front_end(v_t, c_t)
                h_star, state, u_t = _step_core(core, S, M, P, z_t, state, t=i, gate_bias=gate_bias, R_t=R_prev)
                policy, value, logits = heads(h_star)

            action = int(torch.argmax(policy, dim=-1).item())

            prev_policy = policy
            prev_value = value.unsqueeze(-1)
            prev_action_logp = torch.log(policy[:, action].clamp_min(1e-8)).unsqueeze(-1)
            prev_is_feedback = torch.full((1, 1), 1.0 if epoch == "feedback" else 0.0, device=device)
            if epoch == "feedback":
                # the trial's real recorded outcome -- known in advance
                # here (this is a frozen replay of a completed trial, not a
                # live rollout), used as the feedback-tick reward exactly as
                # `_run_trial` uses the trial's actual correctness.
                prev_reward = torch.full((1, 1), 1.0 if correct else 0.0, device=device)

            rec = LogRecord(
                run_id=run_id, model_id=model_id, seed=seed, t=t, trial_id=trial_idx, session=session_id,
                epoch=epoch, load=load, held_items=held_items, held_categories=[""] * len(held_items),
                probe_item=probe_item, probe_category="", in_set=in_set,
                action=action, correct=correct,  # real, recorded outcome; a fixed trial property, not masked here
                h_flat=state["h"][0].tolist() if S == 0 else None,
                h_worker=state["h_worker"][0].tolist() if S == 1 else None,
                h_manager=state["h_manager"][0].tolist() if S == 1 else None,
                delta=float(delta_t[0, 0].item()) if reflective_gate is not None else 0.0,
                reflection_R=float(R_prev[0, 0].item()) if R_prev is not None else 0.0,
                value=float(value.item()), policy=policy[0].tolist(),
                readout_pi=logits[0].tolist(), readout_v=[float(value.item())],
            )
            writer.write(rec)
            t += 1
    return t


def generate_activity_log(
    run_id: str, dandi_data, out_dir: Optional[Path] = None, checkpoint_name: str = "ckpt.pt",
) -> Path:
    """Generates (or overwrites) the activity log for one completed run,
    replaying every session in `dandi_data` for which cached stimulus
    features exist. Returns the output Parquet path.

    `checkpoint_name` (D33): which snapshot to replay. Every activity-derived
    DV in the study -- RSA, alignment, decoding, cross-temporal, dPCA,
    persistence, dynamics -- is computed from this log, so before this
    parameter existed all of them were Gate-B-only and the retained Gate A
    checkpoint reached only `run_geometry.py`'s six weight-derived topology
    metrics. Output filename comes from `activity_log_path`."""
    cfg = _load_full_config()
    model_id, S, M, P, T, D, seed = _parse_run_id(run_id)  # noqa: F841 -- T/D never affect the frozen forward pass
    device = torch.device("cpu")  # replay is cheap (forward-only, no batching benefit from GPU here)

    from brainalign_wm.mechanisms.reflective_gate import ReflectiveGate

    pbwm_gate, identity_catch_fraction, flat_units_mult = _run_id_extras(model_id)
    if identity_catch_fraction:
        cfg = {**cfg, "task": {**cfg["task"], "identity_catch_fraction": identity_catch_fraction}}
    if flat_units_mult != 1:
        cfg = {**cfg, "model": {**cfg["model"], "flat_units": cfg["model"]["flat_units"] * flat_units_mult}}
    front_end, core, heads = _build_model(cfg, S, M, P, device, pbwm_gate=pbwm_gate)
    _load_checkpoint(front_end, core, heads, run_id, device, checkpoint_name=checkpoint_name)
    front_end.eval()
    core.eval()
    heads.eval()
    reflective_gate = ReflectiveGate(cfg["mechanisms"]["reflection_lambda"], cfg["mechanisms"]["reflection_beta"]) if M else None

    out_path = activity_log_path(run_id, checkpoint_name, out_dir)
    trials = dandi_data.trials()

    t = 0
    with ParquetLogWriter(out_path) as writer:
        for session_id in dandi_data.sessions():
            pic_to_feature = _stimulus_features_for_session(session_id)
            if pic_to_feature is None:
                continue
            session_trials = trials[trials.session == session_id]
            if len(session_trials) == 0:
                continue
            t = replay_session(
                front_end, core, heads, reflective_gate, S, M, P, session_trials,
                pic_to_feature, cfg, device, run_id, model_id, seed, session_id, writer, t,
            )
    return out_path


def generate_chance_activity_log(model_id: str, seed: int, dandi_data, out_dir: Optional[Path] = None) -> Path:
    """Chance-model negative control (H5 acceptance gate): an
    UNTRAINED (randomly initialized, never optimized) model of the given
    architecture, replayed through the exact same real-session pipeline as
    a trained checkpoint. Wired into `analysis/run_all.py`'s standard
    output (not a one-off script) so every alignment run reports whether
    the metric actually discriminates a trained representation from a
    random one -- if a chance model's normalized alignment is NOT clearly
    below trained cells', the alignment DV is still degenerate."""
    cfg = _load_full_config()
    try:
        S, M, P, T, D = _parse_model_id(model_id)  # noqa: F841 -- T/D never affect the untrained forward pass either
    except ValueError:
        # Same manifest fallback `_parse_run_id` uses, for the pilot/diagnostic
        # naming (`FLATGRU_SUP`). Without it this raised and took the whole
        # H5 acceptance gate down with it -- the one check that says whether
        # the alignment DV discriminates a trained model from a random one.
        arch = _arch_from_manifest(f"{model_id}_s{seed}")
        if arch is None:
            raise
        S, M, P, T, D = arch
    device = torch.device("cpu")

    from brainalign_wm.utils.seeding import seed_everything
    from brainalign_wm.mechanisms.reflective_gate import ReflectiveGate

    seed_everything(seed)
    pbwm_gate, identity_catch_fraction, flat_units_mult = _run_id_extras(model_id)
    if identity_catch_fraction:
        cfg = {**cfg, "task": {**cfg["task"], "identity_catch_fraction": identity_catch_fraction}}
    if flat_units_mult != 1:
        cfg = {**cfg, "model": {**cfg["model"], "flat_units": cfg["model"]["flat_units"] * flat_units_mult}}
    front_end, core, heads = _build_model(cfg, S, M, P, device, pbwm_gate=pbwm_gate)
    front_end.eval()
    core.eval()
    heads.eval()
    reflective_gate = ReflectiveGate(cfg["mechanisms"]["reflection_lambda"], cfg["mechanisms"]["reflection_beta"]) if M else None

    run_id = f"{model_id}_s{seed}_chance"
    out_dir = out_dir or ACTIVITY_LOGS
    out_path = out_dir / f"{run_id}.parquet"
    trials = dandi_data.trials()

    t = 0
    with ParquetLogWriter(out_path) as writer:
        for session_id in dandi_data.sessions():
            pic_to_feature = _stimulus_features_for_session(session_id)
            if pic_to_feature is None:
                continue
            session_trials = trials[trials.session == session_id]
            if len(session_trials) == 0:
                continue
            t = replay_session(
                front_end, core, heads, reflective_gate, S, M, P, session_trials,
                pic_to_feature, cfg, device, run_id, model_id, seed, session_id, writer, t,
            )
    return out_path


def generate_activity_log_reflection_shuffled(
    run_id: str, dandi_data, out_dir: Optional[Path] = None, checkpoint_name: str = "ckpt.pt",
) -> Path:
    """Reflection-shuffle causal control (H2's causal claim):
    re-replays every trial using that SAME trial's own natural R_t sequence
    (read back from the normal activity log, generating it first if
    missing) but time-shuffled WITHIN the trial via
    `mechanisms.reflective_gate.shuffle_reflection` -- destroying R_t's
    temporal alignment with epoch/load/lure while preserving its marginal
    distribution, exactly as the module docstring's causal-control
    machinery is designed for. If H2 holds, alignment computed from this
    log should be markedly lower than from the unshuffled log (see
    `analysis/run_all.py`'s companion lesion-comparison entrypoint)."""
    from brainalign_wm.mechanisms.reflective_gate import ReflectiveGate, shuffle_reflection
    from brainalign_wm.training.logging_schema import read_log

    cfg = _load_full_config()
    model_id, S, M, P, T, D, seed = _parse_run_id(run_id)  # noqa: F841 -- T/D never affect the frozen forward pass
    if not M:
        raise ValueError(f"{run_id} has M=0 (no reflective gate) -- reflection-shuffle lesion is undefined")
    device = torch.device("cpu")

    # The lesion is defined against the SAME checkpoint's unshuffled log --
    # H2's causal claim is within one snapshot, so a Gate A shuffle compared
    # against a Gate B baseline would confound the lesion with duration.
    normal_path = activity_log_path(run_id, checkpoint_name)
    if not normal_path.exists():
        generate_activity_log(run_id, dandi_data, checkpoint_name=checkpoint_name)
    normal_df = read_log(normal_path)

    # hashlib, NOT Python's built-in hash(): hash() on a str is salted per
    # PYTHONHASHSEED, so this seed would differ across process launches --
    # the same non-determinism class `run_grid.py::config_hash` avoids by
    # using sha256 instead of hash().
    import hashlib

    _seed_bytes = hashlib.sha256(f"{run_id}:reflection_shuffle".encode()).digest()[:4]
    gen = torch.Generator().manual_seed(int.from_bytes(_seed_bytes, "big"))
    shuffled_R: dict[tuple, torch.Tensor] = {}
    for (session_id, trial_id), g in normal_df.sort_values("t").groupby(["session", "trial_id"]):
        R_seq = torch.as_tensor(g["reflection_R"].to_numpy(), dtype=torch.float32).view(-1, 1, 1)  # [T,1,1]
        assert R_seq.shape[1] == 1, "replay is batch=1; a batch-major R_seq would silently mis-shuffle across trials"
        shuffled_R[(session_id, trial_id)] = shuffle_reflection(R_seq, generator=gen).view(-1)  # [T]

    pbwm_gate, identity_catch_fraction, flat_units_mult = _run_id_extras(model_id)
    if identity_catch_fraction:
        cfg = {**cfg, "task": {**cfg["task"], "identity_catch_fraction": identity_catch_fraction}}
    if flat_units_mult != 1:
        cfg = {**cfg, "model": {**cfg["model"], "flat_units": cfg["model"]["flat_units"] * flat_units_mult}}
    front_end, core, heads = _build_model(cfg, S, M, P, device, pbwm_gate=pbwm_gate)
    _load_checkpoint(front_end, core, heads, run_id, device, checkpoint_name=checkpoint_name)
    front_end.eval()
    core.eval()
    heads.eval()
    reflective_gate = ReflectiveGate(cfg["mechanisms"]["reflection_lambda"], cfg["mechanisms"]["reflection_beta"])

    out_path = activity_log_path(f"{run_id}__reflection_shuffled", checkpoint_name, out_dir)
    trials = dandi_data.trials()

    t = 0
    with ParquetLogWriter(out_path) as writer:
        for session_id in dandi_data.sessions():
            pic_to_feature = _stimulus_features_for_session(session_id)
            if pic_to_feature is None:
                continue
            session_trials = trials[trials.session == session_id]
            if len(session_trials) == 0:
                continue
            this_session_shuffled = {
                trial_idx: shuffled_R[(session_id, trial_idx)]
                for trial_idx in range(len(session_trials))
                if (session_id, trial_idx) in shuffled_R
            }
            t = replay_session(
                front_end, core, heads, reflective_gate, S, M, P, session_trials,
                pic_to_feature, cfg, device, run_id, model_id, seed, session_id, writer, t,
                shuffled_R_by_trial=this_session_shuffled,
            )
    return out_path
