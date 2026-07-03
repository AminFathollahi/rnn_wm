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
`training.logging_schema.LogRecord`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from brainalign_wm.training.logging_schema import LogRecord, ParquetLogWriter
from brainalign_wm.training.train import ROOT, _build_model, _gate_width, _load_full_config, _step_core
from brainalign_wm.tasks.sternberg import context_vector


def _parse_run_id(run_id: str) -> tuple[str, int, int, int, int]:
    """'M101_s3' -> (model_id='M101', S=1, M=0, L=1, seed=3)."""
    model_part, seed_part = run_id.split("_s")
    S, M, L = int(model_part[1]), int(model_part[2]), int(model_part[3])
    return model_part, S, M, L, int(seed_part)


def _load_checkpoint(front_end, core, heads, run_id: str, device) -> int:
    ckpt_path = ROOT / "results" / "checkpoints" / run_id / "ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint for {run_id} at {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    front_end.load_state_dict(ck["front_end"])
    core.load_state_dict(ck["core"])
    heads.load_state_dict(ck["heads"])
    return ck["step"]


def _stimulus_features_for_session(session_id: str) -> Optional[dict]:
    """Loads the cached (PicID -> feature) map for one session, if present."""
    cache_dir = ROOT / "results" / "feat_cache" / "dataset_stimuli"
    path = cache_dir / f"{session_id}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {str(pid): feat for pid, feat in zip(data["pic_ids"], data["features"])}


def replay_session(
    front_end, core, heads, reflective_gate, S: int, M: int,
    session_trials, pic_to_feature: dict, cfg: dict, device, run_id: str, model_id: str, seed: int,
    writer: ParquetLogWriter, t0: int,
) -> int:
    """Replays every trial of one session through the frozen model, logging
    each tick. Returns the next global tick counter (`t`)."""
    m = cfg["model"]
    t_cfg = cfg["task"]
    feature_dim = m["feature_dim"]
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

        state = core.init_state(1, device) if S == 1 else {"h": core.init_state(1, device)}
        R_prev = reflective_gate.init_state(1, device) if reflective_gate is not None else None

        schedule: list[tuple[str, Optional[np.ndarray]]] = []
        schedule += [("fixation", None)] * t_cfg["fixation_steps"]
        for pid in held_items:
            schedule += [("encode", _feat(pid))] * t_cfg["encode_steps"]
        schedule += [("maintain", None)] * t_cfg["maintain_steps"]
        schedule += [("probe", _feat(probe_item))] * t_cfg["probe_steps"]
        schedule += [("feedback", None)] * t_cfg.get("feedback_steps", 1)
        schedule += [("iti", None)] * t_cfg.get("iti_steps", 1)

        for i, (epoch, feat) in enumerate(schedule):
            v_t = (
                torch.as_tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
                if feat is not None
                else torch.zeros(1, feature_dim, device=device)
            )
            c_t = torch.tensor([context_vector(load, epoch, lure_flag=False)], dtype=torch.float32, device=device)

            gate_bias = None
            if reflective_gate is not None:
                delta_t = torch.zeros(1, 1, device=device)  # replay is a single deterministic pass; no online RPE stream
                R_t = reflective_gate.step(delta_t, R_prev)
                gate_bias = reflective_gate.gate_bias(R_t).expand(1, _gate_width(S, m))
                R_prev = R_t

            with torch.no_grad():
                z_t = front_end(v_t, c_t)
                h_star, state, u_t = _step_core(core, S, M, z_t, state, t=i, gate_bias=gate_bias)
                policy, value, logits = heads(h_star)

            action = int(torch.argmax(policy, dim=-1).item())
            rec = LogRecord(
                run_id=run_id, model_id=model_id, seed=seed, t=t, trial_id=trial_idx,
                epoch=epoch, load=load, held_items=held_items, held_categories=[""] * len(held_items),
                probe_item=probe_item, probe_category="", in_set=in_set,
                action=action, correct=correct,  # real, recorded outcome; a fixed trial property, not masked here
                h_flat=state["h"][0].tolist() if S == 0 else None,
                h_worker=state["h_worker"][0].tolist() if S == 1 else None,
                h_manager=state["h_manager"][0].tolist() if S == 1 else None,
                delta=0.0, reflection_R=float(R_prev[0, 0].item()) if R_prev is not None else 0.0,
                value=float(value.item()), policy=policy[0].tolist(),
                readout_pi=logits[0].tolist(), readout_v=[float(value.item())],
            )
            writer.write(rec)
            t += 1
    return t


def generate_activity_log(run_id: str, dandi_data, out_dir: Optional[Path] = None) -> Path:
    """Generates (or overwrites) the activity log for one completed run,
    replaying every session in `dandi_data` for which cached stimulus
    features exist. Returns the output Parquet path."""
    cfg = _load_full_config()
    model_id, S, M, L, seed = _parse_run_id(run_id)
    device = torch.device("cpu")  # replay is cheap (forward-only, no batching benefit from GPU here)

    from brainalign_wm.mechanisms.reflective_gate import ReflectiveGate

    front_end, core, heads = _build_model(cfg, S, M, device)
    _load_checkpoint(front_end, core, heads, run_id, device)
    front_end.eval()
    core.eval()
    heads.eval()
    reflective_gate = ReflectiveGate(cfg["mechanisms"]["reflection_lambda"], cfg["mechanisms"]["reflection_beta"]) if M else None

    out_dir = out_dir or (ROOT / "results" / "activity_logs")
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
                front_end, core, heads, reflective_gate, S, M, session_trials,
                pic_to_feature, cfg, device, run_id, model_id, seed, writer, t,
            )
    return out_path
