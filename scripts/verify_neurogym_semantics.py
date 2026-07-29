#!/usr/bin/env python3
"""Throwaway check (item 9.3 prep): confirms Bandit-v0/DawTwoStep-v0 never
set terminated/truncated -- `new_trial=True` only flags a trial boundary,
the env auto-advances internally. This justifies `ContinuousBatchEnv` in
`run_tiny_rnn_bandit.py` doing one `reset()` then unlimited `step()` calls,
instead of `NeuroGymBatchEnv`'s per-trial `done` latch.
"""
import neurogym as ngym  # noqa: E402 (no brainalign_wm import needed here, kept path-independent)

for env_id, kwargs in [("Bandit-v0", {"n": 2, "p": (0.1, 0.9)}), ("DawTwoStep-v0", {})]:
    env = ngym.make(env_id, **kwargs)
    env.reset(seed=0)
    n_new_trial = 0
    for t in range(5000):
        _obs, _r, terminated, truncated, info = env.step(env.action_space.sample())
        assert not terminated and not truncated, f"{env_id} terminated/truncated at tick {t}"
        n_new_trial += bool(info.get("new_trial", False))
    print(f"{env_id}: 5000 ticks, no termination, {n_new_trial} new_trial flags "
          f"(~{5000 / n_new_trial:.2f} ticks/trial)")
