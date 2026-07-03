"""TaskGenerator: the top-level interface composing an `ImageTokenBank`, a
`SternbergGenerator`, and a `CurriculumSchedule` into a single trial source.

    gen = TaskGenerator(config, image_bank, seed=0)
    steps = gen.sample_trial(step_idx, total_steps)  # -> list[TrialStep]

Each `TrialStep` (see `sternberg.py`) carries everything needed to build both
the model input (image_id/features + c_t) and a
`brainalign_wm.training.logging_schema.LogRecord` (minus the trainer-only
fields: run_id, model_id, seed, t, action, correct, h_*, delta, reflection_R,
value, policy, readout_*).
"""
from __future__ import annotations

import numpy as np

from brainalign_wm.tasks.curriculum import CurriculumSchedule
from brainalign_wm.tasks.image_token_bank import ImageTokenBank
from brainalign_wm.tasks.sternberg import SternbergGenerator, TrialStep


class TaskGenerator:
    def __init__(self, config: dict, image_bank: ImageTokenBank, seed: int = 0):
        self.cfg = config
        self.bank = image_bank
        self.seed = seed
        self.sternberg = SternbergGenerator(config, image_bank)
        self.curriculum = CurriculumSchedule.from_config(config)
        self._trial_counter = 0

    def sample_trial(self, step_idx: int, total_steps: int, split: str = "train") -> list[TrialStep]:
        """Deterministic given (seed, step_idx): a fresh RandomState is derived
        per call so trial content depends only on (seed, step_idx), not on call
        order -- lets a trainer re-request/replay a specific step_idx exactly."""
        params = self.curriculum.params_for(step_idx, total_steps)
        seed_state = np.random.SeedSequence([self.seed, step_idx]).generate_state(4)
        rng = np.random.RandomState(seed_state)
        trial_id = step_idx
        return self.sternberg.generate_trial(
            rng=rng,
            loads=params["loads"],
            lure_fraction=params["lure_fraction"],
            maintain_steps=params["maintain_steps"],
            trial_id=trial_id,
            load_weights=params["load_weights"],
            split=split,
        )

    def curriculum_params(self, step_idx: int, total_steps: int) -> dict:
        """The sampling-distribution snapshot for this step (for logging)."""
        return self.curriculum.params_for(step_idx, total_steps)
