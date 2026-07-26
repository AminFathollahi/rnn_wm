"""NBackGenerator: generates one n-back sequence (Phase 6, comments.txt §5,
Stage 3 prerequisite -- [BASHIVAN24]'s design, minus the location dimension
this repo has no spatial axis for, see §9.2).

6 task variants: n in {1,2,3} x feature in {identity, category}. At each
position `i` in a length-`sequence_length` sequence, the model sees one
stimulus; for `i < n` there is no valid n-back comparison yet (no judgment
scored, ideal action = no-action); for `i >= n` the stimulus either matches
(same image_id under "identity", same category under "category") the item
shown at position `i-n` or doesn't, at a configurable target rate.

Model-internal only (comments.txt §9.1): no brain-alignment DV, so this
generator is NOT wired into training/train.py -- Phase 7 (METARL, task cue
withheld + previous action/reward as inputs) decides how these steps
actually reach the network. No `c_t` field for that reason: forcing a
Sternberg-shaped context vector here would just be guessed-at plumbing for
an interface Phase 7 hasn't defined yet.

Every trial takes an explicit `numpy.random.RandomState`; no bare
`np.random.*` module-level calls, same invariant as `sternberg.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from brainalign_wm.tasks.image_token_bank import ImageTokenBank

Feature = Literal["identity", "category"]


@dataclass
class NBackStep:
    trial_id: int
    t_in_trial: int
    n: int
    feature: Feature
    image_id: int
    category: str
    is_match: Optional[bool]  # None for i < n (no judgment scored), else the ground truth


class NBackGenerator:
    def __init__(self, config: dict, image_bank: ImageTokenBank):
        self.cfg = config
        self.bank = image_bank
        self.categories = list(config["task"]["categories"])

    def generate_trial(
        self,
        rng: np.random.RandomState,
        n: int,
        feature: Feature,
        sequence_length: int,
        match_fraction: float,
        trial_id: int,
        split: str = "train",
    ) -> list[NBackStep]:
        if sequence_length <= n:
            raise ValueError(f"sequence_length={sequence_length} must be > n={n} (no scoreable position otherwise)")

        steps: list[NBackStep] = []
        for i in range(sequence_length):
            if i < n:
                cat = str(rng.choice(self.categories))
                image_id = self.bank.sample(1, rng, split=split, category=cat)[0]
                is_match = None
            else:
                back = steps[i - n]
                is_match = bool(rng.random_sample() < match_fraction)
                if is_match:
                    if feature == "identity":
                        image_id, cat = back.image_id, back.category
                    else:  # category: same category, a DIFFERENT exemplar image
                        cat = back.category
                        image_id = self.bank.sample(1, rng, split=split, category=cat, exclude={back.image_id})[0]
                else:
                    if feature == "identity":
                        cat = str(rng.choice(self.categories))
                        image_id = self.bank.sample(1, rng, split=split, category=cat, exclude={back.image_id})[0]
                    else:  # category: force a DIFFERENT category than back's
                        other_cats = [c for c in self.categories if c != back.category] or self.categories
                        cat = str(rng.choice(other_cats))
                        image_id = self.bank.sample(1, rng, split=split, category=cat)[0]

            steps.append(
                NBackStep(
                    trial_id=trial_id, t_in_trial=i, n=n, feature=feature,
                    image_id=image_id, category=cat, is_match=is_match,
                )
            )
        return steps
