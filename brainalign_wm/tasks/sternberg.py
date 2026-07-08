"""SternbergGenerator: generates one trial of the image-Sternberg
working-memory task.

Epoch labels match `brainalign_wm.training.logging_schema.VALID_EPOCHS`
exactly: fixation, encode, maintain, probe, feedback, iti. All encoding
items share the single 'encode' epoch label in this model-side schema; a
finer encode1/encode2/encode3 split exists only in the harmonized neural
condition schema built separately by the NWB adapter.

Every trial takes an explicit `numpy.random.RandomState`; there are no bare
`np.random.*` module-level calls anywhere in this file, so trial content is
fully determined by the seed passed in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from brainalign_wm.tasks.image_token_bank import ImageTokenBank

# [WM_family, aux_family, load1, load2, load3, epoch_encode, epoch_maintain, epoch_probe, lure_flag, rule_flag]
C_DIM = 10


@dataclass
class TrialStep:
    trial_id: int
    t_in_trial: int
    epoch: str
    load: int
    held_items: list[int]
    held_categories: list[str]
    probe_item: Optional[int]
    probe_category: Optional[str]
    in_set: Optional[bool]  # populated only during probe/feedback (the "live" response window); None elsewhere
    lure_flag: bool
    image_id: Optional[int]  # the item shown this tick (encode/probe epochs), else None ("no stimulus")
    c_t: list[float]
    is_identity_catch: bool = False  # §9.4a identity-report catch trial
    identity_catch_category: Optional[str] = None  # target category for the aux head, catch trials only


def context_vector(epoch: str, lure_flag: bool, encoded_count: int = 0, aux_family: bool = False) -> list[float]:
    """`encoded_count` is how many items have been shown so far (1-indexed,
    incrementing as each item is presented) -- fires the load1/2/3 one-hot
    only during `encode`. Zero at every other epoch: the network must carry
    load forward in its own recurrent state through maintain/probe rather
    than reading it off a live exogenous broadcast. `aux_family` (§9.4a) is
    set for every tick of an identity-report catch trial -- the trial-
    level task-family cue, analogous to `WM_family` (which stays 1
    regardless; the trial is still fundamentally a WM trial, just with an
    auxiliary identity readout at the query point)."""
    c = [0.0] * C_DIM
    c[0] = 1.0  # WM_family (only family in Core)
    c[1] = 1.0 if aux_family else 0.0  # aux_family: identity-report catch trial (§9.4a)
    if epoch == "encode" and 1 <= encoded_count <= 3:
        c[1 + encoded_count] = 1.0  # load1/2/3 one-hot -> indices 2,3,4
    c[5] = 1.0 if epoch == "encode" else 0.0
    c[6] = 1.0 if epoch == "maintain" else 0.0
    c[7] = 1.0 if epoch == "probe" else 0.0
    c[8] = 1.0 if (epoch == "probe" and lure_flag) else 0.0
    c[9] = 0.0  # rule_flag (reserved, Extended/Stretch)
    return c


class SternbergGenerator:
    def __init__(self, config: dict, image_bank: ImageTokenBank):
        self.cfg = config
        self.bank = image_bank
        t = config["task"]
        self.categories = list(t["categories"])
        self.fixation_steps = int(t["fixation_steps"])
        self.encode_steps = int(t["encode_steps"])
        self.probe_steps = int(t["probe_steps"])
        self.feedback_steps = int(t["feedback_steps"])
        self.iti_steps = int(t["iti_steps"])

    def generate_trial(
        self,
        rng: np.random.RandomState,
        loads: list[int],
        lure_fraction: float,
        maintain_steps: int,
        trial_id: int,
        load_weights: Optional[list[float]] = None,
        split: str = "train",
        identity_catch_fraction: float = 0.0,
    ) -> list[TrialStep]:
        p = None
        if load_weights is not None:
            p = np.asarray(load_weights, dtype=float)
            p = p / p.sum()
        load = int(rng.choice(loads, p=p))

        held_items: list[int] = []
        held_categories: list[str] = []
        exclude: set[int] = set()
        for _ in range(load):
            cat = str(rng.choice(self.categories))
            img_id = self.bank.sample(1, rng, split=split, category=cat, exclude=exclude)[0]
            held_items.append(img_id)
            held_categories.append(cat)
            exclude.add(img_id)

        # Identity-report catch trial (§7.1/§9.4a):
        # replaces the probe step entirely with a query for which category
        # occupied a GIVEN serial position -- fixed to position 1 (the
        # first-encoded item) so no extra per-trial signal is needed to
        # communicate "which position" to the network. Off (0.0) in Core.
        is_catch = bool(rng.random_sample() < identity_catch_fraction)
        identity_catch_category = held_categories[0] if is_catch else None

        # in_set is drawn first (P=0.5); lure_fraction then conditions only the
        # not-in-set branch, i.e. a fraction of not-in-set probes are lures.
        # Catch trials skip the in/out judgment entirely (no probe image).
        in_set = bool(rng.random_sample() < 0.5)
        is_lure = False
        if is_catch:
            probe_item, probe_category = None, None
        elif in_set:
            probe_item = int(rng.choice(held_items))
            probe_category = held_categories[held_items.index(probe_item)]
        else:
            is_lure = bool(rng.random_sample() < lure_fraction)
            if is_lure:
                probe_category = str(rng.choice(held_categories))
            else:
                other_cats = [c for c in self.categories if c not in held_categories] or self.categories
                probe_category = str(rng.choice(other_cats))
            probe_item = self.bank.sample(1, rng, split=split, category=probe_category, exclude=exclude)[0]

        steps: list[TrialStep] = []
        t = 0

        def emit(epoch: str, n: int, image_id: Optional[int], encoded_count: int = 0) -> None:
            nonlocal t
            for _ in range(n):
                steps.append(
                    TrialStep(
                        trial_id=trial_id,
                        t_in_trial=t,
                        epoch=epoch,
                        load=load,
                        held_items=list(held_items),
                        held_categories=list(held_categories),
                        probe_item=probe_item,
                        probe_category=probe_category,
                        in_set=(in_set if epoch in ("probe", "feedback") else None) if not is_catch else None,
                        lure_flag=is_lure,
                        image_id=image_id,
                        c_t=context_vector(epoch, is_lure, encoded_count, aux_family=is_catch),
                        is_identity_catch=is_catch,
                        identity_catch_category=identity_catch_category,
                    )
                )
                t += 1

        emit("fixation", self.fixation_steps, None)
        for item_num, item_id in enumerate(held_items, start=1):
            emit("encode", self.encode_steps, item_id, encoded_count=item_num)
        emit("maintain", maintain_steps, None)
        emit("probe", self.probe_steps, None if is_catch else probe_item)
        emit("feedback", self.feedback_steps, None)
        emit("iti", self.iti_steps, None)
        return steps
