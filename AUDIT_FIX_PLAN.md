# Audit fix + clean grid re-run — execution plan

Status as of 2026-07-05: **plan approved, execution not yet started.**
Branch `fix/audit-2026-07-04` has been created (empty — no commits on it yet).
`comments.txt` (repo root, untracked) is the senior-review audit this plan
responds to; read it in full before resuming. This file is the resumable
plan + design record for a future session to pick up from cold.

## Origin

The user asked to apply every fix in `comments.txt`, fan out subagents to
check for anything the audit missed, then delete the invalidated
`results/` (12 completed runs spanning 3 different git commits mid-grid —
not analyzable) and re-run one clean 8-hour training grid. `comments.txt`
found the current results scientifically invalid for two independent
blocker reasons (see its Section A): the L=1 arm is confounded with
supervision density and sits at chance (A1), and the alignment metric
itself is degenerate — raw alignment exceeds the noise ceiling for 5/6
cells and normalized alignment clips to 1.0 for everything including a
chance model (A2). Plus reproducibility bugs (B) and unwired analysis code
(C).

## Scope decisions already made with the user (do not re-litigate)

1. **Load one-hot stays as a model input.** `tasks/sternberg.py::context_vector`
   keeps one-hot-encoding load into `c_t` indices 2-4. comments.txt's A2a
   "strongly consider removing it" is declined this pass — the *required*
   part of A2a (identity-based RDM conditions instead of load-keyed ones)
   already removes the main statistical shortcut, and removing the model
   input too would add extra learnability risk on top of the A1 training-
   signal rewrite happening in the same pass. **Must add a `RESPONSES`
   section to `comments.txt`** explaining this and every other judgment call
   below (user explicitly asked for this).
2. **H5/H6 get wired in this pass**, not scoped out: `dpca.py`, `encoding.py`,
   `cross_temporal.py`, `persistence.py` all get wired into `run_all.py`,
   plus new figures F4-F6. Only F1 (design schematic) and F8 (Tier-B
   cross-dataset replication, needs its own adapter validation against
   001187) are scoped OUT of this pass — document this explicitly, don't
   silently drop it.

## Key architectural decision: per-session identity RDMs + pooled coarse RDMs

This is the resolution to A2a/A2b/A2c/A2d/A2e/A2f/B3 together, worked out
during planning (not spelled out verbatim in comments.txt, but implied by
combining its individual fixes):

- Real Tier-A PicIDs are **not consistent across sessions/patients**
  (different numbering conventions per `DECISIONS.md`'s M8 notes) — held-item
  identity is only meaningful *within* one session. So the
  **maintenance/encode epoch** (identity + load conditions, per A2a/A2b) is
  computed as a **per-session** crossnobis RDM: model replay restricted to
  that session's trials vs. that session's neural trials, raw alignment
  against that session's own split-half ceiling. This is exactly what
  `rsa.py::noise_ceiling_from_dataset`/`_session_condition_rdm` already
  compute — A2c's "same representation for raw and ceiling" requirement is
  satisfied for free once alignment itself is also computed per-session.
  Aggregate across sessions for the headline number; **keep the per-session
  rows (tagged with patient via `dandi_data.patient_of(session)`) for the
  LME** — this is exactly B3's requested patient factor.
- The **probe epoch** (coarse in_set/lure/correct conditions) genuinely
  generalizes across sessions, so it stays **pooled** — but built correctly
  per A2d: for each unit, compute its condition-mean using ONLY that unit's
  own session's matching trials, then concatenate units at the condition
  level (never average a unit over a trial it did not record — this is what
  `dandi_nwb.py::rates(None, ...)` currently gets wrong, zero-filling
  out-of-session trials instead). The ceiling for this pooled object must be
  computed via repeated random per-condition-per-session half-splits of the
  *same* pooled representation (not the current per-session-RDM-shape-
  matching hack), which fixes A2c/A2e for the pooled path too.

This per-session/pooled split is the shared foundation for H1 (region
dissociation), H2 (causal reflection-shuffle), H3/H4 (main alignment), and
H5/H6.

## A1a design (the training-signal confound fix) — worked out in detail

Uses `tasks/curriculum.py::CurriculumSchedule.params_for()`'s existing
`phase` field (`"warmup"`/`"ramp"`/`"target"`, already computed every step
via `task_gen.curriculum_params(step, total_steps)` — no new plumbing
needed to know the phase).

- **Warmup phase** (first `warmup_frac`≈20% of steps): L=0 keeps today's
  dense per-tick CE exactly as-is (`tick_weight` trick, unchanged). L=1 keeps
  its existing per-trial node-perturbation update timing, but during warmup
  only, the scalar `reward` passed to `apply_update` becomes the *fraction
  of ticks whose greedy action matched the ideal target action*
  (`_target_action`) instead of bare trial-end correct/incorrect — a denser,
  smoother training signal, implemented as a one-line change to how `reward`
  is computed pre-warmup (trace/update timing untouched).
- **Ramp+target phases** (remaining ~80%): L=0 switches to REINFORCE with a
  value baseline, backpropagated through time. L=1 is unchanged (already
  sparse trial-end reward in this regime). This makes L=0 vs L=1 differ
  *only* in credit assignment for the bulk of training, on an identical
  reward signal (resolves A1a's actual requirement).

Concrete REINFORCE loss design for `_run_trial`'s `mode="bptt"` path (add a
`signal: "ce"|"reinforce"` parameter, selected by `phase != "warmup"` for
L=0 only; L=1 never uses this parameter — its own local update owns the
warmup/non-warmup reward-shaping switch separately):
- At every **probe-epoch** tick, collect `(logp_a, value_pred)` where
  `logp_a = torch.log(policy[0, action].clamp_min(1e-8))` for the tick's
  **sampled** action (`action` is already sampled via `torch.multinomial`
  in the existing bptt path) — keep the computation graph (don't detach
  `policy`/`value` here, unlike the `prev_*` bookkeeping which stays
  detached as today).
- At the **feedback** tick, additionally collect `value_pred` alone (for the
  value loss only — the feedback-epoch's target action is trivial fixation,
  not behaviorally meaningful, so no policy-gradient term there).
- After the trial loop (reward now known): for each collected probe-tick
  term, `advantage = (reward_tensor - value_pred.detach())`;
  `policy_loss += -logp_a * advantage`; `value_loss += F.mse_loss(value_pred,
  reward_tensor)` (summed over probe ticks + the feedback tick's value_pred).
  Add a small entropy bonus (`entropy_coef * mean_entropy`, subtracted from
  the loss to encourage exploration — REINFORCE without one tends to
  collapse prematurely; add `train.entropy_coef` to `config.yaml`, e.g.
  `0.01`). Final `total_loss = policy_loss/n_probe_ticks +
  value_weight*value_loss/n_value_terms - entropy_coef*mean_entropy`.
- Document in `DECISIONS.md` + fill the H3 caveat into `preregistration.md`
  ("H3 only defined for cells that clear the behavioral gate").

**A1b** (rung-2 adaptive-baseline no-op): `local_learning.py`'s
`adaptive_baseline` branch is currently byte-identical to `else`. Give it a
real, distinct effect — e.g. a faster EMA decay constant when
`adaptive_baseline=True` so it actually reduces reward variance differently
from rung 1.

**A1c** (batching) — de-risked by exploration: the model/mechanism code
(`HRLCore`, `MaskedGRUCell`, `FlatGRUCore`, `Heads`, `ReflectiveGate`,
`NodePerturbationLearner.accumulate`'s `einsum("bi,bj->ij", ...)/batch`)
is **already written batch-generically** — only `train.py`'s
`_run_trial`/`_run_trial_local`/`evaluate_accuracy`/`_image_feature` and the
task-generation call site are hardcoded to batch=1. Batching plan:
- Since trial tick-length is fully determined by `load` (all other draws —
  held items, in_set, lure — vary content, not tick count), draw **one
  shared load per batch** from the curriculum's `load_weights`, then
  generate `B` independent trials at that load (each with its own RNG derived
  from `SeedSequence([seed, step, b])`). No padding needed — add a
  `TaskGenerator.sample_batch(step_idx, total_steps, batch_size, split)` 
  method in `tasks/generator.py` implementing this.
- Rewrite `_run_trial`/`_run_trial_local`/`evaluate_accuracy` to accept
  `list[list[TrialStep]]` (B trials, all same length T): tick loop stays
  over `i in range(T)`, but stacks `v_t`/`c_t`/targets over the batch dim at
  each tick (since `in_set`/`lure_flag` differ per trial even at fixed
  load, `c_t` and `_target_action` must be computed **per-b**, not shared).
  `_image_feature` needs a batched variant (stack per-b image features).
- Per-b bookkeeping (last_probe_action, true_in_set, correct, reward) needs
  arrays instead of scalars.
- Add `train.batch_size` to `config.yaml`.

**B5/B6** ride along: document that action-sampling is now load-bearing for
L=0 (REINFORCE) in `train.py`'s docstring. Fix the heads' 2x-effective-LR
quirk: `NodePerturbationLearner` currently gets `weight_hh=weight_ih=heads.pi.weight`
(same param twice) for `Heads.pi`/`Heads.value`, so `apply_update` adds the
update twice to the same tensor since `presyn=h_star` is identical for both
"slots". Add a single-weight mode (track one `TracedWeight`, not two) for
plain `Linear` heads instead of leaving the duplicated-slot quirk live.

## Remaining fixes (unchanged from comments.txt, see it for line-level detail)

- **A2** full rewrite: `neural/dataset_contract.py` (region-family filtering
  via existing `region_family()` helper), `neural/adapters/dandi_nwb.py`
  (fix `rates(None,...)`/`response_patterns` cross-session zero-fill;
  region-family filtering), `analysis/rsa.py` (generalize
  `_session_condition_rdm` to take a condition-labeling function; rewrite
  `noise_ceiling_from_dataset` to intersect by condition identity, delete
  dead `common_conds`/`valid` = B4; add pooled split-half ceiling variant),
  `analysis/run_all.py` (rewrite around per-session + pooled split; raise
  min-shared-conditions floor with loud failure = A2f; emit per-cell
  headline CSV + long per-session CSV for the LME), 
  `training/generate_activity_logs.py` (live per-tick `delta_t` from the
  replay's own policy/value/feedback instead of `torch.zeros` stub = A2g;
  carry real held-item identity through to logs — category stays absent for
  real data, not in the NWB schema), `analysis/stats.py` (patient variance
  component via statsmodels `vc_formula` when a `patient` column is present
  = B3; document VIF check for accuracy-vs-L collinearity).
- **B1**: `run_grid.py` — replace salted `hash()` fingerprint with
  `hashlib.sha256` over the full resolved config; dump
  `results/resolved_config.yaml`.
- **B2**: gate boundary `>=` in `train.py` + `preregistration.md`.
- **C1**: reflection-shuffle causal control — regenerate M=1 activity logs
  with `mechanisms.reflective_gate.shuffle_reflection` applied to replay's
  `R_t`; recompute alignment; compare; new figure panel (F4).
- **C2**: region-family (MTL/MFC) alignment wired into `run_all.py` + LME
  region factor.
- **C3/C4** (in scope per user): wire `dpca.py`/`encoding.py`/
  `cross_temporal.py`/`persistence.py` into `run_all.py` —
  H5 = `cross_temporal_decoding` + `stability_index` on held-item identity
  across maintenance timebins (model: per-tick hidden states already
  logged; neural: time-resolved bins already available via `rates`),
  compared model-vs-brain via `stats.compare_distributions`; H6 =
  `persistent_activity_index`/`load_tuning`/`selectivity_anova` per unit,
  compared via `compare_distributions`; encoding R² (ridge, ceiling-
  normalized) on the shared condition set as a complementary column.
  Figures F4 (reflection-shuffle), F5 (persistence/selectivity, H6), F6
  (stability-index dynamic/oblique, H5) added to `figures/make_all.py`.

## Section D — delete before re-run (only after everything above is committed)

Delete: `results/manifest.jsonl`, `results/checkpoints/`,
`results/activity_logs/`, `results/alignment_results.csv`,
`results/figures/`, `RUN_REPORT.md`. Keep `results/.gitkeep`.

**Keep** (not invalidated, expensive to rebuild): `results/cifar100_raw/`,
`stimuli/`, `results/feat_cache/` — no change to `feature_dim`/encoder/
`bottleneck`/category set/stimulus pool this pass (the load-one-hot-input
decision above keeps `context_vector`'s `C_DIM` untouched, so this
exception doesn't trigger).

## Validation before launching the real grid

1. Re-run `neural/sim_brain/recovery_gate.py` after the RSA/pseudopopulation
   rewrite — must still PASS.
2. `pytest -q` full suite + new tests: pseudopopulation has no structural
   zeros, epoch-appropriate condition schema, reflection-shuffle lesion,
   per-session ceiling/raw same-representation, REINFORCE/batching smoke
   test.
3. Smoke-tier run of all 8 cells x 1 seed; check comments.txt Section H
   acceptance gates (both L arms get a real shot; normalized_alignment not
   identically 1.0; chance control scores below trained; raw ≤ ceiling;
   n_shared_conditions comfortably above floor).
4. **Independent adversarial review**: fan out fresh (non-fork) subagents to
   review the full diff for correctness bugs beyond what comments.txt
   caught, especially in the REINFORCE/batching rewrite and the
   pseudopopulation/ceiling rewrite, before committing to the 8h run.
5. Append the `RESPONSES` section to `comments.txt` (see scope-decisions
   section above).

## Freeze, delete, launch

1. Commit everything on `fix/audit-2026-07-04`; fill `preregistration.md`
   commit hash + date; record in `DECISIONS.md`.
2. Execute the Section D deletions.
3. Benchmark batched wall-clock on the smoke tier; pick a tier/seed count
   that fits the user's 8h budget with margin left for `run_all.py` +
   figures afterward (prefer completing the full 8-cell factorial at >=1
   seed cleanly per comments.txt E3; scale seeds up if batching gives
   headroom).
4. Launch the grid in the background; do not commit code until it
   finishes; monitor via scheduled check-ins, not tight polling.
5. On completion: run `analysis/run_all.py`, `figures/make_all.py`, verify
   acceptance gates H1-H5 on the real grid, report final results to the
   user.

## Task checklist (mirror of this session's TaskCreate list, for a fresh session to recreate)

- [x] 0. Branch created: `fix/audit-2026-07-04` (done, no commits yet)
- [ ] 1. A1a: matched sparse-reward training signal (warmup + REINFORCE)
- [ ] 2. A1b: fix rung-2 adaptive-baseline no-op
- [ ] 3. A1c: batch trials
- [ ] 4. B5/B6: document REINFORCE sampling policy; fix heads 2x-LR quirk
- [ ] 5. A2 foundation: dataset_contract region-family + dandi_nwb pseudopopulation fix
- [ ] 6. A2: rewrite rsa.py (per-session identity RDMs, pooled coarse RDMs, real ceilings)
- [ ] 7. A2: rewrite run_all.py around per-session + pooled architecture
- [ ] 8. A2g: live reflection stream in generate_activity_logs.py replay
- [ ] 9. B3: patient variance component in stats.py
- [ ] 10. B1: real sha256 config hash + resolved_config.yaml dump
- [ ] 11. B2: gate boundary >= consistently
- [ ] 12. C1: reflection-shuffle causal control wired in
- [ ] 13. C2: region dissociation (MTL/MFC) wired into run_all.py + LME
- [ ] 14. C3/C4: wire H5/H6 analyses, encoding R2, figures F4-F6
- [ ] 15. Validate: recovery gate + pytest + new tests
- [ ] 16. Validate: smoke-tier 8-cell run + acceptance gates H1-H5
- [ ] 17. Independent adversarial review (fan out subagents)
- [ ] 18. Append RESPONSES section to comments.txt
- [ ] 19. Freeze: commit, fill preregistration.md + DECISIONS.md
- [ ] 20. Delete invalidated results (Section D)
- [ ] 21. Benchmark batched wall-clock; pick tier/seed count for 8h budget
- [ ] 22. Launch + monitor the 8h grid run
- [ ] 23. Post-grid analysis: run_all.py, figures, final acceptance gate check + report

## Files read/understood already (don't need to re-read from scratch)

`training/train.py`, `mechanisms/local_learning.py`, `analysis/run_all.py`,
`analysis/rsa.py`, `analysis/rdm.py`, `neural/adapters/dandi_nwb.py`,
`tasks/sternberg.py`, `tasks/generator.py`, `tasks/curriculum.py`,
`tasks/image_token_bank.py`, `training/generate_activity_logs.py`,
`run_grid.py`, `analysis/stats.py`, `mechanisms/reflective_gate.py`,
`neural/dataset_contract.py`, `neural/sim_brain/spiking_generator.py`,
`neural/sim_brain/recovery_gate.py`, `models/hrl.py`, `models/gru_cell.py`,
`models/heads.py`, `models/flat_gru.py`, `models/front_end.py`,
`training/logging_schema.py`, `figures/make_all.py`, `analysis/dpca.py`,
`analysis/encoding.py`, `analysis/cross_temporal.py`,
`analysis/persistence.py`, `configs/config.yaml`, `DECISIONS.md`,
`preregistration.md`, `utils/device.py` (GPU: RTX 5070 Ti Laptop, sm_120,
confirmed idle at plan time).
