# Decisions log

Append-only. Record any non-obvious choice (protocol §0 rule 7), especially deviations.

## Resolved before build (from protocol §17)
- **Neural target:** human single-neuron ephys (Sternberg), not fMRI.
- **Datasets:** combine all — Tier A pooled (`000469` + `000673`, ~1,800 WM units, MTL+MFC), Tier B `001187` (MTL), Tier C `000574` (verbal, coarse conditions only). Read via h5py.
- **Modality:** visual WM only for Core; verbal = coarse-condition replication (no encoder path).
- **Stimuli:** train on a broad image pool; align using the datasets' exact embedded images.
- **Compute:** single machine, one GPU (RTX 5070 Ti). Grid runs sequentially, breadth-before-depth, resumable overnight.
- **PyTorch:** upgrade to the CUDA-12.8 / sm_120 build (`make setup`); build device-agnostic with CPU fallback.

## During build
- **2026-07-03: broad ImageTokenBank pool = CIFAR-100, grouped into 4 superordinate categories.** §7.3 needs a *broad naturalistic* training pool distinct from the exact dataset stimuli used only for alignment (§7.4). No such pool existed on disk. Chose CIFAR-100 (auto-downloads via torchvision, ~170MB, license-clean) grouped by fine-label into `faces` (people fine classes; no genuine face closeups available -- logged as a limitation), `animals`, `objects`, `places` -- matching the picture-Sternberg category structure (§7.1). Images are 32x32 (upsampled by the ResNet transform); acceptable because this pool trains a *general* WM operation, not the alignment comparison itself (which uses the datasets' embedded `StimulusTemplates`). See `scripts/build_stimuli_pool.py`.
- **2026-07-03: skip env clone, upgrade `wm_dynamics` in place.** §0.2 Step 0.1 offers cloning to `wm_dynamics_gpu` as an "optional safety" since other projects share `wm_dynamics`. Disk check: root filesystem has only 12GB free vs. the env's 8.2GB size — cloning would leave ~3.8GB free, too tight for the torch/torchvision cu128 wheel download (~1-2GB) plus install. Upgrading in place instead (this was always the fallback per §0.2). Risk noted: other conda envs/projects depending on `wm_dynamics`'s torch==2.5.1 will now see torch upgraded; device-agnostic code + CPU fallback (`utils/device.py`) is unaffected either way.
- **2026-07-03: `manager_units` retuned 128 -> 64 for param-budget match (§5.2, M2 gate).** With `worker_units=196` (fixed by the 14x14 grid, §5.2 — a meaningful spatial-resolution choice, not a free capacity knob), `g_dim=128`, `worker_pool_block=2` (-> summary_dim 49), `bottleneck=128` (shared/frozen across cells, §4.1): worker-side params = 266,952; flat core (`flat_units=256`) = 296,448. At `manager_units=128` the HRL core totals 352,200 (+18.8% vs. flat) -- outside the ±5% tolerance (`param_budget_tol`), i.e. HRL would "win" partly via extra capacity, exactly the confound §5.2/§15 warns about. Solved 3·Nm² + 281·Nm + 128 ∈ [flat·0.95 − worker, flat·1.05 − worker] for the manager-side param count (manager GRU cell + top-down projection W_g) -> Nm ∈ [~38, ~83]; chose `manager_units=64` (round number, mid-range) -> HRL core totals 297,352 (+0.3% vs. flat, well inside tolerance). Achieved counts + the matching helper live in `brainalign_wm/models/hrl.py` (`compute_param_counts`), asserted by `tests/test_models.py`.
- **2026-07-03: M5 (sim_brain + analysis toolkit) built directly, not via subagent.** An earlier attempt to delegate M5 to a background fork reported "completed" but silently produced zero files (likely truncated by a session-limit boundary hit mid-run) -- caught by checking `ls` on the target directories rather than trusting the fork's self-report. Rebuilt `neural/sim_brain/spiking_generator.py`, `analysis/{rdm,rsa,dpca,persistence,cross_temporal,encoding,stats}.py`, and `neural/sim_brain/recovery_gate.py` directly. **Recovery gate passes** (protocol §8.3): noise ceiling lower=0.880/upper=0.893 (sane); RSA alignment between two independent same-generative-process draws ("matching planted geometry") = 0.860 of ceiling vs. 0.251 raw for a scrambled control (>0.6 margin); dPCA load- and item-marginalizations both clear the scrambled control by >0.03 (real 0.359/0.169 vs. scrambled 0.025/0.113) once `time` was declared as its own dPCA factor (see next entry); cross-temporal item decoding: 0.491 accuracy (real) vs. 0.195 (scrambled) at chance=0.200.
- **2026-07-03: fixed two real bugs found while getting the recovery gate to a legitimate pass (not weakened to force green).** (1) `crossnobis_rdm`'s condition-label handling used `np.asarray(list_of_tuples, dtype=object)` and `arr == a_tuple`, both of which numpy silently mis-handles for equal-length-tuple labels (builds a 2D array / tries to broadcast the tuple as an axis) -- replaced with pure-Python dict-based grouping (`brainalign_wm/analysis/rdm.py::_group_indices`). (2) the dPCA gate check initially declared only `["load","item"]` as factors while feeding a 4D tensor `[unit, load, item, time]` -- `time` (a real, unaccounted-for axis) never got marginalized out, so time-locked variance leaked equally into the `load` and `item` marginalizations and masked the true per-factor signal (item real≈scrambled, 0.275 vs 0.279). Declaring `time` as an explicit third factor fixed it (item real=0.169 vs scrambled=0.113). Also separately: the per-unit rate model was `base_rate + softplus(sum of all planted-axis terms)` -- a large `load` term could saturate the softplus and crush the `item`/`category` terms' effective contribution (an unwanted nonlinear interaction between two axes meant to be independently recoverable); switched to purely additive-then-clip (`brainalign_wm/neural/sim_brain/spiking_generator.py::_rate_trace`), and increased `w_load`'s tuning-weight scale (2.0 -> 8.0) so the load axis clears the sampling-noise floor at the default trial counts (`sim_brain.trials_per_condition=8`).
- **2026-07-03: M6 (`dandi_nwb` adapter) built directly against the real mounted data, schema re-verified by direct h5py introspection (matches protocol §8.4 exactly for both 000469 and 000673). 5/5 tests pass on real data (contract validation, MTL+MFC regions present, valid conditions, rate-tensor shapes, sane noise ceiling).
- **2026-07-03: M7 (`training/train.py::train_one`) built directly; several real bugs found and fixed via targeted diagnostics, not assumption:**
  1. **Flat model + M=1 (cells M010/M011) had no gate to bias.** `FlatGRUCore` (S=0) uses plain `nn.GRUCell`, which has no hook for the reflective gate's update-gate bias -- §6.1 is written in terms of "the manager," which doesn't exist for a flat model. The reflective signal was being computed but silently dropped, making M010/M011 behaviorally identical to M000/M001 (no crash, just wrong -- would have quietly invalidated part of the factorial). Fixed by adding `_GatedFlatCore` (`training/train.py`): for S=0×M=1, the flat GRU uses the same gate-capable `MaskedGRUCell` as the HRL manager (mask=None, dense) and is biased directly -- the flat model acts as its own gated unit.
  2. **Node-perturbation traced the wrong perturbation shape.** The eligibility-trace weight matrices (`weight_hh`/`weight_ih`) have `3*hidden` output rows (one per GRU gate-preactivation unit: reset/update/candidate), but the original local-learning code perturbed the POST-gate hidden state `h_t` (dim=`hidden`) -- a shape mismatch that crashed immediately on the first L=1 cell tested. Fixed by adding `_perturbed_gru_step` (`training/train.py`), which manually replicates the GRU math and injects the perturbation at the **pre-activation** level (shape `[batch, 3*hidden]`), matching protocol §6.2's actual node-perturbation formulation (perturb each unit's total input drive) rather than the post-gate output. Also split the heads' single perturbation into two independently-shaped learners (`pi`, `value` -- their output dims, 3 vs. 1, previously shared one incompatible perturbation tensor).
  3. **`ImageTokenBank`'s train/test split wasn't stratified by category** -- a single global shuffle could (and, with a small pool, did) leave a category with zero test images, crashing `sample(..., split="test", category=c)`. Fixed to split per-category (`tasks/image_token_bank.py`).
  4. **Cross-entropy loss was dominated by the trivial "predict fixation" target.** ~83% of ticks in a trial (fixation/encode/maintain/feedback/iti) have a constant, trivially-satisfiable target (fixation); only the probe epoch (~17% of ticks) carries the actual match/non-match decision. Averaged uniformly, the network learned to always predict fixation (loss dropped 0.87->0.14 over 300 trials) while probe-epoch accuracy stayed at chance. Fixed by upweighting the probe epoch's cross-entropy 10x relative to other epochs (`tick_weight` in `_run_trial`).
  5. **Diagnosed (not a bug): the delayed-match-to-sample task needs a real training budget.** After fixes 1-4, short diagnostic runs (600-1500 trials) still showed no learning, which looked like a residual bug. Isolated via ablation: (a) trivial single-tick 4-way category classification through the same front-end+GRU+head pipeline learned to 90% within 600 trials (rules out a gradient-flow/plumbing bug); (b) a plain MLP on concatenated (encode, probe) raw ResNet features learned same/different judgment to ~75-80% within 750 trials (rules out "the task is unlearnable from these features"); (c) the actual recurrent pipeline, run longer (3000 trials, no input noise, higher LR) showed a clear late-emerging learning curve (chance through step ~1800, then 0.55->0.69->0.70->0.76 by step 3000) -- learning does happen, it just needs more trials than a quick interactive check allows, consistent with why the protocol's `dev`/`full` tiers budget 20,000/150,000 steps. Also discovered along the way: my first-pass synthetic test stimuli (random-noise placeholder PNGs, used before the real CIFAR-100 pool finished downloading) produced near-degenerate ResNet features (mean pairwise cosine similarity 0.96 vs. 0.73 for real CIFAR images) -- removed from `stimuli/` once the real pool was ready, since they would have been useless (near-indistinguishable) training/eval items.
  6. **First real training-grid results (full tier, first four cells, seed 0):** M000 (flat, BPTT) reaches load1=1.00/load3=0.85 (both gates clear); M010 (HRL, BPTT) reaches load1=0.975/load3=0.875 (both gates clear); M001 and M011 (local learning) reach load1=0.55/0.45, load3=0.40/0.60 (gates not cleared at seed 0 -- consistent with local learning being the "risky arm" the fallback ladder exists for; both escalated to rung 2, still short of gate).

## M8 (post-training alignment analysis)

- **`train_one` does not log per-tick model activity.** The `LogRecord`/Parquet schema exists precisely so representations can be compared to the neural recordings, but logging every training tick would be enormous and not useful (alignment needs the *trained* model's representations, not a training-time trace). Added `training/generate_activity_logs.py`: for a completed checkpoint, replays each real Tier A session's *actual* recorded trials (their loads, held item identities, probe identity) through the frozen model, driving the visual front end with the dataset's own cached embedded stimulus-image features rather than the broad training pool (the "exact-image alignment" scheme, §7.4) -- and logs every tick via the existing schema.
- **Coverage limitation:** only ~50% of real trials replay successfully in the first pass, because a trial is skipped outright if any of its held/probe PicIDs are missing from that session's cached `StimulusTemplates` feature map (rather than fabricating a substitute). Not yet root-caused whether this reflects genuinely absent images for some PicIDs (e.g. a "no image" sentinel akin to the observed `999` value) or a session-specific PicID-numbering convention (some sessions use 3-digit PicIDs, e.g. 501/502, rather than the 1-5 range the protocol document states) that needs its own harmonization rule. Logged as a follow-up rather than silently coerced.
- **`analysis/run_all.py`** computes a crossnobis RDM from per-trial maintenance-epoch activity (model, via the replay above; neural, via `dandi_nwb.rates()`), pooled across all regions (not yet split by MTL/MFC family -- `NeuralDataset.rates`/`.noise_ceiling` filter by a single canonical region, not a family, so family-level aggregation needs a small adapter extension left for later), and reports alignment raw and as a fraction of the noise ceiling. **Real bug found and fixed:** `crossnobis_rdm` silently zero-fills RDM entries for condition pairs with too few trials to split across folds (see `rdm.py`'s `counts>0` fallback), rather than marking them undefined; with several coarse conditions (load x in-set x correct) occurring only once in a session, this injected enough spurious zero-distance entries to make every cell's alignment reduce to exactly 0.0 -- caught only because four different trained models producing bit-identical alignment scores was implausible on its face. Fixed by filtering both the model and neural sides to conditions with at least `MIN_TRIALS_PER_CONDITION=8` trials before computing crossnobis. After the fix, alignment scores are non-degenerate and vary sensibly across cells (0.31-0.94 raw over the first 4 completed cells).
- **Performance limitation, not yet fixed:** `neural/adapters/dandi_nwb.py::rates()` recomputes spike-count histograms with no caching (unlike `sim_brain/spiking_generator.py`, which was optimized the same way earlier this session), making the *full* Tier A pool (~1800 units, thousands of trials) impractically slow for a single `run_all.py` invocation. Bounded today's runs via `--max-sessions-per-dataset`; the same per-(unit,trial) rate-caching fix used in `spiking_generator.py` should be ported here before running the final alignment analysis on the complete dataset.
- **Figures:** `figures/make_all.py` implements F2 (behavior/gates, from `manifest.jsonl`) and F3 (main alignment, from `alignment_results.csv`); both degrade gracefully (skip with a message, not a crash) if their inputs aren't ready yet. F1, F4-F8 are not implemented (Extended-tier analyses: dynamical-systems mechanism, lesions, oblique sweep, cross-dataset replication).

## Audit fix pass (2026-07-05, branch `fix/audit-2026-07-04`)

Responds to `comments.txt`'s senior-review audit, plus a master-protocol
cross-check. Full detail (section-by-section response, every judgment
call, adversarial-review findings, real-data validation, current status)
is consolidated in **`RESPONSES.md`** at the repo root -- not duplicated
here. Frozen at commit `92e059e`; see `RESPONSES.md` for the running list
of follow-up commits and current grid status.

## v5.0 pass (2026-07-06, comments.txt items 1-4)

- **B2 baseline ceiling** (item 1): `rsa.trial_level_split_half_ceiling`
  replaces the condition-level crossnobis ceiling B2 was wrongly normalized
  by; B2's raw score and ceiling now live on the same per-trial Euclidean
  representation.
- **Load one-hot** (item 2): `tasks/sternberg.py::context_vector` now fires
  the load1/2/3 one-hot only during `encode`, incrementing per item shown;
  zero at every other epoch. Isolated retrain + corr(maintenance_raw_
  alignment, accuracy_load1) recomputation reported in `RESPONSES.md` Part 5.
- **Knob P replaces Knob L in Core** (item 3): see `preregistration.md`'s
  2026-07-06 amendment for the full rationale. Implementation: `PlasticGRUCell`
  (`models/gru_cell.py`) adds Hebbian fast weights (`W_eff = W + alpha*hebb`,
  `hebb` decayed/updated per tick, clipped) to the worker (S=1) or flat core
  (S=0); `HRLCore`/`_GatedFlatCore` gain a `plastic` flag; `train.py`'s
  `_build_model`/`_step_core`/`_init_state`/`_run_trial`/`evaluate_accuracy`
  thread a `P` parameter alongside `S`/`M`. `run_grid.py`'s `CELLS` (8, S/M/P
  bits) and new `LOCAL_LEARNING_CELLS` (4, `M**L` model_ids, `--local-learning`
  flag) replace the old single 8-cell S/M/L list; `configs/config.yaml` gained
  `local_learning_cells:` and `mechanisms.hebb_eta_decay/hebb_eta_hebb/hebb_clip`.
  `analysis/run_all.py` now reads `rec["P"]` for the Core grid and explicitly
  skips `M**L` manifest rows (reported via rung/accuracy only, not RSA-aligned).
- **e-prop rung 3** (item 4): `mechanisms/local_learning.py`'s
  `NodePerturbationLearner` gains an `eprop` flag (rung 3); the eligibility
  trace's "perturbation" input is replaced by `training/train.py::
  _eprop_gru_step`'s per-unit pseudo-derivative (the GRU gate nonlinearities'
  own analytic derivative -- GRU gates are differentiable, so no spiking-style
  surrogate is needed), reusing `trace_step`/`apply_update` unchanged. `train_one`
  gained a second escalation check (`train.rung_check_frac_2`, default 0.75) for
  rung 2 -> 3, mirroring the existing rung 1 -> 2 check. All 8 local-learning
  cells (M00L/M01L/M10L/M11L x seeds 0-1) retrained through the full 1->2->3
  escalation and all 8 escalated to rung 3 without clearing the load1>=0.95
  gate -- results table in RESPONSES.md Part 5. Recorded as the genuine
  result per item 4's instruction, not retried further.
- **Neural session count rebenchmarked, cap raised to the full Tier A pool
  (item 8).** The "impractically slow" worry the M8 section above recorded
  was ALREADY fixed by the time this pass started (`dandi_nwb.py::rates()`
  now has the per-`(unit, epoch)` cache the M8 note asked for -- unclear
  which earlier commit added it, but `git log` shows no dedicated commit
  message for it, so this benchmark re-verifies it's real rather than
  trusting the absence of a complaint). Direct timing of
  `analysis.run_all.align_one_run` (fresh activity-log replay + full
  maintenance/probe alignment across all 3 region levels) on the M000_s0
  checkpoint, varying `--max-sessions-per-dataset`:
  | sessions (both Tier A sets) | wall-clock |
  |---|---|
  | 16 (8+8, the old debugging cap) | 79s |
  | 32 (16+16) | 231s |
  | 45 (24+24, one set exhausted) | 375s |
  | 65 (21+44, full Tier A) | 752s (~12.5 min) |
  Scaling is worse than linear (unit x trial pair count grows with both
  factors) but stays entirely tractable as a standalone analysis pass, not
  a training-budget line item: sessions cost analysis wall-clock only
  (`rates()`/crossnobis), never training wall-clock, so raising them doesn't
  compete with the seed-count decision the E3 benchmark made. **Decision:**
  drop `--max-sessions-per-dataset` entirely (the CLI's own default is
  already `None` = full pool; the 16-session figure that shows up in
  `RESPONSES.md`'s prior write-ups came from manually passing `--max-
  sessions-per-dataset 8` during interactive debugging, not from a coded
  cap) for all analysis passes going forward -- full 65-session Tier A
  pool, unbounded by dataset. Full-grid analysis cost at this setting:
  ~752s/run x up to 16-24 runs (8 cells x 2-3 seeds) = roughly 3.3-5h, run
  as its own pass after training completes (`run_grid.py`'s Step 3), not
  packed into the same 8h training budget. Seed count is UNCHANGED by this
  decision (2-3 seeds, per the E3 wall-clock benchmark) -- session count and
  seed count are orthogonal budgets (sessions -> analysis wall-clock only,
  seeds -> training wall-clock only), so raising one doesn't require
  lowering the other.
- **Ablation battery + identity-catch mechanisms implemented; training
  campaigns launched (items 5-6).** `train.energy_cost_weight` (L2/mean-
  firing-rate penalty on `h_star`, added post-loop so it survives both
  "ce" and "reinforce" signal branches) and `model.recurrent_noise_sigma`
  (Gaussian noise added to the persisting recurrent state, both S=0 and
  S=1, in both train and eval modes) are wired into `train.py`'s
  `_run_trial`/`_step_core`, default 0.0 (off) for every Core cell.
  `M111_pbwm`: `models/gru_cell.py::PBWMManagerCell` (LSTM-style 3-gate
  manager, input/forget R_t-driven via `beta*R_t`, output NOT R_t-driven
  per spec) wired into `HRLCore` via `pbwm_gate`/`reflection_beta`
  constructor args and `_build_model`/`train_one`'s
  `run.get("pbwm_gate", False)`. Identity-catch trials
  (`task.identity_catch_fraction`, §9.4a): `SternbergGenerator` replaces
  the probe step with a fixed-position (serial position 1) category query
  on the configured fraction of trials, tagging the whole trial
  `aux_family=1`; `Heads.identity_aux` (constructed only when the fraction
  is >0, so Core heads/checkpoints are byte-identical to before this
  feature existed) reads the same `h_star` the policy head reads; its
  cross-entropy loss is added post-loop like the energy-cost term. Catch
  trials are masked out of the ordinary policy/value loss (`non_catch`
  mask) since they have no real in/out judgment (`in_set=None`).

  Three gaps found and closed on the 2026-07-07 pass, before launching
  training: (1) `energy_cost_weight`/`recurrent_noise_sigma`/
  `identity_catch_fraction` were only readable from the global
  config.yaml, with no way for distinct ablation arms to each set their
  own value without editing the shared file between launches -- `train_one`
  now folds `run.get(...)` overrides into its local `full_cfg` copy before
  any downstream reader (model construction, task generation, eval-time
  accuracy) consumes it, so every arm's `run` dict carries its own
  override and arms can be interleaved/resumed freely. (2)
  `evaluate_accuracy`/`_run_trial` had no separate identity-report-accuracy
  metric for catch trials -- `_run_trial` now returns a 4th value,
  `identity_catch: {"correct", "total"}`, accumulated whenever
  `heads.identity_aux is not None`; `evaluate_accuracy` reports it as
  `accuracy["identity_catch"]` and excludes catch trials from the
  match/non-match `load{N}` denominator (they have no real in/out
  judgment, so leaving them in would silently dilute those gates). (3)
  `generate_activity_logs.py`'s `_build_model` calls didn't pass
  `pbwm_gate`/`identity_catch_fraction` at all -- replaying an
  `M111_pbwm`/`*_idcatch` checkpoint would have crashed on a
  `load_state_dict` key/shape mismatch; `_run_id_extras(model_id)` now
  derives both from the run_id suffix convention
  (`M111_pbwm`/`M000_idcatch`/`M111_idcatch`) and all three `_build_model`
  call sites use it. `analysis/run_all.py::_is_ablation_or_catch_variant`
  excludes these runs from the Core S x M x P regression (same S/M/P bits
  as their base cell, but a materially different trained representation --
  pooling them in would silently bias that cell's rows); they're reported
  via their own comparison tables in RESPONSES.md instead. Two new
  orchestration scripts (`scripts/run_ablation_battery.py`,
  `scripts/run_identity_catch.py`, run_id convention consumed by both
  fixes above) launch the >=5-seed/arm and >=4-seed/cell campaigns
  respectively, reusing `run_grid.py`'s manifest/report/signal-handling
  machinery. All five smoke-tested end-to-end (including
  `accuracy["identity_catch"]` reporting a real 0-1 value) before the real
  campaigns were launched.

  Item 5 (ablation battery, 5 seeds/arm, dev tier) finished 2026-07-08
  00:11 UTC -- results table in RESPONSES.md Part 5. +energy cost is the
  only arm that beats M111's own accuracy/gate-clear rate (mean
  load1=0.930 vs M111's 0.675-0.90 range, 3/5 seeds clearing load1 vs 0/2
  for M111); +noise is uniformly worse (0/5 gate clears, lowest means on
  all three loads); +pbwm_gate sits in between, roughly matching M111. The
  `_is_ablation_or_catch_variant` guard currently means none of these 15
  runs have RSA alignment computed at all (skipped entirely from
  `align_one_run`, not just from the pooled regression) -- item 5's actual
  comparison DV is "probe-epoch alignment at matched accuracy," so this
  guard needs relaxing (compute alignment, keep excluding only from the
  pooled S x M x P regression rows) before the ablation battery's real
  question can be answered; deferred to the consolidated `run_all.py` pass
  once item 6 also finishes. Item 6 (identity-catch, 4 seeds/cell, dev
  tier) launched 2026-07-08 ~11:07 (local); no code changes needed to
  start it, `run_identity_catch.py` was already correct from the prior
  session's implementation pass -- only item 5 needed babysitting to
  completion and the item-6 launch step, which the monitoring chain missed
  (item 5 finished at 03:41 local but nothing launched item 6 until this
  pass at 11:07 -- the scheduled hourly health-check chain evidently
  stopped hopping at some point after item 5 completed; noted so future
  autonomous monitoring chains build in an explicit "did the process exit"
  check rather than relying solely on the chain continuing to fire).
- **Reviewer suggestions (§4.1-4.5), applied 2026-07-08: two applied in
  full, one applied partially (reporting only), two skipped with
  rationale.**

  **4.1 (performance-matched baselines): applied.** Three flat-GRU control
  arms off M000 (S=0,M=0,P=0): `flat_gru_2x` (`model.flat_units` doubled,
  per-run override folded into `_build_model`'s `full_cfg` copy exactly
  like the item-5 ablation overrides), `flat_gru_l1` (`train.
  l1_weight_penalty` on the core's own weight-matrix parameters, added as
  a parameter-space term directly in `train_one`'s step loop -- distinct
  from `energy_cost_weight`, which is an activation-space penalty inside
  `_run_trial`), `flat_gru_dropout` (`model.core_dropout_p`, standard
  dropout on the S=0 core's hidden state, train-mode only via a new
  `training` flag threaded through `_step_core` -- unlike
  `recurrent_noise_sigma`, which stays on at eval too since it models a
  persistent noise source rather than a training-time regularizer).
  Model_id convention `M000_2x`/`M000_l1`/`M000_dropout`; `_run_id_extras`
  (`generate_activity_logs.py`) extended to a 3-tuple
  `(pbwm_gate, identity_catch_fraction, flat_units_mult)` since `_2x`
  changes the core's actual parameter shapes (replay would otherwise crash
  on `load_state_dict`, same bug class item 5/6 caught proactively) --
  `_l1`/`_dropout` need no replay-time handling since neither changes a
  parameter shape. `run_all.py`'s existing `model_id != "M{S}{M}{P}"` guard
  already covers these three model_ids with no change needed. Launched via
  new `scripts/run_perf_matched_baselines.py --seeds 3 --tier dev`
  (3 arms x 3 seeds = 9 runs), smoke-tested end-to-end first (including a
  `load_state_dict` round-trip check on the `_2x` arm's doubled-width
  checkpoint) before real training; running in the background alongside
  item 6 (both CPU-only per `utils/device.py`'s sm_120 fallback -- 32
  cores, <10% load before either campaign started, so parallel execution
  was judged safe rather than queuing item 6 -> item 4.1 sequentially).

  **4.2 (internal brain-like properties): applied.** New
  `analysis/network_properties.py`: `weight_entropy` (Shannon entropy of a
  weight tensor's value histogram), `gru_effective_connectivity` (collapses
  any GRU-family `weight_hh` [n_gates*H, H] into an [H,H] connectivity
  graph by summing |gate block| per (i,j) -- generic across 3-gate GRU
  cells and the 4-gate `PBWMManagerCell`), `modularity_q` (Louvain
  community modularity via `networkx`, added to `pyproject.toml`'s main
  deps), `small_worldness` (Watts-Strogatz sigma on a thresholded,
  largest-connected-component subgraph -- deliberately coarse
  `n_random`/`n_iter` defaults, same tractability tradeoff
  `rsa.py::within_session_noise_ceiling` makes for its own resample count).
  Cross-temporal decoding stability (H5) was NOT reimplemented --
  `dynamics_and_persistence.py::stability_index_for_session` already does
  this. `mixed_selectivity_index` (Rigotti et al. 2013 nonlinear mixed
  selectivity: per-unit load x item interaction-SS / total-SS from a
  standard ANOVA decomposition) uses **held-item identity, not category**,
  as the second factor -- found while implementing that
  `generate_activity_logs.py`'s real-data replay path
  (`generate_activity_log`) hardcodes `held_categories=[""] * len(...)`
  and `probe_category=""` (the DANDI item-identity mapping carries no
  semantic category label, so this was never wired up); `held_items`
  (session-local item index) is always populated and is a legitimate
  delay-period variable in its own right, so the metric was defined on
  that instead of blocking on a DANDI-adapter category-labeling fix that's
  out of scope for this pass. New `scripts/analyze_network_properties.py`
  computes all four metrics for every completed checkpoint already in
  `results/manifest.jsonl` (weight/connectivity metrics need only the
  checkpoint; mixed-selectivity additionally needs an existing
  `results/activity_logs/{run_id}.parquet`, generating one fresh is out of
  scope here -- that parquet is the neural-alignment pipeline's own
  artifact). All 10 new unit tests
  (`tests/test_network_properties.py`) pass, including a synthetic-data
  check that `mixed_selectivity_index` correctly isolates a planted
  XOR-like interaction unit from purely-additive units at matched
  across-cell variance.

  **4.3 (multi-metric alignment battery): skipped.** `rsa.py`'s own
  docstring states the operating principle plainly: "no alignment
  magnitude is interpretable in isolation" -- every existing DV
  (crossnobis RSA, both raw and normalized) is reported against a
  carefully-validated, representation-matched noise ceiling
  (`within_session_noise_ceiling`, itself through several audit-fixed
  iterations: N3's split-half-vs-fold-reshuffle bug, the B2 trial-level
  ceiling fix from item 1, etc.). Bolting on CKA/Procrustes/Ridge-encoding/
  DSA without an equally rigorous, metric-matched ceiling for each would
  produce additional numbers that LOOK like independent confirmatory
  evidence but aren't actually validated to the same standard as the
  existing pipeline -- worse than not reporting them at all, since a
  reader can't tell which of five metrics to trust without that same
  design work repeated five times. This is a real, well-scoped follow-up
  (each metric needs its own ceiling-estimation design pass, not just an
  API call), not something to rush alongside items 4.1/4.2/5/6 in one
  session -- left for a dedicated future pass.

  **4.4 (evolutionary conditioning control): skipped.** A structurally
  different training paradigm (population-based evolutionary optimization
  pretraining a subtask, not a config knob on the existing BPTT/node-
  perturbation pipeline) -- new hyperparameters (population size,
  mutation/crossover operators, generation count, subtask curriculum), a
  materially larger compute budget on top of the campaigns already
  running, and a hypothesis (evolved-prior vs. learned-from-scratch
  inductive bias) the current protocol doesn't include or motivate.
  Implementing this well would need its own protocol section and design
  pass, not a bolt-on; flagged as a possible future study rather than
  attempted at reduced rigor to fit this session.

  **4.5 (effect sizes/power): applied partially -- reporting only, no new
  training.** Added `stats.bayes_factor_bic` (BF10 via the Wagenmakers
  2007 / Raftery 1995 BIC approximation -- the simple closed-form version,
  not the numerically-integrated JZS Bayesian t-test, since this is a
  reporting complement to the existing Mann-Whitney/rank-biserial
  comparisons, not a replacement for them) and applied it retroactively to
  item 5's already-collected ablation-arm-vs-M111-Core accuracy data (see
  RESPONSES.md Part 5) -- this needed no new compute, just a function over
  data already on disk. The suggestion's other half, "run the full
  150k-step grid with 8 seeds," was NOT launched: it's a genuine ~10-20x
  increase over the dev-tier/2-3-seed budget this project deliberately
  fixed (see the wall-clock benchmark / grid budget decision entry above),
  on a single local machine already running two background training
  campaigns (items 5+6, now also 4.1) -- unilaterally committing many
  additional hours-to-days of this machine's compute crosses from
  engineering judgment into a resource decision that needs the user's
  sign-off, not an autonomous one. Available on request
  (`run_grid.py --tier full --seeds 8`).

- **2026-07-08: 5-arm ablation battery arm T/D "on" magnitude (`train.topo_loss_weight_on`/`dale_penalty_weight_on` = 0.01 each).** The design added arms T (topographic smoothness) and D (Dale's law) with a per-cell binary bit each (model_id positions 4/5), but `run_grid.py`'s cell dicts only carry the bit itself, not a loss weight -- `train_one` had to translate `run["T"]`/`run["D"]` into an actual nonzero `topo_loss_weight`/`dale_penalty_weight` for cells with the bit on (an explicit override in `run` still takes priority, for supplementary arms that want a custom value). Chose 0.01 for both as a starting point, matching the existing `energy_cost_weight` ablation arm's magnitude (0.01) rather than `l1_weight_penalty`'s (0.001) -- no principled reason to prefer either scale a priori, and per comments.txt §6.1 this is meant to be tuned during the Phase B dev-tier sanity check if training destabilizes, with any change recorded here.
- **2026-07-08: 3 two-arm interaction-probe cells added to the ablation battery (user request), bringing it from 12 to 15 cells.** The 12-cell reference-anchored design (baseline/full + 5 knock-one-out + 5 add-one) only resolves each arm's NET interaction lumped across all other arms (comments.txt §2's stated scope limitation), not any specific pairwise interaction. Added M10010 (S+T, topography on its native worker-grid substrate, isolated from M/P/D -- closes the T substrate-entanglement gap comments.txt §2.2 flags, since M00010 alone only tests T on the flat substrate), M00011 (T+D on the flat substrate, isolated from S/M/P), and M10001 (S+D "bio-plausible backbone", isolated from M/P/T). These resolve only these 3 specific pairs; the rest (S×M, S×P, M×P, M×T, M×D, P×T, P×D) remain lumped/unresolved, per comments.txt's addendum.

- **2026-07-08: Rutishauser data-preprocessing verification (user-requested) found one real gap in the neural adapter (fixed) and one near-miss (tried, then reverted after a follow-up literature check).** A research pass cross-checking `dandi_nwb.py` against the datasets' own NWB structure and source papers (Kyzar et al. 2024, Daume et al.) confirmed WM-session identification, column mapping, outcome-field semantics, and epoch timing windows are correct for Tier A (000469+000673) -- the maintenance window (1.5s) has real headroom against an empirically-measured ~2.5-2.7s median true delay duration.
  1. **Tier B (001187) could not load at all** (real fix, kept): its Sternberg trials live under `intervals/WM_trials`, not the top-level `intervals/trials` 000469/000673 use (001187 also has an unrelated `intervals/LTM_trials` New/Old-recognition task that must not be picked up as WM data). `find_wm_sessions`/`_load_session` now check both candidate group names (`TRIALS_GROUP_CANDIDATES`); 001187's `WM_trials` schema matches 000673's exactly, so it reuses that `COLUMN_MAPS` entry. Verified against real data: 001187 now loads (3 sessions/23 units checked, contract-valid). Tier B is loadable but still not wired into any analysis path -- that remains future scope. Regression-tested (`tests/test_dandi_adapter.py::test_tier_b_001187_loads_via_wm_trials_group`).
  2. **Isolation-distance unit QC: implemented, then reverted.** The first verification pass noted `units/waveforms_isolation_distance` exists and matches what Kyzar et al. 2024 report, which was over-read as license to add a numeric inclusion threshold (`min_isolation_distance=20.0`, applied to `_apply_firing_qc`). A follow-up literature check found this was wrong on two counts: (a) none of the Rutishauser/Daume/Kaminski papers use isolation distance or SNR as a numeric inclusion threshold -- they're reported only as post-hoc descriptive statistics; unit curation happened qualitatively during OSort spike-sorting (firing-rate/waveform stability, ISI distribution, refractory violations) and is already reflected in which units made it into the released `units` table, so adding a new numeric cutoff would be an invented criterion with no support in the methodology being replicated; (b) a NaN isolation distance is the *expected* value for a unit that is the only cluster on its channel (no second cluster to compute a Mahalanobis distance against), not a quality flag -- the reverted code was treating it as a QC failure, which would have systematically (and wrongly) discarded some of the least-ambiguous units. Reverted in full (adapter, config, callers, tests); unit inclusion is bare firing rate only, as before. Left here as a record of a plausible-sounding but unsupported change that got caught before it touched any real results.

- **2026-07-08: Phase B (pipeline validation) complete; loss decreases and no NaNs with topo+dale both active.** Smoke tier (`--tier smoke --seeds 1`) trained all 15 ablation-battery cells end-to-end with no errors. A dev-tier (`--tier dev`, 20k steps) run of M11111 (full reference: S=M=P=T=D=1, both new mechanisms at their chosen "on" magnitude 0.01) went from near-chance accuracy (0.5/0.65/0.525 at the smoke tier's 100 steps) to load1=0.675/load2=0.775/load3=0.675 at 20k steps -- genuine learning progress -- with every checkpoint tensor confirmed NaN-free. Behavioral gates (load1>=0.95, load3>=0.80) are not expected to clear at dev tier (comments.txt: dev tier is a sanity-check tier, not a results tier) and did not; this is expected, not a failure signal. No adjustment to `topo_loss_weight_on`/`dale_penalty_weight_on` was needed.
