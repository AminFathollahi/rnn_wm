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

Responds to `comments.txt`, a senior-review audit finding the first grid's
12 completed runs scientifically invalid (L=1 confounded with supervision
density and at chance; the alignment DV degenerate -- raw exceeding the
noise ceiling for 5/6 cells; runs spanning 3 different git commits
mid-grid). Full section-by-section responses to every judgment call are
appended to `comments.txt` itself (a `RESPONSES` section, R1-R11) per the
user's request that all such calls be explained inline; this entry is the
durable project-history record of the same work, cross-referenced by Rn.

- **A1a (training-signal confound, R3):** L=0 now trains via REINFORCE
  (policy-gradient + value baseline, backprop through time) on the SAME
  sparse trial-end reward L=1 uses, for the ramp+target curriculum phases
  (~80% of steps); both arms share a matched-density warmup phase (~20%)
  for tractability (L=0: unchanged dense CE; L=1: node-perturbation reward
  becomes "fraction of ticks matching the ideal action" instead of bare
  correct/incorrect, warmup-only). Policy-gradient terms collected only at
  probe-epoch ticks (`training/train.py::_run_trial`, `signal="reinforce"`).
- **A1a/A2a real-data finding (R4):** literal held-item-identity conditions
  (A2a's original ask) have essentially zero repeated trials on real Tier-A
  data -- verified empirically: every held item across 70 load-1 trials in
  a representative `000673` session was distinct. Condition-averaged RSA
  cannot work on raw item identity here. Resolution: impute a category
  label per stimulus image via nearest-neighbor in the model's own frozen-
  encoder feature space against its training-pool category centroids
  (`analysis/stimulus_categories.py`), condition the maintenance-epoch RDM
  on `(sorted held-category multiset, load)` instead of raw item identity.
- **A2 architecture (R6):** maintenance/encode-epoch alignment is computed
  PER SESSION (category conditions are session-specific -- imputed
  per-session from that session's own cached stimulus features), each
  against a new `rsa.within_session_noise_ceiling` (repeated resampled
  fold-splits of ONE session -- a cross-session LOSO ceiling is undefined
  when conditions don't recur across sessions). The probe epoch's coarse
  `(load, in_set, correct)` conditions DO recur across sessions, so that
  path stays pooled, built via a new `analysis/pseudopopulation.py` (each
  unit's condition mean from only that unit's own session's trials --
  never diluted by another session's zero-filled entries, which
  `dandi_nwb.rates(None,...)`/`response_patterns` both did previously),
  with the ceiling computed on that same pooled representation.
  Per-session maintenance rows carry the session's patient
  (`dandi_data.patient_of`), resolving B3's request for a real patient
  factor in the mixed-effects model.
- **A1b:** rung 2's `adaptive_baseline` now uses a genuinely faster EMA
  decay constant (`baseline_decay_adaptive`) distinct from rung 1's --
  previously byte-identical to the `else` branch (dead distinction).
- **A1c (batching):** one training step now runs a batch of `B` trials
  (`train.batch_size`, config), sharing one load drawn per batch (tick
  count is fully determined by load, so no padding is needed -- see
  `tasks/generator.py::TaskGenerator.sample_batch`). Two real bugs found
  while implementing this (both would have silently corrupted any B>1 run,
  not just been suboptimal):
  1. **Reflective-gate broadcast bug (R9):** `surprise()`/`step()`/
     `gate_bias()` mis-broadcast `[B,1]*[B]` to `[B,B]` for B>1 (harmless
     at B=1 by coincidence). Fixed by keeping the `[B,1]` convention for
     reward/value/action-logp consistently through that path.
  2. **Node-perturbation batch-averaging bug (R10):** naively averaging the
     eligibility trace across the batch BEFORE the reward is known replaces
     `mean_b[(r_b-baseline)*xi_b]` with `mean_b[r_b-baseline]*mean_b[xi_b]`
     -- the product of averages instead of the average of products --
     destroying the reward-perturbation correlation node perturbation
     exploits. Fixed by keeping PER-SAMPLE traces (`[B, *param.shape]`)
     in `mechanisms/local_learning.py` until the reward-weighted average at
     `apply_update`. Regression test:
     `test_mechanisms.py::test_batched_node_perturbation_preserves_per_sample_correlation`.
- **B5/B6 (R11):** fixed (not just documented) the heads' 2x-effective-
  local-learning-rate quirk: `NodePerturbationLearner` now has a genuine
  single-weight mode (`weight_ih=None`) for `Heads.pi`/`Heads.value` (plain
  `Linear` layers), instead of pointing `weight_hh`/`weight_ih` at the same
  parameter (which applied the three-factor update twice per trial).
- **B1:** `run_grid.py`'s config fingerprint is now `hashlib.sha256` over
  the full resolved config (model+mechanisms+task+train+gates+neural+tier),
  computed once per grid invocation and dumped verbatim to
  `results/resolved_config.yaml`; the old `abs(hash(json.dumps(tier_subset)))
  % 1e8` was salted per-process (`PYTHONHASHSEED`) and only covered the
  tier subset, both independently wrong.
- **B2:** gate boundary is `>=` (not the original strict `>`) in `train.py`
  and `preregistration.md`, applied consistently (`run_grid.py`'s scaffold
  stub too, for key-naming consistency even though it doesn't share the
  bug).
- **B3:** `analysis/stats.py::mixed_effects_alignment` adds a `patient`
  variance component (`vc_formula`) when the input has a `patient` column,
  fed by the new per-session maintenance rows; added `accuracy_vif` for the
  documented accuracy-vs-L collinearity check.
- **C1 (reflection-shuffle causal control):** wired in
  `training/generate_activity_logs.py::generate_activity_log_reflection_shuffled`
  -- re-replays every M=1 trial using that trial's OWN natural R_t sequence
  (read back from the normal log) but time-shuffled within the trial
  (`shuffle_reflection`), and `analysis/run_all.py::
  reflection_shuffle_lesion_for_run` compares normal-vs-shuffled maintenance
  alignment. F4 figure added.
- **C2 (region dissociation):** `dandi_nwb.py`'s `units`/`rates`/
  `response_patterns` now accept `"MTL"`/`"MFC"` as `region`, resolved via
  the existing `dataset_contract.region_family()` helper; `run_all.py`
  computes both epoch paths at `region in (None, "MTL", "MFC")`.
- **C3/C4 (H5/H6, wired per explicit user instruction, not scoped out):**
  new `analysis/dynamics_and_persistence.py` wires `cross_temporal_decoding`
  (H5: stability index, dynamic-vs-stable delay coding) and
  `persistent_activity_index` (H6) against real per-session model+neural
  data, using LOAD as the shared decodable/indexed factor (item/category
  identity recurs too rarely per session for a stratified decode with
  several folds -- see the A2a finding above). Compared model-vs-brain via
  `stats.compare_distributions`. F5 (persistence) and F6 (dynamic/stable)
  figures added. Encoding-model R^2 and full real-data dPCA marginalization
  were considered but ultimately not wired in this pass (time; H5/H6's
  existing machinery already exercises the core real-vs-model comparison) --
  see `comments.txt` RESPONSES R2. F1 (design schematic) and F8 (Tier-B
  cross-dataset replication) explicitly scoped out (R2).
- **Performance (R7, real bug found while validating the rewrite against
  real data, not separately requested):** `dandi_nwb.py`'s `rates()`/
  `response_patterns()` previously recomputed every spike histogram from
  scratch on every call -- already flagged above as a follow-up, but the
  new pipeline calls `rates()` far more often (once per session, per
  region, per noise-ceiling resample) than the original single-pass code,
  making this non-optional. Added a per-`(unit, epoch)` cache at
  `self.bin_ms`, mirroring `sim_brain`'s existing `_rate_cache`; also
  cached `trials()` (previously rebuilt via `pd.concat` on every call).
- **Fold-count bug found while validating the per-session path against
  real data (R8):** `_session_condition_rdm` derived its crossnobis fold
  count from the SINGLE RAREST condition across the whole session -- with
  the fine-grained category-multiset schema, one or two 1-trial "singleton"
  conditions used to zero out the fold count (hence the RDM) for the ENTIRE
  session, even though most conditions were well-populated. Fixed by
  dropping rare (`< min_trials_per_condition`) conditions before computing
  fold count, mirroring the min-trials filter `run_all.py` already applied
  model-side.
- **`alignment_results.csv`'s schema changed:** no longer a single
  `normalized_alignment` column -- now `maintenance_normalized_alignment`
  (per-session category schema, mean across sessions) and
  `probe_normalized_alignment` (pooled coarse schema), reflecting the two
  epoch-appropriate paths above. Long-format per-session rows (with the
  patient factor) are in the new `results/alignment_by_session.csv`.

## Adversarial review findings (2026-07-05, same audit-fix pass)

Two fresh subagents (no context from my own implementation reasoning)
reviewed the full diff independently. Full detail in `comments.txt`'s
`RESPONSES` section (R12-R16); durable summary here:

- **Region-family fallback bug (R12):** `rsa.py::_session_condition_rdm`
  and `dynamics_and_persistence.py::neural_session_epoch_timeseries` both
  matched units to a session via `startswith`, then fell back to using
  EVERY session's units when the target session had none in the requested
  region. 31% of real Tier-A sessions have zero MFC units -- for those,
  the fallback combined with `rates()`'s cross-session zero-fill produced
  an all-zero RDM reported as a spuriously valid "ok" row, contaminating
  the C2 region-dissociation analysis. Fixed: exact session match, return
  `None` (no fallback) when a session has no units in that region.
- **Incomplete NaN guard in `rdm.py::crossnobis_rdm` (R13):** checked only
  one condition of each pair's fold-means, not both; fixed.
- **Salted-hash reproducibility bug reintroduced (R14):** the reflection-
  shuffle lesion's RNG seed used Python's built-in `hash()` on a string --
  the exact defect B1 fixed for the grid's config hash. Fixed with
  `hashlib.sha256`, same pattern as B1 (also found and fixed in
  `run_grid.py`'s `--scaffold` stub).
- **Nested-not-crossed random effects (R15):** `stats.py::
  mixed_effects_alignment`'s patient variance component nested patient
  within seed instead of crossing them (confirmed via log-likelihood
  comparison), understating patient-driven structure. Fixed using the
  standard statsmodels crossed-effects workaround (dummy constant group,
  both factors as `vc_formula` terms); generalized to an `extra_vc_col`
  parameter.
- **Region pseudo-replication in the main LME (R16):** `run_all.py` was
  feeding pooled+MTL+MFC rows for the same session into one flat model --
  correlated subsets of the same data, not independent observations.
  Split into two models: the main S*M*L*accuracy model on pooled rows
  only, plus a separate H1/C2 region-dissociation model (MTL/MFC rows,
  `session` as the R15-generalized crossed factor).
- **`trial_id`-collision bug found via direct wall-clock profiling (not
  raised by either reviewer):** `run_all.py::_model_epoch_patterns` grouped
  activity-log rows by `trial_id` alone; `trial_id` resets to 0 per session
  in the replay log, so the probe-epoch pooled path (multi-session) was
  silently merging unrelated trials from different sessions sharing a
  trial index -- this alone explained why probe-epoch alignment reported
  "insufficient_shared_conditions" for every cell in the first post-fix
  validation pass. Fixed by grouping on `(session, trial_id)`; regression
  test added. Also standardized features in `cross_temporal.py`'s
  per-timebin `LogisticRegression` fits (H5) -- unscaled raw firing
  rates/activations were causing routine `lbfgs` non-convergence.
- **Validated post-fix** on a smoke-tier (100-step) 8-cell run at 8
  sessions/dataset: probe-epoch alignment now reports `status="ok"` with
  8 shared conditions for all 8 cells (previously always
  "insufficient_shared_conditions"); `normalized_alignment` is not
  identically 1.0 and `raw_alignment` stays below `noise_ceiling_upper`
  for every row (H2's DV-sanity acceptance gates); the region-dissociation
  model correctly uses `statsmodels.MixedLM + session variance component`
  (not the OLS fallback); the main S*M*L model falls back to
  OLS+cluster-bootstrap at this stage only because a 1-seed smoke run
  gives MixedLM's `groups=seed` a single level (expected, resolves once
  the real grid runs >=8 seeds). The chance-vs-trained gate (H5) is a
  near-tie at smoke tier (chance=0.150 vs trained=0.151) -- expected, not
  a DV bug: 100 training steps do not train these models past chance
  behavior (consistent with this project's own prior finding, above, that
  even 1500-3000 steps showed no learning in diagnostic tests); this gate
  should be re-checked once the real grid's `dev`/`full` tier runs
  complete.
