# Work queue

Durable across sessions and compactions. `[ ]` open, `[~]` assigned and in
flight, `[x]` done with the commit that closed it. Executors tick their own
line in the same commit that does the work. Nothing is complete until its
acceptance output is in `executor.md`.

## Standing rules for every executor

- No network training launches. "Training" means fitting recurrent network
  weights. Prepare and verify those commands, put them in
  `../training_commands.txt`, and stop. Everything downstream of a saved
  checkpoint -- replay, measurement, alignment, intervention sweeps,
  aggregation -- is analysis: implement it AND run it, and report the real
  numbers. An implementation without its result table is not finished.
- Repair measurements and reuse saved checkpoints first. A new training run
  must answer a specified question that existing checkpoints cannot answer.
  A parser, aggregation, neural-window, or documentation repair does not by
  itself justify retraining. Only a verified training defect affecting a saved
  run justifies replacing that run; enumerate the affected runs first.
- Reuse existing functions; never a second RDM builder, bootstrap, or
  aggregation path.
- Docstrings and comments at a minimum. No internal references anywhere in
  code, names, or written artifacts -- including this project's own
  shorthand ("Gate A"/"Gate B", "the campaign", section or phase numbers,
  decision ids, document names). Write what the thing is.
- One item per commit, explicit `git add <path>`, never `git add -A`.
- Never delete experiment outputs; archive with a clear name.
- Report a null or a broken thing you did not fix, rather than smoothing it.

## Settled, do not reopen (2026-09-12)

Three questions raised by the external audit are closed by user decision.
They are recorded here so no later round spends compute reopening them.

1. **Model step duration and human epoch duration need not match.** The
   model averages its 15 maintenance states; the neural side averages the
   recorded delay. Neither is stretched, interpolated, or assigned seconds
   per model step. Timing ratios between datasets (encoding medians ~1.02 s
   in 000469 against ~2.02 s in 000673, both delay medians ~2.69 s) are a
   sampling fact about the recordings, not a defect in epoch-averaged
   similarity analysis, and they license no training sweep.
2. **The ten-entry task cue stays.** Its width is unchanged for every
   checkpoint on disk, and any continuation study encodes task identity
   inside the existing ten entries rather than widening the interface.
3. **No full-grid or whole-architecture retraining.** Flat and hierarchical
   cores are already matched at the effective-synapse level (73,728 against
   74,052, 0.44%), so a 196-unit flat replacement grid repairs nothing. The
   locality control in A7b is a useful budget-matched comparison. The crossed
   controls in A7c separately test width and connectivity; they extend the
   experiment rather than replace completed models.

The timing sweep that used to sit in this queue is gone; what remained of it
is the data-quality item B7. The task-cue schema item is gone as a separate
line; what remained of it is A9(c).

## File ownership while the current round is in flight

| Owner | Files |
|---|---|
| memory-demand ids | `scripts/run_identity_catch.py`, `generate_activity_logs.py`, `scripts/align_extended_variants.py`, `run_grid.py` guard, `../training_commands.txt` |
| locality-matched flat arm | `run_grid.py` arm, `train.py` flat core + model override, `generate_activity_logs.py` replay override, `../training_commands.txt` |
| alignment | `contrasts.py`, `scripts/analyze_alignment_contrasts.py`, `run_all.py` alignment/probe region |
| dynamics | `merge_analysis_shards.py`, `analysis.sh`, `dynamics_and_persistence.py`, `persistence.py`, `cross_temporal.py`, `run_all.py` dynamics region |
| partial RSA | `task_structure.py`, `rsa.py`, `rdm.py` |
| intervention | new modules and scripts only |

## A. Training

- [x] A1 Three leaky rate cores (E/I, dynamic-synapse, low-rank) -- `57d8f50`.
- [x] A2 Parameter match and gradient-flow report -- `5662e0e`.
- [x] A3 `run_grid.py --substrate`, per-substrate resolved config -- `33b1b5e`.
- [x] A4 Replay path reads the recorded substrate -- `7d3a07f`.
- [x] A5 Variant pairing for the three substrates -- `8119c3b`.
- [x] A6 `../training_commands.txt`: 48 substrate runs + 24 memory-demand runs,
      72 unique ids, verified by enumeration. Ready to launch.
- [x] C8 Memory-demand run ids namespaced by signal (`build_run_id`,
      `6368e52`); readers fixed (`generate_activity_logs._run_id_extras`,
      `6368e52`); substrate-arm guard now rejects T/D cells (`197bd59`);
      `../training_commands.txt` unblocked (26 memory-demand runs total).
- [x] A7a Capacity audit, closing the width question. Measured from the
      configuration and the built cores: flat 128 units = 73,728 effective
      structural synapses; hierarchical 196-unit worker + 24-unit manager =
      74,052; a 0.44% difference. Structural counting includes input,
      recurrent-surviving and feedback projections and excludes biases and
      the plastic mixing coefficient. The per-cell units / structural /
      active / allocated table is in `PROJECT_IMPLEMENTATION_STATUS.md` 3.1;
      plastic cells are matched structurally but not in active parameter
      count, which stays a stated limitation. Conclusion: the flat core is
      already budget-matched, a 196-unit flat grid is an optional wide
      control and not a repair, and flat 128 keeps its 16x8 sheet.
- [x] A7b Locality-matched flat control (trained 2026-09-14; all 16 runs
      recorded complete at the full tier, re-verified 2026-09-17).
      Structure is confounded with recurrent sparsity. The hierarchical
      worker's recurrence is locality-masked on a sheet at density 0.10; the
      flat core's is dense. Local connectivity, rather than hierarchy, is a
      plausible alternative source of brain-like geometry, and that is what
      this control tests. The
      sheet's status differs too: flat topography is imposed on an arbitrary
      unit ordering while hierarchical topography is the connectivity itself.
      Train `M00000` and `M00010` as flat cores of 289 units on a 17x17
      sheet, locality-masked by the same construction the worker uses: 8
      seeds, supervised signal only, 16 runs. Solved and measured before
      assignment, mask seed 0 -- target density 0.0681 gives 72,612 effective
      structural synapses and a mean in-degree of 19.75, against the worker's
      19.69 and against flat 73,728 / hierarchical 74,052. The three-way
      synapse spread is 1.98% relative to the smallest count, inside
      `param_budget_tol`, and the sheet is
      square like the worker's.
      The recurrent init is matched too. `reset_parameters` draws
      `weight_hh` uniformly with half-width 1/sqrt(hidden units), and the mask is applied after,
      so a masked core's recurrent gain at initialization depends on its
      width: measured candidate-block spectral radius is 0.584 flat dense,
      0.208 for the worker, 0.168 for the control at its own width. The
      control draws at the worker's per-synapse scale instead, giving 0.204
      against 0.208. `weight_ih` is deliberately left alone -- the worker's
      96-wide input against the control's 64 is the top-down pathway itself,
      which is under test rather than a nuisance.
      Why in-degree and not density. Width, synapse count and connection
      density cannot all be held at once, so one of them is solved for. Mean
      in-degree is the per-unit quantity that defines a sparse local regime
      -- it is what the configuration's own comments track, and what made
      density 0.65 unacceptable in the withdrawn arm. Holding the nominal
      0.10 at 289 units would have put in-degree 38% above the worker's while
      buying nothing; holding in-degree costs 1.5% of the synapse budget and
      nothing else.
      What it estimates. Hierarchical minus locality-matched flat compares
      architectural packages at similar structural budget, worker in-degree,
      and initial candidate-matrix radius. Width, exact mask density, spatial
      extent, input projections and the manager system still differ. The
      kernel family is shared; its calibrated scale and realized geometry
      need not be identical. Locality-matched flat minus dense flat changes
      width, connectivity and initialization together. Neither contrast
      independently identifies a pure hierarchy or pure sparsity effect.
      What it does not control: total units, 289 against 196+24 against 128.
      Units, synapses and in-degree cannot all be matched at once; the
      preregistered criterion is synapses, and unit count is handled by the
      subsampling rule for per-population analyses. Subsampling equalizes
      measured feature counts, not capacity during training. Do not report
      the arm as fully matched. Candidate-weight radius is an initialization
      diagnostic, not the spectral radius of the complete GRU Jacobian.
      This replaces the grid-matched worker arm (`d39c536`, reverted). At 128
      worker units the preregistered density gives 46,701 synapses, and
      restoring the budget would need ~0.65, a mean in-degree of ~84, no
      longer sparse local connectivity; shrinking the hierarchical side
      bought grid geometry at the cost of the capacity match and left
      sparsity uncontrolled. Matching the flat side buys both. That arm also
      had no replay path -- the activity-log generator rebuilds a model from
      the default config before loading a checkpoint, so any run trained at a
      non-default width would have failed the state-dict load and produced no
      log to analyze. The replacement fixes that.
      Existing runs are untouched. LAUNCHED 2026-09-13 00:53 as part of the
      168-run outstanding set (see the launch record below).
- [x] A7c Cross width, connectivity and topographic loss to estimate their
      effects separately. Use flat GRUs with widths {128,289}, connectivity
      {dense, random sparse, spatially local sparse}, and T={0,1}, with
      M=P=D=0. Initial scope: SUP, eight paired seeds, 12 cells/96 total runs.
      Existing dense128 cells supply 16 runs. The prepared local289 arm uses a
      different recurrent initialization scale and remains a separate budget
      control. The other five width/connectivity combinations require 80 runs.
      Match the random and local masks at each width in active edge count and
      in/out-degree sequence where feasible, using degree-preserving rewiring;
      log the realized masks and do not pick one based on alignment. Keep
      sparse in-degree near the worker's value across widths. State which
      initialization policy is held fixed for each planned contrast before
      enumeration: neither input nor recurrent scaling may change silently
      with the connectivity label. Audit compatibility of existing checkpoints
      and the prepared local289 initialization; if they cannot belong to one
      factorial, keep them as separate budget controls and explicitly identify
      the additional matched runs needed. Do not claim a complete factorial
      assembled from inconsistent initialization policies.
      Compare widths within connectivity, dense versus random sparse within
      width, random versus local sparse within width, and T within every cell;
      report their interactions. Connection-count changes in the dense/sparse
      contrast are part of the intervention. Compare hierarchy to the existing
      budget control separately. Preserve training/evaluation schedules, seed
      pairing, all applicable analyses and ROI/population coverage. This scope
      does not cross every biological mechanism or training signal.
      Implemented in `fd44b95`, verified 2026-09-13: all 96 factorial ids and
      the 16 budget-control ids enumerate uniquely with zero manifest overlap;
      local and random masks at each width share an identical edge count and
      degree sequence (2,517 edges at 128, 5,708 at 289) and differ only in
      topology; every factorial cell draws its recurrent weights at its own
      width, so initialization does not move with the connectivity label.
      `../training_commands.txt` section 6 carries the 80 commands and reports
      168 outstanding runs. LAUNCHED 2026-09-13 00:53 (see the launch record
      below).
- [x] A9 Multi-task versus WM-only continuation from matched saved
      checkpoints, using the implemented Sternberg, two-arm bandit, two-step
      decision, delayed-match-to-sample, go/no-go and context-decision tasks
      on one shared recurrent model. Prerequisites, in order, before any
      continuation is prepared:
      (a) verify every task's observation/action/reward contract, state
          reset, delayed reward credit, and above-chance learning;
      (b) audit the non-WM loop's missing reflective input and topographic
          loss, and make mechanism scope explicit and consistent;
      (c) encode task identity inside the existing ten-entry cue -- indices
          [0,8,9] as Sternberg 100, bandit 010, two-step 001, delayed match
          110, go/no-go 101, context decision 011 -- keeping indices 1..7 for
          the auxiliary WM flag, encoding position and WM epoch, zeroed on
          other tasks. Persist the schema version and named task mapping with
          checkpoints; keep the existing thirteen-entry reader for any
          checkpoint written under it. Task labels must never encode the
          correct action or future outcome. Before continuing any checkpoint,
          verify that WM cue vectors, noiseless forward states and action
          outputs are unchanged, that all six codes are distinct, and that
          both arms use identical input interfaces and adapters.
      Then branch each selected checkpoint into WM-only and multi-task
      continuation with identical starting parameters, seeds, optimizer
      restart policy, shared modules and evaluation trials. Match total
      additional optimizer updates for the primary comparison and report
      task-specific exposures; equal-WM-exposure needs its own budget and is
      not obtained by interleaving. Advance the WM curriculum by WM exposure
      only. Select representative flat/hierarchical and mechanism cells in
      advance. Evaluate WM and auxiliary behaviour and every applicable
      existing analysis on the same recorded WM trials, keeping signal, seed,
      region, load, population and patient coverage explicit. Label the
      result as an effect of additional multi-task training, never as
      multi-task training from initialization. Prepare unique commands and
      run ids; no full-grid restart.
      Closed 2026-09-17. (a) The contract report passes with 0 failures over
      all six tasks -- distinct ten-entry codes, the WM cue unchanged at all
      six epochs, every observation reaching the same (4, 64) bottleneck, one
      shared 3-way head mapped onto each native action space, seed-reproducible
      per-trial reset, and reward credited to the action that earned it. All
      five auxiliary tasks learn above their own measured random-policy reward
      within 400 updates (bandit 0.898 vs 0.369, dawtwostep 0.694 vs 0.264,
      delaymatchsample 0.498 vs 0.316, gonogo 0.516 vs 0.233,
      contextdecisionmaking 0.283 vs 0.165). (b) The auxiliary loop was missing
      the reflective gate and the topographic penalty and credited only the
      tick that received a reward; all three are fixed and mechanism scope is
      stated at the loop (`cd6975e`). (c) The ten-entry cue landed in
      `a8c0718`. Continuation enumeration verified: 64 unique run_ids, none
      colliding with the manifest, all 64 source checkpoints present, batch
      size following the configured value so each arm matches the run it
      extends (`c3625a6`). Two scope facts recorded with the arm: bandit and
      dawtwostep resolve in 1-2 ticks and exercise almost no recurrent
      dynamics, and bandit exposes no ground-truth target. Commands are in
      `../training_commands.txt`; nothing was launched.

## Launch record

All 168 outstanding runs were launched 2026-09-13 00:53 (+03:30) on user
authorization, after the gate passed: `make verify-gpu` clean, full
`pytest -q` 486 passed, `make audit-campaign` 0 violations, all 168 run ids
re-enumerated unique and disjoint from the manifest's 266, and the new flat
cores' peak GPU memory measured rather than estimated (worst case 216.9 MiB
allocator peak plus a 196 MiB CUDA context, so eight concurrent runs occupy
~3.3 GiB of the 12,227 MiB card and `--workers 8` binds before the memory
gate does).

The fourteen prepared invocations run one at a time under a driver, in the
order the command inventory lists them -- rate substrates (48), memory-demand
arm (24), locality-matched flat control (16), crossed flat controls (80) --
so no two invocations contend for the card while each fills it with its own
worker pool. Per-invocation logs are `logs/<name>.log`; the driver's own
start/exit ledger is `logs/campaign_driver.log` and its pid is in
`logs/campaign_driver.pid`.

The uncommitted working-tree change to `train.py` at launch time was confined
to `run_multitask_neurogym_trial`, which the Sternberg grid never calls, so
every launched run uses committed code on its own path.

## B. Measurement repairs

- [~] B1 The equal-performance comparison has no data: the criterion
      checkpoints were never aligned, and the shard merger has no suffix
      support. `scripts/run_perf_matched_baselines.py` also constructs
      unsuffixed run ids, so SUP and RL variants can collide or be parsed as
      the wrong training signal. Add the signal suffix to those ids and every
      matching reader, add `--suffix`, and add a resumable criterion pass to
      `analysis.sh` with its own shard root. Test enumeration, resume, manifest,
      checkpoint, replay, and analysis lookup for both signals. Do not run the
      sweep.
- [~] B2 Standardized persistence means are on the order of hundreds.
      Diagnose the cause, fix the estimator, add a regression test. The
      corrected estimator is in the working tree uncommitted; the saved
      `results/dynamics_persistence.csv` still holds the old cross-scale
      values and must be regenerated and retired, not reinterpreted.
- [ ] B3 Probe-alignment task-structure baselines: finish the out-of-sample
      semipartial wiring and report it per region.
      `results/task_structure_baselines.csv` does not exist yet.
- [ ] B4 Maintenance sensitivity curve and epoch-by-method 2x2: is the
      maintenance/probe gap the epoch or the estimator? The 2x2
      implementation exists; its result table does not.
- [ ] B5 Full-delay maintenance RSA. For each recorded trial, define the
      neural maintenance window from `t_maintain` up to, but not including,
      `t_probe`; bin spikes at the existing 50 ms resolution and average the
      available duration to obtain one neural vector per trial. Define each
      rate as total spikes in the half-open window divided by its exact
      duration; if the final bin is partial, weight by bin duration rather
      than averaging it equally with full bins. Exclude malformed windows and
      document their counts. Use the existing maintenance condition labels,
      within-load crossnobis RDM builder, condition-level RSA within each
      session, noise-ceiling calculation, patient-clustered inference, and
      aggregation code. This uses the whole recorded delay instead of a fixed
      1.5 s slice; it does not align model steps to seconds -- the model's 15
      maintenance states are still averaged as they are. Report trial
      delay-duration and bin-count distributions, excluded or malformed
      trials, usable conditions and sessions, and raw and normalized RSA for
      every existing region, training signal, cell and seed, worker and
      manager separately wherever applicable. Compare directly against the
      existing fixed-window RSA on the same included trials, with paired
      per-session differences and patient-clustered intervals. Acceptance
      requires tests showing that no spike at or after `t_probe` enters
      maintenance, that variable trial lengths produce the correct epoch-mean
      vectors, and that the fixed-window path is unchanged, plus the complete
      result and comparison tables in `executor.md`.
- [ ] B6 State the seed/session/patient/region estimand and add equal-patient
      maintenance summaries. For the mean alignment of independently trained
      networks, compute RSA separately for each seed, session and region
      first; averaging seed RDMs first estimates an ensemble instead. Average
      a patient's sessions, then patients with equal weights; compare against
      the existing equal-session estimate, which is a different estimand, not
      an error. Match bootstrap weights to the point estimate and pair seeds
      and available neural observations across cell contrasts. Seed and
      patient arithmetic means commute in a complete equally weighted table;
      specify missing-data support explicitly. Keep named regions separate and
      do not choose a subject's best region on the tested alignment values.
      Preserve probe pseudopopulation RSA as its own estimand; do not label it
      a mean of patient correlations. Acceptance requires an unequal-session
      synthetic example, coverage counts and paired differences, and explicit
      evidence of whether weighting changes any scientific comparison.
- [ ] B7 Timestamp quality control on the recorded sessions. Expand
      `docs/tutorial_timing_audit.json` to per-session and per-load duration
      distributions, separating stimulus exposure, inter-image gaps, response
      latency and analysis windows. Quarantine and document the one
      nonpositive maintenance duration in 000469 explicitly, verify that the
      replay and analysis paths already exclude malformed events, and report
      the affected rows. Rerun only the timing-sensitive analyses, and only if
      the quarantine changes them. No retraining: epoch durations are not
      required to match between the model and the recordings.

## C. Science

- [~] C1 Worker and manager analyzed separately, never pooled. Per-population
      is the primary DV for every hierarchical cell, each population against
      every region. Flat-vs-hierarchical runs twice (flat vs worker, flat vs
      manager), unit-count matched by subsampling. The pooled numbers in
      `results/` are superseded, not deleted. Partly wired in the working tree;
      `results/alignment_contrasts.csv` still has no population column.
      A7b's 289-unit flat control is not unit-matched to the worker either,
      so subsampling stays the rule for it; what it removes is the capacity
      and sparsity confound, not the width one.
- [~] C2 Every contrast under both training signals, with coverage stated per
      contrast and unlicensed rows flagged with a reason, not dropped.
- [x] C3 Partial rank correlation of neural and model RDMs given the frozen
      encoder's RDM, per region, with intervals. Region ranking is the
      output -- `4ecfc70`, extended with a maintenance-epoch-selected third
      run in a follow-up commit (`scripts/rsa_encoder_partial.py`). The two
      alignment-selected runs give the identical ordering; the
      accuracy-selected run does not.
- [x] C4 Memory-location intervention in the plastic cells -- `c8e557d`.
      1,080 accuracy rows over 72 checkpoints x 5 conditions x 3 loads, plus
      1,296 carrier-deviation rows. Selectivity holds for the synaptic
      disruption (off/on 0.039 during the delay) and for the activity
      disruption placed before encoding (0.107); the delay-activity condition
      reads 0.72 because holding the trace freezes it while the unlesioned
      trace keeps being written -- drift, not injected corruption, and the
      free-trace variant moves it only to 0.78. Recorded as a limitation of
      that condition; no rows regenerated. Both carriers hold memory
      (0.15-0.46 accuracy cost during the delay against 0.00-0.05 before
      encoding); the eight hierarchical reinforcement checkpoints are at
      chance and their zero drop is a floor.
- [~] C5 Does maintenance alignment predict robustness to longer delays,
      distractors and novel stimulus combinations, among accuracy-matched
      networks, on conditions unused for selection. `results/robustness.csv`
      exists but has only 11 runs, and its predictor column
      `maintenance_normalized_alignment` is exactly 0.0 in 9 of the 11 --
      the normalize-then-clip step maps every negative alignment to zero, so
      the regression has almost no predictor variance. Repeat against
      `maintenance_signed_raw_alignment`, which does vary in the same file,
      state the accuracy-matched sample size, and report the null honestly if
      it stays null at this n rather than quietly reporting the clipped fit.
- [x] C6 Survey `/media/amin/ADATA HD710 PRO/Research/Representation/Working
      Memory/data` for an independent confirmation set. Present are 000004,
      000469, 000574, 000673, 001187, ds004752, ds005034, ds005489, ds005557,
      ds006848, Panichello_2024, Wolff, Watters, Inagaki, Campbell,
      Soldado-Magraner, kai miller, PFC-3, alagapan, CLAM-tACS. Only propose
      a download if a planned analysis actually needs one.
      Surveyed 2026-09-16. The Daume 2024 medial temporal lobe
      working-memory maintenance recordings (20 GB, 46 files) are already on
      the volume and are the independent confirmation set: separate patients,
      separate acquisition, same task family and file standard. The verbal
      Sternberg collection is a different stimulus modality and the
      declarative-memory collection has no maintenance delay, so neither
      confirms this claim. No download is proposed. Wiring the confirmation
      set through the existing reader is a measurement item, not a training
      one.
- [~] C7 Distributions, not only pooled means: per load, per condition field,
      per region, per patient and their crossings, with counts behind every
      row and the spread reported, not just the centre. Say which conditions
      carry the pooled effect and which cancel inside it.

## D. Housekeeping

- [x] D1 Full `pytest -q` and `make verify-gpu` -- 2026-09-13: 486 passed,
      `verify-gpu` clean (RTX 5070 Ti Laptop, torch 2.11.0+cu128, sm_120
      present), `make audit-campaign` 0 violations. One nondeterminism was
      found and fixed while clearing this: `measurement_validation`'s mixture
      generator seeded its RNG from `hash()` of a string, which Python salts
      per process, so the same nominal seed produced a different mixture every
      interpreter run and the monotonicity test failed for some values of
      `PYTHONHASHSEED` (113 and 157 reproduce it). Seeding now derives from a
      crc32 of the encoded key; the file's seven tests pass under every hash
      seed tried. No saved result depended on the old derivation.
- [x] D2 `advisor.md` entries for the September decisions.
- [x] D3 After the round lands: re-read `training_commands.txt` end to end and
      confirm every command still enumerates the run ids it claims.
      Rewritten 2026-09-17 around the continuation study, now the only
      outstanding training -- all 168 runs of the previous four sections are
      complete. Its verification, status and completion blocks were executed
      as written: 64 unique run_ids, 0 manifest collisions, 0 missing source
      checkpoints, 0/64 complete, completion check exits 1.
- [x] D5 `contrasts.enabled_arm_count` raised on any tagged `model_id`
      (`invalid literal for int() with base 10: '_'`), so the contrast stage
      would have broken on every pending run the first time one was analyzed.
      Fixed in `12fbfcb`: the count reads the five leading bits, and the
      exploratory arm-count correlation is restricted to the battery's own
      untagged cells, since a control variant and the local-learning family
      are separate designs rather than extra points on it.
- [x] D6 Arm D is applied ~3.3x more strongly to hierarchical cells than to
      flat ones. `_dale_penalty` returns a SUM of per-population means: one
      mean for the flat cell, worker mean plus manager mean for the
      hierarchical core. Measured at initialization with `dale_ei_split` 0.8:
      flat 0.0219 against hierarchical 0.0711 (worker 0.0181 + manager
      0.0530; the 24-unit dense manager's weights are large because its init
      uniform initialization bound is 1/sqrt(24)). So at the same nominal `dale_penalty_weight_on` the
      D arm is a different-sized intervention on the two architectures, in a
      battery whose purpose is to separate them. Affects every completed
      D=1 cell. Saved effects describe the implemented penalties; neither
      their signs nor magnitudes are guaranteed to survive a normalization
      change. Cross-S contrasts must state the different penalty scope.
      Averaging populations removes the population-count multiplier but does
      not equalize weight scale or gradient strength. State the limitation
      and specify the intended normalization before any targeted follow-up;
      do not prescribe a replacement D grid from the initial penalty ratio.
      Closed 2026-09-16. Re-measured over the eight seeds the study trains
      rather than one draw: flat 0.0221 (sd 0.0001) against hierarchical
      0.0692 (sd 0.0012), worker 0.0179 plus manager 0.0513, a ratio of 3.13.
      The limitation and the intended normalization -- weight each
      population's mean violation by its share of the core's surviving
      recurrent synapses -- are recorded, with the required annotation for
      any contrast that crosses the architecture arm while the penalty is on.
      No replacement grid is prescribed and no completed cell is changed.
- [x] D7 Recurrent initialization gain is not comparable across the S arm in
      the completed battery. Measured candidate-block spectral radius at
      init: flat dense 128 = 0.584, hierarchical worker 196 at density 0.10 =
      0.208. The mask is applied after a draw whose scale depends only on
      width, so a masked core starts with roughly a third of the dense core's
      recurrent gain. No fix is possible retroactively, and the GRU's
      additive path makes this far less severe than it would be for an
      ungated core, but it is a named alternative account of any S effect and
      must appear as one. A7b approximately matches candidate-weight gain to the worker, but
      also changes width and connectivity. It is a budget control, not an
      isolated hierarchy intervention; A7c addresses the separate contrasts.
      Closed 2026-09-16. Re-measured over eight seeds: candidate-block
      spectral radius 0.608 (sd 0.016) flat dense at 128 units against 0.201
      (sd 0.004) for the 196-unit worker at realized density 0.1005, a ratio
      of 3.03; the 24-unit dense manager is 0.606 (sd 0.045). Recorded as a
      named alternative account of any architecture effect.
- [x] D4 Stale flat-topography defaults corrected -- `312f14c`. The fallbacks
      in `train.py` and `scripts/probe_peak_memory.py` said 16x16; the
      configured and tested flat sheet is 16x8 at 128 units, and every
      fallback now agrees with `model.flat_grid`. Completed run behaviour is
      unchanged.

- [x] D8 Clarified the dPCA label in the status Markdown/TeX/PDF and module
      description. `analysis/dpca.py` marginalizes the condition tensor and
      runs ordinary PCA within each marginalization; it does not implement
      Kobak et al.'s full regularized encoder/decoder estimator. Preserve the
      saved marginal variance fractions as descriptive decomposition results,
      and do not label their axes as a validated full-dPCA fit. No model
      retraining is required for this documentation correction.

## E. Repository hygiene

The code must read standalone: a reader holding only the repository must
understand every name, comment and docstring without a second document. 562
pointers to internal review stages, spec sections, decision ids and this
project's private shorthand survive across 66 tracked Python files, plus
`configs/config.yaml`, the `Makefile` and the status Markdown. Each one is
removed by keeping its technical content and dropping the pointer -- the
sentence explains what the code does and why, never who asked for it.

The chronology and advisory documents are excluded: they are the internal
record and cite themselves by design.

- [ ] E1 `run_grid.py` (34 remaining). A pass already stripped the
      parenthetical pointers; what is left needs prose rewrites. Uncommitted
      in the working tree -- read it before editing. Verify with
      `tests/test_run_grid_concurrency.py`.
- [ ] E2 `brainalign_wm/training/train.py` (77). The largest single block.
      Verify with `tests/test_training.py`.
- [ ] E3 `brainalign_wm/analysis/run_all.py` (18) and
      `brainalign_wm/training/generate_activity_logs.py` (14). Both carry
      other agents' uncommitted work -- coordinate before touching.
- [ ] E4 Tests: `tests/test_run_grid_concurrency.py` (14),
      `tests/test_training.py` (12), `tests/test_scaffold.py` (9),
      `tests/test_tasks.py` (6), `tests/test_models.py` (5) and the smaller
      remainder. Test names and docstrings state the behaviour under test.
- [ ] E5 `scripts/` (about 90 across 14 files, densest
      `scripts/run_phase11_pilot.py` at 13). `run_phase11_pilot.py` is also a
      filename keyed to a phase number; rename it for what it runs and update
      every reference, including the repository map.
- [ ] E6 `brainalign_wm/models/` and `brainalign_wm/tasks/` (about 35).
- [ ] E7 `configs/config.yaml` (56), `Makefile` (2) and
      `PROJECT_IMPLEMENTATION_STATUS.md` (1). Config comments describe what a
      key controls and the evidence for its value, not the review that set it.
      Key names themselves stay unchanged -- a rename breaks every saved
      resolved config.
- [ ] E8 Final sweep: re-grep the whole tree, confirm zero remaining, run
      `pytest -q` and `make audit-campaign`.

## Priority and order

Work is serialized: one agent at a time, each finishing and committing before
the next starts, so a session interruption loses at most one item.

1. `../training_commands.txt` -- done 2026-09-17. The 64-run continuation
   study is the only outstanding network training and its commands are
   verified and resumable.
2. C4, C5 -- a finished sweep needs only its selectivity check and result
   table, and a null needs reporting honestly.
3. C1, C2, C7 -- the per-population dependent variable, coverage under both
   signals, and distributions behind the pooled means.
4. B1, B2 -- the equal-performance run ids and the persistence estimator.
   Both have corrected code sitting uncommitted.
5. B3, B4 -- two result tables whose implementations already exist.
6. B5 -- full-delay maintenance RSA, the largest measurement item.
7. B6, B7 -- estimand and timestamp quality control.
8. E1-E8 -- hygiene, last, because every item above edits the same files.
