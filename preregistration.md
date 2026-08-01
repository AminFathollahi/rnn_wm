# Pre-registration (freeze before the full training grid)

Copy the directional predictions and decision table from the master
specification's hypotheses section here, commit, and record the commit
hash. Do not edit after the full grid starts.

## Primary hypotheses (fill in / confirm from the specification)
- H1 (Structure -> anatomy): S=1 raises alignment; worker↔MTL, manager↔MFC dissociation.
- H2 (Modulation -> frontal control): M=1 raises MFC-epoch alignment; reflection-shuffle causal.
- H3 (Learning): L=1 ≥ BPTT alignment at matched behavior, *conditional on the gate*. Audit fix
  A1a (2026-07-05): L=0 now trains via REINFORCE (policy-gradient + value baseline) on the same
  sparse trial-end reward L=1 uses for the ramp+target curriculum phases, so L=0 vs L=1 differ
  ONLY in credit assignment there -- H3 is evaluated on that matched-signal regime. H3 remains
  well-defined ONLY for (S,M) cells where BOTH the L=0 and L=1 arms clear their behavioral gate
  (`load1>=0.95` and `load3>=0.80`); for any cell where L=1 fails to clear the gate even at rung 2,
  report this explicitly as "H3 undefined for this cell (L=1 did not clear the gate)" rather than
  comparing alignment at mismatched behavior.
- H4 (Target): M111 best; S×M interaction > 0.
- H5 (Oblique/dynamic): oblique regime more brain-aligned; stable+dynamic delay code. Wired via
  `analysis/dynamics_and_persistence.py::stability_index_for_session` (audit fix C3/C4), using LOAD
  as the shared decodable factor across maintenance timebins (item/category identity recurs too
  rarely per session for a stratified decode -- see RESPONSES.md's 2026-07-05 entry).
- H6 (Persistent activity): constrained models show more memoranda-selective persistence. Wired via
  `persistence_index_for_session` (audit fix C3/C4), same LOAD-indexed basis as H5.

## Primary region per hypothesis (avoid region-shopping)
- H1-worker / H6: MTL.   H2: MFC.

## Analysis lock
- Metric: crossnobis RSA (maintenance epoch, per-session category-multiset+load conditions --
  audit fix A2a/A2b, see RESPONSES.md) and pooled coarse-condition crossnobis (probe epoch, audit
  fix A2d), both noise-ceiling-normalized; inference by condition-label permutation.
- Model: align ~ S*M*L + accuracy, fit on POOLED-region rows only, with seed and patient as CROSSED
  MixedLM variance components (audit fix B3/R15 -- see `stats.py::mixed_effects_alignment`). A
  SEPARATE region-dissociation model (align ~ S*C(region) + accuracy, MTL/MFC rows only, session as
  the crossed factor) tests H1's worker<->MTL / manager<->MFC dissociation directly -- region levels
  are NOT mixed into the main model (would be pseudo-replication; audit fix R16). FDR across regions
  applied to the region-dissociation model's per-hypothesis p-values.
- Behavioral gate boundary: `>=` (not `>`) on `load1_acc`/`load3_acc` (audit fix B2).
- Seeds: >= 8 (target; actual grid seed count set by the E3 wall-clock budget, recorded in
  `results/resolved_config.yaml` and the manifest's `config_hash` for this run). Noise ceiling:
  per-session repeated-fold-resample reliability for the maintenance path (`within_session_noise_
  ceiling`; a cross-session LOSO ceiling is undefined for session-specific category conditions),
  pooled split-half reliability for the probe path (`pooled_noise_ceiling`) -- both audit fix A2c
  (raw alignment and its ceiling always computed on the SAME representation).
- H2 causal control: reflection-shuffle lesion (audit fix C1) compares normal vs. time-shuffled
  R_t maintenance alignment for every M=1 cell; wired into `run_all.py`'s standard output, not a
  one-off.
- H5 acceptance guard: chance/untrained-model negative control (audit fix, comments.txt Section H)
  wired into `run_all.py::chance_control_check`, re-verified on every pipeline run.

Frozen at commit: 92e059e  (date: 2026-07-05)

---

## Amendment (2026-07-06): H3/H4 framing (audit finding N4)

This file's header says "do not edit after the full grid starts." This
amendment is made anyway, appended (not rewriting the frozen text above),
under an EXPLICIT PI instruction (comments.txt, 2026-07-06 review, item
N4) issued specifically because two real defects were found in the
maintenance-epoch alignment DV (N1: the RDM pooled all loads together,
letting load rather than held-item identity dominate its structure; N2:
per-session rectification produced a spurious positive floor) -- both now
fixed (see `RESPONSES.md` Part 5). No grid seed beyond the already-
completed seed 0-1 dev-tier runs has been launched under this amendment;
H1/H2/H5/H6 and the analysis-lock section above are UNCHANGED. Full
before/after detail (numbers, code, tests) lives in `RESPONSES.md`, per
this project's one-file discipline for audit responses -- this section
only records the resulting decision for H3/H4's status per the
pre-registered rule ("H3 undefined for this cell" if the gate isn't met).

**Finding:** across all 16 completed seed 0-1 runs (8 cells x 2 seeds),
every L=1 (local-learning) cell sits in a narrow, robust 0.375-0.65
load1-accuracy band -- never once close to the `load1>=0.95` gate, at
EITHER seed, for ANY (S,M) combination, including the flagship M111. This
is a training-signal/budget finding, not touched by the N1-N3 analysis
fixes (which only changed how the maintenance alignment DV is computed
from already-trained checkpoints, not `train.py`/`local_learning.py`
themselves) -- so it is not expected to resolve merely from more seeds at
the SAME (dev-tier, 20k-step) budget.

**Decision:** per comments.txt N4's explicit recommendation, H3
("L=1 >= BPTT at matched behavior") is reframed NOW, before any further
seed compute is spent, as: *H3 records the gap rather than presupposing a
matched-behavior comparison exists.* Concretely: report L=1's accuracy
gap to BPTT for every cell (already done in `RESPONSES.md`'s per-seed
table) and continue to mark H3 "undefined for this cell (L=1 did not
clear the gate)" per the ORIGINAL pre-registered rule above -- no change
to that rule's text, just an explicit acknowledgment, ahead of time, that
it is expected to fire for every cell at dev tier. H4 ("M111 best")
inherits the same caveat, since M111 is itself an L=1 cell.

**What this does NOT decide:** whether to (a) spend the remaining 8-seed
budget on tighter CIs for the S/M effects and the now-fixed maintenance DV
(H3/H4 stay undefined, but H1/H2/H5/H6 gain power), or (b) redirect that
budget toward a single, higher-training-budget shot at the L=1 arms
specifically (the dev-tier 20k-step budget is almost certainly the
binding constraint, not the credit-assignment mechanism itself), is a
resource-allocation call for the PI, not implied by this amendment. Per
the user's explicit instruction accompanying this fix pass, NO grid
relaunch (seed 2 or beyond, nor a targeted L=1 run) has been made under
this amendment -- this section only settles the H3/H4 FRAMING question so
that whichever of (a)/(b) is chosen later, the grid is not implicitly
gated on an impossible matched-behavior comparison.

Also corrected here (documentation-only, matching code already in place
before this amendment): the "Noise ceiling" bullet above describes the
maintenance-path ceiling as "per-session repeated-fold-resample
reliability" -- this described the PRE-N3 implementation. Audit fix N3
(2026-07-06) replaced it with a genuine per-session split-half reliability
estimate (disjoint random trial halves, not fold-reshuffles of the SAME
trials); see `rsa.py::within_session_noise_ceiling` and `RESPONSES.md`
Part 5 for why the old version saturated near 1.0 and was not a valid
reliability estimate.

---

## Amendment (2026-07-06): Knob L retired from Core, replaced by Knob P (master protocol v5.0, §17 decision 6)

Per the master protocol's v5.0 revision (comments.txt item 3), the Core
factorial's third bit is no longer L (local-learning-as-training-
algorithm) but **P (synaptic plasticity, Hebbian fast weights, §6.2)** --
all 8 Core cells are BPTT-trained; no cell is training-algorithm-gated.
This directly resolves the finding in the amendment above (every L=1 cell
stuck at 0.375-0.65 load1-accuracy, never near the gate, at dev-tier
budget): P asks a different, always-trainable question (does a fast
synaptic memory trace improve brain alignment?) instead of gambling the
whole Core grid's interpretability on whether node-perturbation scales to
this task.

- **H3 is redefined** (was: "L=1 >= BPTT at matched behavior"; now: "P=1
  increases alignment via a maintenance-period signature consistent with
  activity-silent WM -- lower persistence-index, preserved/improved
  cross-temporal decoding, vs. matched P=0 cells"). The OLD H3 (now H3',
  Extended-tier only) continues as the local-learning mechanism study on
  `M00L/M01L/M10L/M11L` (§6.3) -- node-perturbation escalating through
  rung 2 to rung 3 (e-prop, comments.txt item 4) -- reported on rung
  reached + accuracy, no longer gating Core interpretability.
- **H4** ("M111 best") is unaffected in form (still "the fully-constrained
  cell is best") but M111 now means S=1,M=1,P=1 (Hebbian, BPTT-trained),
  not S=1,M=1,L=1 (node-perturbation) -- a cell that, unlike its
  predecessor, is expected to actually clear the behavioral gates.
- **Analysis lock**: `align ~ S*M*L + accuracy` -> `align ~ S*M*P +
  accuracy` (`stats.py::mixed_effects_alignment`); `run_all.py` treats
  `M**L` model_ids as a separate, unrelated string (not parsed as S/M/P)
  and excludes them from this model entirely.
- Existing seed-0/1 (and partial seed-2) checkpoints for the four old L=0
  cells (M000/M010/M100/M110) carry over UNCHANGED as the new P=0 Core
  cells (P=0 recurrence is architecturally identical to the old L=0
  recurrence). The four old L=1 checkpoints carry over renamed to
  `M00L/M01L/M10L/M11L` for the local-learning study. Four NEW P=1 cells
  (M001/M011/M101/M111) are trained fresh.

This amendment does not reopen or re-litigate H1/H2/H5/H6, which are
unaffected by the L->P swap.

---

## Amendment (2026-07-26): single behavioural gate (comments.txt §3), digital-twin
predictions (C1-C4), and the multi-task/topology/Dale's-law predictions

Per comments.txt §10's explicit instruction, three additions, appended (not
rewriting the frozen text above).

### Gate update
Every earlier gate reference in this file (the `load1>=0.95`/`load3>=0.80`
pair used to frame H3, and any dev-tier ad hoc threshold) is superseded by
comments.txt §3's single, human-derived criterion:

    criterion: load1 >= 0.94, load2 >= 0.91, load3 >= 0.86
    consecutive_evals: 3

Derived from DANDI 000469 (the only dataset covering all three loads;
0.9444/0.9111/0.8667 median human session), not from any published
literature value — see `references.md` §5c. A run is "trained" when it
holds this criterion across 3 consecutive evaluations; there is no second
admission gate. This is the criterion Phase 11.1's pilot (S=0/S=1, seed 0)
was evaluated against, and the one Stage 1/2/3 grids are evaluated against
going forward.

### C1-C4: digital-twin predictions (comments.txt §0)
The companion project (`../wm_dynamics/PAPER_REPORT.tex`, nine datasets —
human single units, iEEG, ECoG, macaque PFC) established four findings the
RNN is tested for reproducing. Pre-registered here as directional
predictions, tested by Phase 8's analysis suite:

- **C1** (content/context rotation, tested by 8.1): the memorandum
  (content) axis rotates more than the task-context axis across the delay
  period, within the same units (companion-paper paired difference 0.102,
  p=0.008).
- **C2** (manifold dimensionality vs load, tested by 8.2): the maintenance
  manifold's participation ratio does NOT expand with load (companion-paper
  pooled slope 0.01, 95% CI [-0.10, 0.12], p=0.87).
- **C3** (single-trial identifiability, tested by 8.3): maintenance dynamics
  are identifiable only from single trials; trial-averaged means manufacture
  a spurious contraction not present in the single-trial ensemble.
- **C4** (dominant mode and control, tested by 8.4): the delay-period
  dynamics form a contracting flow with a single slowest-decaying direction,
  and alignment to that direction — not to random or context directions —
  predicts the causal effect of a simulated perturbation (companion paper:
  this held for real electrical stimulation).

If the RNN reproduces C1-C4, it stands as the simulator on which closed-loop
stimulation policies can be developed before touching a patient (§0). Each
prediction is falsifiable independently; failing one does not invalidate
the others.

### Multi-task diet: task-irrelevant decoding (Phase 5, [BASHIVAN24])
PRE-REGISTERED, from [BASHIVAN24]'s STSF/STMF/MTMF result: the multi-task
diet model retains decodable task-IRRELEVANT information (serial position,
category) at >85%; the WM-only diet model does not. Tested by the existing
task-irrelevant decoding analysis (Phase 8.7-8.8) run on both Stage 1 diet
arms.

### Network topology (Phase 8.9, [SHAKIBA26])
PRE-REGISTERED published ranges for the four topology metrics, reported per
trained cell and NOT collapsed into a single "brain-likeness" score (their
Fig 3 shows the four do not move together):
- entropy (Gaussian-KDE-100): ~3-6 random-like, ~0.6-1.5 intermediate,
  ~0.02-0.08 highly structured.
- modularity Q (Clauset-Newman-Moore): ~0.4-0.5 strongly modular, ~0.1
  weakly constrained.
- small-worldness sigma: ~1.5-2.5 robust small-world, ~1 random-like.
- assortativity r: functionally-initialized networks go disassortative
  (r < 0); spatially-constrained-but-randomly-initialized networks go
  assortative (r ~ 0.4-0.5).

### Arm D (Dale's law) cost prediction (Phase 8.9b/9.5, [SHAKIBA26])
PRE-REGISTERED: arm D (the Dale sign-constraint penalty) COSTS substantial
performance in this battery, growing with `dale_penalty_weight`, because
every Core cell initializes uniform(-1/sqrt(H), +1/sqrt(H)) — the
randomly-initialized condition under which [SHAKIBA26]'s Table 3 shows a
sign constraint is catastrophic (near-chance) unless rescued by a
biologically-derived weight initialization. `M00001_bioinit` (arm D plus a
log-normal, spectral-radius-0.95, mean-0.1 recurrent init, no connectome
data) tests whether initialization rescues the sign constraint here as it
does in [SHAKIBA26]. If D is instead nearly free in the plain battery, that
is evidence the penalty is too weak to be doing anything, not that Dale's
law is harmless — check `dale_penalty_weight` before drawing the latter
conclusion.

This amendment does not reopen or re-litigate H1/H2/H5/H6 or the P-arm
framing above; it adds the C1-C4/multi-task/topology/Dale's-law predictions
and updates the gate reference, per comments.txt §10.

---

## Amendment (2026-07-30): split gate (Gate A / Gate B), Stage 1 GO/NO-GO
(comments.txt Phase 12, §3)

Per comments.txt §10's explicit instruction. The single graded criterion
above (`load1>=0.94, load2>=0.91, load3>=0.86`, all three loads jointly) is
superseded by TWO logically distinct gates, which Round 1 had conflated:

- **Gate A** (§3.1, behavioural matching / inclusion): `load1 >= 0.83`,
  the deduplicated pooled q10 across all three sessions (n=92; see the
  9.8 correction below). Load 1 only — it is the only load all three
  Sternberg datasets (000469, 000673, 001187) share. Checked every
  `eval_every` steps, confirmed after `consecutive_evals=3` consecutive
  passes. Does NOT stop training.
- **Gate B** (§3.2, fixed equal-duration training budget): `gates.max_steps`,
  a single step count applied to every Stage 1 cell so runs are compared at
  matched training duration, not matched accuracy. Set in 12.6 from the
  vanilla-substrate pilot's (12.5) accuracy-plateau (or geometry-plateau,
  per 12.4) trace. `null` until then.

Per-milestone efficiency DVs (§3.3) — `steps_to_<key>_<threshold>`,
`trials_to_...`, `wall_s_to_...`, `joules_to_...`, one set per key in
`gates.criterion` union `gates.extra_milestones` — are pre-registered
descriptive measures, not a third gate; none of them stop training either.
`extra_milestones.load3: 0.80` (000469's own load-3 q10) is currently the
only entry beyond Gate A's own load1 threshold.

### 9.8 correction: 92 vs 111 sessions
The dedup q10 above (0.8344) replaces an earlier, wrong pooled figure of
0.8222 computed from the naive 111-row (undeduplicated) pool. 001187 is a
re-release of MTL recordings already present in 000673 — 19 session
identifiers are shared between the two datasets and were being counted
twice. `scripts/human_behavior_gates.py::dedupe_sessions` now collapses on
`(session, load)` before any pooled statistic is computed; the raw,
undeduplicated 243-row `results/human_behavior.csv` is kept on disk for
auditability, with dedup applied only at the point pooled quantiles are
derived. This is a corrected number, not a re-derivation from a different
choice of statistic — recorded here per comments.txt's instruction that a
changed gate number after data inspection must be documented, with its
reason, not silently swapped.

This amendment does not reopen or re-litigate H1-H6, C1-C4, or the
multi-task/topology/Dale's-law predictions above; it replaces the gate
mechanics only.

---

## Amendment (2026-08-01): recurrent initialization is now a logged variable
for the Stage-1 vanilla diagnostic (comments.txt §13.3)

The statement above ("every Core cell initializes uniform(-1/sqrt(H),
+1/sqrt(H))") stays true for the Core battery and is not changed by this
amendment. What changes: the flat-vanilla NO-GO reported at Phase 12.5 was
found (advisor audit, 2026-08-01) to confound gatedness with the recurrent
init's spectral radius, which had never been varied or recorded as an
explicit run parameter — every vanilla arm silently used whatever radius
`uniform(-1/sqrt(H), 1/sqrt(H))` happened to draw, measured at 0.616 +/-
0.011 over five seeds at H=128.

`VanillaRNNCell` (and `VanillaHRLCore`'s worker and manager, both built from
it) now accept an optional `recurrent_init_spectral_radius`, applied by
rescaling `weight_hh` after the existing draw, computed on the EFFECTIVE
(mask-applied) matrix. `None` (the default, and `configs/config.yaml`'s
`model.recurrent_init_spectral_radius: null`) leaves the draw bit-identical
to every prior run. A run dict may override it, following the exact pattern
already used for `substrate`. This exists solely to make Stage 1's vanilla
diagnostic (comments.txt §13.4) able to hold gatedness, initialization, and
training signal apart; it is not a change to any Core cell's default and
does not touch the GRU substrate.

This amendment does not reopen or re-litigate H1-H6, C1-C4, the gate
mechanics, or the multi-task/topology/Dale's-law predictions above; it adds
one explicit, logged, default-`None` initialization parameter to the
vanilla substrate only.
