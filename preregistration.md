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
  rarely per session for a stratified decode -- see DECISIONS.md's 2026-07-05 entry).
- H6 (Persistent activity): constrained models show more memoranda-selective persistence. Wired via
  `persistence_index_for_session` (audit fix C3/C4), same LOAD-indexed basis as H5.

## Primary region per hypothesis (avoid region-shopping)
- H1-worker / H6: MTL.   H2: MFC.

## Analysis lock
- Metric: crossnobis RSA (maintenance epoch, per-session category-multiset+load conditions --
  audit fix A2a/A2b, see DECISIONS.md) and pooled coarse-condition crossnobis (probe epoch, audit
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
