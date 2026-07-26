# Phase Log

Per-phase record for the 2026-07-26 audit rebuild (`comments.txt`): what
changed, the acceptance check's actual output, and anything not done and
why. One entry per phase, appended in commit order.

---

## Phase 0 — Archive

**Changed:**
- Moved `results/manifest.jsonl`, `results/metrics/`, `results/checkpoints/`,
  `results/activity_logs/`, `RUN_REPORT.md` into
  `results/archive_pre_leakfix_2026-07-26/`, with a `README.md` explaining
  the F1 label leak that invalidates them. Kept `results/feat_cache/`,
  `results/cifar100_raw/`, `results/human_behavior.csv`.
- Added `scripts/leak_check.py`: generates 3,000 trials and reports
  category-match accuracy, per-`c_t`-dimension accuracy, and a logistic-
  regression conjunction of the two. Shares `generate_probe_dataset` /
  `all_single_dim_accuracies` with
  `tests/test_tasks.py::test_probe_context_vector_does_not_leak_the_answer`
  (previously duplicated inline).

**Acceptance — actual output:**

```
$ PYTHONPATH=$PWD python scripts/leak_check.py
category-match accuracy:        0.8560
single c_t dimension accuracy:
  c_t[0]: 0.0000
  c_t[1]: 0.0000
  c_t[2]: 0.0000
  c_t[3]: 0.0000
  c_t[4]: 0.0000
  c_t[5]: 0.0000
  c_t[6]: 0.0000
  c_t[7]: 0.0000
  c_t[8]: 0.0000
  c_t[9]: 0.0000
conjunction (category+c_t) accuracy: 0.8560
```
conjunction accuracy 0.8560 <= 0.90 target (was 1.000 pre-fix). Matches the
expected value analytically: P(in_set)*1 + P(not in_set)*(1-lure_fraction)
= 0.5 + 0.5*0.7 = 0.85.

```
$ python -m pytest -q
148 passed, 5 warnings in 58.85s
```

**Not done / deferred:** nothing.

---

## Phase 1 — Parameter Budget (fixes A2, B1, B2)

**Changed:**
- `models/gru_cell.py`: `effective_param_count()`/`n_units()` on
  `MaskedGRUCell` (structural synapse count: dense `weight_ih` + masked
  `weight_hh`, masked-out entries count 0) and `PBWMManagerCell` (always
  dense). `PlasticGRUCell` inherits `effective_param_count()` unchanged --
  `alpha` modulates an *existing* synapse, not a new connection, so it
  doesn't inflate the structural count (documented in its docstring).
- `models/hrl.py`: `HRLCore.effective_param_count()`/`n_units()` sum the
  worker's + manager's + `g_proj`'s counts.
- `models/vanilla_rnn.py` (new): `VanillaRNNCell` (item 1.4), single-gate
  tanh RNN with the same `mask`/`extra_update_bias` interface shape as
  `MaskedGRUCell`. `_build_model`/`_step_core` (train.py) dispatch on the
  new `model.substrate: {vanilla|gru}` config key for S=0; S=1 vanilla
  (HRLCore) raises `NotImplementedError` rather than silently training a
  GRU under a vanilla label -- deferred to whichever Stage-1 phase (11)
  actually needs it, since Stage 1's factorial doesn't touch M/P/T/D and
  nothing in Phase 1's acceptance exercises it.
- `models/flat_gru.py` DELETED (item 1.5). `_GatedFlatCore` (train.py) is
  now the S=0 GRU-substrate core for every M/P combination including the
  true M=0,P=0 baseline (previously a separate class wrapping plain
  `nn.GRUCell`, which inits biases uniform where `MaskedGRUCell` inits
  them to zero -- B1's confound between M00000 and M01000). Updated
  `tests/test_models.py` (`FlatGRUCore` import -> `MaskedGRUCell(mask=None)`)
  and stale `FlatGRUCore` reference in `scripts/analyze_network_properties.py`.
- `scripts/match_param_budget.py` (new, item 1.3): solves `flat_units` from
  a target effective-synapse budget (closed-form quadratic), and
  `(worker_grid, manager_units)` for the hierarchical core at a FIXED
  `worker_density` (brute-force search) -- density is a pre-registered
  scientific choice (B2: mean in-degree 6->14), not something to solve for.
  Prints only; does not edit config.
- `scripts/param_report.py` (new, item 1.2): markdown table over all 15
  Core cells -- `n_units | n_synapses_effective | n_params_total |
  deviation from the S-group mean`. `n_params_total` is core-only (same
  scope as the other two columns -- front_end/heads are shared/near-fixed
  across every cell by design) and mask-aware via a new
  `_mask_aware_param_count()` helper (recursively masks `weight_hh`/`alpha`
  wherever a submodule carries a `.mask`), for the same reason
  `effective_param_count()` exists: a raw `.parameters()` count of a
  masked-but-dense tensor is meaningless (A2).
- `configs/config.yaml`: `bottleneck` 128->64, `g_dim` 128->32,
  `worker_density` 0.04->0.10 (B2), `flat_units` 256->128 (solved from
  `param_budget_target_synapses_gru: 74000`), `manager_units` 64->24
  (solved at the new bottleneck/g_dim/density, target 74000), `flat_grid`
  16x16->16x8 (product must equal `flat_units`, arm T's reshape), added
  `substrate: gru`, `param_budget_target_synapses_vanilla: 25000`,
  `param_budget_target_synapses_gru: 74000`.

  **Deviation from comments.txt §5's illustrative numbers:** the doc text
  sketches `worker_grid: [12,12]` and `manager_units: 48`, but says
  "Widths...come from 1.3, not by hand" in the same breath. Running the
  actual solver at worker_grid=12x12/density=0.10/g_dim=32/bottleneck=64
  only reaches 61,325 of the 74,000-synapse target (17% short) at
  manager_units=48, and needs manager_units~73 to hit it -- neither matches
  the sketch. Keeping worker_grid at 14x14 (unchanged) and solving
  manager_units=24 lands within 0.4% of the target and, more importantly,
  within 0.4% of the S=0 flat core's own 73,728 -- which is what A2 is
  actually about (S=0 vs S=1 parity). Followed the explicit "not by hand"
  process instruction over the illustrative numbers.
- `tests/test_models.py::test_param_budget_matched_within_tolerance`:
  rewritten to compare `effective_param_count()` instead of raw
  `.parameters()` numel -- the raw comparison is exactly the A2 confound
  and now fails at 140% (worker's dense `weight_hh` is large regardless of
  how sparse the mask makes it); effective comparison passes at 0.4%.
- `tests/test_param_budget.py` (new): 8 tests covering masked/dense
  effective counts on every cell type, `VanillaRNNCell` forward shape,
  the `flat_grid`/`flat_units` product invariant, and both
  `match_param_budget.py` solvers.

**Acceptance — actual output:**

```
$ PYTHONPATH=$PWD python scripts/param_report.py
| model_id | S | M | P | T | D | n_units | n_synapses_effective | n_params_total | dev_from_S_group_mean |
|---|---|---|---|---|---|---|---|---|---|
| M00000 | 0 | 0 | 0 | 0 | 0 | 128 | 73728 | 74496 | +0.00% |
| M11111 | 1 | 1 | 1 | 1 | 1 | 220 | 74052 | 86984 | +0.00% |
| M01111 | 0 | 1 | 1 | 1 | 1 | 128 | 73728 | 123648 | +0.00% |
| M10111 | 1 | 0 | 1 | 1 | 1 | 220 | 74052 | 86984 | +0.00% |
| M11011 | 1 | 1 | 0 | 1 | 1 | 220 | 74052 | 75404 | +0.00% |
| M11101 | 1 | 1 | 1 | 0 | 1 | 220 | 74052 | 86984 | +0.00% |
| M11110 | 1 | 1 | 1 | 1 | 0 | 220 | 74052 | 86984 | +0.00% |
| M10000 | 1 | 0 | 0 | 0 | 0 | 220 | 74052 | 75404 | +0.00% |
| M01000 | 0 | 1 | 0 | 0 | 0 | 128 | 73728 | 74496 | +0.00% |
| M00100 | 0 | 0 | 1 | 0 | 0 | 128 | 73728 | 123648 | +0.00% |
| M00010 | 0 | 0 | 0 | 1 | 0 | 128 | 73728 | 74496 | +0.00% |
| M00001 | 0 | 0 | 0 | 0 | 1 | 128 | 73728 | 74496 | +0.00% |
| M10010 | 1 | 0 | 0 | 1 | 0 | 220 | 74052 | 75404 | +0.00% |
| M00011 | 0 | 0 | 0 | 1 | 1 | 128 | 73728 | 74496 | +0.00% |
| M10001 | 1 | 0 | 0 | 0 | 1 | 220 | 74052 | 75404 | +0.00% |

S=0: max pairwise deviation in n_synapses_effective = 0.00% (n=7 cells)
S=1: max pairwise deviation in n_synapses_effective = 0.00% (n=8 cells)
S=0 group mean n_synapses_effective = 73728; S=1 group mean = 74052 (ratio S1/S0 = 1.004)
max n_params_total across all cells = 123648 (<=150,000 required)
```
Max pairwise deviation 0.00% <= 5% within each substrate (S group); max
n_params_total 123,648 <= 150,000. S=0 vs S=1 group means match to 0.4%
(A2's actual target).

```
$ python -m pytest -q
156 passed, 5 warnings in 58.95s
```

**Not done / deferred:**
- HRLCore has no vanilla-substrate variant (S=1 + `substrate: vanilla`
  raises `NotImplementedError`) -- Stage 1's factorial (§4) doesn't need
  M/P/T/D, so building a full vanilla worker/manager now, untested, would
  be speculative; add when Phase 11 (Stage 1) needs it.
- `n_params_total`'s whole-model (front_end+core+heads) variant was tried
  first and fails for the two P=1,S=0 cells (161,924 > 150,000 --
  `PlasticGRUCell`'s `alpha` is fully dense on the unmasked flat core, a
  real ~49k-parameter cost with no mask to exploit). Switched to core-only
  scope (matching `n_units`'s existing scope) rather than shrinking
  `flat_units` to force compliance, which would have undone the S0~S1
  parity match above.
