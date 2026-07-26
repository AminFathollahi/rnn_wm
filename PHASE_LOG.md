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
