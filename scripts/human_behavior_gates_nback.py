#!/usr/bin/env python3
"""Derive human-behavioral n-back gates from the raw Kai Miller ECoG dataset
(`data/kai miller/memory_nback`, comments.txt §9.1(2)) -- for the same
reason `human_behavior_gates.py` reads Sternberg gates off real human
trials instead of a literature number or a round placeholder: "trained" for
Stage 3 n-back should mean "performs like the human patients on THIS task,"
not an arbitrary ceiling accuracy.

Restricted to n in {1, 2} ("1-back"/"2-back") -- not 0-back (a simple
target-detection control, not really an n-back memory comparison) and not
3-back (not run in this dataset at all).

DATA FORMAT (`README_memory_nback_dataset_notes.docx`, docs unzipped and
read directly): each `data/<pid>_nback.mat` has continuous (not
per-trial) sample-indexed arrays -- `stim` (which house image, 0=blank),
`task` (-1 baseline / 0 / 1 / 2 = which n-back level), `target` (0 blank,
1 non-target cue, 2 targeted/match cue), `response` (an analog
[time x n_fingers] dataglove trace, "in the form of a finger flexion").
Only 2 of 4 patients have a usable `response` signal (comments.txt's
"two of the four have no behavioural responses recorded at all"),
confirmed here, not assumed:
  - AL: no `response` key in the file at all.
  - UG: `response` key present but degenerate (6-12 unique values total,
    essentially flat/saturated -- matches the docx's own note that UG's
    dataglove closures were not actually recorded).
  - CA, CC: genuine multi-hundred-unique-value analog traces (docx: "had
    all elements of the task run appropriately").
Only CA and CC are used below.

TRIAL EXTRACTION: each maximal contiguous run of `target != 0` within a
`task` segment is one stimulus epoch (verified: exactly 100 per task
level per patient, stimulus shown for 600 samples then a 1601-sample ISI
-- matches the docx's "50 stimuli per run" x 2 runs). `target==2` epochs
are targets (correct response = flex); `target==1` are non-targets
(correct response = no flex).

FLEX DETECTION: a per-epoch, per-channel rise (window max minus the mean
of the 200 samples immediately preceding onset) exceeding 20% of that
channel's own whole-recording (max-min) range, in a window from onset to
onset+1600 samples. A GLOBAL median/MAD threshold was tried first and
rejected: CC's baseline drifts across the ~15-minute recording, so a
global z-score threshold missed real, visually obvious flexes (window max
several hundred units above the LOCAL pre-stimulus baseline) because the
global MAD is inflated by that drift. The local/relative version gives
a clean, monotonically-decreasing-with-difficulty accuracy profile for
both CA and CC independently (0-back near ceiling, 2-back the hardest,
matching the docx: "This was often very difficult for our patients") --
that internal consistency is the validation for this heuristic, the same
role monotonicity-in-load plays for the Sternberg gate script.

"accuracy" per epoch = correct response (hit on target, correct-rejection
on non-target) -- the same "blended" definition `response_accuracy`
already uses for Sternberg's match/non-match trials, kept for consistency
with `human_behavior_gates.py`'s gate semantics.

    python scripts/human_behavior_gates_nback.py

Writes `results/human_behavior_nback.csv` (one row per patient x n-level).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brainalign_wm.config import get_path  # noqa: E402
from scripts.human_behavior_gates import wilson_ci  # noqa: E402

DEFAULT_DATA_ROOT = get_path("data_root") / "kai miller" / "memory_nback" / "memory_nback" / "data"
USABLE_PATIENTS = ["ca", "cc"]  # al: no response field; ug: degenerate response field (see module docstring)
WINDOW = 1600         # samples from epoch onset to allow a response in
PRE_BASELINE = 200    # samples before onset used as the local baseline
FLEX_FRAC_OF_RANGE = 0.2  # per-channel rise threshold, as a fraction of that channel's own (max-min)


def _find_epochs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def score_patient(mat_path: Path) -> pd.DataFrame:
    import scipy.io as sio

    d = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    task, target = d["task"], d["target"]
    resp = np.atleast_2d(d["response"])
    rng = resp.max(axis=0) - resp.min(axis=0)
    thresh = FLEX_FRAC_OF_RANGE * rng

    rows = []
    for n in (1, 2):
        hits = misses = false_alarms = correct_rejections = 0
        for s, _e in _find_epochs((task == n) & (target != 0)):
            win_end = min(s + WINDOW, len(target))
            local_base = resp[max(0, s - PRE_BASELINE):s].mean(axis=0) if s > 0 else resp[s:s + 1].mean(axis=0)
            rise = resp[s:win_end].max(axis=0) - local_base
            responded = bool((rise > thresh).any())
            if target[s] == 2:  # targeted cue
                hits += int(responded)
                misses += int(not responded)
            else:  # non-target cue
                false_alarms += int(responded)
                correct_rejections += int(not responded)
        n_correct = hits + correct_rejections
        n_total = hits + misses + false_alarms + correct_rejections
        lo, hi = wilson_ci(n_correct, n_total)
        rows.append({
            "n": n, "n_trials": n_total, "n_correct": n_correct,
            "accuracy": n_correct / n_total if n_total else float("nan"),
            "ci_lo": lo, "ci_hi": hi,
            "hit_rate": hits / (hits + misses) if (hits + misses) else float("nan"),
            "cr_rate": correct_rejections / (false_alarms + correct_rejections)
            if (false_alarms + correct_rejections) else float("nan"),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    args = ap.parse_args(argv)
    data_root = Path(args.data_root)

    all_rows = []
    for pid in USABLE_PATIENTS:
        path = data_root / f"{pid}_nback.mat"
        if not path.exists():
            print(f"[nback-human] {pid}: not found at {path}; skipping.")
            continue
        df = score_patient(path)
        df.insert(0, "patient", pid)
        all_rows.append(df)
        print(f"[nback-human] {pid}: " + ", ".join(
            f"n={r.n} acc={r.accuracy:.3f} (hit={r.hit_rate:.3f}, cr={r.cr_rate:.3f}, n_trials={r.n_trials})"
            for r in df.itertuples()
        ))

    if not all_rows:
        print("[nback-human] no patients scored; data root unreachable?")
        return 1
    df = pd.concat(all_rows, ignore_index=True)

    out = get_path("results") / "human_behavior_nback.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[nback-human] wrote {out} ({len(df)} patient x n rows)\n")

    print("=" * 70)
    print("PROPOSED n-BACK GATES (paste into configs/config.yaml gates.nback)")
    print("=" * 70)
    print("  # Derived from the 2 Kai Miller ECoG patients (CA, CC) with a usable")
    print("  # dataglove response signal -- scripts/human_behavior_gates_nback.py,")
    print("  # results/human_behavior_nback.csv. AL has no response channel; UG's")
    print("  # is degenerate (see script docstring). n=0 and n=3 excluded: 0-back")
    print("  # is a target-detection control, not a memory comparison, and 3-back")
    print("  # was not run in this dataset. `criterion` mirrors gates.criterion's own")
    print("  # convention (median per-patient accuracy, i.e. median 'session'), not a")
    print("  # pooled number -- only 2 patients, so this is the same median-of-N idea")
    print("  # at N=2 rather than the ~20-session N Sternberg's gate uses.")
    print("  criterion:")
    import math
    for n, g in df.groupby("n"):
        median_patient_acc = float(g.accuracy.median())
        n_correct, n_total = int(g.n_correct.sum()), int(g.n_trials.sum())
        pooled_acc = n_correct / n_total
        lo, hi = wilson_ci(n_correct, n_total)
        print(f"    n{n}: {math.floor(median_patient_acc * 100) / 100:.2f}"
              f"    # median patient accuracy {median_patient_acc:.4f}; pooled {pooled_acc:.4f} "
              f"[{lo:.3f},{hi:.3f}], n={n_total} trials, {g.patient.nunique()} patients")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
