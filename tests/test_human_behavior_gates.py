"""Phase 12 (§3.1/9.8): `dedupe_sessions` must collapse a session that
appears in more than one dataset release (001187 re-releasing 000673
recordings) so a pooled quantile counts it once, not once per release.
Without this, the naive 111-row pool gives load-1 q10=0.8222 instead of
the correct 92-session 0.8344 -- a real, silent shift in the study's
inclusion gate."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.human_behavior_gates import dedupe_sessions, session_quantiles


def test_dedupe_pooled_n_is_union_not_sum():
    a = pd.DataFrame([
        {"dataset": "000673", "session": "sub-10_ses-1_P68CS", "load": 1, "accuracy": 68 / 70},
        {"dataset": "000673", "session": "sub-11_ses-1_UNIQUE", "load": 1, "accuracy": 0.80},
    ])
    b = pd.DataFrame([
        {"dataset": "001187", "session": "sub-10_ses-1_P68CS", "load": 1, "accuracy": 68 / 70},  # re-release, same session
        {"dataset": "001187", "session": "sub-12_ses-1_OTHER", "load": 1, "accuracy": 0.90},
    ])
    combined = pd.concat([a, b], ignore_index=True)
    assert len(combined) == 4  # naive concat: sum, double-counts the shared session

    deduped = dedupe_sessions(combined)
    assert len(deduped) == 3  # union: 3 distinct sessions, not 4 rows
    assert set(deduped["session"]) == {"sub-10_ses-1_P68CS", "sub-11_ses-1_UNIQUE", "sub-12_ses-1_OTHER"}

    q = session_quantiles(deduped, 1)
    assert q["n"] == 3


def test_human_percentiles_survives_dataset_code_csv_roundtrip(tmp_path, monkeypatch):
    """`human_behavior.csv` stores dataset codes as "000469" etc; pandas'
    default int-inference on read_csv silently drops the leading zeros
    (469 != "000469"), which made the 000469-only load2/3 filter in
    `_human_percentiles` match zero rows every time the CSV was read back
    from disk (train.py never hit this in the same-process script run,
    only on reload -- the bug that made human_percentile_load{2,3} always
    null in Phase 12.1's own acceptance run)."""
    import pandas as pd

    from brainalign_wm.training import train as train_mod

    monkeypatch.setattr(train_mod, "RESULTS", tmp_path / "results")
    (tmp_path / "results").mkdir()
    pd.DataFrame([
        {"dataset": "000469", "session": "s1", "load": 1, "accuracy": 0.70},
        {"dataset": "000469", "session": "s2", "load": 1, "accuracy": 0.90},
        {"dataset": "000469", "session": "s1", "load": 2, "accuracy": 0.60},
        {"dataset": "000469", "session": "s2", "load": 2, "accuracy": 0.80},
    ]).to_csv(tmp_path / "results" / "human_behavior.csv", index=False)

    out = train_mod._human_percentiles({"load1": 0.80, "load2": 0.70, "load3": 0.80})
    assert out["load1"] == 0.5   # beats 1 of 2 sessions (0.70 <= 0.80)
    assert out["load2"] == 0.5   # beats 1 of 2 sessions (0.60 <= 0.70)
    assert out["load3"] is None  # no load-3 rows in this fixture
