"""comments.txt §20.6: the only pure-logic piece of the D37 sign diagnostic
worth a standalone test is the tick<->bin matched-position arithmetic --
everything else in the script is thin glue over already-tested RSA/rdm
machinery and requires real Tier-A data to exercise meaningfully."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_maintenance_sign import _matched_bin_indices


def test_matched_bin_indices_spans_the_full_bin_range_monotonically():
    idx = _matched_bin_indices(n_ticks=15, n_bins=30)
    assert len(idx) == 15
    assert idx[0] == 0
    assert idx[-1] == 29
    assert idx == sorted(idx)  # matched position must be monotone in tick


def test_matched_bin_indices_handles_equal_tick_and_bin_counts():
    assert _matched_bin_indices(n_ticks=5, n_bins=5) == [0, 1, 2, 3, 4]


def test_matched_bin_indices_handles_a_single_tick():
    assert _matched_bin_indices(n_ticks=1, n_bins=30) == [0]
