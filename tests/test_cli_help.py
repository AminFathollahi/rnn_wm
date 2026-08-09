"""comments.txt §23.2: `run_grid.py --help` crashed with `TypeError: %o format:
a number is required, not dict` because a help string contained a bare `%`
(argparse runs help text through `%`-formatting). `--help` is the first thing
anyone runs when unsure of a flag, and nothing in the suite ever invoked it.

This sweeps every argparse entry point rather than that one file: the defect is
a class, not an instance, and a new script can reintroduce it at any time."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _argparse_entry_points() -> list[Path]:
    candidates = sorted(ROOT.glob("*.py")) + sorted(ROOT.glob("scripts/*.py"))
    return [p for p in candidates
            if "argparse" in p.read_text() and "__main__" in p.read_text()]


def test_there_are_entry_points_to_check():
    assert _argparse_entry_points(), "glob found nothing; the sweep below would be vacuous"


@pytest.mark.parametrize("script", _argparse_entry_points(), ids=lambda p: p.name)
def test_help_exits_zero(script):
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=300, cwd=ROOT,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
    )
    assert proc.returncode == 0, f"{script.name} --help failed:\n{proc.stderr[-2000:]}"
    assert "usage" in proc.stdout.lower()
