"""Ensure every command-line entry point can render its help text.

An argparse help string containing a bare percent sign once made
`run_grid.py --help` raise a formatting `TypeError`. Help is a public entry
point and needs the same regression coverage as execution.

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
