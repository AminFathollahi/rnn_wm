#!/usr/bin/env python3
"""List the completed run ids the post-training analysis should cover.

Covers the ablation battery and every control and extension arm built on it:
a cell selector is five arm bits or the local-learning short form, optionally
carrying an arm tag. A run qualifies when its id is exactly its cell selector,
training signal and seed, so a run trained to a different duration or under
any signal other than the two the study compares stays out."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainalign_wm.config import get_path


CELL = re.compile(r"M(?:[01]{5}|[01]{2}L)(?:_[a-z0-9]+)?$")


def main() -> int:
    manifest = get_path("results") / "manifest.jsonl"
    latest: dict[str, dict] = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Smoke and dev probes reuse campaign run ids; continuation rows carry no tier.
        if record.get("tier", "full") != "full":
            continue
        latest[record.get("run_id", "")] = record

    run_ids = []
    for run_id, record in latest.items():
        model_id = str(record.get("model_id", ""))
        supervision = record.get("supervision")
        if (
            record.get("status") == "completed"
            and not record.get("archived", False)
            and CELL.fullmatch(model_id)
            and supervision in {"SUP", "RL"}
            and re.fullmatch(rf"{re.escape(model_id)}_{supervision}_s\d+", run_id)
        ):
            run_ids.append(run_id)

    for run_id in sorted(run_ids):
        print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
