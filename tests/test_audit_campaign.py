"""comments.txt §20.3: `scripts/audit_campaign.py`'s tier-poisoning check must
fire on a real instance of the defect it exists to catch -- a smoke-tier
`completed` row for a core cell sitting next to a genuine full-tier one --
and only on the smoke-tier row, not the full-tier one."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_campaign as ac


def test_tier_poisoning_fires_once_for_the_smoke_row(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({
            "run_id": "M11011_SUP_s0", "status": "completed", "tier": "smoke",
            "wall_clock_s": 14.6, "config_hash": "28ac84b1eefa",
        }) + "\n"
        + json.dumps({
            "run_id": "M00000_SUP_s0", "status": "completed", "tier": "full",
            "wall_clock_s": 8527.0, "config_hash": "d60d4f8cc7e2",
        }) + "\n"
    )
    ac.violations.clear()
    ac.warnings.clear()
    orig_manifest = ac.MANIFEST
    ac.MANIFEST = manifest
    try:
        rows = ac.load_rows()
        ac.check_tier_poisoning(rows, "full")
    finally:
        ac.MANIFEST = orig_manifest

    tier_violations = [v for v in ac.violations if v.startswith("[tier-poisoning]")]
    assert len(tier_violations) == 1
    assert "M11011_SUP_s0" in tier_violations[0]
    assert "M00000_SUP_s0" not in tier_violations[0]
