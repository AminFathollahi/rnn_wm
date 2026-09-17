import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import torch
from brainalign_wm.config import load_config
from scripts.run_continuations import enumerate_continuations, verify_source


def test_source_verification_runs_for_a_reflective_cell():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = next(r for r in enumerate_continuations(models=("M11111",))
               if r["run_id"].endswith("_s0") and r["M"] == 1)
    out = verify_source(run, cfg, device)
    assert out["source_state_sha256"]
    assert out["adapter_state_sha256"]
