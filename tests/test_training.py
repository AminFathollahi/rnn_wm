"""Training entrypoint test: all eight cells complete a forward pass and
weight update and emit a well-formed result at a minimal step budget (the
smoke tier). Skips if the stimuli pool has not been built yet."""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
STIMULI_READY = (ROOT / "stimuli" / "faces").exists()

pytestmark = pytest.mark.skipif(not STIMULI_READY, reason="stimuli/ pool not built yet")

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("cell", CFG["cells"])
def test_cell_smoke(cell):
    import shutil

    from brainalign_wm.training import train as T

    S, M, L = int(cell[1]), int(cell[2]), int(cell[3])
    run = {"model_id": cell, "S": S, "M": M, "L": L, "seed": 0, "run_id": f"SMOKETEST_{cell}"}
    ckpt_dir = ROOT / "results" / "checkpoints" / run["run_id"]
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    try:
        result = T.train_one(run, {"steps": 6, "scaffold_sleep_s": 0})
    finally:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

    assert result["status"] == "completed"
    assert set(result["gates"].keys()) == {"load1>0.95", "load3>0.80"}
    assert set(result["accuracy"].keys()) == {"load1", "load2", "load3"}
    if L == 1:
        assert result["rung"] in (1, 2)
    else:
        assert result["rung"] == 0
