"""dandi_nwb adapter tests: returns schema-valid rates, regions, and
conditions, and a real noise ceiling, against the actual mounted external
drive data. Skips gracefully if the drive is not mounted."""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
DATA_ROOT = Path(CFG["paths"]["data_root"])

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "000469").exists(), reason=f"external USB not mounted at {DATA_ROOT}"
)


@pytest.fixture(scope="module")
def adapter():
    from brainalign_wm.neural.adapters.dandi_nwb import DandiSternbergTierA

    return DandiSternbergTierA(
        DATA_ROOT,
        datasets=("000469", "000673"),
        min_firing_hz=CFG["neural"]["min_firing_hz"],
        bin_ms=CFG["neural"]["bin_ms"],
        max_sessions_per_dataset=2,  # keep the test fast; full pool used at M7/M8
    )


def test_contract_validates(adapter):
    from brainalign_wm.neural.dataset_contract import validate_dataset

    validate_dataset(adapter, region=None)


def test_regions_include_mtl_and_mfc(adapter):
    regs = set(adapter.regions())
    mtl_hit = regs & {"hippocampus", "amygdala", "entorhinal"}
    mfc_hit = regs & {"dACC", "preSMA", "vmPFC"}
    assert mtl_hit, f"no MTL regions found in {regs}"
    assert mfc_hit, f"no MFC regions found in {regs}"


def test_conditions_nonempty_and_load_in_range(adapter):
    assert len(adapter.conditions) > 0
    for c in adapter.conditions:
        assert c.load in (1, 2, 3)


def test_rates_shape(adapter):
    rates = adapter.rates(None, adapter.bin_ms, ["maintain"])
    assert rates.shape[0] == len(adapter.units())
    assert rates.shape[1] == len(adapter.trials())
    assert (rates >= 0).all()


def test_noise_ceiling_sane(adapter):
    lower, upper = adapter.noise_ceiling(None, "maintain")
    assert 0.0 <= lower <= upper <= 1.0 + 1e-6
