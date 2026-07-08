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


def test_stimulus_cache_covers_real_trial_picids(adapter):
    """Regression guard: `cache_stimulus_features` must resolve each
    session's OWN PicID convention (direct `image_<PicID>` key for 000673;
    a 1-indexed position into `order_of_images` for 000469 -- these are
    genuinely different across datasets, confirmed against the mounted NWB
    files). Caching only the direct-key interpretation silently failed to
    match nearly all of 000469's real trial PicIDs (0% for most sessions
    checked), which meant `generate_activity_logs.py` skipped almost every
    000469 trial (and, since 000469 is the only Tier A dataset with a
    load=2 arm, nearly every load=2 condition system-wide) without any
    error -- found via a real post-grid analysis run, not anticipated in
    advance. This test would have caught it: real trial PicIDs must mostly
    resolve to a cached feature, not mostly miss.

    Checks whatever is currently on disk at `results/feat_cache/dataset_stimuli/`
    (run `python -m brainalign_wm.encoders.cache_features` first) rather than
    regenerating it -- that step encodes thousands of images through ResNet
    and would make the suite noticeably slower on every run."""
    from brainalign_wm.training.generate_activity_logs import _stimulus_features_for_session

    trials = adapter.trials()
    if not any((ROOT / "results" / "feat_cache" / "dataset_stimuli" / f"{s}.npz").exists() for s in trials.session.unique()):
        pytest.skip("no cached stimulus features on disk; run `python -m brainalign_wm.encoders.cache_features` first")
    total, kept = 0, 0
    for sess in trials.session.unique():
        cache = _stimulus_features_for_session(sess)
        for row in trials[trials.session == sess].itertuples():
            held = [int(x) for x in row.held_items if int(x) != 0]
            probe = int(row.probe_item)
            total += 1
            kept += bool(cache) and all(str(p) in cache for p in held + [probe])
    assert total > 0
    assert kept / total > 0.9, f"only {kept}/{total} real trials matched a cached stimulus feature"


def test_tier_b_001187_loads_via_wm_trials_group():
    """001187's Sternberg trials live under `intervals/WM_trials`, not the
    top-level `intervals/trials` 000469/000673 use -- and it also has an
    unrelated `intervals/LTM_trials` New/Old task that must not be picked
    up as a WM session."""
    if not (DATA_ROOT / "001187").exists():
        pytest.skip(f"001187 not present under {DATA_ROOT}")
    from brainalign_wm.neural.adapters.dandi_nwb import DandiSternbergTierA
    from brainalign_wm.neural.dataset_contract import validate_dataset

    tier_b = DandiSternbergTierA(
        DATA_ROOT, datasets=("001187",), min_firing_hz=CFG["neural"]["min_firing_hz"],
        bin_ms=CFG["neural"]["bin_ms"], max_sessions_per_dataset=2,
    )
    assert len(tier_b.sessions()) > 0
    assert len(tier_b.units()) > 0
    validate_dataset(tier_b, region=None)
