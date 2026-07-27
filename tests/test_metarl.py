"""Phase 7 (comments.txt §5, SUP/RL/METARL): `_select_signal`'s SUP/RL/
legacy behavior, `MetaRLAdapter`'s shape, `sample_metarl_block`'s block
content/determinism, and `run_metarl_block`'s mechanics -- finite loss and
gradients (S=0 and S=1), correct snapshot shapes, and (the mechanism the
whole arm depends on) that recurrent state genuinely PERSISTS across
sequences within a block rather than being reset. Skips entirely if the
stimuli pool isn't built (n-back reuses `ImageTokenBank`, same as
Sternberg)."""
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from brainalign_wm.training.train import (
    MetaRLAdapter,
    _build_model,
    _select_signal,
    run_metarl_block,
    sample_metarl_block,
)
from brainalign_wm.tasks.nback import NBackGenerator

ROOT = Path(__file__).resolve().parents[1]
FULL_CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
STIMULI_READY = (ROOT / "stimuli" / "faces").exists()
pytestmark_needs_stimuli = pytest.mark.skipif(
    not STIMULI_READY, reason="stimuli/ pool not built yet -- run scripts/build_stimuli_pool.py"
)


def _make_bank():
    from brainalign_wm.tasks.image_token_bank import ImageTokenBank

    return ImageTokenBank(
        stimuli_root=ROOT / "stimuli", categories=FULL_CFG["task"]["categories"],
        feature_cache_path=ROOT / "results" / "feat_cache" / "test_image_token_bank.npy", seed=0,
    )


# ---------------- _select_signal (no stimuli needed) ----------------

def test_select_signal_legacy_matches_pre_phase7_behavior():
    assert _select_signal("legacy", "warmup") == "ce"
    assert _select_signal("legacy", "ramp") == "reinforce"
    assert _select_signal("legacy", "target") == "reinforce"


def test_select_signal_sup_is_always_ce():
    assert _select_signal("SUP", "warmup") == "ce"
    assert _select_signal("SUP", "ramp") == "ce"
    assert _select_signal("SUP", "target") == "ce"


def test_select_signal_rl_is_always_reinforce():
    assert _select_signal("RL", "warmup") == "reinforce"
    assert _select_signal("RL", "ramp") == "reinforce"
    assert _select_signal("RL", "target") == "reinforce"


# ---------------- MetaRLAdapter (no stimuli needed) ----------------

def test_metarl_adapter_shape():
    adapter = MetaRLAdapter(feature_dim=512, n_actions=3, bottleneck_dim=64)
    v_t = torch.randn(5, 512)
    prev_action_onehot = torch.zeros(5, 3)
    prev_reward = torch.zeros(5, 1)
    z_t = adapter(v_t, prev_action_onehot, prev_reward)
    assert z_t.shape == (5, 64)


# ---------------- sample_metarl_block ----------------

@pytestmark_needs_stimuli
def test_block_content_length_and_label_validity():
    bank = _make_bank()
    gen = NBackGenerator(FULL_CFG, bank)
    batch, n_labels, feature_labels = sample_metarl_block(
        gen, FULL_CFG, seed=0, step_idx=0, batch_size=4, block_size=3,
    )
    assert len(batch) == 4
    expected_T = 3 * FULL_CFG["nback"]["sequence_length"]
    for steps, n, feature in zip(batch, n_labels, feature_labels):
        assert len(steps) == expected_T
        assert n in FULL_CFG["nback"]["n_values"]
        assert feature in FULL_CFG["nback"]["features"]
        # every step in this instance carries the SAME (n, feature) -- fixed
        # but unsignalled per block, item 7's whole premise.
        assert all(s.n == n and s.feature == feature for s in steps)


@pytestmark_needs_stimuli
def test_block_content_deterministic_given_seed():
    bank = _make_bank()
    gen = NBackGenerator(FULL_CFG, bank)
    batch_a, n_a, f_a = sample_metarl_block(gen, FULL_CFG, seed=1, step_idx=2, batch_size=3, block_size=2)
    batch_b, n_b, f_b = sample_metarl_block(gen, FULL_CFG, seed=1, step_idx=2, batch_size=3, block_size=2)
    assert n_a == n_b and f_a == f_b
    for steps_a, steps_b in zip(batch_a, batch_b):
        assert [s.image_id for s in steps_a] == [s.image_id for s in steps_b]
        assert [s.is_match for s in steps_a] == [s.is_match for s in steps_b]


@pytestmark_needs_stimuli
def test_n_and_feature_vary_across_blocks_in_one_batch():
    """Item 7.1's decoding analysis needs `(n, feature)` to vary ACROSS
    block instances in one batch (unlike Sternberg's shared-load batch) --
    a large-enough batch must draw more than one distinct value of each."""
    bank = _make_bank()
    gen = NBackGenerator(FULL_CFG, bank)
    _batch, n_labels, feature_labels = sample_metarl_block(
        gen, FULL_CFG, seed=0, step_idx=0, batch_size=24, block_size=2,
    )
    assert len(set(n_labels)) > 1
    assert len(set(feature_labels)) > 1


# ---------------- run_metarl_block ----------------

def _tiny_cfg():
    return {**FULL_CFG, "nback": {**FULL_CFG["nback"], "sequence_length": 4}}


@pytestmark_needs_stimuli
@pytest.mark.parametrize("S", [0, 1])
def test_run_metarl_block_finite_loss_and_grad(S):
    cfg = _tiny_cfg()
    bank = _make_bank()
    gen = NBackGenerator(cfg, bank)
    device = torch.device("cpu")
    m = cfg["model"]
    _front_end, core, heads = _build_model(cfg, S=S, M=0, P=0, device=device)
    adapter = MetaRLAdapter(m["feature_dim"], m["action_dim"], m["bottleneck"]).to(device)

    batch, _n, _f = sample_metarl_block(gen, cfg, seed=0, step_idx=0, batch_size=3, block_size=2)
    out = run_metarl_block(
        adapter, core, heads, S, 0, 0, bank, batch, sequence_length=cfg["nback"]["sequence_length"],
        feature_dim=m["feature_dim"], n_actions=m["action_dim"], device=device, mode="bptt",
    )
    assert out["loss"] is not None
    assert torch.isfinite(out["loss"])
    assert out["h_star"].shape == (3, 2, m["worker_units"] + m["manager_units"] if S == 1 else m["flat_units"])
    if S == 1:
        assert out["h_worker"].shape == (3, 2, m["worker_units"])
        assert out["h_manager"].shape == (3, 2, m["manager_units"])
    else:
        assert out["h_worker"] is None and out["h_manager"] is None

    out["loss"].backward()
    assert adapter.w.weight.grad is not None
    assert torch.isfinite(adapter.w.weight.grad).all()


@pytestmark_needs_stimuli
def test_recurrent_state_persists_across_sequences_within_a_block():
    """The mechanism METARL depends on entirely: state must NOT reset
    between the K n-back sequences within one block, since the network's
    only way to infer the block's withheld (n, feature) is by carrying
    information across sequence boundaries in its own recurrent state.
    Isolate the SECOND sequence's steps and run it alone (state reset,
    `block_size=1` in effect) vs. running it as sequence 2 of a real
    2-sequence block (state carried from sequence 1) -- with identical
    weights and greedy (`mode="eval"`) action selection, the two must give
    DIFFERENT `h_star` snapshots, since only the carried-over state
    differs."""
    cfg = _tiny_cfg()
    bank = _make_bank()
    gen = NBackGenerator(cfg, bank)
    device = torch.device("cpu")
    m = cfg["model"]
    _front_end, core, heads = _build_model(cfg, S=0, M=0, P=0, device=device)
    adapter = MetaRLAdapter(m["feature_dim"], m["action_dim"], m["bottleneck"]).to(device)

    batch, _n, _f = sample_metarl_block(gen, cfg, seed=3, step_idx=0, batch_size=1, block_size=2)
    seq_len = cfg["nback"]["sequence_length"]
    second_sequence_alone = [batch[0][seq_len:]]

    with torch.no_grad():
        full_block = run_metarl_block(
            adapter, core, heads, 0, 0, 0, bank, batch, sequence_length=seq_len,
            feature_dim=m["feature_dim"], n_actions=m["action_dim"], device=device, mode="eval",
        )
        isolated = run_metarl_block(
            adapter, core, heads, 0, 0, 0, bank, second_sequence_alone, sequence_length=seq_len,
            feature_dim=m["feature_dim"], n_actions=m["action_dim"], device=device, mode="eval",
        )

    snapshot_within_block = full_block["h_star"][0, 1]  # sequence 2's snapshot, state carried
    snapshot_isolated = isolated["h_star"][0, 0]  # same steps, state reset first
    assert not np.allclose(snapshot_within_block, snapshot_isolated), (
        "sequence 2's hidden state is identical whether or not sequence 1 ran first -- "
        "state is not actually persisting across the block"
    )
