"""Selectivity of the two memory-location lesions (activity vs. synaptic
trace) on `PlasticGRUCell`, both flat (S=0) and hierarchical-worker (S=1)
state layouts.
"""
import torch

import pandas as pd
import pytest

from brainalign_wm.analysis.memory_location import (
    lesion_state, lesion_step, summarize_interventions, summarize_selectivity,
)
from brainalign_wm.models.gru_cell import PlasticGRUCell
from brainalign_wm.models.hrl import HRLCore


def _flat_state(batch=4, hidden=8, input_dim=5, seed=0):
    torch.manual_seed(seed)
    cell = PlasticGRUCell(input_dim, hidden)
    with torch.no_grad():
        cell.alpha.copy_(torch.randn_like(cell.alpha) * 0.5)

    class Core:
        def __init__(self):
            self.cell = cell

        def __call__(self, z_t, h_prev, hebb_prev=None, extra_update_bias=None):
            return cell(z_t, h_prev, hebb_prev, extra_update_bias=extra_update_bias)

        def readout_state(self, h_t):
            return h_t

    core = Core()
    state = {"h": torch.randn(batch, hidden).clamp(-1, 1), "hebb": torch.randn(batch, 3 * hidden, hidden).clamp(-2, 2)}
    z_t = torch.randn(batch, input_dim)
    return core, state, z_t


def test_activity_lesion_freezes_synapse_flat():
    core, state, z_t = _flat_state()
    gen = torch.Generator().manual_seed(0)
    _, natural, _ = lesion_step(core, 0, 0, 1, z_t, state, t=0, gate_bias=None, target=None, frac=1.0, gen=gen)
    _, lesioned, _ = lesion_step(core, 0, 0, 1, z_t, state, t=0, gate_bias=None, target="activity", frac=1.0, gen=gen)
    assert torch.equal(lesioned["hebb"], state["hebb"]), "activity lesion must leave the trace bit-identical"
    assert not torch.equal(lesioned["h"], natural["h"]), "activity lesion must actually change the hidden state"


def test_synaptic_lesion_leaves_activity_input_untouched_flat():
    core, state, z_t = _flat_state()
    gen_a = torch.Generator().manual_seed(1)
    gen_b = torch.Generator().manual_seed(1)
    _, control, _ = lesion_step(core, 0, 0, 1, z_t, state, t=0, gate_bias=None, target=None, frac=0.0, gen=gen_a)
    _, lesioned, _ = lesion_step(core, 0, 0, 1, z_t, state, t=0, gate_bias=None, target="synaptic", frac=1.0, gen=gen_b)
    assert not torch.equal(lesioned["hebb"], state["hebb"])
    assert not torch.equal(lesioned["h"], control["h"]), "corrupted trace should propagate into h via w_eff"


def test_lesion_bounds_match_each_channels_own_range():
    core, state, z_t = _flat_state()
    gen = torch.Generator().manual_seed(2)
    _, lesioned_h, _ = lesion_step(core, 0, 0, 1, z_t, state, t=0, gate_bias=None, target="activity", frac=1.0, gen=gen)
    assert lesioned_h["h"].abs().max() <= 1.0 + 1e-5
    _, lesioned_hebb, _ = lesion_step(core, 0, 0, 1, z_t, state, t=0, gate_bias=None, target="synaptic", frac=1.0, gen=gen)
    assert lesioned_hebb["hebb"].abs().max() <= core.cell.hebb_clip + 1e-5


def test_activity_lesion_freezes_synapse_hierarchical():
    torch.manual_seed(0)
    core = HRLCore(input_dim=5, worker_units=9, manager_units=4, grid=(3, 3), plastic=True)
    batch = 3
    state = core.init_state(batch)
    z_t = torch.randn(batch, 5)
    gen = torch.Generator().manual_seed(0)
    _, lesioned, _ = lesion_step(core, 1, 0, 1, z_t, state, t=0, gate_bias=None, target="activity", frac=1.0, gen=gen)
    assert torch.equal(lesioned["hebb_worker"], state["hebb_worker"])


def test_lesion_state_changes_only_selected_channel():
    core, state, _ = _flat_state()
    activity = lesion_state(core, 0, state, "activity", 1.0, torch.Generator().manual_seed(3))
    synaptic = lesion_state(core, 0, state, "synaptic", 1.0, torch.Generator().manual_seed(3))
    assert torch.equal(activity["hebb"], state["hebb"])
    assert not torch.equal(activity["h"], state["h"])
    assert torch.equal(synaptic["h"], state["h"])
    assert not torch.equal(synaptic["hebb"], state["hebb"])
    with pytest.raises(ValueError):
        lesion_state(core, 0, state, "other", 1.0, torch.Generator().manual_seed(3))


def test_intervention_summary_uses_paired_control():
    df = pd.DataFrame([
        {"run_id": "a", "S": 0, "supervision": "SUP", "condition": "control", "load": 1, "accuracy": 0.9},
        {"run_id": "a", "S": 0, "supervision": "SUP", "condition": "delay_activity", "load": 1, "accuracy": 0.6},
        {"run_id": "b", "S": 0, "supervision": "SUP", "condition": "control", "load": 1, "accuracy": 0.8},
        {"run_id": "b", "S": 0, "supervision": "SUP", "condition": "delay_activity", "load": 1, "accuracy": 0.7},
    ])
    summary = summarize_interventions(df)
    row = summary.loc[summary["condition"].eq("delay_activity")].iloc[0]
    assert row["n_runs"] == 2
    assert row["mean_drop"] == pytest.approx(0.2)


def test_selectivity_summary_pairs_each_lesion_with_its_off_target_carrier():
    df = pd.DataFrame([
        {"condition": "act", "target": "activity",
         "activity_deviation_during": 2.0, "synaptic_deviation_during": 0.5,
         "activity_deviation_after": 1.0, "synaptic_deviation_after": 4.0},
        {"condition": "act", "target": "activity",
         "activity_deviation_during": 4.0, "synaptic_deviation_during": 0.1,
         "activity_deviation_after": 3.0, "synaptic_deviation_after": 6.0},
        {"condition": "syn", "target": "synaptic",
         "activity_deviation_during": 0.2, "synaptic_deviation_during": 8.0,
         "activity_deviation_after": 0.4, "synaptic_deviation_after": 2.0},
    ])
    out = summarize_selectivity(df).set_index(["condition", "window"])
    act = out.loc[("act", "during")]
    assert act["target"] == "activity"
    assert act["on_target_deviation"] == pytest.approx(3.0)
    assert act["off_target_deviation"] == pytest.approx(0.3)
    assert act["off_over_on"] == pytest.approx(0.1)
    syn = out.loc[("syn", "during")]
    assert syn["target"] == "synaptic"
    assert syn["on_target_deviation"] == pytest.approx(8.0)
    assert syn["off_target_deviation"] == pytest.approx(0.2)
    assert out.loc[("act", "after")]["off_target_deviation"] == pytest.approx(5.0)


def test_erase_removes_the_trace_and_noise_overshoots_it():
    core, state, z_t = _flat_state()
    erased = lesion_state(core, 0, state, "synaptic", 1.0, torch.Generator().manual_seed(0), mode="erase")
    noised = lesion_state(core, 0, state, "synaptic", 1.0, torch.Generator().manual_seed(0), mode="noise")
    assert torch.equal(erased["hebb"], torch.zeros_like(state["hebb"]))
    assert noised["hebb"].norm() > state["hebb"].norm()


def test_held_trace_stays_at_its_pre_lesion_value_across_the_window():
    """The trace a held activity lesion carries through the window is the one
    it entered with, so its distance from the unlesioned trace measures how
    far the unlesioned trace travelled, not corruption written in."""
    core, state, z_t = _flat_state()
    gen = torch.Generator().manual_seed(0)
    entry = state["hebb"].clone()
    s = state
    for t in range(5):
        _, s, _ = lesion_step(core, 0, 0, 1, z_t, s, t=t, gate_bias=None, target="activity",
                              frac=1.0, gen=gen, hold_trace=True)
        assert torch.equal(s["hebb"], entry)

    free = state
    gen = torch.Generator().manual_seed(0)
    for t in range(5):
        _, free, _ = lesion_step(core, 0, 0, 1, z_t, free, t=t, gate_bias=None, target="activity",
                                 frac=1.0, gen=gen, hold_trace=False)
    assert not torch.equal(free["hebb"], entry)
