from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np

from brainalign_wm.neural.adapters.dandi_nwb import (
    DandiSternbergTierA,
    _resolve_stimulus_images,
    find_wm_sessions,
)


def _write_nwb(
    path: Path,
    dataset: str,
    identifier: str,
    pic_ids: tuple[int, int] = (101, 102),
    positional: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enc_cols = (
        ("loadsEnc1_PicIDs", "loadsEnc2_PicIDs", "loadsEnc3_PicIDs")
        if dataset == "000469"
        else ("PicIDs_Encoding1", "PicIDs_Encoding2", "PicIDs_Encoding3")
    )
    probe_col = "loadsProbe_PicIDs" if dataset == "000469" else "PicIDs_Probe"
    trial_name = "WM_trials" if dataset == "001187" else "trials"

    with h5py.File(path, "w") as h:
        h.create_dataset("identifier", data=np.bytes_(identifier))
        trials = h.create_group(f"intervals/{trial_name}")
        values = {
            "id": [0],
            "loads": [1],
            enc_cols[0]: [pic_ids[0]],
            enc_cols[1]: [0],
            enc_cols[2]: [0],
            probe_col: [pic_ids[1]],
            "probe_in_out": [1],
            "response_accuracy": [1],
            "timestamps_FixationCross": [0.5],
            "timestamps_Maintenance": [2.0],
            "timestamps_Probe": [4.0],
            "timestamps_Response": [4.5],
            "start_time": [0.0],
            "stop_time": [5.0],
        }
        for name, value in values.items():
            trials.create_dataset(name, data=value)

        if dataset == "001187":
            ltm = h.create_group("intervals/LTM_trials")
            ltm.create_dataset("id", data=[0])

        templates = h.create_group("stimulus/templates/StimulusTemplates")
        image_ids = (21, 22) if positional else pic_ids
        refs = []
        for image_id in image_ids:
            image = templates.create_dataset(
                f"image_{image_id}", data=np.full((2, 2, 3), image_id, dtype=np.uint8)
            )
            refs.append(image.ref)
        if positional:
            templates.create_dataset("order_of_images", data=np.asarray(refs, dtype=h5py.ref_dtype))

        electrodes = h.create_group("general/extracellular_ephys/electrodes")
        electrodes.create_dataset("location", data=np.asarray([b"hippocampus_left"]))
        units = h.create_group("units")
        units.create_dataset("id", data=[0])
        units.create_dataset("electrodes", data=[0])
        units.create_dataset("spike_times", data=[1.0, 2.5])
        units.create_dataset("spike_times_index", data=[2])


def test_tier_b_wm_table_and_direct_stimuli(tmp_path):
    path = tmp_path / "001187" / "sub-1" / "session.nwb"
    _write_nwb(path, "001187", "sub-1_ses-1_P1CS")

    assert find_wm_sessions(path.parents[1]) == [path]
    with h5py.File(path, "r") as h:
        images = _resolve_stimulus_images(h, "001187")
    assert set(images) == {"101", "102"}


def test_tier_a_positional_stimuli_remain_supported(tmp_path):
    path = tmp_path / "000469" / "sub-1" / "session.nwb"
    _write_nwb(path, "000469", "sub-1_ses-2_P1CS", pic_ids=(1, 2), positional=True)

    with h5py.File(path, "r") as h:
        images = _resolve_stimulus_images(h, "000469")
    assert set(images) == {"1", "2"}
    assert int(images["1"][0, 0, 0]) == 21


def test_cache_enumerates_tier_a_and_tier_b(tmp_path, monkeypatch):
    from brainalign_wm.encoders import resnet18_encoder
    from brainalign_wm.neural.adapters.dandi_nwb import cache_stimulus_features

    _write_nwb(
        tmp_path / "data" / "000469" / "sub-1" / "a.nwb",
        "000469",
        "sub-1_ses-2_P1CS",
        pic_ids=(1, 2),
        positional=True,
    )
    _write_nwb(
        tmp_path / "data" / "001187" / "sub-2" / "b.nwb",
        "001187",
        "sub-2_ses-1_P2CS",
    )
    monkeypatch.setattr(
        resnet18_encoder,
        "encode_images",
        lambda images: np.arange(len(images) * 3, dtype=float).reshape(len(images), 3),
    )
    cfg = {
        "paths": {"data_root": tmp_path / "data", "feature_cache": tmp_path / "cache"},
        "neural": {"datasets_tierA": ["000469"], "datasets_tierB": ["001187"]},
    }

    cache_stimulus_features(cfg)

    cache = tmp_path / "cache" / "dataset_stimuli"
    tier_a = np.load(cache / "000469-sub-1_ses-2_P1CS.npz")
    tier_b = np.load(cache / "001187-sub-2_ses-1_P2CS.npz")
    assert set(tier_a["pic_ids"]) == {"1", "2"}
    assert set(tier_b["pic_ids"]) == {"101", "102"}
    assert not list(cache.glob("*.partial"))

    encode = Mock(side_effect=AssertionError("complete caches must be reused"))
    monkeypatch.setattr(resnet18_encoder, "encode_images", encode)
    cache_stimulus_features(cfg)
    encode.assert_not_called()


def test_overlapping_releases_use_001187_once(tmp_path):
    _write_nwb(
        tmp_path / "000673" / "sub-1" / "a.nwb",
        "000673",
        "sub-1_ses-1_P68CS",
    )
    _write_nwb(
        tmp_path / "001187" / "sub-9" / "b.nwb",
        "001187",
        "sub-9_ses-1_P68CS",
    )
    dataset = DandiSternbergTierA(tmp_path, datasets=("000673", "001187"))

    assert dataset.sessions() == ["001187-sub-9_ses-1_P68CS"]
    assert dataset.patient_of(dataset.sessions()[0]) == "P68CS"


def test_distinct_sessions_across_releases_are_retained(tmp_path):
    _write_nwb(
        tmp_path / "000673" / "sub-1" / "a.nwb",
        "000673",
        "sub-1_ses-1_P68CS",
    )
    _write_nwb(
        tmp_path / "001187" / "sub-9" / "b.nwb",
        "001187",
        "sub-9_ses-2_P68CS",
    )
    dataset = DandiSternbergTierA(tmp_path, datasets=("001187", "000673"))

    assert set(dataset.sessions()) == {
        "000673-sub-1_ses-1_P68CS",
        "001187-sub-9_ses-2_P68CS",
    }
