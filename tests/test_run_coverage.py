import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from scripts.list_active_analysis_runs import CELL


def test_cell_selector_accepts_every_trained_arm():
    for model_id in ("M00000", "M11111", "M00L", "M11L",
                     "M00000_dynsyn", "M00000_local289", "M10010_idcatch",
                     "M00000_contwm", "M11111_contmulti"):
        assert CELL.fullmatch(model_id), model_id


def test_cell_selector_rejects_named_pilots():
    for model_id in ("FLATGRU", "HIERGRU", "M00000_H128", "M0000", "M000000"):
        assert not CELL.fullmatch(model_id), model_id


def _qualifies(model_id, supervision, run_id):
    return bool(
        CELL.fullmatch(model_id)
        and supervision in {"SUP", "RL"}
        and re.fullmatch(rf"{re.escape(model_id)}_{supervision}_s\d+", run_id)
    )


def test_run_id_must_be_exactly_cell_signal_seed():
    assert _qualifies("M00000", "SUP", "M00000_SUP_s0")
    assert _qualifies("M00000_contwm", "SUP", "M00000_contwm_SUP_s7")
    # a different training duration is a separate run of the same cell
    assert not _qualifies("M00000", "SUP", "M00000_SUP_s0_budget80000")
    # the earlier hybrid signal is not one of the two the study compares
    assert not _qualifies("M00000", "legacy", "M00000_legacy_s0")
    # a tag on the run id that the cell selector does not carry
    assert not _qualifies("M00000", "SUP", "M00000_dynsyn_SUP_s0")


def test_later_smoke_row_neither_listed_nor_hides_full_row(tmp_path, monkeypatch, capsys):
    import json
    import scripts.list_active_analysis_runs as listing

    rows = [
        {"run_id": "M00000_SUP_s0", "model_id": "M00000", "supervision": "SUP", "status": "completed", "tier": "full"},
        {"run_id": "M00000_SUP_s0", "model_id": "M00000", "supervision": "SUP", "status": "error", "tier": "smoke"},
        {"run_id": "M00000_SUP_s1", "model_id": "M00000", "supervision": "SUP", "status": "completed", "tier": "smoke"},
        {"run_id": "M00000_contwm_SUP_s0", "model_id": "M00000_contwm", "supervision": "SUP", "status": "completed"},
    ]
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(listing, "get_path", lambda name: tmp_path)
    listing.main()
    assert capsys.readouterr().out.split() == ["M00000_SUP_s0", "M00000_contwm_SUP_s0"]
