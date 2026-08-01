"""Diagnostic run_id naming (comments.txt §13.4/§14.1): the init-radius
segment is vanilla-specific and must not appear for other substrates, since
`--recurrent-init-spectral-radius` is inert on the GRU path."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase11_pilot import build_diagnostic_model_id


def test_vanilla_run_id_carries_init_segment():
    model_id = build_diagnostic_model_id(s=0, substrate="vanilla", radius=1.0, supervision="legacy")
    assert model_id == "VANFLAT_INIT100_LEGACY"
    assert "INIT" in model_id


def test_gru_run_id_omits_init_segment():
    model_id = build_diagnostic_model_id(s=0, substrate="gru", radius=None, supervision="legacy")
    assert model_id == "FLATGRU_LEGACY"
    assert "INIT" not in model_id
