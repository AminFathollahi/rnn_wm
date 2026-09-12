"""Model overrides for checkpoint-compatible training variants."""
from __future__ import annotations


LOCALITY_BUDGET_TAG = "local289"
LOCALITY_BUDGET = {
    "flat_units": 289,
    "flat_grid": [17, 17],
    "flat_density": 0.0681,
    "flat_recurrent_init_units": 196,
}

FLAT_CONTROLS = {
    "local128": {
        "flat_units": 128,
        "flat_grid": [16, 8],
        "flat_density": 0.1495,
        "flat_connectivity": "local",
        "flat_mask_seed": 0,
        "flat_recurrent_init_units": 128,
    },
    "random128": {
        "flat_units": 128,
        "flat_grid": [16, 8],
        "flat_density": 0.1495,
        "flat_connectivity": "random",
        "flat_mask_seed": 0,
        "flat_recurrent_init_units": 128,
    },
    "dense289": {
        "flat_units": 289,
        "flat_grid": [17, 17],
        "flat_density": None,
        "flat_connectivity": "dense",
        "flat_mask_seed": 0,
        "flat_recurrent_init_units": 289,
    },
    "local289native": {
        "flat_units": 289,
        "flat_grid": [17, 17],
        "flat_density": 0.0681,
        "flat_connectivity": "local",
        "flat_mask_seed": 0,
        "flat_recurrent_init_units": 289,
    },
    "random289": {
        "flat_units": 289,
        "flat_grid": [17, 17],
        "flat_density": 0.0681,
        "flat_connectivity": "random",
        "flat_mask_seed": 0,
        "flat_recurrent_init_units": 289,
    },
}


def overrides_for_model(model_id: str) -> dict:
    """Return model overrides encoded by a tagged model identifier."""
    parts = set(model_id.split("_"))
    if LOCALITY_BUDGET_TAG in parts:
        return dict(LOCALITY_BUDGET)
    for tag, overrides in FLAT_CONTROLS.items():
        if tag in parts:
            return dict(overrides)
    return {}
