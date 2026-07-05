"""Frozen model-activity logging contract.

Every model tick during training or evaluation emits one `LogRecord`,
written to Parquet keyed by (run_id, t). The schema is validated on write,
so a malformed record fails loudly rather than corrupting downstream
representational-similarity, demixed-PCA, or encoding-model analyses.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

VALID_EPOCHS = {"fixation", "encode", "maintain", "probe", "feedback", "iti"}


@dataclass
class LogRecord:
    run_id: str
    model_id: str
    seed: int
    t: int
    trial_id: int
    session: str
    epoch: str
    load: int
    held_items: list[int]
    held_categories: list[str]
    probe_item: int
    probe_category: str
    in_set: bool
    action: int
    correct: Optional[bool]
    h_flat: Optional[list[float]]
    h_worker: Optional[list[float]]
    h_manager: Optional[list[float]]
    delta: float
    reflection_R: float
    value: float
    policy: list[float]
    readout_pi: list[float]
    readout_v: list[float]

    def validate(self) -> None:
        if self.epoch not in VALID_EPOCHS:
            raise ValueError(f"invalid epoch {self.epoch!r}; must be one of {VALID_EPOCHS}")
        if self.load < 1:
            raise ValueError(f"load must be >=1, got {self.load}")
        if self.h_flat is None and self.h_worker is None and self.h_manager is None:
            raise ValueError("at least one of h_flat/h_worker/h_manager must be populated")
        if self.h_flat is not None and (self.h_worker is not None or self.h_manager is not None):
            raise ValueError("h_flat is mutually exclusive with h_worker/h_manager (flat vs HRL)")


# Arrow schema mirrors the dataclass field order; list-of-float columns use a
# nullable list type so flat-model runs (h_worker=h_manager=None) stay valid.
_FLOAT_LIST = pa.list_(pa.float32())
_INT_LIST = pa.list_(pa.int32())
_STR_LIST = pa.list_(pa.string())

ARROW_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("model_id", pa.string()),
        ("seed", pa.int32()),
        ("t", pa.int32()),
        ("trial_id", pa.int32()),
        ("session", pa.string()),
        ("epoch", pa.string()),
        ("load", pa.int32()),
        ("held_items", _INT_LIST),
        ("held_categories", _STR_LIST),
        ("probe_item", pa.int32()),
        ("probe_category", pa.string()),
        ("in_set", pa.bool_()),
        ("action", pa.int32()),
        ("correct", pa.bool_()),
        ("h_flat", _FLOAT_LIST),
        ("h_worker", _FLOAT_LIST),
        ("h_manager", _FLOAT_LIST),
        ("delta", pa.float32()),
        ("reflection_R", pa.float32()),
        ("value", pa.float32()),
        ("policy", _FLOAT_LIST),
        ("readout_pi", _FLOAT_LIST),
        ("readout_v", _FLOAT_LIST),
    ]
)

_FIELD_NAMES = [f.name for f in fields(LogRecord)]


def record_to_row(rec: LogRecord) -> dict:
    rec.validate()
    return {name: getattr(rec, name) for name in _FIELD_NAMES}


class ParquetLogWriter:
    """Buffered Parquet writer for `LogRecord`s; one file per run_id.

    Buffers rows and flushes in batches to avoid one-row-per-write overhead
    over a ~200-tick x thousands-of-trials unroll.
    """

    def __init__(self, path: str | Path, batch_size: int = 2048):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self._buffer: list[dict] = []
        self._writer: Optional[pq.ParquetWriter] = None

    def write(self, rec: LogRecord) -> None:
        self._buffer.append(record_to_row(rec))
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=ARROW_SCHEMA)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, ARROW_SCHEMA)
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self) -> None:
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "ParquetLogWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_log(path: str | Path):
    """Read a run's activity log back as a pandas DataFrame (for analysis)."""
    import pandas as pd

    return pq.read_table(path).to_pandas()
