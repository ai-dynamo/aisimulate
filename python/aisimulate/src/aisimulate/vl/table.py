# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host cost table: frontend stages measured on one serving environment.

One table holds every row collected on a serving host. Rows are keyed by model
and by the frontend, feature transport and image workload they were measured on;
`predict` and `recommend` look a row up exactly and a missing row fails closed
with the collect command that measures it. Rows are measured data: nothing here
interpolates or defaults a cost.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, PositiveInt

from ..config.common import StrictModel
from ..config.engine import (
    FrontendKind,
    FrontendMeasurementConfig,
    FrontendPredictionConfig,
    FrontendStageConfig,
    HostProfileConfig,
)

SGLANG_VERSION = "0.5.19"
"""The sglang release whose frontend the stage boundaries were defined on."""

SGLANG_REVISION = "0bcd822377da7b5718e674eaf9c870d349424dd1"
"""Immutable sgl-project/sglang commit of that release; the behavior model's reference."""


class TableEnvironment(StrictModel):
    """The serving environment every row of a table was measured on."""

    cpu: str
    host: str
    threads: PositiveInt
    sglang_version: str
    python: str


class FrontendRow(StrictModel):
    model: str
    measured_for: FrontendMeasurementConfig
    stages: list[FrontendStageConfig] = Field(min_length=1)
    provenance: dict[str, Any] = Field(default_factory=dict)


class HostCostTable(StrictModel):
    schema_version: Literal[1] = 1
    environment: TableEnvironment
    rows: list[FrontendRow] = Field(default_factory=list)

    def digest(self, row: FrontendRow) -> str:
        """Content digest of a row's measured costs and environment; provenance does not change it."""
        content = {
            "environment": self.environment.model_dump(mode="json"),
            "model": row.model,
            "measured_for": row.measured_for.model_dump(mode="json"),
            "stages": [stage.model_dump(mode="json") for stage in row.stages],
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]


class MissingRow(ValueError):
    """The table holds no row for the requested workload; the message carries the command that measures it."""


def load_table(path: str | Path) -> HostCostTable:
    try:
        return HostCostTable.model_validate(json.loads(Path(path).read_text()))
    except OSError as exc:
        raise ValueError(f"could not read host cost table {path}: {exc}") from exc


def save_table(path: str | Path, table: HostCostTable) -> None:
    """Write atomically so a reader never sees a partial table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as temporary:
        temporary.write(json.dumps(table.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    os.replace(temporary.name, path)


def update_table(path: str | Path, environment: TableEnvironment, row: FrontendRow) -> HostCostTable:
    """Add `row` to the table at `path`, creating it for `environment` or replacing the same measurement.

    The read-modify-write is serialized across collectors sharing the table; the
    measurement itself holds no lock. A table belongs to one serving environment,
    so a recording from another CPU or sglang release is refused.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        table = load_table(path) if path.exists() else HostCostTable(environment=environment)
        recorded = table.environment
        if (recorded.cpu, recorded.sglang_version) != (environment.cpu, environment.sglang_version):
            raise ValueError(
                f"host cost table {path} was measured on {recorded.cpu} with sglang {recorded.sglang_version}; "
                f"this recording is from {environment.cpu} with sglang {environment.sglang_version}. "
                "Keep one table per serving environment."
            )
        kept = [
            existing
            for existing in table.rows
            if (existing.model, existing.measured_for) != (row.model, row.measured_for)
        ]
        table = table.model_copy(update={"rows": [*kept, row]})
        save_table(path, table)
    return table


def collect_command(path: str | Path, *, model: str, measurement: FrontendMeasurementConfig) -> str:
    """The exact command that adds the missing row, ready to paste."""
    parts = [
        shlex.quote(sys.executable),
        "-m",
        "aisimulate.vl.collect",
        "--table",
        shlex.quote(str(path)),
        "--model",
        shlex.quote(model),
        "--frontend",
        measurement.frontend,
        "--images",
        f"{measurement.height}x{measurement.width}x{measurement.count}",
        "--encoding",
        measurement.encoding,
    ]
    if measurement.frontend == "rust":
        parts += ["--tp", "1" if measurement.feature_transport == "inline" else "2"]
    for name, value in (("--min-pixels", measurement.min_pixels), ("--max-pixels", measurement.max_pixels)):
        if value is not None:
            parts += [name, str(value)]
    parts += ["--sglang-python", f"<python with sglang {SGLANG_VERSION} installed>"]
    return " ".join(parts)


def lookup(
    table: HostCostTable, path: str | Path, *, model: str, measurement: FrontendMeasurementConfig
) -> FrontendRow:
    """The row measured for exactly this model and workload on the pinned sglang release."""
    if not table.environment.sglang_version.startswith(SGLANG_VERSION):
        raise ValueError(
            f"host cost table {path} was measured with sglang {table.environment.sglang_version}; "
            f"the frontend model is defined on {SGLANG_VERSION}"
        )
    for row in table.rows:
        if row.model == model and row.measured_for == measurement:
            return row
    raise MissingRow(
        f"host cost table {path} has no row for {model} {measurement.describe()}. "
        "Measure it on the serving host, then rerun:\n  " + collect_command(path, model=model, measurement=measurement)
    )


def resolve_frontend(
    config: HostProfileConfig, *, model: str, images: Mapping[str, Any], tensor: int
) -> tuple[FrontendPredictionConfig, str]:
    """The frontend stages a prediction of `images` on `tensor` ranks runs with, and their row's digest."""
    measurement = FrontendMeasurementConfig.for_workload(config.frontend, images, tensor)
    table = load_table(config.path)
    row = lookup(table, config.path, model=model, measurement=measurement)
    return FrontendPredictionConfig(stages=list(row.stages), measured_for=row.measured_for), table.digest(row)


__all__ = [
    "SGLANG_REVISION",
    "SGLANG_VERSION",
    "FrontendKind",
    "FrontendRow",
    "HostCostTable",
    "MissingRow",
    "TableEnvironment",
    "collect_command",
    "load_table",
    "lookup",
    "resolve_frontend",
    "save_table",
    "update_table",
]
