# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host cost table: measured frontend stages keyed by serving host, SGLang revision, model, frontend and shape.

One file holds every row collected for a serving environment. `predict` and
`recommend` look a row up by the exact image shape and text length they
simulate; a missing row fails closed with the collect command that measures
it. Rows are measured data: nothing here interpolates or defaults a cost.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, PositiveInt

from ..config.common import StrictModel
from ..config.engine import FrontendPredictionConfig, FrontendStageConfig, HostProfileConfig

SGLANG_REVISION = "0bcd822377da7b5718e674eaf9c870d349424dd1"
"""sgl-project/sglang v0.5.19: the release whose frontend the stage boundaries were defined on."""

SGLANG_VERSION = "0.5.19"

FrontendKind = Literal["python", "rust"]


class Shape(StrictModel):
    """The sampled workload a row is valid for."""

    height: PositiveInt
    width: PositiveInt
    count: PositiveInt = 1
    encoding: Literal["png", "jpeg"] = "png"
    min_pixels: PositiveInt | None = None
    max_pixels: PositiveInt | None = None
    text_tokens: PositiveInt

    @classmethod
    def from_images(cls, images: Mapping[str, Any], text_tokens: int) -> Shape:
        return cls(
            height=int(images["height"]),
            width=int(images["width"]),
            count=int(images.get("count", 1)),
            encoding=str(images.get("encoding", "png")),
            min_pixels=images.get("min_pixels"),
            max_pixels=images.get("max_pixels"),
            text_tokens=int(text_tokens),
        )

    def describe(self) -> str:
        pixels = "".join(
            f" {name}={value}"
            for name, value in (("min_pixels", self.min_pixels), ("max_pixels", self.max_pixels))
            if value is not None
        )
        return f"{self.height}x{self.width}x{self.count} {self.encoding} text_tokens={self.text_tokens}{pixels}"


class RowIdentity(StrictModel):
    """Where and on what a row was measured."""

    cpu: str
    sglang_revision: str
    model: str
    frontend: FrontendKind


class FrontendRow(StrictModel):
    identity: RowIdentity
    shape: Shape
    stages: list[FrontendStageConfig] = Field(min_length=1)
    provenance: dict[str, Any] = Field(default_factory=dict)

    def digest(self) -> str:
        """Content digest of the measured costs; provenance does not change it."""
        content = {
            "identity": self.identity.model_dump(mode="json"),
            "shape": self.shape.model_dump(mode="json"),
            "stages": [stage.model_dump(mode="json") for stage in self.stages],
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]


class HostCostTable(StrictModel):
    schema_version: Literal[3] = 3
    rows: list[FrontendRow] = Field(default_factory=list)


class MissingRow(ValueError):
    """The table holds no row for the requested workload; the message carries the command that measures it."""


def load_table(path: str | Path) -> HostCostTable:
    return HostCostTable.model_validate(json.loads(Path(path).read_text()))


def save_table(path: str | Path, table: HostCostTable) -> None:
    """Write atomically so a reader never sees a partial table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(table.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def upsert_row(table: HostCostTable, row: FrontendRow) -> HostCostTable:
    """Replace the row measured for the same identity and shape, or append."""
    kept = [existing for existing in table.rows if (existing.identity, existing.shape) != (row.identity, row.shape)]
    kept.append(row)
    return table.model_copy(update={"rows": kept})


def collect_command(path: str | Path, *, model: str, frontend: FrontendKind, shape: Shape) -> str:
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
        frontend,
        "--images",
        f"{shape.height}x{shape.width}x{shape.count}",
        "--encoding",
        shape.encoding,
        "--text-tokens",
        str(shape.text_tokens),
    ]
    for name, value in (("--min-pixels", shape.min_pixels), ("--max-pixels", shape.max_pixels)):
        if value is not None:
            parts += [name, str(value)]
    parts += ["--sglang-python", f"<python with sglang {SGLANG_VERSION} installed>"]
    return " ".join(parts)


def lookup(table: HostCostTable, path: str | Path, *, model: str, frontend: FrontendKind, shape: Shape) -> FrontendRow:
    """The row measured for exactly this workload on the pinned SGLang revision.

    The measuring host's CPU is not part of the key: predictions run away from
    the serving host. Rows from several CPUs for one key are ambiguous and are
    reported rather than picked from.
    """
    matches = [
        row
        for row in table.rows
        if row.identity.sglang_revision == SGLANG_REVISION
        and row.identity.model == model
        and row.identity.frontend == frontend
        and row.shape == shape
    ]
    if not matches:
        raise MissingRow(
            f"host cost table {path} has no row for {frontend} {shape.describe()} "
            f"(sglang {SGLANG_REVISION[:7]}, {model}). Measure it on the serving host, then rerun:\n  "
            + collect_command(path, model=model, frontend=frontend, shape=shape)
        )
    cpus = sorted({row.identity.cpu for row in matches})
    if len(cpus) > 1:
        raise ValueError(
            f"host cost table {path} holds rows for {frontend} {shape.describe()} from several CPUs "
            f"({', '.join(cpus)}); keep one table per serving host"
        )
    return matches[-1]


def resolve_frontend(
    config: HostProfileConfig, *, model: str, images: Mapping[str, Any], text_tokens: int | None
) -> tuple[FrontendPredictionConfig, str]:
    """The frontend stages a prediction runs with, and the digest of the row they came from."""
    if text_tokens is None:
        raise ValueError("host_profile requires a synthetic workload with a fixed text length")
    shape = Shape.from_images(images, text_tokens)
    row = lookup(load_table(config.path), config.path, model=model, frontend=config.frontend, shape=shape)
    return FrontendPredictionConfig(stages=list(row.stages)), row.digest()
