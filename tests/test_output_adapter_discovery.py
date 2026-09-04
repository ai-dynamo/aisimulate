# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from aisimulate.output_adapter import (
    OUTPUT_ADAPTER_API_VERSION,
    OutputAdapterExecutionError,
    OutputAdapterResolutionError,
    resolve_output_adapters,
    write_output_adapters,
)


class _Adapter:
    api_version = OUTPUT_ADAPTER_API_VERSION

    def __init__(self, name: str) -> None:
        self.name = name

    def write(self, config, *, result, output_dir):
        del config, result
        path = Path("artifact.txt")
        (output_dir / path).write_text("artifact\n")
        return [path]


@dataclass
class _EntryPoint:
    name: str
    value: str
    provider: object
    loads: int = 0

    def load(self):
        self.loads += 1
        if isinstance(self.provider, Exception):
            raise self.provider
        return self.provider


def test_only_selected_output_adapter_is_loaded() -> None:
    selected = _EntryPoint("dgd", "example:create_dgd", lambda: _Adapter("dgd"))
    unused = _EntryPoint("chart", "example:create_chart", lambda: _Adapter("chart"))

    resolved = resolve_output_adapters(["dgd"], entry_points=[selected, unused])

    assert list(resolved) == ["dgd"]
    assert selected.loads == 1
    assert unused.loads == 0


def test_missing_output_adapter_lists_available_names() -> None:
    available = _EntryPoint("chart", "example:create_chart", lambda: _Adapter("chart"))

    with pytest.raises(OutputAdapterResolutionError, match="dgd.*installed adapters: chart"):
        resolve_output_adapters(["dgd"], entry_points=[available])

    assert available.loads == 0


@pytest.mark.parametrize(
    ("adapter", "message"),
    [
        (_Adapter("wrong"), "returned name"),
        (type("OldAdapter", (_Adapter,), {"api_version": 0})("dgd"), "API version"),
        (type("IncompleteAdapter", (), {"name": "dgd", "api_version": OUTPUT_ADAPTER_API_VERSION})(), "write"),
    ],
)
def test_invalid_output_adapter_abi_is_rejected(adapter, message) -> None:
    with pytest.raises(OutputAdapterResolutionError, match=message):
        resolve_output_adapters(["dgd"], injected={"dgd": adapter}, entry_points=[])


def test_reported_artifacts_must_be_relative_existing_paths(tmp_path) -> None:
    class InvalidPathAdapter(_Adapter):
        def write(self, config, *, result, output_dir):
            del config, result, output_dir
            return ["../outside"]

    with pytest.raises(OutputAdapterExecutionError, match="outside the output directory"):
        write_output_adapters(
            {"dgd": InvalidPathAdapter("dgd")},
            {"dgd": {}},
            result=object(),
            output_dir=tmp_path,
        )


def test_output_adapter_writes_into_supplied_directory(tmp_path) -> None:
    artifacts = write_output_adapters(
        {"dgd": _Adapter("dgd")},
        {"dgd": {"name": "example"}},
        result=object(),
        output_dir=tmp_path,
    )

    assert artifacts == {"dgd": (Path("artifact.txt"),)}
    assert (tmp_path / "artifact.txt").read_text() == "artifact\n"
