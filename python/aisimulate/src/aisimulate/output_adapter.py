# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery and invocation for post-recommendation output adapters."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .sweeper.result import SweepResult

OUTPUT_ADAPTER_API_VERSION = 1
OUTPUT_ADAPTER_ENTRY_POINT_GROUP = "aisimulate.output_adapters"


@runtime_checkable
class RecommendationOutputAdapter(Protocol):
    """Write additional artifacts for a completed recommendation."""

    name: str
    api_version: int

    def write(
        self,
        config: Mapping[str, Any],
        *,
        result: SweepResult,
        output_dir: Path,
    ) -> Sequence[str | Path]:
        """Write artifacts and return their paths relative to ``output_dir``."""


class OutputAdapterResolutionError(RuntimeError):
    """A selected recommendation output adapter could not be resolved."""


class OutputAdapterExecutionError(RuntimeError):
    """A selected recommendation output adapter failed to write its artifacts."""


def validate_output_adapter(adapter: Any, *, requested_name: str) -> RecommendationOutputAdapter:
    """Validate one resolved output adapter."""

    if getattr(adapter, "name", None) != requested_name:
        raise OutputAdapterResolutionError(
            f"output adapter {requested_name!r} returned name {getattr(adapter, 'name', None)!r}"
        )
    api_version = getattr(adapter, "api_version", None)
    if type(api_version) is not int or api_version != OUTPUT_ADAPTER_API_VERSION:
        raise OutputAdapterResolutionError(
            f"output adapter {requested_name!r} uses API version {api_version!r}; "
            f"AISimulate requires {OUTPUT_ADAPTER_API_VERSION}"
        )
    if not callable(getattr(adapter, "write", None)):
        raise OutputAdapterResolutionError(f"output adapter {requested_name!r} is missing callable: write")
    return adapter


def resolve_output_adapters(
    names: Iterable[str],
    *,
    injected: Mapping[str, RecommendationOutputAdapter] | None = None,
    entry_points: Iterable[importlib.metadata.EntryPoint] | None = None,
) -> dict[str, RecommendationOutputAdapter]:
    """Resolve selected output adapters without importing unrelated packages."""

    requested = list(dict.fromkeys(names))
    injected = injected or {}
    installed = (
        list(entry_points)
        if entry_points is not None
        else list(importlib.metadata.entry_points().select(group=OUTPUT_ADAPTER_ENTRY_POINT_GROUP))
    )
    resolved: dict[str, RecommendationOutputAdapter] = {}
    for name in requested:
        if name in injected:
            resolved[name] = validate_output_adapter(injected[name], requested_name=name)
            continue
        matches = [entry for entry in installed if entry.name == name]
        if not matches:
            available = sorted(set(injected) | {entry.name for entry in installed})
            raise OutputAdapterResolutionError(
                f"output adapter {name!r} is unavailable; installed adapters: "
                f"{', '.join(available) if available else '<none>'}"
            )
        if len(matches) > 1:
            raise OutputAdapterResolutionError(f"output adapter {name!r} has multiple installed providers")
        entry = matches[0]
        try:
            constructor = entry.load()
            adapter = constructor() if callable(constructor) else constructor
        except Exception as exc:
            raise OutputAdapterResolutionError(
                f"failed to load output adapter {name!r} from {entry.value!r}: {type(exc).__name__}: {exc}"
            ) from exc
        resolved[name] = validate_output_adapter(adapter, requested_name=name)
    return resolved


def write_output_adapters(
    adapters: Mapping[str, RecommendationOutputAdapter],
    configs: Mapping[str, Mapping[str, Any]],
    *,
    result: SweepResult,
    output_dir: Path,
) -> dict[str, tuple[Path, ...]]:
    """Invoke selected adapters and validate their reported artifact paths."""

    artifacts: dict[str, tuple[Path, ...]] = {}
    for name, adapter in adapters.items():
        try:
            reported = adapter.write(configs[name], result=result, output_dir=output_dir)
            if isinstance(reported, (str, bytes)):
                raise TypeError("must return a sequence of relative paths")
            paths: list[Path] = []
            for raw_path in reported:
                path = Path(raw_path)
                if path.is_absolute() or not path.parts or ".." in path.parts:
                    raise ValueError(f"returned path outside the output directory: {raw_path!r}")
                if not (output_dir / path).exists():
                    raise ValueError(f"reported missing artifact: {path}")
                paths.append(path)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise OutputAdapterExecutionError(f"output adapter {name!r} failed: {type(exc).__name__}: {exc}") from exc
        artifacts[name] = tuple(paths)
    return artifacts


__all__ = [
    "OUTPUT_ADAPTER_API_VERSION",
    "OUTPUT_ADAPTER_ENTRY_POINT_GROUP",
    "OutputAdapterExecutionError",
    "OutputAdapterResolutionError",
    "RecommendationOutputAdapter",
    "resolve_output_adapters",
    "validate_output_adapter",
    "write_output_adapters",
]
