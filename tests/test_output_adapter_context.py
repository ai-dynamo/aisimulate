# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A live output adapter can receive the run context without breaking legacy adapters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from aisimulate.output_adapter import (
    OUTPUT_ADAPTER_API_VERSION,
    RecommendationOutputCallbacks,
    RecommendationOutputContext,
    resolve_output_callbacks,
)

WORKLOAD = SimpleNamespace(isl=1024, osl=256)


class _Base:
    api_version = OUTPUT_ADAPTER_API_VERSION

    def write(self, config: Any, *, result: Any, output_dir: Any) -> list[str]:
        return []


def _adapter(name: str, subscribe: Any) -> Any:
    namespace = {"name": name, "subscribe": subscribe}
    return type("Adapter", (_Base,), namespace)()


def test_context_is_passed_when_subscribe_accepts_it() -> None:
    seen: dict[str, Any] = {}
    candidates: list[Any] = []

    def subscribe(self: Any, config: Any, context: RecommendationOutputContext) -> Any:
        seen.update(config=config, context=context)
        return RecommendationOutputCallbacks(on_candidate=candidates.append)

    callbacks = resolve_output_callbacks(
        {"live": {"x": 1}},
        injected={"live": _adapter("live", subscribe)},
        context=RecommendationOutputContext(workload=WORKLOAD),
    )

    assert seen["config"] == {"x": 1}
    assert seen["context"].workload is WORKLOAD
    assert callbacks.on_candidate is not None
    callbacks.on_candidate(SimpleNamespace(candidate_id="c1"))
    assert [c.candidate_id for c in candidates] == ["c1"]


def test_context_is_passed_through_var_keyword_subscribe() -> None:
    seen: dict[str, Any] = {}

    def subscribe(self: Any, config: Any, **kwargs: Any) -> None:
        seen.update(kwargs)

    resolve_output_callbacks(
        {"live": {}},
        injected={"live": _adapter("live", subscribe)},
        context=RecommendationOutputContext(workload=WORKLOAD),
    )

    assert seen["context"].workload is WORKLOAD


def test_legacy_subscribe_without_context_still_works() -> None:
    calls: list[Any] = []

    def subscribe(self: Any, config: Any) -> None:
        calls.append(config)

    callbacks = resolve_output_callbacks(
        {"legacy": {"y": 2}},
        injected={"legacy": _adapter("legacy", subscribe)},
        context=RecommendationOutputContext(workload=WORKLOAD),
    )

    assert calls == [{"y": 2}]
    assert callbacks.on_candidate is None and callbacks.on_round is None


def test_no_context_means_subscribe_is_called_as_before() -> None:
    seen: list[Any] = []

    def subscribe(self: Any, config: Any, context: Any = None) -> None:
        seen.append(context)

    resolve_output_callbacks(
        {"live": {}}, injected={"live": _adapter("live", subscribe)}
    )

    assert seen == [None]
