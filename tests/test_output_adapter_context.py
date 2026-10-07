# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A live output adapter can receive the run context without breaking legacy adapters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from aisimulate.output_adapter import (
    OUTPUT_ADAPTER_API_VERSION,
    OutputAdapterExecutionError,
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
    assert seen["context"].workload == WORKLOAD
    assert seen["context"].workload is not WORKLOAD
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

    assert seen["context"].workload == WORKLOAD


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

    resolve_output_callbacks({"live": {}}, injected={"live": _adapter("live", subscribe)})

    assert seen == [None]


# -- signature compatibility (review of aisimulate#384) ------------------------


def _legacy_positional_only(self: Any, config: Any, context: Any = None, /) -> None:
    legacy_calls.append(("positional_only", config, context))


def _legacy_var_positional(self: Any, config: Any, *context: Any) -> None:
    legacy_calls.append(("var_positional", config, context))


def _legacy_config_named_context(self: Any, context: Any) -> None:
    legacy_calls.append(("named_context", context, None))


legacy_calls: list[Any] = []


@pytest.mark.parametrize(
    "subscribe, label, expected_second",
    [
        (_legacy_positional_only, "positional_only", None),
        (_legacy_var_positional, "var_positional", ()),
        (_legacy_config_named_context, "named_context", None),
    ],
)
def test_legacy_signatures_that_cannot_take_context_keep_the_legacy_call(subscribe, label, expected_second) -> None:
    legacy_calls.clear()

    resolve_output_callbacks(
        {"legacy": {"z": 3}},
        injected={"legacy": _adapter("legacy", subscribe)},
        context=RecommendationOutputContext(workload=WORKLOAD),
    )

    assert legacy_calls == [(label, {"z": 3}, expected_second)]


def test_positional_only_context_with_var_keyword_does_not_break_subscription() -> None:
    calls: list[Any] = []

    def subscribe(self: Any, config: Any, context: Any = None, /, **kwargs: Any) -> None:
        calls.append((config, context, kwargs))

    resolve_output_callbacks(
        {"live": {"q": 1}},
        injected={"live": _adapter("live", subscribe)},
        context=RecommendationOutputContext(workload=WORKLOAD),
    )

    # Whether the context arrives via **kwargs depends on the interpreter's signature binding
    # (3.13 accepts it, 3.11 does not); either way the call must succeed and config arrive.
    assert len(calls) == 1
    assert calls[0][:2] == ({"q": 1}, None)


def test_a_type_error_from_the_adapter_body_is_not_retried_as_legacy() -> None:
    calls: list[Any] = []

    def subscribe(self: Any, config: Any, context: Any) -> None:
        calls.append(context)
        raise TypeError("adapter bug")

    with pytest.raises(OutputAdapterExecutionError, match="adapter bug"):
        resolve_output_callbacks(
            {"live": {}},
            injected={"live": _adapter("live", subscribe)},
            context=RecommendationOutputContext(workload=WORKLOAD),
        )

    assert len(calls) == 1


# -- isolation (review of aisimulate#384) ---------------------------------------


def test_subscribers_cannot_change_the_workload_the_search_uses_or_each_other() -> None:
    workload = SimpleNamespace(isl=1024, concurrency=8)
    seen: list[tuple[int, int]] = []

    def mutating(self: Any, config: Any, context: RecommendationOutputContext) -> None:
        context.workload.isl = 1
        context.workload.concurrency = 64

    def observing(self: Any, config: Any, context: RecommendationOutputContext) -> None:
        seen.append((context.workload.isl, context.workload.concurrency))

    resolve_output_callbacks(
        {"a": {}, "b": {}},
        injected={"a": _adapter("a", mutating), "b": _adapter("b", observing)},
        context=RecommendationOutputContext(workload=workload),
    )

    assert (workload.isl, workload.concurrency) == (1024, 8)
    assert seen == [(1024, 8)]


def test_keyword_only_context_is_passed() -> None:
    seen: dict[str, Any] = {}

    def subscribe(self: Any, config: Any, *, context: RecommendationOutputContext) -> None:
        seen.update(config=config, context=context)

    resolve_output_callbacks(
        {"live": {"k": 1}},
        injected={"live": _adapter("live", subscribe)},
        context=RecommendationOutputContext(workload=WORKLOAD),
    )

    assert seen["config"] == {"k": 1}
    assert seen["context"].workload == WORKLOAD


def test_nested_workload_state_is_isolated_between_subscribers() -> None:
    from copy import deepcopy

    workload = SimpleNamespace(isl=1024, load=[1, 2, {"concurrency": 8}])
    observed: list[Any] = []

    def mutating(self: Any, config: Any, context: RecommendationOutputContext) -> None:
        context.workload.load.append(3)
        context.workload.load[2]["concurrency"] = 64

    def observing(self: Any, config: Any, context: RecommendationOutputContext) -> None:
        observed.append(deepcopy(context.workload.load))

    resolve_output_callbacks(
        {"a": {}, "b": {}},
        injected={"a": _adapter("a", mutating), "b": _adapter("b", observing)},
        context=RecommendationOutputContext(workload=workload),
    )

    assert workload.load == [1, 2, {"concurrency": 8}]
    assert observed == [[1, 2, {"concurrency": 8}]]


def test_mutating_a_retained_context_from_a_callback_does_not_reach_the_search() -> None:
    workload = SimpleNamespace(isl=1024, load=[1, 2])
    retained: list[RecommendationOutputContext] = []

    def subscribe(self: Any, config: Any, context: RecommendationOutputContext) -> RecommendationOutputCallbacks:
        retained.append(context)

        def on_candidate(record: Any) -> None:
            retained[0].workload.isl = 1
            retained[0].workload.load.append(99)

        return RecommendationOutputCallbacks(on_candidate=on_candidate)

    callbacks = resolve_output_callbacks(
        {"live": {}},
        injected={"live": _adapter("live", subscribe)},
        context=RecommendationOutputContext(workload=workload),
    )
    assert callbacks.on_candidate is not None
    callbacks.on_candidate(SimpleNamespace(candidate_id="c1"))

    assert retained[0].workload.isl == 1  # the adapter's own copy did change
    assert (workload.isl, workload.load) == (1024, [1, 2])
