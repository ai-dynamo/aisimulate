# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY lazy CUDA events and native calls; no GPU/performance claims."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from collector.glm53flash_contract import BACKENDS, canonical_json
from collector.glm53flash_observer import NativeOperationObserver, NativeWorkload

from .test_glm53flash_contract import manifest, sample_row

pytestmark = pytest.mark.unit


class LazyCuda:
    def __init__(self):
        self.clock, self.stream, self.synced = 0, 7, False
        self.log, self.events = [], []

    def current_stream(self):
        return self.stream

    def is_current_stream_capturing(self):
        return False

    def synchronize(self):
        self.log.append(("sync",))
        self.synced = True

    def Event(self, *, enable_timing):  # noqa: N802 - actual public Torch API
        assert enable_timing
        cuda, index = self, len(self.events)

        class Event:
            initialized = False

            def record(self, stream):
                assert stream == cuda.stream
                if not self.initialized:
                    cuda.log.append(("initialize", index))
                    self.initialized = True
                cuda.log.append(("record", index))
                cuda.synced = False
                self.time = cuda.clock

            def elapsed_time(self, end):
                assert cuda.synced, "event read before existing observer synchronization"
                cuda.log.append(("read", index))
                return end.time - self.time

        event = Event()
        self.events.append(event)
        self.log.append(("construct", index))
        return event


def fixture(*, pool=True, collective=True, included=False):
    cuda = LazyCuda()
    row = sample_row()
    geometry = json.loads(row["geometry"])
    geometry["backend"] = "sglang"
    row.update(
        backend="sglang",
        backend_version=BACKENDS["sglang"][0],
        backend_revision=BACKENDS["sglang"][1],
        geometry=canonical_json(geometry),
    )
    graph = manifest(row)
    if collective:
        graph["phases"]["context"].append(
            {
                "name": "attention_allreduce_0",
                "component": "primitive",
                "geometry": canonical_json(
                    {
                        "backend": "sglang",
                        "checkpoint_format": "fp8",
                        "role": "allreduce",
                        "token_selection": "all_scheduled",
                    }
                ),
            }
        )
    observer = NativeOperationObserver(graph, row, 0, torch_module=SimpleNamespace(cuda=cuda))

    class Native:
        failure = False
        second_failure = False

        def first(self, value):
            cuda.log.append(("native", "first"))
            cuda.clock += 1
            if included:
                self.second(value)
            if collective:
                self.reduce(value)
            if self.failure:
                raise RuntimeError("TEST_ONLY native failure")
            cuda.clock += 1
            return value

        def second(self, value):
            cuda.log.append(("native", "second"))
            if self.second_failure:
                raise RuntimeError("TEST_ONLY native second-part failure")
            cuda.clock += 3
            return value

        def reduce(self, value):
            cuda.log.append(("native", "reduce"))
            cuda.clock += 2
            return value

    native = Native()
    observer.wrap(native, "first", "attention_0")
    observer.wrap(native, "second", "attention_0", included_by_same_operation=included)
    if collective:
        observer.wrap_collective(native, "reduce", ("attention_allreduce_0",))
    if pool:
        observer.enable_prefill_event_pool()
    return observer, native, cuda


def workload(sample=0, *, query=128):
    return NativeWorkload("context", 1, query, 0, "full_prefill", ("r",), (), sample, sample + 1)


def run(observer, native, sample=0, *, query=128, included=False):
    observer.begin(workload(sample, query=query))
    token = object()
    assert native.first(token) is token
    if not included:
        assert native.second(token) is token
    return observer.end()


def finish(observer):
    # Simulate the real helper's final setup/whole reads after observer.end().
    for name in ("setup", "whole"):
        start, end = observer.event_pool.pairs[name]
        start.elapsed_time(end)
    return observer.event_pool.finish()


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("profiled", [False, True])
def test_direct_retained_calls_preserve_events_native_results_and_profiled_ranges(pool, profiled):
    observer, native, cuda = fixture(pool=pool)
    observer.begin(workload())
    marker = object()
    observer.profiler = marker if profiled else None

    @contextmanager
    def scope(name, *, scope_name):
        assert profiled, "retained native calls must not allocate a profiling scope"
        cuda.log.append(("scope-enter", name))
        try:
            yield
        finally:
            cuda.log.append(("scope-exit", name))

    observer._range = scope
    before = len(cuda.log)
    token = object()
    assert native.first(token) is token
    assert native.second(token) is token
    measured = cuda.log[before:]
    assert [entry[1] for entry in measured if entry[0] == "native"] == ["first", "reduce", "second"]
    assert len([entry for entry in measured if entry[0] == "record"]) == 6
    scopes = [entry for entry in measured if entry[0].startswith("scope-")]
    assert scopes == (
        [
            ("scope-enter", "attention_0"),
            ("scope-enter", "attention_allreduce_0"),
            ("scope-exit", "attention_allreduce_0"),
            ("scope-exit", "attention_0"),
            ("scope-enter", "attention_0"),
            ("scope-exit", "attention_0"),
        ]
        if profiled
        else []
    )
    observer.profiler = None  # This TEST_ONLY marker is not a native Torch profiler.
    rows = observer.end()
    assert sum(row["latency"] for row in rows) == 7
    if pool:
        finish(observer)


def test_pooled_compute_reuses_pre_call_stream_and_still_checks_the_return_stream():
    observer, native, cuda = fixture(collective=False)
    observer.begin(workload())
    original_getter = cuda.current_stream
    observations = []

    def current_stream():
        observations.append(len([entry for entry in cuda.log if entry[0] == "native"]))
        return original_getter()

    cuda.current_stream = current_stream
    token = object()
    assert native.first(token) is token
    assert observations == [0, 1]
    assert native.second(token) is token
    assert observations == [0, 1, 1, 2]
    observer.end()
    finish(observer)


@pytest.mark.parametrize("included", [False, True])
def test_distinct_lazy_handles_initialized_before_native_and_reused_only_after_all_reads(included):
    observer, native, cuda = fixture(included=included)
    first = run(observer, native, included=included)
    pairs = dict(observer.event_pool.pairs)
    assert len({id(event) for pair in pairs.values() for event in pair}) == 2 * len(pairs)
    first_native = next(index for index, event in enumerate(cuda.log) if event[0] == "native")
    assert all(index < first_native for index, event in enumerate(cuda.log) if event[0] in ("initialize", "construct"))
    assert sum(row["latency"] for row in first) == 7
    assert next(row for row in first if row["component"] == "attention")["latency"] == 5
    with pytest.raises(RuntimeError, match="outstanding"):
        observer.begin(workload(1))
    evidence = finish(observer)
    assert evidence["bootstrap_sample"] == 0
    before = len(cuda.log)
    second = run(observer, native, 5, included=included)
    assert not any(event[0] in ("initialize", "construct") for event in cuda.log[before:])
    assert observer.event_pool.pairs == pairs
    assert [row["latency"] for row in second] == [row["latency"] for row in first]
    finish(observer)


@pytest.mark.parametrize("defect", ["order", "arity", "missing", "identity", "exception", "stream"])
def test_changed_native_path_is_rejected_without_recycling_outstanding_events(defect):
    observer, native, cuda = fixture(collective=False)
    run(observer, native)
    finish(observer)
    if defect == "identity":
        native.first = lambda value: value
        with pytest.raises(RuntimeError, match="identity changed"):
            observer.begin(workload(1))
        return
    observer.begin(workload(1))
    before = len([event for event in cuda.log if event[0] == "native"])
    if defect == "exception":
        native.failure = True
    if defect == "stream":
        cuda.stream += 1
    with pytest.raises(
        RuntimeError, match="identity/order/arity|incomplete|TEST_ONLY native failure|active native stream"
    ):
        if defect == "order":
            native.second(object())
        elif defect == "arity":
            native.first(value=object())
        else:
            native.first(object())
            observer.end()
    if defect in ("order", "arity", "stream"):
        assert len([event for event in cuda.log if event[0] == "native"]) == before
    with pytest.raises(RuntimeError, match="requires synchronization"):
        observer.event_pool.finish()
    observer.close()
    with pytest.raises(RuntimeError, match="outstanding or closed"):
        observer.event_pool.begin(workload(2))


def test_new_geometry_requires_excluded_bootstrap_and_may_have_its_own_order():
    observer, native, _ = fixture(collective=False)
    run(observer, native)
    finish(observer)
    with pytest.raises(RuntimeError, match="excluded bootstrap"):
        observer.begin(workload(5, query=256))
    observer.begin(workload(0, query=256))
    native.second(object())
    native.first(object())
    observer.end()
    finish(observer)
    observer.begin(workload(5, query=256))
    native.second(object())
    native.first(object())
    observer.end()
    finish(observer)


def test_failed_second_part_cannot_hide_behind_first_part_complete_unit_coverage():
    observer, native, _ = fixture(collective=False)
    run(observer, native)
    finish(observer)
    observer.begin(workload(1))
    native.first(object())
    native.second_failure = True
    with pytest.raises(RuntimeError, match="second-part failure"):
        native.second(object())
    # Both parts contribute to the same physical unit. Set coverage alone
    # would accept the first part, but the outstanding second call must fail.
    with pytest.raises(RuntimeError, match="did not complete"):
        observer.end()
    assert observer.event_pool.active and not observer.event_pool.read_complete


def test_pool_rejects_unbounded_hooks_late_registration_and_other_backends():
    observer, native, _ = fixture(pool=False, collective=False)
    observer.wrap_collective(native, "reduce")
    with pytest.raises(RuntimeError, match="bounded"):
        observer.enable_prefill_event_pool()
    observer, native, _ = fixture(collective=False)
    with pytest.raises(RuntimeError, match="already fixed"):
        observer.wrap(native, "reduce", "attention_0")
    observer, _, _ = fixture(pool=False)
    observer.provenance["backend"] = "vllm"
    with pytest.raises(RuntimeError, match="SGLang"):
        observer.enable_prefill_event_pool()


def test_default_observer_retains_original_lazy_event_lifecycle():
    observer, native, cuda = fixture(pool=False)
    run(observer, native)
    count = len(cuda.events)
    run(observer, native, 5)
    assert len(cuda.events) == 2 * count
    assert observer.event_pool is None
