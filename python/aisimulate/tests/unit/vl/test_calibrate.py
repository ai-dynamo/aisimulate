# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host calibration lowering: measured intervals become engine cost tables."""

import concurrent.futures
import json

import pytest

from aisimulate.vl.calibrate.python_frontend import Recorder, install_recorders
from aisimulate.vl.calibrate.rust_frontend import frontend_from_timing
from aisimulate.vl.calibrate.samples import Span, mean_active_concurrency, stage_costs, union_ms

pytestmark = pytest.mark.unit


def _burst(concurrency: int, service_ms: float, repeats: int) -> list[Span]:
    """`repeats` back-to-back bursts of `concurrency` jobs that overlap exactly."""
    spans = []
    for burst in range(repeats):
        start = burst * int(service_ms * 1e6) * 2
        spans.extend(Span(start, start + int(service_ms * 1e6)) for _ in range(concurrency))
    return spans


def test_stage_costs_take_the_alone_service_and_scale_by_sharing():
    curves = {1: _burst(1, 4.0, 12), 2: _burst(2, 6.0, 6)}
    assert mean_active_concurrency(curves[2]) == [2.0] * 12
    cost, scale = stage_costs(curves)
    assert cost.const_ms == pytest.approx(4.0)
    assert scale == pytest.approx([1.0, 1.5])
    # A drained burst is not steady: two jobs that barely overlap do not count for concurrency 2.
    drained = [
        span
        for burst in range(6)
        for span in (
            Span(burst * 4_000_000, burst * 4_000_000 + 1_000_000),
            Span(burst * 4_000_000 + 900_000, burst * 4_000_000 + 1_900_000),
        )
    ]
    with pytest.raises(ValueError, match="concurrency 2 has 0 steady samples"):
        stage_costs({1: curves[1], 2: drained})


def test_union_ms_counts_parallel_pool_spans_once():
    # Four decodes run in parallel for 10 ms, then the processor takes 10 ms; the
    # request's 25 ms wall leaves 5 ms of loop work, not max(25 - 50, 0).
    decodes = [Span(0, 10_000_000)] * 4
    processor = Span(12_000_000, 22_000_000)
    assert union_ms([*decodes, processor]) == pytest.approx(20.0)
    assert 25.0 - union_ms([*decodes, processor]) == pytest.approx(5.0)


def test_recorders_intercept_the_class_level_worker_entry_points():
    class Processor:
        @classmethod
        def _load_single_item(cls, data, modality):
            return (cls.__name__, data, modality)

        def process_and_combine_mm_data(self, base_output, mm_tokens, processor=None):
            return ([base_output], mm_tokens, {})

    decode_recorder, processor_recorder = Recorder(), Recorder()
    decode_recorder.enabled = processor_recorder.enabled = True
    restore = install_recorders(Processor, decode_recorder, processor_recorder)
    instance = Processor()
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        # The IO pool submits the classmethod resolved through the class, as SGLang does.
        assert pool.submit(instance._load_single_item, "url", "image").result() == ("Processor", "url", "image")
    assert instance.process_and_combine_mm_data("base", "tokens", processor=object()) == (["base"], "tokens", {})
    assert len(decode_recorder.spans) == 1 and len(processor_recorder.spans) == 1
    restore()
    assert "_load_single_item" in Processor.__dict__ and "process_and_combine_mm_data" in Processor.__dict__
    instance._load_single_item("url", "image")
    assert len(decode_recorder.spans) == 1


def _worker_rows(concurrency: int, service_ns: int, repeats: int, offset_ns: int) -> list[dict]:
    rows = []
    for burst in range(repeats):
        start = offset_ns + burst * service_ns * 2
        rows.extend(
            {
                "event": "boundary_span",
                "op": "rust_worker",
                "started_ns": start,
                "ended_ns": start + service_ns,
                "metadata": {"container": True, "ok": True},
            }
            for _ in range(concurrency)
        )
    return rows


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_rust_worker_spans_must_cover_every_worker_level(tmp_path):
    rows = _worker_rows(1, 3_000_000, 12, 0)
    rows.append({"started_ns": 0, "ended_ns": 2_000_000, "image_timings_ns": [[1_000_000, 1_000_000]]})
    with pytest.raises(ValueError, match="concurrency 2 has 0 steady samples.*concurrency 8 has 0 steady samples"):
        frontend_from_timing(_write(tmp_path / "timing.jsonl", rows), mm_workers=8)


def test_rust_worker_spans_lower_to_one_stage_with_a_full_scale(tmp_path):
    rows = []
    for concurrency in range(1, 5):
        rows.extend(_worker_rows(concurrency, (2 + concurrency) * 1_000_000, 12, concurrency * 10**9))
    # Legacy decode/patchify rows and other boundary ops are provenance, not worker occupancy.
    rows.append({"started_ns": 0, "ended_ns": 1_000_000, "image_timings_ns": [[500_000, 500_000]]})
    rows.append({"event": "boundary_span", "op": "rust_hash", "started_ns": 0, "ended_ns": 10, "metadata": {}})
    frontend = frontend_from_timing(_write(tmp_path / "timing.jsonl", rows), mm_workers=4)
    (stage,) = frontend.stages
    assert (stage.resource, stage.unit) == ("mm_worker", "request")
    assert stage.cost.const_ms == pytest.approx(3.0)
    assert stage.concurrency_scale == pytest.approx([1.0, 4 / 3, 5 / 3, 2.0])
