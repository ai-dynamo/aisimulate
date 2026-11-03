# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from audit_coding import bounds as coding_bounds
from audit_coding import validate_counts
from audit_matrix import errors, interval
from audit_shared_frontend import error as frontend_error
from audit_shared_frontend import latencies as frontend_latencies
from collect import qualify_cuda_trace_tables
from instrument import inject_forward_nvtx
from predict_matrix import client_ttft_ms, qualify_ready_event, replay_origin_ns
from profile_preflight import fields as profile_fields
from response_gate_diagnostic import gate_time, summary


def test_submission_replay_preserves_original_client_ttft_boundary():
    origin = 4_000_000_000_000_000
    reference = {"start_ns": str(origin + 3_000_000)}
    client_arrival = {"first_token_ms": 100.0, "arrival_time_ms": 3.0, "ttft_ms": 97.0}
    submitted_arrival = {"first_token_ms": 100.0, "arrival_time_ms": 10.0, "ttft_ms": 90.0}
    assert client_ttft_ms(client_arrival, reference, origin) == 97.0
    assert client_ttft_ms(submitted_arrival, reference, origin) == 97.0
    assert submitted_arrival["ttft_ms"] != client_ttft_ms(submitted_arrival, reference, origin)


def test_recorded_native_tail_requires_explicit_conservation():
    event = {"schema": "aisimulate.glm-rsi.output-ready.v2", "generated_output_tokens": 130,
             "output_tokens": 128, "discarded_tail_tokens": 2, "finished_len": 128}
    qualify_ready_event(event)
    with pytest.raises(ValueError, match="conservation"):
        qualify_ready_event({**event, "discarded_tail_tokens": 0})
    with pytest.raises(ValueError, match="conservation"):
        qualify_ready_event({**event, "finished_len": None})
    with pytest.raises(ValueError, match="explicit"):
        qualify_ready_event({"output_tokens": 130})


def test_independent_audit_pools_requests_and_preserves_stream_denominator():
    # Unequal reference magnitudes must not turn into a ratio of means.
    result = errors([(100, 50), (100, 1000)])
    assert result["n"] == 2 and result["mape"] == 95
    assert result["mean_absolute_ms"] == 475
    assert interval([
        {"at_ns": "1000000000", "output_tokens": 3},
        {"at_ns": "1127000000", "output_tokens": 128},
    ]) == (1000000000, 1.0)


@pytest.mark.parametrize("pairs", [[], [(1, 0)], [(float("nan"), 1)], [(1, float("inf"))]])
def test_independent_audit_rejects_undefined_control_scores(pairs):
    with pytest.raises(ValueError):
        errors(pairs)


def test_submission_zero_based_origin_restores_client_transport_delay():
    client_origin = 4_000_000_000_000_000
    reference = {"start_ns": str(client_origin + 3_000_000)}
    measured = [
        {"start_ns": str(client_origin), "dispatch": {"at_ns": str(client_origin + 20_000_000)}},
        {"start_ns": str(client_origin + 3_000_000), "dispatch": {"at_ns": str(client_origin + 10_000_000)}},
    ]
    assert replay_origin_ns(measured, "client_start") == client_origin
    submission_origin = replay_origin_ns(measured, "server_submission")
    assert submission_origin == client_origin + 10_000_000
    assert client_ttft_ms({"first_token_ms": 90.0}, reference, submission_origin) == 97.0
    assert client_ttft_ms({"first_token_ms": 100.0}, reference, client_origin) == 97.0


def test_independent_coding_audit_preserves_tail_counts_and_units():
    value = {"schema": "aisimulate.glm-rsi.output-ready.v2", "generated_output_tokens": 18,
             "output_tokens": 16, "discarded_tail_tokens": 2, "finished_len": 16}
    validate_counts(value)
    with pytest.raises(ValueError, match="conservation"):
        validate_counts({**value, "discarded_tail_tokens": 0})
    assert coding_bounds([
        {"at_ns": "1000000000", "output_tokens": 3},
        {"at_ns": "1030000000", "output_tokens": 16},
    ]) == (1000000000, 1030000000, 2.0)


def test_delivery_gate_uses_only_work_known_when_output_becomes_ready():
    arrivals, free_at = [1.0, 5.0, 15.0], [11.0, 21.0, 31.0]
    assert gate_time(0.0, arrivals, free_at) == 0.0
    assert gate_time(3.0, arrivals, free_at) == 11.0
    assert gate_time(5.0, arrivals, free_at) == 21.0  # Explicit input-first tie.
    assert gate_time(7.0, arrivals, free_at) == 21.0
    assert gate_time(40.0, arrivals, free_at) == 40.0
    assert summary([(0.0, 2.0)])["mape"] == 100.0  # Defined zero prediction stays scored.


def test_native_shared_frontend_audit_keeps_zero_delivery_intervals():
    row = {"arrival_time_ms": 1, "first_token_ms": 11, "last_token_ms": 11,
           "terminal_time_ms": 11, "ttft_ms": 10, "itl_ms": 0,
           "output_length": 128, "requested_output_length": 128}
    assert frontend_latencies(row) == {"ttft": 10, "tpot": 0}
    assert frontend_error(0, 0.01) == 100
    assert frontend_error(0, 0) is None
    with pytest.raises(ValueError, match="causal"):
        frontend_latencies({**row, "terminal_time_ms": 10})
    with pytest.raises(ValueError, match="conserved"):
        frontend_latencies({**row, "requested_output_length": 129})


def test_profiler_cpu_preflight_requires_one_named_protocol():
    source = "class ProfileReq:\n    output_dir: str\n    record_shapes: bool = False\n"
    assert profile_fields(source, "ProfileReq") == {"output_dir", "record_shapes"}
    with pytest.raises(ValueError, match="Unknown runtime class"):
        profile_fields(source + source, "ProfileReq")
    with pytest.raises(ValueError, match="Unknown runtime class"):
        profile_fields(source, "Missing")


def test_forward_nvtx_patch_preserves_backend_label_and_rejects_reinstallation():
    source = "def forward(self, batch):\n    ctx = profile_range(build_step_span_name(batch, is_draft_worker=False))\n"
    modified = inject_forward_nvtx(source)
    assert 'build_step_span_name(batch, is_draft_worker=False)' in modified
    assert '__import__("rsi_forward_nvtx").span(' in modified
    with pytest.raises(ValueError, match="already installed"):
        inject_forward_nvtx(modified)
    with pytest.raises(ValueError, match="Unknown shaped"):
        inject_forward_nvtx("x = other_span()")


def test_cuda_trace_probe_requires_gpu_activity_and_projected_markers(tmp_path):
    kernels = tmp_path / "kernels.csv"
    projected = tmp_path / "projected.csv"
    kernels.write_text("Device,Duration (ns)\n" + "".join(f"B300 ({i}),100\n" for i in range(8)))
    header = "Name,Projected Duration (ns)\n"
    rows = [f"rsi-smoke-replay-gpu-{i},100\n" for i in range(8)]
    projected.write_text(header + "".join(rows))
    assert qualify_cuda_trace_tables(kernels, projected)["gpu_projected_marker_ids"] == list(range(8))
    projected.write_text(header + "".join(rows[:-1]))
    with pytest.raises(ValueError, match="GPU-projected"):
        qualify_cuda_trace_tables(kernels, projected)
    projected.write_text(header + "".join(rows))
    kernels.write_text("Device,Duration (ns)\n" + "".join(f"B300 ({i}),100\n" for i in range(7)))
    with pytest.raises(ValueError, match="eight devices"):
        qualify_cuda_trace_tables(kernels, projected)


def test_forward_nvtx_receipt_never_reads_weights_and_balances_failed_span(tmp_path, monkeypatch):
    calls = []
    cuda = SimpleNamespace(current_device=lambda: 0, nvtx=SimpleNamespace(
        range_push=lambda name: calls.append("push"), range_pop=lambda: calls.append("pop")))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(
        sys.modules, "sglang.srt.utils.nvtx_utils", SimpleNamespace(profile_range=lambda _: nullcontext()))
    monkeypatch.setenv("RSI_FORWARD_NVTX_ROOT", str(tmp_path))
    path = Path(__file__).with_name("rsi_forward_nvtx.py")
    spec = importlib.util.spec_from_file_location("isolated_nvtx_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tensor = SimpleNamespace(dtype="torch.bfloat16", shape=(4, 4))
    model = SimpleNamespace(named_parameters=lambda: [("decoder.weight", tensor)],
                            parameters=lambda: [tensor], named_modules=lambda: [])
    runner = SimpleNamespace(model=model, is_draft_worker=True,
                             ps=SimpleNamespace(tp_rank=0, tp_size=8, gpu_id=0))
    with pytest.raises(RuntimeError, match="test failure"), module.span("test", runner=runner):
        raise RuntimeError("test failure")
    assert calls == ["push", "pop"]
    receipt = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert receipt["parameter_dtype_counts"] == {"torch.bfloat16": 1}
    assert receipt["draft_parameters"] == [{"name": "decoder.weight", "dtype": "torch.bfloat16", "shape": [4, 4]}]
