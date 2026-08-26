# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.replay import ReplayOutputRequirements


def _engine() -> dict:
    return {
        "model": "example/model",
        "hardware": "h200_sxm",
        "context_length": 1024,
        "workers": {
            "aggregated": {
                "kv_cache": {"capacity": {"type": "fixed", "blocks": 128}},
                "timing": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
            }
        },
    }


def _run(config: dict):
    parsed = CorePredictionConfig.model_validate(config)
    spec = prediction_to_replay_spec(parsed)
    return (
        EngineReplayRunnerFactory()
        .create(0)
        .run(
            spec,
            output_requirements=ReplayOutputRequirements(
                include_raw_report=True,
                capture_per_request=True,
            ),
        )
    )


def test_prediction_spec_separates_perf_identity_from_fixed_timing() -> None:
    parsed = CorePredictionConfig.model_validate({"engine": _engine()})
    deployment = prediction_to_replay_spec(parsed).backend_deployment

    assert deployment.performance_model_metadata == {
        "aggregated": {
            "provider": "aic",
            "config": {
                "backend": "vllm",
                "backend_version": None,
                "system": "h200_sxm",
                "model_path": "example/model",
                "tp_size": 1,
                "attention_dp_size": 1,
                "moe_tp_size": None,
                "moe_ep_size": None,
                "nextn": None,
            },
        }
    }
    assert "aic_model_path" not in deployment.agg_engine_args
    assert deployment.agg_engine_args["timing_model"]["type"] == "fixed"


def test_prediction_compiles_aic_engine_and_cached_prefix_controls() -> None:
    parsed = CorePredictionConfig.model_validate(
        {
            "traffic": {
                "source": {
                    "type": "synthetic",
                    "input_tokens": 8,
                    "output_tokens": 2,
                    "cached_prefix_tokens": 3,
                },
                "load": {"type": "concurrency", "concurrency": 1},
                "stop": {"requests": 1},
            },
            "engine": {
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "context_length": 1024,
                "nextn": 2,
                "nextn_accepted": 1.25,
                "enable_chunked_prefill": True,
                "gemm_quant_mode": "fp8",
                "kvcache_quant_mode": "fp8",
                "free_gpu_memory_fraction": 0.85,
                "workers": {"aggregated": {}},
            },
        }
    )

    spec = prediction_to_replay_spec(parsed)
    args = spec.backend_deployment.agg_engine_args

    assert spec.workload["cached_prefix_tokens"] == 3
    assert args["max_model_len"] == 1024
    assert args["enable_chunked_prefill"] is True
    assert args["aic_nextn"] == 2
    assert args["aic_nextn_accepted"] == 1.25
    assert args["aic_gemm_dtype"] == "fp8"
    assert args["aic_kv_cache_dtype"] == "fp8"
    assert args["gpu_memory_utilization"] == 0.85


def test_engine_stack_runs_ordered_synthetic_sessions() -> None:
    report = _run(
        {
            "traffic": {
                "source": {
                    "type": "synthetic-session",
                    "new_input_tokens_per_turn": 8,
                    "output_tokens_per_turn": 2,
                    "session": {"turns": 3, "inter_turn_delay_ms": 5},
                },
                "load": {"type": "concurrency", "concurrency": 1},
                "stop": {"sessions": 2},
            },
            "engine": _engine(),
        }
    )

    assert report.metrics["completed_requests"] == 6
    records = report.metadata["native_report"]["per_request"]
    assert [(row["session_id"], row["turn_index"]) for row in records] == [
        ("session_0", 0),
        ("session_0", 1),
        ("session_0", 2),
        ("session_1", 0),
        ("session_1", 1),
        ("session_1", 2),
    ]
    assert [row["input_length"] for row in records[:3]] == [8, 18, 28]


def test_engine_stack_reuses_exact_cached_prefix_from_public_traffic() -> None:
    engine = _engine()
    engine["workers"]["aggregated"]["kv_cache"]["block_size"] = 4
    report = _run(
        {
            "traffic": {
                "source": {
                    "type": "synthetic",
                    "input_tokens": 8,
                    "output_tokens": 2,
                    "cached_prefix_tokens": 4,
                },
                "load": {"type": "concurrency", "concurrency": 1},
                "stop": {"requests": 2},
            },
            "engine": engine,
        }
    )

    records = report.metadata["native_report"]["per_request"]
    assert [row["reused_input_tokens"] for row in records] == [0, 4]


def test_engine_stack_runs_mooncake_delta(tmp_path) -> None:
    trace = tmp_path / "delta.jsonl"
    rows = [
        {
            "request_id": "r0",
            "session_id": "s0",
            "input_length": 4,
            "output_length": 2,
            "hash_ids": [1],
            "timestamp": 0,
        },
        {
            "request_id": "r1",
            "session_id": "s0",
            "input_length": 2,
            "output_length": 1,
            "hash_ids": [2],
            "delay": 5,
        },
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = _run(
        {
            "traffic": {
                "source": {
                    "type": "trace",
                    "paths": [str(trace)],
                    "format": "mooncake-delta",
                    "block_size": 4,
                },
                "load": {"type": "trace_timestamps"},
            },
            "engine": _engine(),
        }
    )

    assert report.metrics["completed_requests"] == 2
    records = report.metadata["native_report"]["per_request"]
    assert records[1]["input_length"] == 8
    assert records[1]["arrival_time_ms"] >= records[0]["last_token_ms"] + 5


def test_engine_stack_runs_applied_compute_agentic(tmp_path) -> None:
    trace = tmp_path / "applied.jsonl"
    trace.write_text(
        json.dumps(
            {
                "num_turns": 1,
                "input_prompt_length": 8,
                "assistant_response_length": [2],
                "tool_call_output_length": [3],
                "tool_call_latency": [0.005],
                "final_assistant_response_length": 1,
            }
        )
        + "\n"
    )

    report = _run(
        {
            "traffic": {
                "source": {
                    "type": "trace",
                    "paths": [str(trace)],
                    "format": "applied_compute_agentic",
                    "block_size": 4,
                },
                "load": {"type": "concurrency", "concurrency": 1},
                "stop": {"max_virtual_time_seconds": 60},
            },
            "engine": _engine(),
        }
    )

    assert report.metrics["completed_requests"] == 2
    records = report.metadata["native_report"]["per_request"]
    assert records[1]["input_length"] == 13


def test_engine_stack_runs_legacy_agentic_mooncake(tmp_path) -> None:
    trace = tmp_path / "agentic.jsonl"
    rows = [
        {
            "request_id": "root",
            "session_id": "session",
            "input_length": 4,
            "output_length": 2,
            "hash_ids": [1],
            "timestamp": 0,
        },
        {
            "request_id": "child",
            "session_id": "session",
            "input_length": 4,
            "output_length": 1,
            "hash_ids": [1],
            "wait_for": ["root"],
            "delay": 2,
            "tool_wait_ms": 3,
        },
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = _run(
        {
            "traffic": {
                "source": {
                    "type": "trace",
                    "paths": [str(trace)],
                    "format": "agentic_mooncake",
                    "block_size": 4,
                },
                "load": {"type": "trace_timestamps", "speedup": 1},
            },
            "engine": _engine(),
        }
    )

    assert report.metrics["completed_requests"] == 2
    records = report.metadata["native_report"]["per_request"]
    by_id = {record["request_id"]: record for record in records}
    root, child = by_id["root"], by_id["child"]
    assert child["arrival_time_ms"] >= root["last_token_ms"] + 5


def test_engine_stack_runs_native_dynamo_trace(tmp_path) -> None:
    trace = tmp_path / "dynamo.jsonl"
    rows = []
    for index, start in enumerate((100, 120)):
        rows.append(
            {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": start + 10,
                "request": {
                    "request_id": f"r{index}",
                    "output_tokens": 2,
                    "request_received_ms": start,
                    "total_time_ms": 10,
                    "replay": {
                        "trace_block_size": 4,
                        "input_length": 4,
                        "input_sequence_hashes": [11],
                    },
                },
            }
        )
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = _run(
        {
            "traffic": {
                "source": {
                    "type": "trace",
                    "paths": [str(trace)],
                    "format": "dynamo",
                    "block_size": 4,
                },
                "load": {"type": "trace_timestamps", "speedup": 2},
            },
            "engine": _engine(),
        }
    )

    assert report.metrics["completed_requests"] == 2
    records = report.metadata["native_report"]["per_request"]
    assert records[1]["arrival_time_ms"] == 10
