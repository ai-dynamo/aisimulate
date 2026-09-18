# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.replay.reporting import format_report_table
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.replay import ReplayOutputRequirements

_TRACE_FIXTURES = Path(__file__).parent / "e2e/configs/unified_cli/fixtures/traces"


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


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
def test_prediction_preserves_context_limit_for_all_backends(backend: str, mode: str) -> None:
    engine = _engine()
    worker = engine["workers"]["aggregated"]
    roles = ["aggregated"] if mode == "aggregated" else ["prefill", "decode"]
    engine.update(backend=backend, mode=mode, workers=dict.fromkeys(roles, worker))
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate({"engine": engine})).backend_deployment
    for role in roles:
        payload = getattr(deployment, f"{'agg' if role == 'aggregated' else role}_engine_args")
        assert payload["max_model_len"] == 1024


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
                "forward_model": "op_level",
            },
        }
    }
    assert "aic_model_path" not in deployment.agg_engine_args
    assert deployment.agg_engine_args["timing_model"]["type"] == "fixed"


def test_aic_timing_power_publication_tracks_current_data_coverage() -> None:
    report = _run(
        {
            "traffic": {
                "source": {
                    "type": "synthetic",
                    "input_tokens": 128,
                    "output_tokens": 4,
                },
                "load": {"type": "concurrency", "concurrency": 1},
                "stop": {"requests": 1},
            },
            "engine": {
                "mode": "aggregated",
                "model": "Qwen/Qwen3-30B-A3B",
                "hardware": "b200_sxm",
                "backend": "vllm",
                "backend_version": "current",
                "context_length": 4096,
                "workers": {
                    "aggregated": {
                        "parallelism": {
                            "replicas": 1,
                            "tensor": 4,
                            "pipeline": 1,
                            "attention_data": 1,
                            "moe_tensor": 1,
                            "moe_expert": 4,
                        },
                        "scheduler": {
                            "max_batched_tokens": 8192,
                            "max_sequences": 256,
                        },
                        "kv_cache": {
                            "block_size": 64,
                            "prefix_caching": True,
                            "capacity": {"type": "fixed", "blocks": 4096},
                        },
                        "timing": {"type": "default"},
                    }
                },
            },
        }
    )

    coverage = report.metrics["power_coverage"]
    assert coverage == 0.0
    assert report.metrics["power_w"] is None
    diagnostics = report.metadata["native_report"]["power_diagnostics"]
    assert diagnostics["schema_version"] == "1.0"
    assert diagnostics["scope"] == "active_forward_pass_per_gpu"
    assert diagnostics["power_coverage"] == pytest.approx(coverage)
    assert [phase["name"] for phase in diagnostics["phases"]] == [
        "prefill",
        "decode",
    ]
    phase_energies = [phase["energy_wms"] for phase in diagnostics["phases"] if "energy_wms" in phase]
    if phase_energies:
        assert diagnostics["energy_wms"] == pytest.approx(sum(phase_energies))
    else:
        assert "energy_wms" not in diagnostics
    assert diagnostics["latency_ms"] == pytest.approx(sum(phase["latency_ms"] for phase in diagnostics["phases"]))
    assert diagnostics["covered_latency_ms"] == pytest.approx(
        sum(phase["covered_latency_ms"] for phase in diagnostics["phases"])
    )
    for phase in diagnostics["phases"]:
        operations = phase["operations"]
        assert [operation["name"] for operation in operations] == sorted(operation["name"] for operation in operations)
        for operation in operations:
            assert operation["source_kind"] in {
                "measured",
                "transferred",
                "modeled",
                "mixed",
                "other",
                "missing",
            }
            if operation["status"] == "missing":
                assert "energy_wms" not in operation
                assert operation["uncovered_reason"]


def test_power_report_table_surfaces_available_power_and_coverage() -> None:
    table = format_report_table({"power_w": 487.5, "power_coverage": 0.95})
    active_power_row = next(line for line in table.splitlines() if "Active Power per GPU (W)" in line)
    coverage_row = next(line for line in table.splitlines() if "Power Data Coverage (%)" in line)

    assert "487.50" in active_power_row
    assert "95.00" not in active_power_row
    assert "95.00" in coverage_row
    assert "487.50" not in coverage_row


def test_power_report_table_surfaces_withheld_power_as_unavailable() -> None:
    table = format_report_table({"power_coverage": 0.42})
    active_power_row = next(line for line in table.splitlines() if "Active Power per GPU (W)" in line)
    coverage_row = next(line for line in table.splitlines() if "Power Data Coverage (%)" in line)

    assert "unavailable (insufficient energy coverage)" in active_power_row
    assert "42.00" in coverage_row
    assert "42.00" not in active_power_row


def test_b200_power_survives_native_json_and_runner_normalization() -> None:
    # Pin the measured-data identity and workload from migration section 4.11.
    report = _run(
        {
            "traffic": {
                "source": {"type": "synthetic", "input_tokens": 1024, "output_tokens": 128},
                "load": {"type": "concurrency", "concurrency": 64},
                "stop": {"requests": 100},
            },
            "engine": {
                "mode": "aggregated",
                "model": "meta-llama/Meta-Llama-3.1-8B",
                "hardware": "b200_sxm",
                "backend": "trtllm",
                "backend_version": "1.3.0rc20",
                "workers": {"aggregated": {"parallelism": {"tensor": 2, "replicas": 1}}},
            },
        }
    )

    assert report.metrics["completed_requests"] == 100
    native_summary = report.metadata["native_report"]
    for name, expected in {"power_w": 655.9411158961074, "power_coverage": 0.9070317503277924}.items():
        assert native_summary[name] == pytest.approx(expected)
        assert report.metrics[name] == native_summary[name]


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


@pytest.mark.parametrize("interval", [None, 0, 2], ids=["default", "disabled", "two_rounds"])
def test_sglang_prefill_decode_interval_reaches_native_scheduler(tmp_path, interval) -> None:
    trace = tmp_path / "interval.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"request_id": "running", "timestamp": 0, "input_length": 8, "output_length": 6, "hash_ids": [1]},
                {"request_id": "waiting", "timestamp": 0.5, "input_length": 8, "output_length": 2, "hash_ids": [2]},
            ]
        ),
        encoding="utf-8",
    )
    engine = _engine()
    engine["backend"] = "sglang"
    if interval is not None:
        engine["workers"]["aggregated"]["scheduler"] = {"prefill_decode_interval": interval}
    report = _run(
        {
            "engine": engine,
            "traffic": {
                "source": {"type": "trace", "paths": [str(trace)], "format": "mooncake", "block_size": 8},
                "load": {"type": "trace_timestamps"},
            },
        }
    )
    running, waiting = sorted(report.metadata["native_report"]["per_request"], key=lambda row: row["arrival_time_ms"])

    # Both forward types take exactly 1 ms. The second request arrives during
    # the first EXTEND. With N=2, decode rounds occupy [1,2] and [2,3] before
    # the waiting request's EXTEND at [3,4]. With N=0, it extends at [1,2]
    # and the existing request's first decode instead occupies [2,3].
    assert running["first_token_ms"] == pytest.approx(1.0)
    assert waiting["arrival_time_ms"] == pytest.approx(0.5)
    assert waiting["first_admit_ms"] == pytest.approx(3.0 if interval else 1.0)
    assert waiting["first_token_ms"] == pytest.approx(4.0 if interval else 2.0)
    assert running["ttst_ms"] == pytest.approx(1.0 if interval else 2.0)
    assert (running["output_length"], waiting["output_length"]) == (6, 2)


@pytest.mark.parametrize("seeded", [False, True], ids=["fresh", "completed_prior_wave"])
def test_sglang_interval_idle_tail_does_not_relax_later_kv_admission(tmp_path, seeded) -> None:
    rows = [{"timestamp": 0, "input_length": 4, "output_length": 1, "hash_ids": [1] * 4}] if seeded else []
    rows += [
        {"timestamp": 100, "input_length": 4, "output_length": 200, "hash_ids": [2] * 4},
        {"timestamp": 100.5, "input_length": 4, "output_length": 106, "hash_ids": [3] * 4},
    ]
    trace = tmp_path / "interval-idle-pressure.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    engine = _engine()
    engine["backend"] = "sglang"
    worker = engine["workers"]["aggregated"]
    worker["scheduler"] = {"max_sequences": 2, "prefill_decode_interval": 20}
    worker["kv_cache"] = {
        "block_size": 1,
        "prefix_caching": False,
        "capacity": {"type": "fixed", "blocks": 256},
    }
    report = _run(
        {
            "engine": engine,
            "traffic": {
                "source": {"type": "trace", "paths": [str(trace)], "format": "mooncake", "block_size": 1},
                "load": {"type": "trace_timestamps"},
            },
        }
    )
    waiting = max(report.metadata["native_report"]["per_request"], key=lambda row: row["arrival_time_ms"])
    # With the normal reservation ratio the first request holds enough KV budget
    # to keep the second waiting until t=300. The trace reader normalizes its
    # origin, so compare delays. A completed earlier wave and its idle cooldown
    # must not lower that reservation and admit the second request at t=121.
    assert waiting["first_admit_ms"] - waiting["arrival_time_ms"] == pytest.approx(199.5)
    assert report.metrics["completed_requests"] == len(rows)


def test_engine_stack_runs_mooncake_delta() -> None:
    report = _run(
        {
            "traffic": {
                "source": {
                    "type": "trace",
                    "paths": [str(_TRACE_FIXTURES / "mooncake-delta.jsonl")],
                    "format": "mooncake-delta",
                    "block_size": 512,
                },
                "load": {"type": "trace_timestamps"},
            },
            "engine": _engine(),
        }
    )

    assert report.metrics["completed_requests"] == 2
    records = report.metadata["native_report"]["per_request"]
    assert records[1]["input_length"] == 28
    assert records[1]["arrival_time_ms"] >= records[0]["last_token_ms"] + 5


def test_engine_stack_runs_applied_compute_agentic() -> None:
    report = _run(
        {
            "traffic": {
                "source": {
                    "type": "trace",
                    "paths": [str(_TRACE_FIXTURES / "applied-compute-agentic.jsonl")],
                    "format": "applied_compute_agentic",
                    "block_size": 512,
                },
                "load": {"type": "concurrency", "concurrency": 1},
                "stop": {"max_virtual_time_seconds": 60},
            },
            "engine": _engine(),
        }
    )

    assert report.metrics["completed_requests"] == 3
    records = report.metadata["native_report"]["per_request"]
    assert [record["input_length"] for record in records] == [16, 28, 40]


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


def test_engine_stack_runs_native_dynamo_trace() -> None:
    report = _run(
        {
            "traffic": {
                "source": {
                    "type": "trace",
                    "paths": [str(_TRACE_FIXTURES / "dynamo-standard.jsonl")],
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


def _fpm_engine() -> dict:
    return {
        "model": "MiniMaxAI/MiniMax-M2.7",
        "hardware": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.25.1",
        "context_length": 8192,
        "workers": {
            "aggregated": {
                "parallelism": {"tensor": 4, "moe_tensor": 4, "moe_expert": 1},
                "kv_cache": {"prefix_caching": False},
                "timing": {"type": "default", "forward_model": "fpm"},
            }
        },
    }


def test_prediction_spec_lowers_fpm_forward_model_into_canonical_timing() -> None:
    parsed = CorePredictionConfig.model_validate({"engine": _fpm_engine()})
    deployment = prediction_to_replay_spec(parsed).backend_deployment

    timing = deployment.agg_engine_args["timing_model"]
    assert timing["type"] == "external"
    assert timing["provider"] == "aic"
    assert timing["config"]["estimation_mode"] == "fpm_interpolation"
    assert timing["config"]["fallback_policy"] == "deny"
    assert timing["config"]["worker_type"] == "aggregated"
    assert "aic_forward_model" not in deployment.agg_engine_args
    assert deployment.performance_model_metadata["aggregated"]["config"]["forward_model"] == "fpm"


def test_prediction_spec_lowers_default_timing_into_canonical_auto_selection() -> None:
    engine = _fpm_engine()
    engine["workers"]["aggregated"]["timing"] = {"type": "default"}
    parsed = CorePredictionConfig.model_validate({"engine": engine})
    deployment = prediction_to_replay_spec(parsed).backend_deployment

    assert "aic_forward_model" not in deployment.agg_engine_args
    assert deployment.agg_engine_args["timing_model"]["config"]["estimation_mode"] == "auto"
    assert deployment.performance_model_metadata["aggregated"]["config"]["forward_model"] == "op_level"


def test_prediction_spec_lowers_forward_model_per_role_in_disaggregated_mode() -> None:
    engine = _fpm_engine()
    engine["mode"] = "disaggregated"
    worker = engine["workers"].pop("aggregated")
    engine["workers"] = {
        "prefill": {**worker, "timing": {"type": "default"}},
        "decode": {**worker, "timing": {"type": "default", "forward_model": "fpm"}},
    }
    parsed = CorePredictionConfig.model_validate({"engine": engine})
    deployment = prediction_to_replay_spec(parsed).backend_deployment

    assert "aic_forward_model" not in deployment.prefill_engine_args
    assert "aic_forward_model" not in deployment.decode_engine_args
    assert deployment.decode_engine_args["timing_model"]["type"] == "external"
    assert deployment.decode_engine_args["timing_model"]["provider"] == "aic"
    assert deployment.decode_engine_args["timing_model"]["config"]["fallback_policy"] == "deny"
    prefill = deployment.prefill_engine_args["timing_model"]["config"]
    decode = deployment.decode_engine_args["timing_model"]["config"]
    assert prefill["estimation_mode"] == "auto"
    assert prefill["worker_type"] == "prefill"
    assert decode["estimation_mode"] == "fpm_interpolation"
    assert decode["worker_type"] == "decode"
    assert deployment.performance_model_metadata["prefill"]["config"]["forward_model"] == "op_level"
    assert deployment.performance_model_metadata["decode"]["config"]["forward_model"] == "fpm"


_SMALL_TRAFFIC = {
    "source": {"type": "synthetic", "input_tokens": 1024, "output_tokens": 32},
    "load": {"type": "concurrency", "concurrency": 4},
    "stop": {"requests": 8},
}


def test_engine_stack_replays_fpm_timing_from_the_bundled_cell(monkeypatch) -> None:
    # The bundled MiniMax-M2.7 cell is collected at vLLM 0.25.1, outside the queryable version slots.
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")

    report = _run({"engine": _fpm_engine(), "traffic": _SMALL_TRAFFIC})

    assert report.metrics["completed_requests"] == 8


def test_engine_stack_fpm_timing_fails_closed_without_a_matching_cell(
    monkeypatch,
) -> None:
    # tp2 has no FPM cell for this model on h200_sxm. The FPM path must refuse rather than fall
    # back to op_level; the same shape still replays under op_level timing. This is a wiring
    # check for the data path, not an accuracy statement about either model.
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    engine = _fpm_engine()
    engine["workers"]["aggregated"]["parallelism"] = {
        "tensor": 2,
        "moe_tensor": 2,
        "moe_expert": 1,
    }

    with pytest.raises(RuntimeError, match="FPM"):
        _run({"engine": engine, "traffic": _SMALL_TRAFFIC})

    engine["workers"]["aggregated"]["timing"] = {"type": "default"}
    assert _run({"engine": engine, "traffic": _SMALL_TRAFFIC}).metrics["completed_requests"] == 8


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_engine_stack_runs_weka_directory_with_one_agentic_lane(backend: str) -> None:
    corpus = _TRACE_FIXTURES / "weka"
    engine = {**_engine(), "backend": backend}
    report = _run(
        {
            "traffic": {
                "source": {"type": "trace", "paths": [str(corpus)], "format": "weka"},
                "load": {"type": "trace_timestamps", "agentic_lanes": 1},
            },
            "engine": engine,
        }
    )

    assert report.metrics["completed_requests"] == 3
    native = report.metadata["native_report"]
    assert native["agentic_input_format"] == "weka"
    assert native["agentic_lanes"] == 1
    assert native["agentic_qualification"] == "functional_only"
    assert len(native["agentic_lifecycle_digest"]) == 64
    assert native["agentic_lifecycle_event_count"] > native["completed_requests"]
    assert len(native["agentic_play_outcomes"]) == 2
    assert all(outcome["status"] == "completed" for outcome in native["agentic_play_outcomes"])
    assert min(record["dispatched_at_ms"] for record in native["per_request"]) == 0
    assert native["weka_nested_timestamp_basis"] == "absolute"
    assert native["agentic_model_projection"] == {
        "policy": "project_to_configured_target",
        "source_models": ["model", "other-model"],
        "target_model": "example/model",
    }
    records = native["per_request"]
    by_play: dict[str, list[dict]] = {}
    for record in records:
        by_play.setdefault(record["play_id"], []).append(record)
    ordered = sorted(
        by_play,
        key=lambda play_id: min(record["dispatched_at_ms"] for record in by_play[play_id]),
    )
    assert len(ordered) == 2
    assert [play_id.rsplit(":play:", 1)[1] for play_id in ordered] == [
        "play-a",
        "play-b",
    ]
    assert min(record["dispatched_at_ms"] for record in by_play[ordered[1]]) >= max(
        record["terminal_time_ms"] for record in by_play[ordered[0]]
    )
    prefill_only = by_play[ordered[1]][0]
    assert prefill_only["requested_output_length"] == 0
    assert prefill_only["output_length"] == 0
    assert prefill_only["first_token_ms"] is None


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_engine_stack_auto_infers_raw_weka_relative_timestamps(backend: str) -> None:
    config = {
        "traffic": {
            "source": {
                "type": "trace",
                "paths": [str(_TRACE_FIXTURES / "weka-relative.json")],
                "format": "weka",
            },
            "load": {"type": "trace_timestamps", "agentic_lanes": 1},
        },
        "engine": {**_engine(), "backend": backend},
    }
    report = _run(config)

    native = report.metadata["native_report"]
    assert native["weka_nested_timestamp_basis"] == "relative"
    assert native["agentic_input_format"] == "weka"
    assert report.metrics["completed_requests"] == 4
    repeated = _run(config).metadata["native_report"]
    for key in ("agentic_graph", "agentic_lifecycle_digest", "agentic_play_outcomes", "per_request"):
        assert repeated[key] == native[key], key
    assert [outcome["status"] for outcome in native["agentic_play_outcomes"]] == ["completed"]


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_agentx_m1_default_python_result_retains_qualification_without_dynamo(backend: str) -> None:
    config = {
        "traffic": {
            "source": {"type": "trace", "format": "weka", "paths": [str(_TRACE_FIXTURES / "weka-relative.json")]},
            "load": {"type": "trace_timestamps", "agentic_lanes": 1},
        },
        "engine": {**_engine(), "backend": backend},
    }
    code = textwrap.dedent("""
        import importlib.abc
        import json
        import sys

        assert not any(name.split(".")[0] in {"aisimulate", "dynamo"} for name in sys.modules)

        class RejectDynamo(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "dynamo" or fullname.startswith("dynamo."):
                    raise AssertionError("public AISimulate Weka replay must not import Dynamo")

        sys.meta_path.insert(0, RejectDynamo())

        from aisimulate import CorePredictionConfig, EngineReplayRunnerFactory
        from aisimulate.compiler import prediction_to_replay_spec

        config = CorePredictionConfig.model_validate(json.load(sys.stdin))
        runner = EngineReplayRunnerFactory().create(0)
        try:
            report = runner.run(prediction_to_replay_spec(config))
        finally:
            runner.close()
        print(json.dumps({"metrics": report.metrics, "metadata": report.metadata}))
    """)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps(config),
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    report = json.loads(completed.stdout)
    assert report["metrics"]["completed_requests"] == 4
    assert report["metadata"]["agentic_qualification"] == "functional_only"
    assert report["metadata"]["agentic_lanes"] == 1
    assert report["metadata"]["agentic_model_projection"]["target_model"] == "example/model"
    assert "native_report" not in report["metadata"]


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_agentx_m1_gate_config_preserves_local_source_block_size(backend: str, monkeypatch) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    gate = runpy.run_path(str(scripts / "qualify_agentx_m1.py"))
    source = _TRACE_FIXTURES / "weka-relative.json"
    block_size = json.loads(source.read_text())["block_size"]
    assert block_size == 4
    config = gate["prediction_config"](source, "weka", block_size, backend)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    assert spec.workload["trace_block_size"] == block_size
    assert spec.backend_deployment.agg_engine_args["block_size"] == block_size
    report = gate["run_python"](config)
    evidence = gate["evidence"](report)
    assert evidence["agentic_graph"]["block_size"] == block_size
    assert report["completed_requests"] == 4


def test_predict_detail_uses_real_native_evidence(tmp_path, capsys):
    import yaml

    from aisimulate.main import main

    config = {
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 128, "output_tokens": 4},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        },
        "engine": {
            "model": "Qwen/Qwen3-30B-A3B",
            "hardware": "b200_sxm",
            "backend": "vllm",
            "backend_version": "current",
            "context_length": 4096,
            "workers": {
                "aggregated": {
                    "parallelism": {"tensor": 4, "moe_tensor": 1, "moe_expert": 4},
                    "scheduler": {"max_batched_tokens": 8192, "max_sequences": 256},
                    "kv_cache": {"block_size": 64},
                }
            },
        },
    }
    path = tmp_path / "native-detail.yaml"
    path.write_text(yaml.safe_dump(config))
    out = tmp_path / "out"
    assert (
        main(
            [
                "predict",
                "-c",
                str(path),
                "--detail",
                "all",
                "--format",
                "json",
                "--output-dir",
                str(out),
            ]
        )
        == 0
    )
    stdout = json.loads(capsys.readouterr().out)
    saved = json.loads((out / "prediction.json").read_text())
    assert stdout["details"] == saved["details"]
    from jsonschema import validate

    schema = json.loads((Path(__file__).resolve().parents[1] / "docs/cli/prediction-details.schema.json").read_text())
    validate(stdout["details"], schema)
    sections = stdout["details"]["sections"]
    assert set(sections) == {"summary", "memory", "time", "energy"}
    assert saved["power_diagnostics"]["power_w"] is None
    assert stdout["summary"]["power_w"] is None
    memory = sections["memory"]["roles"]["aggregated"]
    assert memory["status"] == "available"
    assert memory["stage"] == "before_native_capacity_adjustments"
    assert "num_gpu_blocks" not in memory
    assert memory["memory_breakdown"]["weights_bytes"] > 0
    assert memory["estimated_num_gpu_blocks"] == memory["total_kv_size_tokens"] // 64
    assert memory["total_gpu_capacity_bytes"] > memory["total_kv_size_bytes"] > 0
    assert sections["time"]["serving_metrics"]["mean_ttft_ms"] == stdout["summary"]["mean_ttft_ms"]
    assert "wall_time_ms" not in sections["time"]["serving_metrics"]
    assert "duration_ms" not in sections["time"]["serving_metrics"]
    assert stdout["summary"]["duration_ms"] > 0
    assert "phases" not in sections["time"]
    # Repeating the same prediction without diagnostics keeps modeled metrics identical.
    plain_out = tmp_path / "plain"
    assert main(["predict", "-c", str(path), "--format", "json", "--output-dir", str(plain_out)]) == 0
    plain = json.loads(capsys.readouterr().out)
    for key, value in stdout["summary"].items():
        if key not in {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}:
            assert plain[key] == value, key


def test_fpm_detail_distinguishes_memory_budget_from_runtime_capacity(tmp_path, capsys, monkeypatch):
    import yaml

    from aisimulate.main import main

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    path = tmp_path / "fpm.yaml"
    path.write_text(yaml.safe_dump({"engine": _fpm_engine(), "traffic": _SMALL_TRAFFIC}))
    assert (
        main(["predict", "-c", str(path), "--detail", "all", "--format", "json", "--output-dir", str(tmp_path / "out")])
        == 0
    )
    sections = json.loads(capsys.readouterr().out)["details"]["sections"]
    memory = sections["memory"]["roles"]["aggregated"]
    assert memory["scope"] == "capacity_estimate_per_rank"
    assert memory["stage"] == "before_native_capacity_adjustments"
    assert memory["estimated_num_gpu_blocks"] > 0
    assert "num_gpu_blocks" not in memory
    assert set(sections) == {"summary", "memory", "time", "energy"}
    assert sections["time"]["serving_metrics"]["mean_ttft_ms"] > 0
