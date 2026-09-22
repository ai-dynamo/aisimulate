# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serving checks retain native benchmark records and fail closed on mismatched evidence."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from aisimulate import main as cli
from aisimulate.support import serving_validation as serving

from .test_support_validation import validation_case as _replay_validation_case  # noqa: F401

pytestmark = pytest.mark.unit


def _save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def materialized_workload(monkeypatch):
    """Author only the isolated tokenizer boundary; native derived replay still runs."""

    def materialize(recipe, output, _python):
        play = json.loads(Path(recipe["benchmark_trace"]["path"]).read_text())
        inputs, payloads = [], []
        for index, source in enumerate(play["requests"]):
            tokens = [block for block in source["hash_ids"] for _ in range(play["block_size"])][: source["in"]]
            inputs.append({"turn_index": index, "input_token_ids": tokens, "input_length": len(tokens)})
            payloads.append(
                {
                    "payload": {
                        "model": recipe["scope"]["model_projection"],
                        "messages": [{"role": "user", "content": "synthetic"}],
                        "add_generation_prompt": True,
                        "continue_final_message": False,
                    }
                }
            )
        _save(output / "tokenization.json", {"chat_template": "synthetic", "requests": inputs})
        (output / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in payloads))
        _save(output / "helper.json", {"fixture": "synthetic tokenizer boundary"})
        _save(output / "input.json", {"trace": recipe["trace"]})
        return {
            "helper": serving._identity(output / "helper.json"),
            "input": serving._identity(output / "input.json"),
            "installation": {"source": {"vcs_info": {"commit_id": serving.AIPERF_REVISION}}},
        }

    monkeypatch.setattr(serving, "_materialize_payloads", materialize)


def write_server_tokenization(recipe_path, recipe):
    frozen = json.loads(Path(recipe["matched_workload"]["tokenization"]["path"]).read_text())
    payloads = [
        json.loads(line) for line in Path(recipe["matched_workload"]["payloads"]["path"]).read_text().splitlines()
    ]
    rows = []
    for tokens, payload in zip(frozen["requests"], payloads, strict=True):
        request = {
            key: payload["payload"][key]
            for key in ("model", "messages", "add_generation_prompt", "continue_final_message")
        }
        request["add_special_tokens"] = False
        rows.append(
            {
                "turn_index": tokens["turn_index"],
                "request": request,
                "response": {"tokens": tokens["input_token_ids"], "count": tokens["input_length"]},
            }
        )
    path = recipe_path.parent / "server-tokenization.json"
    _save(path, {"recipe": serving._identity(recipe_path), "status": "matched", "requests": rows})
    return path


@pytest.fixture
def serving_case(request, tmp_path, materialized_workload):
    args, request, _plan, trace, prediction_output = request.getfixturevalue("_replay_validation_case")
    # This invokes the actual ordinary prediction/compiler/native FPM path;
    # imported AIPerf-shaped observations below are synthetic contract fixtures.
    assert cli.main(args) == 0
    config = yaml.safe_load((prediction_output / "predict.yaml").read_text())
    worker = config["engine"]["workers"]["aggregated"]
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text('{"tokenizer_class":"synthetic"}')
    expected = {
        "model": request.identity.model,
        "model_revision": request.identity.model_revision,
        "backend": "vllm",
        "backend_version": request.identity.framework_version,
        "image_digest": "sha256:" + "a" * 64,
        "hardware": request.identity.gpu,
        "parallelism": request.parallelism(),
        "precision": {"weights": "fp8", "fmha": "bfloat16", "kv": "fp8"},
        "attention_groups": [{"backend_name": "FLASH_ATTN", "layer_names": ["layer.0"]}],
        "graph_config": {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2]},
        "scheduler": worker["scheduler"],
        "context_length": config["engine"]["context_length"],
        "gpu_memory_utilization": 0.73,
        "runtime_config": {
            "model_config": {
                "dtype": "torch.bfloat16",
                "quantization": "fp8",
                "revision": request.identity.model_revision,
            },
            "cache_config": {"cache_dtype": "fp8"},
            "scheduler_config": {"max_num_batched_tokens": 64, "max_num_seqs": 4},
        },
    }
    recipe_path = serving.prepare_serving_validation(
        trace=trace,
        prediction_config=prediction_output / "predict.yaml",
        output=tmp_path / "serving",
        endpoint="http://127.0.0.1:8000",
        tokenizer=tokenizer,
        expected_execution=expected,
        aiperf_python=Path("/benchmark/bin/python"),
    )
    recipe = json.loads(recipe_path.read_text())
    prediction_report = prediction_output / "prediction/prediction.json"
    prediction = json.loads(Path(recipe["matched_workload"]["prediction_report"]["path"]).read_text())
    artifacts = Path(recipe["artifacts_dir"])
    artifacts.mkdir()
    rows = []
    for index, row in enumerate(prediction["per_request"]):
        rows.append(
            {
                "metadata": {
                    "conversation_id": recipe["play_id"],
                    "turn_index": index,
                    "x_request_id": f"http-{index}",
                    "was_cancelled": False,
                    "benchmark_phase": "profiling",
                    "request_start_ns": 1_000_000_000 + round(row["arrival_time_ms"] * 1_000_000),
                    "request_end_ns": 1_000_000_000 + round(row["last_token_ms"] * 1_000_000),
                },
                "metrics": {
                    "input_sequence_length": {"value": row["input_length"], "unit": "tokens"},
                    "output_sequence_length": {"value": row["output_length"], "unit": "tokens"},
                    "time_to_first_token": {"value": row["ttft_ms"], "unit": "ms"},
                    "inter_token_latency": {"value": row["itl_ms"], "unit": "ms"},
                },
                "error": None,
            }
        )
    records = artifacts / "profile_export.jsonl"
    records.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = artifacts / "profile_export_aiperf.json"
    _save(summary, {"was_cancelled": False, "is_complete": True, "request_count": {"avg": len(rows)}})
    runtime_dir = Path(recipe["serving_observer"]["output_dir"])
    runtime_dir.mkdir()
    provenance_path = Path(recipe["serving_observer"]["provenance_path"])
    provenance = json.loads(provenance_path.read_text())
    runtime_sources = []
    native_config = copy.deepcopy(expected["runtime_config"])
    native_config["cache_config"]["enable_prefix_caching"] = True
    for tp in range(expected["parallelism"]["tensor"]):
        native_path = runtime_dir / f"fpm-execution-worker-dp0-tp{tp}-pp0.json"
        _save(
            native_path,
            {
                "schema_name": "aisimulate_fpm_runtime_execution",
                "schema_version": 1,
                "status": "observed",
                "execution_provenance": provenance,
                "provenance_source": serving._identity(provenance_path),
                "dp_rank": 0,
                "tp_rank": tp,
                "pp_rank": 0,
                **{key: expected[key] for key in ("attention_groups", "graph_config", "backend_version")},
                "resolved_config": native_config,
            },
        )
        runtime_sources.append(serving._identity(native_path))
    receipt = {
        "schema_version": serving.SCHEMA,
        "recipe": serving._identity(recipe_path),
        "status": "completed",
        "exit_code": 0,
        "command": ["python", *recipe["command_arguments"]],
        "installation": {"source": {"vcs_info": {"commit_id": serving.AIPERF_REVISION}}},
        "artifacts": [serving._identity(records), serving._identity(summary)],
        "runtime_observations": runtime_sources,
        "serving_provenance": serving._identity(provenance_path),
        "observer_modules": recipe["serving_observer"]["module_files"],
        "server_tokenization": serving._identity(write_server_tokenization(recipe_path, recipe)),
    }
    _save(recipe_path.parent / "benchmark-execution.json", receipt)
    observed = {
        "schema_version": serving.SCHEMA,
        "recipe": serving._identity(recipe_path),
        "observed_execution": expected,
        "scope": recipe["scope"],
        "sources": [serving._identity(tokenizer / "tokenizer_config.json")],
    }
    observed_path = recipe_path.parent / "observed-execution.json"
    _save(observed_path, observed)
    return SimpleNamespace(
        request=request,
        recipe_path=recipe_path,
        recipe=recipe,
        prediction_report=prediction_report,
        observed_path=observed_path,
        observed=observed,
        rows=rows,
        records=records,
        trace=trace,
        prediction_config=prediction_output / "predict.yaml",
        tokenizer=tokenizer,
    )


def _assess(case, **kwargs):
    return serving.validate_serving_measurements(
        case.recipe_path,
        prediction_report=case.prediction_report,
        execution_evidence_path=case.observed_path,
        output_report=case.recipe_path.parent / "assessment.json",
        **kwargs,
    )


def _rewrite_records(case):
    case.records.write_text("\n".join(json.dumps(row) for row in case.rows) + "\n")
    receipt_path = case.recipe_path.parent / "benchmark-execution.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"] = [
        serving._identity(case.records),
        serving._identity(case.records.parent / "profile_export_aiperf.json"),
    ]
    _save(receipt_path, receipt)


def test_serving_gate_compares_native_requests_and_keeps_accuracy_scope(serving_case):
    report = _assess(serving_case)
    assert report["status"] == "passed"
    assert report["forward"]["status"] == "unavailable"
    assert report["scope"]["qualification"] == "one_complete_single_stream_weka_play"
    assert report["metrics"]["ttft"]["p95_relative_error"] == 0
    assert report["metrics"]["tpot"]["p95_relative_error"] == 0
    assert report["metrics"]["output_throughput"]["relative_error"] < 1e-6
    assert len(report["samples"]) == 2
    assert report["samples"][0]["serving_request_id"] == "http-0"
    assert serving.check_serving_validation_report(serving_case.recipe_path.parent / "assessment.json") == report


def test_target_tokenized_replay_preserves_source_identity_and_dependencies(serving_case):
    case = serving_case
    matched = case.recipe["matched_workload"]
    rows = [json.loads(row) for row in Path(matched["trace"]["path"]).read_text().splitlines()]
    original = json.loads(case.prediction_report.read_text())["per_request"]
    assert rows[0]["schema"] == "dynamo.agentic_mooncake"
    assert [row["request_id"] for row in rows[1:]] == [row["request_id"] for row in original]
    assert rows[2]["dependencies"] == [
        {"request_id": original[0]["request_id"], "trigger": "completion", "relation": "sequence", "delay_ms": 100.0}
    ]
    assert rows[2]["not_before_ms"] == 100.0
    config = yaml.safe_load(Path(matched["prediction_config"]["path"]).read_text())
    source_config = yaml.safe_load(case.prediction_config.read_text())
    assert config["engine"] == source_config["engine"]
    assert config["traffic"]["source"]["format"] == "agentic_mooncake"
    assert "mooncake_trace" in case.recipe["command_arguments"]


def test_native_prefix_observation_must_match_replay_even_after_protocol_normalization(serving_case):
    case = serving_case
    receipt_path = case.recipe_path.parent / "benchmark-execution.json"
    receipt = json.loads(receipt_path.read_text())
    path = Path(receipt["runtime_observations"][0]["path"])
    observed = json.loads(path.read_text())
    observed["resolved_config"]["cache_config"]["enable_prefix_caching"] = False
    _save(path, observed)
    receipt["runtime_observations"][0] = serving._identity(path)
    _save(receipt_path, receipt)
    report = _assess(case)
    assert report["status"] == "failed"
    assert any("prefix cache setting" in issue for issue in report["failures"])


@pytest.mark.parametrize("mismatch", [False, True])
def test_actual_server_tokenization_preflight_checks_every_frozen_payload(serving_case, monkeypatch, mismatch):
    case = serving_case
    tokenization = json.loads(Path(case.recipe["matched_workload"]["tokenization"]["path"]).read_text())
    calls = []

    def response(request, **_kwargs):
        index = len(calls)
        calls.append(json.loads(request.data))
        expected = tokenization["requests"][index]
        tokens = list(expected["input_token_ids"])
        if mismatch and index == 1:
            tokens[-1] += 1
        return io.BytesIO(json.dumps({"tokens": tokens, "count": len(tokens)}).encode())

    monkeypatch.setattr(serving, "urlopen", response)
    if mismatch:
        with pytest.raises(ValueError, match="tokenizer/chat template differs"):
            serving._tokenize_serving(case.recipe_path, case.recipe)
    else:
        path = serving._tokenize_serving(case.recipe_path, case.recipe)
        assert json.loads(path.read_text())["status"] == "matched"
    assert len(calls) == 2
    assert all(not row["add_special_tokens"] for row in calls)


def test_missing_server_tokenization_cannot_qualify_serving(serving_case):
    case = serving_case
    path = case.recipe_path.parent / "benchmark-execution.json"
    receipt = json.loads(path.read_text())
    receipt.pop("server_tokenization")
    _save(path, receipt)
    report = _assess(case)
    assert report["status"] == "incomplete"
    assert any("/tokenize" in issue for issue in report["missing_evidence"])


def test_serving_dispatch_tolerance_is_editable_without_remeasurement(serving_case):
    case = serving_case
    case.rows[1]["metadata"]["request_start_ns"] += 6_000_000
    case.rows[1]["metadata"]["request_end_ns"] += 6_000_000
    _rewrite_records(case)
    first = _assess(case)
    assert first["status"] == "failed"
    second = serving.validate_serving_measurements(
        case.recipe_path,
        prediction_report=case.prediction_report,
        execution_evidence_path=case.observed_path,
        output_report=case.recipe_path.parent / "new-policy.json",
        max_dispatch_delay_ms=7.0,
    )
    assert second["status"] == "passed"
    assert second["policy"]["max_dispatch_delay_ms"] == 7.0


def test_reassessment_rejects_forged_status_and_changed_source(serving_case):
    report = _assess(serving_case)
    report["status"] = "failed"
    path = serving_case.recipe_path.parent / "assessment.json"
    _save(path, report)
    with pytest.raises(ValueError, match="independent reassessment"):
        serving.check_serving_validation_report(path)
    report["status"] = "passed"
    _save(path, report)
    serving_case.records.write_text(serving_case.records.read_text() + "\n")
    with pytest.raises(ValueError, match="independent reassessment"):
        serving.check_serving_validation_report(path)


@pytest.mark.parametrize("api_time,expected_arrival", [(0.0, 308.6875), (0.3, 300.0)])
def test_native_single_stream_matches_supported_timing_subsets(request, api_time, expected_arrival):
    args, _request, _plan, trace, output = request.getfixturevalue("_replay_validation_case")
    play = json.loads(trace.read_text())
    play["requests"][0].update(t=120.0, api_time=api_time)
    play["requests"][1].update(t=120.3, api_time=0.3)
    _save(trace, play)
    assert cli.main(args) == 0
    report = json.loads((output / "prediction/prediction.json").read_text())
    rows = report["per_request"]
    assert rows[0]["arrival_time_ms"] == 0
    assert rows[1]["arrival_time_ms"] == pytest.approx(expected_arrival)
    assert serving._timing_subset(play) == ("zero_idle_gaps" if api_time else "zero_recorded_api_time")


@pytest.mark.parametrize("field", serving.EXECUTION_FIELDS)
def test_missing_effective_execution_never_qualifies_accuracy(serving_case, field):
    serving_case.observed["observed_execution"][field] = None
    _save(serving_case.observed_path, serving_case.observed)
    report = _assess(serving_case)
    assert report["status"] == "incomplete"
    assert any(field in issue for issue in report["missing_evidence"])


@pytest.mark.parametrize("field", ["model_revision", "precision", "attention_groups", "graph_config", "parallelism"])
def test_different_serving_execution_fails_even_if_latencies_match(serving_case, field):
    serving_case.observed["observed_execution"][field] = "different"
    _save(serving_case.observed_path, serving_case.observed)
    report = _assess(serving_case)
    assert report["status"] == "failed"
    assert any(field in issue for issue in report["failures"])


@pytest.mark.parametrize("change", ["warmup", "cancelled", "wrong_shape", "duplicate", "wrong_units", "missing"])
def test_bad_request_evidence_does_not_pass(serving_case, change):
    row = serving_case.rows[0]
    if change == "warmup":
        row["metadata"]["benchmark_phase"] = "warmup"
    elif change == "cancelled":
        row["metadata"]["was_cancelled"] = True
    elif change == "wrong_shape":
        row["metrics"]["input_sequence_length"]["value"] += 1
    elif change == "duplicate":
        serving_case.rows[1] = copy.deepcopy(row)
    elif change == "wrong_units":
        row["metrics"]["time_to_first_token"]["unit"] = "seconds"
    else:
        serving_case.rows.pop()
    _rewrite_records(serving_case)
    report = _assess(serving_case)
    assert report["status"] != "passed"
    assert report["metrics"]["output_throughput"]["status"] == "incomplete"


def test_latency_and_throughput_are_independent_gates_with_editable_limits(serving_case):
    for row in serving_case.rows:
        row["metrics"]["time_to_first_token"]["value"] *= 2
    _rewrite_records(serving_case)
    report = _assess(serving_case)
    assert report["status"] == "failed"
    assert report["metrics"]["ttft"]["status"] == "failed"
    assert report["metrics"]["tpot"]["status"] == "passed"
    assert report["metrics"]["output_throughput"]["status"] == "passed"
    second = serving.validate_serving_measurements(
        serving_case.recipe_path,
        prediction_report=serving_case.prediction_report,
        execution_evidence_path=serving_case.observed_path,
        output_report=serving_case.recipe_path.parent / "reviewed-policy-assessment.json",
        latency_p95_error=0.6,
    )
    assert second["status"] == "passed"
    assert second["recipe"] == report["recipe"]


@pytest.mark.parametrize("change", ["nested", "overlap", "prefix_fork", "multiple_plays", "mixed_timing"])
def test_unsupported_trace_preparation_remains_incomplete(serving_case, tmp_path, change):
    case = serving_case
    play = json.loads(case.trace.read_text())
    if change == "nested":
        play["requests"].append({"type": "subagent", "requests": [play["requests"][0]]})
    elif change == "overlap":
        play["requests"][0]["api_time"] = 1
    elif change == "prefix_fork":
        play["requests"][1]["hash_ids"] = [8, 9]
    elif change == "mixed_timing":
        play["requests"][0]["api_time"] = 0.05
    case.trace.write_text(json.dumps(play) + "\n" + (json.dumps(play) + "\n" if change == "multiple_plays" else ""))
    recipe_path = serving.prepare_serving_validation(
        trace=case.trace,
        prediction_config=case.prediction_config,
        output=tmp_path / "unsupported",
        endpoint="http://localhost:8000",
        tokenizer=case.tokenizer,
        expected_execution=case.recipe["expected_execution"],
    )
    recipe = json.loads(recipe_path.read_text())
    assert recipe["status"] == "incomplete"
    assert recipe["issues"]
    assert recipe["command_arguments"] is None
    with pytest.raises(ValueError, match="preparation is incomplete"):
        serving.run_serving_benchmark(recipe_path, aiperf_python=Path("python"))


def test_real_producer_command_checks_pin_and_does_not_execute_saved_commands(serving_case, monkeypatch):
    case = serving_case
    (case.recipe_path.parent / "benchmark-execution.json").unlink()
    case.records.unlink()
    (case.records.parent / "profile_export_aiperf.json").unlink()
    case.records.parent.rmdir()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "-c":
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "version": "pinned",
                        "source": {
                            "url": serving.AIPERF_SOURCE + ".git",
                            "vcs_info": {"commit_id": serving.AIPERF_REVISION},
                        },
                    }
                )
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(serving.subprocess, "run", run)
    monkeypatch.setattr(serving, "_tokenize_serving", write_server_tokenization)
    receipt_path = serving.run_serving_benchmark(case.recipe_path, aiperf_python=Path("/benchmark/bin/python"))
    assert json.loads(receipt_path.read_text())["status"] == "completed"
    assert calls[1][0][1:] == case.recipe["command_arguments"]
    assert "--no-fixed-schedule" in calls[1][0]
    assert "--scenario" not in calls[1][0]
    assert "--warmup-request-count" not in calls[1][0]
    case.recipe["command_arguments"] = ["-c", "raise AssertionError('must not execute')"]
    _save(case.recipe_path, case.recipe)
    with pytest.raises(ValueError, match="differs from the fixed"):
        serving.run_serving_benchmark(case.recipe_path, aiperf_python=Path("/benchmark/bin/python"))
    assert len(calls) == 2


def test_changed_trace_is_rejected_before_execution(serving_case):
    case = serving_case
    case.trace.write_text(case.trace.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after preparation"):
        _assess(case)


def test_matched_replay_preserves_prefix_inside_source_block(serving_case, tmp_path, monkeypatch):
    case = serving_case
    original_materialize = serving._materialize_payloads
    # Explicit synthetic tokenizer output matching the real-tokenizer review case:
    # a 96-token shared prefix ends inside the source's second 64-token block.
    token_ids = [list(range(97)), [*range(96), *range(100, 167)]]

    def materialize(recipe, output, python):
        result = original_materialize(recipe, output, python)
        _save(
            output / "tokenization.json",
            {
                "requests": [
                    {"turn_index": i, "input_token_ids": tokens, "input_length": len(tokens)}
                    for i, tokens in enumerate(token_ids)
                ]
            },
        )
        return result

    monkeypatch.setattr(serving, "_materialize_payloads", materialize)
    prediction = yaml.safe_load(case.prediction_config.read_text())
    prediction["engine"]["workers"]["aggregated"]["kv_cache"]["block_size"] = 16
    source = json.loads(Path(case.recipe["benchmark_trace"]["path"]).read_text())
    assert source["block_size"] == 64
    matched = serving._prepare_matched_workload(case.recipe, prediction, tmp_path / "exact-prefix", Path("python"))
    replay = json.loads(Path(matched["prediction_report"]["path"]).read_text())
    graph = [json.loads(line) for line in Path(matched["trace"]["path"]).read_text().splitlines()]
    original = json.loads(case.prediction_report.read_text())
    assert replay["completed_requests"] == 2
    assert replay["per_request"][1]["reused_input_tokens"] == 96
    assert graph[0]["block_size"] == 1
    assert [row["hash_ids"] for row in graph[1:]] == token_ids
    for index, (row, old) in enumerate(zip(graph[1:], original["per_request"], strict=True)):
        for key in ("request_id", "play_id", "session_id"):
            assert row[key] == old[key]
        assert row["not_before_ms"] == source["requests"][index]["t"] * 1000
        assert row["recorded_api_time_ms"] == 0
        assert row["output_length"] == source["requests"][index]["out"]
    assert graph[1]["dependencies"] == []
    assert graph[2]["dependencies"] == [
        {"request_id": graph[1]["request_id"], "trigger": "completion", "relation": "sequence", "delay_ms": 100}
    ]
    coverage = json.loads(Path(matched["coverage_report"]["path"]).read_text())
    assert coverage["status"] == "covered"
    assert coverage["queries"]["unsupported"] == 0


def test_supplied_forward_samples_are_separate_from_request_latency(serving_case):
    case = serving_case
    forward = case.recipe_path.parent / "forward.json"
    _save(
        forward,
        {
            "recipe": serving._identity(case.recipe_path),
            "sources": [serving._identity(case.records)],
            "samples": {
                "prefill": [
                    {
                        "scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 64},
                        "measured_ms": 0.01,
                        "unit": "ms",
                    }
                ],
                "decode": [
                    {
                        "scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 65},
                        "measured_ms": 2,
                        "unit": "ms",
                    }
                ],
            },
        },
    )
    report = _assess(case, forward_evidence_path=forward)
    assert report["status"] == "failed"
    assert report["forward"]["phases"]["prefill"]["status"] == "failed"
    assert report["forward"]["phases"]["decode"]["status"] == "passed"
    assert report["metrics"]["ttft"]["status"] == "passed"
