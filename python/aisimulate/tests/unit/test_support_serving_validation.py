# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serving checks retain native benchmark records and fail closed on mismatched evidence."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from aisimulate import main as cli
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.support import serving_validation as serving
from aisimulate_core.ais_perf_model import AisPerfModelError

from .test_support_validation import validation_case as _replay_validation_case  # noqa: F401

pytestmark = pytest.mark.unit


def _save(path, value):
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("hicache", [None, {"num_host_blocks": 32}])
def test_serving_scope_rejects_hicache_before_hbm_only_qualification(tmp_path, hicache):
    trace = tmp_path / "trace.json"
    prediction = {
        "engine": {
            "model": "example/model",
            "hardware": "h200_sxm",
            "backend": "sglang",
            "workers": {
                "aggregated": {
                    "kv_cache": {"hicache": hicache},
                    "timing": {
                        "estimation_mode": "fpm_interpolation",
                        "fallback_policy": "deny",
                        "estimator_config": {"fpm_interpolation": {"method": "direct"}},
                    },
                }
            },
        },
        "traffic": {
            "source": {"type": "trace", "format": "weka", "paths": [str(trace)]},
            "load": {"type": "trace_timestamps", "agentic_lanes": 1},
        },
    }
    if hicache is not None:
        with pytest.raises(ValueError, match="HBM-only cache"):
            serving._scope(prediction, trace)
        return
    assert serving._scope(prediction, trace)["cache_storage"] == "hbm_only"


STRICT_DIRECT_FPM = {
    "estimation_mode": "fpm_interpolation",
    "fallback_policy": "deny",
    "estimator_config": {"fpm_interpolation": {"method": "direct"}},
}
FALLBACK_ALLOWED = {**STRICT_DIRECT_FPM, "fallback_policy": "allow"}


@pytest.mark.parametrize(
    ("engine_fields", "timing", "qualifies"),
    [
        pytest.param({}, STRICT_DIRECT_FPM, True, id="stated on the worker"),
        pytest.param(STRICT_DIRECT_FPM, {}, True, id="stated on the engine"),
        pytest.param(
            {"estimation_mode": "fpm_interpolation"},
            {key: STRICT_DIRECT_FPM[key] for key in ("fallback_policy", "estimator_config")},
            True,
            id="mode on the engine, the rest on the worker",
        ),
        pytest.param(FALLBACK_ALLOWED, STRICT_DIRECT_FPM, True, id="strict worker over an engine that allows fallback"),
        pytest.param({}, {}, False, id="stated nowhere"),
        pytest.param(FALLBACK_ALLOWED, {}, False, id="engine allows fallback"),
        pytest.param(STRICT_DIRECT_FPM, {"fallback_policy": "allow"}, False, id="worker allows fallback"),
        pytest.param({"ais_perf_model": STRICT_DIRECT_FPM}, {}, True, id="ais_perf_model"),
        pytest.param({"ais_perf_model": FALLBACK_ALLOWED}, {}, False, id="ais_perf_model allows fallback"),
    ],
)
def test_serving_scope_reads_estimator_settings_from_the_worker_or_the_engine(
    tmp_path, engine_fields, timing, qualifies
):
    trace = tmp_path / "trace.json"
    prediction = {
        "engine": {
            "model": "example/model",
            "hardware": "h200_sxm",
            "backend": "sglang",
            **engine_fields,
            "workers": {"aggregated": {"timing": timing}},
        },
        "traffic": {
            "source": {"type": "trace", "format": "weka", "paths": [str(trace)]},
            "load": {"type": "trace_timestamps", "agentic_lanes": 1},
        },
    }
    if not qualifies:
        with pytest.raises(ValueError, match="ordinary strict direct-FPM prediction"):
            serving._scope(prediction, trace)
        return
    assert serving._scope(prediction, trace)["qualification"] == "one_complete_single_stream_weka_play"


OLD_PATH = "/data/collected.parquet"
ANOTHER_PATH = "/data/another.parquet"
OLD_PATH_ONLY = {"forward_model": "fpm", "fpm_parquet_path": OLD_PATH}


def _strict_direct_fpm_naming(path):
    """Strict direct-FPM settings whose ``estimator_config`` names a parquet file."""
    return {
        **STRICT_DIRECT_FPM,
        "estimator_config": {"fpm_interpolation": {"method": "direct", "fpm_parquet_path": path}},
    }


def _qualified_prediction(tmp_path, engine_fields, timing):
    """Return a config mapping that serving validation accepts, and its trace."""
    trace = tmp_path / "trace.json"
    trace.write_text("{}\n")
    prediction = {
        "engine": {
            "model": "example/model",
            "hardware": "h200_sxm",
            "backend": "sglang",
            **copy.deepcopy(engine_fields),
            "workers": {"aggregated": {"timing": copy.deepcopy(timing)}},
        },
        "traffic": {
            "source": {"type": "trace", "format": "weka", "paths": [str(trace)]},
            "load": {"type": "trace_timestamps", "agentic_lanes": 1},
        },
    }
    assert serving._scope(prediction, trace)["qualification"] == "one_complete_single_stream_weka_play"
    return prediction, trace


def _estimator_config_replay_compiles(prediction):
    compiled = prediction_to_replay_spec(CorePredictionConfig.model_validate(copy.deepcopy(prediction)))
    return compiled.backend_deployment.agg_engine_args["timing_model"]["config"]["estimator_config"]


def _prepare_without_a_replay(tmp_path, config, trace):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    return serving.prepare_serving_validation(
        trace=trace,
        prediction_config=config,
        output=tmp_path / "serving",
        endpoint="http://127.0.0.1:8000",
        tokenizer=tokenizer,
        expected_execution={},
    )


@pytest.mark.parametrize(
    ("engine_fields", "timing", "path"),
    [
        pytest.param({}, {**STRICT_DIRECT_FPM, "fpm_parquet_path": OLD_PATH}, OLD_PATH, id="estimator on the worker"),
        pytest.param(STRICT_DIRECT_FPM, OLD_PATH_ONLY, OLD_PATH, id="estimator on the engine"),
        pytest.param(
            {},
            {**_strict_direct_fpm_naming(OLD_PATH), "fpm_parquet_path": OLD_PATH},
            OLD_PATH,
            id="the worker's estimator_config names the same file",
        ),
        pytest.param(
            _strict_direct_fpm_naming(OLD_PATH),
            OLD_PATH_ONLY,
            OLD_PATH,
            id="the engine's estimator_config names the same file",
        ),
        pytest.param(
            _strict_direct_fpm_naming(ANOTHER_PATH),
            {**STRICT_DIRECT_FPM, "fpm_parquet_path": OLD_PATH},
            OLD_PATH,
            id="the worker's estimator_config replaces one of the engine that names another file",
        ),
        pytest.param(
            {},
            {**STRICT_DIRECT_FPM, "fpm_parquet_path": "data/collected.parquet"},
            "data/collected.parquet",
            id="relative path",
        ),
        pytest.param(
            {},
            {
                "estimation_mode": "fpm_interpolation",
                "fallback_policy": "deny",
                "estimator_config": {
                    "fpm_interpolation": {"method": "direct", "collect_coverage": True},
                    "correction": {"enabled": False},
                },
                "fpm_parquet_path": OLD_PATH,
            },
            OLD_PATH,
            id="next to other estimator settings",
        ),
    ],
)
def test_estimator_config_names_the_file_of_the_old_fpm_parquet_path_as_replay_does(
    tmp_path, engine_fields, timing, path
):
    prediction, _ = _qualified_prediction(tmp_path, engine_fields, timing)
    as_written = copy.deepcopy(prediction)
    *_, estimator_config = serving._replayed_estimator(prediction)
    assert estimator_config["fpm_interpolation"]["fpm_parquet_path"] == path
    assert estimator_config == _estimator_config_replay_compiles(prediction)
    # The matched workload saves this mapping as its config, so it stays as written.
    assert prediction == as_written


@pytest.mark.parametrize(
    ("engine_fields", "timing"),
    [
        pytest.param(
            {},
            {**_strict_direct_fpm_naming(ANOTHER_PATH), "fpm_parquet_path": OLD_PATH},
            id="another file in the worker's estimator_config",
        ),
        pytest.param(
            _strict_direct_fpm_naming(ANOTHER_PATH),
            OLD_PATH_ONLY,
            id="another file in the engine's estimator_config",
        ),
        pytest.param(
            {},
            {**_strict_direct_fpm_naming(None), "fpm_parquet_path": OLD_PATH},
            id="null in the worker's estimator_config",
        ),
    ],
)
def test_old_fpm_parquet_path_that_conflicts_with_estimator_config_is_the_error_replay_raises(
    tmp_path, engine_fields, timing
):
    prediction, trace = _qualified_prediction(tmp_path, engine_fields, timing)
    with pytest.raises(ValueError) as from_replay:
        _estimator_config_replay_compiles(prediction)
    with pytest.raises(ValueError) as from_serving_validation:
        serving._replayed_estimator(prediction)
    assert type(from_serving_validation.value) is type(from_replay.value)
    assert str(from_serving_validation.value) == str(from_replay.value)
    assert (
        str(from_replay.value) == "conflicting fpm_parquet_path and estimator_config.fpm_interpolation.fpm_parquet_path"
    )

    config = tmp_path / "predict.yaml"
    config.write_text(yaml.safe_dump(prediction))
    with pytest.raises(ValueError, match="^conflicting fpm_parquet_path and estimator_config"):
        _prepare_without_a_replay(tmp_path, config, trace)
    assert not (tmp_path / "serving").exists()


@pytest.mark.parametrize("spelling", ["old fpm_parquet_path on the worker timing", "path in estimator_config"])
def test_parquet_file_without_its_metadata_file_stops_preparation_for_either_spelling_of_the_path(tmp_path, spelling):
    parquet = tmp_path / "forward.parquet"
    parquet.write_bytes(b"PAR1")
    timing = (
        _strict_direct_fpm_naming(str(parquet))
        if spelling == "path in estimator_config"
        else {**STRICT_DIRECT_FPM, "fpm_parquet_path": str(parquet)}
    )
    prediction, trace = _qualified_prediction(tmp_path, {}, timing)
    config = tmp_path / "predict.yaml"
    config.write_text(yaml.safe_dump(prediction))
    with pytest.raises(FileNotFoundError, match=re.escape(str(tmp_path / "forward.metadata.json"))):
        _prepare_without_a_replay(tmp_path, config, trace)
    # Nothing is left behind, so the same output folder can be used once the file is there.
    assert not (tmp_path / "serving").exists()


def test_key_written_twice_inside_ais_perf_model_stops_preparation(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text("{}\n")
    config = tmp_path / "predict.yaml"
    # The YAML reader keeps the last value, which alone would be accepted.
    config.write_text(
        "engine:\n"
        "  model: example/model\n"
        "  hardware: h200_sxm\n"
        "  backend: sglang\n"
        "  ais_perf_model:\n"
        "    estimation_mode: auto\n"
        "    fallback_policy: deny\n"
        "    estimation_mode: fpm_interpolation\n"
        "    estimator_config: {fpm_interpolation: {method: direct}}\n"
        "  workers: {aggregated: {}}\n"
        "traffic:\n"
        f"  source: {{type: trace, format: weka, paths: [{json.dumps(str(trace))}]}}\n"
        "  load: {type: trace_timestamps, agentic_lanes: 1}\n"
    )
    assert serving._scope(yaml.safe_load(config.read_text()), trace)
    written_twice = re.escape(
        f"engine.ais_perf_model in {config}: the key 'estimation_mode' is written twice (lines 6 and 8); write it once"
    )
    with pytest.raises(AisPerfModelError, match=f"^{written_twice}$"):
        serving._read_prediction_config(config)
    with pytest.raises(AisPerfModelError, match=f"^{written_twice}$"):
        _prepare_without_a_replay(tmp_path, config, trace)
    assert not (tmp_path / "serving").exists()


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
    return _prepared_case(request, tmp_path)


def _prepared_case(request, tmp_path, respell=None):
    """Replay the onboarding config, prepare serving validation for it and add synthetic serving evidence.

    ``respell`` rewrites the config onboarding wrote before serving validation
    reads it. It gets the config mapping and the folder of the config file and
    returns the mapping to save there, so a test can state the same estimator
    settings in another place of the config.
    """
    args, request, _plan, trace, prediction_output = request.getfixturevalue("_replay_validation_case")
    # This invokes the actual ordinary prediction/compiler/native FPM path;
    # imported AIPerf-shaped observations below are synthetic contract fixtures.
    assert cli.main(args) == 0
    config = yaml.safe_load((prediction_output / "predict.yaml").read_text())
    if respell is not None:
        config = respell(config, prediction_output)
        (prediction_output / "predict.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
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


# Prefill emits token one; the two decode steps read 64 and 65 past-KV tokens.
@pytest.mark.parametrize(
    "api_time,expected_arrival", [(0.0, 300 + 2.5 + (2 + 64 / 1024) + (2 + 65 / 1024)), (0.3, 300.0)]
)
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


@pytest.mark.parametrize("argument", ["latency_p95_error", "throughput_error"])
@pytest.mark.parametrize("limit", [False, True, -0.1, "0.2", float("nan"), float("inf"), float("-inf")])
def test_serving_error_limits_reject_invalid_values_in_preparation_and_assessment(
    serving_case, tmp_path, argument, limit
):
    case = serving_case
    with pytest.raises(ValueError, match=f"{argument} must"):
        serving.prepare_serving_validation(
            trace=case.trace,
            prediction_config=case.prediction_config,
            output=tmp_path / "invalid-limit",
            endpoint="http://127.0.0.1:8000",
            tokenizer=case.tokenizer,
            expected_execution=case.recipe["expected_execution"],
            aiperf_python=Path("/benchmark/bin/python"),
            **{argument: limit},
        )
    with pytest.raises(ValueError, match=f"{argument} must"):
        _assess(case, **{argument: limit})


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


def _forward_samples(case):
    """Save one measured prefill pass that is far from the prediction and one decode pass that is close to it."""
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
    return forward


def test_supplied_forward_samples_are_separate_from_request_latency(serving_case):
    case = serving_case
    report = _assess(case, forward_evidence_path=_forward_samples(case))
    assert report["status"] == "failed"
    assert report["forward"]["phases"]["prefill"]["status"] == "failed"
    assert report["forward"]["phases"]["decode"]["status"] == "passed"
    assert report["metrics"]["ttft"]["status"] == "passed"


ESTIMATOR_FIELDS = ("estimation_mode", "fallback_policy", "estimator_config")
COLLECTED_FORWARD_DATA = [
    "collection/systems/data/h200_sxm/vllm/0.25.1/fpm_forward_perf.metadata.json",
    "collection/systems/data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet",
]
COPIED_FORWARD_DATA = ["validation/external/forward.metadata.json", "validation/external/forward.parquet"]
ANOTHER_REVISION = ["expected execution model_revision differs from ordinary prediction configuration"]


def _copied_forward_data(engine, folder):
    """Copy the forward pass data under the engine's systems path to ``folder/external`` and return the parquet copy."""
    (collected,) = Path(engine["systems_paths"][0]).rglob("fpm_forward_perf.parquet")
    external = folder / "external"
    external.mkdir()
    shutil.copy(collected, external / "forward.parquet")
    shutil.copy(collected.with_suffix(".metadata.json"), external / "forward.metadata.json")
    return external / "forward.parquet"


def _estimator_on_the_engine(config, folder):
    """State the estimator once for the engine and leave the worker without a timing key."""
    engine = config["engine"]
    timing = engine["workers"]["aggregated"].pop("timing")
    engine.update({key: timing[key] for key in ESTIMATOR_FIELDS})
    return config


def _estimator_on_the_engine_with_a_parquet_file(config, folder):
    """State the estimator once for the engine with one more parquet file and leave the worker's timing empty."""
    engine = config["engine"]
    worker = engine["workers"]["aggregated"]
    engine.update({key: worker["timing"][key] for key in ESTIMATOR_FIELDS})
    worker["timing"] = {}
    engine["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] = str(_copied_forward_data(engine, folder))
    return config


def _estimator_on_the_engine_with_an_old_parquet_path(config, folder):
    """State the estimator once for the engine and name one more parquet file by the worker's old path field."""
    engine = config["engine"]
    worker = engine["workers"]["aggregated"]
    engine.update({key: worker["timing"][key] for key in ESTIMATOR_FIELDS})
    # The timing takes the old path field only next to ``forward_model: fpm``.
    worker["timing"] = {"forward_model": "fpm", "fpm_parquet_path": str(_copied_forward_data(engine, folder))}
    return config


def _estimator_under_ais_perf_model(config, folder):
    """State the estimator, the data paths and the profile under ``engine.ais_perf_model`` only.

    The systems path and one more parquet file are written relative to the
    folder of the config file, the way a hand-written config names them.
    """
    engine = config["engine"]
    timing = engine["workers"]["aggregated"].pop("timing")
    estimator_config = copy.deepcopy(timing["estimator_config"])
    estimator_config["fpm_interpolation"]["fpm_parquet_path"] = os.path.relpath(
        _copied_forward_data(engine, folder), folder
    )
    engine["ais_perf_model"] = {
        "estimation_mode": timing["estimation_mode"],
        "fallback_policy": timing["fallback_policy"],
        "estimator_config": estimator_config,
        "systems_paths": [os.path.relpath(root, folder) for root in engine.pop("systems_paths")],
        "fpm_profile": engine.pop("fpm_profile"),
    }
    return config


def _work_in_an_unrelated_folder(tmp_path, monkeypatch):
    """Leave the folder of the config, so that a path read from the working directory names nothing."""
    elsewhere = tmp_path / "elsewhere" / "deeper"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)


def _timing_artifacts(recipe, tmp_path):
    return [Path(item["path"]).relative_to(tmp_path.resolve()).as_posix() for item in recipe["timing_artifacts"]]


# How the config states the estimator, and the timing artifacts its data paths name.
STATED_OUTSIDE_THE_WORKER_TIMING = {
    "engine fields, worker without timing": (_estimator_on_the_engine, COLLECTED_FORWARD_DATA),
    "engine fields with a parquet file, empty worker timing": (
        _estimator_on_the_engine_with_a_parquet_file,
        COLLECTED_FORWARD_DATA + COPIED_FORWARD_DATA,
    ),
    "engine fields, old fpm_parquet_path on the worker timing": (
        _estimator_on_the_engine_with_an_old_parquet_path,
        COLLECTED_FORWARD_DATA + COPIED_FORWARD_DATA,
    ),
    "ais_perf_model with paths relative to the config": (
        _estimator_under_ais_perf_model,
        COLLECTED_FORWARD_DATA + COPIED_FORWARD_DATA,
    ),
}


@pytest.mark.parametrize("stated", STATED_OUTSIDE_THE_WORKER_TIMING)
def test_timing_artifacts_are_found_for_an_estimator_stated_outside_the_worker_timing(
    request, tmp_path, monkeypatch, materialized_workload, stated
):
    respell, timing_artifacts = STATED_OUTSIDE_THE_WORKER_TIMING[stated]
    _work_in_an_unrelated_folder(tmp_path, monkeypatch)
    case = _prepared_case(request, tmp_path, respell)
    assert (case.recipe["status"], case.recipe["issues"]) == ("ready", [])
    assert _timing_artifacts(case.recipe, tmp_path) == timing_artifacts


@pytest.mark.parametrize("stated", STATED_OUTSIDE_THE_WORKER_TIMING)
def test_forward_samples_are_priced_for_an_estimator_stated_outside_the_worker_timing(
    request, tmp_path, monkeypatch, materialized_workload, stated
):
    respell, _ = STATED_OUTSIDE_THE_WORKER_TIMING[stated]
    _work_in_an_unrelated_folder(tmp_path, monkeypatch)
    case = _prepared_case(request, tmp_path, respell)
    report = _assess(case, forward_evidence_path=_forward_samples(case))
    # The same data prices the samples as for the config onboarding wrote.
    assert {phase: result["status"] for phase, result in report["forward"]["phases"].items()} == {
        "prefill": "failed",
        "decode": "passed",
    }


def _ten_times_slower_forward_data(engine, folder):
    """Write the forward pass data under the engine's systems path, ten times slower, to ``folder/external``.

    A copy of the collected data prices a pass the same, so it cannot show
    which file priced it. Return the parquet file that was written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    (collected,) = Path(engine["systems_paths"][0]).rglob("fpm_forward_perf.parquet")
    external = folder / "external"
    external.mkdir()
    table = pq.read_table(collected)
    rows = [{**row, "latency_ms": row["latency_ms"] * 10} for row in table.to_pylist()]
    slower = external / "forward.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), slower)
    metadata = json.loads(collected.with_suffix(".metadata.json").read_text())
    metadata["parquet_sha256"] = hashlib.sha256(slower.read_bytes()).hexdigest()
    _save(external / "forward.metadata.json", metadata)
    return slower


@pytest.mark.parametrize("estimator_stated_on", ["worker timing", "engine"])
def test_forward_samples_are_priced_from_the_file_of_the_old_fpm_parquet_path_as_replay_is(
    request, tmp_path, materialized_workload, estimator_stated_on
):
    def respell(config, folder):
        engine = config["engine"]
        worker = engine["workers"]["aggregated"]
        if estimator_stated_on == "engine":
            engine.update({key: worker["timing"][key] for key in ESTIMATOR_FIELDS})
            worker["timing"] = {"forward_model": "fpm"}
        worker["timing"]["fpm_parquet_path"] = str(_ten_times_slower_forward_data(engine, folder))
        return config

    case = _prepared_case(request, tmp_path, respell)
    assert (case.recipe["status"], case.recipe["issues"]) == ("ready", [])
    assert _timing_artifacts(case.recipe, tmp_path) == COLLECTED_FORWARD_DATA + COPIED_FORWARD_DATA
    matched = case.recipe["matched_workload"]
    source_config = yaml.safe_load(case.prediction_config.read_text())
    assert yaml.safe_load(Path(matched["prediction_config"]["path"]).read_text())["engine"] == source_config["engine"]

    report = _assess(case, forward_evidence_path=_forward_samples(case))
    predicted = {phase: result["samples"][0]["predicted_ms"] for phase, result in report["forward"]["phases"].items()}
    # The collected data prices these two passes at 2.5 ms and 2 + 65 / 1024 ms.
    assert predicted == {"prefill": pytest.approx(25.0), "decode": pytest.approx(20 + 650 / 1024)}
    # Replay priced the cold 64-token prefill of the first request from the same file.
    replayed = json.loads(Path(matched["prediction_report"]["path"]).read_text())["per_request"]
    assert replayed[0]["ttft_ms"] == pytest.approx(predicted["prefill"])


def _forward_data_named_only_by_the_old_parquet_path(config, folder):
    """Move the forward pass data out of the engine's systems path and name it by the worker's old path field."""
    engine = config["engine"]
    (collected,) = Path(engine["systems_paths"][0]).rglob("fpm_forward_perf.parquet")
    external = folder / "external"
    external.mkdir()
    shutil.move(collected, external / "forward.parquet")
    shutil.move(collected.with_suffix(".metadata.json"), external / "forward.metadata.json")
    engine["workers"]["aggregated"]["timing"]["fpm_parquet_path"] = str(external / "forward.parquet")
    return config


def test_forward_data_that_only_the_old_fpm_parquet_path_names_is_found(request, tmp_path, materialized_workload):
    case = _prepared_case(request, tmp_path, _forward_data_named_only_by_the_old_parquet_path)
    assert (case.recipe["status"], case.recipe["issues"]) == ("ready", [])
    assert _timing_artifacts(case.recipe, tmp_path) == COPIED_FORWARD_DATA
    report = _assess(case, forward_evidence_path=_forward_samples(case))
    assert {phase: result["status"] for phase, result in report["forward"]["phases"].items()} == {
        "prefill": "failed",
        "decode": "passed",
    }


def _systems_paths_on_the_worker_timing_only(config, folder):
    """Move the systems paths from the engine to the worker's timing."""
    engine = config["engine"]
    engine["workers"]["aggregated"]["timing"]["systems_paths"] = engine.pop("systems_paths")
    return config


def _systems_paths_of_the_engine_name_a_folder_without_data(config, folder):
    """State the systems paths on the worker's timing too and point those of the engine at an empty folder."""
    engine = config["engine"]
    empty = folder / "no-data"
    empty.mkdir()
    engine["workers"]["aggregated"]["timing"]["systems_paths"] = engine["systems_paths"]
    engine["systems_paths"] = [str(empty)]
    return config


def _estimator_config_of_the_engine_names_one_more_parquet_file(config, folder):
    """State the estimator config for the engine too, with a parquet file the worker's timing does not name."""
    engine = config["engine"]
    stated_for_the_engine = copy.deepcopy(engine["workers"]["aggregated"]["timing"]["estimator_config"])
    stated_for_the_engine["fpm_interpolation"]["fpm_parquet_path"] = str(_copied_forward_data(engine, folder))
    engine["estimator_config"] = stated_for_the_engine
    return config


def test_forward_samples_are_priced_for_systems_paths_stated_only_on_the_worker_timing(
    request, tmp_path, materialized_workload
):
    case = _prepared_case(request, tmp_path, _systems_paths_on_the_worker_timing_only)
    assert (case.recipe["status"], case.recipe["issues"]) == ("ready", [])
    report = _assess(case, forward_evidence_path=_forward_samples(case))
    assert {phase: result["status"] for phase, result in report["forward"]["phases"].items()} == {
        "prefill": "failed",
        "decode": "passed",
    }


@pytest.mark.parametrize(
    "respell",
    [
        pytest.param(_systems_paths_of_the_engine_name_a_folder_without_data, id="systems_paths"),
        pytest.param(_estimator_config_of_the_engine_names_one_more_parquet_file, id="estimator_config"),
    ],
)
def test_setting_stated_on_the_worker_timing_and_on_the_engine_is_read_from_the_worker_timing(
    request, tmp_path, materialized_workload, respell
):
    # Replay uses the worker's value when it has one, so only the data that value names is in the recipe.
    case = _prepared_case(request, tmp_path, respell)
    assert (case.recipe["status"], case.recipe["issues"]) == ("ready", [])
    assert _timing_artifacts(case.recipe, tmp_path) == COLLECTED_FORWARD_DATA


def test_matched_config_for_old_fields_is_the_source_config_with_the_derived_trace(serving_case, tmp_path):
    matched = serving_case.recipe["matched_workload"]
    expected = yaml.safe_load(serving_case.prediction_config.read_text())
    assert "ais_perf_model" not in expected["engine"]
    expected["traffic"]["source"] = {"type": "trace", "paths": [matched["trace"]["path"]], "format": "agentic_mooncake"}
    assert Path(matched["prediction_config"]["path"]).read_text() == yaml.safe_dump(expected, sort_keys=False)
    assert _timing_artifacts(serving_case.recipe, tmp_path) == COLLECTED_FORWARD_DATA


def test_matched_config_keeps_ais_perf_model_with_paths_that_name_the_same_files(
    request, tmp_path, monkeypatch, materialized_workload
):
    _work_in_an_unrelated_folder(tmp_path, monkeypatch)
    case = _prepared_case(request, tmp_path, _estimator_under_ais_perf_model)
    source = yaml.safe_load(case.prediction_config.read_text())
    written = source["engine"]["ais_perf_model"]
    assert written["systems_paths"] == ["../collection/systems"]
    assert written["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] == "external/forward.parquet"

    matched = case.recipe["matched_workload"]
    assert Path(matched["prediction_config"]["path"]) == tmp_path / "serving/matched/predict.yaml"
    assert Path(matched["prediction_report"]["path"]) == tmp_path / "serving/matched/prediction/prediction.json"
    # The matched config is saved in another folder, so it names the files by
    # their full paths. Everything else is the config as it was written.
    written["systems_paths"] = [str(tmp_path / "collection/systems")]
    written["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] = str(
        tmp_path / "validation/external/forward.parquet"
    )
    source["traffic"]["source"] = {"type": "trace", "paths": [matched["trace"]["path"]], "format": "agentic_mooncake"}
    assert yaml.safe_load(Path(matched["prediction_config"]["path"]).read_text()) == source


def _prepared_again(case, output, **expected_changes):
    """Prepare serving validation for the case once more, for a serving stack that differs in ``expected_changes``."""
    recipe_path = serving.prepare_serving_validation(
        trace=case.trace,
        prediction_config=case.prediction_config,
        output=output,
        endpoint="http://127.0.0.1:8000",
        tokenizer=case.tokenizer,
        expected_execution={**case.observed["observed_execution"], **expected_changes},
        aiperf_python=Path("/benchmark/bin/python"),
    )
    return json.loads(recipe_path.read_text())


@pytest.mark.parametrize(
    "respell",
    [
        pytest.param(None, id="fpm_profile on the engine"),
        pytest.param(_estimator_under_ais_perf_model, id="fpm_profile under ais_perf_model"),
    ],
)
def test_another_model_revision_than_the_configured_profile_is_reported(
    request, tmp_path, materialized_workload, respell
):
    case = _prepared_case(request, tmp_path, respell)
    recipe = _prepared_again(case, tmp_path / "another-revision", model_revision="another-revision")
    assert (recipe["status"], recipe["issues"]) == ("incomplete", ANOTHER_REVISION)


def _systems_paths_on_the_timing(folder):
    return {"workers": {"aggregated": {"timing": {**STRICT_DIRECT_FPM, "systems_paths": [str(folder / "systems")]}}}}


def _systems_paths_in_an_entry_with_a_state_snapshot(folder):
    """Such an entry keeps every setting for its own worker; its path starts at the folder of the config."""
    entry = {
        **STRICT_DIRECT_FPM,
        "systems_paths": ["systems"],
        "from": {"state_snapshot": {"path": "snapshot", "sha256": "0" * 64}},
    }
    return {"ais_perf_model": entry, "workers": {"aggregated": {}}}


@pytest.mark.parametrize(
    "engine_fields",
    [
        pytest.param(_systems_paths_on_the_timing, id="timing.systems_paths"),
        pytest.param(_systems_paths_in_an_entry_with_a_state_snapshot, id="ais_perf_model entry with a state snapshot"),
    ],
)
def test_systems_paths_stated_for_the_worker_are_searched_for_timing_artifacts(tmp_path, monkeypatch, engine_fields):
    _work_in_an_unrelated_folder(tmp_path, monkeypatch)
    folder = tmp_path / "configs"
    collected = folder / "systems/data/h200_sxm/sglang/1.0/fpm_forward_perf.parquet"
    collected.parent.mkdir(parents=True)
    collected.write_bytes(b"PAR1")
    trace = tmp_path / "trace.json"
    trace.write_text("{}\n")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    prediction = {
        "engine": {"model": "example/model", "hardware": "h200_sxm", "backend": "sglang", **engine_fields(folder)},
        "traffic": {
            "source": {"type": "trace", "format": "weka", "paths": [str(trace)]},
            "load": {"type": "trace_timestamps", "agentic_lanes": 1},
        },
    }
    config = folder / "predict.yaml"
    config.write_text(yaml.safe_dump(prediction))

    recipe_path = serving.prepare_serving_validation(
        trace=trace,
        prediction_config=config,
        output=tmp_path / "serving",
        endpoint="http://127.0.0.1:8000",
        tokenizer=tokenizer,
        expected_execution={},
    )

    recipe = json.loads(recipe_path.read_text())
    assert _timing_artifacts(recipe, tmp_path) == ["configs/systems/data/h200_sxm/sglang/1.0/fpm_forward_perf.parquet"]
    assert "the prediction's collected FPM timing artifacts could not be located" not in recipe["issues"]
