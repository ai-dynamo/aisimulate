# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and assess a small, independently measured serving accuracy check.

The producer is upstream AIPerf, not another implementation of agentic scheduling.
Initially only one complete single-stream Weka play is qualified. Other plays
remain explicit incomplete checks; they must not be flattened or partially run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import uuid
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import yaml

from aisimulate.config import CorePredictionConfig

AIPERF_REVISION = "7db2ba37a62aa80c882bc90eaf61cc8073e2387b"
AIPERF_SOURCE = "https://github.com/ai-dynamo/aiperf"
SCHEMA = "aisimulate-serving-validation/v1"
EXECUTION_FIELDS = (
    "model",
    "model_revision",
    "backend",
    "backend_version",
    "image_digest",
    "hardware",
    "parallelism",
    "precision",
    "attention_groups",
    "graph_config",
    "scheduler",
    "context_length",
    "gpu_memory_utilization",
    "runtime_config",
)


def normalize_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Compare native runtime behavior while allowing checkpoint mount paths to differ.

    Model/checkpoint identity is verified separately. Revision, dtype, quantization,
    cache, scheduler, topology and graph settings remain in this comparison.
    """
    normalized = json.loads(json.dumps(config))
    model = normalized.get("model_config", {})
    for field in ("model", "tokenizer"):
        model.pop(field, None)
    # Collection uses phase-specific KV seeding. Serving's prefix cache flag is
    # checked against the replay scope separately, using the native observation.
    normalized.get("cache_config", {}).pop("enable_prefix_caching", None)
    return normalized


def _identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _threshold(value: float, label: str) -> float:
    value = _number(value, label, positive=True)
    if value > 1:
        raise ValueError(f"{label} must be in (0, 1]")
    return value


def _read_play(trace: Path) -> tuple[dict[str, Any] | None, list[str]]:
    text = trace.read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        documents = [json.loads(line) for line in text.splitlines() if line.strip()]
        if len(documents) != 1:
            return None, ["choose one complete representative Weka play; serving qualification accepts one play"]
        document = documents[0]
    if not isinstance(document, dict) or not isinstance(document.get("requests"), list):
        return None, ["the selected file must contain one complete Weka play object"]
    issues = []
    if not isinstance(document.get("id"), str) or not document["id"]:
        issues.append("the selected play needs its original nonempty Weka id")
    requests = document["requests"]
    if not requests:
        issues.append("the selected play is empty")
    previous = None
    for index, row in enumerate(requests):
        if not isinstance(row, dict) or row.get("type") not in {"n", "s"}:
            issues.append("branching plays are not yet qualified; select a complete single-stream play")
            break
        for key in ("in", "out"):
            if type(row.get(key)) is not int or row[key] <= 0:
                issues.append(f"request {index} needs positive {key} tokens for HTTP serving validation")
        try:
            timestamp = _number(row.get("t"), f"request {index} timestamp")
            _number(row.get("api_time", 0), f"request {index} api_time")
        except ValueError as error:
            issues.append(str(error))
            break
        hashes = row.get("hash_ids")
        if not isinstance(hashes, list) or not hashes:
            issues.append("single-stream qualification requires the original nonempty prefix hash lists")
            break
        if previous is not None and (
            timestamp < previous["t"] + previous.get("api_time", 0)
            or row.get("model") != previous.get("model")
            or hashes[: len(previous["hash_ids"])] != previous["hash_ids"]
        ):
            issues.append(
                "overlap, model changes or non-extending prefixes can create flattened agents; "
                "select a complete single-stream play without these ambiguities"
            )
            break
        previous = row
    if requests and not any(type(row) is dict and type(row.get("out")) is int and row["out"] > 1 for row in requests):
        issues.append("at least one multi-token response is required to assess TPOT")
    return document, issues


def _timing_subset(play: dict[str, Any]) -> str | None:
    """Identify source-invariant equivalence to both native timing bounds.

    AIS enforces max(not-before timestamp, predecessor completion + idle gap).
    AIPerf concurrency uses gaps, and fixed schedule uses timestamps. Either is
    equivalent for the subsets below, regardless of the measured response time.
    """
    pairs = list(pairwise(play["requests"]))
    if all(previous.get("api_time", 0) == 0 for previous, _ in pairs):
        return "zero_recorded_api_time"
    if all(abs(current["t"] - previous["t"] - previous.get("api_time", 0)) < 1e-9 for previous, current in pairs):
        return "zero_idle_gaps"
    return None


def _scope(prediction: dict[str, Any], trace: Path) -> dict[str, Any]:
    config = CorePredictionConfig.model_validate(prediction).model_dump(mode="json", exclude_none=True)
    traffic, engine = config["traffic"], config["engine"]
    source, load = traffic["source"], traffic["load"]
    if (
        source.get("type") != "trace"
        or source.get("format") != "weka"
        or [str(Path(path).expanduser().resolve()) for path in source.get("paths", [])] != [str(trace)]
        or source.get("nested_timestamp_basis") not in {None, "absolute"}
        or load.get("type") != "trace_timestamps"
        or load.get("agentic_lanes") != 1
        or load.get("agentic_snapshot") is not None
        or load.get("agentic_warmup", False)
        or load.get("speedup", 1) != 1
        or traffic.get("stop")
        or engine["mode"] != "aggregated"
        or engine.get("speculation") is not None
    ):
        raise ValueError("serving validation requires complete cold aggregated Weka replay at original speed, one lane")
    worker = engine["workers"]["aggregated"]
    if worker["parallelism"].get("replicas", 1) != 1:
        raise ValueError("serving validation initially qualifies exactly one aggregated worker")
    if worker["kv_cache"].get("host_offload") or worker["kv_cache"].get("g3_offload"):
        raise ValueError("serving validation initially qualifies HBM-only cache")
    timing = worker["timing"]
    if (
        timing.get("type") != "default"
        or timing.get("estimation_mode") != "fpm_interpolation"
        or timing.get("fallback_policy") != "deny"
        or timing.get("estimator_config", {}).get("fpm_interpolation", {}).get("method") != "direct"
    ):
        raise ValueError("serving validation requires ordinary strict direct-FPM prediction")
    return {
        "qualification": "one_complete_single_stream_weka_play",
        "replay": "cold_aggregated",
        "load_policy": "single_session_absolute_and_completion_bounds",
        "agentic_lanes": 1,
        "prefix_caching": worker["kv_cache"]["prefix_caching"],
        "cache_state_at_start": "cold",
        "kv_initialization": "real_request_history",
        "cache_storage": "hbm_only",
        "speculation": "disabled",
        "model_projection": engine["model"],
        "token_count_source": "server_usage",
    }


def _benchmark_arguments(recipe: dict[str, Any]) -> list[str]:
    return [
        "-m",
        "aiperf",
        "profile",
        "--url",
        recipe["endpoint"],
        "--model",
        recipe["scope"]["model_projection"],
        "--tokenizer",
        recipe["tokenizer"],
        "--endpoint-type",
        "chat",
        "--streaming",
        "--use-server-token-count",
        "--input-file",
        recipe.get("matched_workload", {}).get("payloads", recipe["benchmark_trace"])["path"],
        "--custom-dataset-type",
        "mooncake_trace" if recipe.get("matched_workload") else "weka_trace",
        "--fixed-schedule" if recipe["timing_subset"] == "zero_idle_gaps" else "--no-fixed-schedule",
        "--concurrency",
        "1",
        "--num-sessions",
        "1",
        "--request-count",
        str(recipe["request_count"]),
        "--extra-inputs",
        "ignore_eos:true",
        "--random-seed",
        "0",
        "--export-level",
        "records",
        "--artifact-dir",
        recipe["artifacts_dir"],
    ]


def _aiperf_installation(aiperf_python: Path) -> dict[str, Any]:
    probe = (
        "import importlib.metadata,json;d=importlib.metadata.distribution('aiperf');"
        "print(json.dumps({'version':d.version,'source':json.loads(d.read_text('direct_url.json') or '{}')}))"
    )
    checked = subprocess.run([str(aiperf_python), "-c", probe], check=True, text=True, capture_output=True)
    installation = json.loads(checked.stdout)
    source = installation.get("source", {})
    if (
        source.get("vcs_info", {}).get("commit_id") != AIPERF_REVISION
        or source.get("url", "").removesuffix(".git").rstrip("/") != AIPERF_SOURCE
    ):
        raise ValueError(f"install AIPerf from {AIPERF_SOURCE}.git@{AIPERF_REVISION} in the benchmark environment")
    return installation


def _materialize_payloads(recipe: dict[str, Any], output: Path, aiperf_python: Path) -> dict[str, Any]:
    installation = _aiperf_installation(aiperf_python)
    helper = output / "serving_workload.py"
    helper.write_bytes(files("aisimulate.support").joinpath("serving_workload.py").read_bytes())
    inputs = output / "materialization-input.json"
    _write(
        inputs,
        {
            "trace": recipe["benchmark_trace"]["path"],
            "output": str(output),
            "tokenizer": recipe["tokenizer"],
            "model": recipe["scope"]["model_projection"],
            "arguments": _benchmark_arguments(recipe)[2:],
        },
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("AIPERF_")}
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with (
        (output / "materialization.stdout.log").open("w") as stdout,
        (output / "materialization.stderr.log").open("w") as stderr,
    ):
        completed = subprocess.run(
            [str(aiperf_python), str(helper), str(inputs)], env=environment, stdout=stdout, stderr=stderr, check=False
        )
    if completed.returncode:
        raise ValueError(
            f"native AIPerf request materialization failed; inspect {output / 'materialization.stderr.log'}"
        )
    return {"installation": installation, "helper": _identity(helper), "input": _identity(inputs)}


def _prepare_matched_workload(
    recipe: dict[str, Any], prediction: dict[str, Any], output: Path, python: Path
) -> dict[str, Any]:
    from aisimulate.supervision import main as predict

    from .validation import _completion_issues

    output.mkdir()
    materialization = _materialize_payloads(recipe, output, python)
    tokenization_path = output / "tokenization.json"
    tokenization = json.loads(tokenization_path.read_text())
    play = json.loads(Path(recipe["benchmark_trace"]["path"]).read_text())
    original = json.loads(Path(recipe["prediction_report"]["path"]).read_text())
    identities = {int(row["request_id"].rsplit(":", 1)[-1]): row for row in original["per_request"]}
    requests = tokenization["requests"]
    if len(requests) != len(play["requests"]) or sorted(identities) != list(range(len(requests))):
        raise ValueError("materialization or original replay does not cover the complete selected play")
    # Explicit native graph input retains the source serial dependencies. Feeding
    # newly tokenized hashes through Weka's fork detector would change causality.
    header = {
        "schema": "dynamo.agentic_mooncake",
        "version": 2,
        # Preserve equality at every engine cache-block boundary, including
        # prefixes ending inside a larger source Weka block.
        "block_size": 1,
        "hash_id_scope": "local",
        "source": {"format": "target_tokenized_weka", "digest": _identity(tokenization_path)["sha256"]},
    }
    rows = []
    for index, (source, target) in enumerate(zip(play["requests"], requests, strict=True)):
        tokens = target["input_token_ids"]
        if target["turn_index"] != index or target["input_length"] != len(tokens):
            raise ValueError("native materialization has inconsistent input token identities")
        old = identities[index]
        dependencies = []
        if index:
            previous = play["requests"][index - 1]
            dependencies.append(
                {
                    "request_id": identities[index - 1]["request_id"],
                    "trigger": "completion",
                    "relation": "sequence",
                    "delay_ms": (source["t"] - previous["t"] - previous.get("api_time", 0)) * 1000,
                }
            )
        rows.append(
            {
                "request_id": old["request_id"],
                "play_id": old["play_id"],
                "session_id": old["session_id"],
                "model": source["model"],
                "input_length": len(tokens),
                "output_length": source["out"],
                "hash_ids": tokens,
                "not_before_ms": (source["t"] - play["requests"][0]["t"]) * 1000,
                "recorded_api_time_ms": source.get("api_time", 0) * 1000,
                "dependencies": dependencies,
            }
        )
    trace_path = output / "target-tokenized.agentic.jsonl"
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in [header, *rows]))
    derived = json.loads(json.dumps(prediction))
    derived["traffic"]["source"] = {"type": "trace", "paths": [str(trace_path)], "format": "agentic_mooncake"}
    config_path = output / "predict.yaml"
    config_path.write_text(yaml.safe_dump(derived, sort_keys=False))
    status = predict(
        [
            "predict",
            "--config",
            str(config_path),
            "--output-dir",
            str(output / "prediction"),
            "--capture-per-request",
            "--format",
            "json",
        ]
    )
    prediction_path = output / "prediction/prediction.json"
    coverage_path = output / "prediction/fpm-coverage.json"
    if status or not prediction_path.is_file() or not coverage_path.is_file():
        raise ValueError("ordinary prediction on frozen target-tokenized requests did not finish")
    replay = json.loads(prediction_path.read_text())
    issues = _completion_issues(replay)
    if issues or json.loads(coverage_path.read_text()).get("status") != "covered":
        raise ValueError("frozen target-tokenized requests lack complete direct-FPM coverage: " + "; ".join(issues))
    return {
        **materialization,
        "payloads": _identity(output / "requests.jsonl"),
        "tokenization": _identity(tokenization_path),
        "trace": _identity(trace_path),
        "prediction_config": _identity(config_path),
        "prediction_report": _identity(prediction_path),
        "coverage_report": _identity(coverage_path),
    }


def prepare_serving_validation(
    *,
    trace: Path,
    prediction_config: Path,
    output: Path,
    endpoint: str,
    tokenizer: Path,
    expected_execution: dict[str, Any],
    latency_p95_error: float = 0.20,
    throughput_error: float = 0.20,
    prediction_report: Path | None = None,
    aiperf_python: Path | None = None,
    max_dispatch_delay_ms: float = 5.0,
    max_dispatch_delay_fraction: float = 0.01,
) -> Path:
    """Save a reviewed, reproducible producer recipe; never start serving or a benchmark."""
    trace, prediction_config, output = (path.expanduser().resolve() for path in (trace, prediction_config, output))
    tokenizer = tokenizer.expanduser().resolve()
    parsed_url = urlsplit(endpoint)
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.hostname
        or parsed_url.username
        or parsed_url.password
    ):
        raise ValueError("endpoint must be an HTTP(S) serving URL without embedded credentials")
    if not tokenizer.is_dir():
        raise ValueError("tokenizer must be an existing local directory for the pinned checkpoint")
    for source in (trace, prediction_config, tokenizer):
        if source == output or output in source.parents:
            raise ValueError("serving output must not contain an input")
    prediction = yaml.safe_load(prediction_config.read_text(encoding="utf-8"))
    scope = _scope(prediction, trace)
    play, issues = _read_play(trace)
    timing_subset = _timing_subset(play) if play is not None and not issues else None
    if play is not None and not issues and timing_subset is None:
        issues.append(
            "this play has both positive recorded API times and positive idle gaps: AIS replay enforces "
            "both absolute and completion bounds, but pinned AIPerf cannot combine them. Choose a complete "
            "representative play with zero/absent predecessor API times or all zero idle gaps; do not edit "
            "its timestamps to force qualification. Broader trace timing alignment remains required."
        )
    prediction_report = (prediction_report or prediction_config.parent / "prediction/prediction.json").resolve()
    if not prediction_report.is_file():
        issues.append("run ordinary onboarding replay with per-request capture before preparing serving validation")
    if aiperf_python is None:
        issues.append("provide the pinned AIPerf Python for request materialization before matched prediction")
    engine = prediction["engine"]
    worker = engine["workers"]["aggregated"]
    configured = {key: engine.get(key) for key in ("model", "backend", "backend_version", "hardware", "context_length")}
    configured.update(
        model_revision=engine.get("fpm_profile", {}).get("model_revision"),
        parallelism=worker.get("parallelism"),
        scheduler=worker.get("scheduler"),
        gpu_memory_utilization=worker.get("kv_cache", {}).get("capacity", {}).get("memory_fraction"),
    )
    for field, value in configured.items():
        if value is not None and expected_execution.get(field) != value:
            issues.append(f"expected execution {field} differs from ordinary prediction configuration")
    for field in EXECUTION_FIELDS:
        if expected_execution.get(field) in (None, "", {}, []):
            issues.append(f"expected collection execution has no resolved {field}; inspect all phases and ranks")
    tokenizer_files = [
        _identity(path)
        for path in sorted(tokenizer.rglob("*"))
        if path.is_file()
        and path.suffix in {".json", ".model", ".txt", ".tiktoken", ".jinja", ".jinja2", ".vocab", ".bpe"}
    ]
    if not tokenizer_files:
        issues.append("the pinned tokenizer directory has no tokenizer/config artifacts")
    timing_paths = {
        path.resolve()
        for root in engine.get("systems_paths", [])
        for path in Path(root).expanduser().rglob("fpm_forward_perf.*")
        if path.is_file()
    }
    external = worker["timing"].get("estimator_config", {}).get("fpm_interpolation", {}).get("fpm_parquet_path")
    if external:
        timing_paths.update((Path(external).resolve(), Path(external).with_suffix(".metadata.json").resolve()))
    if not timing_paths:
        issues.append("the prediction's collected FPM timing artifacts could not be located")
    output.mkdir(parents=True, exist_ok=False)
    observer_dir = output / "observer"
    observer_dir.mkdir()
    observer_files = []
    runtime = files("collector.fpm_forward.runtime")
    for name in ("fpm_memory_observer.py", "fpm_memory_worker.py"):
        target = observer_dir / name
        target.write_bytes(runtime.joinpath(name).read_bytes())
        observer_files.append(_identity(target))
    benchmark_trace = output / "play.json"
    if play is not None:
        _write(benchmark_trace, play)
    recipe = {
        "schema_version": SCHEMA,
        "status": "incomplete" if issues else "ready",
        "issues": issues,
        "trace": _identity(trace),
        "prediction_config": _identity(prediction_config),
        "prediction_report": _identity(prediction_report) if prediction_report.is_file() else None,
        "benchmark_trace": _identity(benchmark_trace) if play is not None else None,
        "scope": scope,
        "play_id": play.get("id") if play else None,
        "timing_subset": timing_subset,
        "request_count": len(play["requests"]) if play else 0,
        "aiperf": {"repository": AIPERF_SOURCE, "revision": AIPERF_REVISION},
        "endpoint": endpoint,
        "tokenizer": str(tokenizer),
        "tokenizer_files": tokenizer_files,
        "timing_artifacts": [_identity(path) for path in sorted(timing_paths)],
        "expected_execution": expected_execution,
        "policy": {
            "latency_p95_relative_error": _threshold(latency_p95_error, "latency_p95_error"),
            "throughput_relative_error": _threshold(throughput_error, "throughput_error"),
            "max_dispatch_delay_ms": _number(max_dispatch_delay_ms, "max_dispatch_delay_ms"),
            "max_dispatch_delay_fraction": _number(max_dispatch_delay_fraction, "max_dispatch_delay_fraction"),
        },
        "artifacts_dir": str(output / "aiperf"),
        "serving_observer": {
            "module_files": observer_files,
            "worker_class": "fpm_memory_worker.FpmExecutionWorker",
            "provenance_path": str(output / "serving-provenance.json"),
            "output_dir": str(output / "runtime-observations"),
            "launch_arguments": ["--worker-cls", "fpm_memory_worker.FpmExecutionWorker"],
            "environment": {
                "PYTHONPATH": str(observer_dir),
                "FPM_EXECUTION_OUTPUT_DIR": str(output / "runtime-observations"),
                "FPM_EXECUTION_PROVENANCE_FILE": str(output / "serving-provenance.json"),
            },
            "instructions": "Add these arguments/environment to the reviewed ordinary vLLM serving launch. "
            "Make the observer files and provenance available at these absolute paths inside every worker container. "
            "Prepend the observer directory to any existing PYTHONPATH. Start a fresh server or reset its prefix "
            "cache after initialization. A server started without the hook must be relaunched or remain incomplete. "
            "The hook records selected attention groups, effective graphs and resolved runtime configuration. "
            "Retain the scheduler/container launch record, mounted checkpoint revision and nvidia-smi output "
            "as separate identity evidence; the observer does not cryptographically verify model weights or images.",
        },
    }
    if not issues:
        try:
            recipe["matched_workload"] = _prepare_matched_workload(
                recipe, prediction, output / "matched", aiperf_python
            )
        except (OSError, ValueError, RuntimeError) as error:
            issues.append(str(error))
    recipe["status"] = "incomplete" if issues else "ready"
    recipe["command_arguments"] = _benchmark_arguments(recipe) if not issues else None
    path = output / "recipe.json"
    _write(path, recipe)
    _write(
        output / "serving-provenance.json",
        {
            "schema_version": SCHEMA,
            "run_id": str(uuid.uuid4()),
            "purpose": "matched_serving_accuracy",
            "recipe": _identity(path),
            "trace": recipe["trace"],
            "observer_modules": observer_files,
        },
    )
    _write(
        output / "observed-execution.template.json",
        {
            "schema_version": SCHEMA,
            "recipe": _identity(path),
            "observed_execution": dict.fromkeys(EXECUTION_FIELDS),
            "scope": dict.fromkeys(scope),
            "sources": [],
            "note": "Populate from serving initialization and benchmark evidence; missing values never pass. "
            "Do not copy expected values as observations. Record native attention groups and effective graphs. "
            "Benchmark real_prefix/real_kv preparation is separate from serving's actual request history.",
        },
    )
    return path


def _checked_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    if recipe.get("schema_version") != SCHEMA:
        raise ValueError("unsupported serving validation recipe schema")
    if recipe.get("status") != "ready":
        raise ValueError("serving validation preparation is incomplete: " + "; ".join(recipe.get("issues", [])))
    if recipe.get("aiperf") != {"repository": AIPERF_SOURCE, "revision": AIPERF_REVISION}:
        raise ValueError("serving recipe must use the qualified AIPerf source pin")
    if recipe.get("command_arguments") != _benchmark_arguments(recipe):
        raise ValueError("serving command differs from the fixed producer recipe")
    for key in ("trace", "prediction_config", "prediction_report", "benchmark_trace"):
        if _identity(Path(recipe[key]["path"])) != recipe[key]:
            raise ValueError(f"serving recipe {key} changed after preparation")
    for source in [*recipe["tokenizer_files"], *recipe["timing_artifacts"]]:
        if _identity(Path(source["path"])) != source:
            raise ValueError("pinned tokenizer or FPM timing files changed after serving preparation")
    matched = recipe["matched_workload"]
    for key in (
        "helper",
        "input",
        "payloads",
        "tokenization",
        "trace",
        "prediction_config",
        "prediction_report",
        "coverage_report",
    ):
        if _identity(Path(matched[key]["path"])) != matched[key]:
            raise ValueError(f"matched serving {key} changed after preparation")
    for source in recipe["serving_observer"]["module_files"]:
        if _identity(Path(source["path"])) != source:
            raise ValueError("serving observer module changed after preparation")
    provenance = json.loads(Path(recipe["serving_observer"]["provenance_path"]).read_text(encoding="utf-8"))
    if (
        provenance.get("recipe") != _identity(path)
        or provenance.get("observer_modules") != recipe["serving_observer"]["module_files"]
    ):
        raise ValueError("serving observer provenance does not match this recipe and its modules")
    return recipe


def _tokenize_serving(recipe_path: Path, recipe: dict[str, Any]) -> Path:
    """Check actual server rendering, without running a model or warming KV."""
    frozen = json.loads(Path(recipe["matched_workload"]["tokenization"]["path"]).read_text())
    payloads = [
        json.loads(row) for row in Path(recipe["matched_workload"]["payloads"]["path"]).read_text().splitlines()
    ]
    endpoint = recipe["endpoint"].rstrip("/").removesuffix("/v1") + "/tokenize"
    rows = []
    for index, (request, tokens) in enumerate(zip(payloads, frozen["requests"], strict=True)):
        payload = request["payload"]
        body = {key: payload[key] for key in ("model", "messages", "add_generation_prompt", "continue_final_message")}
        body["add_special_tokens"] = False
        http = Request(
            endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
        )
        with urlopen(http, timeout=60) as response:
            actual = json.loads(response.read())
        rows.append({"turn_index": index, "request": body, "response": actual})
        if actual.get("tokens") != tokens["input_token_ids"] or actual.get("count") != tokens["input_length"]:
            path = recipe_path.parent / "server-tokenization.json"
            _write(path, {"recipe": _identity(recipe_path), "endpoint": endpoint, "requests": rows, "status": "failed"})
            raise ValueError(f"serving tokenizer/chat template differs at turn {index}; inspect {path}")
    path = recipe_path.parent / "server-tokenization.json"
    _write(path, {"recipe": _identity(recipe_path), "endpoint": endpoint, "requests": rows, "status": "matched"})
    return path


def run_serving_benchmark(recipe_path: Path, *, aiperf_python: Path) -> Path:
    """Run only the fixed AIPerf producer against a caller-owned ready endpoint.

    Install AIPerf in a separate environment from the pinned Git URL. The PEP 610
    check rejects a mutable or unrelated installation before sending traffic.
    """
    recipe_path = recipe_path.expanduser().resolve()
    recipe = _checked_recipe(recipe_path)
    output = recipe_path.parent
    receipt_path = output / "benchmark-execution.json"
    if receipt_path.exists() or Path(recipe["artifacts_dir"]).exists():
        raise ValueError("use a fresh serving validation directory for each measurement")
    runtime_sources = [
        _identity(path)
        for path in sorted(Path(recipe["serving_observer"]["output_dir"]).glob("fpm-execution-worker-*.json"))
    ]
    if not runtime_sources:
        raise ValueError("start the ordinary serving worker with the prepared observer hook before benchmarking")
    installation = _aiperf_installation(aiperf_python)
    command = [str(aiperf_python), *_benchmark_arguments(recipe)]
    receipt = {
        "schema_version": SCHEMA,
        "recipe": _identity(recipe_path),
        "installation": installation,
        "command": command,
        "status": "incomplete",
        "exit_code": None,
        "serving_provenance": _identity(Path(recipe["serving_observer"]["provenance_path"])),
        "observer_modules": recipe["serving_observer"]["module_files"],
        "runtime_observations": runtime_sources,
    }
    missing, failures = [], []
    _runtime_observations(recipe, receipt, {"observed_execution": recipe["expected_execution"]}, missing, failures)
    if missing or failures:
        raise ValueError("serving worker does not match the prepared execution: " + "; ".join(missing + failures))
    receipt["server_tokenization"] = _identity(_tokenize_serving(recipe_path, recipe))
    _write(receipt_path, receipt)
    # Preserve normal credentials/proxy access; prevent hidden AIPerf settings
    # from changing the reviewed traffic or prefix reconstruction policy.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("AIPERF_")}
    with (output / "aiperf.stdout.log").open("w") as stdout, (output / "aiperf.stderr.log").open("w") as stderr:
        completed = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr, check=False)
    receipt.update(exit_code=completed.returncode, status="completed" if completed.returncode == 0 else "failed")
    receipt["artifacts"] = [
        _identity(path) for path in sorted(Path(recipe["artifacts_dir"]).rglob("profile_export*")) if path.is_file()
    ]
    _write(receipt_path, receipt)
    _checked_recipe(recipe_path)
    return receipt_path


def _metric(row: dict[str, Any], name: str, unit: str) -> float:
    metric = row.get("metrics", {}).get(name)
    if not isinstance(metric, dict) or metric.get("unit") != unit:
        raise ValueError(f"AIPerf {name} is missing or has a different unit from {unit}")
    return _number(metric.get("value"), f"AIPerf {name}")


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    return ordered[lower] + (ordered[math.ceil(index)] - ordered[lower]) * (index - lower)


def _error(predicted: float, measured: float) -> float:
    if measured <= 0:
        raise ValueError("relative timing error requires a positive measured value")
    return abs(predicted - measured) / measured


def _compare_samples(samples: list[dict[str, Any]], metric: str, threshold: float) -> dict[str, Any]:
    errors = [row[f"{metric}_relative_error"] for row in samples if row.get(f"{metric}_relative_error") is not None]
    if not errors:
        return {"status": "incomplete", "sample_count": 0, "issue": f"no valid {metric} samples"}
    p95 = _percentile(errors, 0.95)
    return {
        "status": "passed" if p95 <= threshold else "failed",
        "sample_count": len(errors),
        "p95_relative_error": p95,
        "max_relative_error": max(errors),
        "threshold": threshold,
    }


def _verify_sources(sources: Any, issues: list[str], failures: list[str]) -> None:
    if not isinstance(sources, list) or not sources:
        issues.append("observed execution needs hashed serving initialization evidence")
        return
    for source in sources:
        try:
            if _identity(Path(source["path"])) != source:
                failures.append(f"execution source changed: {source.get('path')}")
        except (OSError, KeyError, TypeError):
            issues.append("an execution evidence source is missing or malformed")


def _runtime_observations(
    recipe: dict[str, Any], receipt: dict[str, Any], observed: dict[str, Any], missing: list[str], failures: list[str]
) -> list[dict[str, Any]]:
    sources = receipt.get("runtime_observations", [])
    if not sources:
        missing.append("native serving worker observations are missing; relaunch with the prepared observer hook")
        return []
    _verify_sources(sources, missing, failures)
    provenance_path = Path(recipe["serving_observer"]["provenance_path"])
    provenance = json.loads(provenance_path.read_text())
    if receipt.get("serving_provenance") != _identity(provenance_path):
        failures.append("benchmark receipt is not bound to the prepared serving provenance")
    if receipt.get("observer_modules") != recipe["serving_observer"]["module_files"]:
        failures.append("benchmark receipt is not bound to the prepared observer modules")
    ranks = set()
    for source in sources:
        try:
            payload = json.loads(Path(source["path"]).read_text())
            if payload.get("schema_name") != "aisimulate_fpm_runtime_execution" or payload.get("schema_version") != 1:
                raise ValueError("unknown native serving observation schema")
            if payload.get("status") != "observed":
                raise ValueError("serving runtime did not resolve selected attention groups and graph configuration")
            if payload.get("execution_provenance") != provenance or payload.get("provenance_source") != _identity(
                provenance_path
            ):
                failures.append("native serving observations belong to a different prepared server launch")
            rank = tuple(payload.get(key) for key in ("dp_rank", "tp_rank", "pp_rank"))
            if any(type(value) is not int or value < 0 for value in rank) or rank in ranks:
                failures.append("native serving observation has a duplicate or invalid worker rank")
            ranks.add(rank)
            for key in ("attention_groups", "graph_config", "backend_version"):
                actual = observed.get("observed_execution", {}).get(key)
                if actual is not None and payload.get(key) != actual:
                    failures.append(f"reported serving {key} differs from the native initialized worker")
            native_config = payload.get("resolved_config")
            if not isinstance(native_config, dict) or not native_config:
                missing.append("native serving worker resolved runtime configuration is missing")
            else:
                prefix = native_config.get("cache_config", {}).get("enable_prefix_caching")
                if not isinstance(prefix, bool):
                    missing.append("native serving prefix cache setting is unresolved")
                elif prefix != recipe["scope"]["prefix_caching"]:
                    failures.append("native serving prefix cache setting differs from the replay scope")
                actual = observed.get("observed_execution", {}).get("runtime_config")
                if actual is not None and normalize_runtime_config(native_config) != actual:
                    failures.append("reported serving runtime_config differs from the native initialized worker")
        except (OSError, ValueError, TypeError) as error:
            missing.append(f"serving worker observation is unavailable: {error}")
    parallel = recipe["expected_execution"].get("parallelism", {})
    expected = {
        (dp, tp, pp)
        for dp in range(parallel.get("attention_data", 1))
        for tp in range(parallel.get("tensor", 1))
        for pp in range(parallel.get("pipeline", 1))
    }
    if ranks != expected:
        missing.append("native serving observations do not cover the selected worker ranks exactly")
    return sources


def _check_server_tokenization(
    recipe_path: Path, recipe: dict[str, Any], receipt: dict[str, Any], missing: list[str], failures: list[str]
) -> None:
    source = receipt.get("server_tokenization")
    if source is None:
        missing.append("actual serving /tokenize evidence is missing")
        return
    try:
        if _identity(Path(source["path"])) != source:
            failures.append("actual serving tokenization evidence changed")
        evidence = json.loads(Path(source["path"]).read_text())
        if evidence.get("recipe") != _identity(recipe_path) or evidence.get("status") != "matched":
            failures.append("serving tokenization does not match the prepared workload")
        frozen = json.loads(Path(recipe["matched_workload"]["tokenization"]["path"]).read_text())
        payloads = [
            json.loads(row) for row in Path(recipe["matched_workload"]["payloads"]["path"]).read_text().splitlines()
        ]
        if len(evidence.get("requests", [])) != len(frozen["requests"]):
            missing.append("serving tokenization did not cover every prepared request")
            return
        for index, (actual, expected, payload) in enumerate(
            zip(evidence["requests"], frozen["requests"], payloads, strict=True)
        ):
            request = {
                key: payload["payload"][key]
                for key in ("model", "messages", "add_generation_prompt", "continue_final_message")
            }
            request["add_special_tokens"] = False
            if (
                actual.get("turn_index") != index
                or actual.get("request") != request
                or actual.get("response", {}).get("tokens") != expected["input_token_ids"]
                or actual.get("response", {}).get("count") != expected["input_length"]
            ):
                failures.append(f"serving tokenization differs from frozen payload at turn {index}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        missing.append(f"serving tokenization evidence is unavailable: {error}")


def _forward_validation(path: Path | None, recipe_path: Path, threshold: float) -> dict[str, Any]:
    if path is None:
        return {"status": "unavailable", "issue": "no independently instrumented serving forward timings supplied"}
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"status": "incomplete", "issue": f"forward evidence is unavailable: {error}"}
    missing, failures = [], []
    _verify_sources(evidence.get("sources"), missing, failures)
    if evidence.get("recipe") != _identity(recipe_path):
        failures.append("forward evidence belongs to a different serving recipe")
    from aisimulate_core.sdk import RustForwardPassPerfModel

    recipe = _checked_recipe(recipe_path)
    prediction = yaml.safe_load(Path(recipe["prediction_config"]["path"]).read_text())
    engine = prediction["engine"]
    worker = engine["workers"]["aggregated"]
    parallel = worker["parallelism"]
    config = {
        "model": engine["model"],
        "system": engine["hardware"],
        "backend": engine["backend"],
        "backend_version": engine["backend_version"],
        "worker_type": "aggregated",
        "tp": parallel.get("tensor", 1),
        "pp": parallel.get("pipeline", 1),
        "attention_dp": parallel.get("attention_data", 1),
        "moe_tp_size": parallel.get("moe_tensor", 1),
        "moe_ep_size": parallel.get("moe_expert", 1),
        "fpm_profile": engine["fpm_profile"],
        "systems_paths": engine["systems_paths"],
        "estimation_mode": "fpm_interpolation",
        "fallback_policy": "deny",
        "estimator_config": worker["timing"]["estimator_config"],
    }
    try:
        model = RustForwardPassPerfModel.best_available(config)
    except (ValueError, RuntimeError) as error:
        return {
            "status": "incomplete",
            "issue": f"forward timing construction failed: {error}",
            "evidence": _identity(path),
        }
    phases: dict[str, Any] = {}
    try:
        for phase in ("prefill", "decode"):
            rows = evidence.get("samples", {}).get(phase, [])
            if not rows:
                phases[phase] = {
                    "status": "unavailable",
                    "sample_count": 0,
                    "issue": "no serving instrumentation for this phase",
                }
                continue
            samples = []
            unsupported = []
            for row in rows:
                try:
                    if row.get("unit") != "ms" or not isinstance(row.get("scheduled_requests"), dict):
                        raise ValueError("forward samples require native scheduled_requests coordinates and ms units")
                    measured = _number(row.get("measured_ms"), "measured forward latency", positive=True)
                    scheduled = row["scheduled_requests"]
                    if (
                        phase == "prefill"
                        and (
                            scheduled.get("num_prefill_requests", 0) < 1 or scheduled.get("num_decode_requests", 0) != 0
                        )
                    ) or (
                        phase == "decode"
                        and (
                            scheduled.get("num_decode_requests", 0) < 1 or scheduled.get("num_prefill_requests", 0) != 0
                        )
                    ):
                        raise ValueError("forward phase does not match scheduled request coordinates")
                    predicted = model.estimate_forward_pass_time_ms({"scheduled_requests": scheduled})
                    predicted = _number(predicted, "native direct-FPM forward estimate", positive=True)
                    samples.append(
                        {**row, "predicted_ms": predicted, "forward_relative_error": _error(predicted, measured)}
                    )
                except (ValueError, RuntimeError) as error:
                    unsupported.append({"scheduled_requests": row.get("scheduled_requests"), "reason": str(error)})
            phases[phase] = {
                **_compare_samples(samples, "forward", threshold),
                "samples": samples,
                "unsupported": unsupported,
            }
            if unsupported:
                missing.append(f"{phase} forward timing queries were unsupported or invalid")
    finally:
        model.close()
    statuses = [phase["status"] for phase in phases.values()]
    if all(status == "unavailable" for status in statuses):
        missing.append("supplied forward evidence contains no measured phase samples")
    status = (
        "failed"
        if failures or "failed" in statuses
        else "incomplete"
        if missing or "incomplete" in statuses
        else "passed"
    )
    return {
        "status": status,
        "phases": phases,
        "missing_evidence": missing,
        "failures": failures,
        "evidence": _identity(path),
    }


def validate_serving_measurements(
    recipe_path: Path,
    *,
    prediction_report: Path,
    execution_evidence_path: Path,
    output_report: Path | None,
    forward_evidence_path: Path | None = None,
    latency_p95_error: float | None = None,
    throughput_error: float | None = None,
    max_dispatch_delay_ms: float | None = None,
    max_dispatch_delay_fraction: float | None = None,
) -> dict[str, Any]:
    """Compare real AIPerf records with ordinary prediction; retain all request evidence."""
    recipe_path = recipe_path.expanduser().resolve()
    recipe = _checked_recipe(recipe_path)
    policy = dict(recipe["policy"])
    if latency_p95_error is not None:
        policy["latency_p95_relative_error"] = _threshold(latency_p95_error, "latency_p95_error")
    if throughput_error is not None:
        policy["throughput_relative_error"] = _threshold(throughput_error, "throughput_error")
    for key, value in (
        ("max_dispatch_delay_ms", max_dispatch_delay_ms),
        ("max_dispatch_delay_fraction", max_dispatch_delay_fraction),
    ):
        if value is not None:
            policy[key] = _number(value, key)
    if output_report is not None and output_report.exists():
        raise ValueError("save each serving assessment to a fresh report path; raw measurements can be reused")
    missing, failures = [], []
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "incomplete",
        "recipe": _identity(recipe_path),
        "scope": recipe["scope"],
        "timing_subset": recipe["timing_subset"],
        "matched_workload": recipe["matched_workload"],
        "policy": policy,
        "missing_evidence": missing,
        "failures": failures,
        "definitions": {
            "ttft_ms": "first token minus request dispatch; serving is client observed, simulation is engine observed",
            "tpot_ms": "per-request mean inter-token latency; only responses with at least two tokens",
            "output_throughput_tokens_per_second": (
                "total output tokens / (last response - first dispatch), including inter-turn delays"
            ),
            "relative_error": "abs(simulated - measured) / measured; p95 uses linear empirical quantile",
            "kv_preparation": "benchmark seeds KV history; serving populates actual request history",
        },
        "samples": [],
    }
    try:
        prediction = json.loads(
            Path(recipe["matched_workload"]["prediction_report"]["path"]).read_text(encoding="utf-8")
        )
        observed = json.loads(execution_evidence_path.read_text(encoding="utf-8"))
        receipt_path = recipe_path.parent / "benchmark-execution.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        missing.append(f"required prediction, execution receipt or serving observation is unavailable: {error}")
        if output_report is not None:
            _write(output_report, report)
        return report
    report["inputs"] = [_identity(path) for path in (prediction_report, execution_evidence_path, receipt_path)]
    if _identity(prediction_report) != recipe["prediction_report"]:
        failures.append("prediction report is not the ordinary replay bound to the prepared recipe")
    if observed.get("recipe") != report["recipe"] or receipt.get("recipe") != report["recipe"]:
        failures.append("serving observations or benchmark receipt belong to a different prepared recipe")
    if receipt.get("status") != "completed" or receipt.get("exit_code") != 0:
        missing.append("AIPerf benchmark did not complete successfully")
    if receipt.get("command", [None])[1:] != _benchmark_arguments(recipe):
        failures.append("executed benchmark command differs from the prepared workload")
    if receipt.get("installation", {}).get("source", {}).get("vcs_info", {}).get("commit_id") != AIPERF_REVISION:
        missing.append("executed AIPerf installation is not bound to the qualified source revision")
    _verify_sources(observed.get("sources"), missing, failures)
    report["runtime_observations"] = _runtime_observations(recipe, receipt, observed, missing, failures)
    _check_server_tokenization(recipe_path, recipe, receipt, missing, failures)
    for field in EXECUTION_FIELDS:
        expected = recipe["expected_execution"].get(field)
        actual = observed.get("observed_execution", {}).get(field)
        if expected in (None, "", {}, []) or actual in (None, "", {}, []):
            missing.append(f"missing expected or observed effective execution: {field}")
        elif expected != actual:
            failures.append(f"effective serving execution differs from collection: {field}")
    for field, expected in recipe["scope"].items():
        actual = observed.get("scope", {}).get(field)
        if actual is None:
            missing.append(f"missing observed serving scope: {field}")
        elif actual != expected:
            failures.append(f"serving and simulated workload scope differ: {field}")
    artifacts = receipt.get("artifacts", [])
    summaries = [item for item in artifacts if Path(item.get("path", "")).name == "profile_export_aiperf.json"]
    if len(summaries) != 1:
        missing.append("one native AIPerf profile_export_aiperf.json summary is required")
    else:
        try:
            source = summaries[0]
            if _identity(Path(source["path"])) != source:
                failures.append("AIPerf summary changed after the producer run")
            summary = json.loads(Path(source["path"]).read_text(encoding="utf-8"))
            if summary.get("was_cancelled") is not False or summary.get("is_complete") is False:
                missing.append("native AIPerf summary did not report an uncancelled, complete run")
            if summary.get("request_count", {}).get("avg") != recipe["request_count"]:
                failures.append("native AIPerf summary request count differs from the selected play")
            if summary.get("error_request_count", {}).get("avg", 0) != 0:
                failures.append("native AIPerf summary includes failed requests")
        except (OSError, ValueError, KeyError, TypeError) as error:
            missing.append(f"AIPerf summary is unavailable: {error}")
    record_files = [item for item in artifacts if Path(item.get("path", "")).name == "profile_export.jsonl"]
    if len(record_files) != 1:
        missing.append("one native AIPerf profile_export.jsonl file is required")
        measured_rows = []
    else:
        record_file = record_files[0]
        try:
            if _identity(Path(record_file["path"])) != record_file:
                failures.append("AIPerf per-request records changed after the producer run")
            measured_rows = [
                json.loads(line) for line in Path(record_file["path"]).read_text().splitlines() if line.strip()
            ]
        except (OSError, ValueError) as error:
            missing.append(f"AIPerf per-request records are unavailable: {error}")
            measured_rows = []
    predicted_rows = prediction.get("per_request", [])
    count = recipe["request_count"]
    if prediction.get("completed_requests") != count or len(predicted_rows) != count:
        missing.append("ordinary prediction did not complete every selected request")
    if len(measured_rows) != count:
        missing.append("serving did not produce exactly one record for every selected request")
    predicted_by_turn = {}
    for row in predicted_rows:
        match = re.search(r":request:outer:(\d+)$", str(row.get("request_id", "")))
        if match is None or int(match[1]) in predicted_by_turn:
            failures.append("prediction request identity is missing, repeated or outside single-stream Weka scope")
        else:
            predicted_by_turn[int(match[1])] = row
    play = json.loads(Path(recipe["benchmark_trace"]["path"]).read_text(encoding="utf-8"))
    tokenization = json.loads(Path(recipe["matched_workload"]["tokenization"]["path"]).read_text())
    seen = set()
    starts_ns, ends_ns, starts_ms, ends_ms = [], [], [], []
    for row in measured_rows:
        metadata = row.get("metadata", {})
        turn = metadata.get("turn_index")
        if (
            metadata.get("conversation_id") != recipe["play_id"]
            or type(turn) is not int
            or turn < 0
            or turn >= count
            or turn in seen
            or turn not in predicted_by_turn
        ):
            failures.append("AIPerf request identity is missing, repeated or does not match the selected play")
            continue
        seen.add(turn)
        simulated = predicted_by_turn[turn]
        if (
            row.get("error") is not None
            or metadata.get("was_cancelled") is not False
            or metadata.get("benchmark_phase") != "profiling"
            or simulated.get("terminal_status") != "completed"
        ):
            failures.append(f"request {turn} failed, was cancelled or includes non-profile traffic")
            continue
        try:
            isl, osl = _metric(row, "input_sequence_length", "tokens"), _metric(row, "output_sequence_length", "tokens")
            if (
                isl != tokenization["requests"][turn]["input_length"]
                or osl != play["requests"][turn]["out"]
                or simulated.get("input_length") != isl
                or simulated.get("output_length") != osl
                or simulated.get("requested_output_length") != osl
            ):
                failures.append(f"request {turn} input/output token counts do not match trace and prediction")
                continue
            measured_ttft = _metric(row, "time_to_first_token", "ms")
            predicted_ttft = _number(simulated.get("ttft_ms"), "simulated TTFT", positive=True)
            start_ns = metadata.get("request_start_ns")
            end_ns = metadata.get("request_end_ns")
            if type(start_ns) is not int or type(end_ns) is not int or start_ns < 0 or end_ns <= start_ns:
                raise ValueError("AIPerf request timestamps must be ordered integer nanoseconds")
            start_ms = _number(simulated.get("arrival_time_ms"), "simulated arrival")
            end_ms = _number(simulated.get("last_token_ms"), "simulated final token", positive=True)
            if end_ms <= start_ms:
                raise ValueError("simulated request timing interval is invalid")
            sample = {
                "turn_index": turn,
                "prediction_request_id": simulated["request_id"],
                "serving_request_id": metadata.get("x_request_id"),
                "input_tokens": int(isl),
                "source_trace_input_tokens": play["requests"][turn]["in"],
                "output_tokens": int(osl),
                "measured_ttft_ms": measured_ttft,
                "predicted_ttft_ms": predicted_ttft,
                "ttft_relative_error": _error(predicted_ttft, measured_ttft),
                "measured_tpot_ms": None,
                "predicted_tpot_ms": None,
                "tpot_relative_error": None,
                "request_start_ns": start_ns,
                "request_end_ns": end_ns,
                "simulated_arrival_ms": start_ms,
                "simulated_last_token_ms": end_ms,
            }
            if osl > 1:
                measured_tpot = _metric(row, "inter_token_latency", "ms")
                predicted_tpot = _number(simulated.get("itl_ms"), "simulated TPOT", positive=True)
                sample.update(
                    measured_tpot_ms=measured_tpot,
                    predicted_tpot_ms=predicted_tpot,
                    tpot_relative_error=_error(predicted_tpot, measured_tpot),
                )
            report["samples"].append(sample)
            starts_ns.append(start_ns)
            ends_ns.append(end_ns)
            starts_ms.append(start_ms)
            ends_ms.append(end_ms)
        except ValueError as error:
            missing.append(f"request {turn}: {error}")
    metrics = {
        name: _compare_samples(report["samples"], name, policy["latency_p95_relative_error"])
        for name in ("ttft", "tpot")
    }
    if len(report["samples"]) == count and count > 0:
        ordered_samples = sorted(report["samples"], key=lambda item: item["turn_index"])
        first_start_ns = ordered_samples[0]["request_start_ns"]
        source_start = play["requests"][0]["t"]
        for previous, current in pairwise(ordered_samples):
            source_previous = play["requests"][previous["turn_index"]]
            source_current = play["requests"][current["turn_index"]]
            expected_gap_ms = (source_current["t"] - source_previous["t"] - source_previous.get("api_time", 0)) * 1000
            not_before_ns = first_start_ns + round((source_current["t"] - source_start) * 1e9)
            ready_ns = max(not_before_ns, previous["request_end_ns"] + round(expected_gap_ms * 1e6))
            schedule_error_ms = (current["request_start_ns"] - ready_ns) / 1e6
            current["load_timing"] = {
                "not_before_ns": not_before_ns,
                "completion_gap_ms": expected_gap_ms,
                "expected_ready_ns": ready_ns,
                "dispatch_error_ms": schedule_error_ms,
            }
            if abs(schedule_error_ms) > max(
                policy["max_dispatch_delay_ms"], expected_gap_ms * policy["max_dispatch_delay_fraction"]
            ):
                failures.append(
                    f"serving dispatch at turn {current['turn_index']} differs from trace bounds; "
                    "check producer load policy or client scheduling lag"
                )
        tokens = sum(row["output_tokens"] for row in report["samples"])
        measured_rate = tokens / ((max(ends_ns) - min(starts_ns)) / 1e9)
        predicted_rate = tokens / ((max(ends_ms) - min(starts_ms)) / 1000)
        relative_error = _error(predicted_rate, measured_rate)
        limit = policy["throughput_relative_error"]
        metrics["output_throughput"] = {
            "status": "passed" if relative_error <= limit else "failed",
            "unit": "tokens/second",
            "measured": measured_rate,
            "predicted": predicted_rate,
            "relative_error": relative_error,
            "threshold": limit,
        }
    else:
        metrics["output_throughput"] = {"status": "incomplete", "issue": "complete matched requests are required"}
    report["metrics"] = metrics
    report["forward"] = _forward_validation(forward_evidence_path, recipe_path, policy["latency_p95_relative_error"])
    statuses = [metric["status"] for metric in metrics.values()]
    if failures or "failed" in statuses or report["forward"]["status"] == "failed":
        report["status"] = "failed"
    elif not missing and all(status == "passed" for status in statuses) and report["forward"]["status"] != "incomplete":
        report["status"] = "passed"
    _checked_recipe(recipe_path)
    if output_report is not None:
        _write(output_report, report)
    return report


def check_serving_validation_report(path: Path) -> dict[str, Any]:
    """Reassess raw evidence, including source hashes, before accepting a saved result."""
    report = json.loads(path.read_text(encoding="utf-8"))
    try:
        recipe = report["recipe"]
        if _identity(Path(recipe["path"])) != recipe:
            raise ValueError("serving recipe changed since assessment")
        for source in report["inputs"]:
            if _identity(Path(source["path"])) != source:
                raise ValueError("serving validation input changed since assessment")
        evidence = report.get("forward", {}).get("evidence")
        if evidence is not None and _identity(Path(evidence["path"])) != evidence:
            raise ValueError("serving forward evidence changed since assessment")
        reassessed = validate_serving_measurements(
            Path(recipe["path"]),
            prediction_report=Path(report["inputs"][0]["path"]),
            execution_evidence_path=Path(report["inputs"][1]["path"]),
            output_report=None,
            forward_evidence_path=Path(evidence["path"]) if evidence else None,
            latency_p95_error=report["policy"]["latency_p95_relative_error"],
            throughput_error=report["policy"]["throughput_relative_error"],
            max_dispatch_delay_ms=report["policy"]["max_dispatch_delay_ms"],
            max_dispatch_delay_fraction=report["policy"]["max_dispatch_delay_fraction"],
        )
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("serving validation report lacks required evidence references") from error
    if reassessed != report:
        raise ValueError("saved serving validation result differs from independent reassessment of its raw evidence")
    return report
