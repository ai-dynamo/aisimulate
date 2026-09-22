# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public onboarding commands with synthetic native records, never silicon claims."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from collector.fpm_forward import cli as collector_cli
from collector.fpm_forward import planner, repeatability
from collector.fpm_forward.config import FPMCollectionOptions
from collector.fpm_forward.database import aggregate_cell, write_formal_database

from aisimulate import main as cli
from aisimulate.support import serving_validation as serving
from aisimulate.support import validation_workflow as workflow
from aisimulate.support.fpm import fpm_cli_args
from aisimulate.support.plan import create_plan
from aisimulate.support.schema import SupportRequest

from .collector.test_fpm_repeatability import _write_campaign
from .test_support_serving_validation import materialized_workload, write_server_tokenization  # noqa: F401
from .test_support_validation import validation_case  # noqa: F401

pytestmark = pytest.mark.unit


def _write(path, value):
    path.write_text(json.dumps(value))


def _synthetic_campaign(root, checkpoint, plan, *, attempt_id, generator_overrides=None):
    _write_campaign(root, checkpoint, plan, attempt_id=attempt_id, generator_overrides=generator_overrides)
    for path in root.rglob("benchmark-dp*.json"):
        payload = json.loads(path.read_text())
        for row in payload["results"]:
            point = row["point"]
            if point["point_type"] == "prefill" and point["total_kv_read_tokens"] == 0:
                row["kv_seed_regime"] = "not_applicable"
                point["sample_reasons"] = []
        for group in payload["iteration_groups"]:
            point = group["point"]
            if point["point_type"] == "prefill" and point["total_kv_read_tokens"] == 0:
                point["sample_reasons"] = []
        _write(path, payload)


@pytest.fixture
def quality_case(validation_case, tmp_path, monkeypatch, materialized_workload):  # noqa: F811
    replay_args, request, root, trace, replay_output = validation_case
    payload = request.model_dump(mode="json")
    payload["identity"]["framework_version"] = "0.27.0"
    payload["fpm_profile"]["deployments"][0].update(backend_version="0.27.0", fmha_quant_mode="fp8")
    payload["collection"]["prefill_cudagraph_policy"] = "runtime"
    request = SupportRequest.model_validate(payload)
    old_root = root
    root = tmp_path / "native-collection"
    replay_args = [value.replace(str(old_root), str(root)) for value in replay_args]
    create_plan(request, root)
    model_config = tmp_path / "model-config.json"
    _write(
        model_config,
        {
            "architectures": [request.fpm_profile.architecture],
            "model_type": "llama",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 1024,
            "max_position_embeddings": 512,
            "torch_dtype": "bfloat16",
            "num_local_experts": request.fpm_profile.num_experts,
            "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic"},
        },
    )
    parsed = collector_cli._parser().parse_args(fpm_cli_args(request, output_dir=root, plan_only=True)[3:])
    options = FPMCollectionOptions.from_args(parsed)
    points = {
        "schema_version": 3,
        "prefill": [
            {"batch_size": batch, "total_prefill_tokens": tokens, "total_kv_read_tokens": batch * context}
            for batch in (1, 2)
            for tokens in (2, 16, 32, 64, 128)
            for context in (0, 16, 64, 128, 256)
        ],
        "decode": [
            {"batch_size": batch, "total_kv_read_tokens": batch * context}
            for batch in (1, 2)
            for context in (1, 4, 16, 64, 128, 256)
        ],
    }
    points_json = json.dumps(points, sort_keys=True, separators=(",", ":"))
    options = replace(
        options, benchmark_points_json=points_json, benchmark_points_sha256=planner._canonical_hash(points)
    )
    overrides = {"K8sConfig": {"k8s_image": "example/runtime@sha256:" + "a" * 64}}
    monkeypatch.setattr(planner, "_git_revision", lambda: "synthetic-workflow-revision")
    plan = planner.build_collection_plan(
        backend="vllm",
        model_path=request.identity.model,
        system=request.identity.gpu,
        selected_ops=set(),
        options=options,
        model_architecture=request.fpm_profile.architecture,
        model_config_path=str(model_config),
        fpm_profile=request.fpm_profile,
        generator_overrides=overrides,
    )
    campaign = root / "fpm-artifacts" / plan.sha256[:16]
    checkpoint = root / "fpm-checkpoint/fpm_forward.json"
    _synthetic_campaign(campaign, checkpoint, plan, attempt_id="synthetic-source", generator_overrides=overrides)
    rows = []
    for cell in plan.cells:
        rows.extend(
            aggregate_cell(plan, cell, campaign / "cells" / cell.cell_id, expected_attempt_id="synthetic-source")
        )
    # Replace the other fixture's explicit synthetic table with this native publication.
    for path in (root / "systems/data").rglob("fpm_forward_perf.*"):
        path.unlink()
    parquet, metadata, skipped = write_formal_database(plan, rows, systems_root=root / "systems/data")
    saved = json.loads(checkpoint.read_text())
    saved["database"] = {
        "status": "passed",
        "missing_cells": [],
        "skipped_first_publisher_wins": list(skipped),
        "plan_cells": len(plan.cells),
        "published_cells": len(plan.cells),
        "parquet": str(parquet),
        "metadata": str(metadata),
    }
    _write(checkpoint, saved)
    calls = []

    def synthetic_repeat(sample_plan, **kwargs):
        calls.append(kwargs)
        raw = Path(kwargs["artifact_root"]) / sample_plan.sha256[:16]
        cp = Path(kwargs["checkpoint_dir"]) / "fpm_forward.json"
        _synthetic_campaign(
            raw,
            cp,
            sample_plan,
            attempt_id=f"synthetic-repeat-{len(calls)}",
            generator_overrides=kwargs["generator_overrides"],
        )
        return 0

    monkeypatch.setattr(repeatability, "run_collection", synthetic_repeat)
    output = tmp_path / "collection-quality"
    args = [
        "onboard",
        "validate-collection",
        "--config",
        str(root / "request.yaml"),
        "--output-dir",
        str(root),
        "--validation-output-dir",
        str(output),
    ]
    return {
        "request": request,
        "root": root,
        "plan": plan,
        "campaign": campaign,
        "checkpoint": checkpoint,
        "args": args,
        "output": output,
        "calls": calls,
        "replay_args": replay_args,
        "replay_output": replay_output,
        "trace": trace,
    }


def test_public_prepare_then_repeat_and_holdout_preserve_originals(quality_case):
    case = quality_case
    originals = {p: p.read_bytes() for p in case["root"].rglob("*") if p.is_file()}
    assert cli.main(case["args"]) == 0
    report = json.loads((case["output"] / "collection-validation.json").read_text())
    assert report["status"] == "incomplete"
    assert report["gates"]["interpolation"]["status"] == "passed"
    assert not case["calls"]
    assert cli.main([*case["args"], "--resume", "--execute"]) == 0
    checked, _, _, _ = workflow.check_collection_report(
        case["output"] / "collection-validation.json", workflow.ValidationPolicy()
    )
    assert checked["status"] == "passed"
    assert len(case["calls"]) == 10
    assert originals == {p: p.read_bytes() for p in originals}


def test_changed_threshold_reuses_raw_samples_but_changed_count_does_not(quality_case, tmp_path):
    case = quality_case
    assert cli.main([*case["args"], "--execute"]) == 0
    repeats = case["output"] / "repeatability"
    saved = {p: p.read_bytes() for p in repeats.rglob("*") if p.is_file()}
    policy = tmp_path / "policy.json"
    _write(policy, {"repeatability": {"max_cv": 0.10}, "interpolation": {"max_p95_relative_error": 0.1}})
    args = [
        *case["args"][:-1],
        str(tmp_path / "reassessed"),
        "--repeatability-dir",
        str(repeats),
        "--policy",
        str(policy),
    ]
    assert cli.main(args) == 0
    assert len(case["calls"]) == 10
    assert saved == {p: p.read_bytes() for p in saved}
    _write(policy, {"repeatability": {"samples": 6}})
    args[args.index("--validation-output-dir") + 1] = str(tmp_path / "wrong-count")
    with pytest.raises(SystemExit, match="2"):
        cli.main(args)
    assert len(case["calls"]) == 10


@pytest.mark.parametrize("condition", ["missing", "failed", "failed_and_missing"])
def test_source_execution_status_survives_report_recheck(quality_case, condition):
    case = quality_case
    paths = sorted(case["campaign"].rglob("fpm-execution-worker-*.json"))
    if condition in {"missing", "failed_and_missing"}:
        paths.pop().unlink()
    if condition in {"failed", "failed_and_missing"}:
        observed = json.loads(paths[0].read_text())
        observed["resolved_config"]["model_config"]["dtype"] = "torch.float32"
        _write(paths[0], observed)
    expected = "incomplete" if condition == "missing" else "failed"
    assert cli.main(case["args"]) == (0 if expected == "incomplete" else 1)
    report_path = case["output"] / "collection-validation.json"
    report, _, _, _ = workflow.check_collection_report(report_path, workflow.ValidationPolicy())
    assert report["status"] == report["gates"]["execution"]["status"] == expected
    cells = report["gates"]["execution"]["cells"]
    assert any(cell["missing_evidence" if expected == "incomplete" else "failures"] for cell in cells)
    report["gates"]["execution"]["status"] = "passed"
    _write(report_path, report)
    with pytest.raises(ValueError, match="execution gate differs"):
        workflow.check_collection_report(report_path, workflow.ValidationPolicy())


@pytest.mark.parametrize("relation", ["ancestor", "same", "descendant", "symlink_descendant"])
def test_collection_reassessment_preserves_reused_attempt_tree(quality_case, tmp_path, relation):
    case = quality_case
    assert cli.main([*case["args"], "--execute"]) == 0
    repeats = case["output"] / "repeatability"
    attempt = next((repeats / "samples").glob("*/sample-*/attempt-*"))
    output = {"ancestor": repeats.parent, "same": repeats, "descendant": attempt / "assessment"}.get(relation)
    if relation == "symlink_descendant":
        link = tmp_path / "repeat-alias"
        link.symlink_to(attempt, target_is_directory=True)
        output = link / "assessment"
    before = {p: p.read_bytes() for p in case["output"].rglob("*") if p.is_file()}
    args = [*case["args"]]
    args[args.index("--validation-output-dir") + 1] = str(output)
    with pytest.raises(SystemExit, match="2"):
        cli.main([*args, "--repeatability-dir", str(repeats)])
    assert before == {p: p.read_bytes() for p in case["output"].rglob("*") if p.is_file()}
    checked, _, _, _ = workflow.check_collection_report(
        case["output"] / "collection-validation.json", workflow.ValidationPolicy()
    )
    assert checked["status"] == "passed"


@pytest.mark.parametrize("which", ["status", "formal", "policy", "sample"])
def test_requalification_rejects_changed_or_forged_evidence(quality_case, which):
    case = quality_case
    assert cli.main([*case["args"], "--execute"]) == 0
    report_path = case["output"] / "collection-validation.json"
    report = json.loads(report_path.read_text())
    if which == "status":
        report["gates"]["interpolation"]["status"] = "failed"
        _write(report_path, report)
    elif which == "policy":
        _write(case["output"] / "policy.json", {"repeatability": {"max_cv": 1.0}})
    elif which == "formal":
        Path(report["inputs"]["formal_data"][0]["path"]).write_bytes(b"changed")
    else:
        next((case["output"] / "repeatability").rglob("benchmark-dp0.json")).write_text("{}")
    with pytest.raises((ValueError, OSError)):
        workflow.check_collection_report(report_path, workflow.ValidationPolicy())


@pytest.mark.parametrize("policy_format", ["json", "yaml"])
@pytest.mark.parametrize(
    "values",
    [
        {"repeatability": {"samples": True}},
        {"serving": {"max_latency_p95_relative_error": float("nan")}},
        {"serving": {"max_latency_p95_relative_error": -0.1}},
        {"serving": {"max_latency_p95_relative_error": "1e-05"}},
        {"serving": {"max_throughput_relative_error": True}},
        {"serving": {"max_throughput_relative_error": float("inf")}},
        {"serving": {"max_throughput_relative_error": float("-inf")}},
        {"interpolation": {"max_unsupported_fraction": 1.1}},
        {"typo": 1},
        [],
    ],
)
def test_validation_policy_rejects_ambiguous_settings(values, tmp_path, policy_format):
    policy = tmp_path / f"policy.{policy_format}"
    policy.write_text(json.dumps(values) if policy_format == "json" else yaml.safe_dump(values))
    with pytest.raises(ValueError):
        workflow._policy(policy)


def _prepare_matched(case, tmp_path, *, policy=None):
    assert cli.main([*case["args"], "--execute"]) == 0
    assert cli.main(case["replay_args"]) == 0
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    _write(tokenizer / "tokenizer_config.json", {"synthetic": True})
    prepared = tmp_path / "prepared"
    args = [
        "onboard",
        "validate-serving",
        "--collection-report",
        str(case["output"] / "collection-validation.json"),
        "--replay-report",
        str(case["replay_output"] / "validation.json"),
    ]
    if policy is not None:
        args.extend(["--policy", str(policy)])
    assert (
        cli.main(
            [
                *args,
                "--action",
                "prepare",
                "--validation-output-dir",
                str(prepared),
                "--endpoint",
                "http://127.0.0.1:8000",
                "--tokenizer",
                str(tokenizer),
                "--aiperf-python",
                "/synthetic/pinned-aiperf/python",
            ]
        )
        == 0
    )
    recipe_path = prepared / "serving/recipe.json"
    recipe = json.loads(recipe_path.read_text())
    return args, recipe_path, recipe, tokenizer


def _synthetic_serving(recipe_path, recipe, tokenizer):
    """AIPerf-shaped fixtures generated from native prediction; no HTTP/GPU execution."""
    prediction_ref = recipe.get("matched_workload", {}).get("prediction_report", recipe["prediction_report"])
    prediction = json.loads(Path(prediction_ref["path"]).read_text())
    artifacts = Path(recipe["artifacts_dir"])
    artifacts.mkdir()
    rows = []
    for index, row in enumerate(prediction["per_request"]):
        rows.append(
            {
                "metadata": {
                    "conversation_id": recipe["play_id"],
                    "turn_index": index,
                    "x_request_id": f"synthetic-{index}",
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
    _write(summary, {"was_cancelled": False, "is_complete": True, "request_count": {"avg": len(rows)}})
    runtime_dir = Path(recipe["serving_observer"]["output_dir"])
    runtime_dir.mkdir()
    provenance = Path(recipe["serving_observer"]["provenance_path"])
    expected = recipe["expected_execution"]
    native_runtime = json.loads(json.dumps(expected["runtime_config"]))
    native_runtime.setdefault("cache_config", {})["enable_prefix_caching"] = recipe["scope"]["prefix_caching"]
    runtime_sources = []
    for dp in range(expected["parallelism"]["attention_data"]):
        for tp in range(expected["parallelism"]["tensor"]):
            path = runtime_dir / f"fpm-execution-worker-dp{dp}-tp{tp}-pp0.json"
            _write(
                path,
                {
                    "schema_name": "aisimulate_fpm_runtime_execution",
                    "schema_version": 1,
                    "status": "observed",
                    "execution_provenance": json.loads(provenance.read_text()),
                    "provenance_source": workflow._identity(provenance),
                    "dp_rank": dp,
                    "tp_rank": tp,
                    "pp_rank": 0,
                    "resolved_config": native_runtime,
                    **{key: expected[key] for key in ("attention_groups", "graph_config", "backend_version")},
                },
            )
            runtime_sources.append(workflow._identity(path))
    _write(
        recipe_path.parent / "benchmark-execution.json",
        {
            "schema_version": serving.SCHEMA,
            "recipe": workflow._identity(recipe_path),
            "status": "completed",
            "exit_code": 0,
            "command": ["python", *recipe["command_arguments"]],
            "installation": {"source": {"vcs_info": {"commit_id": serving.AIPERF_REVISION}}},
            "artifacts": [workflow._identity(records), workflow._identity(summary)],
            "runtime_observations": runtime_sources,
            "serving_provenance": workflow._identity(provenance),
            "observer_modules": recipe["serving_observer"]["module_files"],
            "server_tokenization": workflow._identity(write_server_tokenization(recipe_path, recipe)),
        },
    )
    observed = recipe_path.parent / "observed-execution.json"
    _write(
        observed,
        {
            "schema_version": serving.SCHEMA,
            "recipe": workflow._identity(recipe_path),
            "observed_execution": expected,
            "scope": recipe["scope"],
            "sources": [workflow._identity(tokenizer / "tokenizer_config.json")],
        },
    )
    return observed


@pytest.mark.parametrize("validation_case", ["tp", "dep", "tep"], indirect=True)
def test_public_four_steps_qualify_only_matched_synthetic_scope(quality_case, tmp_path):
    args, recipe_path, recipe, tokenizer = _prepare_matched(quality_case, tmp_path)
    observed = _synthetic_serving(recipe_path, recipe, tokenizer)
    raw = {p: p.read_bytes() for p in recipe_path.parent.rglob("*") if p.is_file()}
    output = tmp_path / "qualified"
    assert (
        cli.main(
            [
                *args,
                "--action",
                "assess",
                "--recipe",
                str(recipe_path),
                "--execution-evidence",
                str(observed),
                "--validation-output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "validation.json").read_text())
    assert report["status"] == "passed"
    assert report["accuracy"] == "qualified_for_evaluated_scope"
    assert report["gates"]["serving"]["forward"]["status"] == "unavailable"
    assert set(report["gates"]["serving"]["metrics"]) == {"ttft", "tpot", "output_throughput"}
    assert raw == {p: p.read_bytes() for p in raw}


@pytest.mark.parametrize("exit_code", [0, 7])
def test_serving_run_returns_producer_status_and_preserves_logs(quality_case, tmp_path, monkeypatch, capsys, exit_code):
    _, recipe_path, recipe, tokenizer = _prepare_matched(quality_case, tmp_path)
    _synthetic_serving(recipe_path, recipe, tokenizer)
    receipt_path = recipe_path.parent / "benchmark-execution.json"
    receipt_path.unlink()
    shutil.rmtree(recipe["artifacts_dir"])
    producer = tmp_path / "synthetic-aiperf-python"
    installation = {
        "version": "synthetic-process-boundary",
        "source": {"url": serving.AIPERF_SOURCE, "vcs_info": {"commit_id": serving.AIPERF_REVISION}},
    }
    producer.write_text(
        "#!/usr/bin/env python3\nimport sys\nif '-c' in sys.argv:\n"
        f" print({json.dumps(installation)!r})\nelse:\n"
        f" print('producer stdout')\n print('producer stderr', file=sys.stderr)\n sys.exit({exit_code})\n"
    )
    producer.chmod(0o755)
    monkeypatch.setattr(serving, "_tokenize_serving", write_server_tokenization)
    result = cli.main(
        [
            "onboard",
            "validate-serving",
            "--action",
            "run",
            "--recipe",
            str(recipe_path),
            "--aiperf-python",
            str(producer),
        ]
    )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["exit_code"] == exit_code
    assert receipt["status"] == ("completed" if exit_code == 0 else "failed")
    assert result == (0 if exit_code == 0 else 1)
    stdout = capsys.readouterr().out
    assert str(receipt_path) in stdout
    for name, content in (("stdout", "producer stdout"), ("stderr", "producer stderr")):
        path = recipe_path.parent / f"aiperf.{name}.log"
        assert path.read_text().strip() == content
        assert str(path) in stdout


@pytest.mark.parametrize("missing", ["receipt", "execution"])
def test_missing_serving_evidence_retains_incomplete_diagnostic(quality_case, tmp_path, missing):
    args, recipe_path, recipe, tokenizer = _prepare_matched(quality_case, tmp_path)
    observed = _synthetic_serving(recipe_path, recipe, tokenizer)
    missing_path = recipe_path.parent / "benchmark-execution.json" if missing == "receipt" else observed
    missing_path.unlink()
    output = tmp_path / "missing-assessment"
    assert (
        cli.main(
            [
                *args,
                "--action",
                "assess",
                "--recipe",
                str(recipe_path),
                "--execution-evidence",
                str(observed),
                "--validation-output-dir",
                str(output),
            ]
        )
        == 1
    )
    combined = json.loads((output / "validation.json").read_text())
    assessed = json.loads((output / "serving-validation.json").read_text())
    gate = combined["gates"]["serving"]
    assert combined["status"] == gate["status"] == assessed["status"] == "incomplete"
    assert gate["missing_evidence"] == assessed["missing_evidence"]
    assert any(str(missing_path) in issue for issue in gate["missing_evidence"])
    assert gate["failures"] == []


@pytest.mark.parametrize("action", ["prepare", "assess"])
@pytest.mark.parametrize("relation", ["ancestor", "same", "descendant"])
@pytest.mark.parametrize(
    "source_kind", ["original_raw", "reused_repeat", "collection_report", "replay_report", "recipe", "tokenizer"]
)
def test_serving_rejects_nested_outputs_before_writing(quality_case, tmp_path, action, source_kind, relation):
    case = quality_case
    args, recipe_path, recipe, tokenizer = _prepare_matched(case, tmp_path)
    observed = _synthetic_serving(recipe_path, recipe, tokenizer)
    repeats = case["output"] / "repeatability"
    reassessed = tmp_path / "reassessed-collection"
    collection_args = [*case["args"]]
    collection_args[collection_args.index("--validation-output-dir") + 1] = str(reassessed)
    assert cli.main([*collection_args, "--repeatability-dir", str(repeats)]) == 0
    report_path = reassessed / "collection-validation.json"
    args[args.index("--collection-report") + 1] = str(report_path)
    source = {
        "original_raw": next((case["campaign"] / "cells").glob("*/raw")),
        "reused_repeat": next((repeats / "samples").glob("*/sample-*/attempt-*")),
        "collection_report": report_path.parent,
        "replay_report": case["replay_output"],
        "recipe": recipe_path.parent,
        "tokenizer": tokenizer,
    }[source_kind]
    output = {"ancestor": source.parent, "same": source, "descendant": source / "must-not-exist"}[relation]
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                *args,
                "--action",
                action,
                "--recipe",
                str(recipe_path),
                "--execution-evidence",
                str(observed),
                "--validation-output-dir",
                str(output),
                "--endpoint",
                "http://localhost:8000",
                "--tokenizer",
                str(tokenizer),
                "--aiperf-python",
                "/not/executed",
            ]
        )
    if relation == "descendant":
        assert not output.exists()
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    checked, original, plan, _ = workflow.check_collection_report(report_path, workflow.ValidationPolicy())
    assert checked["status"] == "passed"
    assert (
        workflow._check_replay(case["replay_output"] / "validation.json", original, checked, plan)["status"] == "passed"
    )
    assert serving._checked_recipe(recipe_path)["status"] == "ready"


@pytest.mark.parametrize(
    ("changed", "expected_status"),
    [
        ("missing_collection", "incomplete"),
        ("prediction", "failed"),
        ("coverage", "failed"),
        ("legacy_replay", "incomplete"),
        ("missing_serving", "incomplete"),
    ],
)
def test_combined_gate_rejects_missing_and_stale_evidence(quality_case, tmp_path, changed, expected_status):
    args, recipe_path, recipe, tokenizer = _prepare_matched(quality_case, tmp_path)
    observed = _synthetic_serving(recipe_path, recipe, tokenizer)
    if changed == "missing_collection":
        Path(args[args.index("--collection-report") + 1]).unlink()
    elif changed in {"prediction", "coverage"}:
        path = (
            quality_case["replay_output"]
            / "prediction"
            / ("prediction.json" if changed == "prediction" else "fpm-coverage.json")
        )
        path.write_text(path.read_text() + "\n")
    elif changed == "legacy_replay":
        path = quality_case["replay_output"] / "validation.json"
        payload = json.loads(path.read_text())
        del payload["prediction_evidence"]
        _write(path, payload)
    else:
        (recipe_path.parent / "benchmark-execution.json").unlink()
    output = tmp_path / "rejected"
    assert (
        cli.main(
            [
                *args,
                "--action",
                "assess",
                "--recipe",
                str(recipe_path),
                "--execution-evidence",
                str(observed),
                "--validation-output-dir",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads((output / "validation.json").read_text())
    assert report["status"] == expected_status
    assert report["accuracy"] == "not_qualified"
    if changed != "missing_collection":
        assert report["gates"]["collection"]["status"] == "passed"


@pytest.mark.parametrize("policy_format", ["json", "yaml"])
@pytest.mark.parametrize("limit", [0.0, 0.00001, 1.1])
def test_serving_error_limits_survive_preparation_assessment_and_recheck(quality_case, tmp_path, limit, policy_format):
    policy = tmp_path / f"policy.{policy_format}"
    values = {"serving": {"max_latency_p95_relative_error": limit, "max_throughput_relative_error": limit}}
    policy.write_text(json.dumps(values) if policy_format == "json" else yaml.safe_dump(values))
    args, recipe_path, recipe, tokenizer = _prepare_matched(quality_case, tmp_path, policy=policy)
    assert recipe["policy"]["latency_p95_relative_error"] == limit
    assert recipe["policy"]["throughput_relative_error"] == limit
    observed = _synthetic_serving(recipe_path, recipe, tokenizer)
    raw = {path: path.read_bytes() for path in recipe_path.parent.rglob("*") if path.is_file()}
    output = tmp_path / "assessed"
    assert (
        cli.main(
            [
                *args,
                "--action",
                "assess",
                "--recipe",
                str(recipe_path),
                "--execution-evidence",
                str(observed),
                "--validation-output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = serving.check_serving_validation_report(output / "serving-validation.json")
    assert report["status"] == "passed"
    assert report["policy"] == recipe["policy"]
    assert report["metrics"]["ttft"]["p95_relative_error"] == 0
    assert report["metrics"]["tpot"]["p95_relative_error"] == 0
    assert report["metrics"]["output_throughput"]["relative_error"] == 0
    assert raw == {path: path.read_bytes() for path in raw}


@pytest.mark.parametrize(
    "duration_scale,strict_limit,relaxed_limit,failed_metric",
    [
        (1.5, 0.2, 0.5, "ttft"),
        (1.000001, 0.0, 0.00001, "ttft"),
        (0.48, 1.0, 1.1, "ttft"),
        (3.4, 1.0, 1.1, "output_throughput"),
    ],
)
def test_serving_threshold_reassessment_preserves_measurements(
    quality_case, tmp_path, duration_scale, strict_limit, relaxed_limit, failed_metric
):
    args, recipe_path, recipe, tokenizer = _prepare_matched(quality_case, tmp_path)
    observed = _synthetic_serving(recipe_path, recipe, tokenizer)
    records = Path(recipe["artifacts_dir"]) / "profile_export.jsonl"
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    previous = None
    for row in rows:
        metadata = row["metadata"]
        duration = metadata["request_end_ns"] - metadata["request_start_ns"]
        if previous is not None:
            metadata["request_start_ns"] = previous + 100_000_000
        metadata["request_end_ns"] = metadata["request_start_ns"] + round(duration * duration_scale)
        previous = metadata["request_end_ns"]
        for key in ("time_to_first_token", "inter_token_latency"):
            row["metrics"][key]["value"] *= duration_scale
    records.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    receipt_path = recipe_path.parent / "benchmark-execution.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"] = [workflow._identity(Path(ref["path"])) for ref in receipt["artifacts"]]
    _write(receipt_path, receipt)
    raw = {path: path.read_bytes() for path in recipe_path.parent.rglob("*") if path.is_file()}
    assess = [*args, "--action", "assess", "--recipe", str(recipe_path), "--execution-evidence", str(observed)]
    strict_policy = tmp_path / "strict.json"
    _write(
        strict_policy,
        {"serving": {"max_latency_p95_relative_error": strict_limit, "max_throughput_relative_error": strict_limit}},
    )
    assert cli.main([*assess, "--validation-output-dir", str(tmp_path / "strict"), "--policy", str(strict_policy)]) == 1
    report = json.loads((tmp_path / "strict/validation.json").read_text())
    assert report["gates"]["serving"]["metrics"][failed_metric]["status"] == "failed"
    assert serving.check_serving_validation_report(tmp_path / "strict/serving-validation.json")["status"] == "failed"
    policy = tmp_path / "relaxed.json"
    _write(
        policy,
        {"serving": {"max_latency_p95_relative_error": relaxed_limit, "max_throughput_relative_error": relaxed_limit}},
    )
    assert cli.main([*assess, "--validation-output-dir", str(tmp_path / "relaxed"), "--policy", str(policy)]) == 0
    assert serving.check_serving_validation_report(tmp_path / "relaxed/serving-validation.json")["status"] == "passed"
    assert raw == {path: path.read_bytes() for path in raw}


def test_replay_accepts_verified_memory_finalization_and_rejects_changed_capacity(tmp_path, monkeypatch):
    from collector.fpm_forward.repeatability import load_repeatability_source

    from aisimulate import supervision
    from aisimulate.support.finalization import finalize

    from .test_onboard_finalization import build_completed_collection

    original, root = build_completed_collection(tmp_path)
    target = tmp_path / "resolved"
    finalize(original, root, target)
    request = SupportRequest.from_yaml(target / "request.yaml")
    trace = tmp_path / "trace.json"
    _write(
        trace,
        {
            "id": "synthetic-memory",
            "models": ["source"],
            "block_size": 16,
            "hash_id_scope": "local",
            "requests": [{"t": 0, "type": "s", "model": "source", "in": 16, "out": 2, "hash_ids": [1]}],
        },
    )
    monkeypatch.setattr(supervision, "main", cli.main)
    replay = tmp_path / "replay-resolved"
    args = [
        "onboard",
        "validate-fpm",
        "--config",
        str(target / "request.yaml"),
        "--output-dir",
        str(target),
        "--trace",
        str(trace),
        "--validation-output-dir",
        str(replay),
    ]
    assert cli.main(args) == 0
    plan = load_repeatability_source(next((root / "fpm-artifacts").iterdir()))
    collection = {
        "inputs": {
            "collection_directory": str(root),
            "formal_data": [
                workflow._identity(path) for path in sorted((root / "systems/data").rglob("fpm_forward_perf.*"))
            ],
        }
    }
    assert workflow._check_replay(replay / "validation.json", original, collection, plan)["status"] == "passed"

    changed = request.model_dump(mode="json")
    changed["fpm_profile"]["deployments"][0]["resources"]["runtime_memory"]["kv_cache_bytes"] += 1024
    altered = SupportRequest.model_validate(changed)
    bad_root = tmp_path / "altered-resolved"
    create_plan(altered, bad_root)
    for source in (target / "systems/data").rglob("*"):
        if source.is_file():
            destination = bad_root / source.relative_to(target)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    args = [
        str(value).replace(str(target), str(bad_root)).replace(str(replay), str(tmp_path / "altered-replay"))
        for value in args
    ]
    assert cli.main(args) == 0
    with pytest.raises(ValueError, match="resources differ"):
        workflow._check_replay(tmp_path / "altered-replay/validation.json", original, collection, plan)
