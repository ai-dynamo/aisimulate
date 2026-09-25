# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-bound, editable collection and serving qualification for onboarding.

Validation files are separate from the collection plan. This module composes the
collector and native evaluators; it does not select another timing estimator or
change the runtime's collection grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from aisimulate.config.common import StrictModel, load_yaml

from .plan import check_plan, request_id
from .schema import SupportRequest

Ratio = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class RepeatabilityPolicy(StrictModel):
    samples: int = Field(default=5, strict=True, ge=2)
    max_points_per_cell: int = Field(default=12, strict=True, ge=1)
    max_cv: Ratio = 0.05


class InterpolationPolicy(StrictModel):
    max_points_per_phase: int = Field(default=16, strict=True, ge=1)
    seed: int = Field(default=42, strict=True, ge=0)
    max_p95_relative_error: Ratio = 0.20
    max_unsupported_fraction: float = Field(default=0.0, strict=True, ge=0, le=1, allow_inf_nan=False)


class ServingPolicy(StrictModel):
    max_latency_p95_relative_error: Ratio = 0.20
    max_throughput_relative_error: Ratio = 0.20
    max_dispatch_delay_ms: Ratio = 5.0
    max_dispatch_delay_fraction: Ratio = 0.01


class ValidationPolicy(StrictModel):
    schema_version: Literal["aisimulate-onboarding-validation-policy/v1"] = "aisimulate-onboarding-validation-policy/v1"
    repeatability: RepeatabilityPolicy = Field(default_factory=RepeatabilityPolicy)
    interpolation: InterpolationPolicy = Field(default_factory=InterpolationPolicy)
    serving: ServingPolicy = Field(default_factory=ServingPolicy)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _checked(ref: dict[str, Any]) -> Path:
    path = Path(ref["path"])
    if _identity(path) != ref:
        raise ValueError(f"validation evidence changed: {path}")
    return path


def _policy(path: str | Path | None) -> ValidationPolicy:
    if not path:
        return ValidationPolicy()
    try:
        # YAML 1.1 treats JSON exponent notation such as 1e-05 as a string.
        values = _json(Path(path))
    except (OSError, json.JSONDecodeError):
        values = load_yaml(path)
    return ValidationPolicy.model_validate(values)


def _fresh(output: Path, *inputs: Path) -> None:
    if output.is_symlink():
        raise ValueError("validation output must not be a symbolic link")
    output = output.resolve()
    for source in inputs:
        source = source.resolve()
        if output == source or output in source.parents or (source.is_dir() and source in output.parents):
            raise ValueError("validation output must be separate from all source artifacts")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("validation output must be fresh; resume the same policy or choose a new directory")
    output.mkdir(parents=True, exist_ok=True)


def _referenced_paths(value: Any) -> list[Path]:
    """Collect artifact identities and preserved directories before creating outputs."""
    paths = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "path",
                "collection_directory",
                "directory",
                "tokenizer",
                "artifacts_dir",
                "output_dir",
            } and isinstance(item, str):
                paths.append(Path(item))
            else:
                paths.extend(_referenced_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_referenced_paths(item))
    return paths


def _status(gates: dict[str, dict[str, Any]]) -> str:
    statuses = [item["status"] for item in gates.values()]
    return "failed" if "failed" in statuses else "passed" if all(x == "passed" for x in statuses) else "incomplete"


def _execution_status(frozen: dict[str, Any]) -> str:
    return _status(
        {
            cell["cell_id"]: {
                "status": "passed" if cell["execution"]["status"] == "qualified" else cell["execution"]["status"]
            }
            for cell in frozen["cells"]
        }
    )


def _collection_source(request: SupportRequest, root: Path) -> tuple[Any, Path, Path, list[dict[str, Any]]]:
    """Verify formal publication against every native row before qualification."""
    import pyarrow.parquet as pq
    from collector.fpm_forward.config import PrefillSamplingProfile
    from collector.fpm_forward.database import aggregate_cell, validate_formal_database_commit
    from collector.fpm_forward.planner import backend_identity_columns
    from collector.fpm_forward.repeatability import load_repeatability_source

    check_plan(request, root)
    checkpoints = list((root / "fpm-checkpoint").rglob("fpm_forward.json"))
    if len(checkpoints) != 1:
        raise ValueError("collection validation requires exactly one formal collector checkpoint")
    checkpoint_path = checkpoints[0]
    checkpoint = _json(checkpoint_path)
    sha = checkpoint.get("plan_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("invalid source collection plan identity")
    campaign = root / "fpm-artifacts" / sha[:16]
    plan = load_repeatability_source(campaign)
    payload = plan.to_dict()
    if (
        plan.sha256 != sha
        or plan.model_path != request.identity.model
        or plan.system != request.identity.gpu
        or plan.backend != request.identity.framework
        or plan.capability.aic_database_version != request.identity.framework_version
        or request.fpm_profile is None
        or payload.get("fpm_profile") != request.fpm_profile.model_dump(mode="json")
    ):
        raise ValueError("source collection differs from the original onboarding identity/profile")
    scheduler = request.scheduler_limits()
    sampling = PrefillSamplingProfile.build(
        max_isl=scheduler["max_batched_tokens"],
        max_batch_size=scheduler["max_sequences"],
        max_cudagraph_capture_size=request.collection.max_prefill_cudagraph_size,
        cudagraph_policy=request.collection.prefill_cudagraph_policy,
    )
    if (
        plan.options.vllm_max_model_len != request.search.context_length
        or plan.options.max_num_batched_tokens != scheduler["max_batched_tokens"]
        or plan.options.max_num_seqs != scheduler["max_sequences"]
        or (plan.options.gpu_memory_utilization or request.collection.memory_fraction)
        != request.collection.memory_fraction
        or plan.options.prefill_cudagraph_policy != request.collection.prefill_cudagraph_policy
        or plan.options.prefill_sampling != sampling
    ):
        raise ValueError("source collection runtime limits differ from the reviewed onboarding settings")
    deployment = request.profile_deployment()
    if {cell.workload_kind for cell in plan.cells} != {"prefill", "decode"}:
        raise ValueError("collection validation requires complete prefill and decode cells")
    database = checkpoint.get("database", {})
    if (
        database.get("status") != "passed"
        or database.get("missing_cells") != []
        or database.get("skipped_first_publisher_wins") != []
        or database.get("plan_cells") != len(plan.cells)
        or database.get("published_cells") != len(plan.cells)
    ):
        raise ValueError("collection validation requires complete formal publication without reused cells")
    parquet, metadata = (Path(database[name]).resolve() for name in ("parquet", "metadata"))
    if any(not path.is_relative_to(root / "systems/data") for path in (parquet, metadata)):
        raise ValueError("formal data must belong to this source campaign")
    validate_formal_database_commit(parquet, metadata, plan)
    rows = []
    for cell in plan.cells:
        for field in ("tp", "pp", "dp", "cp", "moe_tp", "moe_ep"):
            if getattr(cell.topology, field) != getattr(deployment, field):
                raise ValueError("collection topology differs from onboarding")
        for field in ("gemm_quant_mode", "moe_quant_mode", "fmha_quant_mode", "comm_quant_mode", "kv_cache_dtype"):
            if getattr(cell, field) != getattr(deployment, field):
                raise ValueError("collection precision differs from onboarding")
        if any(
            getattr(deployment, name) != value for name, value in backend_identity_columns(cell.backend_policy).items()
        ):
            raise ValueError("collection backend identity differs from onboarding")
        entry = checkpoint.get("cells", {}).get(cell.cell_id, {})
        if entry.get("status") != "passed" or not entry.get("attempt_id"):
            raise ValueError(f"source cell is not complete: {cell.cell_id}")
        rows.extend(
            aggregate_cell(plan, cell, campaign / "cells" / cell.cell_id, expected_attempt_id=entry["attempt_id"])
        )
    canonical = lambda row: json.dumps(row, sort_keys=True, allow_nan=False)
    if sorted(map(canonical, rows)) != sorted(map(canonical, pq.read_table(parquet).to_pylist())):
        raise ValueError("published formal rows differ from verified native collection")
    return plan, campaign, checkpoint_path, [_identity(parquet), _identity(metadata)]


def _holdout(request: SupportRequest, root: Path, output: Path, policy: ValidationPolicy) -> dict[str, Any]:
    from .interpolation_validation import evaluate_interpolation_holdout

    return evaluate_interpolation_holdout(
        request, systems_root=root / "systems", output_dir=output, **policy.interpolation.model_dump()
    )


def _repeat_plan(plan: Any, campaign: Path, checkpoint: Path, policy: ValidationPolicy) -> dict[str, Any]:
    from collector.fpm_forward.repeatability import freeze_repeatability_plan

    return freeze_repeatability_plan(
        plan,
        campaign,
        checkpoint,
        samples=policy.repeatability.samples,
        max_points_per_cell=policy.repeatability.max_points_per_cell,
        cv_threshold=policy.repeatability.max_cv,
    )


def _assess_repeats(directory: Path, policy: ValidationPolicy, expected_plan: dict[str, Any]) -> dict[str, Any]:
    from collector.fpm_forward.repeatability import PLAN_FILENAME, REPORT_FILENAME, assess_repeatability

    frozen = _json(directory / PLAN_FILENAME)
    if (
        {key: value for key, value in frozen.items() if key not in {"policy", "sha256"}}
        != {key: value for key, value in expected_plan.items() if key not in {"policy", "sha256"}}
        or frozen["policy"]["samples"] != policy.repeatability.samples
        or frozen["policy"]["max_points_per_cell"] != policy.repeatability.max_points_per_cell
    ):
        raise ValueError("repeatability measurements do not match the source or selected sample policy")
    return assess_repeatability(frozen, _json(directory / REPORT_FILENAME), cv_threshold=policy.repeatability.max_cv)


def validate_collection(args: argparse.Namespace) -> int:
    from collector.fpm_forward.repeatability import load_repeatability_deployment, run_repeatability

    request = SupportRequest.from_yaml(args.config)
    root = Path(args.output_dir).expanduser().resolve()
    output = Path(args.validation_output_dir).expanduser().resolve()
    policy = _policy(args.policy or (output / "policy.json" if args.resume else None))
    if args.retry_failed and not (args.execute and args.resume):
        raise ValueError("--retry-failed requires --execute --resume")
    if args.repeatability_dir and (args.execute or args.resume):
        raise ValueError("reuse --repeatability-dir only with a fresh offline assessment")
    plan, campaign, checkpoint, formal = _collection_source(request, root)
    frozen = _repeat_plan(plan, campaign, checkpoint, policy)
    inputs = {
        "request": _identity(Path(args.config)),
        "collection_directory": str(root),
        "support_plan": _identity(root / "support-plan.json"),
        "collection_plan": _identity(campaign / "collection-plan.json"),
        "checkpoint": _identity(checkpoint),
        "formal_data": formal,
    }
    if args.resume:
        if _json(output / "campaign.json") != inputs or _json(output / "repeatability-selection.json") != frozen:
            raise ValueError("validation campaign inputs changed; preserve it and use a fresh output directory")
        if ValidationPolicy.model_validate(_json(output / "policy.json")) != policy:
            raise ValueError("validation policy changed; reassess in a fresh output directory")
    else:
        _fresh(
            output,
            root,
            Path(args.config),
            *([Path(args.policy)] if args.policy else []),
            *([Path(args.repeatability_dir)] if args.repeatability_dir else []),
        )
        _write(output / "policy.json", policy.model_dump(mode="json"))
        _write(output / "campaign.json", inputs)
        _write(output / "repeatability-selection.json", frozen)
    report: dict[str, Any] = {
        "schema_version": "aisimulate-collection-validation/v1",
        "status": "incomplete",
        "accuracy": "not_assessed",
        "policy": _identity(output / "policy.json"),
        "inputs": inputs,
        "selection": _identity(output / "repeatability-selection.json"),
        "gates": {
            "point_validity": {"status": "passed"},
            "execution": {
                "status": _execution_status(frozen),
                "cells": [{"cell_id": c["cell_id"], **c["execution"]} for c in frozen["cells"]],
            },
            "repeatability": {"status": "incomplete", "issues": ["bounded GPU repeats have not completed"]},
            "interpolation": {"status": "incomplete"},
        },
    }
    report_path = output / "collection-validation.json"
    _write(report_path, report)
    holdout_dir = output / "holdout"
    if args.resume:
        with tempfile.TemporaryDirectory(prefix="aisimulate-holdout-check-") as temporary:
            holdout = _holdout(request, root, Path(temporary) / "holdout", policy)
        old = _json(holdout_dir / "interpolation-validation.json")
        _compare_holdout(old, holdout)
    else:
        holdout = _holdout(request, root, holdout_dir, policy)
    report["gates"]["interpolation"] = {
        "status": holdout["status"],
        "report": _identity(holdout_dir / "interpolation-validation.json"),
    }
    _write(report_path, report)
    repeats = Path(args.repeatability_dir).resolve() if args.repeatability_dir else output / "repeatability"
    if args.execute:
        run_repeatability(
            plan,
            generator_overrides=load_repeatability_deployment(campaign),
            source_campaign_dir=campaign,
            source_checkpoint_path=checkpoint,
            output_dir=repeats,
            resume=args.resume and repeats.exists(),
            retry_failed=args.retry_failed,
            samples=policy.repeatability.samples,
            max_points_per_cell=policy.repeatability.max_points_per_cell,
            cv_threshold=policy.repeatability.max_cv,
        )
    if repeats.exists():
        assessment = _assess_repeats(repeats, policy, frozen)
        _write(output / "repeatability-assessment.json", assessment)
        report["gates"]["repeatability"] = {
            "status": assessment["status"],
            "directory": str(repeats),
            "assessment": _identity(output / "repeatability-assessment.json"),
        }
    report["status"] = _status(report["gates"])
    _write(report_path, report)
    print(f"Collection validation: {report['status']}; serving accuracy not assessed. Saved {report_path}")
    return (
        0
        if report["status"] == "passed"
        or (report["status"] == "incomplete" and not args.execute and not args.repeatability_dir)
        else 1
    )


def _compare_holdout(saved: dict[str, Any], current: dict[str, Any]) -> None:
    for key in ("status", "policy", "phases", "predictions", "native_query_coverage"):
        if saved.get(key) != current.get(key):
            raise ValueError(f"saved holdout {key} differs from native reassessment")
    for key, value in saved["artifacts"].items():
        if key == "source":
            for source in value.values():
                _checked(source)
        else:
            _checked(value)


def check_collection_report(
    path: Path, policy: ValidationPolicy
) -> tuple[dict[str, Any], SupportRequest, Any, dict[str, Any]]:
    """Recompute all mandatory collection gates from preserved native/formal data."""
    report = _json(path)
    if report.get("schema_version") != "aisimulate-collection-validation/v1":
        raise ValueError("unsupported collection validation report")
    previous_policy = _policy(_checked(report["policy"]))
    if previous_policy.repeatability != policy.repeatability or previous_policy.interpolation != policy.interpolation:
        raise ValueError("collection validation policy differs; reassess with the selected policy")
    inputs = report["inputs"]
    request = SupportRequest.from_yaml(_checked(inputs["request"]))
    root = Path(inputs["collection_directory"])
    for key in ("support_plan", "collection_plan", "checkpoint"):
        _checked(inputs[key])
    plan, campaign, checkpoint, formal = _collection_source(request, root)
    if formal != inputs["formal_data"]:
        raise ValueError("source formal data changed since collection validation")
    frozen = _repeat_plan(plan, campaign, checkpoint, policy)
    if _json(_checked(report["selection"])) != frozen:
        raise ValueError("source runtime evidence or repeatability selection changed")
    gates = report["gates"]
    expected_execution = _execution_status(frozen)
    if gates["point_validity"]["status"] != "passed" or gates["execution"]["status"] != expected_execution:
        raise ValueError("saved collection integrity/execution gate differs from native evidence")
    if gates["execution"].get("cells") != [{"cell_id": c["cell_id"], **c["execution"]} for c in frozen["cells"]]:
        raise ValueError("saved execution inspection differs from native observations")
    with tempfile.TemporaryDirectory(prefix="aisimulate-holdout-check-") as temporary:
        holdout = _holdout(request, root, Path(temporary) / "holdout", policy)
    _compare_holdout(_json(_checked(gates["interpolation"]["report"])), holdout)
    if gates["interpolation"]["status"] != holdout["status"]:
        raise ValueError("saved interpolation gate differs from native reassessment")
    repeat = gates["repeatability"]
    if "directory" in repeat:
        assessment = _assess_repeats(Path(repeat["directory"]), policy, frozen)
        if _json(_checked(repeat["assessment"])) != assessment or repeat["status"] != assessment["status"]:
            raise ValueError("saved repeatability gate differs from raw samples")
    elif repeat["status"] != "incomplete":
        raise ValueError("repeatability gate has no measurement evidence")
    if report["status"] != _status(gates):
        raise ValueError("collection validation status disagrees with required gates")
    return report, request, plan, frozen


def _check_replay(path: Path, original: SupportRequest, collection: dict[str, Any], source_plan: Any) -> dict[str, Any]:
    from .finalization import _merge_resources, _verify_collection, finalization_manifest, verify_finalized_data
    from .validation import _completion_issues, _fpm_artifacts, _prediction_config

    report = _json(path)
    root = _checked(report["collection_plan"]).parent
    request = SupportRequest.from_yaml(root / "request.yaml")
    check_plan(request, root)
    if request_id(request) != request_id(original):
        manifest = finalization_manifest(request)
        if (
            manifest is None
            or manifest.get("source_request_id") != request_id(original)
            or manifest.get("source_collection_plan_sha256") != source_plan.sha256
            or Path(manifest.get("source_directory", "")).resolve()
            != Path(collection["inputs"]["collection_directory"])
        ):
            raise ValueError("replay request is neither the collection request nor its finalized memory profile")
        # Finalization may change resources/provenance, never model, topology, precision or collection settings.
        payload = request.model_dump(mode="json")
        payload["fpm_profile"]["provenance"] = original.fpm_profile.provenance
        selected_index = original.fpm_profile.deployments.index(original.profile_deployment())
        payload["fpm_profile"]["deployments"][selected_index]["resources"] = (
            original.profile_deployment().resources.model_dump(mode="json")
        )
        if payload != original.model_dump(mode="json"):
            raise ValueError("finalized replay changed non-memory onboarding inputs")
        for artifact in manifest["source_artifacts"]:
            if hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError("finalization source evidence changed")
        verify_finalized_data(request, root)
        observations, verified_manifest, _, _ = _verify_collection(
            original,
            Path(collection["inputs"]["collection_directory"]),
            memory_revision=manifest.get("memory_revision"),
        )
        expected_resources = _merge_resources(
            observations,
            verified_manifest,
            source_references=manifest.get("observation_provenance") == "source_references",
        )
        if request.profile_deployment().resources.model_dump(mode="json") != expected_resources:
            raise ValueError("finalized replay resources differ from verified collection memory")
    request.profile_deployment().resources.require_memory()
    trace = _checked(report["trace"])
    config_path = _checked(report["prediction_config"])
    if load_yaml(config_path) != _prediction_config(request, root, trace):
        raise ValueError("replay configuration differs from the reviewed direct-FPM workload")
    if report["collection_id"] != request_id(request) or report["fpm_artifacts"] != _fpm_artifacts(root):
        raise ValueError("replay collection identity or formal data changed")
    source_hashes = sorted((x["sha256"], x["size_bytes"]) for x in collection["inputs"]["formal_data"])
    if sorted((x["sha256"], x["size_bytes"]) for x in report["fpm_artifacts"]) != source_hashes:
        raise ValueError("replay formal data differs from the validated collection")
    if "prediction_evidence" not in report or "coverage_evidence" not in report:
        raise KeyError("replay lacks bound prediction/coverage evidence; rerun onboard validate-fpm")
    prediction_path = _checked(report["prediction_evidence"])
    coverage_path = _checked(report["coverage_evidence"])
    if str(prediction_path) != report["prediction_report"] or str(coverage_path) != report["coverage_report"]:
        raise ValueError("replay evidence identities differ from the recorded paths")
    prediction = _json(prediction_path)
    coverage = _json(coverage_path)
    issues = _completion_issues(prediction)
    queries = coverage.get("queries", {})
    if (
        report.get("prediction_exit_code") != 0
        or coverage.get("status") != "covered"
        or queries.get("unsupported") != 0
        or queries.get("measured", 0) + queries.get("interpolated", 0) <= 0
        or report.get("fpm_query_coverage") != coverage
    ):
        issues.append("native query coverage is incomplete or inconsistent")
    return {
        "status": "passed" if not issues else "incomplete",
        "issues": issues,
        "report": _identity(path),
        "trace": _identity(trace),
        "prediction_config": _identity(config_path),
        "prediction_report": _identity(prediction_path),
        "coverage_report": _identity(Path(report["coverage_report"])),
    }


def _serving_expected(request: SupportRequest, plan: Any, frozen: dict[str, Any]) -> dict[str, Any]:
    from collector.fpm_forward.repeatability import load_repeatability_deployment

    from .serving_validation import normalize_runtime_config

    deployment = request.profile_deployment()
    workers = [worker for cell in frozen["cells"] for worker in cell["execution"]["observed_workers"]]
    expected = {
        "model": request.identity.model,
        "model_revision": request.identity.model_revision,
        "backend": request.identity.framework,
        "backend_version": request.identity.framework_version,
        "hardware": request.identity.gpu,
        "parallelism": request.parallelism(),
        "precision": {
            "weights": deployment.gemm_quant_mode,
            "fmha": deployment.fmha_quant_mode,
            "kv": deployment.kv_cache_dtype,
        },
        "scheduler": request.scheduler_limits(),
        "context_length": request.search.context_length,
        "gpu_memory_utilization": request.collection.memory_fraction,
    }
    for key in ("attention_groups", "graph_config"):
        values = [worker.get(key) for worker in workers]
        expected[key] = values[0] if values and all(item == values[0] for item in values) else None
    configurations = [normalize_runtime_config(worker.get("resolved_config", {})) for worker in workers]
    expected["runtime_config"] = (
        configurations[0] if configurations and all(item == configurations[0] for item in configurations) else None
    )
    overrides = load_repeatability_deployment(frozen["source_campaign_dir"])
    image = plan.options.slurm_container_image or overrides.get("K8sConfig", {}).get("k8s_image", "")
    match = re.search(r"(?:^|@)(sha256:[0-9a-f]{64})$", image)
    expected["image_digest"] = match[1] if match else None
    return expected


def _gate_failure(exc: Exception) -> dict[str, Any]:
    return {
        "status": "incomplete" if isinstance(exc, (FileNotFoundError, KeyError)) else "failed",
        "issues": [str(exc)],
    }


def validate_serving(args: argparse.Namespace) -> int:
    from .serving_validation import (
        prepare_serving_validation,
        run_serving_benchmark,
        validate_serving_measurements,
    )

    if args.action == "run":
        if not args.recipe or not args.aiperf_python:
            raise ValueError("serving run requires --recipe and --aiperf-python; it launches the prepared benchmark")
        receipt_path = run_serving_benchmark(Path(args.recipe), aiperf_python=Path(args.aiperf_python))
        receipt = _json(receipt_path)
        print(
            f"Serving benchmark: {receipt['status']} (producer exit {receipt['exit_code']}). Saved {receipt_path}; "
            f"logs: {receipt_path.parent / 'aiperf.stdout.log'}, {receipt_path.parent / 'aiperf.stderr.log'}"
        )
        return 0 if receipt["status"] == "completed" and receipt["exit_code"] == 0 else 1
    if not args.validation_output_dir or not args.collection_report or not args.replay_report:
        raise ValueError(
            "serving prepare/assess requires --validation-output-dir, --collection-report and --replay-report"
        )
    collection_path, replay_path = Path(args.collection_report).resolve(), Path(args.replay_report).resolve()
    output = Path(args.validation_output_dir).expanduser().resolve()
    policy_path = args.policy
    if policy_path is None and collection_path.is_file():
        policy_path = _json(collection_path).get("policy", {}).get("path")
    policy = _policy(policy_path)
    inputs = [collection_path.parent, replay_path.parent]
    if args.recipe:
        inputs.append(Path(args.recipe).parent)
    for value in (policy_path, args.tokenizer, args.aiperf_python, args.execution_evidence, args.forward_evidence):
        if value:
            inputs.append(Path(value))
    for value in (collection_path, replay_path, args.recipe, args.execution_evidence, args.forward_evidence):
        if value and Path(value).is_file():
            document = _json(Path(value))
            inputs.extend(_referenced_paths(document))
            # Replay may use a separate memory-finalized collection directory.
            if value == replay_path and document.get("collection_plan", {}).get("path"):
                inputs.append(Path(document["collection_plan"]["path"]).parent)
    _fresh(output, *inputs)
    _write(output / "policy.json", policy.model_dump(mode="json"))
    combined: dict[str, Any] = {
        "schema_version": "aisimulate-onboarding-validation/v1",
        "status": "incomplete",
        "accuracy": "not_qualified",
        "policy": _identity(output / "policy.json"),
        "scope": "selected representative collection coordinates and one complete matched Weka play",
        "gates": {name: {"status": "incomplete"} for name in ("collection", "replay", "serving")},
        "limitations": [
            "This bounded assessment does not establish accuracy outside the evaluated configurations and workload."
        ],
    }
    target = output / "validation.json"
    _write(target, combined)
    active_gate = "collection"
    try:
        collection, original, plan, frozen = check_collection_report(collection_path, policy)
        combined["gates"]["collection"] = {
            "status": collection["status"],
            "report": _identity(collection_path),
            "checks": collection["gates"],
        }
        active_gate = "replay"
        replay = _check_replay(replay_path, original, collection, plan)
        combined["gates"]["replay"] = replay
        active_gate = "collection"
        expected = _serving_expected(original, plan, frozen)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        combined["gates"][active_gate] = _gate_failure(exc)
    else:
        try:
            if args.action == "prepare":
                if not args.endpoint or not args.tokenizer:
                    raise ValueError("serving prepare requires --endpoint and --tokenizer")
                recipe_path = prepare_serving_validation(
                    trace=Path(replay["trace"]["path"]),
                    prediction_config=Path(replay["prediction_config"]["path"]),
                    prediction_report=Path(replay["prediction_report"]["path"]),
                    output=output / "serving",
                    endpoint=args.endpoint,
                    tokenizer=Path(args.tokenizer),
                    expected_execution=expected,
                    latency_p95_error=policy.serving.max_latency_p95_relative_error,
                    throughput_error=policy.serving.max_throughput_relative_error,
                    max_dispatch_delay_ms=policy.serving.max_dispatch_delay_ms,
                    max_dispatch_delay_fraction=policy.serving.max_dispatch_delay_fraction,
                    aiperf_python=Path(args.aiperf_python) if args.aiperf_python else None,
                )
                combined["recipe"] = _identity(recipe_path)
                combined["gates"]["serving"] = {
                    "status": "incomplete",
                    "issues": _json(recipe_path)["issues"] + ["matched serving measurement not assessed"],
                }
            else:
                if not args.recipe or not args.execution_evidence:
                    raise ValueError("serving assess requires --recipe and --execution-evidence")
                recipe = _json(Path(args.recipe))
                if (
                    recipe["expected_execution"] != expected
                    or recipe["trace"] != replay["trace"]
                    or recipe["prediction_config"] != replay["prediction_config"]
                    or recipe["prediction_report"] != replay["prediction_report"]
                ):
                    raise ValueError("serving recipe does not match validated collection execution and replay")
                report_path = output / "serving-validation.json"
                serving = validate_serving_measurements(
                    Path(args.recipe),
                    prediction_report=Path(replay["prediction_report"]["path"]),
                    execution_evidence_path=Path(args.execution_evidence),
                    output_report=report_path,
                    forward_evidence_path=Path(args.forward_evidence) if args.forward_evidence else None,
                    latency_p95_error=policy.serving.max_latency_p95_relative_error,
                    throughput_error=policy.serving.max_throughput_relative_error,
                    max_dispatch_delay_ms=policy.serving.max_dispatch_delay_ms,
                    max_dispatch_delay_fraction=policy.serving.max_dispatch_delay_fraction,
                )
                combined["gates"]["serving"] = {
                    "status": serving["status"],
                    "report": _identity(report_path),
                    "metrics": serving.get("metrics", {}),
                    "forward": serving.get("forward", {"status": "unavailable"}),
                    "missing_evidence": serving["missing_evidence"],
                    "failures": serving["failures"],
                }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            combined["gates"]["serving"] = _gate_failure(exc)
    combined["status"] = _status(combined["gates"])
    if combined["status"] == "passed":
        combined["accuracy"] = "qualified_for_evaluated_scope"
    _write(target, combined)
    print(f"Onboarding validation: {combined['status']}; accuracy: {combined['accuracy']}. Saved {target}")
    if (
        args.action == "prepare"
        and combined.get("recipe")
        and _json(Path(combined["recipe"]["path"]))["status"] == "ready"
    ):
        return 0
    return 0 if combined["status"] == "passed" else 1


def add_quality_parsers(actions: Any) -> None:
    collection = actions.add_parser(
        "validate-collection",
        help="Inspect collection and interpolation with editable quality policy; GPU repeats require --execute.",
    )
    collection.add_argument(
        "-c", "--config", required=True, help="Original reviewed collection request, before memory finalization."
    )
    collection.add_argument("--output-dir", required=True, help="Original completed collection plan directory.")
    collection.add_argument("--validation-output-dir", required=True, help="Separate fresh assessment directory.")
    collection.add_argument(
        "--policy", help="Editable validation policy JSON/YAML; omitted fields use documented defaults."
    )
    collection.add_argument(
        "--execute",
        action="store_true",
        help="Launch the frozen bounded repeatability subset using the original collector deployment.",
    )
    collection.add_argument(
        "--resume",
        action="store_true",
        help="Continue this exact policy and source campaign, preserving repeat attempts.",
    )
    collection.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed repeat measurements in fresh attempt directories; requires --execute --resume.",
    )
    collection.add_argument(
        "--repeatability-dir",
        help="Reuse preserved repeat measurements in a fresh offline assessment; selection/count must match.",
    )
    serving = actions.add_parser(
        "validate-serving",
        help="Prepare, run, or assess matched serving and combine mandatory onboarding validation gates.",
    )
    serving.add_argument("--action", choices=("prepare", "run", "assess"), required=True)
    serving.add_argument(
        "--validation-output-dir",
        help="Fresh directory for preparation or assessment; never replaces raw serving evidence.",
    )
    serving.add_argument(
        "--collection-report", help="Verified collection-validation.json; required for prepare/assess."
    )
    serving.add_argument(
        "--replay-report", help="Existing validate-fpm validation.json for the selected complete single-stream play."
    )
    serving.add_argument(
        "--policy", help="Validation policy JSON/YAML; defaults to the collection report's frozen policy."
    )
    serving.add_argument("--recipe", help="Prepared recipe.json; required for run/assess.")
    serving.add_argument("--endpoint", help="Caller-managed HTTP(S) serving endpoint; prepare never launches a server.")
    serving.add_argument("--tokenizer", help="Local tokenizer directory for the pinned model; required for prepare.")
    serving.add_argument(
        "--aiperf-python",
        help="Python with pinned AIPerf and target tokenizer; prepare freezes payloads, run sends traffic.",
    )
    serving.add_argument(
        "--execution-evidence",
        help="Observed serving execution JSON, including actual runtime evidence sources; required for assess.",
    )
    serving.add_argument(
        "--forward-evidence",
        help="Optional separately instrumented matched forward timings; absence is reported unavailable.",
    )
