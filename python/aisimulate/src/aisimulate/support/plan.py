# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create ordinary prediction/recommendation inputs without resolving a model."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig

from .fpm import fpm_cli_args
from .schema import SupportRequest

_LOCK_NAME = ".support.lock"


def request_id(request: SupportRequest) -> str:
    """Identify reusable collection inputs, independent of evaluation traffic."""

    collection = request.model_dump(mode="json", exclude_none=True)
    collection.pop("workload")
    for key in ("objective", "seed"):
        collection["search"].pop(key)
    payload = json.dumps(collection, sort_keys=True, separators=(",", ":"))
    return "onboarding-" + hashlib.sha256(payload.encode()).hexdigest()


def validation_id(request: SupportRequest) -> str:
    """Bind generated validation examples to the complete saved request."""
    payload = json.dumps(request.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":"))
    return "validation-" + hashlib.sha256(payload.encode()).hexdigest()


def _configs(
    request: SupportRequest, root: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, int]]]:
    preset = request.parallelism()
    workload = request.workload
    common = {
        "traffic": {
            "source": {
                "type": "synthetic",
                "input_tokens": workload.input_tokens,
                "output_tokens": workload.output_tokens,
            },
            "load": {"type": "concurrency", "concurrency": workload.concurrency},
            "stop": {"requests": workload.request_count},
        },
        "evaluation": {"sla": {"ttft_ms": workload.slo.ttft_ms, "itl_ms": workload.slo.tpot_ms}},
    }
    engine = {
        "mode": "aggregated",
        "model": request.identity.model,
        "hardware": request.identity.gpu,
        "backend": request.identity.framework,
        "backend_version": request.identity.framework_version,
        "systems_paths": [str(root / "systems")],
        "context_length": request.search.context_length,
    }
    # Recommendation otherwise expands scheduler defaults into extra domains.
    worker = {
        "scheduler": request.scheduler_limits(),
        "timing": {"type": "default", "estimation_mode": "fpm_interpolation", "fallback_policy": "deny"},
    }
    if request.collection.gpu_memory_utilization is not None:
        worker["kv_cache"] = {"capacity": {"memory_fraction": request.collection.gpu_memory_utilization}}
    if request.fpm_profile is not None:
        engine["fpm_profile"] = request.fpm_profile.model_dump(mode="json")
        worker["timing"]["estimator_config"] = {"fpm_interpolation": {"method": "direct"}}
        if request.profile_deployment().resources.cache_layout == "grouped":
            worker.setdefault("kv_cache", {})["prefix_caching"] = False
    prediction = {
        **common,
        "engine": {**engine, "workers": {"aggregated": {**worker, "parallelism": preset}}},
    }
    CorePredictionConfig.model_validate(prediction)
    recommendation = {
        **common,
        "engine": {**engine, "workers": {"aggregated": {**worker, "parallelism": {"preset": [preset]}}}},
        "optimization": {
            "target": request.search.objective,
            # Match this validation worker even when it exceeds the runtime's default cap.
            # This is derived scope, not a declaration of available deployment GPUs.
            "constraints": {"max_candidate_gpus": request.worker_gpus},
        },
        "optimizer": {"algorithm": "random", "max_trials": 1, "parallelism": 1, "seed": request.search.seed},
    }
    CoreRecommendationConfig.model_validate(recommendation)
    return prediction, {"pilot": recommendation}, [preset]


def _commands(request: SupportRequest, root: Path, recommendation_names: list[str]) -> dict[str, Any]:
    commands = {
        "fpm_plan_local": fpm_cli_args(request, output_dir=root, plan_only=True),
        "fpm_run_local": [
            "aisimulate",
            "onboard",
            "collect-fpm",
            "--config",
            str(root / "request.yaml"),
            "--output-dir",
            str(root),
            "--execute",
        ],
        "predict": [
            [
                "aisimulate",
                "predict",
                "--config",
                str(root / "predict/pilot.yaml"),
                "--output-dir",
                str(root / "predict-results/pilot"),
                "--format",
                "json",
            ]
        ],
        "recommend": [
            [
                "aisimulate",
                "recommend",
                "--config",
                str(root / f"recommend/{name}.yaml"),
                "--output-dir",
                str(root / f"recommend-results/{name}"),
                "--format",
                "json",
            ]
            for name in recommendation_names
        ],
    }
    if request.fpm_profile is not None and not request.profile_deployment().resources.memory_ready:
        commands["finalize"] = [
            "aisimulate",
            "onboard",
            "finalize",
            "--config",
            str(root / "request.yaml"),
            "--output-dir",
            str(root),
            "--resolved-output-dir",
            str(root.with_name(root.name + "-resolved")),
        ]
    return commands


def _plan_documents(request: SupportRequest, root: Path) -> tuple[dict[str, Any], dict[Path, bytes]]:
    source_spec = Path(__file__).resolve().parents[2] / "aisimulate_core/systems" / f"{request.identity.gpu}.yaml"
    if not source_spec.is_file():
        raise ValueError(
            f"GPU {request.identity.gpu!r} has no packaged system specification; "
            "new GPU architectures require integration"
        )
    prediction, recommendations, presets = _configs(request, root)
    commands = _commands(request, root, list(recommendations))
    plan = {
        "schema_version": "aisimulate-support-plan/v2",
        "request_id": request_id(request),
        "validation_id": validation_id(request),
        "search": {
            "candidate_count": len(presets),
            "candidates": [{"parallelism": preset, "required_gpus": request.worker_gpus} for preset in presets],
            "detail": (
                "Prediction and recommendation are single-worker validation examples for the selected topology. "
                "The recommendation GPU cap equals that worker's required GPUs; no available allocation is declared. "
                "Set deployment replicas and optimization budgets in ordinary predict/recommend configs."
            ),
            "baseline_rule": "Selected single worker; model parallelism legality and memory fit remain unchecked.",
        },
        "fpm": {
            "status": "planned",
            "collection_gpus_required": request.worker_gpus,
            "resource_requirement": (
                "At least this many GPUs are required for one selected worker (attention TP times attention DP). "
                "Actual available GPUs, placement and collector runtime resources remain unchecked."
            ),
            "plan_command": commands["fpm_plan_local"],
            "runtime_limits": request.collection_settings(),
            "sampling": (
                "Dynamo self-benchmark generates the prefill/decode grid from CUDA graph sizes and the reviewed "
                "runtime limits. The new-token budget, per-request context, scheduler sequence bound and total KV "
                "capacity are separate quantities. Exact points resolve at runtime; validation traffic and SLAs "
                "do not change these collection settings. Smoke and limited runs do not publish formal FPM data."
            ),
        },
        "prerequisites": [
            {
                "id": "model_integration",
                "status": "not_checked",
                "detail": (
                    "Verify pinned model metadata, memory/cache accounting, and chosen parallelism. Without an "
                    "FPM profile, predict/recommend uses a registered analytical class and SOL transfer; follow "
                    "python/aisimulate/docs/add_a_new_model.md for that route. Supply an FPM identity/resource "
                    "profile to use direct interpolation without a class. Per-operation silicon data is not required."
                ),
            },
            {
                "id": "runtime",
                "status": "not_checked",
                "detail": (
                    "Prepare and verify the declared model/tokenizer revisions, vLLM version, visible GPUs, "
                    "model access, and the packaged Dynamo/Kubernetes/Generator collector runtime. Local invocation "
                    "uses that existing runtime. The collector does not apply revision/version declarations; "
                    "use a pinned local model snapshot where needed."
                ),
            },
            {
                "id": "fpm_data",
                "status": "not_checked",
                "detail": "Collect matching prefill/decode timings into the local systems tree before prediction or "
                "recommendation. Creating a plan establishes neither timing coverage nor memory fit.",
            },
        ],
        "accuracy": {
            "status": "not_assessed",
            "detail": "No measured accuracy or E2E validation is claimed by this plan.",
        },
        "outputs": {
            "request": str(root / "request.yaml"),
            "prediction_configs": [str(root / "predict/pilot.yaml")],
            "recommendation_configs": [str(root / f"recommend/{name}.yaml") for name in recommendations],
            "recommendation_results": [str(root / f"recommend-results/{name}") for name in recommendations],
            "commands": str(root / "commands.json"),
            "systems_root": str(root / "systems"),
        },
    }
    if "prefill_cudagraph_policy" in request.collection.model_fields_set:
        plan["fpm"]["launch"] = (
            "AISimulate launches and manages benchmark workers when collection is executed; no separately "
            "launched HTTP server is required. vLLM initializes the model, caches and CUDA graphs; Dynamo "
            "self-benchmark generates and measures the feasible grid. Initialization and planning do not launch GPUs."
        )
    if request.fpm_profile is not None:
        from aisimulate_core.sdk.memory import estimate_kv_cache

        deployment = request.profile_deployment()
        scheduler = prediction["engine"]["workers"]["aggregated"]["scheduler"]
        estimate = (
            estimate_kv_cache(
                request.identity.model,
                request.identity.gpu,
                request.identity.framework,
                backend_version=request.identity.framework_version,
                max_num_tokens=scheduler["max_batched_tokens"],
                max_batch_size=scheduler["max_sequences"],
                memory_fraction_kind="of_total",
                memory_fraction_value=request.collection.memory_fraction,
                tp_size=deployment.tp,
                pp_size=deployment.pp,
                attention_dp_size=deployment.dp,
                moe_tp_size=deployment.moe_tp,
                moe_ep_size=deployment.moe_ep,
                fpm_profile=request.fpm_profile,
                context_length=request.search.context_length,
            )
            if deployment.resources.memory_ready
            else {"memory_source": "pending", "simulation_ready": False}
        )
        if not deployment.resources.memory_ready:
            plan["fpm"]["scheduling_policy"] = (
                "Synchronous scheduling in both phases; current collector prefill benchmarking requires it. "
                "This is a collection policy, not the default for arbitrary vLLM serving."
                if request.identity.framework_version == "0.27.0"
                else "Native timing collection policy; runtime memory observation is unavailable for this vLLM version."
            )
        elif deployment.resources.cache_layout == "grouped":
            if estimate["request_peak_cache_bytes"] > estimate["total_kv_size_bytes"]:
                raise ValueError(
                    "declared FPM resources leave insufficient rank-local bytes for grouped cache peak allocation"
                )
        elif estimate["total_kv_size_tokens"] <= request.search.context_length:
            raise ValueError("declared FPM resources leave insufficient rank-local KV capacity for the runtime context")
        plan["resources"] = estimate
        plan["search"]["baseline_rule"] = (
            "Selected profile deployment; declared topology and rank-local resource bounds pass CPU admission. "
            "Serving-runtime compatibility and silicon accuracy remain unchecked."
        )
        plan["prerequisites"][0].update(
            status="declared_profile_validated",
            detail=(
                "The pinned identity, declared deployment, scheduler envelope and resource budget validate without "
                "an analytical model class. Verify the profile's resource provenance against the serving runtime. "
                "Generated configs use direct interpolation; matching timing coverage is still required."
            ),
        )
        plan["prerequisites"][1]["detail"] = (
            "Prepare and verify the declared model/tokenizer revisions, visible GPUs, model access, and the "
            "packaged Dynamo/Kubernetes/Generator collector runtime. This plan does not download a pinned "
            "checkpoint or inspect the running runtime. During collection execution, the collector checks "
            "the observed Pod vLLM version against the profile's literal backend version before benchmarking."
        )
        plan["prerequisites"][2]["detail"] = (
            "Collect matching prefill/decode timings into the local systems tree before prediction or recommendation. "
            "The profile resource estimate does not establish timing coverage or measured accuracy."
        )
        if not deployment.resources.memory_ready:
            plan["search"]["baseline_rule"] = (
                "Selected profile deployment; memory remains pending runtime initialization. "
                "Collection is permitted; prediction, recommendation and replay require finalized memory."
            )
            plan["prerequisites"][0].update(
                status="runtime_memory_pending",
                detail=(
                    "Identity, precision, topology and cache geometry are declared. Runtime memory is collected "
                    "during worker initialization for supported vLLM 0.27.0 layouts; other versions can collect "
                    "timings but leave memory unresolved. No activation or non-KV byte declaration is required. "
                    "Run onboard finalize after successful collection, then review the resolved profile."
                ),
            )
        elif deployment.resources.memory_source == "runtime":
            plan["search"]["baseline_rule"] = (
                "Selected deployment uses observed cache capacity for the exact recorded runtime settings. "
                "Timing coverage and silicon accuracy require separate validation."
            )
            plan["prerequisites"][0].update(
                status="runtime_memory_resolved",
                detail=(
                    "Observed cache capacity and grouped geometry are bound to this deployment and runtime envelope."
                ),
            )
        plan["outputs"]["fpm_model_profile"] = str(root / "fpm-model-profile.json")
    yaml_documents = {
        Path("request.yaml"): request.model_dump(mode="json", exclude_none=True),
        Path("predict/pilot.yaml"): prediction,
        **{Path(f"recommend/{name}.yaml"): config for name, config in recommendations.items()},
    }
    documents = {path: yaml.safe_dump(data, sort_keys=False).encode() for path, data in yaml_documents.items()}
    if request.fpm_profile is not None:
        documents[Path("fpm-model-profile.json")] = (
            json.dumps(request.fpm_profile.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        ).encode()
        from .finalization import finalization_manifest

        manifest = finalization_manifest(request)
        if manifest is not None:
            documents[Path("finalization.json")] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    documents[Path("systems") / source_spec.name] = source_spec.read_bytes()
    documents[Path("commands.json")] = (json.dumps(commands, indent=2, sort_keys=True) + "\n").encode()
    plan["generated_files_sha256"] = {
        path.as_posix(): hashlib.sha256(content).hexdigest() for path, content in documents.items()
    }
    documents[Path("support-plan.json")] = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
    return plan, documents


@contextmanager
def plan_lock(root: Path) -> Iterator[None]:
    """Keep one stable lock inode; the OS releases ownership on process exit."""

    lock = root / _LOCK_NAME
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"onboarding lock must be a regular file: {lock}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"another onboarding operation holds {lock}") from exc
        yield
    finally:
        os.close(descriptor)


def _check_paths(root: Path) -> None:
    for relative in (
        "request.yaml",
        "support-plan.json",
        "fpm-model-profile.json",
        "commands.json",
        "predict",
        "predict/pilot.yaml",
        "recommend",
        "recommend/pilot.yaml",
        "systems",
        "systems/data",
        "fpm-checkpoint",
        "fpm-artifacts",
    ):
        if (root / relative).is_symlink():
            raise ValueError(f"refusing symlinked plan output {root / relative}")
    pending = [root / relative for relative in ("systems/data", "fpm-checkpoint", "fpm-artifacts")]
    while pending:
        path = pending.pop()
        if path.is_symlink():
            raise ValueError(f"refusing symlinked plan output {path}")
        if path.is_dir():
            pending.extend(path.iterdir())


def check_plan(request: SupportRequest, root: Path, *, allow_missing: bool = False) -> None:
    """Verify saved identity before collecting or reusing any local output."""

    _check_paths(root)
    try:
        prior = json.loads((root / "support-plan.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"a readable onboarding plan is required in {root}; run aisimulate onboard plan first"
        ) from exc
    if not isinstance(prior, dict) or prior.get("schema_version") != "aisimulate-support-plan/v2":
        raise ValueError(
            f"legacy or invalid onboarding plan in {root}; regenerate in a new output directory. "
            "Legacy plans used validation traffic to choose collection bounds and cannot be reused implicitly."
        )
    if prior.get("request_id") != request_id(request):
        raise ValueError(f"refusing to mix a different request identity in {root}; choose a new output directory")
    try:
        saved = SupportRequest.from_yaml(root / "request.yaml")
    except ValueError as exc:
        raise ValueError(f"saved request in {root} was modified or cannot be read") from exc
    if request_id(saved) != prior["request_id"] or validation_id(saved) != prior.get("validation_id"):
        raise ValueError(f"saved request identity in {root} was modified or differs from the requested plan")
    expected, documents = _plan_documents(saved, root)
    if prior != expected:
        raise ValueError(
            f"generated plan input {root / 'support-plan.json'} was modified; choose a new output directory"
        )
    # Recompute the manifest rather than trusting hashes in an edited plan. An
    # evaluation-only refresh must never bless modified collector inputs.
    for relative, content in documents.items():
        destination = root / relative
        if destination.is_symlink():
            raise ValueError(f"refusing symlinked plan output {destination}")
        if not destination.exists() and allow_missing:
            continue
        if not destination.is_file() or destination.read_bytes() != content:
            raise ValueError(
                f"generated plan input {destination} was modified or missing; choose a new output directory"
            )
    from .finalization import verify_finalized_data

    verify_finalized_data(saved, root)


def create_plan(request: SupportRequest, output_dir: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Write a plan or safely refresh validation inputs for the same collection."""

    root = Path(output_dir).expanduser().resolve()
    plan, documents = _plan_documents(request, root)
    root.mkdir(parents=True, exist_ok=True)
    with plan_lock(root):
        _check_paths(root)
        nonempty = any(path.name != _LOCK_NAME for path in root.iterdir())
        if nonempty:
            if not overwrite:
                raise ValueError(f"output directory {root} is nonempty; use overwrite only for the same collection")
            check_plan(request, root, allow_missing=True)
        # Inspect every generated file before writing any file. Existing data,
        # checkpoints and results are never replaced or deleted. Only verified
        # generated inputs may change after an evaluation-only request edit.
        pending: dict[Path, bytes] = {}
        for relative, content in documents.items():
            destination = root / relative
            if destination.is_symlink():
                raise ValueError(f"refusing symlinked plan output {destination}")
            if not destination.exists() or destination.read_bytes() != content:
                pending[relative] = content
        (root / "systems/data").mkdir(parents=True, exist_ok=True)
        for relative, content in pending.items():
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with destination.open("xb") as handle:
                    handle.write(content)
            else:
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
                        temporary = Path(handle.name)
                        handle.write(content)
                    os.replace(temporary, destination)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
    return plan
