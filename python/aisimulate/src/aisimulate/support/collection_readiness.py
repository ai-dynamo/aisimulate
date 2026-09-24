# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only collection readiness from attempt-bound native evidence."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import SupportRequest

REPORT_FILENAME = "fpm-readiness.json"


def _cell_report(plan, cell, campaign: Path, entry: Any) -> dict[str, Any]:
    from collector.fpm_forward.database import aggregate_cell
    from collector.fpm_forward.execution_evidence import _effective_launch, file_evidence, inspect_execution_evidence
    from collector.fpm_forward.native_artifact import _rank_artifacts, validate_native_collection

    directory = campaign / "cells" / cell.cell_id
    raw = directory / "raw"
    entry = entry if isinstance(entry, dict) else {}
    result = {
        "cell": cell.to_dict(),
        "checkpoint_status": entry.get("status", "missing"),
        "attempt_id": entry.get("attempt_id"),
        "status": "incomplete",
        "blockers": [],
        "raw_artifact_root": str(raw),
        "native_regime_counts": {},
        "formal_regime_counts": {},
        "direct_eligible_points": 0,
        "coverage": "unverified",
        "rank_diagnostics": [],
        "launch_evidence": [],
        "cache_observation_evidence": [],
    }
    try:
        result["launch_evidence"] = [
            file_evidence(directory / name)
            for name in ("generator-request.json", "run.sh", "fpm_env.sh", "collector-runtime-env.sh")
            if (directory / name).is_file()
        ]
        result["cache_observation_evidence"] = [
            file_evidence(path) for path in sorted(raw.glob("**/fpm-memory-*.json"))
        ]
        # Preserve diagnostics even when protocol validation rejects the attempt.
        # They remain explicitly unverified; never use these counts for admission.
        for path, payload in _rank_artifacts(raw):
            result["rank_diagnostics"].append(
                {
                    "source": file_evidence(path),
                    "verified": False,
                    "kvwarm": payload.get("kvwarm"),
                    "regime_counts": dict(
                        Counter(
                            row["kv_seed_regime"]
                            if isinstance(row.get("kv_seed_regime"), str)
                            else "unreported"
                            if row.get("kv_seed_regime") is None
                            else "invalid"
                            for row in payload.get("results", [])
                            if isinstance(row, dict)
                        )
                    ),
                }
            )
        attempt = entry.get("attempt_id")
        if not isinstance(attempt, str) or not attempt:
            raise ValueError("checkpoint has no current attempt identity; inspect existing artifacts before retrying")
        collection = validate_native_collection(
            cell, raw, expected_plan_sha256=plan.sha256, expected_attempt_id=attempt
        )
        result.update(
            native_regime_counts=dict(Counter(item.kv_seed_regime or "unreported" for item in collection.points)),
            kvwarm=collection.kvwarm_meta,
            observed_backend_version=collection.backend_version,
            runtime_run_id=collection.runtime_run_id,
            runtime_grid_digest=collection.runtime_grid_digest,
        )
        for diagnostic in result["rank_diagnostics"]:
            diagnostic["verified"] = True
        rows = aggregate_cell(plan, cell, directory, expected_attempt_id=attempt)
        formal_counts = Counter(row["kv_seed_regime"] for row in rows)
        # This is the existing direct consumer's row eligibility rule, not an
        # interpolation implementation or a claim of query-domain coverage.
        eligible = len(rows) - formal_counts["fake_fallback"]
        result.update(
            formal_regime_counts=dict(formal_counts),
            direct_eligible_points=eligible,
            coverage="partial" if formal_counts["fake_fallback"] else "sampled_points_only",
            execution=inspect_execution_evidence(cell, raw, collection, plan=plan),
        )
        if cell.workload_kind == "decode" and collection.kvwarm_meta is None:
            result["blockers"].append("decode KV warm-up metadata is unreported; inspect the runtime protocol")
        if not eligible:
            result["blockers"].append(
                "no direct-eligible timing points; inspect KV warm-up eligibility/skip reason and fallback logs; "
                "an unchanged full collection is not justified"
            )
        captures = plan.options.prefill_sampling.cudagraph_capture_sizes
        if cell.workload_kind == "prefill" and captures is not None and not plan.options.enforce_eager:
            flags = _effective_launch(plan, directory)
            configured = json.loads(flags.get("--compilation-config", "{}"))
            expected = {
                "cudagraph_capture_sizes": list(captures),
                "max_cudagraph_capture_size": plan.options.prefill_sampling.max_cudagraph_capture_size,
            }
            if not isinstance(configured, dict) or any(configured.get(key) != value for key, value in expected.items()):
                result["blockers"].append(
                    "saved prefill launch does not preserve the reviewed explicit graph configuration"
                )
            for worker in result["execution"]["observed_workers"]:
                graph = worker.get("graph_config")
                if isinstance(graph, dict) and any(
                    key in graph and graph[key] != value for key, value in expected.items()
                ):
                    result["blockers"].append(
                        "observed prefill graph configuration differs from the reviewed explicit settings"
                    )
            # Dynamo b83b1d9304ebfc624709ac46db32b1b6f1ff1615 instrumented_scheduler.py:2302-2350,6223-6236:
            # Full native captures reflect compilation_config; phase lists may be filtered.
            native_expected = {
                "capture_sizes": expected["cudagraph_capture_sizes"],
                "max_capture_size": expected["max_cudagraph_capture_size"],
            }
            for native in result["execution"]["native_graph_config"]:
                graph = native["config"]
                if isinstance(graph, dict) and any(
                    key in graph and graph[key] != value for key, value in native_expected.items()
                ):
                    result["blockers"].append(
                        "native prefill graph configuration differs from the reviewed explicit settings"
                    )
        if result["execution"]["failures"]:
            result["blockers"].extend(result["execution"]["failures"])
        if entry.get("cleanup_error"):
            result["blockers"].append(
                "current attempt has unresolved resource cleanup; retain its cleanup error and inspect"
            )
        if entry.get("status") != "passed":
            result["blockers"].append(
                "current attempt is not passed; inspect or resume saved evidence before collection"
            )
        result["status"] = "blocked" if result["blockers"] else "ready"
    except (OSError, TypeError, ValueError, KeyError) as error:
        result["blockers"].append(str(error))
        result["status"] = "blocked" if result["rank_diagnostics"] else "incomplete"
    result["next_action"] = (
        "full collection may proceed; repeatability, query coverage, memory and serving accuracy remain separate"
        if result["status"] == "ready"
        else "inspect preserved evidence and resolve the blocker before an unchanged full collection; "
        "qualify a runtime/launch change with a new bounded campaign when required"
    )
    return result


def _matches_request(request: SupportRequest, plan) -> bool:
    if (
        plan.model_path != request.identity.model
        or plan.system != request.identity.gpu
        or plan.backend != request.identity.framework
        or plan.capability.aic_database_version != request.identity.framework_version
        or plan.fpm_profile != request.fpm_profile
    ):
        return False
    scheduler = request.scheduler_limits()
    parallel = request.parallelism()
    return (
        plan.options.vllm_max_model_len == request.search.context_length
        and plan.options.max_num_batched_tokens == scheduler["max_batched_tokens"]
        and plan.options.max_num_seqs == scheduler["max_sequences"]
        and plan.options.prefill_cudagraph_policy == request.collection.prefill_cudagraph_policy
        and plan.options.max_prefill_cudagraph_size == request.collection.max_prefill_cudagraph_size
        and (plan.options.gpu_memory_utilization or 0.9) == request.collection.memory_fraction
        and all(
            cell.topology.tp == parallel["tensor"]
            and cell.topology.dp == parallel["attention_data"]
            and cell.topology.moe_tp == parallel["moe_tensor"]
            and cell.topology.moe_ep == parallel["moe_expert"]
            for cell in plan.cells
        )
    )


def assess_readiness(
    request: SupportRequest,
    root: Path,
    checkpoint_root: Path,
    *,
    expected_plan=None,
) -> dict[str, Any]:
    """Inspect saved sources without model fetching, reruns or checkpoint writes.

    A formal attempt supersedes smoke evidence for the same cell, even when it
    failed. Only the selected checkpoint's current attempt can qualify a cell.
    """
    from collector.fpm_forward.execution_evidence import file_evidence
    from collector.fpm_forward.repeatability import load_repeatability_deployment, load_repeatability_source
    from collector.fpm_forward.runner import CHECKPOINT_SCHEMA

    campaigns = []
    selected = {}
    blockers = []
    for smoke in (True, False):
        checkpoint_path = checkpoint_root / ("fpm_forward_smoke.json" if smoke else "fpm_forward.json")
        if not checkpoint_path.exists():
            continue
        report = {"kind": "smoke" if smoke else "formal", "cells": [], "errors": []}
        campaigns.append(report)
        try:
            report["checkpoint"] = file_evidence(checkpoint_path)
            checkpoint = json.loads(checkpoint_path.read_text())
            if not isinstance(checkpoint, dict):
                raise ValueError("collector checkpoint must be an object")
            sha = checkpoint.get("plan_sha256")
            if (
                checkpoint.get("schema") != CHECKPOINT_SCHEMA
                or not isinstance(sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", sha)
                or not isinstance(checkpoint.get("cells"), dict)
            ):
                raise ValueError("invalid collector checkpoint identity or cell records")
            campaign = root / "fpm-artifacts" / sha[:16]
            if smoke:
                campaign /= "smoke"
            plan = load_repeatability_source(campaign)
            if plan.sha256 != sha or not _matches_request(request, plan):
                raise ValueError("saved collection differs from the reviewed onboarding request")
            if expected_plan is not None and plan.sha256 != expected_plan.sha256:
                raise ValueError(
                    "saved collection differs from the current frozen runtime/launch plan; use a new campaign"
                )
            deployment = load_repeatability_deployment(campaign)
            report.update(
                plan_sha256=plan.sha256,
                source_plan=file_evidence(campaign / "collection-plan.json"),
                runtime_identity={
                    "model": plan.model_path,
                    "model_revision": request.identity.model_revision,
                    "system": plan.system,
                    "backend": plan.backend,
                    "backend_version": plan.capability.aic_database_version,
                    "collector_revision": plan.aic_revision,
                    "generator_config_sha256": plan.generator_config_sha256,
                    "deployment": deployment,
                    "executor": plan.options.executor,
                    "slurm_image": plan.options.slurm_container_image,
                    "runtime_observation": plan.to_dict().get("runtime_observation"),
                    "scope": "frozen requested identity; observed backend and launch evidence are reported per cell",
                },
            )
            for cell in plan.cells:
                entry = checkpoint["cells"].get(cell.cell_id)
                cell_report = _cell_report(plan, cell, campaign, entry)
                report["cells"].append(cell_report)
                if cell.cell_id not in selected or cell.cell_id in checkpoint["cells"]:
                    selected[cell.cell_id] = {"campaign": report["kind"], "plan_sha256": plan.sha256, **cell_report}
        except (OSError, TypeError, ValueError, KeyError) as error:
            report["errors"].append(str(error))
            blockers.append(f"{report['kind']}: {error}")
    if len({item["plan_sha256"] for item in selected.values()}) > 1:
        blockers.append("selected phase evidence belongs to different frozen runtime/launch plans")
    if expected_plan is not None and set(selected) != {cell.cell_id for cell in expected_plan.cells}:
        blockers.append("not every current planned cell has saved readiness evidence")
    if {item["cell"]["workload_kind"] for item in selected.values()} != {"prefill", "decode"}:
        blockers.append("readiness requires both prefill and decode for every selected configuration")
    for item in selected.values():
        blockers.extend(
            f"{item['cell']['workload_kind']} {item['cell']['cell_id']}: {reason}" for reason in item["blockers"]
        )
    ready = bool(selected) and not blockers and all(item["status"] == "ready" for item in selected.values())
    return {
        "status": "ready" if ready else "blocked" if campaigns else "incomplete",
        "ready_for_full_collection": ready,
        "scope": "sampled native timing readiness only; not complete query coverage, memory or serving accuracy",
        "request_identity": request.identity.model_dump(mode="json"),
        "blockers": blockers,
        "selected_cells": list(selected.values()),
        "campaigns": campaigns,
    }


def resume_without_workers(plan, checkpoint_path: Path) -> bool:
    """Recognize only the runner's guaranteed no-worker resume path.

    Running/interrupted/missing cells can launch cleanup or workers. Do not
    admit those states, even if their raw files look complete. Failed and
    cleanup-failed cells are skipped without --resume-retry-failed, which the
    guided CLI does not expose. The collector still revalidates recovery.
    """
    from collector.fpm_forward.runner import CHECKPOINT_SCHEMA

    try:
        checkpoint = json.loads(checkpoint_path.read_text())
        return (
            isinstance(checkpoint, dict)
            and checkpoint.get("schema") == CHECKPOINT_SCHEMA
            and checkpoint.get("plan_sha256") == plan.sha256
            and isinstance(checkpoint.get("cells"), dict)
            and bool(plan.cells)
            and all(
                isinstance(entry := checkpoint["cells"].get(cell.cell_id), dict)
                and entry.get("status") in {"passed", "failed", "cleanup_failed"}
                and isinstance(entry.get("attempt_id"), str)
                and bool(entry["attempt_id"])
                for cell in plan.cells
            )
        )
    except (OSError, TypeError, ValueError):
        return False
