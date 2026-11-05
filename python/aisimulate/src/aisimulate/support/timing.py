# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Onboarding evidence for the timing model selected by ordinary prediction.

The compiler owns the mapping to canonical identity. Rust owns source resolution
and its dependency closure; this module only retains that verified output.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from aisimulate.config import CorePredictionConfig


def canonical_timing_config(prediction: dict[str, Any]) -> dict[str, Any]:
    from aisimulate.compiler import _worker_engine_args

    engine = CorePredictionConfig.model_validate(prediction).engine
    if engine.mode != "aggregated" or engine.workers.aggregated is None:
        raise ValueError("onboarding timing validation requires an aggregated worker")
    payload = _worker_engine_args(engine, engine.workers.aggregated, "aggregated", transfer_bytes_per_token=None)
    timing = payload.get("timing_model", {})
    if timing.get("type") != "external" or timing.get("provider") != "aic":
        raise ValueError("onboarding timing validation requires the canonical forward-pass provider")
    config = dict(timing["config"])
    # The engine adapter appends cache-capacity inputs beside the canonical
    # timing fields. They belong to engine memory sizing, not forward timing.
    for key in (
        "gpu_memory_utilization",
        "mem_fraction_static",
        "free_gpu_memory_fraction",
        "cuda_graph_reserved_bytes",
    ):
        config.pop(key, None)
    return config


def selected_timing_mode(prediction: dict[str, Any]) -> str:
    engine = prediction["engine"]
    # An omitted role timing block inherits engine controls, just like an
    # explicitly empty block in ordinary prediction.
    timing = engine["workers"]["aggregated"].get("timing", {})
    return timing.get("estimation_mode") or engine.get("estimation_mode", "auto")


def learned_timing_evidence(prediction: dict[str, Any], *, pin: bool = False) -> dict[str, Any]:
    """Construct once at validation setup, retaining native source and readiness facts."""
    from aisimulate_core.sdk import RustForwardPassPerfModel

    if selected_timing_mode(prediction) != "fpm_learned":
        raise ValueError("learned validation requires explicit estimation_mode: fpm_learned")
    model = RustForwardPassPerfModel.best_available(canonical_timing_config(prediction))
    try:
        resolved = model.diagnostics()["provenance"]["config"]
        # Legacy explicit paths intentionally keep their old public config
        # shape. Their native dependency inventory is still fully pinned.
        dependencies = model.dependency_info()
        if not dependencies.get("files"):
            raise ValueError("seeded learned validation requires complete native dependency records")
        result = {
            "config": resolved,
            "readiness": model.readiness_report(),
            "dependencies": dependencies["files"],
            "dependency_info": dependencies,
        }
        if pin:
            timing = prediction["engine"]["workers"]["aggregated"].setdefault("timing", {})
            timing.update(
                estimation_mode=resolved["estimation_mode"],
                fallback_policy=resolved["fallback_policy"],
                estimator_config=resolved["estimator_config"],
            )
        return result
    finally:
        model.close()


def dependency_identities(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Use the native hashes, without resolving or hashing a source a second time."""
    return [
        {"path": item["path"], "sha256": item["sha256"], "size_bytes": item["byte_length"]}
        for item in evidence["dependencies"]
    ]


class LearnedTimingDependencyError(ValueError):
    """Recorded learned timing can no longer be validated for assessment."""


def verify_timing_dependencies(evidence: dict[str, Any], *, config: dict[str, Any] | None = None) -> None:
    """Revalidate the native closure once at a replay/serving assessment boundary.

    File checks provide precise drift diagnostics. Explicit native verification
    rechecks fingerprints, including directory additions and removals;
    Python never reproduces those dependency rules. No prediction or tuning is
    performed, and the verification model is immediately closed.
    """
    from aisimulate_core.sdk import RustForwardPassPerfModel

    from ..estimator_readiness import learned_unready_message

    try:
        for ref in dependency_identities(evidence):
            path = Path(ref["path"])
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if path.stat().st_size != ref["size_bytes"] or digest != ref["sha256"]:
                raise ValueError(f"learned timing dependency changed: {path}")
        model = RustForwardPassPerfModel.best_available(evidence["config"] if config is None else config)
        try:
            model.verify_dependencies()
            if model.diagnostics()["provenance"]["config"] != evidence["config"]:
                raise ValueError("learned replay model differs from its recorded resolved configuration")
            # Legacy explicit sources keep their public configuration shape.
            # Compare Rust's full closure too: additions to an original-A data
            # directory change its fingerprint without changing a pinned file.
            if model.dependency_info() != evidence.get("dependency_info"):
                raise ValueError("learned timing native dependency closure changed")
            readiness = model.readiness_report()
            if readiness.get("execution", {}).get("ready") is not True:
                raise ValueError(learned_unready_message(readiness))
        finally:
            model.close()
    except (OSError, ValueError, RuntimeError) as error:
        raise LearnedTimingDependencyError(
            f"learned timing dependencies are unavailable or changed: {error}"
        ) from error
