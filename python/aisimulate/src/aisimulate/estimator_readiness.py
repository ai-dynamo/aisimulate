# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render Core-owned estimator readiness and selection failures for offline callers.

Learned execution gates consume the native required-store decision; regression
fallback messages retain the reasons each earlier estimator was rejected.
"""

from __future__ import annotations

import re
from typing import Any

FPM_SELF_SERVICE_GUIDE = "docs/perf-model/fpm-self-service/README.md"

_UNSUPPORTED_ARCHITECTURE = re.compile(r"architecture (\S+) is not supported")
_MODE_PREFIX = re.compile(r"^(OpLevel|FpmInterpolation|FpmRegression|FpmLearned): ")
_MODE_LABELS = {
    "OpLevel": "op_level",
    "FpmInterpolation": "fpm_interpolation",
    "FpmRegression": "fpm_regression",
    "FpmLearned": "fpm_learned",
}
_WRAPPER_PREFIXES = (
    "unsupported model for Rust core estimator: ",
    "compile_engine: ",
    "ValueError: ",
)
_PERF_DATA_MISSING = re.compile(r"perf database error: (.+)", re.DOTALL)
# Core reports other problems (unresolved git-lfs pointers, malformed or
# unparseable tables) with the same prefix. Only a lookup miss gets the
# missing-measurement explanation; anything else keeps its original text.
_MISSING_ROW = re.compile(
    r"data missing for|data empty for|data unavailable for|no rows in |has no compatible data|has data for|"
    r"no data to anchor"
)


def unready_estimator_message(diagnostics: dict[str, Any]) -> str:
    """Return the user-facing reason an estimator is not ready for replay."""

    provenance = diagnostics.get("provenance") or {}
    config = provenance.get("config") or {}
    if config.get("estimation_mode") == "fpm_learned":
        return learned_unready_message(diagnostics.get("readiness_report") or {})
    failures = [str(failure) for failure in provenance.get("selection_failures") or []]
    if not failures and diagnostics.get("last_warning"):
        failures = str(diagnostics["last_warning"]).split("; ")
    if not failures:
        # The caller explicitly selected regression without observations.
        return (
            "fpm_regression estimator is not ready: no workload store has a usable fit; tune it with observed "
            "forward passes or restore a calibrated state snapshot"
        )

    model = config.get("model")
    system = config.get("system")
    backend = config.get("backend")
    version = config.get("backend_version")
    systems_paths = config.get("systems_paths") or None

    version_message = _unavailable_version_message(system, backend, version, systems_paths)
    if version_message is not None:
        return version_message

    for failure in failures:
        match = _UNSUPPORTED_ARCHITECTURE.search(failure)
        if match is not None:
            return _unsupported_architecture_message(match.group(1), model)

    reasons = _distinct_reasons(failures)
    return (
        f"no timing data for {model} with {backend} {_version_label(system, backend, version, systems_paths)} "
        f"on {system}: {reasons}. Prediction will not fall back to an untrained regression estimator. "
        "Try another backend, backend_version or parallelism; to add a new model, follow the FPM "
        f"self-service guide ({FPM_SELF_SERVICE_GUIDE})"
    )


def estimator_execution_ready(diagnostics: dict[str, Any]) -> bool:
    """Read the native decision without deriving phase requirements in Python."""
    config = (diagnostics.get("provenance") or {}).get("config") or {}
    if config.get("estimation_mode") == "fpm_learned":
        return diagnostics.get("readiness_report", {}).get("execution", {}).get("ready") is True
    return diagnostics.get("readiness") == "ready"


def learned_unready_message(report: dict[str, Any]) -> str:
    """Render native required-store lifecycle details for offline callers."""
    execution = report.get("execution") or {}
    unavailable = execution.get("unavailable") or []
    details = []
    for store in unavailable:
        reason = store.get("not_ready_reason") or "unavailable"
        detail = f"{store['workload_kind']}: {reason}"
        details.append(detail)
    missing = "; ".join(details) or "Core did not provide execution readiness"
    return (
        f"fpm_learned estimator is not ready for this execution: {missing}. "
        "Select a bundle or snapshot covering the required stores, or fit and export those stores "
        "in the calling application. Prediction does not train the model or substitute another estimator"
    )


def perf_data_missing_message(error: BaseException | str) -> str | None:
    """Explain a missing performance-table row hit during replay, or return None."""

    match = _PERF_DATA_MISSING.search(str(error))
    if match is None or _MISSING_ROW.search(match.group(1)) is None:
        return None
    detail = match.group(1).strip()
    return (
        f"missing performance data: {detail}. The selected performance data has no measurement for a shape this model "
        "and parallelism need. Try another parallelism, backend or backend_version; to add a new model, "
        f"follow the FPM self-service guide ({FPM_SELF_SERVICE_GUIDE})"
    )


def _unavailable_version_message(
    system: str | None,
    backend: str | None,
    version: str | None,
    systems_paths: list[str] | None,
) -> str | None:
    if not system or not backend or not version:
        return None
    from aisimulate_core.sdk import perf_database

    slots = perf_database.get_version_slots(system, backend, systems_paths=systems_paths)
    if slots is None:
        available = perf_database.get_supported_databases(systems_paths=systems_paths).get(system, {})
        if available.get(backend):
            return None
        backends = ", ".join(sorted(name for name, versions in available.items() if versions)) or "none"
        return f"no timing data for {backend} on {system}; backends with data on {system}: {backends}"
    try:
        perf_database.resolve_query_version(system, backend, version, systems_paths=systems_paths)
    except ValueError:
        listed = ", ".join(f"{slot_version} ({slot})" for slot, slot_version in slots.items())
        return f"no timing data for {backend} {version} on {system}; available: {listed}"
    return None


def _version_label(system: str | None, backend: str | None, version: str | None, systems_paths) -> str:
    if version not in {"current", "previous", "next"} or not system or not backend:
        return str(version)
    from aisimulate_core.sdk import perf_database

    try:
        return (
            f"{version} ({perf_database.resolve_query_version(system, backend, version, systems_paths=systems_paths)})"
        )
    except ValueError:
        return str(version)


def _unsupported_architecture_message(architecture: str, model: str | None) -> str:
    from aisimulate_core.sdk.common import ARCHITECTURE_TO_MODEL_FAMILY

    supported = ", ".join(sorted(ARCHITECTURE_TO_MODEL_FAMILY))
    return (
        f"architecture {architecture} ({model}) is not supported by the op-level model; "
        f"supported architectures: {supported}. To simulate a new model, follow the FPM self-service "
        f"guide ({FPM_SELF_SERVICE_GUIDE})"
    )


def _distinct_reasons(failures: list[str]) -> str:
    by_reason: dict[str, list[str]] = {}
    for failure in failures:
        match = _MODE_PREFIX.match(failure)
        mode = _MODE_LABELS[match.group(1)] if match is not None else None
        reason = failure[match.end() :] if match is not None else failure
        for prefix in _WRAPPER_PREFIXES:
            reason = reason.removeprefix(prefix)
        by_reason.setdefault(reason.strip().rstrip("."), []).extend([mode] if mode else [])
    return "; ".join(f"{'/'.join(modes)}: {reason}" if modes else reason for reason, modes in by_reason.items())
