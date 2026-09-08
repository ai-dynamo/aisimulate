#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the AISimulate 0.13 modeled-power qualification ledger.

The default mode validates structure and coverage without pretending that a
pending gate passed. ``--require-release-ready`` is the fail-closed release
gate: it also requires every release-blocking entry to carry immutable passing
evidence and requires an approved silicon-accuracy threshold.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "docs" / "power" / "qualification-matrix.json"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
ISSUE_RE = re.compile(r"^AIC-[0-9]+$")

DIMENSIONS = {
    "model_families": {"dense", "moe", "any"},
    "deployments": {"aggregated", "disaggregated", "afd", "epd", "any"},
    "runners": {"standard", "dynamo", "any"},
    "timing_backends": {
        "default_op_level",
        "default_fpm",
        "fixed",
        "polynomial",
        "any",
    },
    "coverage_states": {"covered", "undercovered", "not_applicable"},
}
EXPECTATIONS = {
    "modeled_power",
    "unavailable_power",
    "operation_energy_evidence",
    "aic_parity",
    "silicon_accuracy",
    "runner_output",
    "packaged_output",
    "integration_available",
    "data_invariants",
}
STATUSES = {"pending", "blocked", "passed", "failed"}


class QualificationError(ValueError):
    """Raised when the qualification ledger is malformed or not releasable."""


def _expect(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _expect_keys(
    value: Any, required: set[str], context: str, errors: list[str]
) -> None:
    _expect(isinstance(value, dict), f"{context} must be an object", errors)
    if not isinstance(value, dict):
        return
    missing = required - value.keys()
    _expect(not missing, f"{context} is missing {sorted(missing)}", errors)
    unexpected = value.keys() - required
    _expect(
        not unexpected, f"{context} has unexpected keys {sorted(unexpected)}", errors
    )


def _validate_matrix(matrix: Any, context: str, errors: list[str]) -> None:
    _expect_keys(matrix, set(DIMENSIONS), context, errors)
    if not isinstance(matrix, dict):
        return
    for dimension, allowed in DIMENSIONS.items():
        values = matrix.get(dimension)
        _expect(
            isinstance(values, list) and values,
            f"{context}.{dimension} must be a non-empty array",
            errors,
        )
        if not isinstance(values, list):
            continue
        _expect(
            all(isinstance(value, str) for value in values),
            f"{context}.{dimension} must contain strings",
            errors,
        )
        if not all(isinstance(value, str) for value in values):
            continue
        _expect(
            len(values) == len(set(values)),
            f"{context}.{dimension} contains duplicates",
            errors,
        )
        unknown = set(values) - allowed
        _expect(
            not unknown,
            f"{context}.{dimension} contains unsupported values {sorted(unknown)}",
            errors,
        )
        _expect(
            "any" not in values or len(values) == 1,
            f"{context}.{dimension} cannot combine 'any' with concrete values",
            errors,
        )


def _validate_assertion(
    assertion: Any, matrix: dict[str, list[str]], context: str, errors: list[str]
) -> None:
    _expect_keys(assertion, {"when", "expectation", "thresholds"}, context, errors)
    if not isinstance(assertion, dict):
        return
    when = assertion.get("when")
    _expect(isinstance(when, dict), f"{context}.when must be an object", errors)
    if isinstance(when, dict):
        for dimension, selected in when.items():
            _expect(
                dimension in DIMENSIONS,
                f"{context}.when has unknown dimension {dimension!r}",
                errors,
            )
            _expect(
                isinstance(selected, list) and selected,
                f"{context}.when.{dimension} must be non-empty",
                errors,
            )
            if isinstance(selected, list):
                _expect(
                    all(isinstance(item, str) for item in selected),
                    f"{context}.when.{dimension} must contain strings",
                    errors,
                )
            if (
                dimension in matrix
                and isinstance(selected, list)
                and all(isinstance(item, str) for item in selected)
            ):
                _expect(
                    set(selected) <= set(matrix[dimension]),
                    f"{context}.when.{dimension} is outside the gate matrix",
                    errors,
                )
    _expect(
        assertion.get("expectation") in EXPECTATIONS,
        f"{context}.expectation is unsupported",
        errors,
    )
    _expect(
        isinstance(assertion.get("thresholds"), dict),
        f"{context}.thresholds must be an object",
        errors,
    )


def _validate_evidence(evidence: Any, context: str, errors: list[str]) -> None:
    _expect_keys(
        evidence,
        {"result", "artifact", "sha256", "source_revision", "recorded_at"},
        context,
        errors,
    )
    if not isinstance(evidence, dict):
        return
    _expect(
        evidence.get("result") in {"pass", "fail"},
        f"{context}.result must be pass or fail",
        errors,
    )
    artifact = evidence.get("artifact")
    _expect(
        isinstance(artifact, str) and bool(artifact.strip()),
        f"{context}.artifact must be non-empty",
        errors,
    )
    if isinstance(artifact, str) and artifact:
        path_parts = Path(artifact).parts
        _expect(
            artifact.startswith("https://")
            or (
                "://" not in artifact
                and not Path(artifact).is_absolute()
                and ".." not in path_parts
            ),
            f"{context}.artifact must be HTTPS or a repository-relative path",
            errors,
        )
    _expect(
        isinstance(evidence.get("sha256"), str)
        and bool(DIGEST_RE.fullmatch(evidence["sha256"])),
        f"{context}.sha256 must be 64 lowercase hex",
        errors,
    )
    _expect(
        isinstance(evidence.get("source_revision"), str)
        and bool(SHA_RE.fullmatch(evidence["source_revision"])),
        f"{context}.source_revision must be a 40-character commit SHA",
        errors,
    )
    recorded_at = str(evidence.get("recorded_at", ""))
    try:
        parsed_time = datetime.fromisoformat(recorded_at.removesuffix("Z") + "+00:00")
    except ValueError:
        parsed_time = None
    _expect(
        recorded_at.endswith("Z")
        and parsed_time is not None
        and parsed_time.utcoffset() == UTC.utcoffset(parsed_time),
        f"{context}.recorded_at must be UTC ISO-8601",
        errors,
    )


def _validate_gate(gate: Any, index: int, errors: list[str]) -> None:
    context = f"gates[{index}]"
    _expect_keys(
        gate,
        {
            "id",
            "category",
            "title",
            "release_blocking",
            "depends_on",
            "matrix",
            "assertions",
            "execution",
            "notes",
        },
        context,
        errors,
    )
    if not isinstance(gate, dict):
        return
    _expect(
        isinstance(gate.get("id"), str)
        and bool(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", gate["id"])),
        f"{context}.id is invalid",
        errors,
    )
    _expect(
        gate.get("category")
        in {
            "contract",
            "data",
            "functional",
            "diagnostics",
            "parity",
            "accuracy",
            "runner",
            "packaging",
            "integration",
        },
        f"{context}.category is unsupported",
        errors,
    )
    _expect(
        isinstance(gate.get("title"), str) and bool(gate.get("title", "").strip()),
        f"{context}.title must be non-empty",
        errors,
    )
    _expect(
        isinstance(gate.get("release_blocking"), bool),
        f"{context}.release_blocking must be boolean",
        errors,
    )
    depends_on = gate.get("depends_on")
    _expect(
        isinstance(depends_on, list), f"{context}.depends_on must be an array", errors
    )
    if isinstance(depends_on, list):
        _expect(
            all(ISSUE_RE.fullmatch(str(item)) for item in depends_on),
            f"{context}.depends_on must contain Linear issue IDs",
            errors,
        )

    matrix = gate.get("matrix")
    _validate_matrix(matrix, f"{context}.matrix", errors)
    assertions = gate.get("assertions")
    _expect(
        isinstance(assertions, list) and assertions,
        f"{context}.assertions must be non-empty",
        errors,
    )
    if isinstance(matrix, dict) and isinstance(assertions, list):
        for assertion_index, assertion in enumerate(assertions):
            _validate_assertion(
                assertion, matrix, f"{context}.assertions[{assertion_index}]", errors
            )

    execution = gate.get("execution")
    _expect_keys(
        execution,
        {"kind", "status", "command", "command_status", "evidence", "blocked_by"},
        f"{context}.execution",
        errors,
    )
    if not isinstance(execution, dict):
        return
    kind = execution.get("kind")
    status = execution.get("status")
    command = execution.get("command")
    command_status = execution.get("command_status")
    evidence = execution.get("evidence")
    blocked_by = execution.get("blocked_by")
    _expect(
        kind in {"automated", "manual"},
        f"{context}.execution.kind is unsupported",
        errors,
    )
    _expect(status in STATUSES, f"{context}.execution.status is unsupported", errors)
    if kind == "automated":
        _expect(
            isinstance(command, str) and bool(command.strip()),
            f"{context}.execution.command is required",
            errors,
        )
        _expect(
            command_status in {"available", "planned"},
            f"{context}.execution.command_status must be available or planned",
            errors,
        )
    elif kind == "manual":
        _expect(
            command is None,
            f"{context}.execution.command must be null for manual evidence",
            errors,
        )
        _expect(
            command_status == "not_applicable",
            f"{context}.execution.command_status must be not_applicable for manual evidence",
            errors,
        )
    _expect(
        isinstance(evidence, list),
        f"{context}.execution.evidence must be an array",
        errors,
    )
    if isinstance(evidence, list):
        for evidence_index, item in enumerate(evidence):
            _validate_evidence(
                item, f"{context}.execution.evidence[{evidence_index}]", errors
            )
        if status in {"passed", "failed"}:
            _expect(
                bool(evidence), f"{context} cannot be {status} without evidence", errors
            )
        if status == "passed":
            _expect(
                command_status != "planned",
                f"{context} cannot pass while its command is only planned",
                errors,
            )
        if status in {"pending", "blocked"}:
            _expect(
                not evidence, f"{context} cannot retain evidence while {status}", errors
            )
        if status == "passed":
            _expect(
                all(
                    item.get("result") == "pass"
                    for item in evidence
                    if isinstance(item, dict)
                ),
                f"{context} passed with non-passing evidence",
                errors,
            )
        if status == "failed":
            _expect(
                any(
                    item.get("result") == "fail"
                    for item in evidence
                    if isinstance(item, dict)
                ),
                f"{context} failed without failing evidence",
                errors,
            )
    _expect(
        isinstance(blocked_by, list),
        f"{context}.execution.blocked_by must be an array",
        errors,
    )
    if status == "blocked":
        _expect(
            bool(blocked_by), f"{context} is blocked without naming a blocker", errors
        )
    elif isinstance(blocked_by, list):
        _expect(not blocked_by, f"{context} names blockers but is not blocked", errors)
    _expect(
        isinstance(gate.get("notes"), str) and bool(gate.get("notes", "").strip()),
        f"{context}.notes must be non-empty",
        errors,
    )


def _gate_by_id(
    document: dict[str, Any], gate_id: str, errors: list[str]
) -> dict[str, Any]:
    gate = next(
        (
            item
            for item in document.get("gates", [])
            if isinstance(item, dict) and item.get("id") == gate_id
        ),
        None,
    )
    _expect(gate is not None, f"required gate {gate_id!r} is missing", errors)
    return gate or {}


def _expect_dimension(
    gate: dict[str, Any], name: str, values: set[str], errors: list[str]
) -> None:
    matrix = gate.get("matrix")
    raw = matrix.get(name, []) if isinstance(matrix, dict) else []
    actual = (
        set(raw)
        if isinstance(raw, list) and all(isinstance(item, str) for item in raw)
        else set()
    )
    _expect(
        actual == values,
        f"gate {gate.get('id')!r} must cover {name}={sorted(values)}, got {sorted(actual)}",
        errors,
    )


def _expect_gate_contract(
    gate: dict[str, Any],
    *,
    release_blocking: bool,
    matrix: dict[str, set[str]],
    assertions: list[dict[str, Any]],
    errors: list[str],
) -> None:
    gate_id = gate.get("id")
    _expect(
        gate.get("release_blocking") is release_blocking,
        f"gate {gate_id!r} release_blocking must be {release_blocking}",
        errors,
    )
    for name, values in matrix.items():
        _expect_dimension(gate, name, values, errors)
    _expect(
        gate.get("assertions") == assertions,
        f"gate {gate_id!r} assertions do not match the 0.13 qualification contract",
        errors,
    )


def _validate_required_coverage(document: dict[str, Any], errors: list[str]) -> None:
    modeled = _gate_by_id(document, "modeled-power-matrix", errors)
    _expect_gate_contract(
        modeled,
        release_blocking=True,
        matrix={
            "model_families": {"dense", "moe"},
            "deployments": {"aggregated", "disaggregated"},
            "runners": {"standard", "dynamo"},
            "timing_backends": {"default_op_level"},
            "coverage_states": {"covered", "undercovered"},
        },
        assertions=[
            {
                "when": {"coverage_states": ["covered"]},
                "expectation": "modeled_power",
                "thresholds": {
                    "power_coverage_ratio_minimum": 0.9,
                    "power_w_minimum_exclusive": 0,
                },
            },
            {
                "when": {"coverage_states": ["undercovered"]},
                "expectation": "unavailable_power",
                "thresholds": {"power_coverage_ratio_maximum_exclusive": 0.9},
            },
            {
                "when": {"runners": ["standard", "dynamo"]},
                "expectation": "runner_output",
                "thresholds": {
                    "required_fields": [
                        "power_w",
                        "power_coverage",
                        "power_provenance",
                    ]
                },
            },
        ],
        errors=errors,
    )

    unsupported = _gate_by_id(document, "unsupported-timing-no-fabrication", errors)
    _expect_gate_contract(
        unsupported,
        release_blocking=True,
        matrix={
            "model_families": {"dense", "moe"},
            "deployments": {"aggregated", "disaggregated"},
            "runners": {"standard", "dynamo"},
            "timing_backends": {"default_fpm", "fixed", "polynomial"},
            "coverage_states": {"not_applicable"},
        },
        assertions=[
            {
                "when": {"timing_backends": ["default_fpm", "fixed", "polynomial"]},
                "expectation": "unavailable_power",
                "thresholds": {
                    "forbid_numeric_power_w": True,
                    "forbid_measured_provenance": True,
                },
            }
        ],
        errors=errors,
    )

    operation = _gate_by_id(document, "operation-energy-evidence", errors)
    _expect_gate_contract(
        operation,
        release_blocking=True,
        matrix={
            "model_families": {"dense", "moe"},
            "deployments": {"aggregated", "disaggregated"},
            "runners": {"standard", "dynamo"},
            "timing_backends": {"default_op_level"},
            "coverage_states": {"covered", "undercovered"},
        },
        assertions=[
            {
                "when": {"coverage_states": ["covered", "undercovered"]},
                "expectation": "operation_energy_evidence",
                "thresholds": {
                    "required_fields": [
                        "phase",
                        "operation",
                        "latency_ms",
                        "energy_w_ms",
                        "covered_latency_ms",
                        "source",
                    ],
                    "reconciliation_relative_tolerance": 0.01,
                },
            }
        ],
        errors=errors,
    )

    data = _gate_by_id(document, "power-data-invariants", errors)
    _expect_gate_contract(
        data,
        release_blocking=True,
        matrix={
            "model_families": {"dense", "moe"},
            "deployments": {"any"},
            "runners": {"any"},
            "timing_backends": {"default_op_level"},
            "coverage_states": {"covered", "undercovered"},
        },
        assertions=[
            {
                "when": {},
                "expectation": "data_invariants",
                "thresholds": {
                    "finite_nonnegative_values": True,
                    "power_and_limit_columns_paired": True,
                },
            }
        ],
        errors=errors,
    )

    parity = _gate_by_id(document, "aic-power-parity", errors)
    _expect_gate_contract(
        parity,
        release_blocking=True,
        matrix={
            "model_families": {"dense", "moe"},
            "deployments": {"aggregated", "disaggregated"},
            "runners": {"standard"},
            "timing_backends": {"default_op_level"},
            "coverage_states": {"covered"},
        },
        assertions=[
            {
                "when": {},
                "expectation": "aic_parity",
                "thresholds": {
                    "maximum_relative_error_ratio": 0.01,
                    "reference_must_be_immutable": True,
                },
            }
        ],
        errors=errors,
    )

    accuracy = _gate_by_id(document, "silicon-power-accuracy", errors)
    policy = document.get("policy")
    accuracy_policy = (
        policy.get("silicon_accuracy", {}) if isinstance(policy, dict) else {}
    )
    _expect_gate_contract(
        accuracy,
        release_blocking=True,
        matrix={
            "model_families": {"dense", "moe"},
            "deployments": {"aggregated", "disaggregated"},
            "runners": {"standard", "dynamo"},
            "timing_backends": {"default_op_level"},
            "coverage_states": {"covered"},
        },
        assertions=[
            {
                "when": {},
                "expectation": "silicon_accuracy",
                "thresholds": accuracy_policy,
            }
        ],
        errors=errors,
    )

    package = _gate_by_id(document, "application-wheel", errors)
    _expect_gate_contract(
        package,
        release_blocking=True,
        matrix={
            "model_families": {"any"},
            "deployments": {"any"},
            "runners": {"standard", "dynamo"},
            "timing_backends": {"any"},
            "coverage_states": {"not_applicable"},
        },
        assertions=[
            {
                "when": {},
                "expectation": "packaged_output",
                "thresholds": {
                    "distribution_count": 1,
                    "wheel_smoke_required": True,
                    "packaged_data_required": True,
                },
            }
        ],
        errors=errors,
    )

    for integration, deployment in (
        ("afd-power-integration", "afd"),
        ("epd-power-integration", "epd"),
    ):
        gate = _gate_by_id(document, integration, errors)
        _expect_gate_contract(
            gate,
            release_blocking=False,
            matrix={
                "model_families": {"dense", "moe"},
                "deployments": {deployment},
                "runners": {"standard", "dynamo"},
                "timing_backends": {"default_op_level"},
                "coverage_states": {"covered", "undercovered"},
            },
            assertions=[
                {
                    "when": {},
                    "expectation": "integration_available",
                    "thresholds": {"required": False},
                }
            ],
            errors=errors,
        )
        _expect(
            gate.get("execution", {}).get("status") == "blocked",
            f"{integration} must remain blocked until its integration exists",
            errors,
        )


def validate_document(
    document: Any,
    *,
    require_release_ready: bool = False,
    expected_revision: str | None = None,
) -> None:
    errors: list[str] = []
    _expect_keys(
        document,
        {
            "$schema",
            "schema_version",
            "release_target",
            "release_state",
            "candidate_revision",
            "policy",
            "gates",
        },
        "document",
        errors,
    )
    if not isinstance(document, dict):
        raise QualificationError("\n".join(errors))
    _expect(
        document.get("$schema") == "./qualification-matrix.schema.json",
        "$schema must be './qualification-matrix.schema.json'",
        errors,
    )
    _expect(
        document.get("schema_version") == "1.0", "schema_version must be '1.0'", errors
    )
    _expect(
        document.get("release_target") == "0.13.0",
        "release_target must be '0.13.0'",
        errors,
    )
    _expect(
        document.get("release_state") in {"not_qualified", "qualified"},
        "release_state is unsupported",
        errors,
    )
    candidate_revision = document.get("candidate_revision")
    _expect(
        candidate_revision is None
        or (
            isinstance(candidate_revision, str)
            and bool(SHA_RE.fullmatch(candidate_revision))
        ),
        "candidate_revision must be null or a 40-character commit SHA",
        errors,
    )
    _expect(
        expected_revision is None
        or (
            isinstance(expected_revision, str)
            and bool(SHA_RE.fullmatch(expected_revision))
        ),
        "expected_revision must be a 40-character commit SHA",
        errors,
    )

    policy = document.get("policy")
    _expect_keys(
        policy,
        {
            "minimum_power_coverage_ratio",
            "aic_parity_relative_tolerance",
            "silicon_accuracy",
        },
        "policy",
        errors,
    )
    if isinstance(policy, dict):
        _expect(
            policy.get("minimum_power_coverage_ratio") == 0.9,
            "minimum power coverage must remain 0.9",
            errors,
        )
        _expect(
            policy.get("aic_parity_relative_tolerance") == 0.01,
            "AIC parity tolerance must remain 0.01",
            errors,
        )
        accuracy_policy = policy.get("silicon_accuracy")
        _expect_keys(
            accuracy_policy,
            {
                "metric",
                "maximum_error_pct",
                "minimum_points_per_case",
                "threshold_status",
            },
            "policy.silicon_accuracy",
            errors,
        )
        if isinstance(accuracy_policy, dict):
            _expect(
                accuracy_policy.get("metric") == "power_w_mape_pct",
                "silicon accuracy metric must be power_w_mape_pct",
                errors,
            )
            _expect(
                accuracy_policy.get("threshold_status")
                in {"pending_approval", "approved"},
                "silicon threshold status is unsupported",
                errors,
            )
            if accuracy_policy.get("threshold_status") == "pending_approval":
                _expect(
                    accuracy_policy.get("maximum_error_pct") is None,
                    "unapproved silicon threshold must not carry a value",
                    errors,
                )
                _expect(
                    accuracy_policy.get("minimum_points_per_case") is None,
                    "unapproved sample threshold must not carry a value",
                    errors,
                )
            else:
                maximum = accuracy_policy.get("maximum_error_pct")
                minimum = accuracy_policy.get("minimum_points_per_case")
                _expect(
                    isinstance(maximum, (int, float))
                    and not isinstance(maximum, bool)
                    and 0 < maximum < 100,
                    "approved silicon maximum error must be between 0 and 100",
                    errors,
                )
                _expect(
                    isinstance(minimum, int)
                    and not isinstance(minimum, bool)
                    and minimum > 0,
                    "approved silicon minimum points must be positive",
                    errors,
                )

    gates = document.get("gates")
    _expect(
        isinstance(gates, list) and gates, "gates must be a non-empty array", errors
    )
    if isinstance(gates, list):
        for index, gate in enumerate(gates):
            _validate_gate(gate, index, errors)
        ids = [gate.get("id") for gate in gates if isinstance(gate, dict)]
        if all(isinstance(gate_id, str) for gate_id in ids):
            _expect(len(ids) == len(set(ids)), "gate ids must be unique", errors)
        _validate_required_coverage(document, errors)

    if require_release_ready or document.get("release_state") == "qualified":
        _expect(
            expected_revision is not None,
            "release qualification requires an explicit expected_revision",
            errors,
        )
        _expect(
            bool(SHA_RE.fullmatch(str(candidate_revision or ""))),
            "qualified release requires a candidate_revision",
            errors,
        )
        if expected_revision is not None and candidate_revision is not None:
            _expect(
                candidate_revision == expected_revision,
                "candidate_revision does not match expected_revision",
                errors,
            )
        if isinstance(policy, dict):
            accuracy_policy = policy.get("silicon_accuracy", {})
            _expect(
                accuracy_policy.get("threshold_status") == "approved",
                "silicon accuracy threshold is not approved",
                errors,
            )
        if isinstance(gates, list):
            unfinished = [
                gate.get("id")
                for gate in gates
                if isinstance(gate, dict)
                and gate.get("release_blocking")
                and gate.get("execution", {}).get("status") != "passed"
            ]
            _expect(
                not unfinished,
                f"release-blocking gates are not passed: {unfinished}",
                errors,
            )
            for gate in gates:
                if (
                    not isinstance(gate, dict)
                    or not gate.get("release_blocking")
                    or gate.get("execution", {}).get("status") != "passed"
                ):
                    continue
                evidence = gate.get("execution", {}).get("evidence", [])
                mismatched = [
                    item.get("source_revision")
                    for item in evidence
                    if isinstance(item, dict)
                    and item.get("source_revision") != candidate_revision
                ]
                _expect(
                    not mismatched,
                    f"gate {gate.get('id')!r} evidence does not match candidate_revision: {mismatched}",
                    errors,
                )
        _expect(
            document.get("release_state") == "qualified",
            "release_state must be 'qualified'",
            errors,
        )

    if errors:
        raise QualificationError("\n".join(f"- {error}" for error in errors))


def _reject_json_constant(constant: str) -> None:
    raise QualificationError(f"ledger contains non-JSON constant {constant!r}")


def load_and_validate(
    path: Path,
    *,
    require_release_ready: bool = False,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    document = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    validate_document(
        document,
        require_release_ready=require_release_ready,
        expected_revision=expected_revision,
    )
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", nargs="?", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--require-release-ready", action="store_true")
    parser.add_argument(
        "--expected-revision",
        help="exact candidate commit required by --require-release-ready",
    )
    args = parser.parse_args(argv)
    try:
        document = load_and_validate(
            args.ledger,
            require_release_ready=args.require_release_ready,
            expected_revision=args.expected_revision,
        )
    except (OSError, json.JSONDecodeError, QualificationError) as error:
        print(f"power qualification failed: {error}", file=sys.stderr)
        return 1
    print(
        f"power qualification ledger is valid: release={document['release_target']} "
        f"state={document['release_state']} gates={len(document['gates'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
