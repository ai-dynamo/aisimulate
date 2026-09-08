# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_power_qualification.py"
LEDGER = ROOT / "docs" / "power" / "qualification-matrix.json"
SCHEMA = ROOT / "docs" / "power" / "qualification-matrix.schema.json"
SPEC = importlib.util.spec_from_file_location("validate_power_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUALIFICATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFICATION)


def _document() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def _gate(document: dict, gate_id: str) -> dict:
    return next(gate for gate in document["gates"] if gate["id"] == gate_id)


def test_checked_in_power_qualification_ledger_is_structurally_valid() -> None:
    document = QUALIFICATION.load_and_validate(LEDGER)

    assert document["release_state"] == "not_qualified"
    assert QUALIFICATION.main([]) == 0


def test_schema_is_versioned_and_machine_readable() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"]["const"] == "1.0"
    assert schema["properties"]["release_target"]["const"] == "0.13.0"
    assert {"gate", "matrix", "assertion", "execution", "evidence"} <= set(schema["$defs"])


def test_release_check_fails_closed_on_pending_evidence_and_threshold() -> None:
    with pytest.raises(QUALIFICATION.QualificationError) as failure:
        QUALIFICATION.validate_document(_document(), require_release_ready=True)

    message = str(failure.value)
    assert "silicon accuracy threshold is not approved" in message
    assert "modeled-power-matrix" in message
    assert "release_state must be 'qualified'" in message


def test_qualified_state_cannot_bypass_release_check() -> None:
    document = _document()
    document["release_state"] = "qualified"

    with pytest.raises(QUALIFICATION.QualificationError, match="release-blocking gates"):
        QUALIFICATION.validate_document(document)


def test_matrix_must_keep_all_runner_model_and_deployment_dimensions() -> None:
    document = _document()
    _gate(document, "modeled-power-matrix")["matrix"]["runners"] = ["standard"]

    with pytest.raises(QUALIFICATION.QualificationError, match="dynamo.*standard"):
        QUALIFICATION.validate_document(document)


def test_unsupported_timing_backends_must_remain_unavailable() -> None:
    document = _document()
    assertion = _gate(document, "unsupported-timing-no-fabrication")["assertions"][0]
    assertion["expectation"] = "modeled_power"

    with pytest.raises(QUALIFICATION.QualificationError, match="must require unavailable power"):
        QUALIFICATION.validate_document(document)


def test_passed_gate_requires_immutable_passing_evidence() -> None:
    document = _document()
    gate = _gate(document, "power-data-invariants")
    gate["execution"]["status"] = "passed"

    with pytest.raises(QUALIFICATION.QualificationError, match="cannot be passed without evidence"):
        QUALIFICATION.validate_document(document)

    gate["execution"]["evidence"] = [
        {
            "result": "pass",
            "artifact": "https://example.invalid/power-data-invariants.json",
            "sha256": "1" * 64,
            "source_revision": "2" * 40,
            "recorded_at": "2026-09-08T12:00:00Z",
        }
    ]
    QUALIFICATION.validate_document(document)


def test_silicon_gate_and_policy_thresholds_cannot_drift() -> None:
    document = _document()
    policy = document["policy"]["silicon_accuracy"]
    policy.update(
        {
            "maximum_error_pct": 10.0,
            "minimum_points_per_case": 3,
            "threshold_status": "approved",
        }
    )

    with pytest.raises(QUALIFICATION.QualificationError, match="thresholds must match"):
        QUALIFICATION.validate_document(document)

    _gate(document, "silicon-power-accuracy")["assertions"][0]["thresholds"] = copy.deepcopy(policy)
    QUALIFICATION.validate_document(document)


def test_afd_and_epd_remain_explicitly_blocked_not_silently_dropped() -> None:
    document = _document()
    afd = _gate(document, "afd-power-integration")
    afd["execution"]["status"] = "pending"
    afd["execution"]["blocked_by"] = []

    with pytest.raises(QUALIFICATION.QualificationError, match="afd-power-integration must remain blocked"):
        QUALIFICATION.validate_document(document)
