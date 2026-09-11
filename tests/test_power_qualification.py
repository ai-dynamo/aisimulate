# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_power_qualification.py"
LEDGER = ROOT / "docs" / "power" / "qualification-matrix.json"
SCHEMA = ROOT / "docs" / "power" / "qualification-matrix.schema.json"
TEST_ARTIFACT_REF = "docs/power/evidence/power-data-invariants-bb31ab4.json"
TEST_ARTIFACT = ROOT / TEST_ARTIFACT_REF
SPEC = importlib.util.spec_from_file_location("validate_power_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUALIFICATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFICATION)


def _document() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def _gate(document: dict, gate_id: str) -> dict:
    return next(gate for gate in document["gates"] if gate["id"] == gate_id)


def _passing_evidence(revision: str, artifact: str = TEST_ARTIFACT_REF) -> dict:
    return {
        "result": "pass",
        "artifact": artifact,
        "sha256": hashlib.sha256((QUALIFICATION.ROOT / artifact).read_bytes()).hexdigest(),
        "source_revision": revision,
        "recorded_at": "2026-09-08T12:00:00Z",
    }


def _qualified_document(revision: str, artifact_root: Path) -> dict:
    document = _document()
    document["candidate_revision"] = revision
    document["release_state"] = "qualified"
    policy = document["policy"]["silicon_accuracy"]
    policy.update(
        {
            "maximum_error_pct": 10.0,
            "minimum_points_per_case": 3,
            "threshold_status": "approved",
        }
    )
    _gate(document, "silicon-power-accuracy")["assertions"][0]["thresholds"] = copy.deepcopy(policy)
    for gate in document["gates"]:
        if not gate["release_blocking"]:
            continue
        gate["execution"]["status"] = "passed"
        if gate["execution"]["command_status"] == "planned":
            gate["execution"]["command_status"] = "available"
        if gate["id"] == "power-data-invariants":
            report = json.loads(TEST_ARTIFACT.read_text(encoding="utf-8"))
            report["source_revision"] = revision
        else:
            report = {
                "schema_version": "1.0",
                "gate_id": gate["id"],
                "source_revision": revision,
                "matrix": copy.deepcopy(gate["matrix"]),
                "assertion_results": [
                    {**copy.deepcopy(assertion), "result": "pass"} for assertion in gate["assertions"]
                ],
                "units": QUALIFICATION._expected_units(gate),
                "anomalies": [],
                "details": {"test_fixture": True},
            }
        artifact_ref = f"artifacts/{gate['id']}.json"
        artifact_path = artifact_root / artifact_ref
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(report), encoding="utf-8")
        gate["execution"]["evidence"] = [_passing_evidence(revision, artifact_ref)]
    return document


def _rewrite_evidence_report(document: dict, gate_id: str, update: Callable[[dict], None]) -> None:
    evidence = _gate(document, gate_id)["execution"]["evidence"][0]
    artifact = QUALIFICATION.ROOT / evidence["artifact"]
    report = json.loads(artifact.read_text(encoding="utf-8"))
    update(report)
    artifact.write_text(json.dumps(report), encoding="utf-8")
    evidence["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()


@pytest.fixture
def qualified_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str], dict]:
    artifact_root = tmp_path / "repository"
    artifact_root.mkdir()
    monkeypatch.setattr(QUALIFICATION, "ROOT", artifact_root)
    return lambda revision: _qualified_document(revision, artifact_root)


def test_checked_in_power_qualification_ledger_is_structurally_valid() -> None:
    document = QUALIFICATION.load_and_validate(LEDGER)
    invariant = _gate(document, "power-data-invariants")
    evidence = invariant["execution"]["evidence"][0]
    artifact = ROOT / evidence["artifact"]

    assert document["release_state"] == "not_qualified"
    assert invariant["execution"]["status"] == "passed"
    assert evidence["result"] == "pass"
    assert artifact.is_file()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == evidence["sha256"]
    assert QUALIFICATION.main([]) == 0


def test_power_data_invariant_evidence_is_complete_and_machine_readable() -> None:
    document = _document()
    evidence = _gate(document, "power-data-invariants")["execution"]["evidence"][0]
    with (ROOT / evidence["artifact"]).open(encoding="utf-8") as artifact_file:
        result = json.load(artifact_file)

    assert result["schema_version"] == "1.0"
    assert result["gate_id"] == "power-data-invariants"
    assert result["source_revision"] == evidence["source_revision"]
    assert result["anomalies"] == []
    details = result["details"]
    assert details["discovery_root"] == "python/aisimulate/src/aiconfigurator_core/systems/data"
    assert details["total_discovered_parquet_count"] == 762
    assert details["skipped_file_anomalies"] == []
    assert details["files_scanned"] == len(details["files"]) == 3
    assert details["files_without_power_columns"] == 759
    assert details["rows_scanned"] == sum(file_result["rows_scanned"] for file_result in details["files"])
    assert details["rows_scanned"] == 122362
    assert details["checks"]["power_and_limit_columns_paired"] == {
        "result": "pass",
        "files_missing_pair": [],
    }
    assert details["checks"]["finite_nonnegative_values"] == {
        "result": "pass",
        "power": {
            "unit": "W",
            "nan_count": 0,
            "positive_infinity_count": 0,
            "negative_infinity_count": 0,
            "negative_count": 0,
        },
        "power_limit": {
            "unit": "W",
            "nan_count": 0,
            "positive_infinity_count": 0,
            "negative_infinity_count": 0,
            "non_positive_count": 0,
        },
    }
    assert all(
        file_result["power_columns_present"] == ["power", "power_limit"]
        and file_result["pairing_result"] == "pass"
        and file_result["key_columns"]
        and file_result["power"]["unit"] == "W"
        and file_result["power_limit"]["unit"] == "W"
        and file_result["power"]["positive_infinity_count"] == 0
        and file_result["power"]["negative_infinity_count"] == 0
        and file_result["power_limit"]["positive_infinity_count"] == 0
        and file_result["power_limit"]["negative_infinity_count"] == 0
        for file_result in details["files"]
    )


def test_schema_is_versioned_and_machine_readable() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"]["const"] == "1.0"
    assert schema["properties"]["release_target"]["const"] == "0.13.0"
    assert {"gate", "matrix", "assertion", "execution", "evidence"} <= set(schema["$defs"])

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(_document())


def test_schema_rejects_unsupported_dimensions() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    document = _document()
    _gate(document, "modeled-power-matrix")["matrix"]["runners"] = ["bogus-runner"]

    errors = list(Draft202012Validator(schema).iter_errors(document))

    assert any(error.validator == "enum" and "bogus-runner" in error.message for error in errors)


def test_schema_rejects_passing_planned_or_evidence_free_gates() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    revision = "a" * 40

    planned = _document()
    planned_gate = _gate(planned, "modeled-power-matrix")
    planned_gate["execution"]["status"] = "passed"
    planned_gate["execution"]["evidence"] = [_passing_evidence(revision)]
    assert list(validator.iter_errors(planned))

    evidence_free = _document()
    evidence_free_gate = _gate(evidence_free, "power-data-invariants")
    evidence_free_gate["execution"]["status"] = "passed"
    evidence_free_gate["execution"]["evidence"] = []
    assert list(validator.iter_errors(evidence_free))


def test_dependency_free_validator_rejects_schema_identity_drift() -> None:
    document = _document()
    document["$schema"] = "https://example.invalid/other-schema.json"

    with pytest.raises(QUALIFICATION.QualificationError, match=r"\$schema must be"):
        QUALIFICATION.validate_document(document)


def test_release_check_fails_closed_on_pending_evidence_and_threshold() -> None:
    with pytest.raises(QUALIFICATION.QualificationError) as failure:
        QUALIFICATION.validate_document(_document(), require_release_ready=True)

    message = str(failure.value)
    assert "silicon accuracy threshold is not approved" in message
    assert "modeled-power-matrix" in message
    assert "release_state must be 'qualified'" in message


def test_release_check_accepts_one_fully_qualified_candidate_revision(
    qualified_document: Callable[[str], dict],
) -> None:
    revision = "a" * 40

    QUALIFICATION.validate_document(
        qualified_document(revision),
        require_release_ready=True,
        expected_revision=revision,
    )


def test_release_check_rejects_remote_artifacts_without_network_access(
    qualified_document: Callable[[str], dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    document = qualified_document(revision)
    artifact_url = "https://artifacts.example/power-matrix.json"
    evidence = _gate(document, "modeled-power-matrix")["execution"]["evidence"][0]
    evidence["artifact"] = artifact_url
    network_called = False

    def unexpected_network_call(*_args: object, **_kwargs: object) -> None:
        nonlocal network_called
        network_called = True

    monkeypatch.setattr("socket.create_connection", unexpected_network_call)

    with pytest.raises(QUALIFICATION.QualificationError, match="repository-relative path"):
        QUALIFICATION.validate_document(
            document,
            require_release_ready=True,
            expected_revision=revision,
        )

    assert not network_called


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (lambda report: report.update(gate_id="application-wheel"), "gate_id"),
        (lambda report: report["matrix"].update(runners=["standard"]), "matrix"),
        (
            lambda report: report["assertion_results"][0].update(result="fail"),
            "every gate assertion",
        ),
        (lambda report: report.update(units={"power_w": "kW"}), "units"),
        (lambda report: report["anomalies"].append("unexpected warning"), "anomalies"),
        (lambda report: report.update(source_revision="b" * 40), "source_revision"),
    ],
)
def test_release_check_binds_evidence_to_owning_gate(
    update: Callable[[dict], None],
    message: str,
    qualified_document: Callable[[str], dict],
) -> None:
    revision = "a" * 40
    document = qualified_document(revision)
    _rewrite_evidence_report(document, "modeled-power-matrix", update)

    with pytest.raises(QUALIFICATION.QualificationError, match=message):
        QUALIFICATION.validate_document(
            document,
            require_release_ready=True,
            expected_revision=revision,
        )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (
            lambda report: report["details"]["checks"]["finite_nonnegative_values"]["power"].update(
                positive_infinity_count=1
            ),
            "positive_infinity_count",
        ),
        (
            lambda report: report["details"].update(total_discovered_parquet_count=761),
            "discovery counts",
        ),
        (
            lambda report: report["details"]["skipped_file_anomalies"].append("unreadable.parquet"),
            "skipped_file_anomalies",
        ),
    ],
)
def test_data_invariant_report_rejects_incomplete_discovery_or_finiteness(
    update: Callable[[dict], None],
    message: str,
    qualified_document: Callable[[str], dict],
) -> None:
    revision = "a" * 40
    document = qualified_document(revision)
    _rewrite_evidence_report(document, "power-data-invariants", update)

    with pytest.raises(QUALIFICATION.QualificationError, match=message):
        QUALIFICATION.validate_document(
            document,
            require_release_ready=True,
            expected_revision=revision,
        )


def test_release_check_bounds_local_artifact_size(
    qualified_document: Callable[[str], dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    document = qualified_document(revision)
    monkeypatch.setattr(QUALIFICATION, "ARTIFACT_MAX_BYTES", 1)

    with pytest.raises(QUALIFICATION.QualificationError, match="exceeds the 1-byte limit"):
        QUALIFICATION.validate_document(
            document,
            require_release_ready=True,
            expected_revision=revision,
        )


@pytest.mark.parametrize(
    ("artifact", "digest", "message"),
    [
        ("artifacts/modeled-power-matrix.json", "0" * 64, "SHA-256 mismatch"),
        ("docs/power/evidence/missing.json", "0" * 64, "artifact is unavailable"),
    ],
)
def test_release_check_rejects_unverifiable_artifacts(
    artifact: str,
    digest: str,
    message: str,
    qualified_document: Callable[[str], dict],
) -> None:
    revision = "a" * 40
    document = qualified_document(revision)
    evidence = _gate(document, "modeled-power-matrix")["execution"]["evidence"][0]
    evidence["artifact"] = artifact
    evidence["sha256"] = digest

    with pytest.raises(QUALIFICATION.QualificationError, match=message):
        QUALIFICATION.validate_document(
            document,
            require_release_ready=True,
            expected_revision=revision,
        )


def test_cli_release_gate_fails_closed_and_checks_expected_revision(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    qualified_document: Callable[[str], dict],
) -> None:
    assert QUALIFICATION.main(["--require-release-ready"]) == 1
    assert "power qualification failed" in capsys.readouterr().err

    revision = "a" * 40
    ledger = tmp_path / "qualification-matrix.json"
    ledger.write_text(json.dumps(qualified_document(revision)), encoding="utf-8")

    assert QUALIFICATION.main([str(ledger), "--require-release-ready", "--expected-revision", revision]) == 0
    assert "state=qualified" in capsys.readouterr().out

    assert (
        QUALIFICATION.main(
            [
                str(ledger),
                "--require-release-ready",
                "--expected-revision",
                "b" * 40,
            ]
        )
        == 1
    )
    assert "does not match expected_revision" in capsys.readouterr().err


def test_release_check_rejects_stale_or_mixed_candidate_evidence(
    qualified_document: Callable[[str], dict],
) -> None:
    revision = "a" * 40
    document = qualified_document(revision)

    with pytest.raises(QUALIFICATION.QualificationError, match="does not match expected_revision"):
        QUALIFICATION.validate_document(
            document,
            require_release_ready=True,
            expected_revision="b" * 40,
        )

    _gate(document, "application-wheel")["execution"]["evidence"][0]["source_revision"] = "c" * 40
    with pytest.raises(
        QUALIFICATION.QualificationError,
        match="evidence does not match candidate_revision",
    ):
        QUALIFICATION.validate_document(
            document,
            require_release_ready=True,
            expected_revision=revision,
        )

    with pytest.raises(QUALIFICATION.QualificationError, match="expected_revision"):
        QUALIFICATION.validate_document(
            document,
            require_release_ready=True,
            expected_revision=123,  # type: ignore[arg-type]
        )

    document["candidate_revision"] = int("1" * 40)
    with pytest.raises(QUALIFICATION.QualificationError, match="candidate_revision"):
        QUALIFICATION.validate_document(document)


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

    with pytest.raises(QUALIFICATION.QualificationError, match="qualification contract"):
        QUALIFICATION.validate_document(document)


def test_passed_gate_requires_immutable_passing_evidence() -> None:
    document = _document()
    gate = _gate(document, "power-data-invariants")
    passing_evidence = copy.deepcopy(gate["execution"]["evidence"])
    gate["execution"]["status"] = "passed"
    gate["execution"]["evidence"] = []

    with pytest.raises(QUALIFICATION.QualificationError, match="cannot be passed without evidence"):
        QUALIFICATION.validate_document(document)

    gate["execution"]["evidence"] = passing_evidence
    QUALIFICATION.validate_document(document)

    gate["execution"]["evidence"][0]["sha256"] = int("1" * 64)
    with pytest.raises(QUALIFICATION.QualificationError, match="sha256"):
        QUALIFICATION.validate_document(document)

    gate["execution"]["evidence"][0]["sha256"] = "1" * 64
    gate["execution"]["evidence"][0]["artifact"] = "http://example.invalid/power-data-invariants.json"
    with pytest.raises(QUALIFICATION.QualificationError, match="repository-relative"):
        QUALIFICATION.validate_document(document)


def test_loader_rejects_non_json_constants(tmp_path: Path) -> None:
    ledger = tmp_path / "qualification-matrix.json"
    ledger.write_text(
        LEDGER.read_text(encoding="utf-8").replace(
            '"minimum_power_coverage_ratio": 0.9',
            '"minimum_power_coverage_ratio": NaN',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(QUALIFICATION.QualificationError, match="non-JSON constant"):
        QUALIFICATION.load_and_validate(ledger)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_additional_gate_thresholds_reject_nonfinite_numbers(value: float) -> None:
    document = _document()
    extra_gate = copy.deepcopy(document["gates"][0])
    extra_gate["id"] = "optional-extra-gate"
    extra_gate["release_blocking"] = False
    extra_gate["assertions"][0]["thresholds"] = {"optional_limit": value}
    extra_gate["execution"]["status"] = "pending"
    extra_gate["execution"]["evidence"] = []
    document["gates"].append(extra_gate)

    with pytest.raises(QUALIFICATION.QualificationError, match="non-finite"):
        QUALIFICATION.validate_document(document)


@pytest.mark.parametrize("gate_id", ["afd-power-integration", "application-wheel"])
def test_malformed_execution_accumulates_validation_errors(gate_id: str) -> None:
    document = _document()
    _gate(document, gate_id)["execution"] = []

    with pytest.raises(QUALIFICATION.QualificationError, match="execution must be an object"):
        QUALIFICATION.validate_document(
            document,
            require_release_ready=True,
            expected_revision="a" * 40,
        )


def test_malformed_accuracy_policy_accumulates_validation_errors() -> None:
    document = _document()
    document["policy"]["silicon_accuracy"] = []

    with pytest.raises(QUALIFICATION.QualificationError, match="silicon_accuracy must be an object"):
        QUALIFICATION.validate_document(
            document,
            require_release_ready=True,
            expected_revision="a" * 40,
        )


def test_planned_command_cannot_be_presented_as_passing() -> None:
    document = _document()
    gate = _gate(document, "modeled-power-matrix")
    gate["execution"]["status"] = "passed"
    gate["execution"]["evidence"] = [
        {
            "result": "pass",
            "artifact": "artifacts/power-matrix.json",
            "sha256": "1" * 64,
            "source_revision": "2" * 40,
            "recorded_at": "2026-09-08T12:00:00Z",
        }
    ]

    with pytest.raises(QUALIFICATION.QualificationError, match="command is only planned"):
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

    with pytest.raises(QUALIFICATION.QualificationError, match="qualification contract"):
        QUALIFICATION.validate_document(document)

    _gate(document, "silicon-power-accuracy")["assertions"][0]["thresholds"] = copy.deepcopy(policy)
    QUALIFICATION.validate_document(document)


def test_silicon_approval_rejects_boolean_thresholds() -> None:
    document = _document()
    policy = document["policy"]["silicon_accuracy"]
    policy.update(
        {
            "maximum_error_pct": True,
            "minimum_points_per_case": True,
            "threshold_status": "approved",
        }
    )
    _gate(document, "silicon-power-accuracy")["assertions"][0]["thresholds"] = copy.deepcopy(policy)

    with pytest.raises(QUALIFICATION.QualificationError, match="silicon"):
        QUALIFICATION.validate_document(document)


@pytest.mark.parametrize(
    ("gate_id", "path", "weakened_value"),
    [
        (
            "modeled-power-matrix",
            ("assertions", 0, "thresholds", "power_coverage_ratio_minimum"),
            0.5,
        ),
        (
            "unsupported-timing-no-fabrication",
            ("assertions", 0, "thresholds", "forbid_numeric_power_w"),
            False,
        ),
        (
            "operation-energy-evidence",
            ("assertions", 0, "thresholds", "reconciliation_relative_tolerance"),
            0.5,
        ),
        (
            "power-data-invariants",
            ("assertions", 0, "thresholds", "finite_nonnegative_values"),
            False,
        ),
        ("aic-power-parity", ("matrix", "runners"), ["dynamo"]),
        (
            "silicon-power-accuracy",
            ("matrix", "timing_backends"),
            ["default_fpm"],
        ),
        (
            "application-wheel",
            ("assertions", 0, "thresholds", "wheel_smoke_required"),
            False,
        ),
        (
            "afd-power-integration",
            ("assertions", 0, "thresholds", "required"),
            True,
        ),
    ],
)
def test_release_qualification_contracts_cannot_be_weakened(
    gate_id: str, path: tuple[str | int, ...], weakened_value: object
) -> None:
    document = _document()
    target: object = _gate(document, gate_id)
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = weakened_value  # type: ignore[index]

    with pytest.raises(QUALIFICATION.QualificationError):
        QUALIFICATION.validate_document(document)


def test_release_blocking_gates_cannot_be_downgraded() -> None:
    document = _document()
    _gate(document, "application-wheel")["release_blocking"] = False

    with pytest.raises(QUALIFICATION.QualificationError, match="release_blocking"):
        QUALIFICATION.validate_document(document)


def test_afd_and_epd_remain_explicitly_blocked_not_silently_dropped() -> None:
    document = _document()
    afd = _gate(document, "afd-power-integration")
    afd["execution"]["status"] = "pending"
    afd["execution"]["blocked_by"] = []

    with pytest.raises(
        QUALIFICATION.QualificationError,
        match="afd-power-integration must remain blocked",
    ):
        QUALIFICATION.validate_document(document)
