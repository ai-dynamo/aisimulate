# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from aisimulate.output import format_prediction_stdout
from aisimulate.replay.reporting import format_power_diagnostics


def _diagnostics() -> dict:
    return {
        "schema_version": "1.0",
        "scope": "active_forward_pass_per_gpu",
        "power_w_unit": "W",
        "energy_unit": "W-ms",
        "latency_unit": "ms",
        "coverage_gate": 0.9,
        "publication_status": "withheld",
        "power_coverage": 0.8,
        "phases": [
            {
                "name": "prefill",
                "energy_wms": 3200.0,
                "latency_ms": 10.0,
                "covered_latency_ms": 8.0,
                "power_coverage": 0.8,
                "publication_status": "withheld",
                "power_w": None,
                "source": "mixed",
                "source_kind": "mixed",
                "operations": [
                    {
                        "name": "attention",
                        "latency_ms": 2.0,
                        "covered_latency_ms": 0.0,
                        "power_coverage": 0.0,
                        "source": "empirical",
                        "source_kind": "measured",
                        "status": "missing",
                        "uncovered_reason": "timing provider returned no positive energy evidence",
                    },
                    {
                        "name": "gemm",
                        "energy_wms": 3200.0,
                        "latency_ms": 8.0,
                        "covered_latency_ms": 8.0,
                        "power_coverage": 1.0,
                        "energy_contribution": 1.0,
                        "source": "silicon",
                        "source_kind": "measured",
                        "status": "available",
                    },
                ],
            },
            {
                "name": "decode",
                "energy_wms": 1600.0,
                "latency_ms": 4.0,
                "covered_latency_ms": 4.0,
                "power_coverage": 1.0,
                "publication_status": "available",
                "power_w": 400.0,
                "source": "estimated",
                "source_kind": "modeled",
                "operations": [],
            },
        ],
    }


def test_power_diagnostics_table_is_bounded_and_energy_specific() -> None:
    rendered = format_power_diagnostics(_diagnostics(), top_n=1)

    assert "active forward-pass energy diagnostics (per GPU)" in rendered
    assert "power=N/A" in rendered
    assert "coverage=80.00%" in rendered
    assert "3,200.00 W-ms" in rendered
    assert "gemm" in rendered
    assert "attention" not in rendered
    assert "... 1 more in prediction.json" in rendered
    assert "available modeled:estimated" in rendered
    assert "withheld mixed:mixed" in rendered
    assert "wall-clock or provisioned-fleet energy" in rendered

    complete = format_power_diagnostics(_diagnostics(), top_n=2)
    assert "timing provider returned no positive energy evidence" in complete
    assert "missing measured:empirical" in complete


def test_power_diagnostics_json_stdout_keeps_complete_operation_evidence() -> None:
    output = format_prediction_stdout(
        {"power_coverage": 0.8},
        "json",
        power_diagnostics=_diagnostics(),
        diagnostics_top_n=1,
    )

    parsed = json.loads(output)
    operations = parsed["power_diagnostics"]["phases"][0]["operations"]
    assert [operation["name"] for operation in operations] == ["attention", "gemm"]
    assert "energy_wms" not in operations[0]
    assert operations[0]["status"] == "missing"


def test_unsupported_diagnostics_explain_missing_provider_evidence() -> None:
    rendered = format_power_diagnostics(
        {
            "publication_status": "unsupported",
            "coverage_gate": 0.9,
            "unavailable_reason": "typed timing-energy evidence is unavailable",
            "phases": [],
        }
    )

    assert "status=unsupported" in rendered
    assert "typed timing-energy evidence is unavailable" in rendered


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), float("-inf")])
def test_invalid_measurements_are_unavailable_and_sort_last(value):
    from aisimulate.replay.reporting import (
        _format_energy,
        _format_latency,
        _format_percent,
        _format_power,
        _operation_sort_key,
    )

    for formatter in (_format_energy, _format_latency, _format_percent, _format_power):
        assert formatter(value) == "N/A"
    ordered = sorted(
        [{"name": "invalid", "energy_wms": value}, {"name": "valid", "energy_wms": 1.0}], key=_operation_sort_key
    )
    assert [op["name"] for op in ordered] == ["valid", "invalid"]


def test_compatibility_diagnostics_remain_visible_with_other_details() -> None:
    from aisimulate.output import format_prediction_stdout

    diagnostics = {
        "publication_status": "unsupported",
        "power_w": None,
        "power_coverage": None,
        "phases": [],
    }
    memory = {"status": "available", "scope": "capacity_estimate_per_rank", "roles": {}}
    energy = {"status": "unsupported", "diagnostics": diagnostics}
    for sections in ({"memory": memory}, {"memory": memory, "energy": energy}):
        rendered = format_prediction_stdout(
            {"power_w": None, "power_coverage": None},
            "table",
            details={"schema_version": "1.0", "sections": sections, "skipped": {}},
            power_diagnostics=diagnostics,
        )
        assert rendered.count("AISimulate active forward-pass energy diagnostics (per GPU)") == 1


def _energy_detail_payload():
    from aisimulate.detail import build_prediction_details

    return build_prediction_details({"power_diagnostics": _diagnostics()}, ("energy",))


def _energy_detail_validator():
    from pathlib import Path

    from jsonschema import Draft202012Validator

    schema = json.loads((Path(__file__).parents[1] / "docs/cli/prediction-details.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_energy_schema_accepts_complete_measured_and_missing_operation_evidence():
    _energy_detail_validator().validate(_energy_detail_payload())


@pytest.mark.parametrize("skipped,valid", [({}, True), ({"energy": "not requested"}, True), ({"energy": ""}, False)])
def test_energy_schema_requires_a_reason_when_energy_is_skipped(skipped, valid):
    payload = {"schema_version": "1.0", "sections": {}, "skipped": skipped}
    assert _energy_detail_validator().is_valid(payload) is valid


@pytest.mark.parametrize("level", ["aggregate", "phase"])
@pytest.mark.parametrize(
    "status,watts,valid",
    [
        ("available", 400.0, True),
        ("available", None, False),
        ("withheld", 400.0, False),
        ("withheld", None, True),
        ("unsupported", 400.0, False),
        ("unsupported", None, True),
        ("not_observed", 400.0, False),
        ("not_observed", None, True),
        ("missing", 400.0, False),
        ("missing", None, True),
        ("invented", 400.0, False),
        ("invented", None, False),
    ],
)
def test_energy_publication_status_matches_nullable_watts(level, status, watts, valid):
    from aisimulate.detail import energy_diagnostics

    payload = _energy_detail_payload()
    diagnostics = payload["sections"]["energy"]["diagnostics"]
    record = diagnostics if level == "aggregate" else diagnostics["phases"][1]
    coverage = {"withheld": 0.8, "unsupported": None, "not_observed": 0.0}.get(status, 1.0)
    record.update(publication_status=status, power_w=watts, power_coverage=coverage)
    if level == "aggregate":
        payload["sections"]["energy"]["status"] = status
    else:
        valid = valid and coverage is not None
    assert _energy_detail_validator().is_valid(payload) is valid
    if valid:
        exported = energy_diagnostics({"power_diagnostics": diagnostics})
        record = exported if level == "aggregate" else exported["phases"][1]
        assert record["power_w"] == watts
        assert record["publication_status"] == status
    else:
        with pytest.raises(ValueError, match="publication_status|numeric power_w"):
            energy_diagnostics({"power_diagnostics": diagnostics})


@pytest.mark.parametrize("phases", [None, {}, "invalid", [None]])
def test_energy_export_rejects_malformed_phase_containers(phases):
    from aisimulate.detail import energy_diagnostics

    diagnostics = _diagnostics()
    diagnostics["phases"] = phases
    with pytest.raises(ValueError, match="energy phases must be a list of phase records"):
        energy_diagnostics({"power_diagnostics": diagnostics})


@pytest.mark.parametrize("level", ["aggregate", "phase"])
@pytest.mark.parametrize(
    "status,coverage,valid",
    [
        ("withheld", 0.0, True),
        ("withheld", 0.89, True),
        ("withheld", 0.9, False),
        ("withheld", 1.0, False),
        ("withheld", None, False),
        ("missing", 0.0, False),
        ("missing", 0.89, False),
        ("missing", 0.9, True),
        ("missing", 1.0, True),
        ("missing", None, False),
        ("unsupported", 0.0, False),
        ("unsupported", 1.0, False),
        ("unsupported", None, True),
        ("not_observed", 0.0, True),
        ("not_observed", 0.89, False),
        ("not_observed", 1.0, False),
        ("not_observed", None, True),
    ],
)
def test_energy_publication_status_matches_coverage(level, status, coverage, valid):
    from aisimulate.detail import energy_diagnostics

    payload = _energy_detail_payload()
    diagnostics = payload["sections"]["energy"]["diagnostics"]
    record = diagnostics if level == "aggregate" else diagnostics["phases"][1]
    record.update(publication_status=status, power_w=None, power_coverage=coverage)
    record["latency_ms"] = 0.0 if status == "not_observed" else 4.0
    if level == "aggregate":
        payload["sections"]["energy"]["status"] = status
        record["unavailable_reason"] = "explicit producer absence reason"
    else:
        valid = valid and coverage is not None
    assert _energy_detail_validator().is_valid(payload) is valid
    if valid:
        exported = energy_diagnostics({"power_diagnostics": diagnostics})
        exported_record = exported if level == "aggregate" else exported["phases"][1]
        assert exported_record["power_coverage"] == coverage
        assert exported_record["publication_status"] == status
        if level == "aggregate":
            assert exported_record["unavailable_reason"] == record["unavailable_reason"]
    else:
        with pytest.raises(ValueError, match="publication_status"):
            energy_diagnostics({"power_diagnostics": diagnostics})


def test_energy_schema_rejects_simultaneous_output_and_skipped_reason():
    payload = _energy_detail_payload()
    assert _energy_detail_validator().is_valid(payload)
    payload["skipped"]["energy"] = "also skipped"
    assert not _energy_detail_validator().is_valid(payload)


@pytest.mark.parametrize("value", ["invalid", 42, None, {}, ["nested"]])
@pytest.mark.parametrize("level", ["phase", "operation"])
def test_energy_schema_rejects_malformed_nested_records(level, value):
    payload = _energy_detail_payload()
    phases = payload["sections"]["energy"]["diagnostics"]["phases"]
    if level == "phase":
        phases[0] = value
    else:
        phases[0]["operations"][0] = value
    assert not _energy_detail_validator().is_valid(payload)


@pytest.mark.parametrize(
    "level,field,value",
    [
        ("phase", "latency_ms", -1),
        ("phase", "covered_latency_ms", -1),
        ("phase", "power_coverage", 1.01),
        ("phase", "power_w", 500),  # The phase has only 80% coverage.
        ("phase", "publication_status", "invented"),
        ("phase", "source_kind", "invented"),
        ("phase", "unknown_field", 1),
        ("operation", "latency_ms", True),
        ("operation", "energy_wms", -1),
        ("operation", "power_coverage", -0.01),
        ("operation", "energy_contribution", 1.01),
        ("operation", "status", "invented"),
        ("operation", "source_kind", "invented"),
        ("operation", "name", ""),
        ("operation", "unknown_field", 1),
    ],
)
def test_energy_schema_rejects_invalid_nested_evidence(level, field, value):
    payload = _energy_detail_payload()
    phase = payload["sections"]["energy"]["diagnostics"]["phases"][0]
    record = phase if level == "phase" else phase["operations"][0]
    record[field] = value
    assert not _energy_detail_validator().is_valid(payload)


@pytest.mark.parametrize("level", ["phase", "operation"])
@pytest.mark.parametrize("field", ["latency_ms", "covered_latency_ms", "power_coverage", "source", "source_kind"])
def test_energy_schema_requires_nested_evidence_and_provenance(level, field):
    payload = _energy_detail_payload()
    phase = payload["sections"]["energy"]["diagnostics"]["phases"][0]
    record = phase if level == "phase" else phase["operations"][0]
    del record[field]
    assert not _energy_detail_validator().is_valid(payload)
