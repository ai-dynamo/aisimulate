# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable checks for the AIC-compatible power contract."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

ROOT = Path(__file__).parents[1]
SCHEMA_PATH = ROOT / "docs" / "schemas" / "power-metrics-v1.schema.json"
EXAMPLES_PATH = ROOT / "tests" / "fixtures" / "power-contract-v1.json"


def validate_strict_json(
    validator: Draft202012Validator,
    metrics: dict[str, float | None],
) -> None:
    """Reject host-language non-finite numbers before schema validation."""
    payload = json.dumps(metrics, allow_nan=False)
    validator.validate(json.loads(payload))


@pytest.fixture(scope="module")
def power_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validated_fixture_number(value: Any, name: str) -> Fraction:
    """Validate raw fixture evidence before any scaling can hide its sign."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} evidence must be finite and non-negative")
    exact = Fraction(str(value))
    if exact < 0:
        raise ValueError(f"{name} evidence must be finite and non-negative")
    return exact


def derive_contract_metrics(case: dict[str, Any]) -> dict[str, float | None]:
    """Evaluate fixture decimals exactly; convert to floats only for JSON output.

    This independent oracle uses the same exact sums for coverage and the gate,
    so decimal inputs at 90% qualify without widening the threshold by epsilon.
    """
    roles = case["roles"]
    all_roles_energy_aware = all(role["energy_aware"] for role in roles)

    total_latency_ms = Fraction(0)
    covered_latency_ms = Fraction(0)
    total_energy_wms = Fraction(0)
    for role in roles:
        scale = validated_fixture_number(role.get("scale", 1.0), "scale")
        for operation in role["operations"]:
            latency_ms = validated_fixture_number(operation["latency_ms"], "latency") * scale
            energy_wms = validated_fixture_number(operation["energy_wms"], "energy") * scale
            total_latency_ms += latency_ms
            total_energy_wms += energy_wms
            if energy_wms > 0.0:
                covered_latency_ms += latency_ms

    if not all_roles_energy_aware:
        return {"power_w": None, "power_coverage": None}

    if total_latency_ms <= 0.0:
        return {"power_w": None, "power_coverage": 0.0}

    power_coverage = covered_latency_ms / total_latency_ms
    metrics: dict[str, float | None] = {"power_w": None, "power_coverage": float(power_coverage)}
    try:
        power_w = float(total_energy_wms / total_latency_ms)
    except OverflowError:
        return metrics
    if power_coverage >= Fraction(9, 10) and math.isfinite(power_w) and power_w > 0.0:
        metrics["power_w"] = power_w
    return metrics


@pytest.mark.parametrize(
    "metrics",
    [
        {"power_w": None, "power_coverage": None},
        {"power_w": None, "power_coverage": 0.0},
        {"power_w": None, "power_coverage": 0.899999},
        {"power_w": None, "power_coverage": 1.0},
        {"power_w": 487.5, "power_coverage": 0.9},
        {"power_w": 510.25, "power_coverage": 1.0, "duration_ms": 20.0},
    ],
)
def test_power_contract_accepts_supported_availability_states(
    power_validator: Draft202012Validator,
    metrics: dict[str, float | None],
) -> None:
    validate_strict_json(power_validator, metrics)


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"power_w": None},
        {"power_coverage": None},
        {"power_coverage": 0.9},
        {"power_w": 487.5},
        {"power_w": 487.5, "power_coverage": None},
        {"power_w": 487.5, "power_coverage": 0.899999},
        {"power_w": 0.0, "power_coverage": 1.0},
        {"power_w": -1.0, "power_coverage": 1.0},
        {"power_w": None, "power_coverage": -0.01},
        {"power_w": None, "power_coverage": 1.01},
        {"power_w": "unavailable", "power_coverage": 1.0},
        {"power_w": None, "power_coverage": "unavailable"},
        {"power_w": True, "power_coverage": 1.0},
        {"power_w": None, "power_coverage": False},
        {"power_w": math.nan, "power_coverage": 1.0},
        {"power_w": math.inf, "power_coverage": 1.0},
        {"power_w": -math.inf, "power_coverage": 1.0},
        {"power_w": 487.5, "power_coverage": math.nan},
        {"power_w": 487.5, "power_coverage": math.inf},
        {"power_w": 487.5, "power_coverage": -math.inf},
        {"power_w": None, "power_coverage": math.nan},
        {"power_w": None, "power_coverage": math.inf},
        {"power_w": None, "power_coverage": -math.inf},
    ],
)
def test_power_contract_rejects_fabricated_or_invalid_metrics(
    power_validator: Draft202012Validator,
    metrics: dict[str, Any],
) -> None:
    with pytest.raises((ValidationError, ValueError)):
        validate_strict_json(power_validator, metrics)


@pytest.mark.parametrize(
    "case",
    json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))["cases"],
    ids=lambda case: case["name"],
)
def test_reproducible_examples_match_documented_aic_semantics(
    power_validator: Draft202012Validator,
    case: dict[str, Any],
) -> None:
    actual = derive_contract_metrics(case)
    expected = case["expected"]

    assert actual.keys() == expected.keys() == {"power_w", "power_coverage"}
    assert actual == expected
    validate_strict_json(power_validator, expected)
    validate_strict_json(power_validator, actual)
    assert json.loads(json.dumps(actual, allow_nan=False)) == actual


def test_reproducible_examples_record_provenance() -> None:
    fixture = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    provenance = fixture["provenance"]

    assert provenance["kind"] == "synthetic_contract_fixture"
    assert provenance["measured"] is False
    assert provenance["accuracy_evidence"] is False
    assert len(provenance["aic_semantics_revision"]) == 40
    assert len(provenance["fpe_energy_revision"]) == 40
    assert all((ROOT / path).is_file() for path in provenance["reference_paths"])


@pytest.mark.parametrize(
    "case",
    json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))["invalid_evidence_cases"],
    ids=lambda case: case["name"],
)
def test_reproducible_examples_reject_invalid_operation_evidence(
    case: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        derive_contract_metrics(case)


def test_public_docs_keep_availability_separate_from_semantics() -> None:
    contract = (ROOT / "docs" / "power-model.md").read_text(encoding="utf-8")
    migration = (ROOT / "docs" / "cli" / "migrate-from-aiconfigurator.md").read_text(encoding="utf-8")
    core_api = (ROOT / "docs" / "core-api.md").read_text(encoding="utf-8")

    assert "This PR does not change current AIC or FPE runtime behavior" in contract
    assert "the contract\ndoes not by itself make modeled power available" in contract
    assert "exactly `0.90` is sufficient;\n`0.899` is not" in contract
    assert "Once the follow-up runtime work adds a conforming producer" in contract
    assert "fixtures/power-contract-v1.json" in contract
    assert "[modeled-power contract](../power-model.md)" in migration
    assert "Typed per-op energy alone does\nnot make unified replay power available" in core_api


def test_exact_oracle_withholds_power_that_overflows_float(
    power_validator: Draft202012Validator,
) -> None:
    case = {
        "roles": [
            {
                "energy_aware": True,
                "operations": [{"latency_ms": "1e-400", "energy_wms": 1.0}],
            }
        ]
    }
    actual = derive_contract_metrics(case)
    assert actual == {"power_w": None, "power_coverage": 1.0}
    validate_strict_json(power_validator, actual)
