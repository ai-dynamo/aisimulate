# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable checks for the AIC-compatible power contract."""

from __future__ import annotations

import json
import math
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


def derive_contract_metrics(case: dict[str, Any]) -> dict[str, float]:
    """Independently evaluate the documented AIC-compatible formulas."""
    roles = case["roles"]
    all_roles_energy_aware = all(role["energy_aware"] for role in roles)

    total_latency_ms = 0.0
    covered_latency_ms = 0.0
    total_energy_wms = 0.0
    for role in roles:
        scale = float(role.get("scale", 1.0))
        for operation in role["operations"]:
            latency_ms = float(operation["latency_ms"]) * scale
            energy_wms = float(operation["energy_wms"]) * scale
            if not math.isfinite(latency_ms) or latency_ms < 0.0:
                raise ValueError("latency evidence must be finite and non-negative")
            if not math.isfinite(energy_wms) or energy_wms < 0.0:
                raise ValueError("energy evidence must be finite and non-negative")
            total_latency_ms += latency_ms
            total_energy_wms += energy_wms
            if energy_wms > 0.0:
                covered_latency_ms += latency_ms

    if not all_roles_energy_aware:
        return {}

    if total_latency_ms <= 0.0:
        return {"power_coverage": 0.0}

    power_coverage = covered_latency_ms / total_latency_ms
    metrics = {"power_coverage": power_coverage}
    power_w = total_energy_wms / total_latency_ms
    if power_coverage >= 0.9 and math.isfinite(power_w) and power_w > 0.0:
        metrics["power_w"] = power_w
    return metrics


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"power_coverage": 0.0},
        {"power_coverage": 0.899999},
        {"power_w": 487.5, "power_coverage": 0.9},
        {"power_w": 510.25, "power_coverage": 1.0, "duration_ms": 20.0},
    ],
)
def test_power_contract_accepts_supported_availability_states(
    power_validator: Draft202012Validator,
    metrics: dict[str, float],
) -> None:
    validate_strict_json(power_validator, metrics)


@pytest.mark.parametrize(
    "metrics",
    [
        {"power_w": 487.5},
        {"power_w": 487.5, "power_coverage": 0.899999},
        {"power_w": 0.0, "power_coverage": 1.0},
        {"power_w": None, "power_coverage": 1.0},
        {"power_coverage": -0.01},
        {"power_coverage": 1.01},
        {"power_w": math.nan, "power_coverage": 1.0},
        {"power_w": math.inf, "power_coverage": 1.0},
        {"power_w": -math.inf, "power_coverage": 1.0},
        {"power_w": 487.5, "power_coverage": math.nan},
        {"power_w": 487.5, "power_coverage": math.inf},
        {"power_w": 487.5, "power_coverage": -math.inf},
    ],
)
def test_power_contract_rejects_fabricated_or_invalid_metrics(
    power_validator: Draft202012Validator,
    metrics: dict[str, float | None],
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

    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value)
    validate_strict_json(power_validator, actual)


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
