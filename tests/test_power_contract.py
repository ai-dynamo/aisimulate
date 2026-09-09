# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable checks for the AIC-compatible power contract."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

ROOT = Path(__file__).parents[1]
SCHEMA_PATH = ROOT / "docs" / "schemas" / "power-metrics-v1.schema.json"


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


def test_public_docs_keep_availability_separate_from_semantics() -> None:
    contract = (ROOT / "docs" / "power-model.md").read_text(encoding="utf-8")
    migration = (ROOT / "docs" / "cli" / "migrate-from-aiconfigurator.md").read_text(encoding="utf-8")
    core_api = (ROOT / "docs" / "core-api.md").read_text(encoding="utf-8")

    assert "This PR does not change current AIC or FPE runtime behavior" in contract
    assert "the contract\ndoes not by itself make modeled power available" in contract
    assert "exactly `0.90` is sufficient;\n`0.899` is not" in contract
    assert "Once the follow-up runtime work adds a conforming producer" in contract
    assert "[modeled-power contract](../power-model.md)" in migration
    assert "Typed per-op energy alone does\nnot make unified replay power available" in core_api
