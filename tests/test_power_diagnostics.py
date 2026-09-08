# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

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
    assert "modeled:estimated" not in rendered
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
