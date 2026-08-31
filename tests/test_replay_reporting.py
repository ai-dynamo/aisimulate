# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aisimulate.replay.reporting import format_report_table


def test_report_table_surfaces_power_and_coverage() -> None:
    table = format_report_table(
        {
            "completed_requests": 1,
            "output_throughput_tok_s": 10.0,
            "request_throughput_rps": 1.0,
            "power_w": 487.5,
            "power_coverage": 0.95,
        }
    )

    assert "Power per GPU (W)" in table
    assert "487.50" in table
    assert "Power Data Coverage (%)" in table
    assert "95.00" in table


def test_report_table_marks_gated_power_unavailable() -> None:
    table = format_report_table(
        {
            "completed_requests": 1,
            "output_throughput_tok_s": 10.0,
            "request_throughput_rps": 1.0,
            "power_coverage": 0.42,
        }
    )

    assert "Power per GPU (W)" in table
    assert "N/A" in table
    assert "42.00" in table
