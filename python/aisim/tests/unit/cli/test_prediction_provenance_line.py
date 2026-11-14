# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The prediction timing line names the estimator, source, data version and database mode."""

from __future__ import annotations

import pytest

from aisim.output import format_prediction_stdout

pytestmark = pytest.mark.unit


def _timing_line(provenance: dict[str, object]) -> str:
    summary = {"prediction_provenance": {"aggregated": provenance}}
    (line,) = [line for line in format_prediction_stdout(summary, "table").splitlines() if line.startswith("Timing (")]
    return line


def test_timing_line_ends_with_the_selected_database_mode() -> None:
    line = _timing_line(
        {"estimator": "op_level", "timing_source": "aic", "data_version": "0.24.0", "database_mode": "HYBRID"}
    )
    assert line == "Timing (aggregated): estimator=op_level; source=aic; data_version=0.24.0; database_mode=HYBRID"


def test_timing_line_omits_the_database_mode_when_the_estimator_has_none() -> None:
    line = _timing_line(
        {
            "estimator": "fpm_regression",
            "timing_source": "aic",
            "data_version": None,
            "database_mode": None,
            "data_version_unavailable_reason": "selected estimator does not expose a data version",
        }
    )
    assert line == (
        "Timing (aggregated): estimator=fpm_regression; source=aic; "
        "data_version=unavailable (selected estimator does not expose a data version)"
    )
    assert "database_mode" not in line
