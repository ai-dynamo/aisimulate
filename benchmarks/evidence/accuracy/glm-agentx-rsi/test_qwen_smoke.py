# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
from qwen_smoke import verify


def test_archived_smoke_checks_nested_physical_fields_and_excludes_execution_wall_time():
    row = {"request_id": "stable", "phase": "profile", "terminal_status": "completed", "first_token_ms": 10.0}
    reference = {"per_request": [row], "telemetry": {"requests": [copy.deepcopy(row)]},
                 "state_cache": {"allocated_bytes": 64}, "duration_ms": 20.0, "wall_time_ms": 100.0}
    candidate = copy.deepcopy(reference)
    candidate["wall_time_ms"] = 200.0
    verify(reference, candidate, 1)
    candidate["telemetry"]["requests"][0]["first_token_ms"] = 11.0
    with pytest.raises(ValueError, match="Numerical telemetry"):
        verify(reference, candidate, 1)
    candidate = copy.deepcopy(reference)
    candidate["state_cache"]["allocated_bytes"] = 128
    with pytest.raises(ValueError, match="Cache accounting"):
        verify(reference, candidate, 1)
    with pytest.raises(ValueError, match="count"):
        verify(reference, reference, 2)
