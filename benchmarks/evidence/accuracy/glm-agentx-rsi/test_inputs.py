# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest
from audit_inputs import audit, counts


def test_geometry_strata_preserve_missing_zero_and_frozen_accounting():
    rows = [{"key": key, "phase": "profile", "status": "completed", "input_length": 200,
             "cached_tokens": 100, "output_length": 2, "ttft_ms": 10, "tpot_ms": 2}
            for key in ("same_fresh", "more_fresh", "unknown", "missing")]
    frozen = [{"request_id": hashlib.sha256(row["key"].encode()).hexdigest()[:16],
               "ttft_ms_ape": 1, "tpot_ms_ape": 1} for row in rows]
    predicted = [{"request_id": row["key"], "phase": "profile", "terminal_status": "completed",
                  "input_length": 220, "first_admission_reused_input_tokens": cache,
                  "output_length": 2, "ttft_ms": 15, "itl_ms": 0}
                 for row, cache in zip(rows, (120, 0, None), strict=False)]
    result = audit(rows, predicted, frozen)
    assert result["geometry"]["fresh_within_64_tokens"] == 1
    assert result["geometry"]["predicted_more_fresh"] == 1
    assert result["geometry"]["unavailable_geometry"] == 1
    assert result["geometry"]["input_equal"] == 0
    assert result["geometry"]["known_input"] == 3
    assert result["geometry"]["known_measured_cache"] == 3
    assert result["geometry"]["known_predicted_cache"] == 2
    assert result["deltas_tokens"]["input"]["n"] == 3
    assert result["deltas_tokens"]["fresh"]["sum_signed"] == 120
    assert result["metrics"]["ttft"]["lost"] == 1
    assert result["metrics"]["ttft"]["groups"]["all_retained"] == {"n": 3, "mape": 50}
    assert result["metrics"]["tpot"]["groups"]["all_retained"] == {"n": 3, "mape": 100}
    assert result["metrics"]["ttft"]["groups"]["input_within_64_tokens"] == {"n": 3, "mape": 50}
    assert counts({"input_length": 1, "cached_tokens": 0}, "cached_tokens") == (1, 0, 1)
    with pytest.raises(ValueError, match="Invalid input/cache"):
        counts({"input_length": 1, "cached_tokens": 2}, "cached_tokens")
    with pytest.raises(ValueError, match="Duplicate frozen"):
        audit(rows, predicted, frozen + frozen[:1])
