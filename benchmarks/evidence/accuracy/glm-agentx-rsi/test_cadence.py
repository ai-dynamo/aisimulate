# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from audit_cadence import cadence


def observations():
    return [{"rid": rid, "rank": 0, "at_ns": str(at * 1_000_000), "output_tokens": count,
             "batch_requests": 2} for at, count in ((0, 1), (10, 4), (20, 7), (30, 10), (40, 13))
            for rid in ("a", "b")]


def test_common_nonterminal_cadence_counts_exclusions_and_excess():
    result = cadence(observations(), {"a", "b"}, target=10)
    assert result["retained_intervals"] == 2
    assert result["excluded_intervals"] == {"outside_common_nonterminal_coverage": 2}
    assert result["generated_excess_tokens"] == 6
    assert result["cadence_ms"]["mean"] == 10
    assert result["adjacent_intervals"] == 4


def test_partial_groups_remain_in_accounting_and_empty_coverage_is_unavailable():
    rows = observations()
    rows = [row for row in rows if not (row["rid"] == "b" and row["at_ns"] == "10000000")]
    result = cadence(rows, {"a", "b"}, target=10)
    assert result["retained_intervals"] == 0
    assert result["excluded_intervals"]["incomplete_membership"] == 2
    assert "unavailable" in result


def test_duplicate_missing_and_nonmonotone_counters_reject():
    rows = observations()
    with pytest.raises(ValueError, match="duplicate"):
        cadence(rows + rows[:1], {"a", "b"}, target=10)
    with pytest.raises(ValueError, match="Missing planned"):
        cadence([row for row in rows if row["rid"] == "a"], {"a", "b"}, target=10)
    rows[4]["output_tokens"] = 0
    with pytest.raises(ValueError, match="backward"):
        cadence(rows, {"a", "b"}, target=10)
