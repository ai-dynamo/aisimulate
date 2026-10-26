# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from collector.campaign_selection import select_campaign_cases

pytestmark = pytest.mark.unit


def test_scope_preserves_native_cases_and_accounts_excluded_shapes():
    cases = [
        ["fp8", 1, 32, 32],
        ["bfloat16", 1, 32, 32],
        ["bfloat16", 1, 99, 32],
        ["bfloat16", 2, 99, 32],
        ["fp8_block", 1, 32, 32],
    ]
    logger = Mock()
    selected = select_campaign_cases(cases, ["'bfloat16'", "'fp8_block'"], [", 99, 32]"], op="gemm", logger=logger)
    assert selected == [cases[1], cases[4]]
    assert selected[0] is cases[1]
    assert "outside_include=1 explicitly_excluded=2" in logger.info.call_args.args[0]
    assert len(cases) == 5


def test_literal_filters_and_empty_selection():
    logger = Mock()
    cases = [["bf16", 1], ["bf16", 2]]
    assert select_campaign_cases(cases, None, [".*"], op="x", logger=logger) == cases
    assert select_campaign_cases(cases, None, ["bf16"], op="x", logger=logger) == []
    assert select_campaign_cases(cases, None, None, op="x", logger=logger) is cases


@pytest.mark.parametrize("include,exclude", [([""], None), (None, [""]), (None, [1])])
def test_reject_invalid_scope_filters(include, exclude):
    with pytest.raises(ValueError, match="nonempty literal"):
        select_campaign_cases([[1]], include, exclude, op="x", logger=Mock())


def test_executor_excludes_before_limit_and_preserves_native_callback(monkeypatch):
    collector_dir = str(Path(__file__).resolve().parents[3] / "collector")
    monkeypatch.syspath_prepend(collector_dir)
    import collect

    from collector import capabilities

    cases = [["bf16", 1], ["bf16", 2], ["fp8", 2]]
    callback = Mock()
    module = SimpleNamespace(get_cases=lambda: cases, run=callback)
    monkeypatch.setitem(sys.modules, "campaign_test_module", module)
    monkeypatch.setattr(collect, "logger", Mock())
    monkeypatch.setattr(capabilities, "filter_cases", lambda cases, **kwargs: (cases, 0))
    observed = []

    def execute(name, op, getter, runner, processes, **kwargs):
        selected = getter()
        observed.extend(selected)
        for case in selected:
            runner(*case)
        return []

    monkeypatch.setattr(collect, "collect_module_safe", execute)
    errors = collect.collect_ops(
        1,
        [
            {
                "name": "vllm",
                "type": "gemm",
                "module": "campaign_test_module",
                "get_func": "get_cases",
                "run_func": "run",
                "perf_filename": "fixture.txt",
            }
        ],
        limit=1,
        case_filters=["bf16"],
        case_exclude_filters=["1]"],
        sm_version=100,
    )
    assert errors == []
    assert observed == [["bf16", 2]]
    callback.assert_called_once_with("bf16", 2, perf_filename="fixture.txt")
