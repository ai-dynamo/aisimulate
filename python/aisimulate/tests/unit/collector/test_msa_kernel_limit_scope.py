# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The M3 index-head kernel limit is the DECODE scorer's: a guard keyed on the
head count alone dropped 1808 prefill rows on the 0.30 sweep (2026-09-28)
that the 0.29 sweep had collected."""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[3] / "collector" / "vllm" / "collect_msa_module.py"


def _predicate():
    """Load only the pure predicate — the module itself imports vllm."""
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "msa_decode_scorer_kernel_limit")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SRC), "exec"), ns)
    return ns["msa_decode_scorer_kernel_limit"]


@pytest.mark.parametrize("idx_heads,num_heads", [(3, 48), (6, 96)])
def test_non_power_of_two_index_heads_fail_only_where_the_decode_scorer_runs(idx_heads, num_heads):
    limit = _predicate()
    assert limit(idx_heads, num_heads, is_context=False, query_len=1)  # generation
    assert limit(idx_heads, num_heads, is_context=True, query_len=1)  # one-token context query
    assert limit(idx_heads, num_heads, is_context=True, query_len=16) is None  # prefill path
    assert limit(idx_heads, num_heads, is_context=True, query_len=32768) is None


@pytest.mark.parametrize("idx_heads", [1, 2, 4, 8])
def test_power_of_two_index_heads_never_limited(idx_heads):
    limit = _predicate()
    for is_context, q in ((False, 1), (True, 1), (True, 4096)):
        assert limit(idx_heads, idx_heads * 16, is_context=is_context, query_len=q) is None


def test_builder_passes_the_query_length():
    text = SRC.read_text()
    assert "query_len=seq_len if is_context else 1," in text
    assert "msa_decode_scorer_kernel_limit(" in text
