# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Missing staged inputs must not silently become zero-token native batches."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector/sglang/collect_mla_module.py"


def validate(batch, expected):
    node = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_validate_dsa_forward_tokens"
    )
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)
    ns[node.name](batch, expected)


def tensor(tokens):
    return SimpleNamespace(numel=lambda: tokens)


@pytest.mark.parametrize("tokens", [1, 129, 2048, 8192])
def test_real_query_tensor_counts_must_match(tokens):
    batch = SimpleNamespace(
        input_ids=tensor(tokens),
        positions=tensor(tokens),
        out_cache_loc=tensor(tokens),
        num_token_non_padded_cpu=tokens,
    )
    validate(batch, tokens)
    batch.input_ids = None
    batch.num_token_non_padded_cpu = 0
    with pytest.raises(ValueError, match="input_ids has None"):
        validate(batch, tokens)


@pytest.mark.parametrize("field", ["positions", "out_cache_loc", "num_token_non_padded_cpu"])
def test_decode_requires_one_new_token_per_request(field):
    batch = SimpleNamespace(
        input_ids=tensor(4), positions=tensor(4), out_cache_loc=tensor(4), num_token_non_padded_cpu=4
    )
    setattr(batch, field, 0 if field == "num_token_non_padded_cpu" else tensor(8192))
    with pytest.raises(ValueError, match=field):
        validate(batch, 4)
