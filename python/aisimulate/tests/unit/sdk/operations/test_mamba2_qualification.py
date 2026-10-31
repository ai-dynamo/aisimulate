# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba2Kernel carries its recurrent-state dtype and speculative width."""

from __future__ import annotations

import copy
import pickle

import pytest

import aisimulate.sdk.operations as ops

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"state_dtype": "float16", "token_width": 4}, {"state_dtype": "float16", "token_width": 4}),
        ({}, {"state_dtype": "unrecorded", "token_width": 1}),
    ],
)
def test_qualification_kwargs_survive_copy_and_pickle(kwargs, expected):
    op = ops.Mamba2Kernel(
        "generation_mamba_ssm", 1, "selective_state_update", "generation", 4096, 64, 64, 128, 4, 4, 128, **kwargs
    )
    for clone in (copy.deepcopy(op), pickle.loads(pickle.dumps(op))):
        assert clone.__getnewargs_ex__()[1] == expected
