# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba2Kernel carries its recurrent-state dtype and speculative width."""

from __future__ import annotations

import copy
import pickle

import pytest

import aisimulate.sdk.operations as ops

pytestmark = pytest.mark.unit


def _kernel(**kwargs):
    return ops.Mamba2Kernel(
        "generation_mamba_ssm",
        1,
        "selective_state_update",
        "generation",
        hidden_size=4096,
        nheads=64,
        head_dim=64,
        d_state=128,
        d_conv=4,
        n_groups=4,
        chunk_size=128,
        **kwargs,
    )


def test_qualification_kwargs_survive_copy_and_pickle():
    op = _kernel(state_dtype="float16", token_width=4)
    for clone in (copy.deepcopy(op), pickle.loads(pickle.dumps(op))):
        assert clone.__getnewargs_ex__()[1] == {"state_dtype": "float16", "token_width": 4}


def test_unqualified_default_is_preserved_on_round_trip():
    clone = copy.deepcopy(_kernel())
    assert clone.__getnewargs_ex__()[1] == {"state_dtype": "unrecorded", "token_width": 1}
