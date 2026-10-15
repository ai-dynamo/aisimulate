# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shipped DSA table views expose measured full and indexer-reuse rows."""

import pytest

from aisimulate.sdk.operations.dsa import ContextDSAModule, GenerationDSAModule

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("system", "backend", "version", "expect_skip_rows"),
    [
        ("h200_sxm", "vllm", "0.24.0", True),
        ("b200_sxm", "sglang", "0.5.14", True),
        ("gb200", "sglang", "0.5.14", True),
    ],
)
def test_shipped_skip_row_availability(system, backend, version, expect_skip_rows):
    """Pins the shipped skip-row availability the engine degradation keys off."""
    from aisimulate_core.sdk.perf_database import get_database

    db = get_database(system, backend, version)
    ContextDSAModule.load_data(db)
    GenerationDSAModule.load_data(db)

    assert bool(db._context_dsa_module_skip_data) is expect_skip_rows
    assert bool(db._generation_dsa_module_skip_data) is expect_skip_rows
    # Either way the FULL table loads — the model must never die at load.
    assert db._context_dsa_module_data
    assert db._generation_dsa_module_data
