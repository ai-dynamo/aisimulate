# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the ``Operation._engine_query`` kwarg shim.

The shim maps legacy per-op ``query(**kwargs)`` shapes onto the compiled
engine's single-op evaluation. No production caller remains, but the surface
stays pinned by the single-oracle contract tests. These tests exercise the
shim directly, including the ``verify``-phase ``KDAKernel`` mapping (the
speculative Kimi-K3 shape).
"""

from __future__ import annotations

import pytest

from aisimulate.sdk.operations import KDAKernel
from aisimulate.sdk.perf_database import get_database

pytestmark = pytest.mark.unit

_SYSTEM, _BACKEND, _VERSION = "h200_sxm", "sglang", "0.5.16"
_LEGACY_VERIFY_MS = 0.00067456  # merge-base KDAKernel("chunk_kda", "verify") @ b=1, s=4096


@pytest.fixture(scope="module")
def database():
    db = get_database(_SYSTEM, _BACKEND, _VERSION, allow_unlisted_version=True)
    if db is None:
        pytest.skip(f"{_SYSTEM}/{_BACKEND}/{_VERSION} data missing")
    return db


def _verify_kda() -> KDAKernel:
    return KDAKernel("context_kda_verify", 1.0, "chunk_kda", "verify", 7168, 24, 128, 24, 128, 4, draft_tokens=3)


def test_engine_query_shim_evaluates_verify_kda(database):
    """The generation-shaped legacy kwargs (x/batch_size/beam_width/s/prefix/
    model_name + gen imbalance scale) must map the ``verify`` phase onto the
    engine's generation-like evaluation and return the merge-base value."""
    result = _verify_kda()._engine_query(
        database,
        x=1,
        batch_size=1,
        beam_width=1,
        s=4096,
        prefix=0,
        model_name="kimi-k3-fixture",
        gen_seq_imbalance_correction_scale=1.0,
    )
    assert float(result) == pytest.approx(_LEGACY_VERIFY_MS, rel=1e-9)


def test_verify_kda_spec_serialization_keeps_phase_and_draft_tokens(database):
    """The evaluation-context mapping (verify -> generation-like) must NOT
    leak into the wire: the serialized OpSpec keeps phase="verify" and
    draft_tokens."""
    import json

    from aisimulate_core.sdk.engine import build_ops_json

    spec = json.loads(build_ops_json([_verify_kda()]))[0]
    (_, payload) = next(iter(spec.items()))
    assert payload["phase"] == "verify"
    assert payload["draft_tokens"] == 3
