# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 model facts: vllm needs block-size 128 because the sparse layers' backend accepts only that KV page
size while the dense head's FlashAttention accepts MultipleOf(16) (op-probe harness, L40, 2026-10-05)."""

import pytest

from aisimulate.generator.facts.apply import apply_model_default_args
from aisimulate.generator.facts.request_resolution import model_profile_for_path
from aisimulate.generator.facts.resolve import _FACTS_DIR, load_backend_version_matrix, resolve_facts

pytestmark = pytest.mark.unit


def _latest_dynamo(backend: str) -> str:
    matrix = load_backend_version_matrix(str(_FACTS_DIR / "runtimes" / "dynamo.yaml"))
    return next(v for v, backends in matrix.items() if backend in backends)


@pytest.mark.parametrize("path", ["MiniMaxAI/MiniMax-M3", "MiniMaxAI/MiniMax-M3-MXFP8", "nvidia/MiniMax-M3-NVFP4"])
def test_minimax_m3_paths_resolve_to_the_profile(path):
    assert model_profile_for_path(path) == "minimax-m3"


def test_minimax_m3_vllm_default_is_block_size_128_only():
    def tokens_for(backend: str) -> list[str]:
        facts = resolve_facts(model_profile_id="minimax-m3", hardware="h200", transport="nvlink",
                              dynamo_version=_latest_dynamo(backend), backend=backend)
        out: list[str] = []
        apply_model_default_args(out, facts.model, backend=backend, system=None, role="agg", variant=None)
        return out

    assert tokens_for("vllm") == ["--block-size", "128"]
    # sglang / trtllm carry no M3 default from this profile
    assert tokens_for("sglang") == []
    assert tokens_for("trtllm") == []
