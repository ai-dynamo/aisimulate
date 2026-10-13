# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 has its own facts profile: V4's `DeepSeek-V4*` glob used to
hand it block-size 256, which vllm 0.30.0 rejects for V4.1's 64-token kernel
blocks (op-probe harness A/B, 2026-10-01)."""
import pytest

from aisimulate.generator.facts.apply import apply_model_default_args
from aisimulate.generator.facts.request_resolution import model_profile_for_path
from aisimulate.generator.facts.resolve import resolve_facts

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "path,profile",
    [
        ("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-v4"),
        ("deepseek-ai/DeepSeek-V4-Pro", "deepseek-v4"),
        ("sgl-project/DeepSeek-V4-Flash-FP8", "deepseek-v4"),
        ("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-v4.1"),
        ("nvidia/DeepSeek-V4.1-Flash-NVFP4", "deepseek-v4.1"),
    ],
)
def test_v4_and_v41_resolve_to_their_own_profiles(path, profile):
    assert model_profile_for_path(path) == profile


def _vllm_defaults(profile_id: str) -> list[str]:
    facts = resolve_facts(model_profile_id=profile_id, hardware="h200", transport="nvlink",
                          dynamo_version=_latest_dynamo(), backend="vllm")
    tokens: list[str] = []
    apply_model_default_args(tokens, facts.model, backend="vllm", system=None, role="agg", variant=None)
    return tokens


def _latest_dynamo() -> str:
    from aisimulate.generator.facts.resolve import _FACTS_DIR, load_backend_version_matrix
    matrix = load_backend_version_matrix(str(_FACTS_DIR / "runtimes" / "dynamo.yaml"))
    return next(v for v, backends in matrix.items() if "vllm" in backends)


def test_v41_vllm_block_size_is_64_not_v4s_256():
    v41 = _vllm_defaults("deepseek-v4.1")
    v4 = _vllm_defaults("deepseek-v4")
    assert v41[v41.index("--block-size") + 1] == "64"
    assert v4[v4.index("--block-size") + 1] == "256"
    assert "--trust-remote-code" in v41 and "--no-enable-flashinfer-autotune" in v41
