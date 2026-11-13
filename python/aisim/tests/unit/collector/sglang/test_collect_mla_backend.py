# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pins the standalone-MLA collector's default-backend map to SGLang serving's
MLA attention backend selection, and the declared gap below the mapped
platform set {89, 90, 100, 103, 120}."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_collector_function(name, namespace):
    source_path = REPO_ROOT / "collector" / "sglang" / "collect_mla.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    loaded = dict(namespace)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), loaded)
    return loaded[name]


def _selector(sm_version, *, cuda_ok):
    return _load_collector_function(
        "_select_default_mla_backend",
        {
            "get_sm_version": lambda: sm_version,
            "_cuda_version_at_least": lambda *_a: cuda_ok,
        },
    )


@pytest.mark.unit
def test_sglang_mla_backend_map_follows_serving_selection():
    # The DeepSeek V3/R1 trtllm_mla special case needs SM100/103 and CUDA 12.8.
    assert _selector(100, cuda_ok=True)() == "trtllm_mla"
    assert _selector(103, cuda_ok=True)() == "trtllm_mla"
    # Hopper takes fa3; everything else on the mapped platforms takes triton.
    assert _selector(90, cuda_ok=True)() == "fa3"
    assert _selector(90, cuda_ok=False)() == "triton"
    assert _selector(89, cuda_ok=True)() == "triton"
    assert _selector(89, cuda_ok=False)() == "triton"
    assert _selector(120, cuda_ok=False)() == "triton"

    for unsupported_sm in (80, 86):
        with pytest.raises(ValueError, match=r"No SGLang MLA backend mapping"):
            _selector(unsupported_sm, cuda_ok=True)()


def _geometry():
    return _load_collector_function(
        "_mla_attention_geometry",
        {"KV_LORA_RANK": 512, "QK_NOPE_HEAD_DIM": 128, "QK_ROPE_HEAD_DIM": 64, "V_HEAD_DIM": 128},
    )


@pytest.mark.unit
def test_trtllm_mla_prefill_is_serving_mha_call():
    # DeepseekV3 on trtllm_mla prefills eagerly through MHA_CHUNKED_KV: attn_mha with
    # one 192-wide KV head per query head and a 128-wide value, not the latent MQA.
    assert _geometry()("trtllm_mla", True) == (192, 128, False)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("backend", "is_context_phase"),
    [("trtllm_mla", False), ("fa3", True), ("fa3", False), ("triton", True), ("triton", False)],
)
def test_absorbed_lanes_run_attn_mqa(backend, is_context_phase):
    # Decode everywhere, and fa3/triton prefill under the default breakable prefill
    # graph, run the absorbed method: one 576-wide latent KV head, 512-wide output.
    assert _geometry()(backend, is_context_phase) == (576, 512, True)


@pytest.mark.unit
def test_geometry_rejects_unaudited_backend():
    with pytest.raises(ValueError, match="attention geometry for backend 'flashmla'"):
        _geometry()("flashmla", True)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("backend", "kv_dtype", "expected"),
    [
        ("trtllm_mla", "bf16", "bfloat16"),
        ("trtllm_mla", "fp8", "fp8"),
        ("fa3", "bf16", "bfloat16"),
        ("fa3", "fp8", "bfloat16"),
        ("triton", "bf16", "bfloat16"),
    ],
)
def test_mla_compute_label_follows_kernel_precision(backend, kv_dtype, expected):
    # Blackwell quantizes BF16 inputs internally; Hopper's absorbed 576-dim
    # MLA keeps BF16 compute even with FP8 storage. Do not conflate the axes.
    label = _load_collector_function(
        "_mla_compute_dtype",
        {"torch": SimpleNamespace(dtype=str, float8_e4m3fn="fp8")},
    )
    assert label(backend, kv_dtype) == expected


@pytest.mark.unit
@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
def test_mla_compute_label_rejects_unaudited_backend(kv_dtype):
    label = _load_collector_function(
        "_mla_compute_dtype",
        {"torch": SimpleNamespace(dtype=str, float8_e4m3fn="fp8")},
    )
    with pytest.raises(ValueError, match="compute-precision mapping for backend 'unknown_mla'"):
        label("unknown_mla", kv_dtype)


@pytest.mark.unit
@pytest.mark.parametrize("op", ["mla_context", "mla_generation"])
def test_sglang_mla_plan_declares_the_unmapped_platforms(op):
    # SM80/86 have no backend mapping: the plan leaves them out under their
    # skips.py entry, reported, never as a silent empty getter.
    from collector.op_plans import plan
    from collector.skips import SKIPS

    unmapped = next(entry for entry in SKIPS if entry.get("op") == ["mla_context", "mla_generation"])
    fp8 = next(entry for entry in SKIPS if entry.get("dtype") == ["fp8", "fp8_static", "fp8_block"])
    sm80 = plan(op, "sglang", 80)
    assert sm80.cases == []
    assert [(region.region, region.reason) for region in sm80.removed] == [
        ("dtype=bfloat16", repr(unmapped)),
        ("dtype=fp8", repr(fp8)),
    ]
    # The triton platforms plan fp8 KV like every other backend; a failure is recorded.
    for sm in (89, 120):
        assert {case.kv_cache_dtype for case in plan(op, "sglang", sm).cases} == {"bfloat16", "fp8"}


@pytest.mark.unit
def test_dsa_module_family_op_floors_cover_skip_indexer_variants():
    # The skip-indexer variants run the same SM90+ sparse attention kernel
    # family as the full DSA modules; skips.py must cover all four so sub-SM90
    # platforms drop them with a logged reason.
    from collector.skips import filter_cases

    for op in (
        "dsa_context_module",
        "dsa_generation_module",
        "dsa_context_module_skip_indexer",
        "dsa_generation_module_skip_indexer",
    ):
        assert filter_cases([[1]], "sglang", op, 89)[0] == []
        assert filter_cases([[1]], "sglang", op, 90)[0] == [[1]]
