# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash (Glm5Next) population and runtime gating of the vLLM/SGLang mHC collectors."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

COLLECTOR_DIR = Path(__file__).resolve().parents[3] / "collector"
DSV4 = "DeepseekV4ForCausalLM"
GLM = "Glm5NextForConditionalGeneration"
GLM_MODEL = "zai-org/GLM-5.3-Flash"
GLM_NVFP4 = "nvidia/GLM-5.3-Flash-NVFP4"


def _load(monkeypatch, backend: str):
    # The collectors import torch at module scope; population never touches it.
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    from collector import case_generator

    case_generator._load_model_cases_data.cache_clear()
    path = COLLECTOR_DIR / backend / "collect_mhc_module.py"
    spec = importlib.util.spec_from_file_location(f"_mhc_{backend}_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _token_counts():
    from collector.case_generator import get_common_mhc_test_cases

    return get_common_mhc_test_cases()[0].num_tokens_list


def test_vllm_population_keeps_dsv4_and_adds_glm_call_sites(monkeypatch):
    module = _load(monkeypatch, "vllm")
    tokens = _token_counts()
    cases = module.get_mhc_module_test_cases()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))

    dsv4 = [case for case in cases if not case["id"].startswith("mhc_glm5next_")]
    # DeepSeek-V4 population is unchanged: legacy ids and 4-element params,
    # one physical case per (phase, hidden, hc_mult).
    assert {case["id"] for case in dsv4} == {
        f"mhc_{phase}_hs{hidden}_hcm4_{t}" for phase in ("pre", "post") for hidden in (4096, 7168) for t in tokens
    }
    assert all(len(case["params"]) == 4 for case in dsv4)

    glm = [case for case in cases if case["id"].startswith("mhc_glm5next_")]
    assert {tuple(case["params"]) for case in glm} == {
        (op, t, 4096, 4, GLM, GLM_MODEL)
        for op in ("pre", "post", "fused_post_pre", "expand", "contract")
        for t in tokens
    }


@pytest.mark.parametrize("model_path", [GLM_MODEL, GLM_NVFP4])
def test_vllm_targeted_glm_plan_is_glm_only(monkeypatch, model_path):
    module = _load(monkeypatch, "vllm")
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    cases = module.get_mhc_module_test_cases()
    assert cases
    # The NVFP4 artifact is an alias of the one physical (BF16) mHC case.
    assert {case["params"][4] for case in cases} == {GLM}
    assert {case["params"][5] for case in cases} == {GLM_MODEL}


def test_sglang_population_keeps_dsv4_and_adds_glm_call_sites(monkeypatch):
    module = _load(monkeypatch, "sglang")
    cases = module.get_mhc_module_test_cases()
    dsv4 = [case for case in cases if not case["id"].startswith("mhc_glm5next_")]
    assert [case["params"] for case in dsv4] == [
        ["pre", "deepseek-ai/DeepSeek-V4-Flash"],
        ["post", "deepseek-ai/DeepSeek-V4-Flash"],
        ["pre", "deepseek-ai/DeepSeek-V4-Pro"],
        ["post", "deepseek-ai/DeepSeek-V4-Pro"],
    ]
    glm = [case["params"] for case in cases if case["id"].startswith("mhc_glm5next_")]
    # No fused post->pre call exists in SGLang's Glm5Next path.
    assert glm == [[op, GLM_MODEL, GLM] for op in ("pre", "post", "expand", "contract")]


def test_glm_row_conventions_match_the_shared_table(monkeypatch):
    vllm = _load(monkeypatch, "vllm")
    sglang = _load(monkeypatch, "sglang")
    # pre/post (and vLLM fused_post_pre) fold both per-layer sites like the
    # DeepSeek-V4 rows; expand/contract happen once per forward.
    assert vllm._GLM5NEXT_NUM_SITES == {"pre": 2, "post": 2, "fused_post_pre": 2, "expand": 1, "contract": 1}
    assert sglang._GLM5NEXT_NUM_SITES == {"pre": 2, "post": 2, "expand": 1, "contract": 1}
    assert vllm.MHC_NUM_SITES == 2


@pytest.mark.parametrize(
    ("backend", "architecture", "version", "audited"),
    [
        ("vllm", DSV4, "0.24.0", True),
        ("vllm", DSV4, "0.25.0", True),
        ("vllm", DSV4, "0.30.0+glm53tail.eb4704514fdf", False),
        ("vllm", DSV4, "0.31.0", False),
        # GLM moved to stock 0.31.0 (no overlay); the 0.30.0+glm53tail runtime
        # is no longer an audited GLM mHC runtime.
        ("vllm", GLM, "0.31.0", True),
        ("vllm", GLM, "0.30.0+glm53tail.eb4704514fdf", False),
        ("vllm", GLM, "0.30.0", False),
        ("vllm", GLM, "0.25.0", False),
        ("vllm", GLM, "0.27.1", False),
        ("sglang", DSV4, "0.5.14", True),
        ("sglang", DSV4, "0.5.20", False),
        ("sglang", GLM, "0.5.20", True),
        ("sglang", GLM, "0.5.14", False),
        ("sglang", "UnknownForCausalLM", "0.5.20", False),
    ],
)
def test_each_architecture_runs_only_on_its_audited_release(monkeypatch, backend, architecture, version, audited):
    module = _load(monkeypatch, backend)
    from collector.version_resolver import _check_compat

    # The file-level __compat__ admits every audited release.
    if audited:
        assert _check_compat(module.__compat__, version)
        module._require_audited_runtime(architecture, version)
    else:
        with pytest.raises(module.MhcRuntimeNotAuditedError):
            module._require_audited_runtime(architecture, version)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_audited_releases_never_overlap_across_architectures(monkeypatch, backend):
    # mhc_module_perf is keyed (op_name, hc_mult, hidden_size) without an
    # architecture column, and DeepSeek-V4-Flash and GLM-5.3-Flash share
    # (4, 4096). Disjoint audited releases guarantee the two architectures'
    # distinct invocations never land in one <backend>/<version> table.
    module = _load(monkeypatch, backend)
    from collector.version_resolver import _check_compat

    candidates = ["0.24.0", "0.25.0", "0.27.1", "0.30.0", "0.31.0", "0.5.14", "0.5.16", "0.5.20"]
    for version in candidates:
        admitted = [arch for arch, spec in module._ARCHITECTURE_COMPAT.items() if _check_compat(spec, version)]
        assert len(admitted) <= 1, (version, admitted)


def test_vllm_file_compat_admits_glm_runtime_and_keeps_dsv4_releases(monkeypatch):
    module = _load(monkeypatch, "vllm")
    from collector.version_resolver import _check_compat

    for version in ("0.24.0", "0.25.0", "0.31.0"):
        assert _check_compat(module.__compat__, version), version
    assert not _check_compat(module.__compat__, "0.32.0")


def test_vllm_glm_sites_import_the_0_31_0_module_location():
    # vllm.models.glm5next.nvidia.model moved to .common.model at 0.31.0.
    source = (COLLECTOR_DIR / "vllm" / "collect_mhc_module.py").read_text(encoding="utf-8")
    assert "from vllm.models.glm5next.common.model import Glm5NextDecoderLayer" in source
    assert "vllm.models.glm5next.nvidia.model" not in source


def test_worker_rejects_unknown_architecture(monkeypatch):
    module = _load(monkeypatch, "vllm")
    with pytest.raises(module.MhcRuntimeNotAuditedError):
        module.run_mhc_module_worker("pre", 16, 4096, 4, "UnknownForCausalLM", perf_filename="/tmp/x_perf.txt")
