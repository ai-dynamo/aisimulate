# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mHC case population per backend, and the runtime gating of the vLLM/SGLang mHC collectors."""

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


def _planned_rows(monkeypatch, backend: str, model_path: str | None = None) -> list[dict]:
    """What the backend's getter queues (mHC has no SM-dependent region)."""
    from collector import case_generator
    from collector.op_plans import plan_cases

    if model_path is None:
        monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    else:
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    case_generator._load_model_cases_data.cache_clear()
    return plan_cases("mhc_module", backend, 100)


def test_trtllm_plans_the_dsv4_sites_only(monkeypatch):
    tokens = _token_counts()
    cases = _planned_rows(monkeypatch, "trtllm")
    # G-trtllm-mhc-glm5next-not-served: the module API is DeepSeek-V4's.
    assert [case["id"] for case in cases] == [
        f"mhc_{site}_hs{hidden}_hcm4_{t}" for hidden in (4096, 7168) for site in ("pre", "post") for t in tokens
    ]
    assert all(len(case["params"]) == 4 for case in cases)


def test_vllm_plans_the_glm_call_sites_only(monkeypatch):
    vllm = _load(monkeypatch, "vllm")
    tokens = _token_counts()
    cases = _planned_rows(monkeypatch, "vllm")
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    # G-vllm-mhc-dsv4-shared-key: DeepSeek-V4 rows would share GLM's key.
    assert all(case["id"].startswith("mhc_glm5next_") for case in cases)
    assert {tuple(case["params"]) for case in cases} == {
        (site, t, 4096, 4, GLM, GLM_MODEL) for site in vllm.GLM5NEXT_VLLM_OPS for t in tokens
    }


@pytest.mark.parametrize("model_path", [GLM_MODEL, GLM_NVFP4])
def test_vllm_targeted_glm_plan_is_glm_only(monkeypatch, model_path):
    cases = _planned_rows(monkeypatch, "vllm", model_path)
    assert cases
    # The NVFP4 artifact is an alias of the one physical (BF16) mHC case.
    assert {case["params"][4] for case in cases} == {GLM}
    assert {case["params"][5] for case in cases} == {GLM_MODEL}


def test_sglang_plans_one_row_per_dsv4_site_and_model(monkeypatch):
    cases = _planned_rows(monkeypatch, "sglang")
    # G-sglang-mhc-glm5next-not-ported: GLM has no route for the pinned SGLang.
    assert [case["params"] for case in cases] == [
        ["pre", "deepseek-ai/DeepSeek-V4-Flash"],
        ["post", "deepseek-ai/DeepSeek-V4-Flash"],
        ["pre", "deepseek-ai/DeepSeek-V4-Pro"],
        ["post", "deepseek-ai/DeepSeek-V4-Pro"],
    ]
    assert cases[0]["id"] == "mhc_pre_hs4096_hcm4_deepseek-ai_DeepSeek-V4-Flash"


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
        # DeepSeek-V4 is not audited on 0.31.0 (shared architecture-free key, see the module)
        ("vllm", DSV4, "0.31.0", False),
        ("vllm", GLM, "0.31.0", True),
        ("vllm", GLM, "0.30.0", False),
        ("sglang", DSV4, "0.5.21", True),
        ("sglang", DSV4, "0.5.20", False),
        ("sglang", GLM, "0.5.21", False),  # not ported to 0.5.21 yet
        ("sglang", "UnknownForCausalLM", "0.5.21", False),
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

    candidates = ["0.24.0", "0.25.0", "0.27.1", "0.30.0", "0.31.0", "0.5.14", "0.5.16", "0.5.20", "0.5.21"]
    for version in candidates:
        admitted = [arch for arch, spec in module._ARCHITECTURE_COMPAT.items() if _check_compat(spec, version)]
        assert len(admitted) <= 1, (version, admitted)


def test_vllm_glm_sites_import_the_0_31_0_module_location():
    # vllm.models.glm5next.nvidia.model moved to .common.model at 0.31.0.
    source = (COLLECTOR_DIR / "vllm" / "collect_mhc_module.py").read_text(encoding="utf-8")
    assert "from vllm.models.glm5next.common.model import Glm5NextDecoderLayer" in source
    assert "vllm.models.glm5next.nvidia.model" not in source


def test_worker_rejects_unknown_architecture(monkeypatch):
    module = _load(monkeypatch, "vllm")
    with pytest.raises(module.MhcRuntimeNotAuditedError):
        module.run_mhc_module_worker("pre", 16, 4096, 4, "UnknownForCausalLM", perf_filename="/tmp/x_perf.txt")
