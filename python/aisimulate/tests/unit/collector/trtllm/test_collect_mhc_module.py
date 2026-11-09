# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from collector.case_generator import get_common_mhc_test_cases
from collector.model_cases import build_collection_case_plan
from collector.registry_types import PerfFile
from collector.trtllm.registry import REGISTRY

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_module_with_torch_stub(monkeypatch):
    """Import the collector with a stub torch so case population is testable
    without a GPU env (torch is only used inside run-path functions)."""
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    module_path = REPO_ROOT / "collector" / "trtllm" / "collect_mhc_module.py"
    spec = importlib.util.spec_from_file_location("trtllm_mhc_target", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The getter plans for the device's SM; mHC has no SM-dependent region.
    monkeypatch.setattr(module, "get_sm_version", lambda: 100)
    return module


def test_registry_entry_wires_mhc_module():
    entries = {entry.op: entry for entry in REGISTRY}
    entry = entries["mhc_module"]
    assert entry.module == "collector.trtllm.collect_mhc_module"
    assert entry.get_func == "get_mhc_module_test_cases"
    assert entry.run_func == "run_mhc_module_worker"
    assert entry.perf_filename == PerfFile.MHC_MODULE
    # New collector stays default-open: SM markers only with probe evidence
    # (failure_handling.md decision tree).
    assert entry.unverified is False
    assert entry.unverified_sms == ()


def test_trtllm_dsv4_plan_schedules_mhc_module():
    plan = build_collection_case_plan(backend="trtllm", model_path="sgl-project/DeepSeek-V4-Pro-FP8")
    assert "mhc_module" in plan.selected_ops
    assert "mhc_module" in plan.ops


def test_mhc_case_population_dedups_model_aliases(monkeypatch):
    module = _load_module_with_torch_stub(monkeypatch)

    cases = module.get_mhc_module_test_cases()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"

    # Physical dedup: the 4 DeepSeek-V4 model paths collapse to 2 geometries x
    # 2 sites (skips.py leaves GLM-5.3-Flash out of TRT-LLM); the full
    # sweep is (site, hidden, hc_mult) x num_tokens.
    keys = {(case["params"][0], case["params"][2], case["params"][3]) for case in cases}
    assert keys == {
        ("pre", 4096, 4),
        ("pre", 7168, 4),
        ("post", 4096, 4),
        ("post", 7168, 4),
    }

    expected_tokens = {
        (case.phase, case.hidden_size, case.hc_mult): case.num_tokens_list for case in get_common_mhc_test_cases()
    }
    assert len(cases) == sum(len(tokens) for tokens in expected_tokens.values())
    for case in cases:
        phase, num_tokens, hidden_size, hc_mult = case["params"]
        assert num_tokens in expected_tokens[(phase, hidden_size, hc_mult)]


def test_mhc_worker_params_match_run_signature(monkeypatch):
    module = _load_module_with_torch_stub(monkeypatch)

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(module, "run_mhc_module", fake_run)
    case = module.get_mhc_module_test_cases()[0]
    module.run_mhc_module_worker(*case["params"], perf_filename="/tmp/out/mhc_module_perf.txt")

    assert captured["ops"] == [case["params"][0]]
    assert captured["num_tokens_cases"] == [case["params"][1]]
    assert captured["hidden_size"] == case["params"][2]
    assert captured["hc_mult"] == case["params"][3]
    assert captured["perf_filename"] == "mhc_module_perf.txt"
    assert captured["output_path"] == "/tmp/out"


def _stub_trtllm_module(monkeypatch, name: str, **attrs) -> ModuleType:
    """Install ``name`` (and stub parents) so the collector's lazy TRT-LLM imports resolve."""
    parts = name.split(".")
    for depth in range(1, len(parts)):
        parent = ".".join(parts[:depth])
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, ModuleType(parent))
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_sites_are_built_by_servings_make_mhc_hook(monkeypatch):
    module = _load_module_with_torch_stub(monkeypatch)
    monkeypatch.setattr(module.torch, "no_grad", contextlib.nullcontext, raising=False)
    seen = {}

    class _Param:
        def normal_(self):
            return self

        def fill_(self, value):
            return self

    class _Site:
        fn = base = scale = _Param()

        def to(self, device):
            seen["device"] = device
            return self

    class DeepseekV4DecoderLayer:
        # Serving's hook reads only self.config (modeling_deepseekv4.py:1923-1937@v1.4.0rc2).
        def _make_mhc(self):
            seen["config"] = vars(self.config)
            return _Site()

    _stub_trtllm_module(
        monkeypatch, "tensorrt_llm._torch.models.modeling_deepseekv4", DeepseekV4DecoderLayer=DeepseekV4DecoderLayer
    )
    module._make_mhc_site(7168, 4, device="cuda:1")
    assert seen == {"config": {"hc_mult": 4, "hidden_size": 7168, "hc_sinkhorn_iters": 20}, "device": "cuda:1"}


@pytest.mark.parametrize(("num_splits", "label"), [(16, "dg_splitk"), (1, "dg_nosplit")])
def test_pre_backend_label_reads_the_deepgemm_split(monkeypatch, num_splits, label):
    module = _load_module_with_torch_stub(monkeypatch)
    monkeypatch.setattr(module.torch, "no_grad", contextlib.nullcontext, raising=False)
    mhc_cuda = ModuleType("mhc_cuda")
    mhc_cuda.mhc_gemm_rms_fma_cuda = lambda *args, **kwargs: None
    mhc_cuda.mhc_gemm_rms_dg_cuda = lambda *args, **kwargs: None
    originals = (mhc_cuda.mhc_gemm_rms_fma_cuda, mhc_cuda.mhc_gemm_rms_dg_cuda)
    _stub_trtllm_module(monkeypatch, "tensorrt_llm._torch.modules.mhc", mhc_cuda=mhc_cuda)

    class _Site:
        def pre_mapping(self, residual):
            # MhcPreMappingRunner.forward's DeepGEMM calls (mhc_cuda.py:365-369@v1.4.0rc2).
            return mhc_cuda.mhc_gemm_rms_dg_cuda(residual, None, 16, 24, 16384, num_splits=num_splits)

    assert module._observe_pre_backend(_Site(), object()) == label
    assert (mhc_cuda.mhc_gemm_rms_fma_cuda, mhc_cuda.mhc_gemm_rms_dg_cuda) == originals
