# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash prefix-cache KDA state checkpoint copy: plan, cases, contract, gates."""

import ast
import importlib.metadata
import sys
import types
from collections import Counter
from pathlib import Path

import pytest
import yaml
from collector import glm53_mamba_state_copy_common as common
from collector.case_generator import get_common_glm53_mamba_state_copy_test_cases
from collector.model_cases import build_collection_case_plan
from collector.op_catalog import family_for_perf_file, load_family_map
from collector.provenance import load_closures
from collector.registry_types import PerfFile
from collector.sglang import collect_glm53_mamba_state_copy as sglang_copy
from collector.sglang.registry import REGISTRY as SGLANG_REGISTRY
from collector.version_resolver import _check_compat
from collector.vllm import collect_glm53_mamba_state_copy as vllm_copy
from collector.vllm.registry import REGISTRY as VLLM_REGISTRY

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
OP = "glm53_mamba_state_checkpoint_copy"
FP8 = "zai-org/GLM-5.3-Flash"
NVFP4 = "nvidia/GLM-5.3-Flash-NVFP4"
EXPECTED_PAIRS = {
    (1, 0),
    (32, 0),
    (1, 1),
    (32, 1),
    (2, 2),
    (32, 2),
    (4, 4),
    (32, 4),
    (8, 8),
    (32, 8),
    (16, 16),
    (32, 16),
    (32, 32),
}


def _compat(relative_path: str) -> str:
    tree = ast.parse((REPO_ROOT / relative_path).read_text())
    return next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__compat__" for t in node.targets)
    )


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("model_path", [FP8, NVFP4])
def test_glm_plans_activate_the_op_on_sglang_and_vllm(backend, model_path):
    plan = build_collection_case_plan(backend=backend, model_path=model_path, gpu_type="gb300")
    assert OP in plan.ops


def test_trtllm_and_other_models_never_plan_the_op():
    assert OP not in build_collection_case_plan(backend="trtllm", model_path=FP8, gpu_type="gb300").ops
    assert OP not in build_collection_case_plan(backend="sglang", model_path="Qwen/Qwen3-32B").ops


@pytest.mark.parametrize("model_filter", [None, FP8, NVFP4])
def test_common_cases_cover_every_shard_and_copy_batch_pair(monkeypatch, model_filter):
    if model_filter:
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_filter)
    else:
        monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    cases = get_common_glm53_mamba_state_copy_test_cases()
    assert {c.model_name for c in cases} == {FP8}  # NVFP4 is a shape-only alias
    assert {(c.num_layers, c.num_heads, c.head_dim, c.conv_kernel_size) for c in cases} == {(34, 64, 128, 4)}
    by_tp = {}
    for case in cases:
        by_tp.setdefault(case.tp_size, []).append((case.batch_size, case.num_copy_requests))
    assert set(by_tp) == {1, 2, 4, 8}
    for pairs in by_tp.values():
        assert len(pairs) == len(set(pairs)) == 13
        assert set(pairs) == EXPECTED_PAIRS
    assert all(0 <= c.num_copy_requests <= c.batch_size for c in cases)


def test_other_model_filter_yields_no_cases(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "Qwen/Qwen3-32B")
    assert get_common_glm53_mamba_state_copy_test_cases() == []


def test_backend_getters_split_variants_by_framework(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    sglang_cases = sglang_copy.get_glm53_mamba_state_copy_test_cases()
    vllm_cases = vllm_copy.get_glm53_mamba_state_copy_test_cases()
    assert Counter(case[0] for case in sglang_cases) == {"sglang_decode": 52, "sglang_prefill": 52}
    assert Counter(case[0] for case in vllm_cases) == {"vllm_precopy": 52}
    for cases in (sglang_cases, vllm_cases):
        assert len({tuple(case) for case in cases}) == len(cases)
    assert vllm_cases[0] == ["vllm_precopy", 1, 34, 64, 128, 4, 1, 0, FP8]


@pytest.mark.parametrize(
    "registry,module",
    [
        (SGLANG_REGISTRY, "collector.sglang.collect_glm53_mamba_state_copy"),
        (VLLM_REGISTRY, "collector.vllm.collect_glm53_mamba_state_copy"),
    ],
)
def test_registry_routes_the_op_to_its_module_and_table(registry, module):
    (entry,) = [e for e in registry if e.op == OP]
    assert entry.module == module
    assert entry.perf_filename == PerfFile.GLM53_MAMBA_STATE_COPY
    assert str(entry.perf_filename) == "glm53_mamba_state_copy_perf.txt"
    assert not entry.unverified and not entry.unverified_sms and not entry.versions
    loaded = sys.modules[module]
    assert callable(getattr(loaded, entry.get_func))
    assert callable(getattr(loaded, entry.run_func))


def test_table_maps_to_its_own_family_and_hash_closures_cover_shared_code():
    family_map = load_family_map()
    assert family_for_perf_file(str(PerfFile.GLM53_MAMBA_STATE_COPY), family_map) == "mamba_state_copy"
    closures = load_closures(REPO_ROOT / "collector" / "hash_closures.yaml")
    for module in (
        "collector.sglang.collect_glm53_mamba_state_copy",
        "collector.vllm.collect_glm53_mamba_state_copy",
    ):
        assert "collector/glm53_mamba_state_copy_common.py" in closures[module]
        assert "collector/cases/base_ops/glm53_mamba_state_checkpoint_copy.yaml" in closures[module]


def test_kernel_sources_have_backend_translations():
    data = yaml.safe_load((REPO_ROOT / "collector" / "kernel_source_backends.yaml").read_text())
    mapped = {(m["framework"], m["kernel_source"]): m["backend"] for m in data["mappings"]}
    assert mapped[("sglang", "track_mamba_states_all_layers_kernel")] == "triton"
    assert mapped[("sglang", "index_gather_put_per_layer")] == "torch"
    assert mapped[("vllm", "precopy_mamba_align_fused_kernel")] == "triton"
    for framework, kernel_source, _graph_mode in common.VARIANTS.values():
        assert (framework, kernel_source) in mapped


def test_row_contract_columns_and_variant_metadata():
    row = common.build_row(
        variant="sglang_prefill",
        tp_size=2,
        num_layers=34,
        num_heads=32,
        head_dim=128,
        conv_dim=12288,
        conv_width=3,
        ssm_bytes_per_req_layer=2097152,
        conv_bytes_per_req_layer=73728,
        batch_size=4,
        num_copy_requests=4,
        latency=7.5,
        gpu_time=1.0,
    )
    assert tuple(row) == (
        "variant",
        "tp_size",
        "num_layers",
        "num_heads",
        "head_dim",
        "conv_dim",
        "conv_width",
        "ssm_bytes_per_req_layer",
        "conv_bytes_per_req_layer",
        "batch_size",
        "num_copy_requests",
        "graph_mode",
        "latency",
        "gpu_time",
    )
    assert row["graph_mode"] == "eager"
    assert common.VARIANTS == {
        "sglang_decode": ("sglang", "track_mamba_states_all_layers_kernel", "cuda_graph"),
        "sglang_prefill": ("sglang", "index_gather_put_per_layer", "eager"),
        "vllm_precopy": ("vllm", "precopy_mamba_align_fused_kernel", "eager"),
    }


def test_persist_row_writes_op_name_and_kernel_source():
    calls = []

    def fake_log_perf(**kwargs):
        calls.append(kwargs)
        return True

    row = {"variant": "vllm_precopy"}
    common.persist_row(
        row,
        framework_label="VLLM",
        version="0.31.0",
        device_name="GB300",
        perf_filename="x",
        log_perf=fake_log_perf,
    )
    assert calls[0]["op_name"] == OP
    assert calls[0]["kernel_source"] == "precopy_mamba_align_fused_kernel"
    assert calls[0]["item_list"] == [row]
    with pytest.raises(RuntimeError, match="failed to persist"):
        common.persist_row(
            row, framework_label="VLLM", version="v", device_name="d", perf_filename="x", log_perf=lambda **_: False
        )


@pytest.mark.parametrize(
    "framework,version,accepted",
    [
        ("sglang", "0.5.20", True),
        ("sglang", "0.5.19", False),
        ("sglang", "0.5.14", False),
        ("sglang", "0.5.21", False),
        ("vllm", "0.31.0", True),
        ("vllm", "0.31.0+cu130", True),
        ("vllm", "0.30.0", False),
        ("vllm", "0.30.0+glm53tail.eb4704514fdf", False),
        ("vllm", "0.29.0", False),
        ("vllm", "0.31.1", False),
        ("vllm", "0.24.0", False),
    ],
)
def test_runtime_gate_admits_only_the_audited_release(framework, version, accepted):
    path = f"collector/{framework}/collect_glm53_mamba_state_copy.py"
    assert _check_compat(_compat(path), version) is accepted
    if accepted:
        common.require_audited_runtime(framework, version)
    else:
        with pytest.raises(common.MambaStateCopyRuntimeNotAuditedError):
            common.require_audited_runtime(framework, version)


def test_sglang_run_raises_classified_error_before_touching_the_gpu(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.5.19")
    with pytest.raises(common.MambaStateCopyRuntimeNotAuditedError):
        sglang_copy.run_glm53_mamba_state_copy("sglang_decode", 2, 34, 64, 128, 4, 1, 1, FP8, perf_filename="unused")


def test_vllm_run_raises_classified_error_before_touching_the_gpu(monkeypatch):
    vllm_pkg = types.ModuleType("vllm")
    vllm_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    monkeypatch.setitem(sys.modules, "vllm.version", types.SimpleNamespace(__version__="0.29.0"))
    with pytest.raises(common.MambaStateCopyRuntimeNotAuditedError):
        vllm_copy.run_glm53_mamba_state_copy("vllm_precopy", 2, 34, 64, 128, 4, 1, 1, FP8, perf_filename="unused")


@pytest.mark.parametrize(
    "framework,variant,tp,batch,copies",
    [
        ("sglang", "vllm_precopy", 2, 1, 1),
        ("vllm", "sglang_decode", 2, 1, 1),
        ("sglang", "sglang_decode", 3, 1, 1),
        ("sglang", "sglang_prefill", 2, 2, 3),
        ("vllm", "vllm_precopy", 2, 0, 0),
    ],
)
def test_validate_case_rejects_inconsistent_cases(framework, variant, tp, batch, copies):
    with pytest.raises(ValueError):
        common.validate_case(framework, variant, tp, 64, batch, copies)


def test_geometry_check_fails_loudly_on_mismatch():
    kwargs = dict(num_heads=64, tp_size=2, head_dim=128, conv_kernel_size=4, local_head_dim=128, conv_width=3)
    common.check_local_geometry(local_heads=32, conv_dim=12288, **kwargs)
    with pytest.raises(RuntimeError, match="geometry"):
        common.check_local_geometry(local_heads=32, conv_dim=24576, **kwargs)


def test_vllm_mamba_groups_split_like_the_hybrid_kv_cache_grouping():
    groups = vllm_copy.mamba_group_layer_names(34)
    assert [len(g) for g in groups] == [9, 9, 8, 8]
    flat = [name for group in groups for name in group]
    assert sorted(flat) == sorted(set(flat)) and len(flat) == 34
    assert groups[1][:2] == ["language_model.model.layers.1.linear_attn", "language_model.model.layers.5.linear_attn"]


def test_vllm_precopy_idx_mapping_matches_serving_int32_upload():
    # vLLM 0.31.0 uploads idx_mapping as int32 (v1/worker/gpu/model_runner.py:1336-1338).
    tree = ast.parse(Path(vllm_copy.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_time_precopy")
    source = ast.get_source_segment(Path(vllm_copy.__file__).read_text(), fn)
    assert "idx_mapping = torch.arange(batch_size, dtype=torch.int32, device=device)" in source


def test_collector_modules_import_without_torch():
    # The getters run on the CPU planner, so neither module may import torch
    # or the framework at module scope.
    for module in (sglang_copy, vllm_copy, common):
        tree = ast.parse(Path(module.__file__).read_text())
        top_level_imports = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_level_imports.add((node.module or "").split(".")[0])
        assert not top_level_imports & {"torch", "sglang", "vllm"}
