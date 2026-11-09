# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic native-interface fixtures, not GPU or performance qualification."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.fpm_forward import runtime_k31, runtime_memory
from collector.fpm_forward.runtime import fpm_memory_observer as legacy
from collector.fpm_forward.runtime_instrumentation import load_instrumentation
from collector.fpm_forward.runtime_observations import _SEMANTICS, _physical_views, validate_observations

from .test_fpm_runner import _cell
from .test_fpm_runtime_memory import _Storage, _typed
from .test_runtime_observations import _replace_launch, observation_fixture

pytestmark = pytest.mark.unit


def _adapter(monkeypatch):
    monkeypatch.setitem(sys.modules, "fpm_memory_observer", legacy)
    return importlib.import_module("collector.fpm_forward.runtime.instrumentation.kimi_k31")


class _Tensor:
    def __init__(self, storage, offset, shape, dtype, element_size):
        self.storage, self.offset, self.shape = storage, offset, shape
        self.dtype, self.size = dtype, element_size

    def untyped_storage(self):
        return self.storage

    def element_size(self):
        return self.size

    def storage_offset(self):
        return self.offset // self.size

    def stride(self):
        result, step = [], 1
        for dim in reversed(self.shape[1:]):
            result.append(step)
            step *= dim
        return (512 // self.size, *reversed(result))


def _native_fixture(monkeypatch, *, uniform=False):
    adapter = _adapter(monkeypatch)
    storage = _Storage(1234, 5120)
    layers, groups, placements = {}, [], []
    for stem, page, width in (("mla", 128, 32), ("indexer", 64, 16), ("kda", 64, None)):
        spec = _typed("vllm.v1.kv_cache_interface", "MambaSpec" if stem == "kda" else "MLAAttentionSpec")
        spec.block_size, spec.page_size_bytes = (4096 if stem == "kda" else 4), page
        if stem == "kda":
            spec.shapes, spec.dtypes = ((2, 4), (2, 4)), ("torch.bfloat16", "torch.float32")
            spec.state_content_size_bytes, spec.page_size_padded = 48, 64
            spec.mamba_cache_mode, spec.num_speculative_blocks = "none", 0
            spec.num_prefill_checkpoint_blocks, spec.prefill_checkpoint_alignment = 1, 64
            spec.max_num_blocks_per_req = lambda *_: 1
        else:
            spec.dtype, spec.cache_dtype_str = "torch.uint8", "fp8"
        names = [f"{stem}.{i}" for i in range(2)]
        groups.append(SimpleNamespace(layer_names=names, kv_cache_spec=spec, host_resident=False, is_eagle_group=False))
        placements.append(
            SimpleNamespace(size=5120, layers=names, layer_stride=page, block_stride=512, offset=0, host_resident=False)
        )
        for i, name in enumerate(names):
            module, class_name = {
                "kda": ("vllm.models.kimi_k3.nvidia.kda", "KimiK3DeltaAttention"),
                "mla": ("vllm.model_executor.layers.attention.mla_attention", "MLAAttention"),
                "indexer": ("vllm.models.kimi_k3.common.dsa_cache", "KimiK3IndexerCache"),
            }[stem]
            layer = _typed(module, class_name)
            layer.kv_cache = (
                (
                    _Tensor(storage, i * page, (10, 2, 4), "torch.bfloat16", 2),
                    _Tensor(storage, i * page + 16, (10, 2, 4), "torch.float32", 4),
                )
                if stem == "kda"
                else _Tensor(storage, i * page, (10, 4, width), "torch.uint8", 1)
            )
            layers[name] = layer
    if uniform:
        # Native packed grouping places MLA and indexer in the SAME token
        # block table while retaining their different per-layer page sizes.
        placements[1].offset = 256
        for name in groups[1].layer_names:
            layers[name].kv_cache.offset += 256
        wrapped = []
        for entries in (groups[:2], groups[2:]):
            spec = _typed("vllm.v1.kv_cache_interface", "UniformTypeKVCacheSpecs")
            spec.kv_cache_specs = {name: entry.kv_cache_spec for entry in entries for name in entry.layer_names}
            spec.block_size = entries[0].kv_cache_spec.block_size
            spec.page_size_bytes = sum(child.page_size_bytes for child in spec.kv_cache_specs.values())
            wrapped.append(
                SimpleNamespace(
                    layer_names=list(spec.kv_cache_specs), kv_cache_spec=spec, host_resident=False, is_eagle_group=False
                )
            )
        groups = wrapped
    config = SimpleNamespace(num_blocks=10, kv_cache_groups=groups, kv_cache_tensors=placements)
    vllm = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=4096, hf_config=SimpleNamespace(architectures=["KimiK31ForConditionalGeneration"])
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False, mamba_cache_mode="none"),
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context=layers),
    )
    worker = SimpleNamespace(
        vllm_config=vllm,
        _fpm_cache_initialized=True,
        _fpm_available_cache_bytes=5200,
        model_runner=SimpleNamespace(kv_cache_config=config, kv_caches=[x.kv_cache for x in layers.values()]),
    )
    pool = SimpleNamespace(
        num_gpu_blocks=10, get_num_free_blocks=lambda: 9, null_block=SimpleNamespace(is_null=True, block_id=0)
    )
    scheduler = SimpleNamespace(
        vllm_config=vllm,
        kv_cache_manager=SimpleNamespace(
            block_pool=pool,
            watermark_blocks=0,
            coordinator=SimpleNamespace(single_type_managers=[SimpleNamespace(block_pool=pool) for _ in groups]),
        ),
    )
    cache, scheduled = adapter.worker_memory(worker), adapter.scheduler_memory(scheduler, config)
    for record in (cache, scheduled):
        record["pool_id"] = "pool"
        for group in record["groups"]:
            group.update(pool_id="pool", extra_retained_tokens=0, attention_chunk_size=None)
    cache["semantics"] = {**_SEMANTICS, "retention": "full_or_state"}
    scheduled.update(group_pool_ids=["pool"] * len(groups), permanent_reserved_block_ids=[0])
    return cache, scheduled, worker


def test_native_mixed_state_and_overlaid_groups_charge_actual_padded_pool(monkeypatch):
    cache, scheduled, _ = _native_fixture(monkeypatch)
    groups, accounting = _physical_views(cache, scheduled, "fp8")
    assert accounting["page_size_bytes"] == 512
    assert [group["kind"] for group in groups] == ["attention", "attention", "state"]
    assert groups[-1]["blocks_per_request"] == 1
    assert groups[-1]["page_size_bytes"] == 512
    assert [v["dtype"] for v in cache["layer_components"]["kda.0"]] == ["torch.bfloat16", "torch.float32"]
    assert cache["allocated_cache_bytes"] == 5120
    assert cache["groups"][-1]["state_content_size_bytes"] == 48
    assert cache["groups"][-1]["spec_page_size_bytes"] == 64


def test_native_uniform_wrapper_preserves_one_attention_block_table(monkeypatch):
    cache, scheduled, _ = _native_fixture(monkeypatch, uniform=True)
    groups, accounting = _physical_views(cache, scheduled, "fp8")
    assert len(groups) == 2
    assert groups[0]["kind"] == "attention" and groups[0]["num_layers"] == 4
    assert groups[1]["kind"] == "state" and groups[1]["blocks_per_request"] == 1
    assert accounting["page_size_bytes"] == 512
    assert cache["groups"][0]["spec_page_size_bytes"] == 384
    assert cache["groups"][0]["layer_specs"]["indexer.0"]["spec_page_size_bytes"] == 64
    cache["tensor_allocations"][1]["offset"] = 128
    for name in ("indexer.0", "indexer.1"):
        cache["layer_components"][name][0]["storage_offset_bytes"] -= 128
    with pytest.raises(ValueError, match="within one group overlap"):
        _physical_views(cache, scheduled, "fp8")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda g: g["layer_specs"].pop("indexer.0"), "every native per-layer"),
        (lambda g: g.update(spec_page_size_bytes=385), "page sum"),
        (lambda g: g["layer_specs"]["indexer.0"].update(block_size_tokens=8), "incompatible per-layer"),
    ],
)
def test_native_uniform_wrapper_requires_exact_child_geometry(monkeypatch, mutation, match):
    cache, scheduled, _ = _native_fixture(monkeypatch, uniform=True)
    for record in (cache, scheduled):
        mutation(record["groups"][0])
    with pytest.raises(ValueError, match=match):
        _physical_views(cache, scheduled, "fp8")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c, s: c["layer_components"]["kda.0"].pop(), "component count"),
        (lambda c, s: c["layer_components"]["kda.0"][1].update(dtype="torch.uint8"), "shape/dtype"),
        (lambda c, s: c["layer_components"]["kda.0"][1].update(storage_offset_bytes=0), "offsets"),
        (lambda c, s: c["layer_components"]["kda.0"][0]["stride_bytes"].__setitem__(0, 256), "coverage"),
        (lambda c, s: c["tensor_allocations"][-1].update(layer_stride=32), "overlap"),
        (lambda c, s: c["tensor_allocations"][-1].update(block_stride=64), "block-outer"),
        (lambda c, s: c["tensor_allocations"][-1].update(host_resident=True), "device"),
        (lambda c, s: c["tensor_allocations"].pop(), "omit"),
        (lambda c, s: c["layer_components"]["mla.0"][0]["shape"].__setitem__(0, 20), "coverage"),
        (lambda c, s: s["groups"][-1].update(mamba_cache_mode="align"), "geometry"),
    ],
)
def test_native_component_or_allocation_corruption_is_rejected(monkeypatch, mutation, match):
    cache, scheduled, _ = _native_fixture(monkeypatch)
    mutation(cache, scheduled)
    with pytest.raises(ValueError, match=match):
        _physical_views(cache, scheduled, "fp8")


@pytest.mark.parametrize(
    "key,value",
    [
        ("mamba_cache_mode", "align"),
        ("num_speculative_blocks", 1),
        ("blocks_per_request", 2),
        ("block_size_tokens", 128),
    ],
)
def test_native_state_mode_cannot_be_relabelled_fixed_one_page(monkeypatch, key, value):
    cache, scheduled, _ = _native_fixture(monkeypatch)
    for record in (cache, scheduled):
        record["groups"][-1][key] = value
    with pytest.raises(ValueError, match="one non-speculative state page"):
        _physical_views(cache, scheduled, "fp8")


def _moe():
    return {
        "enumeration": "model.named_modules:RoutedExperts",
        "expected_layer_names": ["model.layers.1.experts"],
        "layers": [
            {
                "name": "model.layers.1.experts",
                "layer_type": "vllm.model_executor.layers.fused_moe.routed_experts.RoutedExperts",
                "method_type": "vllm.model_executor.layers.quantization.mxfp4.Mxfp4MoEMethod",
                "kernel_type": "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernel",
                "backend": "FLASHINFER_TRTLLM_MXFP4_MXFP8",
                "descriptors": {
                    x: {"dtype": d} for x, d in (("_a1", "mxfp8"), ("_a2", "mxfp8"), ("_w1", "mxfp4"), ("_w2", "mxfp4"))
                },
                "weight_shapes": {"w13_weight": [8, 64, 32], "w2_weight": [8, 32, 64]},
            }
        ],
    }


def test_initialized_expert_arithmetic_selects_exact_mode_and_rejects_metadata_guess():
    evidence = _moe()
    assert runtime_k31.moe_precision(evidence) == "w4a8_mxfp4_mxfp8_trtllm"
    config = {
        "model_config": {
            "dtype": "bfloat16",
            "quantization": "mxfp4",
            "architectures": ["KimiK31ForConditionalGeneration"],
        },
        "quantization_config": {"type": "vllm.model_executor.layers.quantization.mxfp4.Mxfp4Config"},
    }
    cell = replace(_cell(), gemm_quant_mode="bfloat16", moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm")
    runtime_memory._validate_precision(config, cell, evidence=evidence)
    with pytest.raises(ValueError, match="initialized MoE backend"):
        runtime_memory._validate_precision(config, cell)
    for mode in ("w4a16_mxfp4", "w4a8_mxfp4_mxfp8", "bfloat16"):
        with pytest.raises(ValueError, match="planned GEMM/MoE"):
            runtime_memory._validate_precision(config, replace(cell, moe_quant_mode=mode), evidence=evidence)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda e: e["layers"][0]["descriptors"]["_a2"].update(dtype="None"), "activation/weight"),
        (lambda e: e["layers"][0]["descriptors"]["_w2"].update(dtype="nvfp4"), "activation/weight"),
        (lambda e: e["layers"][0].update(backend="FLASHINFER_TRTLLM_MXFP4_BF16"), "backend"),
        (lambda e: e["layers"][0].update(kernel_type="NoneType"), "method/kernel"),
        (lambda e: e["expected_layer_names"].append("unobserved.experts"), "coverage"),
        (lambda e: e["layers"].append(copy.deepcopy(e["layers"][0])), "coverage"),
    ],
)
def test_initialized_precision_is_not_inferred_from_model_dtype(mutation, match):
    evidence = _moe()
    mutation(evidence)
    with pytest.raises(ValueError, match=match):
        runtime_k31.moe_precision(evidence)


def test_preview_bundle_uses_exact_source_map_not_invented_git_commit():
    path = Path(runtime_k31.__file__).parent / "runtime/vllm-kimi-k31.example.json"
    bundle = load_instrumentation(path)
    assert runtime_k31.is_audited_runtime(bundle.manifest["runtime"])
    altered = copy.deepcopy(bundle.manifest["runtime"])
    altered["source_files"]["vllm/v1/kv_cache_interface.py"] = "0" * 64
    assert not runtime_k31.is_audited_runtime(altered)
    assert "instrumentation/kimi_k31.py" in bundle.files
    assert json.loads(path.read_text())["runtime"]["version"] == runtime_k31.VERSION


def test_native_expert_enumeration_requires_every_initialized_kimi_moe(monkeypatch):
    adapter = _adapter(monkeypatch)
    routed = _typed("vllm.model_executor.layers.fused_moe.routed_experts", "RoutedExperts")
    kimi = _typed("vllm.models.kimi_k3.nvidia.model", "KimiMoE")
    method = _typed("vllm.model_executor.layers.quantization.mxfp4", "Mxfp4MoEMethod")
    method.moe_kernel = _typed("vllm.model_executor.layers.fused_moe.modular_kernel", "FusedMoEKernel")
    method.mxfp4_backend = SimpleNamespace(name="FLASHINFER_TRTLLM_MXFP4_MXFP8")
    method.moe_quant_config = SimpleNamespace(
        **{
            key: SimpleNamespace(dtype=dtype)
            for key, dtype in (("_a1", "mxfp8"), ("_a2", "mxfp8"), ("_w1", "mxfp4"), ("_w2", "mxfp4"))
        }
    )
    routed.quant_method = method
    routed.w13_weight, routed.w2_weight = SimpleNamespace(shape=(8, 64, 32)), SimpleNamespace(shape=(8, 32, 64))
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.fused_moe.routed_experts", SimpleNamespace(RoutedExperts=type(routed))
    )
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3.nvidia.model", SimpleNamespace(KimiMoE=type(kimi)))
    modules = {"layer.1.moe": kimi, "layer.1.moe.experts": routed}
    owner = SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(named_modules=lambda: modules.items())))
    evidence = adapter.initialized_moe(owner)
    assert evidence["expected_layer_names"] == ["layer.1.moe.experts"]
    assert runtime_k31.moe_precision(evidence) == "w4a8_mxfp4_mxfp8_trtllm"
    method.moe_quant_config._a2.dtype = None
    with pytest.raises(ValueError, match="activation/weight"):
        runtime_k31.moe_precision(adapter.initialized_moe(owner))
    modules["layer.2.moe"] = type(kimi)()
    with pytest.raises(ValueError, match="exactly one audited RoutedExperts"):
        adapter.initialized_moe(owner)


@pytest.mark.parametrize("setting,value", [("enable_prefix_caching", True), ("mamba_cache_mode", "align")])
def test_observer_preserves_unsupported_runtime_mode_as_failure(monkeypatch, setting, value):
    adapter = _adapter(monkeypatch)
    _, _, worker = _native_fixture(monkeypatch)
    setattr(worker.vllm_config.cache_config, setting, value)
    with pytest.raises(ValueError, match="prefix caching off"):
        adapter.worker_memory(worker)


def test_preview_source_digest_checks_actual_files_and_wheel_commit(monkeypatch, tmp_path):
    _adapter(monkeypatch)
    observer = importlib.import_module("collector.fpm_forward.runtime.instrumentation.observer")
    path = tmp_path / "source.py"
    path.write_text("# synthetic source identity fixture\n")
    source_map = {"vllm/source.py": hashlib.sha256(path.read_bytes()).hexdigest()}
    source_revision = hashlib.sha256(json.dumps(source_map, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "runtime": {"version": runtime_k31.VERSION, "source_revision": source_revision, "source_files": source_map}
    }
    monkeypatch.setattr(observer.importlib.metadata, "version", lambda _: runtime_k31.VERSION)
    monkeypatch.setattr(observer, "_installed_file", lambda _: path)
    real_import = observer.importlib.import_module
    build = SimpleNamespace(__commit_id__="g266e88c05")
    monkeypatch.setattr(
        observer.importlib, "import_module", lambda name: build if name == "vllm._version" else real_import(name)
    )
    payload = {"launch": {"deployment": {"image": "fixture@sha256:" + "a" * 64}}}
    observer._runtime(payload, manifest, runtime_k31.VERSION)
    assert payload["source_identity"]["vllm_commit_id"] == "g266e88c05"
    assert payload["runtime"]["source_files"] == source_map
    path.write_text("# changed after source audit\n")
    with pytest.raises(ValueError, match="source hash differs"):
        observer._runtime(payload, manifest, runtime_k31.VERSION)
    build.__commit_id__ = "g99999999"
    with pytest.raises(ValueError, match="VCS identifier contradicts"):
        observer._runtime(payload, manifest, runtime_k31.VERSION)


def _complete_fixture(tmp_path, monkeypatch):
    path, launches = observation_fixture(tmp_path, version=runtime_k31.VERSION)
    launch = launches["tp2"]
    launch["worker_type"] = "decode"
    launch["precision"].update(
        gemm_quant_mode="bfloat16",
        moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
        kvcache_quant_mode="fp8",
        attention_backend="trtllm_mla",
    )
    _replace_launch(path, launches)
    manifest_path = tmp_path / "manifest.yaml"
    manifest = json.loads(manifest_path.read_text())
    pinned = json.loads((Path(runtime_k31.__file__).parent / "runtime/vllm-kimi-k31.example.json").read_text())[
        "runtime"
    ]
    manifest["runtime"] = pinned
    manifest_path.write_text(json.dumps(manifest))
    bundle = load_instrumentation(manifest_path)
    cache, scheduled, _ = _native_fixture(monkeypatch, uniform=True)
    index = json.loads(path.read_text())
    attempt = index["configurations"]["tp2"]["attempts"][0]
    attempt["bundle"]["sha256"] = bundle.sha256
    del attempt["phases"]["prefill"]
    phase = attempt["phases"]["decode"]
    context_path = tmp_path / phase["launch_manifest"]["path"]
    context = json.loads(context_path.read_text())
    context["bundle_sha256"] = bundle.sha256
    context_path.write_text(json.dumps(context))
    phase["launch_manifest"]["sha256"] = hashlib.sha256(context_path.read_bytes()).hexdigest()
    for ref in phase["artifacts"]:
        record_path = tmp_path / ref["path"]
        record = json.loads(record_path.read_text())
        record["bundle_sha256"] = bundle.sha256
        record["runtime"] = {**pinned, "image": launch["deployment"]["image"]}
        config = record["resolved_config"]
        config["model_config"].update(
            dtype="bfloat16", quantization="mxfp4", architectures=["KimiK31ForConditionalGeneration"]
        )
        config["quantization_config"] = {"type": "vllm.model_executor.layers.quantization.mxfp4.Mxfp4Config"}
        config["cache_config"].update(cache_dtype="fp8", mamba_cache_mode="none", enable_prefix_caching=False)
        config["attention_config"] = {"backend": "FLASHINFER_MLA_SPARSE", "indexer_kv_dtype": "fp8"}
        record["cache"] = copy.deepcopy(cache if record["kind"] == "worker" else scheduled)
        if record["kind"] == "worker":
            record["initialized_moe"] = _moe()
            record["attention_backend"] = ["FLASHINFER_MLA_SPARSE", "KIMI_K3_KDA", "DEEPSEEK_V32_INDEXER"]
        record_path.write_text(json.dumps(record))
        ref["sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(index))
    return path, launches


def test_complete_rank_import_preserves_native_grouping_state_and_precision(tmp_path, monkeypatch):
    path, launches = _complete_fixture(tmp_path, monkeypatch)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    assert len(result["resources"]["cache_groups"]) == 2
    assert result["resources"]["cache_groups"][-1]["blocks_per_request"] == 1
    assert result["resources"]["runtime_memory"]["kv_cache_bytes"] == 9 * 512
    assert len(result["provenance"]["rank_capacities"]) == 4
    for artifact in result["provenance"]["artifacts"]:
        evidence = artifact["evidence"]
        if evidence["kind"] == "worker":
            assert runtime_k31.moe_precision(evidence["initialized_moe"]) == "w4a8_mxfp4_mxfp8_trtllm"


def test_one_rank_missing_initialized_precision_keeps_import_incomplete(tmp_path, monkeypatch):
    path, launches = _complete_fixture(tmp_path, monkeypatch)
    index = json.loads(path.read_text())
    refs = index["configurations"]["tp2"]["attempts"][0]["phases"]["decode"]["artifacts"]
    target = tmp_path / refs[0]["path"]
    record = json.loads(target.read_text())
    record.pop("initialized_moe")
    target.write_text(json.dumps(record))
    refs[0]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    path.write_text(json.dumps(index))
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete" and result["resources"] is None
    assert "initialized MoE backend" in str(result["diagnostics"])
