# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from collector.fpm_forward import planner, runner, runtime_memory
from collector.fpm_forward.runtime import fpm_memory_observer as observer
from collector.fpm_forward.types import ParallelTopology

from aisimulate_core.fpm_profile import FpmResourceProfile

from .test_fpm_profile_collection import _plan, _profile, no_models_or_timing_data  # noqa: F401
from .test_fpm_runner import _cell, _native_payload, _write_provenance

pytestmark = pytest.mark.unit


def _typed(module, name):
    return type(name, (), {"__module__": module})()


class _Storage:
    device = "cuda:0"

    def __init__(self, pointer, size):
        self.pointer, self.size = pointer, size

    def data_ptr(self):
        return self.pointer

    def nbytes(self):
        return self.size


class _Tensor:
    def __init__(self, storage):
        self.storage = storage

    def untyped_storage(self):
        return self.storage


def _vllm_config(*, tp=2, dp=2):
    quant = _typed("vllm.model_executor.layers.quantization.modelopt", "ModelOptNvFp4Config")
    quant.quant_method = "NVFP4"
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="example/model",
            revision="immutable",
            dtype="torch.bfloat16",
            quantization="modelopt_fp4",
            max_model_len=4096,
            enforce_eager=False,
        ),
        cache_config=SimpleNamespace(
            cache_dtype="bfloat16", gpu_memory_utilization=0.9, kv_cache_memory_bytes=None, enable_prefix_caching=True
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024, max_num_seqs=64, async_scheduling=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            data_parallel_size=dp,
            enable_expert_parallel=True,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            mode="VLLM_COMPILE",
            cudagraph_mode="FULL_AND_PIECEWISE",
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64],
            max_cudagraph_capture_size=64,
            static_forward_context={},
        ),
        kernel_config=SimpleNamespace(moe_backend="auto"),
        quant_config=quant,
        offload_config=SimpleNamespace(
            offload_backend="auto",
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0, offload_num_in_group=1, offload_prefetch_step=1),
        ),
    )


@pytest.fixture(autouse=True)
def fake_runtime_offloader(monkeypatch):
    monkeypatch.setattr(observer, "_offloader_type", lambda: "vllm.model_executor.offloader.base.NoopOffloader")


def _initialized(*, blocks=100, free=98, packed=False, layers=False):
    """Two physical buffers, five logical layer views and three retention groups.

    With ``layers``, the allocation follows vLLM 0.31.0: one buffer backs every
    tensor, each tensor names its layers in ``layers`` with a ``layer_stride``,
    and the convolution layer's cache is a tuple of two views.
    """
    config = _vllm_config()
    one_buffer = packed or layers
    storages = [_Storage(100, blocks * (128 if one_buffer else 64))]
    storages.append(storages[0] if one_buffer else _Storage(200, blocks * 64))
    tensors = [_Tensor(storage) for storage in storages]
    context = {}
    groups = []
    for prefix, count, window, block in (("full", 2, None, 16), ("slide", 2, 32, 16), ("conv", 1, 4, 4)):
        names = [f"{prefix}.{index}" for index in range(count)]
        for index, name in enumerate(names):
            module, cls = (
                ("vllm.models.inkling.nvidia.sconv_swa_attn", "InklingConvState")
                if prefix == "conv"
                else ("vllm.model_executor.layers.attention", "Attention")
            )
            layer = _typed(module, cls)
            layer.kv_cache = (
                (_Tensor(storages[0]), _Tensor(storages[0])) if layers and prefix == "conv" else tensors[index]
            )
            context[name] = layer
        spec = _typed("vllm.v1.kv_cache_interface", "FullAttentionSpec" if window is None else "SlidingWindowSpec")
        spec.block_size, spec.page_size_bytes, spec.sliding_window, spec.dtype = block, 64, window, "torch.bfloat16"
        groups.append(SimpleNamespace(layer_names=names, kv_cache_spec=spec, is_eagle_group=False))
    config.compilation_config.static_forward_context = context
    cache_config = SimpleNamespace(
        num_blocks=blocks,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            SimpleNamespace(
                size=storages[0].size,
                shared_by=["full.0", "slide.0", "conv.0"],
                block_stride=128 if packed else 0,
                offset=0,
            ),
            SimpleNamespace(
                size=storages[1].size,
                shared_by=["full.1", "slide.1"],
                block_stride=128 if packed else 0,
                offset=64 if packed else 0,
            ),
        ],
    )
    if layers:
        # Layer l of a tensor starts at offset + l * layer_stride; stride 0 keeps
        # each tensor's aliased layers on one region of the layer-outer buffer.
        for index, tensor in enumerate(cache_config.kv_cache_tensors):
            tensor.layers = vars(tensor).pop("shared_by")
            tensor.layer_stride, tensor.block_stride, tensor.offset = 0, 64, index * blocks * 64
    worker = SimpleNamespace(
        vllm_config=config,
        _fpm_cache_initialized=True,
        _fpm_available_cache_bytes=blocks * 128 + 200,
        model_runner=SimpleNamespace(
            kv_cache_config=cache_config,
            kv_caches=[layer.kv_cache for layer in context.values()],
            shared_kv_cache_layers={},
        ),
        peak_activation_memory=123,
        cudagraph_memory_estimate=456,
    )
    pool = SimpleNamespace(
        num_gpu_blocks=blocks, get_num_free_blocks=lambda: free, null_block=SimpleNamespace(is_null=True, block_id=0)
    )
    manager = SimpleNamespace(
        block_pool=pool,
        watermark_blocks=0,
        coordinator=SimpleNamespace(single_type_managers=[SimpleNamespace(block_pool=pool) for _group in groups]),
    )
    scheduler = SimpleNamespace(vllm_config=config, kv_cache_manager=manager)
    return worker, scheduler, cache_config


def _evidence(tmp_path, monkeypatch):
    cell = replace(
        _cell(dp=2),
        topology=ParallelTopology(tp=2, pp=1, dp=2, moe_tp=1, moe_ep=4, cp=1),
        kv_cache_dtype="bfloat16",
        fmha_quant_mode="bfloat16",
    )
    monkeypatch.setattr(observer.importlib.metadata, "version", lambda _name: "0.27.0")
    for dp in range(2):
        pod = tmp_path / f"pod-{dp}"
        pod.mkdir()
        _write_provenance(pod / "collector-provenance.json", cell_id=cell.cell_id)
        provenance = json.loads((pod / "collector-provenance.json").read_text())
        provenance["runtime"]["backend_version"] = "0.27.0"
        (pod / "collector-provenance.json").write_text(json.dumps(provenance))
        (pod / "benchmark.json").write_text(json.dumps(_native_payload(phase="prefill", rank=dp, dp=2)))
        monkeypatch.setattr(observer, "RESULTS_DIR", pod)
        for tp in range(2):
            worker, scheduler, config = _initialized(blocks=100 - dp * 10, free=98 - dp * 10)
            observer.observe("worker", worker, dp_rank=dp, tp_rank=tp, pp_rank=0)
        observer.observe("scheduler", scheduler, dp_rank=dp, cache_config=config)
    return cell


def _resolve(cell, root, **kwargs):
    return runtime_memory.resolve_runtime_resources(
        cell,
        root,
        expected_plan_sha256="plan-sha",
        expected_attempt_id="attempt",
        expected_backend_version="0.27.0",
        expected_context_length=4096,
        expected_max_num_tokens=1024,
        expected_max_batch_size=64,
        expected_gpu_memory_utilization=0.9,
        **kwargs,
    )


@pytest.mark.parametrize("packed", [False, True])
def test_physical_grouped_capacity_deduplicates_aliases_and_charges_padding(packed):
    worker, scheduler, config = _initialized(packed=packed)
    observed = observer.worker_memory(worker)
    assert observed["allocated_cache_bytes"] == 12_800
    assert len(observed["storages"]) == (1 if packed else 2)
    pool = observer.scheduler_memory(scheduler, config)
    assert pool["initial_free_blocks"] == 98
    assert pool["reserved_blocks"] == 2
    assert [group["kind"] for group in observed["groups"]] == ["attention", "attention", "convolution"]
    assert [group["sliding_window"] for group in observed["groups"]] == [None, 32, 4]


def _spec(name, **fields):
    spec = _typed("vllm.v1.kv_cache_interface", name)
    vars(spec).update(fields)
    return spec


def _group(names, spec):
    return SimpleNamespace(layer_names=names, kv_cache_spec=spec, is_eagle_group=False)


def _mamba(**fields):
    # GLM-5.3-Flash KDA state at TP 4: bf16 convolution and fp32 SSM views, padded to the MLA page.
    defaults = {
        "block_size": 2176,
        "page_size_bytes": 1_114_112,
        "mamba_cache_mode": "align",
        "num_speculative_blocks": 0,
        "num_prefill_checkpoint_blocks": 0,
        "shapes": ((6144, 3), (16, 128, 128)),
        "dtypes": ("torch.bfloat16", "torch.float32"),
        "real_page_size_bytes": 1_085_440,
    }
    return _spec("MambaSpec", **{**defaults, **fields})


def test_cache_groups_map_specs_by_class_and_record_state_evidence():
    full = _spec("FullAttentionSpec", block_size=16, page_size_bytes=8192, sliding_window=None, dtype="torch.bfloat16")
    tail = _spec("KpoolTailSpec", block_size=4, page_size_bytes=71_808, sliding_window=4, dtype="torch.bfloat16")
    config = SimpleNamespace(
        kv_cache_groups=[_group(["attn.0"], full), _group(["tail.0"], tail), _group(["kda.0", "kda.1"], _mamba())]
    )
    attention, tail_group, mamba = observer.cache_groups(config)
    keys = {"layer_names", "spec_type", "block_size_tokens", "spec_page_size_bytes", "sliding_window", "dtype"}
    assert set(attention) == keys | {"kind", "layer_specs"}
    assert set(tail_group) == set(mamba) == keys | {"kind", "layer_specs", "state"}
    assert [group["kind"] for group in (attention, tail_group, mamba)] == ["attention", "state", "state"]
    assert attention["layer_specs"] == {
        "attn.0": {
            "spec_type": "vllm.v1.kv_cache_interface.FullAttentionSpec",
            "page_size_bytes": 8192,
            "dtype": "torch.bfloat16",
            "tokens_per_state": None,
            "cache_role": None,
            "storage_block_size": None,
        }
    }
    assert tail_group["state"] == {"blocks_per_request": 1}
    assert tail_group["sliding_window"] is None
    assert mamba["dtype"] is None
    assert mamba["state"] == {
        "mamba_cache_mode": "align",
        "num_speculative_blocks": 0,
        "num_prefill_checkpoint_blocks": 0,
        "shapes": [[6144, 3], [16, 128, 128]],
        "dtypes": ["torch.bfloat16", "torch.float32"],
        "real_page_size_bytes": 1_085_440,
        "blocks_per_request": 1,
    }
    assert mamba["layer_specs"]["kda.1"]["spec_type"].endswith(".MambaSpec")
    kinds = {
        **dict.fromkeys(
            (
                "FullAttentionSpec",
                "MLAAttentionSpec",
                "SinkFullAttentionSpec",
                "RSWASpec",
                "HiddenStateCacheSpec",
                "SlidingWindowSpec",
                "SlidingWindowMLASpec",
            ),
            "attention",
        ),
        **dict.fromkeys(("KpoolTailSpec", "CircularBufferSpec", "MambaSpec"), "state"),
    }
    assert {name: observer.spec_kind(_spec(name)) for name in kinds} == kinds


def test_uniform_type_groups_record_their_inner_specs():
    role = Enum("SparseCacheRole", {"SPARSE": "sparse", "INDEXER": "indexer"}, type=str)
    latent, indexer = (
        _spec(
            "MLAAttentionSpec",
            block_size=2176,
            page_size_bytes=page,
            sliding_window=None,
            dtype="torch.uint8",
            tokens_per_state=tokens,
            cache_role=cache_role,
            storage_block_size=storage,
        )
        for page, tokens, cache_role, storage in ((1_114_112, 1, role.SPARSE, None), (71_808, 4, role.INDEXER, 128))
    )
    names = ["layers.3.mla", "layers.3.indexer"]
    uniform = _spec(
        "UniformTypeKVCacheSpecs",
        block_size=2176,
        page_size_bytes=1_185_920,
        kv_cache_specs=dict(zip(names, (latent, indexer), strict=True)),
    )
    config = SimpleNamespace(kv_cache_groups=[_group(names, uniform)])
    (group,) = observer.cache_groups(config)
    assert group["kind"] == "attention"
    assert group["spec_type"] == "vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs"
    assert (group["block_size_tokens"], group["spec_page_size_bytes"]) == (2176, 1_185_920)
    assert (group["sliding_window"], group["dtype"]) == (None, "torch.uint8")
    assert "state" not in group
    assert group["layer_specs"] == {
        "layers.3.mla": {
            "spec_type": "vllm.v1.kv_cache_interface.MLAAttentionSpec",
            "page_size_bytes": 1_114_112,
            "dtype": "torch.uint8",
            "tokens_per_state": 1,
            "cache_role": "SPARSE",
            "storage_block_size": None,
        },
        "layers.3.indexer": {
            "spec_type": "vllm.v1.kv_cache_interface.MLAAttentionSpec",
            "page_size_bytes": 71_808,
            "dtype": "torch.uint8",
            "tokens_per_state": 4,
            "cache_role": "INDEXER",
            "storage_block_size": 128,
        },
    }
    # A bf16 latent beside the uint8 indexer has no single group dtype; each layer keeps its own.
    latent.dtype = "torch.bfloat16"
    (group,) = observer.cache_groups(config)
    assert group["dtype"] is None
    assert [spec["dtype"] for spec in group["layer_specs"].values()] == ["torch.bfloat16", "torch.uint8"]


def test_mixed_uniform_group_and_unknown_class_are_unsupported():
    latent = _spec("MLAAttentionSpec", block_size=2176, page_size_bytes=1_114_112, dtype="torch.uint8")
    mixed = _spec(
        "UniformTypeKVCacheSpecs",
        block_size=2176,
        page_size_bytes=2_228_224,
        kv_cache_specs={"mla.0": latent, "kda.0": _mamba()},
    )
    with pytest.raises(
        ValueError, match="unsupported runtime cache spec: .*UniformTypeKVCacheSpecs.*MLAAttentionSpec.*MambaSpec"
    ):
        observer.cache_groups(SimpleNamespace(kv_cache_groups=[_group(["mla.0", "kda.0"], mixed)]))
    for spec in (_spec("HiSparseHotSpec"), _typed("vendor.kv_cache_interface", "FullAttentionSpec")):
        name = f"{type(spec).__module__}.{type(spec).__name__}"
        with pytest.raises(ValueError, match=re.escape(f"unsupported runtime cache spec: {name}")):
            observer.cache_groups(SimpleNamespace(kv_cache_groups=[_group(["layer.0"], spec)]))


@pytest.mark.parametrize(
    "change,match",
    [
        ("names", "do not match its layer names"),
        ("window", "disagree on sliding window"),
        ("state", "disagree on state evidence"),
    ],
)
def test_uniform_group_layers_must_agree_on_names_retention_and_state(change, match):
    names = ["kda.0", "kda.1"]
    specs = [_mamba(), _mamba(shapes=((6144, 4), (16, 128, 128))) if change == "state" else _mamba()]
    if change == "window":
        specs = [
            _spec("SlidingWindowSpec", block_size=16, page_size_bytes=64, sliding_window=window, dtype="torch.bfloat16")
            for window in (32, 64)
        ]
    uniform = _spec(
        "UniformTypeKVCacheSpecs",
        block_size=specs[0].block_size,
        page_size_bytes=sum(spec.page_size_bytes for spec in specs),
        kv_cache_specs=dict(zip(names, specs, strict=True)),
    )
    with pytest.raises(ValueError, match=match):
        observer.cache_groups(
            SimpleNamespace(kv_cache_groups=[_group(names[::-1] if change == "names" else names, uniform)])
        )


def test_worker_memory_reads_layers_and_tuple_caches():
    worker, _scheduler, _cache = _initialized(layers=True)
    assert isinstance(worker.vllm_config.compilation_config.static_forward_context["conv.0"].kv_cache, tuple)
    observed = observer.worker_memory(worker)
    assert observed["allocated_cache_bytes"] == 100 * 128
    assert [(item["layers"], item["shared_by"], item["layer_stride"]) for item in observed["tensor_allocations"]] == [
        (["full.0", "slide.0", "conv.0"], ["full.0", "slide.0", "conv.0"], 0),
        (["full.1", "slide.1"], ["full.1", "slide.1"], 0),
    ]
    assert [group["kind"] for group in observed["groups"]] == ["attention", "attention", "convolution"]
    # Tensors of vLLM 0.27.0 and 0.28.0 have no layer stride, so their records name none.
    legacy = observer.worker_memory(_initialized()[0])
    assert all(
        item["layers"] == item["shared_by"] and "layer_stride" not in item for item in legacy["tensor_allocations"]
    )


def test_block_outer_tensor_region_ends_at_its_last_page():
    worker, _scheduler, cache = _initialized(layers=True)
    # Block-outer (kv_cache_interface.py:1422-1428): each 128-byte block holds the
    # 64-byte pages of both layers, so the last page ends exactly at the buffer end.
    tensor = cache.kv_cache_tensors[1]
    tensor.layer_stride, tensor.block_stride, tensor.offset = 64, 128, 0
    allocation = observer.worker_memory(worker)["tensor_allocations"][1]
    assert {key: allocation[key] for key in ("layers", "layer_stride", "block_stride", "offset")} == {
        "layers": ["full.1", "slide.1"],
        "layer_stride": 64,
        "block_stride": 128,
        "offset": 0,
    }


@pytest.mark.parametrize(
    "change,match",
    [
        ("region", "exceed their backing allocation"),
        ("last_page", "exceed their backing allocation"),
        ("page", "page exceeds its block stride"),
        ("storage", "layer strides use multiple backing allocations"),
    ],
)
def test_layer_strided_tensors_share_one_buffer_that_holds_every_region(change, match):
    worker, _scheduler, cache = _initialized(layers=True)
    tensor = cache.kv_cache_tensors[1]
    if change == "region":
        # The second layer starts one byte later, so its last block passes the end of the buffer.
        tensor.layer_stride = 1
    elif change == "last_page":
        # Block-outer, one page later: the second layer's last page ends one page past the buffer.
        tensor.layer_stride, tensor.block_stride, tensor.offset = 64, 128, 64
    elif change == "page":
        # A 64-byte page does not fit in a 32-byte block stride.
        tensor.block_stride = 32
    else:
        context = worker.vllm_config.compilation_config.static_forward_context
        moved = _Storage(300, tensor.size)
        for name in tensor.layers:
            context[name].kv_cache = _Tensor(moved)
        worker.model_runner.kv_caches = [layer.kv_cache for layer in context.values()]
    with pytest.raises(ValueError, match=match):
        observer.worker_memory(worker)


def test_scheduler_memory_records_block_sizes_and_mamba_mode():
    _worker, scheduler, cache = _initialized()
    legacy = observer.scheduler_memory(scheduler, cache)
    assert (legacy["block_size"], legacy["hash_block_size"], legacy["mamba_cache_mode"]) == (None, None, None)
    scheduler.block_size = scheduler.hash_block_size = 2176
    scheduler.kv_cache_manager.block_pool.hash_block_size = 2176
    scheduler.vllm_config.cache_config.mamba_cache_mode = "align"
    record = observer.scheduler_memory(scheduler, cache)
    assert (record["block_size"], record["hash_block_size"], record["mamba_cache_mode"]) == (2176, 2176, "align")
    # The worker record carries the group kind; the scheduler copy keeps the layer specs.
    assert all("kind" not in group and "layer_specs" in group for group in record["groups"])
    scheduler.kv_cache_manager.block_pool.hash_block_size = 4
    with pytest.raises(ValueError, match="hash block size"):
        observer.scheduler_memory(scheduler, cache)


def test_resolve_runtime_resources_requires_all_ranks_and_preserves_graph_evidence(tmp_path, monkeypatch):
    cell = _evidence(tmp_path, monkeypatch)
    result = _resolve(cell, tmp_path)
    resources = FpmResourceProfile.model_validate(result)
    assert resources.memory_source == "runtime"
    assert resources.runtime_memory.kv_cache_bytes == 88 * 128
    assert [group.page_size_bytes for group in resources.cache_groups] == [128, 128, 128]
    assert [group.num_layers for group in resources.cache_groups] == [2, 2, 1]
    assert resources.activations_bytes is None
    provenance = json.loads(resources.runtime_memory.provenance)
    assert provenance["runtime_settings"]["compilation_config"]["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16, 32, 64]
    assert "enable_prefix_caching" not in provenance["runtime_settings"]["cache_config"]
    assert len(provenance["artifacts"]) == 6
    assert provenance["artifacts"][-1]["evidence"]["cache"]["diagnostics"]["peak_activation_memory"] == 123


@pytest.mark.parametrize("backend", ["uva", "prefetch"])
def test_active_weight_offload_is_observed_and_cannot_resolve(tmp_path, monkeypatch, backend):
    original = _initialized

    def offloaded(**kwargs):
        worker, scheduler, cache = original(**kwargs)
        offload = worker.vllm_config.offload_config
        offload.offload_backend = backend
        offload.uva.cpu_offload_gb = 8 if backend == "uva" else 0
        offload.prefetch.offload_group_size = 4 if backend == "prefetch" else 0
        return worker, scheduler, cache

    monkeypatch.setattr(sys.modules[__name__], "_initialized", offloaded)
    monkeypatch.setattr(observer, "_offloader_type", lambda: f"vllm.model_executor.offloader.{backend}.Offloader")
    cell = _evidence(tmp_path, monkeypatch)
    for path in tmp_path.glob("*/fpm-memory-*.json"):
        evidence = json.loads(path.read_text())
        assert evidence["status"] == "unresolved"
        assert evidence["resolved_config"]["offload_config"]["offload_backend"] == backend
        if evidence["kind"] == "worker":
            assert backend in evidence["effective_offloader"]
    with pytest.raises(ValueError, match="unresolved"):
        _resolve(cell, tmp_path)


def _edit_memory(root, edit):
    for path in root.glob("*/fpm-memory-*.json"):
        evidence = json.loads(path.read_text())
        edit(evidence)
        path.write_text(json.dumps(evidence))


@pytest.mark.parametrize("change", ["fp8", "unknown_quantization", "async", "offload", "effective_offloader"])
def test_resolver_rejects_consistent_but_unsupported_runtime_policy(tmp_path, monkeypatch, change):
    cell = _evidence(tmp_path, monkeypatch)

    def edit(evidence):
        config = evidence["resolved_config"]
        if change == "fp8":
            config["model_config"]["quantization"] = "fp8"
            config["quantization_config"] = {
                "type": "vllm.model_executor.layers.quantization.fp8.Fp8Config",
                "activation_scheme": "dynamic",
                "weight_block_size": None,
            }
        elif change == "unknown_quantization":
            config["model_config"]["quantization"] = "compressed-tensors"
        elif change == "async":
            config["scheduler_config"]["async_scheduling"] = True
        elif change == "offload":
            config["offload_config"]["uva"]["cpu_offload_gb"] = 8
        else:
            evidence["effective_offloader"] = "vllm.model_executor.offloader.uva.UVAOffloader"

    _edit_memory(tmp_path, edit)
    with pytest.raises(ValueError, match="precision|async_scheduling|offload"):
        _resolve(cell, tmp_path)


@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16", "fp8"])
def test_fp8_block_weights_preserve_and_verify_selected_bf16_cache(tmp_path, monkeypatch, cache_dtype):
    cell = replace(
        _evidence(tmp_path, monkeypatch),
        gemm_quant_mode="fp8_block",
        weight_quantization="fp8_block",
        moe_quant_mode="fp8_block",
    )

    def edit(evidence):
        config = evidence["resolved_config"]
        config["model_config"]["quantization"] = "fp8"
        config["cache_config"]["cache_dtype"] = cache_dtype
        config["quantization_config"] = {
            "type": "vllm.model_executor.layers.quantization.fp8.Fp8Config",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }

    _edit_memory(tmp_path, edit)
    if cache_dtype == "fp8":
        with pytest.raises(ValueError, match="KV precision differs from the planned identity"):
            _resolve(cell, tmp_path)
    else:
        resources = FpmResourceProfile.model_validate(_resolve(cell, tmp_path))
        settings = json.loads(resources.runtime_memory.provenance)["runtime_settings"]
        assert settings["cache_config"]["cache_dtype"] == cache_dtype
        assert settings["model_config"]["dtype"] == "torch.bfloat16"
        assert settings["quantization_config"]["weight_block_size"] == [128, 128]


def test_modelopt_nvfp4_allows_bfloat16_attention_and_shared_experts(tmp_path, monkeypatch):
    cell = _evidence(tmp_path, monkeypatch)
    cell = replace(cell, gemm_quant_mode="bfloat16", weight_quantization="bfloat16", moe_quant_mode="nvfp4")
    resources = FpmResourceProfile.model_validate(_resolve(cell, tmp_path))
    assert resources.runtime_memory.kv_cache_bytes == 88 * 128
    settings = json.loads(resources.runtime_memory.provenance)["runtime_settings"]
    assert settings["model_config"]["dtype"] == "torch.bfloat16"
    assert settings["quantization_config"]["quant_method"] == "NVFP4"


@pytest.mark.parametrize(
    "change", ["missing", "missing_mode", "wrong_type", "wrong_max", "unsorted", "eager_with_graphs"]
)
def test_missing_or_inconsistent_graph_evidence_cannot_finalize(tmp_path, monkeypatch, change):
    cell = _evidence(tmp_path, monkeypatch)

    def edit(evidence):
        config = evidence["resolved_config"]
        graph = config["compilation_config"]
        if change == "missing":
            graph.clear()
        elif change == "missing_mode":
            graph.pop("mode")
        elif change == "wrong_type":
            graph["cudagraph_capture_sizes"] = [True, 64]
        elif change == "wrong_max":
            graph["max_cudagraph_capture_size"] = 32
        elif change == "unsorted":
            graph["cudagraph_capture_sizes"] = [64, 32]
        else:
            config["model_config"]["enforce_eager"] = True

    _edit_memory(tmp_path, edit)
    with pytest.raises(ValueError, match="graph configuration"):
        _resolve(cell, tmp_path)


@pytest.mark.parametrize("change", ["graph", "weights", "dtype"])
def test_scheduler_must_agree_with_worker_memory_settings(tmp_path, monkeypatch, change):
    cell = _evidence(tmp_path, monkeypatch)

    def edit(evidence):
        if evidence["kind"] != "scheduler":
            return
        config = evidence["resolved_config"]
        if change == "graph":
            config["compilation_config"].update(
                cudagraph_mode="NONE", cudagraph_capture_sizes=[], max_cudagraph_capture_size=0
            )
        elif change == "weights":
            config["model_config"]["quantization"] = "fp8"
        else:
            config["model_config"]["dtype"] = "torch.float16"

    _edit_memory(tmp_path, edit)
    with pytest.raises(ValueError, match="precision|between scheduler and worker"):
        _resolve(cell, tmp_path)


@pytest.mark.parametrize("eager", [False, True])
def test_audited_none_graph_configuration_is_valid(tmp_path, monkeypatch, eager):
    cell = _evidence(tmp_path, monkeypatch)

    def edit(evidence):
        config = evidence["resolved_config"]
        config["model_config"]["enforce_eager"] = eager
        config["compilation_config"].update(
            mode=0 if eager else 3, cudagraph_mode="NONE", cudagraph_capture_sizes=[], max_cudagraph_capture_size=0
        )

    _edit_memory(tmp_path, edit)
    result = _resolve(cell, tmp_path)
    assert result["runtime_memory"]["kv_cache_bytes"] == 88 * 128


def test_audited_worker_graph_downgrade_preserves_scheduler_snapshot(tmp_path, monkeypatch):
    cell = _evidence(tmp_path, monkeypatch)

    def edit(evidence):
        if evidence["kind"] == "worker":
            graph = evidence["resolved_config"]["compilation_config"]
            evidence["initial_compilation_config"] = dict(graph)
            graph["cudagraph_mode"] = "PIECEWISE"

    _edit_memory(tmp_path, edit)
    result = _resolve(cell, tmp_path)
    provenance = json.loads(result["runtime_memory"]["provenance"])
    assert provenance["runtime_settings"]["compilation_config"]["cudagraph_mode"] == "PIECEWISE"


@pytest.mark.parametrize("declared_mode", ["FULL_AND_PIECEWISE", "FULL_DECODE_ONLY"])
def test_formal_memory_checks_reviewed_initial_graph_before_validated_worker_downgrade(
    tmp_path, monkeypatch, declared_mode
):
    cell = _evidence(tmp_path, monkeypatch)

    def downgrade(record):
        if record["kind"] == "worker":
            graph = record["resolved_config"]["compilation_config"]
            record["initial_compilation_config"] = dict(graph)
            graph["cudagraph_mode"] = "PIECEWISE"

    _edit_memory(tmp_path, downgrade)
    expected = {"cudagraph_mode": declared_mode, "max_cudagraph_capture_size": 64}
    if declared_mode == "FULL_AND_PIECEWISE":
        result = _resolve(cell, tmp_path, expected_compilation_config=expected)
        assert result["runtime_memory"]["kv_cache_bytes"] == 88 * 128
    else:
        with pytest.raises(ValueError, match="reviewed serving role"):
            _resolve(cell, tmp_path, expected_compilation_config=expected)


@pytest.mark.parametrize("resolved_mode", ["PIECEWISE", "NONE"])
def test_full_graph_cannot_fall_back_to_unsupported_attention_mode(tmp_path, monkeypatch, resolved_mode):
    cell = _evidence(tmp_path, monkeypatch)

    def edit(evidence):
        graph = evidence["resolved_config"]["compilation_config"]
        graph["cudagraph_mode"] = "FULL"
        if evidence["kind"] == "worker":
            evidence["initial_compilation_config"] = dict(graph)
            graph["cudagraph_mode"] = resolved_mode

    _edit_memory(tmp_path, edit)
    with pytest.raises(ValueError, match="configurations differ between scheduler and worker"):
        _resolve(cell, tmp_path)


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing_worker", "worker rank evidence is incomplete"),
        ("missing_scheduler", "scheduler rank evidence is incomplete"),
        ("duplicate_worker", "duplicate runtime memory worker rank"),
        ("stale", "different plan, attempt"),
        ("wrong_context", "launch mismatch"),
        ("wrong_dtype", "KV precision"),
        ("wrong_gpu_fraction", "launch mismatch"),
        ("graph_mismatch", "differ across worker ranks"),
        ("unknown_spec", "unsupported runtime cache spec"),
        ("nonintegral", "non-integral pool page"),
        ("watermark", "pool capacity/reservations"),
        ("second_pool", "pool capacity/reservations"),
        ("duplicate_storage", "storage accounting"),
        ("wrong_retention", "geometry"),
        ("timing_failure", "terminal result"),
    ],
)
def test_runtime_memory_rejects_incomplete_stale_mixed_or_unsupported_evidence(tmp_path, monkeypatch, change, match):
    cell = _evidence(tmp_path, monkeypatch)
    path = tmp_path / "pod-0/fpm-memory-worker-dp0-tp0-pp0.json"
    payload = json.loads(path.read_text())
    scheduler_path = tmp_path / "pod-0/fpm-memory-scheduler-dp0.json"
    scheduler = json.loads(scheduler_path.read_text())
    if change == "missing_worker":
        path.unlink()
    elif change == "missing_scheduler":
        scheduler_path.unlink()
    elif change == "duplicate_worker":
        (tmp_path / "pod-0/fpm-memory-duplicate.json").write_text(path.read_text())
    elif change == "stale":
        payload["collector_provenance"]["attempt_id"] = "earlier-attempt"
    elif change == "wrong_context":
        payload["resolved_config"]["model_config"]["max_model_len"] = 2048
    elif change == "wrong_dtype":
        payload["resolved_config"]["cache_config"]["cache_dtype"] = "fp8"
    elif change == "wrong_gpu_fraction":
        payload["resolved_config"]["cache_config"]["gpu_memory_utilization"] = 0.8
    elif change == "graph_mismatch":
        payload["resolved_config"]["compilation_config"]["cudagraph_capture_sizes"] = [1, 2, 4]
        payload["resolved_config"]["compilation_config"]["max_cudagraph_capture_size"] = 4
    elif change == "unknown_spec":
        payload["cache"]["groups"][0]["spec_type"] = "vllm.v1.kv_cache_interface.HiSparseHotSpec"
        scheduler["cache"]["groups"][0]["spec_type"] = payload["cache"]["groups"][0]["spec_type"]
    elif change == "nonintegral":
        payload["cache"]["allocated_cache_bytes"] -= 1
        payload["cache"]["storages"][0]["size_bytes"] -= 1
    elif change == "watermark":
        scheduler["cache"]["watermark_blocks"] = 1
    elif change == "second_pool":
        scheduler["cache"]["pool_count"] = 2
    elif change == "duplicate_storage":
        payload["cache"]["storages"].append(payload["cache"]["storages"][0])
    elif change == "wrong_retention":
        payload["cache"]["groups"][1]["sliding_window"] = 64
    elif change == "timing_failure":
        native = tmp_path / "pod-0/benchmark.json"
        value = json.loads(native.read_text())
        value["status"] = "failed"
        native.write_text(json.dumps(value))
    if change != "missing_worker":
        path.write_text(json.dumps(payload))
    if change != "missing_scheduler":
        scheduler_path.write_text(json.dumps(scheduler))
    with pytest.raises(ValueError, match=match):
        _resolve(cell, tmp_path)


@pytest.mark.parametrize("change", ["unknown_spec", "alias", "missing_storage", "watermark", "pool", "version"])
def test_unsupported_observation_records_unresolved_without_running_or_replacing_runtime(tmp_path, monkeypatch, change):
    worker, scheduler, config = _initialized()
    monkeypatch.setattr(observer, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        observer.importlib.metadata, "version", lambda _name: "0.29.0" if change == "version" else "0.27.0"
    )
    _write_provenance(tmp_path / "collector-provenance.json", cell_id="test")
    kind, owner = "worker", worker
    if change == "unknown_spec":
        config.kv_cache_groups[0].kv_cache_spec = SimpleNamespace()
    elif change == "alias":
        config.kv_cache_tensors[0].block_stride = 128
        config.kv_cache_tensors[1].block_stride = 128
    elif change == "missing_storage":
        worker.model_runner.kv_caches = []
    elif change == "watermark":
        kind, owner = "scheduler", scheduler
        scheduler.kv_cache_manager.watermark_blocks = 2
    elif change == "pool":
        kind, owner = "scheduler", scheduler
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].block_pool = object()
    observer.observe(
        kind,
        owner,
        dp_rank=0,
        tp_rank=0 if kind == "worker" else None,
        pp_rank=0 if kind == "worker" else None,
        cache_config=config,
    )
    payload = json.loads(next(tmp_path.glob("fpm-memory-*.json")).read_text())
    assert payload["status"] == "unresolved"
    assert payload["error"]
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("resolved_mode", ["PIECEWISE", "NONE"])
@pytest.mark.parametrize("initialization_count", [1, 2])
def test_worker_profile_time_graph_downgrade_resolves(tmp_path, monkeypatch, resolved_mode, initialization_count):
    cell = _evidence(tmp_path, monkeypatch)

    class Worker:
        def determine_available_memory(self):
            self.vllm_config.compilation_config.cudagraph_mode = resolved_mode
            return self._fpm_available_cache_bytes

        def initialize_from_config(self, config):
            assert config is self.model_runner.kv_cache_config

        def compile_or_warm_up_model(self):
            return "compiled"

    tp_group = SimpleNamespace(rank_in_group=0)
    modules = {
        "vllm.distributed": {
            "get_pp_group": lambda: SimpleNamespace(rank_in_group=0),
            "get_tp_group": lambda: tp_group,
        },
        "vllm.v1.worker.gpu_worker": {"Worker": Worker},
    }
    monkeypatch.setitem(sys.modules, "fpm_memory_observer", observer)
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "fpm_memory_worker", Path(observer.__file__).with_name("fpm_memory_worker.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for dp in range(2):
        pod = tmp_path / f"pod-{dp}"
        monkeypatch.setattr(observer, "RESULTS_DIR", pod)
        for tp in range(2):
            (pod / f"fpm-memory-worker-dp{dp}-tp{tp}-pp0.json").unlink()
            initialized, _, cache = _initialized(blocks=100 - dp * 10, free=98 - dp * 10)
            worker = module.FpmResourceWorker()
            worker.__dict__.update(vars(initialized))
            worker._fpm_cache_initialized = False
            worker.vllm_config.parallel_config.data_parallel_rank = dp
            tp_group.rank_in_group = tp
            for _ in range(initialization_count):
                assert worker.determine_available_memory() == (100 - dp * 10) * 128 + 200
                worker.initialize_from_config(cache)
            assert worker.compile_or_warm_up_model() == "compiled"

    result = _resolve(cell, tmp_path)
    assert result["runtime_memory"]["kv_cache_bytes"] == 88 * 128
    provenance = json.loads(result["runtime_memory"]["provenance"])
    assert provenance["runtime_settings"]["compilation_config"]["cudagraph_mode"] == resolved_mode
    for artifact in provenance["artifacts"]:
        evidence = artifact["evidence"]
        if evidence["kind"] == "worker":
            assert evidence["initial_compilation_config"]["cudagraph_mode"] == "FULL_AND_PIECEWISE"


def test_worker_and_scheduler_wrappers_delegate_before_observation(monkeypatch, caplog):
    events = []

    class Worker:
        def determine_available_memory(self):
            events.append("profile")
            return 1234

        def initialize_from_config(self, config):
            events.append(("initialize", config))

        def compile_or_warm_up_model(self):
            events.append("warmup")
            return "compiled"

    class Scheduler:
        def __init__(self, vllm_config, kv_cache_config, *args, **kwargs):
            events.append(("scheduler", vllm_config, kv_cache_config, args, kwargs))
            self._fpm_dp_rank = 3

    modules = {
        "fpm_memory_observer": {
            "observe": lambda *args, **kwargs: events.append(("observe", args, kwargs)),
            "observe_execution": lambda *args, **kwargs: events.append(("observe_execution", args, kwargs)),
            "observe_cpu": lambda *args, **kwargs: events.append(("observe_cpu", args, kwargs)),
            "compilation_config": lambda _config: {"cudagraph_mode": "FULL"},
        },
        "vllm.distributed": {
            "get_pp_group": lambda: SimpleNamespace(rank_in_group=0),
            "get_tp_group": lambda: SimpleNamespace(rank_in_group=1),
        },
        "vllm.v1.worker.gpu_worker": {"Worker": Worker},
        "dynamo.vllm.instrumented_scheduler": {"InstrumentedScheduler": Scheduler},
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)

    def load(filename):
        path = Path(observer.__file__).with_name(filename + ".py")
        spec = importlib.util.spec_from_file_location(filename, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    worker_module = load("fpm_memory_worker")
    worker = worker_module.FpmResourceWorker()
    worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_index=3, data_parallel_rank=0))
    assert worker.determine_available_memory() == 1234
    worker.initialize_from_config("cache")
    assert worker.compile_or_warm_up_model() == "compiled"
    assert events[:3] == ["profile", ("initialize", "cache"), "warmup"]
    assert events[3][2] == {"dp_rank": 3, "tp_rank": 1, "pp_rank": 0}
    assert events[3][0] == "observe_cpu"
    assert events[4][0] == "observe_execution"
    assert events[5][0] == "observe"
    assert worker._fpm_available_cache_bytes == 1234
    assert worker._fpm_cache_initialized is True
    load("fpm_memory_scheduler").FpmResourceInstrumentedScheduler("config", "cache", "manager", block_size=16)
    assert events[6] == ("scheduler", "config", "cache", ("manager",), {"block_size": 16})
    assert events[7] == ("observe_cpu", ("scheduler",), {"dp_rank": 3})
    assert events[8][2] == {"dp_rank": 3, "cache_config": "cache"}
    before = len(events)
    load("fpm_memory_scheduler").FpmExecutionInstrumentedScheduler("config", "cache")
    assert events[before:] == [
        ("scheduler", "config", "cache", (), {}),
        ("observe_cpu", ("scheduler",), {"dp_rank": 3}),
    ]

    def fail(_self, *_args):
        raise RuntimeError("real runtime failed")

    for method, args in (
        ("determine_available_memory", ()),
        ("initialize_from_config", ("cache",)),
        ("compile_or_warm_up_model", ()),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(Worker, method, fail)
            before = len(events)
            with pytest.raises(RuntimeError, match="real runtime failed"):
                getattr(worker, method)(*args)
            assert len(events) == before

    def missing_graph(_config):
        raise AttributeError("compilation configuration is unavailable")

    monkeypatch.setattr(worker_module, "compilation_config", missing_graph)
    worker = worker_module.FpmResourceWorker()
    worker.vllm_config = SimpleNamespace()
    assert worker.determine_available_memory() == 1234
    assert worker._fpm_initial_compilation_config is None
    assert "FPM initial graph configuration is unavailable" in caplog.text
    worker.initialize_from_config("cache")
    assert worker._fpm_initial_compilation_config is None


def test_ordinary_serving_worker_observes_initialized_runtime_without_dynamo(tmp_path, monkeypatch):
    provenance = {
        "schema": "aisimulate-serving-validation/v1",
        "run_id": "serving-123",
        "purpose": "matched_serving_accuracy",
    }
    provenance_path = tmp_path / "serving-provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    output = tmp_path / "observations"
    monkeypatch.setenv("FPM_EXECUTION_OUTPUT_DIR", str(output))
    monkeypatch.setenv("FPM_EXECUTION_PROVENANCE_FILE", str(provenance_path))
    monkeypatch.setattr(observer.importlib.metadata, "version", lambda _name: "0.28.0")

    class Worker:
        def compile_or_warm_up_model(self):
            self.vllm_config.compilation_config.cudagraph_mode = "PIECEWISE"
            self.model_runner.attn_groups = [
                [
                    SimpleNamespace(
                        backend=type("ResolvedBackend", (), {"__module__": "actual.runtime"}),
                        layer_names=["layer.0"],
                        kv_cache_group_id=0,
                    )
                ]
            ]
            return "compiled"

    monkeypatch.setitem(sys.modules, "fpm_memory_observer", observer)
    monkeypatch.setitem(sys.modules, "dynamo.vllm.instrumented_scheduler", None)
    for name, values in {
        "vllm.distributed": {
            "get_pp_group": lambda: SimpleNamespace(rank_in_group=0),
            "get_tp_group": lambda: SimpleNamespace(rank_in_group=0),
        },
        "vllm.v1.worker.gpu_worker": {"Worker": Worker},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "fpm_memory_worker", Path(observer.__file__).with_name("fpm_memory_worker.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.FpmExecutionWorker()
    worker.vllm_config = _vllm_config(tp=1, dp=1)
    worker.vllm_config.parallel_config.data_parallel_rank = 0
    worker.model_runner = SimpleNamespace(attn_groups=[])
    assert worker.compile_or_warm_up_model() == "compiled"
    sidecar = output / "fpm-execution-worker-dp0-tp0-pp0.json"
    result = json.loads(sidecar.read_text())
    assert result["status"] == "observed"
    assert result["execution_provenance"] == provenance
    assert result["provenance_source"]["path"] == str(provenance_path)
    assert "collector_provenance" not in result
    assert result["attention_groups"][0]["backend_class"] == "actual.runtime.ResolvedBackend"
    assert result["graph_config"]["cudagraph_mode"] == "PIECEWISE"
    assert not hasattr(worker, "_fpm_available_cache_bytes")
    assert not list(output.glob("fpm-memory-*.json"))
    original = sidecar.read_bytes()
    with pytest.raises(RuntimeError, match="duplicate"):
        worker.compile_or_warm_up_model()
    assert sidecar.read_bytes() == original
    # A second rank can share this run, but a different run cannot reuse it.
    observer.observe_execution(worker, dp_rank=0, tp_rank=1, pp_rank=0)
    provenance["run_id"] = "serving-456"
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="different serving run"):
        observer.observe_execution(worker, dp_rank=0, tp_rank=2, pp_rank=0)


@pytest.mark.parametrize(
    "output,provenance",
    [
        ("absolute", None),
        (None, "absolute"),
        ("relative", "absolute"),
        ("absolute", "relative"),
        ("/bad\npath", "absolute"),
    ],
)
def test_serving_execution_observation_requires_paired_absolute_paths(tmp_path, monkeypatch, output, provenance):
    for name, value in (("FPM_EXECUTION_OUTPUT_DIR", output), ("FPM_EXECUTION_PROVENANCE_FILE", provenance)):
        monkeypatch.delenv(name, raising=False)
        if value is not None:
            monkeypatch.setenv(name, str(tmp_path) if value == "absolute" else value)
    with pytest.raises(ValueError, match="supplied together|absolute path"):
        observer.observe_execution(SimpleNamespace(), dp_rank=0, tp_rank=0, pp_rank=0)


@pytest.mark.parametrize("failure", ["purpose", "run_id", "collector", "orphan"])
def test_serving_execution_observation_rejects_missing_or_reused_identity(tmp_path, monkeypatch, failure):
    output = tmp_path / "observations"
    output.mkdir()
    provenance = {"run_id": "unique-run", "purpose": "matched_serving_accuracy"}
    if failure in {"purpose", "run_id"}:
        provenance.pop(failure)
    elif failure == "collector":
        (output / "collector-provenance.json").write_text("{}")
    else:
        (output / "fpm-execution-worker-dp0-tp0-pp0.json").write_text("{}")
    provenance_path = tmp_path / "serving-provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    monkeypatch.setenv("FPM_EXECUTION_OUTPUT_DIR", str(output))
    monkeypatch.setenv("FPM_EXECUTION_PROVENANCE_FILE", str(provenance_path))
    with pytest.raises(ValueError, match="requires a serving run_id|fresh directory"):
        observer.observe_execution(SimpleNamespace(), dp_rank=0, tp_rank=0, pp_rank=0)


def _pending_plan(version="0.27.0"):
    profile = _profile()
    for deployment in profile["deployments"]:
        deployment["backend_version"] = version
        for key in ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes"):
            deployment["resources"].pop(key)
    return _plan(profile, collector_config={"runtime_backend_version": version})


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_pending_memory_plans_without_model_and_renders_audited_hooks(tmp_path):
    plan = _pending_plan()
    assert all(item.disposition == "unknown" for item in plan.topology_memory_admission)
    assert all(
        item.estimated_non_kv_bytes is None
        for decision in plan.topology_memory_admission
        for item in decision.estimates
    )
    assert plan.to_dict()["runtime_memory_policy"]["async_scheduling"] is False
    runtime_memory.validate_saved_plan(plan.to_dict())
    for cell in plan.cells:
        output = tmp_path / cell.cell_id
        output.mkdir()
        runner._render_cell(plan, cell, output, {})
        script = (output / "run.sh").read_text()
        assert "--worker-cls fpm_memory_worker.FpmResourceWorker" in script
        assert "--scheduler-cls fpm_memory_scheduler.FpmResourceInstrumentedScheduler" in script
        assert script.count("--no-async-scheduling") == 1


@pytest.mark.parametrize(
    "flag",
    ["--worker-cls", "--scheduler-cls", "--kv-cache-memory-bytes", "--num-gpu-blocks-override", "--async-scheduling"],
)
@pytest.mark.usefixtures("no_models_or_timing_data")
def test_pending_memory_rejects_conflicting_runtime_policy(flag):
    plan = _pending_plan()
    cell = plan.cells[0]
    cell = replace(
        cell, backend_policy=planner.BackendPolicy("conflict", {"params": {"agg": {"extra_cli_args": [flag]}}}, {})
    )
    with pytest.raises(ValueError, match="memory"):
        runner._cell_generator_overrides(plan, cell, {})


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_saved_plan_identity_reads_immutable_facts_without_model_resolution():
    plan = _plan(_profile())
    saved = plan.to_dict()
    identity = runtime_memory.saved_plan_identity(saved)
    assert identity.model_path == plan.model_path
    assert identity.capability.aic_database_version == plan.capability.aic_database_version
    assert identity.options.warmup_iterations == plan.options.warmup_iterations
    assert runtime_memory.cell_from_dict(saved["cells"][0]) == plan.cells[0]
    saved["options"]["global_warmup_iterations"] += 1
    with pytest.raises(ValueError, match="SHA-256"):
        runtime_memory.saved_plan_identity(saved)


def test_saved_cell_rejects_lossy_extra_fields():
    payload = _cell().to_dict()
    payload["unrecognized"] = True
    with pytest.raises(ValueError, match="contract"):
        runtime_memory.cell_from_dict(payload)


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_unaudited_runtime_keeps_native_timing_launch_and_pending_memory():
    plan = _pending_plan("0.25.1")
    assert plan.to_dict()["runtime_memory_policy"]["observation"] == "unavailable_for_runtime"
    for cell in plan.cells:
        args = runner._cell_generator_overrides(plan, cell, {})["params"]["agg"]["extra_cli_args"]
        assert "--worker-cls" not in args
        assert "--scheduler-cls" not in args
        assert ("--no-async-scheduling" in args) == (cell.workload_kind == "prefill")
        assert runner._observe_runtime_memory(plan, cell) is False


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_historical_v10_plan_keeps_hash_and_reaches_native_aggregation(tmp_path):
    from collector.fpm_forward import database

    from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, LEGACY_EXECUTION_IDENTITY

    # Captured from the actual schema-10 producer at 4d702ff6b756b21e74b76f30273e9828fdde6861.
    # Absolute source paths are immutable provenance; this reader must not open them.
    fixture = Path(__file__).parent / "fixtures/fpm_collection_plan_v10.json"
    original = fixture.read_bytes()
    payload = json.loads(original)
    frozen = runtime_memory.saved_plan_identity(payload)
    assert frozen.sha256 == "ea99845deee06804fd35a7a0999b0118446df1ac8ededbf7c813345360ba6bf7"
    assert [cell.cell_id for cell in frozen.cells] == ["fpm-d111aeb8823a9bee", "fpm-66274b2ba8ae195c"]
    assert frozen.options.benchmark_points_json is None
    rows = []
    for cell in frozen.cells:
        unit = tmp_path / cell.cell_id / "raw/node0000"
        unit.mkdir(parents=True)
        provenance_path = unit / "collector-provenance.json"
        _write_provenance(provenance_path, cell_id=cell.cell_id, plan_sha256=frozen.sha256, attempt_id="historical")
        provenance = json.loads(provenance_path.read_text())
        provenance["runtime"]["backend_version"] = "0.27.0"
        provenance_path.write_text(json.dumps(provenance))
        (unit / "benchmark.json").write_text(json.dumps(_native_payload(phase=cell.workload_kind, rank=0, dp=1)))
        rows.extend(database.aggregate_cell(frozen, cell, tmp_path / cell.cell_id, expected_attempt_id="historical"))
    assert len(rows) == 2
    for row in rows:
        assert row["source_plan_sha256"] == frozen.sha256
        assert tuple(row[name] for name in EXECUTION_COLUMNS) == LEGACY_EXECUTION_IDENTITY
    assert fixture.read_bytes() == original


@pytest.mark.parametrize("schema_version", [10, 11])
@pytest.mark.usefixtures("no_models_or_timing_data")
def test_saved_plan_rejects_cell_identity_from_other_schema(schema_version):
    payload = _pending_plan().to_dict()
    payload["schema_version"] = schema_version
    if schema_version == 11:
        for cell in payload["cells"]:
            cell.pop("execution_identity")
            cell.pop("input_text_sha256")
    with pytest.raises(ValueError, match="execution identity does not match its schema"):
        runtime_memory.validate_saved_plan(payload)


def test_saved_cell_preserves_complete_nonlegacy_execution_identity():
    payload = _cell().to_dict()
    payload["execution_identity"] = {
        "model_config_sha256": "a" * 64,
        "execution_profile": "decoder_replay",
        "engram_residency": "hbm_tp_sharded",
        "input_modality": "text",
    }
    payload["input_text_sha256"] = "b" * 64
    assert runtime_memory.cell_from_dict(payload).to_dict() == payload
    payload["execution_identity"].pop("engram_residency")
    with pytest.raises(ValueError, match="invalid execution identity"):
        runtime_memory.cell_from_dict(payload)


def test_saved_cell_rejects_partial_execution_identity():
    payload = _cell().to_dict()
    payload.pop("input_text_sha256")
    with pytest.raises(ValueError, match="incomplete execution identity"):
        runtime_memory.cell_from_dict(payload)


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_historical_formal_publication_reports_unsupported_migration_without_changes(tmp_path):
    from collector.fpm_forward import database

    parquet = tmp_path / "fpm_forward_perf.parquet"
    metadata = tmp_path / "fpm_forward_perf.metadata.json"
    parquet.write_bytes(b"historical sealed data stays unchanged")
    metadata.write_text(json.dumps({"schema_name": "aic_fpm_forward_perf", "schema_version": 6}))
    original = parquet.read_bytes(), metadata.read_bytes()
    with pytest.raises(ValueError, match="historical schema-6.*Automatic migration is unsupported"):
        database.validate_formal_database_commit(parquet, metadata, _pending_plan())
    assert (parquet.read_bytes(), metadata.read_bytes()) == original


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_custom_label_round_trips_independently_of_observed_memory_runtime():
    profile = _profile()
    for deployment in profile["deployments"]:
        deployment["backend_version"] = "local-vllm-branch"
        for key in ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes"):
            deployment["resources"].pop(key)
    plan = _plan(profile, collector_config={"runtime_backend_version": "0.27.0"})
    saved = runtime_memory.saved_plan_identity(plan.to_dict())
    assert saved.backend_version == "local-vllm-branch"
    assert saved.runtime_backend_version == "0.27.0"
    assert runner._observe_runtime_memory(plan, plan.cells[0])
    assert plan.to_dict()["runtime_memory_policy"]["selected_vllm_version"] == "0.27.0"
    unknown = _plan(profile)
    assert not runner._observe_runtime_memory(unknown, unknown.cells[0])
