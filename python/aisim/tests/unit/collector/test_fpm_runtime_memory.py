# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from collector.fpm_forward import cli, planner, repeatability, runner, runtime_memory
from collector.fpm_forward.capabilities import GENERIC_RUNTIME_TEMPLATE_ID
from collector.fpm_forward.config import FPMCollectionOptions
from collector.fpm_forward.runtime import fpm_memory_observer as observer
from collector.fpm_forward.types import ParallelTopology

from aisim_core.fpm_profile import FpmResourceProfile

from .test_fpm_forward import _MODEL_CONFIGS, _args, _mixed_attention_config
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

    With ``layers``, the allocation follows vLLM 0.33.0: one buffer backs every
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


_Role = enum.Enum("KVCacheGroupRole", {"DEFAULT": "default", "HISPARSE_SOURCE": "hisparse_source"}, type=str)


def _initialized_layer_strided(*, blocks=100, free=98, layout="layer_outer"):
    """The groups and layers of _initialized in one buffer, as vLLM 0.32.0 places them.

    v1/core/kv_cache_utils.py:1622-1687 sizes the buffer at bytes_per_block * num_blocks,
    here 128 * blocks, and gives each group one tensor record from byte 0.
    """
    worker, scheduler, cache_config = _initialized(blocks=blocks, free=free)
    storage = _Storage(100, blocks * 128)
    layers = worker.vllm_config.compilation_config.static_forward_context
    for layer in layers.values():
        layer.kv_cache = _Tensor(storage)
    worker.model_runner.kv_caches = [layer.kv_cache for layer in layers.values()]
    layer_stride, block_stride = {"layer_outer": (64 * blocks, 64), "block_outer": (64, 128)}[layout]
    cache_config.kv_cache_tensors = [
        SimpleNamespace(
            size=blocks * 128,
            layers=list(group.layer_names),
            layer_stride=layer_stride,
            block_stride=block_stride,
            offset=0,
            host_resident=False,
        )
        for group in cache_config.kv_cache_groups
    ]
    for group in cache_config.kv_cache_groups:
        group.host_resident, group.role = False, _Role.DEFAULT
    return worker, scheduler, cache_config


def _evidence(tmp_path, monkeypatch, *, version="0.27.0", initialized=None):
    # None means the module's _initialized at call time, which a test may monkeypatch.
    initialized = initialized or _initialized
    cell = replace(
        _cell(dp=2),
        topology=ParallelTopology(tp=2, pp=1, dp=2, moe_tp=1, moe_ep=4, cp=1),
        kv_cache_dtype="bfloat16",
        fmha_quant_mode="bfloat16",
    )
    monkeypatch.setattr(observer.importlib.metadata, "version", lambda _name: version)
    for dp in range(2):
        pod = tmp_path / f"pod-{dp}"
        pod.mkdir()
        _write_provenance(pod / "collector-provenance.json", cell_id=cell.cell_id)
        provenance = json.loads((pod / "collector-provenance.json").read_text())
        provenance["runtime"]["backend_version"] = version
        (pod / "collector-provenance.json").write_text(json.dumps(provenance))
        (pod / "benchmark.json").write_text(json.dumps(_native_payload(phase="prefill", rank=dp, dp=2)))
        monkeypatch.setattr(observer, "RESULTS_DIR", pod)
        for tp in range(2):
            worker, scheduler, config = initialized(blocks=100 - dp * 10, free=98 - dp * 10)
            observer.observe("worker", worker, dp_rank=dp, tp_rank=tp, pp_rank=0)
        observer.observe("scheduler", scheduler, dp_rank=dp, cache_config=config)
    return cell


def _resolve(cell, root, *, expected_backend_version="0.27.0", **kwargs):
    return runtime_memory.resolve_runtime_resources(
        cell,
        root,
        expected_plan_sha256="plan-sha",
        expected_attempt_id="attempt",
        expected_backend_version=expected_backend_version,
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
    # GLM-5.3-Flash KDA state at TP 4 as in its live vLLM 0.33.0 record: bf16 convolution and fp32 SSM views, padded
    # to the MLA page, and one FlashKDA prefill checkpoint block aligned to 16 tokens.
    defaults = {
        "block_size": 2176,
        "page_size_bytes": 1_114_112,
        "mamba_cache_mode": "align",
        "num_speculative_blocks": 0,
        "num_prefill_checkpoint_blocks": 1,
        "prefill_checkpoint_alignment": 16,
        "shapes": ((3, 6144), (16, 128, 128)),
        "dtypes": ("torch.bfloat16", "torch.float32"),
        "real_page_size_bytes": 1_085_440,
    }
    return _spec("MambaSpec", **{**defaults, **fields})


# The pool block of the hybrid groups of _hybrid_caches.
PAGE = 1_114_112


def _hybrid_caches(mamba):
    """The worker and scheduler cache groups of a vLLM 0.33.0 record: one MLA group and the Mamba group ``mamba``."""
    latent = _spec("MLAAttentionSpec", block_size=2176, page_size_bytes=PAGE, sliding_window=None, dtype="torch.uint8")
    config = SimpleNamespace(kv_cache_groups=[_group(["mla.0"], latent), _group(["kda.0", "kda.1"], mamba)])
    groups = observer.cache_groups(config, hybrid_evidence=True)
    classes = ("vllm.model_executor.layers.attention.MLAAttention", "vllm.models.glm5next.common.kda.KDA")
    for group, layer_class in zip(groups, classes, strict=True):
        group["layer_classes"] = dict.fromkeys(group["layer_names"], layer_class)
    return {"groups": groups}, {"groups": [runtime_memory._scheduler_copy(group) for group in groups]}


def test_cache_groups_map_specs_by_class_and_record_state_evidence():
    full = _spec("FullAttentionSpec", block_size=16, page_size_bytes=8192, sliding_window=None, dtype="torch.bfloat16")
    tail = _spec("CircularBufferSpec", block_size=4, page_size_bytes=2048, dtype="torch.bfloat16")
    config = SimpleNamespace(
        kv_cache_groups=[_group(["attn.0"], full), _group(["tail.0"], tail), _group(["kda.0", "kda.1"], _mamba())]
    )
    # Without hybrid evidence the vLLM 0.27.0/0.28.0 allow-list stays.
    with pytest.raises(
        ValueError, match=re.escape("unsupported runtime cache spec: vllm.v1.kv_cache_interface.CircularBufferSpec")
    ):
        observer.cache_groups(config)
    attention, tail_group, mamba = observer.cache_groups(config, hybrid_evidence=True)
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
        }
    }
    assert tail_group["state"] == {"blocks_per_request": 1}
    assert tail_group["sliding_window"] is None
    assert mamba["dtype"] is None
    assert mamba["state"] == {
        "mamba_cache_mode": "align",
        "num_speculative_blocks": 0,
        "num_prefill_checkpoint_blocks": 1,
        "prefill_checkpoint_alignment": 16,
        "shapes": [[3, 6144], [16, 128, 128]],
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
                "HiddenStateCacheSpec",
                "SlidingWindowSpec",
                "SlidingWindowMLASpec",
            ),
            "attention",
        ),
        **dict.fromkeys(("CircularBufferSpec", "MambaSpec"), "state"),
    }
    assert {name: observer.spec_kind(_spec(name)) for name in kinds} == kinds
    # vLLM 0.33.0 has no KpoolTailSpec: the GLM tail is a CircularBufferSpec (models/glm5next/common/attention.py:170).
    with pytest.raises(ValueError, match=re.escape("unsupported runtime cache spec: vllm.v1.kv_cache_interface.Kpool")):
        observer.spec_kind(_spec("KpoolTailSpec"))


def test_cache_groups_record_the_prefill_checkpoint_of_a_flashkda_spec():
    # With FlashKDA, a GLM KDA spec holds one checkpoint block aligned to 16 tokens
    # (models/glm5next/common/kda.py:168-176, model_executor/layers/mamba/kda_checkpoint.py:9-10).
    config = SimpleNamespace(
        kv_cache_groups=[_group(["kda.0"], _mamba(num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=16))]
    )
    (group,) = observer.cache_groups(config, hybrid_evidence=True)
    state = group["state"]
    assert (state["num_prefill_checkpoint_blocks"], state["prefill_checkpoint_alignment"]) == (1, 16)
    assert state["blocks_per_request"] == 1


@pytest.mark.parametrize(("count", "alignment"), [(1, 16), (0, None)])
def test_import_carries_the_checkpoint_evidence_of_a_mamba_group(count, alignment):
    cache, scheduler = _hybrid_caches(
        _mamba(num_prefill_checkpoint_blocks=count, prefill_checkpoint_alignment=alignment)
    )
    (group,) = [item for item in runtime_memory._normalized_groups(cache, scheduler, PAGE) if item["kind"] == "state"]
    assert group["blocks_per_request"] == 1
    assert group["prefill_checkpoint_blocks"] == count
    assert group.get("prefill_checkpoint_alignment") == alignment


@pytest.mark.parametrize(
    ("count", "alignment", "error"),
    [
        (2, 16, "mamba prefill checkpoint blocks must be 0 or 1"),
        (1, None, "mamba prefill checkpoint evidence is inconsistent"),
        (1, 0, "mamba prefill checkpoint evidence is inconsistent"),
        (0, 16, "mamba prefill checkpoint evidence is inconsistent"),
        (1, True, "mamba prefill checkpoint evidence is inconsistent"),
        (1, -16, "mamba prefill checkpoint evidence is inconsistent"),
        # A count that is not an integer is no count.
        (True, 16, "runtime state cache group lacks prefill checkpoint evidence"),
        (1.0, 16, "runtime state cache group lacks prefill checkpoint evidence"),
        ("1", 16, "runtime state cache group lacks prefill checkpoint evidence"),
    ],
)
def test_import_refuses_inconsistent_checkpoint_evidence(count, alignment, error):
    cache, scheduler = _hybrid_caches(
        _mamba(num_prefill_checkpoint_blocks=count, prefill_checkpoint_alignment=alignment)
    )
    with pytest.raises(ValueError, match=error):
        runtime_memory._normalized_groups(cache, scheduler, PAGE)


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
        )
        for page, tokens, cache_role in ((1_114_112, 1, role.SPARSE), (71_808, 4, role.INDEXER))
    )
    names = ["layers.3.mla", "layers.3.indexer"]
    uniform = _spec(
        "UniformTypeKVCacheSpecs",
        block_size=2176,
        page_size_bytes=1_185_920,
        kv_cache_specs=dict(zip(names, (latent, indexer), strict=True)),
    )
    config = SimpleNamespace(kv_cache_groups=[_group(names, uniform)])
    (group,) = observer.cache_groups(config, hybrid_evidence=True)
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
        },
        "layers.3.indexer": {
            "spec_type": "vllm.v1.kv_cache_interface.MLAAttentionSpec",
            "page_size_bytes": 71_808,
            "dtype": "torch.uint8",
            "tokens_per_state": 4,
            "cache_role": "INDEXER",
        },
    }
    # A bf16 latent beside the uint8 indexer has no single group dtype; each layer keeps its own.
    latent.dtype = "torch.bfloat16"
    (group,) = observer.cache_groups(config, hybrid_evidence=True)
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
        observer.cache_groups(
            SimpleNamespace(kv_cache_groups=[_group(["mla.0", "kda.0"], mixed)]), hybrid_evidence=True
        )
    for spec in (_spec("HiSparseHotSpec"), _typed("vendor.kv_cache_interface", "FullAttentionSpec")):
        name = f"{type(spec).__module__}.{type(spec).__name__}"
        with pytest.raises(ValueError, match=re.escape(f"unsupported runtime cache spec: {name}")):
            observer.cache_groups(SimpleNamespace(kv_cache_groups=[_group(["layer.0"], spec)]), hybrid_evidence=True)


def test_rswa_retention_is_reported_not_relabeled():
    # RSWAManager frees the blocks between the prompt and the decode window, which no kind models.
    rswa = _spec(
        "RSWASpec", block_size=16, page_size_bytes=64, sliding_window=None, rswa_window=128, dtype="torch.bfloat16"
    )
    with pytest.raises(
        ValueError, match=re.escape("unsupported runtime cache spec: vllm.v1.kv_cache_interface.RSWASpec")
    ):
        observer.cache_groups(SimpleNamespace(kv_cache_groups=[_group(["layer.0"], rswa)]), hybrid_evidence=True)


def test_uniform_state_group_records_one_state_and_its_inner_specs():
    # A uniform group of GLM-5.3-Flash tail rings, each a CircularBufferSpec (glm5next/common/attention.py:170-179).
    names = ["layers.3.tail", "layers.4.tail"]
    tails = [_spec("CircularBufferSpec", block_size=4, page_size_bytes=2048, dtype="torch.bfloat16") for _name in names]
    uniform = _spec(
        "UniformTypeKVCacheSpecs",
        block_size=4,
        page_size_bytes=2 * 2048,
        kv_cache_specs=dict(zip(names, tails, strict=True)),
    )
    (group,) = observer.cache_groups(SimpleNamespace(kv_cache_groups=[_group(names, uniform)]), hybrid_evidence=True)
    assert (group["kind"], group["state"], group["sliding_window"]) == ("state", {"blocks_per_request": 1}, None)
    assert (group["block_size_tokens"], group["spec_page_size_bytes"], group["dtype"]) == (4, 4096, "torch.bfloat16")
    assert [spec["spec_type"] for spec in group["layer_specs"].values()] == [
        "vllm.v1.kv_cache_interface.CircularBufferSpec"
    ] * 2
    assert [spec["page_size_bytes"] for spec in group["layer_specs"].values()] == [2048, 2048]


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
    specs = [_mamba(), _mamba(shapes=((4, 6144), (16, 128, 128))) if change == "state" else _mamba()]
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
            SimpleNamespace(kv_cache_groups=[_group(names[::-1] if change == "names" else names, uniform)]),
            hybrid_evidence=True,
        )


def test_worker_memory_reads_layers_and_tuple_caches():
    worker, _scheduler, _cache = _initialized(layers=True)
    assert isinstance(worker.vllm_config.compilation_config.static_forward_context["conv.0"].kv_cache, tuple)
    observed = observer.worker_memory(worker, hybrid_evidence=True)
    assert observed["allocated_cache_bytes"] == 100 * 128
    assert [(item["layers"], item["layer_stride"]) for item in observed["tensor_allocations"]] == [
        (["full.0", "slide.0", "conv.0"], 0),
        (["full.1", "slide.1"], 0),
    ]
    assert all("shared_by" not in item for item in observed["tensor_allocations"])
    assert [group["kind"] for group in observed["groups"]] == ["attention", "attention", "convolution"]
    # Without hybrid evidence an allocation record keeps the vLLM 0.27.0/0.28.0 keys only.
    legacy = observer.worker_memory(_initialized()[0])
    assert [set(item) for item in legacy["tensor_allocations"]] == [{"size", "shared_by", "block_stride", "offset"}] * 2


def test_cache_groups_that_mix_layer_spec_evidence_are_refused():
    # A hybrid record holds layer_specs in every group. A record that holds them in one group only is malformed.
    worker, scheduler, config = _initialized(layers=True)
    cache = observer.worker_memory(worker, hybrid_evidence=True)
    scheduled = observer.scheduler_memory(scheduler, config, hybrid_evidence=True)
    del cache["groups"][1]["layer_specs"]
    del scheduled["groups"][1]["layer_specs"]
    with pytest.raises(ValueError, match="runtime memory cache lacks complete layer spec evidence"):
        runtime_memory._normalized_groups(cache, scheduled, cache["groups"][0]["spec_page_size_bytes"])


def test_uniform_group_with_an_unknown_inner_spec_class_is_refused():
    cache, _scheduler = _hybrid_caches(_mamba())
    group = cache["groups"][1]
    # A uniform group whose second inner spec has a class without a cache kind. The scheduler copy keeps the first
    # inner spec (kv_cache_utils.py:1110-1116), so the two copies still agree.
    group["spec_type"] = "vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs"
    group["layer_specs"]["kda.1"] = {
        **group["layer_specs"]["kda.1"],
        "spec_type": "vllm.v1.kv_cache_interface.RSWASpec",
    }
    scheduler = {"groups": [runtime_memory._scheduler_copy(item) for item in cache["groups"]]}
    with pytest.raises(
        ValueError, match=re.escape("unsupported runtime cache spec: 'vllm.v1.kv_cache_interface.RSWASpec'")
    ):
        runtime_memory._normalized_groups(cache, scheduler, PAGE)


@pytest.mark.parametrize("empty", [(), []])
def test_layer_without_cache_tensors_is_refused(empty):
    worker, _scheduler, _cache = _initialized(layers=True)
    worker.vllm_config.compilation_config.static_forward_context["slide.1"].kv_cache = empty
    with pytest.raises(ValueError, match="cache layer binds no tensors"):
        observer.worker_memory(worker, hybrid_evidence=True)


def test_state_group_keeps_its_kind_through_worker_memory():
    worker, _scheduler, cache = _initialized(layers=True)
    # The sliding-window attention layers become a one-block tail state group.
    cache.kv_cache_groups[1].kv_cache_spec = _spec(
        "CircularBufferSpec", block_size=16, page_size_bytes=64, dtype="torch.bfloat16"
    )
    groups = observer.worker_memory(worker, hybrid_evidence=True)["groups"]
    assert [group["kind"] for group in groups] == ["attention", "state", "convolution"]
    assert (groups[1]["state"], groups[1]["sliding_window"]) == ({"blocks_per_request": 1}, None)


def test_block_outer_tensor_region_ends_at_its_last_page():
    worker, _scheduler, cache = _initialized(layers=True)
    # Block-outer (kv_cache_interface.py:1479-1485): each 128-byte block holds the
    # 64-byte pages of both layers, so the last page ends exactly at the buffer end.
    tensor = cache.kv_cache_tensors[1]
    tensor.layer_stride, tensor.block_stride, tensor.offset = 64, 128, 0
    allocation = observer.worker_memory(worker, hybrid_evidence=True)["tensor_allocations"][1]
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
        ("mixed", "unsupported runtime cache tensor record"),
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
    elif change == "mixed":
        # Only the first tensor keeps a layer stride.
        del vars(tensor)["layer_stride"]
    else:
        context = worker.vllm_config.compilation_config.static_forward_context
        moved = _Storage(300, tensor.size)
        for name in tensor.layers:
            context[name].kv_cache = _Tensor(moved)
        worker.model_runner.kv_caches = [layer.kv_cache for layer in context.values()]
    with pytest.raises(ValueError, match=match):
        observer.worker_memory(worker, hybrid_evidence=True)


def _with_engine_core_values(worker, scheduler, cache):
    """Give initialized fakes the cache fields that vLLM 0.27.0 and 0.28.0 also have.

    Both versions have the eight cache settings and the scheduler and block pool
    hash block sizes. The engine core sets block_size and the startup capacity in
    its own process, so the worker keeps None where the scheduler has a value.
    """
    shared = {
        "mamba_cache_mode": "none",
        "mamba_block_size": None,
        "prefix_match_unit": None,
        "mamba_cache_dtype": "auto",
        "mamba_ssm_cache_dtype": "auto",
    }
    worker_cache = worker.vllm_config.cache_config
    vars(worker_cache).update(shared, block_size=None, kv_cache_size_tokens=None, kv_cache_max_concurrency=None)
    scheduler.vllm_config = SimpleNamespace(**vars(worker.vllm_config))
    scheduler.vllm_config.cache_config = SimpleNamespace(
        **{**vars(worker_cache), "block_size": 4, "kv_cache_size_tokens": 6400, "kv_cache_max_concurrency": 1.5625}
    )
    scheduler.block_size = scheduler.hash_block_size = 4
    scheduler.kv_cache_manager.block_pool.hash_block_size = 4
    return worker, scheduler, cache


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, indent=2).encode()


def _main_execution_config():
    """The execution record that main writes for the configuration of _initialized and _with_engine_core_values.

    It holds no hybrid cache field, the same for vLLM 0.27.0, 0.28.0, 0.32.0 and 0.33.0.
    """
    return {
        "model_config": {
            "model": "example/model",
            "revision": "immutable",
            "dtype": "torch.bfloat16",
            "quantization": "modelopt_fp4",
            "max_model_len": 4096,
            "enforce_eager": False,
        },
        "cache_config": {
            "cache_dtype": "bfloat16",
            "gpu_memory_utilization": 0.9,
            "kv_cache_memory_bytes": None,
            "enable_prefix_caching": True,
        },
        "scheduler_config": {"max_num_batched_tokens": 1024, "max_num_seqs": 64, "async_scheduling": False},
        "parallel_config": {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 2,
            "enable_expert_parallel": True,
            "decode_context_parallel_size": 1,
            "prefill_context_parallel_size": 1,
        },
        "compilation_config": {
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
            "max_cudagraph_capture_size": 64,
        },
        "kernel_config": {"moe_backend": "auto"},
        "quantization_config": {
            "type": "vllm.model_executor.layers.quantization.modelopt.ModelOptNvFp4Config",
            "quant_method": "NVFP4",
        },
    }


def test_legacy_runtime_records_stay_byte_identical_with_the_new_fields_present():
    # Expected values are what the observer wrote before hybrid cache support, for the same objects.
    worker, scheduler, cache = _with_engine_core_values(*_initialized())
    resolved = {
        **_main_execution_config(),
        "offload_config": {
            "offload_backend": "auto",
            "uva": {"cpu_offload_gb": 0},
            "prefetch": {"offload_group_size": 0, "offload_num_in_group": 1, "offload_prefetch_step": 1},
        },
    }
    for config in (worker.vllm_config, scheduler.vllm_config):
        assert _json_bytes(observer.resolved_config(config)) == _json_bytes(resolved)
    assert _json_bytes(observer.execution_config(worker.vllm_config)) == _json_bytes(_main_execution_config())
    groups = [
        {
            "layer_names": names,
            "spec_type": f"vllm.v1.kv_cache_interface.{spec}",
            "block_size_tokens": block,
            "spec_page_size_bytes": 64,
            "sliding_window": window,
            "dtype": "torch.bfloat16",
        }
        for names, spec, block, window in (
            (["full.0", "full.1"], "FullAttentionSpec", 16, None),
            (["slide.0", "slide.1"], "SlidingWindowSpec", 16, 32),
            (["conv.0"], "SlidingWindowSpec", 4, 4),
        )
    ]
    attention = "vllm.model_executor.layers.attention.Attention"
    convolution = "vllm.models.inkling.nvidia.sconv_swa_attn.InklingConvState"
    worker_record = {
        "num_blocks": 100,
        "available_cache_bytes": 13_000,
        "allocated_cache_bytes": 12_800,
        "storages": [
            {"device": "cuda:0", "pointer": 100, "size_bytes": 6400},
            {"device": "cuda:0", "pointer": 200, "size_bytes": 6400},
        ],
        "tensor_allocations": [
            {"size": 6400, "shared_by": ["full.0", "slide.0", "conv.0"], "block_stride": 0, "offset": 0},
            {"size": 6400, "shared_by": ["full.1", "slide.1"], "block_stride": 0, "offset": 0},
        ],
        "groups": [
            {**groups[0], "kind": "attention", "layer_classes": {"full.0": attention, "full.1": attention}},
            {**groups[1], "kind": "attention", "layer_classes": {"slide.0": attention, "slide.1": attention}},
            {**groups[2], "kind": "convolution", "layer_classes": {"conv.0": convolution}},
        ],
        "diagnostics": {"peak_activation_memory": 123, "cudagraph_memory_estimate": 456},
    }
    assert _json_bytes(observer.worker_memory(worker)) == _json_bytes(worker_record)
    scheduler_record = {
        "num_blocks": 100,
        "initial_free_blocks": 98,
        "reserved_blocks": 2,
        "null_block_id": 0,
        "watermark_blocks": 0,
        "pool_count": 1,
        "groups": groups,
    }
    assert _json_bytes(observer.scheduler_memory(scheduler, cache)) == _json_bytes(scheduler_record)
    # The block pool hash check belongs to hybrid evidence only.
    scheduler.kv_cache_manager.block_pool.hash_block_size = 16
    assert _json_bytes(observer.scheduler_memory(scheduler, cache)) == _json_bytes(scheduler_record)


def test_legacy_import_accepts_values_that_only_the_engine_core_holds(tmp_path, monkeypatch):
    original = _initialized
    monkeypatch.setattr(
        sys.modules[__name__], "_initialized", lambda **kwargs: _with_engine_core_values(*original(**kwargs))
    )
    cell = _evidence(tmp_path, monkeypatch)
    assert _resolve(cell, tmp_path)["runtime_memory"]["kv_cache_bytes"] == 88 * 128


def test_scheduler_memory_records_block_sizes_mamba_mode_and_startup_capacity():
    _worker, scheduler, cache = _with_engine_core_values(*_initialized())
    scheduler.block_size = scheduler.hash_block_size = 2176
    scheduler.kv_cache_manager.block_pool.hash_block_size = 2176
    scheduler.vllm_config.cache_config.mamba_cache_mode = "align"
    record = observer.scheduler_memory(scheduler, cache, hybrid_evidence=True)
    keys = ("block_size", "hash_block_size", "mamba_cache_mode", "kv_cache_size_tokens", "kv_cache_max_concurrency")
    assert {key: record[key] for key in keys} == {
        "block_size": 2176,
        "hash_block_size": 2176,
        "mamba_cache_mode": "align",
        "kv_cache_size_tokens": 6400,
        "kv_cache_max_concurrency": 1.5625,
    }
    # The worker record carries the group kind; the scheduler copy keeps the layer specs.
    assert all("kind" not in group and "layer_specs" in group for group in record["groups"])
    scheduler.kv_cache_manager.block_pool.hash_block_size = 4
    with pytest.raises(ValueError, match="hash block size"):
        observer.scheduler_memory(scheduler, cache, hybrid_evidence=True)


def test_hybrid_resolved_config_adds_the_cache_settings_but_not_the_startup_capacity():
    worker, scheduler, _cache = _with_engine_core_values(*_initialized())
    cache_config = observer.resolved_config(scheduler.vllm_config, hybrid_evidence=True)["cache_config"]
    assert {key: cache_config[key] for key in ("block_size", "mamba_cache_mode", "mamba_ssm_cache_dtype")} == {
        "block_size": 4,
        "mamba_cache_mode": "none",
        "mamba_ssm_cache_dtype": "auto",
    }
    assert not {"kv_cache_size_tokens", "kv_cache_max_concurrency"} & set(cache_config)
    assert observer.execution_config(worker.vllm_config, hybrid_evidence=True)["cache_config"]["block_size"] is None


@pytest.mark.parametrize("layout", ["layer_outer", "block_outer"])
def test_layer_strided_allocation_charges_one_backing_pool(layout):
    worker, scheduler, config = _initialized_layer_strided(layout=layout)
    observed = observer.worker_memory(worker)
    assert observed["allocated_cache_bytes"] == 12_800
    assert len(observed["storages"]) == 1
    assert [item["layers"] for item in observed["tensor_allocations"]] == [
        ["full.0", "full.1"],
        ["slide.0", "slide.1"],
        ["conv.0"],
    ]
    assert [group["kind"] for group in observed["groups"]] == ["attention", "attention", "convolution"]
    assert observer.scheduler_memory(scheduler, config)["reserved_blocks"] == 2


@pytest.mark.parametrize("layout", ["layer_outer", "block_outer"])
def test_layer_strided_records_carry_only_layer_strided_keys(layout):
    worker, _scheduler, _config = _initialized_layer_strided(layout=layout)
    records = observer.worker_memory(worker)["tensor_allocations"]
    assert records
    for record in records:
        assert set(record) == {"size", "layers", "layer_stride", "block_stride", "offset"}


def _break_layer_strided(worker, change):
    """Make one unsupported change to the first cache group or the first tensor record."""
    config = worker.model_runner.kv_cache_config
    group, record = config.kv_cache_groups[0], config.kv_cache_tensors[0]
    if change == "host_resident_tensor":
        record.host_resident = True
    elif change == "host_resident_group":
        group.host_resident = True
    elif change == "hisparse_role":
        group.role = _Role.HISPARSE_SOURCE
    elif change == "offset_beyond_allocation":
        record.offset = 64
    elif change == "block_stride_below_page":
        record.block_stride = 32
    elif change == "second_storage":
        layer = worker.vllm_config.compilation_config.static_forward_context["conv.0"]
        layer.kv_cache = _Tensor(_Storage(300, 100 * 128))
        worker.model_runner.kv_caches.append(layer.kv_cache)
    elif change == "mixed_record":
        record.shared_by = vars(record).pop("layers")
    elif change == "unknown_layer":
        record.layers = ["full.0", "ghost"]
    elif change == "uniform_wrapper":
        spec = _typed("vllm.v1.kv_cache_interface", "UniformTypeKVCacheSpecs")
        vars(spec).update(vars(group.kv_cache_spec))
        group.kv_cache_spec = spec


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("host_resident_tensor", "host-resident cache tensors"),
        ("host_resident_group", "host-resident cache groups"),
        ("hisparse_role", "unsupported cache group role"),
        ("offset_beyond_allocation", "exceed their backing allocation"),
        ("block_stride_below_page", "page exceeds its block stride"),
        ("second_storage", "multiple backing allocations"),
        ("mixed_record", "unsupported runtime cache tensor record"),
        ("unknown_layer", "missing or duplicate cache layers"),
        ("uniform_wrapper", "unsupported runtime cache spec"),
    ],
)
def test_layer_strided_allocation_rejects_unsupported_records(change, match):
    worker, _scheduler, _config = _initialized_layer_strided()
    _break_layer_strided(worker, change)
    with pytest.raises(ValueError, match=match):
        observer.worker_memory(worker)


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


@pytest.mark.parametrize("version", ["0.32.0", "0.33.0"])
def test_new_versions_observe_and_finalize_layer_strided_memory(tmp_path, monkeypatch, version):
    cell = _evidence(tmp_path, monkeypatch, version=version, initialized=_initialized_layer_strided)
    resolved = _resolve(cell, tmp_path, expected_backend_version=version)
    assert [group["kind"] for group in resolved["cache_groups"]] == ["attention", "attention", "convolution"]
    assert resolved["runtime_memory"]["kv_cache_bytes"] == 88 * 128


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
        ("mamba_spec_without_state", "lacks blocks_per_request evidence"),
        ("mamba_spec_with_state", "runtime memory cache kind is unsupported"),
        ("mla_spec", "unsupported runtime cache spec: 'vllm.v1.kv_cache_interface.MLAAttentionSpec'"),
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
    elif change in ("mamba_spec_without_state", "mamba_spec_with_state"):
        # The vLLM 0.27.0 path refuses every state group.
        for group in (payload["cache"]["groups"][0], scheduler["cache"]["groups"][0]):
            group["spec_type"] = "vllm.v1.kv_cache_interface.MambaSpec"
            if change == "mamba_spec_with_state":
                group["state"] = {"blocks_per_request": 1}
        payload["cache"]["groups"][0]["kind"] = "state"
    elif change == "mla_spec":
        # Hand-made records of every rank with a class outside the vLLM 0.27.0 allow-list.
        mla = "vllm.v1.kv_cache_interface.MLAAttentionSpec"
        for other in sorted(tmp_path.glob("pod-*/fpm-memory-*.json")):
            if other not in (path, scheduler_path):
                record = json.loads(other.read_text())
                record["cache"]["groups"][0]["spec_type"] = mla
                other.write_text(json.dumps(record))
        payload["cache"]["groups"][0]["spec_type"] = scheduler["cache"]["groups"][0]["spec_type"] = mla
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


def test_unaudited_runtime_version_errors_name_every_memory_version(tmp_path, monkeypatch):
    worker, _scheduler, _config = _initialized()
    monkeypatch.setattr(observer, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(observer.importlib.metadata, "version", lambda _name: "0.31.0")
    _write_provenance(tmp_path / "collector-provenance.json", cell_id="test")
    observer.observe("worker", worker, dp_rank=0, tp_rank=0, pp_rank=0)
    payload = json.loads((tmp_path / "fpm-memory-worker-dp0-tp0-pp0.json").read_text())
    assert payload["status"] == "unresolved"
    assert payload["error"] == (
        "ValueError: memory observation currently supports vLLM 0.27.0, 0.32.0, 0.33.0; found 0.31.0"
    )
    with pytest.raises(ValueError) as raised:
        _resolve(_cell(), tmp_path, expected_backend_version="0.31.0")
    assert str(raised.value) == "runtime memory conversion currently supports vLLM 0.27.0, 0.32.0, 0.33.0"


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


@pytest.mark.parametrize("version", ["0.32.0", "0.33.0"])
def test_execution_observation_records_new_versions(tmp_path, monkeypatch, version):
    provenance_path = tmp_path / "serving-provenance.json"
    provenance_path.write_text(json.dumps({"run_id": f"serving-{version}", "purpose": "matched_serving_accuracy"}))
    output = tmp_path / "observations"
    monkeypatch.setenv("FPM_EXECUTION_OUTPUT_DIR", str(output))
    monkeypatch.setenv("FPM_EXECUTION_PROVENANCE_FILE", str(provenance_path))
    monkeypatch.setattr(observer.importlib.metadata, "version", lambda _name: version)

    class Worker:
        def compile_or_warm_up_model(self):
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
    worker.model_runner = SimpleNamespace(
        attn_groups=[], kv_cache_config=SimpleNamespace(kv_cache_groups=[_full_attention_group()])
    )
    assert worker.compile_or_warm_up_model() == "compiled"
    result = json.loads((output / "fpm-execution-worker-dp0-tp0-pp0.json").read_text())
    assert result["status"] == "observed", result.get("error")
    assert result["backend_version"] == version
    assert result["attention_groups"][0]["backend_class"] == "actual.runtime.ResolvedBackend"


def test_execution_observation_skips_empty_attention_group_lists(tmp_path, monkeypatch):
    # Model Runner V2 appends [] for a cache group without layer views
    # (vllm/v1/worker/gpu/attn_utils.py 0.32.0:220-222, 0.33.0:221-223).
    monkeypatch.setattr(observer, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(observer.importlib.metadata, "version", lambda _name: "0.32.0")
    (tmp_path / "collector-provenance.json").write_text("{}")
    group = SimpleNamespace(
        backend=type("ResolvedBackend", (), {"__module__": "actual.runtime"}),
        layer_names=["layer.0"],
        kv_cache_group_id=1,
    )
    worker = SimpleNamespace(
        vllm_config=_vllm_config(tp=1, dp=1), model_runner=SimpleNamespace(attn_groups=[[], [group]])
    )
    observer.observe_execution(worker, dp_rank=0, tp_rank=0, pp_rank=0)
    result = json.loads((tmp_path / "fpm-execution-worker-dp0-tp0-pp0.json").read_text())
    assert result["status"] == "observed", result.get("error")
    assert result["attention_groups"] == [
        {"kv_cache_group_id": 1, "backend_class": "actual.runtime.ResolvedBackend", "layer_names": ["layer.0"]}
    ]


def test_worker_import_error_names_every_audited_version(monkeypatch):
    monkeypatch.setitem(sys.modules, "fpm_memory_observer", observer)
    monkeypatch.setitem(sys.modules, "vllm.distributed", None)
    spec = importlib.util.spec_from_file_location(
        "fpm_memory_worker", Path(observer.__file__).with_name("fpm_memory_worker.py")
    )
    with pytest.raises(RuntimeError) as raised:
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    assert str(raised.value).endswith(
        "execution observation supports vLLM 0.27.0/0.28.0/0.32.0/0.33.0, memory supports 0.27.0/0.32.0/0.33.0"
    )


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


def _full_attention_group():
    # A plain attention model: every layer has one spec, so vLLM 0.33.0 makes one group of that spec
    # (v1/core/kv_cache_planning.py:1138-1142).
    full = _spec("FullAttentionSpec", block_size=16, page_size_bytes=8192, sliding_window=None, dtype="torch.bfloat16")
    return _group(["layer.0", "layer.1"], full)


def _uniform_group(specs):
    # The packed path of vLLM 0.33.0 makes every group a UniformTypeKVCacheSpecs (v1/core/kv_cache_planning.py:1159,
    # :1026, :1037-1039). The group takes the block and page of its inner specs.
    first = next(iter(specs.values()))
    uniform = _spec(
        "UniformTypeKVCacheSpecs",
        kv_cache_specs=specs,
        block_size=first.block_size,
        page_size_bytes=first.page_size_bytes,
    )
    return _group(list(specs), uniform)


def _uniform_mla_group():
    # The GLM-5.3-Flash sparse attention layers: MLA latent specs (models/glm5next/common/attention.py:112-179).
    latent = _spec("MLAAttentionSpec", block_size=2176, page_size_bytes=PAGE, sliding_window=None, dtype="torch.uint8")
    return _uniform_group({"mla.0": latent, "mla.1": latent})


def _mamba_group():
    # The GLM-5.3-Flash KDA state layers (models/glm5next/common/kda.py:168-176).
    return _uniform_group({"kda.0": _mamba(), "kda.1": _mamba()})


def _observe_execution(tmp_path, monkeypatch, *, version, groups, cache_settings=None):
    """Write the execution record of one worker whose initialized cache holds ``groups``; return the record.

    The configuration is the one of _initialized and _with_engine_core_values, updated with ``cache_settings``.
    vLLM 0.33.0 sets Mamba align mode only for a model with Mamba layers when prefix caching is on
    (model_executor/models/config.py:763-787), so the cache configuration holds align mode only when ``groups`` holds
    a Mamba spec, as a group spec or as a layer spec of a uniform group.
    """
    worker = _with_engine_core_values(*_initialized())[0]
    vars(worker.vllm_config.cache_config).update(cache_settings or {})
    mamba = any(
        type(inner).__name__ == "MambaSpec"
        for group in groups
        for inner in getattr(group.kv_cache_spec, "kv_cache_specs", {"": group.kv_cache_spec}).values()
    )
    if mamba:
        vars(worker.vllm_config.cache_config).update(mamba_cache_mode="align", mamba_block_size=2176)
    attention = SimpleNamespace(
        backend=type("ResolvedBackend", (), {"__module__": "actual.runtime"}),
        layer_names=list(groups[0].layer_names),
        kv_cache_group_id=0,
    )
    worker.model_runner = SimpleNamespace(
        attn_groups=[[attention]], kv_cache_config=SimpleNamespace(kv_cache_groups=groups)
    )
    monkeypatch.setattr(observer, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(observer.importlib.metadata, "version", lambda _name: version)
    for name in ("FPM_EXECUTION_OUTPUT_DIR", "FPM_EXECUTION_PROVENANCE_FILE"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "collector-provenance.json").write_text("{}")
    path = tmp_path / "fpm-execution-worker-dp0-tp0-pp0.json"
    path.unlink(missing_ok=True)
    observer.observe_execution(worker, dp_rank=0, tp_rank=0, pp_rank=0)
    record = json.loads(path.read_text())
    assert record["status"] == "observed", record.get("error")
    return record


def _mla_group():
    # An MLA-only model gets one MLAAttentionSpec group (v1/core/kv_cache_planning.py:1138-1142, :314-327, :206-214).
    latent = _spec("MLAAttentionSpec", block_size=2176, page_size_bytes=PAGE, sliding_window=None, dtype="torch.uint8")
    return _group(["mla.0", "mla.1"], latent)


def _glm53_flash_groups():
    """The GLM-5.3-Flash cache groups of a layout whose block axis is not outermost.

    The packed path then returns None (v1/core/kv_cache_planning.py:931-932), and the path that groups layers by
    spec gives each group one spec (:1176, :556, :206-214). The sparse attention layers give MLAAttentionSpec
    (models/glm5next/common/attention.py:112-119), the tail rings CircularBufferSpec (:170-179) and the KDA layers
    MambaSpec (models/glm5next/common/kda.py:168-176).
    """
    tail = _spec("CircularBufferSpec", block_size=4, page_size_bytes=71_808, dtype="torch.bfloat16")
    kda = [_group([f"kda.{index}"], _mamba()) for index in range(4)]
    return [_mla_group(), _group(["tail.0"], tail), *kda]


_HYBRID_CACHE_CONFIGURATIONS = {
    "full_attention": ([_full_attention_group()], False),
    "mla": ([_mla_group()], False),
    # A packed attention-only cache: UniformTypeKVCacheSpecs groups whose layer specs are attention specs.
    "uniform_attention": ([_uniform_mla_group()], False),
    "direct_mamba": ([_full_attention_group(), _group(["kda.0", "kda.1"], _mamba())], True),
    "uniform_mamba": ([_uniform_mla_group(), _mamba_group()], True),
    "glm53_flash": (_glm53_flash_groups(), True),
}


@pytest.mark.parametrize(
    ("version", "case"),
    [("0.33.0", case) for case in _HYBRID_CACHE_CONFIGURATIONS] + [("0.32.0", "glm53_flash")],
)
def test_execution_record_adds_the_hybrid_cache_fields_only_with_a_recurrent_state_group(
    tmp_path, monkeypatch, version, case
):
    groups, recurrent = _HYBRID_CACHE_CONFIGURATIONS[case]
    # The cache configuration holds every plain and every hybrid cache setting, so only the groups decide.
    record = _observe_execution(
        tmp_path, monkeypatch, version=version, groups=groups, cache_settings={"num_gpu_blocks_override": None}
    )
    cache = record["resolved_config"]["cache_config"]
    plain = observer._CONFIG_FIELDS["cache_config"]
    if recurrent and version == "0.33.0":
        assert sorted(cache) == sorted([*plain, *observer._HYBRID_CACHE_FIELDS])
        assert cache["mamba_cache_mode"] == "align"
    else:
        assert sorted(cache) == sorted(plain)


def test_execution_record_of_a_patch_release_has_no_hybrid_cache_fields(tmp_path, monkeypatch):
    # The release check accepts 0.33.1: the imported package reports the metadata version, and the cached source
    # check of the audited release found no changed file.
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.33.1"
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setattr(observer, "_SOURCE_PROBLEMS", {"0.33.0": None})
    record = _observe_execution(
        tmp_path,
        monkeypatch,
        version="0.33.1",
        groups=_glm53_flash_groups(),
        cache_settings={"num_gpu_blocks_override": None},
    )
    assert record["audited_backend_version"] == "0.33.0"
    assert sorted(record["resolved_config"]["cache_config"]) == sorted(observer._CONFIG_FIELDS["cache_config"])


def test_execution_record_of_a_plain_model_on_vllm_0330_keeps_the_bytes_of_main(tmp_path, monkeypatch):
    record = _observe_execution(tmp_path, monkeypatch, version="0.33.0", groups=[_full_attention_group()])
    assert _json_bytes(record["resolved_config"]) == _json_bytes(_main_execution_config())


@pytest.mark.parametrize("version", ["0.27.0", "0.28.0", "0.32.0", "0.33.0", "0.30.0"])
def test_execution_observation_supports_vllm_0_33_0(tmp_path, monkeypatch, version):
    output = tmp_path / "observations"
    provenance_path = tmp_path / "serving-provenance.json"
    provenance_path.write_text(json.dumps({"run_id": "serving-123", "purpose": "matched_serving_accuracy"}))
    monkeypatch.setenv("FPM_EXECUTION_OUTPUT_DIR", str(output))
    monkeypatch.setenv("FPM_EXECUTION_PROVENANCE_FILE", str(provenance_path))
    monkeypatch.setattr(observer.importlib.metadata, "version", lambda _name: version)
    config = _vllm_config(tp=1, dp=1)
    # The cache config holds the hybrid settings for every version here, and only a 0.33.0 record keeps them.
    vars(config.cache_config).update(mamba_cache_mode="align", prefix_match_unit=None)
    group = SimpleNamespace(
        backend=type("ResolvedBackend", (), {"__module__": "actual.runtime"}),
        layer_names=["layer.0"],
        kv_cache_group_id=0,
    )
    # A hybrid cache configuration of vLLM 0.33.0, which the other versions never build: only its groups decide.
    cache = SimpleNamespace(kv_cache_groups=[_uniform_mla_group(), _mamba_group()])
    worker = SimpleNamespace(
        vllm_config=config, model_runner=SimpleNamespace(attn_groups=[[group]], kv_cache_config=cache)
    )
    observer.observe_execution(worker, dp_rank=0, tp_rank=0, pp_rank=0)
    record = json.loads((output / "fpm-execution-worker-dp0-tp0-pp0.json").read_text())
    if version == "0.30.0":
        # An unaudited version stays unresolved, and the error names the audited versions.
        assert record["status"] == "unresolved"
        assert "0.33.0" in record["error"]
    else:
        assert record["status"] == "observed"
        assert record["backend_version"] == version
        assert record["attention_groups"][0]["kv_cache_group_id"] == 0
        cache = record["resolved_config"]["cache_config"]
        if version == "0.33.0":
            assert cache["mamba_cache_mode"] == "align"
            assert "prefix_match_unit" in cache
        else:
            assert {"mamba_cache_mode", "prefix_match_unit"}.isdisjoint(cache)


@pytest.mark.parametrize("version", ["0.33.0", "0.30.0"])
@pytest.mark.usefixtures("no_models_or_timing_data")
def test_execution_inspection_audits_vllm_0_33_0(tmp_path, version):
    from collector.fpm_forward.execution_evidence import inspect_execution_evidence
    from collector.fpm_forward.native_artifact import NativeCollection

    plan = _pending_plan(version)
    cell = plan.cells[0]
    raw = tmp_path / "raw"
    raw.mkdir()
    collection = NativeCollection(
        points=(),
        rank_timings=(),
        backend_version=version,
        collector_attempt_id="attempt",
        runtime_run_id=cell.cell_id,
        runtime_grid_digest="grid",
    )
    # Evidence version 1 skips the measurement evidence, which this audit does not read.
    missing = inspect_execution_evidence(cell, raw, collection, plan=plan, evidence_version=1)["missing_evidence"]
    unaudited = [item for item in missing if item.startswith("execution configuration inspection is unaudited")]
    assert unaudited == ([] if version == "0.33.0" else ["execution configuration inspection is unaudited for 0.30.0"])


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_runner_stages_only_the_execution_observer_for_an_execution_only_version_without_a_profile():
    # vLLM 0.28.0 is audited for execution observation, not for memory observation.
    plan = replace(_pending_plan("0.28.0"), _fpm_profile_json=None)
    assert plan.runtime_instrumentation is None
    for cell in plan.cells:
        assert not cell.execution_identity[0]
        assert runner._observe_runtime_execution(plan, cell) is True
        assert runner._observe_runtime_memory(plan, cell) is False
        args = runner._cell_generator_overrides(plan, cell, {})["params"]["agg"]["extra_cli_args"]
        assert args[args.index("--worker-cls") + 1] == "fpm_memory_worker.FpmExecutionWorker"
        assert args[args.index("--scheduler-cls") + 1] == "fpm_memory_scheduler.FpmExecutionInstrumentedScheduler"


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


@pytest.mark.parametrize("version", ["0.32.0", "0.33.0"])
@pytest.mark.usefixtures("no_models_or_timing_data")
def test_new_memory_versions_plan_audited_hooks(tmp_path, version):
    plan = _pending_plan(version)
    assert plan.to_dict()["runtime_memory_policy"] == {
        "source": "vllm_initialization",
        "supported_vllm_version": version,
        "selected_vllm_version": version,
        "observation": "enabled",
        "evidence_schema_version": 1,
        "async_scheduling": False,
    }
    for cell in plan.cells:
        output = tmp_path / cell.cell_id
        output.mkdir()
        runner._render_cell(plan, cell, output, {})
        script = (output / "run.sh").read_text()
        assert "--worker-cls fpm_memory_worker.FpmResourceWorker" in script
        assert "--scheduler-cls fpm_memory_scheduler.FpmResourceInstrumentedScheduler" in script
        assert script.count("--no-async-scheduling") == 1


@pytest.mark.parametrize("version", ["0.25.1", "0.28.0", "0.31.0"])
@pytest.mark.usefixtures("no_models_or_timing_data")
def test_unaudited_memory_versions_keep_the_legacy_policy(version):
    assert _pending_plan(version).to_dict()["runtime_memory_policy"] == {
        "source": "vllm_initialization",
        "supported_vllm_version": "0.27.0",
        "selected_vllm_version": version,
        "observation": "unavailable_for_runtime",
        "evidence_schema_version": 1,
        "async_scheduling": None,
    }


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_plan_saved_before_the_new_memory_versions_stops_on_resume(tmp_path, monkeypatch):
    # A pending 0.32.0 plan as main 2022b019 saves it: observation unavailable_for_runtime.
    with monkeypatch.context() as patch:
        patch.setattr(planner, "MEMORY_SUPPORTED_VERSIONS", ("0.27.0",))
        saved = _pending_plan("0.32.0")
        payload = saved.to_dict()
    assert payload["runtime_memory_policy"]["observation"] == "unavailable_for_runtime"
    runtime_memory.validate_saved_plan(payload)
    current = _pending_plan("0.32.0").to_dict()
    assert current["runtime_memory_policy"]["observation"] == "enabled"
    assert {key for key in payload.keys() | current.keys() if payload.get(key) != current.get(key)} == {
        "runtime_memory_policy",
        "sha256",
    }
    artifacts, checkpoints = tmp_path / "artifacts", tmp_path / "checkpoints"
    campaign = artifacts / saved.sha256[:16]
    campaign.mkdir(parents=True)
    (campaign / "collection-plan.json").write_text(json.dumps(payload))
    checkpoints.mkdir()
    checkpoint = {"schema": runner.CHECKPOINT_SCHEMA, "plan_sha256": saved.sha256, "cells": {}}
    (checkpoints / "fpm_forward.json").write_text(json.dumps(checkpoint))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    # collect-fpm --resume plans again with the current code and checks the checkpoint before any cell renders.
    with pytest.raises(ValueError, match="checkpoint does not match the current frozen plan"):
        runner.run_collection(
            _pending_plan("0.32.0"),
            generator_overrides={},
            checkpoint_dir=str(checkpoints),
            artifact_root=str(artifacts),
            resume=True,
            retry_failed=False,
        )
    assert not list(artifacts.rglob("run.sh"))
    # The repeatability loader recomputes the policy of a plan with a runtime version, so it cannot match the file.
    with pytest.raises(ValueError, match="cannot be reconstructed exactly by this collector version"):
        repeatability.load_repeatability_source(campaign)
    assert {path: path.read_bytes() for path in before} == before
    # Control: the same plan saved by the current code loads exactly.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "collection-plan.json").write_text(json.dumps(current))
    assert repeatability.load_repeatability_source(fresh).to_dict() == current


@pytest.mark.parametrize(
    "flag",
    [
        "--worker-cls",
        "--scheduler-cls",
        "--kv-cache-memory-bytes",
        "--num-gpu-blocks-override",
        "--async-scheduling",
        "--mamba-cache-mode",
    ],
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
def test_pending_memory_refuses_a_policy_mamba_cache_mode():
    plan = _pending_plan()
    policy = {"params": {"agg": {"extra_cli_args": ["--mamba-cache-mode=align"]}}}
    cell = replace(plan.cells[0], backend_policy=planner.BackendPolicy("conflict", policy, {}))
    with pytest.raises(
        ValueError, match="^backend policy cannot select the Mamba cache mode for runtime memory observation$"
    ):
        runner._cell_generator_overrides(plan, cell, {})


@pytest.mark.parametrize("architecture", ["Glm5NextForConditionalGeneration", None])
@pytest.mark.usefixtures("no_models_or_timing_data")
def test_glm5next_launch_loads_the_language_model_only(tmp_path, architecture):
    # A GLM-5.3-Flash collection plan carries the point list that the planner generates.
    plan = _glm_plan(tmp_path) if architecture is not None else _pending_plan()
    assert (plan.capability.architecture == "Glm5NextForConditionalGeneration") == (architecture is not None)
    for cell in plan.cells:
        args = runner._cell_generator_overrides(plan, cell, {})["params"]["agg"]["extra_cli_args"]
        # GLM probes and collection load the text decoder only (see the engine arguments of the vLLM producer section
        # in docs/glm53flash-fpm.md).
        assert ("--language-model-only" in args) == (architecture is not None)


# A stand-in commit of the GLM-5.3-Flash checkpoint. vLLM loads the revision of a GLM-5.3-Flash plan, and the producer
# and the planner accept 40 lowercase hex characters only.
_GLM_REVISION = "c" * 40


def _glm_plan(
    tmp_path,
    *,
    hash_block_size_tokens=2176,
    max_batch_size=256,
    max_num_tokens=8192,
    model_revision=_GLM_REVISION,
    kda_checkpoint_alignment=None,
    **changes,
):
    """A GLM-5.3-Flash TP 4 collection plan with the cache resources that the runtime probe measured.

    The config is a small stand-in with the hybrid layer types. The options are the limits that onboarding passes:
    ``max_num_tokens`` tokens, ``max_batch_size`` sequences and a 1,048,576-token context. The profile records the
    same token and sequence limits, because accepted runtime memory is exact for its scheduler settings. The producer
    runs no global warm-up iterations. ``changes`` replace option fields before planning. With a
    ``kda_checkpoint_alignment``, each of the four KDA groups holds one prefill checkpoint block of that alignment, as
    the import writes it for a launch that prefills KDA with FlashKDA. Without it, they hold none, as with the Triton
    KDA prefill backend.
    """
    from .test_runtime_observations import glm_tp4_cache_groups

    cache_groups = glm_tp4_cache_groups()
    # A FlashKDA prefill gives each KDA spec one checkpoint block aligned to 16 tokens (vLLM 0.33.0
    # models/glm5next/common/kda.py:168-176, model_executor/layers/mamba/kda_checkpoint.py:9-10), and the import
    # keeps the count and the alignment of a MambaSpec group (collector/fpm_forward/runtime_memory.py:661-675). The
    # Triton backend gives no checkpoint block, and the import then keeps a count of 0 without an alignment.
    for group in cache_groups:
        if "prefill_checkpoint_blocks" in group:
            group.pop("prefill_checkpoint_alignment", None)
            if kda_checkpoint_alignment is None:
                group.update(prefill_checkpoint_blocks=0)
            else:
                group.update(prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=kda_checkpoint_alignment)
    architecture = "Glm5NextForConditionalGeneration"
    path = _mixed_attention_config(tmp_path, architecture, True)
    memory = {
        "kv_cache_bytes": 185_126_092_800,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 1_048_576,
        "provenance": "Runtime probe capacity of GLM-5.3-Flash at TP 4, declared for this test.",
        "enable_prefix_caching": True,
        "mamba_cache_mode": "align",
    }
    if hash_block_size_tokens is not None:
        memory["hash_block_size_tokens"] = hash_block_size_tokens
    profile = {
        "schema_version": 1,
        "model": str(path),
        "model_revision": model_revision,
        "architecture": architecture,
        "context_length": 1_048_576,
        "num_experts": 8,
        "provenance": "Collector test declaration; not a measured memory qualification.",
        "deployments": [
            {
                "system": "b200_sxm",
                "backend": "vllm",
                "backend_version": "0.33.0",
                "tp": 4,
                "dp": 1,
                "moe_tp": 4,
                "moe_ep": 1,
                "gemm_quant_mode": "bfloat16",
                "moe_quant_mode": "bfloat16",
                "fmha_quant_mode": "bfloat16",
                "comm_quant_mode": "half",
                "kv_cache_dtype": "bfloat16",
                "resources": {
                    "runtime_memory": memory,
                    "cache_layout": "grouped",
                    "cache_groups": cache_groups,
                    "max_num_tokens": max_num_tokens,
                    "max_batch_size": max_batch_size,
                    "provenance": "Runtime probe resources of GLM-5.3-Flash at TP 4, declared for this test.",
                },
            }
        ],
    }
    argv = [
        "--model-path",
        str(path),
        "--gpu",
        "b200_sxm",
        "--fpm-max-gpus",
        "4",
        "--fpm-gpu-counts",
        "4",
        "--fpm-parallel-presets",
        "pure_tp",
        "--fpm-max-model-len",
        "1048576",
        "--fpm-max-num-batched-tokens",
        str(max_num_tokens),
        "--fpm-max-num-seqs",
        str(max_batch_size),
        "--fpm-max-prefill-isl",
        str(max_num_tokens),
        "--fpm-max-prefill-batch-size",
        str(max_batch_size),
        "--fpm-prefill-cudagraph-policy",
        "runtime",
        "--fpm-warmup-iterations",
        "0",
    ]
    return _plan(
        profile,
        model_path=str(path),
        model_architecture=architecture,
        model_config_path=str(path),
        has_model_cases=False,
        selected_ops={"attention_context", "attention_generation"},
        options=replace(FPMCollectionOptions.from_args(cli._parser().parse_args(argv)), **changes),
    )


def test_glm_collection_plan_freezes_the_generated_point_list(tmp_path):
    from collector.fpm_forward.glm53flash_points import glm_point_manifest

    plan = _glm_plan(tmp_path)
    _, canonical, digest = glm_point_manifest(
        hash_block_size=2176,
        token_budget=8192,
        prefill_batch_limit=256,
        decode_batch_limit=256,
        measured_context=131072,
        kv_cache_bytes=185_126_092_800,
        page_size_bytes=13_516_800,
        state_groups=5,
        # The KDA groups of the fixture profile hold no prefill checkpoint block.
        checkpoint_alignment=None,
        checkpoint_groups=0,
    )
    saved = plan.to_dict()

    assert plan.options.vllm_max_model_len == 1_048_576
    assert plan.options.benchmark_points_origin == "glm53flash_grid_v1"
    assert (plan.options.benchmark_points_json, plan.options.benchmark_points_sha256) == (canonical, digest)
    assert saved["point_generation"]["source"] == "generated_glm53flash_grid"
    assert saved["point_generation"]["planned_point_count"] == 537
    assert saved["point_generation"]["manifest_sha256"] == digest
    assert saved["options"]["benchmark_points"]["origin"] == "glm53flash_grid_v1"
    # The plan hash covers the list and its origin, so a saved plan validates and a second build reproduces it.
    runtime_memory.validate_saved_plan(saved)
    assert _glm_plan(tmp_path).sha256 == plan.sha256
    with pytest.raises(ValueError) as error:
        _glm_plan(tmp_path, hash_block_size_tokens=None)
    assert str(error.value) == "GLM-5.3-Flash collection requires the accepted profile's hash_block_size_tokens"
    deepseek = planner.build_collection_plan(
        backend="vllm",
        model_path="deepseek-ai/DeepSeek-V4.1-Flash",
        system="b200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        options=FPMCollectionOptions.from_args(_args(fpm_enforce_eager=True)),
        model_architecture="DeepseekV41ForCausalLM",
        has_model_cases=False,
        collector_config={"aic_database_version": "0.27.0"},
    )
    for other in (_pending_plan(), deepseek):
        assert "benchmark_points" not in other.to_dict()["options"]


def test_glm_collection_plan_with_prefill_checkpoint_blocks_keeps_every_prefill(tmp_path):
    from collector.fpm_forward.glm53flash_points import glm_point_manifest

    # The KDA groups of a FlashKDA launch hold one prefill checkpoint block aligned to 16 tokens. vLLM 0.33.0 then saves
    # the checkpoint inside every planned prefill chunk, so the align rule drops no point.
    plan = _glm_plan(tmp_path, kda_checkpoint_alignment=16)
    _, canonical, digest = glm_point_manifest(
        hash_block_size=2176,
        token_budget=8192,
        prefill_batch_limit=256,
        decode_batch_limit=256,
        measured_context=131072,
        kv_cache_bytes=185_126_092_800,
        page_size_bytes=13_516_800,
        state_groups=5,
        checkpoint_alignment=16,
        checkpoint_groups=4,
    )
    saved = plan.to_dict()

    assert (plan.options.benchmark_points_json, plan.options.benchmark_points_sha256) == (canonical, digest)
    assert digest == "2c81f2f524251c18a5a0c33607544695042c48f8151f2b9c3f067d983fa4cbc6"
    assert saved["point_generation"]["planned_point_count"] == 577
    # The capacity term counts the four checkpoint blocks of each request, and the cache still holds every point.
    assert saved["point_generation"]["dropped_points"] == {
        "prefill": {"candidates": 496, "mamba_align_split": 0, "cache_capacity": 0},
        "decode": {"candidates": 97, "token_budget": 16, "cache_capacity": 0},
    }
    runtime_memory.validate_saved_plan(saved)


def test_glm_collection_plan_records_the_points_that_each_rule_drops(tmp_path):
    plan = _glm_plan(tmp_path)
    saved = plan.to_dict()
    # The counts of the generator's log line. Onboarding sets no log handler, so the plan keeps them.
    dropped = {
        "prefill": {"candidates": 496, "mamba_align_split": 40, "cache_capacity": 0},
        "decode": {"candidates": 97, "token_budget": 16, "cache_capacity": 0},
    }

    assert saved["point_generation"]["dropped_points"] == dropped
    assert saved["options"]["benchmark_points"]["dropped"] == dropped
    # Each phase keeps its candidates minus the points that its rules drop.
    assert saved["point_generation"]["planned_point_count"] == sum(
        counts["candidates"] - sum(value for name, value in counts.items() if name != "candidates")
        for counts in dropped.values()
    )
    # The plan hash covers the counts, so a saved plan validates. A plan without a generated list records none.
    runtime_memory.validate_saved_plan(saved)
    assert "dropped_points" not in _pending_plan().to_dict()["point_generation"]


@pytest.mark.parametrize(
    "changes",
    [{"max_num_seqs": None}, {"max_num_seqs": None, "max_prefill_batch_size": None, "max_decode_batch_size": 256}],
    ids=["decode", "prefill"],
)
def test_glm_collection_plan_requires_batch_limits(tmp_path, changes):
    # The runtime falls back to the profile's batch size, but the list needs the limit of each phase.
    with pytest.raises(ValueError) as error:
        _glm_plan(tmp_path, **changes)
    assert str(error.value) == "GLM-5.3-Flash collection requires --fpm-max-num-seqs or the per-phase batch limits"


@pytest.mark.parametrize(
    ("revision", "accepted"),
    [
        ("v1.0", False),
        ("04c4e9e", False),
        ("04C4E9E95C5DA8862DCED7E5056455116F83A7E0", False),
        ("04c4e9e95c5da8862dced7e5056455116f83a7e0", True),
    ],
    ids=["tag", "short_hash", "uppercase", "commit"],
)
def test_glm_collection_plan_requires_a_commit_revision(tmp_path, revision, accepted):
    # vLLM loads this revision, and the producer reads the tokenizer and the configuration of the same revision. The
    # producer refuses anything but 40 lowercase hex characters only after the engine starts, so planning refuses it.
    if accepted:
        assert _glm_plan(tmp_path, model_revision=revision).fpm_profile.model_revision == revision
        return
    with pytest.raises(ValueError) as error:
        _glm_plan(tmp_path, model_revision=revision)
    assert str(error.value) == (
        f"GLM-5.3-Flash collection requires a model revision of 40 lowercase hex characters, got {revision!r}"
    )


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_glm_collection_plan_takes_the_revision_of_the_probe_launch(tmp_path):
    # Like the runner, the planner takes the revision of the accepted probe launch before the one of the profile.
    with pytest.raises(ValueError) as error:
        _probe_backed_glm_plan(tmp_path, launch_revision="v1.0")
    assert str(error.value) == (
        "GLM-5.3-Flash collection requires a model revision of 40 lowercase hex characters, got 'v1.0'"
    )


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_saved_glm_plan_reloads_with_its_generated_point_list(tmp_path):
    from collector.fpm_forward.repeatability import load_repeatability_source

    # Readiness and the runtime bindings reload every saved campaign and compare its hash with the checkpoint.
    for plan, origin in ((_glm_plan(tmp_path), "glm53flash_grid_v1"), (_pending_plan(), None)):
        campaign = tmp_path / "fpm-artifacts" / plan.sha256[:16]
        runner._atomic_json(campaign / "collection-plan.json", plan.to_dict())
        loaded = load_repeatability_source(campaign)
        assert loaded.sha256 == plan.sha256
        assert loaded.options.benchmark_points_origin == origin
        assert loaded.to_dict() == plan.to_dict()


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_bounded_sample_of_a_saved_glm_plan_is_an_explicit_list(tmp_path):
    from collector.fpm_forward.repeatability import _subset_plan, load_repeatability_source

    points = {
        "schema_version": 3,
        "prefill": [{"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0}],
        "decode": [],
    }

    def samples(plan):
        campaign = tmp_path / "fpm-artifacts" / plan.sha256[:16]
        runner._atomic_json(campaign / "collection-plan.json", plan.to_dict())
        source = load_repeatability_source(campaign)
        prefill = next(cell for cell in source.cells if cell.workload_kind == "prefill")
        selected = {"cell_id": prefill.cell_id, "benchmark_points": points}
        bounded = _subset_plan(source, selected)
        assert json.loads(bounded.options.benchmark_points_json) == points
        # The source options with only the list replaced.
        listed = replace(
            source.options,
            benchmark_points_json=bounded.options.benchmark_points_json,
            benchmark_points_sha256=bounded.options.benchmark_points_sha256,
        )
        return source, listed, bounded, _subset_plan(source, selected, comparison_mode="full_grid")

    # A bounded sample measures points that the generator did not choose, so its list is explicit. A full-grid
    # sample measures the generated list itself.
    source, listed, bounded, full = samples(_glm_plan(tmp_path))
    assert source.options.benchmark_points_origin == "glm53flash_grid_v1"
    # Nor does it keep the drop counts of the generated list.
    assert bounded.options == replace(listed, benchmark_points_origin=None, benchmark_points_dropped_json=None)
    assert bounded.to_dict()["point_generation"]["source"] == "frozen_explicit_manifest"
    assert "dropped_points" not in bounded.to_dict()["point_generation"]
    assert full.options == source.options
    # A sample of a plan without an origin changes only its list, as before.
    _source, listed, bounded, _full = samples(_pending_plan())
    assert bounded.options == listed


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_glm_collection_cells_render_the_producer_launch(tmp_path):
    from collector.fpm_forward.runtime_probe import RuntimeProbePlan

    plan = _glm_plan(tmp_path)
    assert (plan.options.warmup_iterations, plan.options.vllm_max_model_len) == (0, 1_048_576)
    assert getattr(plan, "probe_sampling", None) is None
    producer_env = {
        "DYN_FPM_GLM53FLASH_REAL_KV": "1",
        "DYN_FPM_GLM53FLASH_PREFIX_SEED": "1",
        "DYN_FPM_BENCH_COLLECT_IMBALANCED": "1",
        # The plan has no probe launch, so the revision is the one of the accepted profile.
        "DYN_FPM_TOKENIZER_REVISION": _GLM_REVISION,
        # The deepest measured context, which a 1,048,576-token runtime context holds with its seven extra positions.
        "DYN_FPM_GLM53FLASH_MEASURED_CONTEXT": "131072",
        "PYTHONPATH": "/tmp/fpm-bench:/opt/glm53flash-dynamo",
    }

    def launch(plan, cell, smoke=False):
        overrides = runner._cell_generator_overrides(plan, cell, {}, smoke=smoke)
        env = {item["name"]: item["value"] for item in overrides["K8sConfig"]["extra_env"]}
        return overrides["params"]["agg"]["extra_cli_args"], env

    for cell in plan.cells:
        for smoke in (False, True):
            args, env = launch(plan, cell, smoke)
            assert {"--language-model-only", "--cudagraph-metrics"} <= set(args)
            # The producer requires synchronous scheduling. Prefill has the flag already, so each phase has it once.
            assert args.count("--no-async-scheduling") == 1
            assert args[args.index("--benchmark-warmup-iterations") + 1] == "0"
            assert args[args.index("--benchmark-points-file") + 1] == "/tmp/fpm-bench/benchmark-points.json"
            # The native hash block stays the prefix-match unit of the producer.
            assert not any(argument.startswith("--prefix-match-unit") for argument in args)
            assert "DYN_FPM_GLM53FLASH_PREFIX_MATCH_UNIT" not in env
            assert {name: env.get(name) for name in producer_env} == producer_env
            # vLLM loads the revision whose configuration and tokenizer the producer reads. Without the flag it loads
            # the default branch of a Hub ID.
            assert args.count("--revision") == 1
            assert args[args.index("--revision") + 1] == producer_env["DYN_FPM_TOKENIZER_REVISION"]

    with pytest.raises(ValueError) as error:
        launch(replace(plan, options=replace(plan.options, warmup_iterations=5)), plan.cells[0])
    assert str(error.value) == (
        "GLM-5.3-Flash collection requires warmup_iterations 0. The producer runs its own warm-ups"
    )
    listless = replace(
        plan.options, benchmark_points_json=None, benchmark_points_sha256=None, benchmark_points_origin=None
    )
    with pytest.raises(ValueError) as error:
        launch(replace(plan, options=listless), plan.cells[0])
    assert str(error.value) == (
        "GLM-5.3-Flash collection requires a frozen point list; plan with the accepted runtime probe"
    )

    # A runtime probe of the same deployment measures the generic grid, with any global warm-up.
    probe = RuntimeProbePlan(
        "gb300-tp4",
        json.dumps({"identity": {"model": plan.model_path, "model_revision": "a" * 40}}),
        replace(listless, warmup_iterations=5),
        plan.capability,
        plan.cells,
        SimpleNamespace(manifest={"runtime": {"version": "0.33.0"}}),
        plan.sha256,
    )
    for cell in probe.cells:
        args, env = launch(probe, cell)
        assert "--language-model-only" in args
        assert {"--benchmark-points-file", "--cudagraph-metrics"}.isdisjoint(args)
        assert set(producer_env).isdisjoint(env)


def _probe_backed_glm_plan(tmp_path, *, tp=4, precision=None, launch_revision=None):
    """The GLM-5.3-Flash collection plan of guided collection, bound to the accepted runtime probe.

    The planner gets the inputs of the onboarding collector command: the case plan of the Hub ID, whose packaged
    configuration is FP8 with 128 x 128 weight blocks, the limits of the request, a stand-in vLLM 0.33.0
    instrumentation bundle, the pure-TP probe launch on GB300 (TP 4 unless ``tp`` says otherwise) with a stand-in
    Kubernetes deployment, and the profile that the probe import accepts, with the cache resources that the probe
    measured at TP 4. Another ``tp`` keeps those resources. ``precision`` replaces GEMM, MoE, FMHA and KV-cache labels
    of the launch and the profile, which are the labels of that checkpoint by default. ``launch_revision`` replaces
    the model revision of the launch only. Return the plan and the deployment inputs that it was planned with.
    """
    from collector.model_cases import build_collection_case_plan

    from .test_fpm_glm53flash_prefix_seed import LOADED_REVISION
    from .test_runtime_instrumentation import _manifest
    from .test_runtime_observations import glm_tp4_cache_groups

    model, system = "zai-org/GLM-5.3-Flash", "gb300"
    manifest, declaration = _manifest(tmp_path)
    declaration["runtime"]["version"] = "0.33.0"
    manifest.write_text(json.dumps(declaration))
    config = tmp_path / "config.json"
    config.write_bytes((_MODEL_CONFIGS / "zai-org--GLM-5.3-Flash_config.json").read_bytes())
    topology = {"tp": tp, "pp": 1, "dp": 1, "moe_tp": tp, "moe_ep": 1, "cp": 1}
    labels = {
        "gemm_quant_mode": "fp8_block",
        "moe_quant_mode": "fp8_block",
        "fmha_quant_mode": "fp8",
        "kvcache_quant_mode": "fp8",
        **(precision or {}),
    }
    precision = {
        **{name: labels[name] for name in ("gemm_quant_mode", "moe_quant_mode", "fmha_quant_mode")},
        "comm_quant_mode": "half",
    }
    profile = {
        "schema_version": 1,
        "model": model,
        "model_revision": LOADED_REVISION,
        "architecture": "Glm5NextForConditionalGeneration",
        "context_length": 1_048_576,
        "num_experts": 288,
        "provenance": "Collector test declaration; not a measured memory qualification.",
        "deployments": [
            {
                "system": system,
                "backend": "vllm",
                "backend_version": "0.33.0",
                **{name: topology[name] for name in ("tp", "dp", "moe_tp", "moe_ep")},
                **precision,
                "kv_cache_dtype": labels["kvcache_quant_mode"],
                "resources": {
                    "runtime_memory": {
                        "kv_cache_bytes": 185_126_092_800,
                        "gpu_memory_utilization": 0.9,
                        "max_model_len": 1_048_576,
                        "provenance": "Runtime probe capacity of GLM-5.3-Flash at TP 4, declared for this test.",
                        "enable_prefix_caching": True,
                        "mamba_cache_mode": "align",
                        "hash_block_size_tokens": 2176,
                    },
                    "cache_layout": "grouped",
                    "cache_groups": glm_tp4_cache_groups(),
                    "max_num_tokens": 8192,
                    "max_batch_size": 256,
                    "provenance": "Runtime probe resources of GLM-5.3-Flash at TP 4, declared for this test.",
                },
            }
        ],
    }
    image = "example/vllm-runtime@sha256:" + "a" * 64
    launch = {
        "identity": {
            "model": model,
            "model_revision": launch_revision or LOADED_REVISION,
            "model_kind": "moe",
            "framework": "vllm",
            "framework_version": "0.33.0",
            "runtime_framework_version": "0.33.0",
            "gpu": system,
            "interconnect": "nvlink",
        },
        "topology": topology,
        "precision": {**precision, "kvcache_quant_mode": labels["kvcache_quant_mode"]},
        "collection": {
            "max_model_len": 1_048_576,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 256,
            "gpu_memory_utilization": 0.9,
            "prefill_cudagraph_policy": "runtime",
            "warmup_iterations": 0,
        },
        "model_config": {"path": str(config), "sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
        "deployment": {"executor": "kubernetes", "image": image},
    }
    argv = [
        "--model-path",
        model,
        "--gpu",
        system,
        "--fpm-max-gpus",
        str(tp),
        "--fpm-gpu-counts",
        str(tp),
        "--fpm-parallel-presets",
        "pure_tp",
        "--fpm-max-model-len",
        "1048576",
        "--fpm-max-num-batched-tokens",
        "8192",
        "--fpm-max-num-seqs",
        "256",
        "--fpm-max-prefill-isl",
        "8192",
        "--fpm-max-prefill-batch-size",
        "256",
        "--fpm-prefill-cudagraph-policy",
        "runtime",
        "--fpm-gpu-memory-utilization",
        "0.9",
        "--fpm-warmup-iterations",
        "0",
        "--fpm-weight-quantizations",
        labels["gemm_quant_mode"],
        "--fpm-kv-cache-dtypes",
        labels["kvcache_quant_mode"],
    ]
    overrides = {"K8sConfig": {"k8s_image": image}}
    case_plan = build_collection_case_plan(backend="vllm", model_path=model, gpu_type=system)
    plan = planner.build_collection_plan(
        backend="vllm",
        model_path=model,
        system=system,
        selected_ops=set(case_plan.selected_ops),
        options=FPMCollectionOptions.from_args(cli._parser().parse_args(argv)),
        model_architecture=case_plan.model_architecture,
        has_model_cases=bool(case_plan.model_cases_paths),
        fpm_profile=profile,
        collector_config={"runtime_backend_version": "0.33.0", "aic_database_version": "0.33.0"},
        generator_overrides=overrides,
        runtime_instrumentation=str(manifest),
        runtime_launch=launch,
        runtime_configuration=f"gb300-tp{tp}",
    )
    return plan, overrides


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_probe_backed_glm_collection_renders_the_producer_launch(tmp_path):
    plan, _overrides = _probe_backed_glm_plan(tmp_path)
    revision = plan.runtime_launch["identity"]["model_revision"]
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert (plan.runtime_backend_version, plan.options.benchmark_points_origin) == ("0.33.0", "glm53flash_grid_v1")
    adapter = Path(runner.__file__).parent / "runtime/glm53flash/runtime-paths.json"
    producer_env = {
        "DYN_FPM_GLM53FLASH_REAL_KV": "1",
        "DYN_FPM_GLM53FLASH_PREFIX_SEED": "1",
        "DYN_FPM_BENCH_COLLECT_IMBALANCED": "1",
        "DYN_FPM_GLM53FLASH_MEASURED_CONTEXT": "131072",
        # The producer reads the tokenizer and configuration of the revision that the engine loads.
        "DYN_FPM_TOKENIZER_REVISION": revision,
        "PYTHONPATH": "/tmp/fpm-bench:" + json.loads(adapter.read_text())["python_path"],
        "AISIMULATE_RUNTIME_CONTEXT": "/tmp/fpm-bench/runtime-probe-context.json",
        "AISIMULATE_RUNTIME_INSTRUMENTATION": "/tmp/fpm-bench/runtime-instrumentation/manifest.json",
        "AISIMULATE_RUNTIME_OBSERVATION_DIR": "/results",
    }
    for cell in plan.cells:
        for smoke in (False, True):
            overrides = runner._cell_generator_overrides(plan, cell, {}, smoke=smoke)
            args = overrides["params"]["agg"]["extra_cli_args"]
            env = {item["name"]: item["value"] for item in overrides["K8sConfig"]["extra_env"]}
            assert args.count("--revision") == 1
            assert args[args.index("--revision") + 1] == revision
            # Formal collection runs the classes of the probe's instrumentation, not the vLLM 0.27.0 memory observer.
            assert args.count("--worker-cls") == args.count("--scheduler-cls") == 1
            assert args[args.index("--worker-cls") + 1] == "fpm_runtime_instrumentation.ObservedWorker"
            assert args[args.index("--scheduler-cls") + 1] == (
                "fpm_runtime_instrumentation.ObservedInstrumentedScheduler"
            )
            assert not any(argument.startswith(("fpm_memory_worker.", "fpm_memory_scheduler.")) for argument in args)
            # Memory observation and the producer both require synchronous scheduling, and each phase has it once.
            assert args.count("--no-async-scheduling") == 1
            assert "--cudagraph-metrics" in args
            assert args[args.index("--benchmark-warmup-iterations") + 1] == "0"
            assert args[args.index("--benchmark-points-file") + 1] == "/tmp/fpm-bench/benchmark-points.json"
            assert {name: env.get(name) for name in producer_env} == producer_env


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_glm_smoke_stages_the_four_point_subset(tmp_path):
    plan = _glm_plan(tmp_path)
    frozen = json.loads(plan.options.benchmark_points_json)

    def staged(name, cell, smoke):
        directory = tmp_path / name
        directory.mkdir()
        [path] = runner._stage_points_file(plan, directory, cell=cell, smoke=smoke)
        assert path == directory / runner.POINTS_FILENAME
        return path.read_bytes()

    for cell in plan.cells:
        subset = staged(f"smoke-{cell.workload_kind}", cell, True)
        # One 64-token request without a prefix and after one hash block of prefix, and the decode of one and of two
        # requests at the smallest context of the list above one hash block. The KDA groups of the fixture profile hold
        # no prefill checkpoint block, so vLLM 0.33.0 cuts the prefill of two hash blocks after one block of prefix, and
        # the list does not hold it.
        assert json.loads(subset) == {
            "schema_version": 3,
            "prefill": [
                {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0},
                {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 2176},
            ],
            "decode": [
                {"batch_size": 1, "total_kv_read_tokens": 4096},
                {"batch_size": 2, "total_kv_read_tokens": 8192},
            ],
        }
        # Both cells of an aggregated plan stage the same four points, byte for byte.
        assert hashlib.sha256(subset).hexdigest() == "60e85d447ea36458a4d6a7f584d6709e7c02f7b1c1f306c4af18f353c93d14ce"
        # The smoke measures points of the frozen list, so smoke and formal collection share one plan hash.
        for phase in ("prefill", "decode"):
            assert all(point in frozen[phase] for point in json.loads(subset)[phase])
        # A decode cell also records the prefix caching that the producer requires.
        policy = {"decode_prefix_caching": "enabled"} if cell.workload_kind == "decode" else {}
        assert runner._configured_sampling_metadata(plan, cell, smoke=True) == {
            "benchmark_points_sha256": hashlib.sha256(subset).hexdigest(),
            "requested_point_count": {"prefill": 2, "decode": 2}[cell.workload_kind],
            "smoke_manifest": "glm53flash_grid_v1_smoke",
            **policy,
        }
        full = staged(f"formal-{cell.workload_kind}", cell, False)
        assert full.decode() == plan.options.benchmark_points_json
        assert len(json.loads(full)["prefill"]) == 456
        assert runner._configured_sampling_metadata(plan, cell, smoke=False) == {
            "benchmark_points_sha256": plan.options.benchmark_points_sha256,
            "requested_point_count": {"prefill": 456, "decode": 81}[cell.workload_kind],
            "benchmark_points_origin": "glm53flash_grid_v1",
            **policy,
        }


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_glm_smoke_with_prefill_checkpoint_blocks_stages_the_five_point_subset(tmp_path):
    plan = _glm_plan(tmp_path, kda_checkpoint_alignment=16)
    frozen = json.loads(plan.options.benchmark_points_json)

    for cell in plan.cells:
        directory = tmp_path / f"smoke-{cell.workload_kind}"
        directory.mkdir()
        [path] = runner._stage_points_file(plan, directory, cell=cell, smoke=True)
        subset = path.read_bytes()
        # With an internal checkpoint, the list holds the prefill of two hash blocks after one block of prefix, so the
        # smoke takes it as its third prefill point.
        assert json.loads(subset) == {
            "schema_version": 3,
            "prefill": [
                {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0},
                {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 2176},
                {"batch_size": 1, "total_prefill_tokens": 4352, "total_kv_read_tokens": 2176},
            ],
            "decode": [
                {"batch_size": 1, "total_kv_read_tokens": 4096},
                {"batch_size": 2, "total_kv_read_tokens": 8192},
            ],
        }
        assert hashlib.sha256(subset).hexdigest() == "35060cd4787dc7cb14fb00480359fa5c63a907bd514692e0397f839315c018fd"
        for phase in ("prefill", "decode"):
            assert all(point in frozen[phase] for point in json.loads(subset)[phase])
        policy = {"decode_prefix_caching": "enabled"} if cell.workload_kind == "decode" else {}
        assert runner._configured_sampling_metadata(plan, cell, smoke=True) == {
            "benchmark_points_sha256": hashlib.sha256(subset).hexdigest(),
            "requested_point_count": {"prefill": 3, "decode": 2}[cell.workload_kind],
            "smoke_manifest": "glm53flash_grid_v1_smoke",
            **policy,
        }


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_glm_smoke_of_a_one_sequence_request_stages_points_of_one_request(tmp_path):
    # A request with max_batch_size 1 reaches the collector as --fpm-max-num-seqs 1 and --fpm-max-prefill-batch-size 1.
    plan = _glm_plan(tmp_path, max_batch_size=1)
    assert (plan.options.max_num_seqs, plan.options.max_prefill_batch_size) == (1, 1)
    frozen = json.loads(plan.options.benchmark_points_json)
    assert {point["batch_size"] for phase in ("prefill", "decode") for point in frozen[phase]} == {1}

    for cell in plan.cells:
        directory = tmp_path / f"smoke-{cell.workload_kind}"
        directory.mkdir()
        [path] = runner._stage_points_file(plan, directory, cell=cell, smoke=True)
        subset = path.read_bytes()
        # No decode point of two requests exists, so the decode points are one request at two contexts.
        assert json.loads(subset) == {
            "schema_version": 3,
            "prefill": [
                {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0},
                {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 2176},
            ],
            "decode": [
                {"batch_size": 1, "total_kv_read_tokens": 4096},
                {"batch_size": 1, "total_kv_read_tokens": 8192},
            ],
        }
        for phase in ("prefill", "decode"):
            assert all(point in frozen[phase] for point in json.loads(subset)[phase])
        policy = {"decode_prefix_caching": "enabled"} if cell.workload_kind == "decode" else {}
        assert runner._configured_sampling_metadata(plan, cell, smoke=True) == {
            "benchmark_points_sha256": hashlib.sha256(subset).hexdigest(),
            "requested_point_count": {"prefill": 2, "decode": 2}[cell.workload_kind],
            "smoke_manifest": "glm53flash_grid_v1_smoke",
            **policy,
        }
        # Formal collection still stages the whole list under the same plan hash.
        formal_directory = tmp_path / f"formal-{cell.workload_kind}"
        formal_directory.mkdir()
        [formal] = runner._stage_points_file(plan, formal_directory, cell=cell, smoke=False)
        assert formal.read_bytes().decode() == plan.options.benchmark_points_json


_FOUR_POINTS_TEXT = "GLM-5.3-Flash smoke subset requires the generated list to hold its four points"


def _glm_short_request_plan(tmp_path, worker_type):
    """The plan of a GLM-5.3-Flash request limited to one sequence, 4,096 tokens and a 4,096-token context."""
    return _glm_plan(tmp_path, max_batch_size=1, max_num_tokens=4096, worker_type=worker_type, vllm_max_model_len=4096)


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_glm_smoke_of_a_prefill_role_stages_its_two_prefill_points(tmp_path):
    plan = _glm_short_request_plan(tmp_path, "prefill")
    frozen = json.loads(plan.options.benchmark_points_json)
    # The measured context is 4,089 tokens. The list holds both smoke prefill points, but only one decode context above
    # the hash block of 2,176 tokens, so it has no second decode point for the smoke to take.
    assert {
        point["total_kv_read_tokens"] // point["batch_size"]
        for point in frozen["decode"]
        if point["total_kv_read_tokens"] // point["batch_size"] > 2176
    } == {4089}
    [cell] = plan.cells
    assert cell.workload_kind == "prefill"

    # The role never measures a decode point, so the smoke stages the two prefill points and no decode point.
    directory = tmp_path / "smoke"
    directory.mkdir()
    [path] = runner._stage_points_file(plan, directory, cell=cell, smoke=True)
    subset = path.read_bytes()
    assert json.loads(subset) == {
        "schema_version": 3,
        "prefill": [
            {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0},
            {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 2176},
        ],
        "decode": [],
    }
    assert all(point in frozen["prefill"] for point in json.loads(subset)["prefill"])
    assert runner._configured_sampling_metadata(plan, cell, smoke=True) == {
        "benchmark_points_sha256": hashlib.sha256(subset).hexdigest(),
        "requested_point_count": 2,
        "smoke_manifest": "glm53flash_grid_v1_smoke",
    }
    overrides = runner._cell_generator_overrides(plan, cell, {}, smoke=True)
    assert "--benchmark-points-file" in overrides["params"]["agg"]["extra_cli_args"]


@pytest.mark.usefixtures("no_models_or_timing_data")
@pytest.mark.parametrize("worker_type", ["aggregated", None])
def test_glm_smoke_of_an_aggregated_request_still_needs_both_phases(tmp_path, worker_type):
    # The same limits as the prefill role above, collected for both phases, leave the decode phase one point short.
    plan = _glm_short_request_plan(tmp_path, worker_type)
    assert {cell.workload_kind for cell in plan.cells} == {"prefill", "decode"}
    for cell in plan.cells:
        directory = tmp_path / f"smoke-{cell.workload_kind}"
        directory.mkdir()
        with pytest.raises(ValueError) as error:
            runner._stage_points_file(plan, directory, cell=cell, smoke=True)
        assert str(error.value) == _FOUR_POINTS_TEXT


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_glm_smoke_of_a_decode_role_stages_its_two_decode_points(tmp_path):
    # A token budget of 32 holds no prefill point of 64 new tokens, so the list lacks the prefill smoke points.
    limits = {"max_batch_size": 8, "max_num_tokens": 32}
    plan = _glm_plan(tmp_path, worker_type="decode", **limits)
    frozen = json.loads(plan.options.benchmark_points_json)
    assert not any(point["total_prefill_tokens"] == 64 for point in frozen["prefill"])
    [cell] = plan.cells
    assert cell.workload_kind == "decode"

    directory = tmp_path / "smoke"
    directory.mkdir()
    [path] = runner._stage_points_file(plan, directory, cell=cell, smoke=True)
    subset = path.read_bytes()
    # The budget of 32 tokens cannot prefill the suffixes of two requests above one hash block in one step, so the list
    # has one request there, and the smoke takes one request at each of the two smallest contexts above the block.
    assert json.loads(subset) == {
        "schema_version": 3,
        "prefill": [],
        "decode": [
            {"batch_size": 1, "total_kv_read_tokens": 4096},
            {"batch_size": 1, "total_kv_read_tokens": 8192},
        ],
    }
    assert all(point in frozen["decode"] for point in json.loads(subset)["decode"])
    assert runner._configured_sampling_metadata(plan, cell, smoke=True) == {
        "benchmark_points_sha256": hashlib.sha256(subset).hexdigest(),
        "requested_point_count": 2,
        "smoke_manifest": "glm53flash_grid_v1_smoke",
        "decode_prefix_caching": "enabled",
    }

    # An aggregated plan of the same limits measures the prefill phase too, and the list cannot serve it.
    aggregated = _glm_plan(tmp_path, worker_type="aggregated", **limits)
    for cell in aggregated.cells:
        with pytest.raises(ValueError) as error:
            runner._stage_points_file(aggregated, directory, cell=cell, smoke=True)
        assert str(error.value) == _FOUR_POINTS_TEXT


def test_glm_collection_without_a_model_revision_is_refused(tmp_path):
    architecture = "Glm5NextForConditionalGeneration"
    path = _mixed_attention_config(tmp_path, architecture, True)
    points = tmp_path / "points.json"
    points.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "prefill": [{"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0}],
                "decode": [{"batch_size": 1, "total_kv_read_tokens": 4096}],
            }
        )
    )
    # Direct collector use with a user point list, and with neither an accepted probe launch nor an FPM profile.
    with pytest.raises(ValueError) as error:
        planner.build_collection_plan(
            backend="vllm",
            model_path=str(path),
            system="b200_sxm",
            selected_ops={"attention_context", "attention_generation"},
            options=FPMCollectionOptions.from_args(
                _args(fpm_benchmark_points_file=str(points), fpm_warmup_iterations=0)
            ),
            model_architecture=architecture,
            has_model_cases=False,
            model_config_path=str(path),
            collector_config={"aic_database_version": "0.27.0"},
        )
    assert str(error.value) == (
        "GLM-5.3-Flash collection requires a model revision of 40 lowercase hex characters, got None"
    )
    # The runner keeps its own refusal for a plan without either, such as a plan saved before planning refused it.
    plan = replace(_glm_plan(tmp_path), _fpm_profile_json=None)
    assert (plan.runtime_launch, plan.fpm_profile) == (None, None)
    for cell in plan.cells:
        with pytest.raises(ValueError) as error:
            runner._cell_generator_overrides(plan, cell, {})
        assert str(error.value) == (
            "GLM-5.3-Flash collection requires a probe launch or an FPM profile that names the model revision"
        )


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

    from aisim_core.sdk.fpm_identity import EXECUTION_COLUMNS, LEGACY_EXECUTION_IDENTITY

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


@pytest.mark.parametrize(
    "template_id", [None, GENERIC_RUNTIME_TEMPLATE_ID], ids=["template-of-the-plan", "generic-runtime-template"]
)
@pytest.mark.usefixtures("no_models_or_timing_data")
def test_non_glm_plans_keep_the_approved_moe_tp_skip(tmp_path, template_id):
    from collector.fpm_forward import database

    plan = _pending_plan()
    if template_id is not None:
        plan = replace(plan, capability=replace(plan.capability, template_id=template_id))
    # A warm-up skip is refused for the strategies that require the warm-up, so the cell takes the tp strategy.
    decode = next(cell for cell in plan.cells if cell.workload_kind == "decode" and cell.topology.dp == 1)
    cell = replace(decode, parallel_strategy="tp")
    assert not cell.execution_identity[0]
    unit = tmp_path / "raw/node0000"
    unit.mkdir(parents=True)
    provenance_path = unit / "collector-provenance.json"
    _write_provenance(provenance_path, cell_id=cell.cell_id, plan_sha256=plan.sha256, attempt_id="attempt")
    provenance = json.loads(provenance_path.read_text())
    provenance["runtime"]["backend_version"] = plan.runtime_backend_version
    provenance_path.write_text(json.dumps(provenance))
    payload = _native_payload(phase="decode", rank=0, dp=1)
    payload["kvwarm"] = {"enabled": True, "warm_eligible": False, "skip_reason": "moe_tp_balanced_by_construction"}
    (unit / "benchmark.json").write_text(json.dumps(payload))
    rows = database.aggregate_cell(plan, cell, tmp_path, expected_attempt_id="attempt")
    assert [row["kv_seed_regime"] for row in rows] == ["skip:moe_tp_balanced_by_construction"]


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
