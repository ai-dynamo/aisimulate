# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from collector.fpm_forward.bundled_instrumentation import bundled_instrumentation
from collector.fpm_forward.runtime_instrumentation import freeze_instrumentation, load_instrumentation

pytestmark = pytest.mark.unit


def test_bundled_adapter_is_self_contained_and_unknown_version_needs_local_bundle(tmp_path):
    bundle = bundled_instrumentation("0.27.0")
    assert bundle.manifest["runtime"]["version"] == "0.27.0"
    assert bundle.manifest["runtime"]["source_files"]
    assert bundle.manifest["source_notes"] in bundle.files
    frozen = freeze_instrumentation(bundle, tmp_path / "bundle")
    assert load_instrumentation(frozen.manifest_path).sha256 == bundle.sha256
    assert bundled_instrumentation("0.28.0") is None


def test_bundled_0_31_0_adapter_is_self_contained(tmp_path):
    from collector.fpm_forward.runtime_instrumentation import relative_path

    bundle = bundled_instrumentation("0.31.0")
    manifest = bundle.manifest
    assert manifest["runtime"]["version"] == "0.31.0"
    assert manifest["source_notes"] == "instrumentation/vllm-0.31.0.md"
    assert manifest["source_notes"] in bundle.files
    assert all(relative_path(name) == name for name in manifest["runtime"]["source_files"])
    frozen = freeze_instrumentation(bundle, tmp_path / "bundle")
    assert load_instrumentation(frozen.manifest_path).sha256 == bundle.sha256
    assert bundled_instrumentation("0.28.0") is None


@pytest.mark.parametrize("version", ["0.27.0", "0.28.0", "0.31.0"])
def test_frozen_scheduler_export_preserves_native_benchmark_contract(tmp_path, monkeypatch, version):
    import importlib

    _fake_campaign(tmp_path, monkeypatch, version=version)
    frozen = load_instrumentation(tmp_path / "instrumentation/manifest.json")
    scheduler_name = frozen.manifest["scheduler_class"]
    # Every pinned Dynamo args.py consumer requires this substring before importing
    # the class. The immutable source mappings are in each bundle's source notes.
    assert "InstrumentedScheduler" in scheduler_name
    module, name = scheduler_name.rsplit(".", 1)
    scheduler = getattr(importlib.import_module(module), name)
    native = importlib.import_module("dynamo.vllm.instrumented_scheduler").InstrumentedScheduler
    assert scheduler.__name__ == name
    assert issubclass(scheduler, native)


def test_new_bundle_hooks_emit_observation_after_normal_warmup(tmp_path, monkeypatch):
    import importlib
    import json
    import sys
    from types import ModuleType, SimpleNamespace

    from .test_fpm_runtime_memory import _initialized

    bundle = freeze_instrumentation(bundled_instrumentation("0.27.0"), tmp_path / "bundle")
    monkeypatch.syspath_prepend(str(bundle.root))
    modules = {}
    for name in (
        "vllm",
        "vllm.distributed",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu_worker",
        "dynamo",
        "dynamo.vllm",
        "dynamo.vllm.instrumented_scheduler",
    ):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    modules["dynamo.vllm.instrumented_scheduler"].InstrumentedScheduler = object
    modules["vllm.distributed"].get_tp_group = lambda: SimpleNamespace(rank_in_group=0)
    modules["vllm.distributed"].get_pp_group = lambda: SimpleNamespace(rank_in_group=0)
    owner, _, cache = _initialized()
    owner.vllm_config.parallel_config.data_parallel_rank = 0

    class RuntimeWorker:
        def __init__(self):
            self.__dict__.update(owner.__dict__)

        def determine_available_memory(self):
            return 123456

        def initialize_from_config(self, config):
            assert config is cache
            return "cache initialized"

        def compile_or_warm_up_model(self):
            return "normal warmup returned"

    modules["vllm.v1.worker.gpu_worker"].Worker = RuntimeWorker
    output = tmp_path / "results"
    output.mkdir()
    monkeypatch.setenv("AISIMULATE_RUNTIME_OBSERVATION_DIR", str(output))
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            {
                "schema_version": "aisimulate-runtime-probe-launch/v1",
                "attempt_id": "attempt-1",
                "configuration": "tp2",
                "phase": "prefill",
                "bundle_sha256": bundle.sha256,
                "launch": {"identity": {"gpu": "gb300"}, "deployment": {"image": "fake"}},
            }
        )
    )
    monkeypatch.setenv("AISIMULATE_RUNTIME_CONTEXT", str(context))
    monkeypatch.setenv("AISIMULATE_RUNTIME_INSTRUMENTATION", str(bundle.manifest_path))
    module, cls = bundle.manifest["worker_class"].rsplit(".", 1)
    worker = getattr(importlib.import_module(module), cls)()
    assert worker.determine_available_memory() == 123456
    assert worker.initialize_from_config(cache) == "cache initialized"
    assert worker.compile_or_warm_up_model() == "normal warmup returned"
    records = [json.loads(path.read_text()) for path in output.glob("runtime-observation-*.json")]
    assert len(records) == 1
    assert records[0]["lifecycle"] == {
        "measurement": "after_warmup",
        "cache_initialized": True,
        "warmup_completed": True,
        "capture_completed": True,
    }
    assert records[0]["cache"]["available_cache_bytes"] == 123456
    assert records[0]["unresolved_fields"]  # Deliberately incomplete source/model/hardware fake.


@pytest.fixture(autouse=True)
def isolated_runtime_imports():
    import sys

    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "fpm_memory_observer" or name == "instrumentation" or name.startswith("instrumentation.")
    }
    for name in saved:
        del sys.modules[name]
    yield
    for name in list(sys.modules):
        if name == "fpm_memory_observer" or name == "instrumentation" or name.startswith("instrumentation."):
            del sys.modules[name]
    sys.modules.update(saved)


def _v031_runtime(storage, layout, blocks):
    """vLLM 0.31.0 cache objects: one backing storage and block-first layer views.

    ``glm_tp4`` holds the six GLM-5.3-Flash groups at TP 4 with an fp8 cache and no
    speculation (kv_cache_utils.py:1213-1292): G0 the 11 MLA latent and 11 indexer
    layers, G1 the 11 tail layers and G2 to G5 the 34 KDA layers. KDA layer i of each
    Mamba group shares the region of MLA layer i, and tail layer i the region of
    indexer layer i (:1725-1755). The specs keep the static_forward_context order:
    a GLM layer builds its indexer caches before its MLA layer
    (glm5next/common/attention.py:546, :581). Returns the worker and scheduler
    cache configs, the forward-context layers, the attention groups and the kernel
    block sizes.
    """
    from enum import Enum
    from types import SimpleNamespace

    from .test_fpm_runtime_memory import _typed

    group_role = Enum("KVCacheGroupRole", {"DEFAULT": "default", "HISPARSE_SOURCE": "hisparse_source"}, type=str)
    cache_role = Enum("SparseCacheRole", {"SPARSE": "sparse", "INDEXER": "indexer"}, type=str)

    class View:
        """A typed view of the one backing storage; offset and strides in bytes."""

        def __init__(self, shape, stride_bytes, element, offset):
            self.shape, self.stride_bytes, self.element, self.offset = list(shape), stride_bytes, element, offset

        def untyped_storage(self):
            return storage

        def element_size(self):
            return self.element

        def storage_offset(self):
            return self.offset // self.element

        def stride(self):
            return tuple(value // self.element for value in self.stride_bytes)

    def spec(name, **fields):
        value = _typed("vllm.v1.kv_cache_interface", name)
        vars(value).update(fields)
        return value

    def uniform_spec(**fields):
        # UniformTypeKVCacheSpecs (kv_cache_interface.py:1233-1333) overrides none of the three
        # abstract KVCacheSpec properties at :181-191, which raise NotImplementedError.
        def abstract(_self):
            raise NotImplementedError

        properties = dict.fromkeys(("num_heads", "tokens_per_state", "state_content_size_bytes"), property(abstract))
        value = type("UniformTypeKVCacheSpecs", (), {"__module__": "vllm.v1.kv_cache_interface", **properties})()
        vars(value).update(fields)
        return value

    def group(names, kv_cache_spec):
        return SimpleNamespace(
            layer_names=names,
            kv_cache_spec=kv_cache_spec,
            is_eagle_group=False,
            host_resident=False,
            enable_kv_transfer=True,
            role=group_role.DEFAULT,
        )

    def tensor(name, page, offset):
        # Layer-outer: each layer region holds its pages of every block.
        return SimpleNamespace(
            size=storage.size,
            layers=[name],
            layer_stride=page * blocks,
            block_stride=page,
            offset=offset,
            host_resident=False,
        )

    def layer(module, cls, kv_cache):
        value = _typed(module, cls)
        value.kv_cache = kv_cache
        return value

    def backend(module, cls, name):
        # vLLM 0.31.0 backends have no get_kv_cache_block_dim.
        return type(cls, (), {"__module__": module, "get_name": staticmethod(lambda: name)})

    def attention(backend, names, kv_cache_spec, group_id):
        return SimpleNamespace(
            backend=backend, layer_names=names, kv_cache_spec=kv_cache_spec, kv_cache_group_id=group_id
        )

    if layout == "attention":
        full = spec(
            "FullAttentionSpec", block_size=16, page_size_bytes=4096, sliding_window=None, dtype="torch.bfloat16"
        )
        groups = [group(["layer0"], full)]
        layers = {
            "layer0": layer(
                "vllm.model_executor.layers.attention",
                "Attention",
                View([blocks, 1, 16, 128], [4096, 4096, 256, 2], 2, 0),
            )
        }
        cache = SimpleNamespace(
            num_blocks=blocks,
            kv_cache_groups=groups,
            kv_cache_tensors=[tensor("layer0", 4096, 0)],
            kv_cache_layout="LBHNC",
        )
        flashinfer = backend("vllm.v1.attention.backends.flashinfer", "FlashInferBackend", "FLASHINFER")
        return cache, cache, layers, [[attention(flashinfer, ["layer0"], full, 0)]], [16]
    mla = list(range(3, 45, 4))
    kda = [index for index in range(45) if index % 4 != 3]
    # Round robin into cdiv(34, 11) = 4 Mamba groups of 9, 9, 8 and 8 layers (kv_cache_utils.py:1276-1286).
    mamba_groups = [[f"layers.{index}.kda" for index in kda[start::4]] for start in range(4)]
    latent = spec(
        "MLAAttentionSpec",
        block_size=2176,
        page_size_bytes=1_114_112,
        sliding_window=None,
        dtype="torch.uint8",
        tokens_per_state=1,
        cache_role=cache_role.SPARSE,
        storage_block_size=None,
    )
    indexer = spec(
        "MLAAttentionSpec",
        block_size=2176,
        page_size_bytes=71_808,
        sliding_window=None,
        dtype="torch.uint8",
        tokens_per_state=4,
        cache_role=cache_role.INDEXER,
        storage_block_size=128,
    )
    tail = spec(
        "KpoolTailSpec",
        block_size=4,
        page_size_bytes=71_808,
        sliding_window=4,
        extra_retained_tokens=0,
        dtype="torch.bfloat16",
    )
    state = spec(
        "MambaSpec",
        block_size=2176,
        page_size_bytes=1_114_112,
        page_size_padded=1_114_112,
        mamba_cache_mode="align",
        num_speculative_blocks=0,
        num_prefill_checkpoint_blocks=0,
        shapes=((6144, 3), (16, 128, 128)),
        dtypes=("torch.bfloat16", "torch.float32"),
        real_page_size_bytes=1_085_440,
        tokens_per_state=-1,
    )
    attention_names = [name for index in mla for name in (f"layers.{index}.indexer", f"layers.{index}.mla")]
    tails = [f"layers.{index}.tail" for index in mla]
    uniform = {name: latent if name.endswith(".mla") else indexer for name in attention_names}
    groups = [
        group(
            attention_names,
            uniform_spec(block_size=2176, page_size_bytes=13_045_120, kv_cache_specs=uniform),
        ),
        group(
            tails,
            uniform_spec(block_size=4, page_size_bytes=789_888, kv_cache_specs=dict.fromkeys(tails, tail)),
        ),
        *(group(names, state) for names in mamba_groups),
    ]
    tensors, layers = [], {}
    for position, index in enumerate(mla):
        offset = position * 8_912_896
        tensors.append(tensor(f"layers.{index}.mla", 1_114_112, offset))
        # MLA and the indexer bind the view with H squeezed (mla_attention.py:733-735, deepseek_v2.py:652-655).
        layers[f"layers.{index}.mla"] = layer(
            "vllm.model_executor.layers.attention.mla_attention",
            "MLAAttention",
            View([272, 64, 512], [32_768, 512, 1], 1, offset),
        )
        for names in mamba_groups:
            if position < len(names):
                tensors.append(tensor(names[position], 1_114_112, offset))
                # Mamba bind_kv_cache cuts the conv and SSM states from each page (mamba/abstract.py:28-42).
                layers[names[position]] = layer(
                    "vllm.models.glm5next.common.kda",
                    "Glm5NextLinearAttention",
                    (
                        View([8, 6144, 3], [1_114_112, 6, 2], 2, offset),
                        View([8, 16, 128, 128], [1_114_112, 65_536, 512, 4], 4, offset + 36_864),
                    ),
                )
    for position, index in enumerate(mla):
        offset = 98_041_856 + position * 574_464
        for name, cls, view in (
            ("indexer", "Glm5NextIndexerCache", View([136, 32, 132], [4224, 132, 1], 1, offset)),
            ("tail", "Glm5NextTailCache", View([8, 2, 4, 128], [71_808, 1024, 256, 2], 2, offset)),
        ):
            tensors.append(tensor(f"layers.{index}.{name}", 71_808, offset))
            layers[f"layers.{index}.{name}"] = layer("vllm.models.glm5next.common.attention", cls, view)
    gdn = backend("vllm.v1.attention.backends.gdn_attn", "GDNAttentionBackend", "GDN_ATTN")
    # init_attn_backend makes the attention groups in G0 layer order (gpu/attn_utils.py:222-249).
    attn = [
        [
            attention(
                backend("vllm.v1.attention.backends.mla.indexer", "DeepseekV32IndexerBackend", "DEEPSEEK_V32_INDEXER"),
                [f"layers.{index}.indexer" for index in mla],
                indexer,
                0,
            ),
            attention(
                backend(
                    "vllm.v1.attention.backends.mla.flashinfer_mla_sparse",
                    "FlashInferMLASparseTRTLLMBackend",
                    "FLASHINFER_MLA_SPARSE",
                ),
                [f"layers.{index}.mla" for index in mla],
                latent,
                0,
            ),
        ],
        [
            attention(
                backend("vllm.v1.attention.backends.mla.indexer", "KpoolTailBackend", "KPOOL_TAIL"), tails, tail, 1
            )
        ],
        *([attention(gdn, names, state, group_id)] for group_id, names in enumerate(mamba_groups, start=2)),
    ]
    cache = SimpleNamespace(
        num_blocks=blocks, kv_cache_groups=groups, kv_cache_tensors=tensors, kv_cache_layout="LBHNC"
    )
    # The scheduler copy keeps the first inner spec of each uniform group (kv_cache_utils.py:2434-2440):
    # the first indexer for G0 and the first tail for G1.
    scheduler_cache = SimpleNamespace(
        num_blocks=blocks,
        kv_cache_groups=[group(attention_names, indexer), group(tails, tail), *groups[2:]],
        kv_cache_tensors=tensors,
        kv_cache_layout="LBHNC",
    )
    return cache, scheduler_cache, layers, attn, [64, 4, 2176, 2176, 2176, 2176]


def _fake_campaign(
    tmp_path, monkeypatch, *, packed=False, planar=False, version="0.28.0", runner_version=1, layout="attention"
):
    """Synthetic vendor files and GPU objects; verifies mapping, never GPU support.

    vLLM 0.31.0 fakes place every layer in one backing storage and view it block
    axis first; ``layout="glm_tp4"`` holds the GLM-5.3-Flash TP4 cache groups.
    """
    import hashlib
    import importlib
    import importlib.metadata
    import json
    import sys
    from enum import Enum
    from pathlib import Path
    from types import ModuleType, SimpleNamespace

    from .test_fpm_runtime_memory import _Storage, _typed, _vllm_config

    runtime = Path(__file__).parents[3] / "collector/fpm_forward/runtime"
    original = load_instrumentation(runtime / f"vllm-{version}{'.example' if version == '0.28.0' else ''}.json")
    bundle = freeze_instrumentation(original, tmp_path / "instrumentation")
    manifest = bundle.manifest
    fake_site = tmp_path / "site-packages"
    for name in manifest["runtime"]["source_files"]:
        path = fake_site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Synthetic runtime source for {name}\n")
        manifest["runtime"]["source_files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    bundle.manifest_path.write_text(json.dumps(manifest))
    bundle = load_instrumentation(bundle.manifest_path)
    model = tmp_path / "model"
    model.mkdir()
    model_config_path = model / "config.json"
    model_config_path.write_text('{"model_type":"synthetic-test-model"}')
    config = _vllm_config(tp=1, dp=1)
    config.model_config.model = str(model)
    config.model_config.revision = "d" * 40
    config.model_config.quantization = None
    config.model_config.hf_config = SimpleNamespace(_commit_hash=None)
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.enable_expert_parallel = False
    config.quant_config = None
    # GLM: 8 blocks of the 13,045,120-byte pool block, 11 * (1,114,112 + 71,808).
    blocks, page = (8, 13_045_120) if layout == "glm_tp4" else (10, 4096)
    storage = _Storage(1024, blocks * page * (2 if packed else 1))

    class Tensor:
        def __init__(self, offset=0):
            self.offset = offset
            self.shape = (2, blocks, 16, 1, 64) if planar else (blocks, 2, 16, 1, 64)

        def untyped_storage(self):
            return storage

        def element_size(self):
            return 2

        def storage_offset(self):
            return self.offset // 2

        def stride(self):
            return (blocks * 1024, 1024, 64, 64, 1) if planar else (2048 * (2 if packed else 1), 1024, 64, 64, 1)

    class Backend:
        @staticmethod
        def get_name():
            return "FLASHINFER"

        @staticmethod
        def get_kv_cache_block_dim(kernel, heads, head_size, *, cache_dtype_str):
            assert (kernel, heads, head_size, cache_dtype_str) == (16, 1, 64, "auto")
            return 1 if planar else 0

    Backend.__module__ = "vllm.v1.attention.backends.flashinfer"
    mode = Enum("KVQuantMode", ["NONE"])
    scheduler_config, scheduler_fields = config, {}
    if version == "0.31.0":
        cache, scheduler_cache, layers, attn, kernel_block_sizes = _v031_runtime(storage, layout, blocks)
        block_size, mamba_cache_mode = (2176, "align") if layout == "glm_tp4" else (16, "none")
        vars(config.cache_config).update(
            block_size=block_size,
            mamba_cache_mode=mamba_cache_mode,
            mamba_block_size=block_size if mamba_cache_mode == "align" else None,
            prefix_match_unit=None,
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
            kv_cache_size_tokens=None,
            kv_cache_max_concurrency=None,
        )
        if layout == "glm_tp4":
            # The probe launches GLM-5.3-Flash with an fp8_e4m3 cache, which its uint8 specs store.
            config.cache_config.cache_dtype = "fp8_e4m3"
        # The engine core sets the startup capacity in its own process, before it
        # builds the scheduler (engine/core.py:346-351); workers keep None.
        tokens = (blocks - 1) * block_size
        scheduler_config = SimpleNamespace(**vars(config))
        scheduler_config.cache_config = SimpleNamespace(
            **{**vars(config.cache_config), "kv_cache_size_tokens": tokens, "kv_cache_max_concurrency": tokens / 4096}
        )
        scheduler_fields = {"block_size": block_size, "hash_block_size": block_size}
    else:
        groups, layers, allocations, attn = [], {}, [], []
        for index in range(2 if packed else 1):
            name = f"layer{index}"
            spec = _typed("vllm.v1.kv_cache_interface", "FullAttentionSpec")
            spec.block_size = spec.storage_block_size = 16
            spec.page_size_bytes, spec.sliding_window, spec.dtype = page, None, "torch.bfloat16"
            spec.kv_quant_mode, spec.num_kv_heads, spec.head_size = mode.NONE, 1, 64
            layer = _typed("vllm.model_executor.layers.attention", "Attention")
            layer.kv_cache = Tensor(page * index)
            layers[name] = layer
            groups.append(SimpleNamespace(layer_names=[name], kv_cache_spec=spec, is_eagle_group=False))
            allocations.append(
                SimpleNamespace(
                    size=storage.size, shared_by=[name], block_stride=page * 2 if packed else 0, offset=page * index
                )
            )
            attn.append(
                [SimpleNamespace(layer_names=[name], kv_cache_spec=spec, kv_cache_group_id=index, backend=Backend)]
            )
        cache = scheduler_cache = SimpleNamespace(
            num_blocks=blocks, kv_cache_groups=groups, kv_cache_tensors=allocations
        )
        kernel_block_sizes = [16] * len(groups)
    config.compilation_config.static_forward_context = layers
    runner = _typed(
        "vllm.v1.worker.gpu_model_runner" if runner_version == 1 else "vllm.v1.worker.gpu.model_runner",
        "GPUModelRunner",
    )
    runner.kv_cache_config = cache
    runner.kv_caches = [layer.kv_cache for layer in layers.values()]
    runner.shared_kv_cache_layers = {}
    runner.attn_groups = attn
    # The native initialize_kv_cache methods bind different fields. V2 never
    # assigns the private V1 field, even though both classes have the same name.
    setattr(runner, "_kernel_block_sizes" if runner_version == 1 else "kernel_block_sizes", kernel_block_sizes)
    block_objects = [SimpleNamespace(block_id=index, is_null=index == 0) for index in range(blocks)]
    pool = SimpleNamespace(
        blocks=block_objects,
        num_gpu_blocks=blocks,
        null_block=block_objects[0],
        get_num_free_blocks=lambda: blocks - 1,
        free_block_queue=SimpleNamespace(get_all_free_blocks=lambda: block_objects[1:]),
        **({"hash_block_size": scheduler_fields["hash_block_size"]} if scheduler_fields else {}),
    )
    manager = SimpleNamespace(
        block_pool=pool,
        watermark_blocks=0,
        coordinator=SimpleNamespace(
            single_type_managers=[SimpleNamespace(block_pool=pool) for group in cache.kv_cache_groups]
        ),
    )

    class RuntimeWorker:
        fail = None

        def __init__(self):
            self.vllm_config, self.model_runner, self.device = config, runner, "cuda:0"

        def determine_available_memory(self):
            if self.fail == "profile":
                raise RuntimeError("runtime profile failure")
            return storage.size + page

        def initialize_from_config(self, supplied):
            assert supplied is cache
            if self.fail == "initialize":
                raise RuntimeError("runtime initialize failure")
            return "initialized"

        def compile_or_warm_up_model(self):
            if self.fail == "warmup":
                raise RuntimeError("runtime warmup failure")
            return "warmed"

    class RuntimeScheduler:
        def __init__(self, supplied_config, supplied_cache):
            assert supplied_config is scheduler_config and supplied_cache is scheduler_cache
            self.vllm_config, self.kv_cache_manager, self._fpm_dp_rank = scheduler_config, manager, 0
            vars(self).update(scheduler_fields)

    for name in (
        "vllm",
        "vllm._version",
        "vllm.distributed",
        "vllm.v1.worker.gpu_worker",
        "vllm.model_executor.offloader",
        "dynamo",
        "dynamo.vllm.instrumented_scheduler",
        "torch",
    ):
        module = ModuleType(name)
        module.__path__ = [str(fake_site / name.replace(".", "/"))]
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm._version"].__commit_id__ = "g" + manifest["runtime"]["source_revision"][:9]
    sys.modules["vllm.distributed"].get_tp_group = lambda: SimpleNamespace(rank_in_group=0)
    sys.modules["vllm.distributed"].get_pp_group = lambda: SimpleNamespace(rank_in_group=0)
    sys.modules["vllm.v1.worker.gpu_worker"].Worker = RuntimeWorker
    sys.modules["dynamo.vllm.instrumented_scheduler"].InstrumentedScheduler = RuntimeScheduler
    sys.modules["vllm.model_executor.offloader"].get_offloader = lambda: _typed(
        "vllm.model_executor.offloader.base", "NoopOffloader"
    )
    sys.modules["torch"].cuda = SimpleNamespace(
        get_device_properties=lambda _device: SimpleNamespace(
            name="NVIDIA GB300", major=10, minor=3, total_memory=1_000_000_000
        )
    )
    old_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name: version if name == "vllm" else old_version(name))
    monkeypatch.syspath_prepend(str(bundle.root))
    module_name = bundle.manifest["worker_class"].rsplit(".", 1)[0]
    adapter = importlib.import_module(module_name)
    launch = {
        "identity": {
            "model": str(model),
            "model_revision": config.model_config.revision,
            "model_kind": "dense",
            "framework": "vllm",
            "framework_version": version,
            "gpu": "gb300",
            "interconnect": "nvlink",
            "sm": 103,
        },
        "topology": {"tp": 1, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 1, "cp": 1},
        "precision": {
            "gemm_quant_mode": "bfloat16",
            "moe_quant_mode": "bfloat16",
            "fmha_quant_mode": "bfloat16",
            "kvcache_quant_mode": "fp8" if layout == "glm_tp4" else "bfloat16",
            "comm_quant_mode": "half",
            "moe_backend": "auto",
            "attention_backend": "auto",
            "enable_wideep": False,
            "enable_eplb": False,
        },
        "collection": {
            "max_model_len": 4096,
            "max_num_batched_tokens": 1024,
            "max_num_seqs": 64,
            "gpu_memory_utilization": 0.9,
            "prefill_cudagraph_policy": "runtime",
            "max_prefill_cudagraph_size": None,
            "async_scheduling": False,
        },
        "model_config": {
            "path": str(model_config_path),
            "sha256": hashlib.sha256(model_config_path.read_bytes()).hexdigest(),
        },
        "deployment": {"executor": "slurm", "image": "synthetic/runtime@sha256:" + "c" * 64},
    }
    attempt = {
        "attempt_id": "attempt-1",
        "bundle": {"manifest": str(bundle.manifest_path.relative_to(tmp_path)), "sha256": bundle.sha256},
        "phases": {},
    }
    index = {
        "schema_version": "aisimulate-runtime-observations/v1",
        "configurations": {"worker": {"launch": launch, "active_attempt_id": "attempt-1", "attempts": [attempt]}},
    }

    def run(phase="prefill"):
        directory = tmp_path / phase
        directory.mkdir()
        context = {
            "schema_version": "aisimulate-runtime-probe-launch/v1",
            "attempt_id": "attempt-1",
            "configuration": "worker",
            "phase": phase,
            "bundle_sha256": bundle.sha256,
            "launch": launch,
            "expected_ranks": {"workers": [{"dp_rank": 0, "tp_rank": 0, "pp_rank": 0}], "schedulers": [{"dp_rank": 0}]},
        }
        context_path = directory / "launch.json"
        context_path.write_text(json.dumps(context))
        monkeypatch.setenv("AISIMULATE_RUNTIME_CONTEXT", str(context_path))
        monkeypatch.setenv("AISIMULATE_RUNTIME_INSTRUMENTATION", str(bundle.manifest_path))
        monkeypatch.setenv("AISIMULATE_RUNTIME_OBSERVATION_DIR", str(directory))
        worker = adapter.ObservedWorker()
        assert worker.determine_available_memory() == storage.size + page
        assert worker.initialize_from_config(cache) == "initialized"
        assert worker.compile_or_warm_up_model() == "warmed"
        adapter.ObservedInstrumentedScheduler(scheduler_config, scheduler_cache)
        paths = sorted(directory.glob("runtime-observation-*.json"))
        reference = lambda path: {
            "path": str(path.relative_to(tmp_path)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        attempt["phases"][phase] = {
            "launch_manifest": reference(context_path),
            "artifacts": [{**reference(path), "kind": "observation"} for path in paths],
        }
        (tmp_path / "observations.json").write_text(json.dumps(index))
        return [json.loads(path.read_text()) for path in paths]

    return SimpleNamespace(
        run=run,
        launch=launch,
        config=config,
        cache=cache,
        pool=pool,
        site=fake_site,
        source=manifest,
        adapter=adapter,
        base=RuntimeWorker,
        model_config=model_config_path,
        root=tmp_path,
    )


@pytest.mark.parametrize(
    "version,packed,planar", [("0.27.0", False, False), ("0.28.0", False, True), ("0.28.0", True, False)]
)
def test_runtime_records_import_complete_for_source_audited_layouts(tmp_path, monkeypatch, version, packed, planar):
    from collector.fpm_forward.runtime_observations import validate_observations

    campaign = _fake_campaign(tmp_path, monkeypatch, version=version, packed=packed, planar=planar)
    for phase in ("prefill", "decode"):
        records = campaign.run(phase)
        assert len(records) == 2
        assert all(record["unresolved_fields"] == [] for record in records), records
        worker = next(record for record in records if record["kind"] == "worker")
        assert worker["cache"]["layer_tensors"]["layer0"]["block_axis"] == (1 if planar else 0)
        assert worker["model_config_sha256"] == campaign.launch["model_config"]["sha256"]
        # Only vLLM 0.31.0 records carry the new cache evidence.
        assert all("prefix_caching" not in record["cache"] for record in records)
        assert all("kv_cache_layout" not in record["raw_cache_config"] for record in records)
        allocation = worker["raw_cache_config"]["tensor_allocations"][0]
        assert set(allocation) == {"size", "shared_by", "block_stride", "offset"}
    result = validate_observations(tmp_path / "observations.json", {"worker": campaign.launch})["worker"]
    assert result["status"] == "complete", result["diagnostics"]


def test_bundled_lifecycle_through_verified_entrypoint_imports_both_phases(tmp_path, monkeypatch):
    import importlib
    import json
    import os
    import sys
    from pathlib import Path

    from collector.fpm_forward.runtime_instrumentation import runtime_binding
    from collector.fpm_forward.runtime_observations import validate_observations
    from collector.fpm_forward.runtime_probe import observation_artifacts

    campaign = _fake_campaign(tmp_path, monkeypatch, version="0.27.0")
    frozen = load_instrumentation(tmp_path / "instrumentation/manifest.json")
    runtime = Path(__file__).parents[3] / "collector/fpm_forward/runtime"
    monkeypatch.syspath_prepend(str(runtime))
    for phase in ("prefill", "decode"):
        # Each native phase is a fresh process. Retain the fixture's fake runtime
        # dependencies while clearing only adapters before invoking the entrypoint.
        for name in list(sys.modules):
            if name in {"fpm_memory_observer", "fpm_runtime_instrumentation", "instrumentation"} or name.startswith(
                "instrumentation."
            ):
                monkeypatch.delitem(sys.modules, name)
        original_adapter = campaign.adapter
        entrypoint = importlib.import_module("fpm_runtime_instrumentation")

        def worker():
            context_path = Path(os.environ["AISIMULATE_RUNTIME_CONTEXT"])
            context = json.loads(context_path.read_text())
            context["instrumentation_binding"] = runtime_binding()
            context_path.write_text(json.dumps(context))
            return entrypoint.ObservedWorker()

        # The fixture closes over its adapter module, so redirect those two
        # exports to the runtime entrypoint rather than bypassing native hooks.
        monkeypatch.setattr(original_adapter, "ObservedWorker", worker)
        monkeypatch.setattr(
            original_adapter,
            "ObservedInstrumentedScheduler",
            lambda *args: entrypoint.ObservedInstrumentedScheduler(*args),
        )
        try:
            records = campaign.run(phase)
            assert all(record["instrumentation_binding"]["status"] == "verified" for record in records)
            assert all(record["instrumentation_binding"]["files"] == frozen.files for record in records)
            assert issubclass(entrypoint.ObservedWorker, campaign.base)
            # Probe and formal collection use this same artifact enumerator.
            references = observation_artifacts(tmp_path, tmp_path / phase)
            assert any("runtime-instrumentation-imports-" in item["path"] for item in references)
            index_path = tmp_path / "observations.json"
            index = json.loads(index_path.read_text())
            index["configurations"]["worker"]["attempts"][0]["phases"][phase]["artifacts"] = references
            index_path.write_text(json.dumps(index))
        finally:
            if entrypoint._bundle is not None:
                sys.meta_path.remove(entrypoint._bundle)
            monkeypatch.delitem(sys.modules, "fpm_runtime_instrumentation")
    result = validate_observations(tmp_path / "observations.json", {"worker": campaign.launch})["worker"]
    assert result["status"] == "complete", result
    assert result["provenance"]["instrumentation"]["import_binding"] == "verified"


@pytest.mark.parametrize("version", ["0.27.0", "0.28.0"])
@pytest.mark.parametrize("runner_version", [1, 2])
def test_native_runner_selection_uses_its_initialized_kernel_block_sizes(
    tmp_path, monkeypatch, version, runner_version
):
    from collector.fpm_forward.runtime_observations import validate_observations

    campaign = _fake_campaign(tmp_path, monkeypatch, version=version, runner_version=runner_version)
    runner = campaign.adapter.ObservedWorker().model_runner
    other_field = "kernel_block_sizes" if runner_version == 1 else "_kernel_block_sizes"
    assert not hasattr(runner, other_field)
    for phase in ("prefill", "decode"):
        records = campaign.run(phase)
        assert all(record["unresolved_fields"] == [] for record in records), records
        worker = next(record for record in records if record["kind"] == "worker")
        assert worker["cache"]["layer_tensors"]["layer0"]["kernel_block_size_tokens"] == 16
        assert worker["cache"]["model_runner"]["class"] == f"{type(runner).__module__}.GPUModelRunner"
    result = validate_observations(tmp_path / "observations.json", {"worker": campaign.launch})["worker"]
    assert result["status"] == "complete", result["diagnostics"]


@pytest.mark.parametrize("failure", ["unknown_runner", "unaudited_source", "missing_field"])
def test_runner_mapping_never_guesses_a_compatible_field(tmp_path, monkeypatch, failure):
    import importlib

    campaign = _fake_campaign(tmp_path, monkeypatch, runner_version=2)
    runner = campaign.adapter.ObservedWorker().model_runner
    if failure == "unknown_runner":
        type(runner).__module__ = "vendor.unreviewed_runner"
    elif failure == "unaudited_source":
        campaign.source["runtime"]["source_files"].pop("vllm/v1/worker/gpu/model_runner.py", None)
    else:
        del runner.kernel_block_sizes
        runner._kernel_block_sizes = [16]
    observer = importlib.import_module("instrumentation.observer")
    worker = campaign.adapter.ObservedWorker()
    worker.determine_available_memory()
    worker.initialize_from_config(campaign.cache)
    with pytest.raises((ValueError, AttributeError)):
        observer._layer_views(
            worker,
            observer.legacy.worker_memory(worker),
            campaign.source["runtime"]["source_files"],
            "0.28.0",
        )


def test_changed_vendor_source_is_unresolved_with_raw_measurements_retained(tmp_path, monkeypatch):
    import hashlib

    campaign = _fake_campaign(tmp_path, monkeypatch)
    source = campaign.site / "vllm/v1/worker/gpu_model_runner.py"
    source.write_text("# Synthetic unreviewed vendor patch\n")
    for record in campaign.run():
        assert "runtime" in record["unresolved_fields"]
        assert "source hash differs" in record["error"]
        assert (
            record["runtime"]["source_files"]["vllm/v1/worker/gpu_model_runner.py"]
            == hashlib.sha256(source.read_bytes()).hexdigest()
        )
        assert record["cache"]["num_blocks"] == 10


@pytest.mark.parametrize("extra", [None, 8])
def test_v028_extra_retention_is_not_silently_discarded(tmp_path, monkeypatch, extra):
    campaign = _fake_campaign(tmp_path, monkeypatch)
    spec = campaign.cache.kv_cache_groups[0].kv_cache_spec
    spec.__class__.__name__ = "SlidingWindowSpec"
    spec.sliding_window = 128
    if extra is not None:
        spec.extra_retained_tokens = extra
    for record in campaign.run():
        assert "cache.groups" in record["unresolved_fields"]
        assert record["cache"]["groups"][0]["extra_retained_tokens"] == extra


def test_v027_absent_extra_retention_maps_to_audited_zero(tmp_path, monkeypatch):
    campaign = _fake_campaign(tmp_path, monkeypatch, version="0.27.0")
    spec = campaign.cache.kv_cache_groups[0].kv_cache_spec
    spec.__class__.__name__ = "SlidingWindowSpec"
    spec.sliding_window = 128
    for record in campaign.run():
        assert record["unresolved_fields"] == []
        assert record["cache"]["groups"][0]["extra_retained_tokens"] == 0


@pytest.mark.parametrize("failure", ["profile", "initialize", "warmup"])
def test_underlying_worker_exceptions_propagate_without_completion_evidence(tmp_path, monkeypatch, failure):
    campaign = _fake_campaign(tmp_path, monkeypatch)
    campaign.base.fail = failure
    with pytest.raises(RuntimeError, match=f"runtime {failure} failure"):
        campaign.run()
    assert not list(tmp_path.rglob("runtime-observation-*.json"))


def test_loaded_config_hash_comes_from_actual_runtime_path(tmp_path, monkeypatch):
    import hashlib

    campaign = _fake_campaign(tmp_path, monkeypatch)
    campaign.model_config.write_text('{"model_type":"changed-runtime-config"}')
    for record in campaign.run():
        assert record["model_config_sha256"] == hashlib.sha256(campaign.model_config.read_bytes()).hexdigest()
        assert record["model_config_sha256"] != campaign.launch["model_config"]["sha256"]


def test_backend_without_source_mapping_is_unresolved(tmp_path, monkeypatch):
    campaign = _fake_campaign(tmp_path, monkeypatch)
    worker = campaign.adapter.ObservedWorker()
    worker.model_runner.attn_groups[0][0].backend.__module__ = "vendor.unaudited_backend"
    record = next(record for record in campaign.run() if record["kind"] == "worker")
    assert "cache.layer_tensors" in record["unresolved_fields"]
    assert "backend source" in record["error"]


def test_speculative_runtime_is_unresolved_without_erasing_raw_config(tmp_path, monkeypatch):
    campaign = _fake_campaign(tmp_path, monkeypatch)
    campaign.config.speculative_config = {"method": "synthetic-extra-tokens"}
    record = next(record for record in campaign.run() if record["kind"] == "worker")
    assert "cache.semantics" in record["unresolved_fields"]
    assert record["resolved_config"]["speculative_config"] == campaign.config.speculative_config
    assert record["cache"]["semantics"]["speculative"] is True


def test_non_hf_loaded_config_is_not_bound_to_incidental_config_json(tmp_path, monkeypatch):
    campaign = _fake_campaign(tmp_path, monkeypatch)
    campaign.config.model_config.config_format = "mistral"
    for record in campaign.run():
        assert "model_config_sha256" in record["unresolved_fields"]
        assert "model_config_sha256" not in record


def test_formal_runtime_hook_preserves_existing_execution_evidence(tmp_path, monkeypatch):
    import json

    campaign = _fake_campaign(tmp_path, monkeypatch)
    original_init = campaign.base.__init__

    def initialized(self):
        original_init(self)
        (tmp_path / "prefill" / "collector-provenance.json").write_text(json.dumps({"cell_id": "synthetic"}))

    monkeypatch.setattr(campaign.base, "__init__", initialized)
    records = campaign.run()
    assert all(record["unresolved_fields"] == [] for record in records)
    files = list((tmp_path / "prefill").glob("fpm-execution-worker-*.json"))
    assert len(files) == 1
    evidence = json.loads(files[0].read_text())
    assert evidence["status"] == "observed"
    assert evidence["attention_groups"][0]["backend_class"].startswith("vllm.v1.attention.backends.flashinfer.")


def test_unknown_cache_spec_keeps_its_raw_runtime_fields(tmp_path, monkeypatch):
    campaign = _fake_campaign(tmp_path, monkeypatch)
    spec = campaign.cache.kv_cache_groups[0].kv_cache_spec
    spec.__class__.__name__ = "NewUnsupportedSpec"
    records = campaign.run()
    assert all("cache" in record["unresolved_fields"] for record in records)
    assert all(
        record["raw_cache_config"]["groups"][0]["spec_type"].endswith(".NewUnsupportedSpec") for record in records
    )
    assert all(record["raw_cache_config"]["groups"][0]["page_size_bytes"] == 4096 for record in records)


def test_underlying_scheduler_exception_propagates_without_scheduler_record(tmp_path, monkeypatch):
    import sys

    campaign = _fake_campaign(tmp_path, monkeypatch)
    scheduler = sys.modules["dynamo.vllm.instrumented_scheduler"].InstrumentedScheduler

    def fail(*args, **kwargs):
        raise RuntimeError("native scheduler initialization failed")

    monkeypatch.setattr(scheduler, "__init__", fail)
    with pytest.raises(RuntimeError, match="native scheduler initialization failed"):
        campaign.run()
    assert not list(tmp_path.rglob("runtime-observation-scheduler-*.json"))


def test_model_sidecar_hashes_are_read_from_the_actual_loaded_config_directory(tmp_path, monkeypatch):
    import hashlib

    campaign = _fake_campaign(tmp_path, monkeypatch)
    sidecar = campaign.model_config.parent / "hf_quant_config.json"
    sidecar.write_text('{"quantization":{"quant_algo":"NVFP4"}}')
    campaign.launch["model_config"]["source_files"] = {"hf_quant_config.json": "0" * 64}
    for record in campaign.run():
        assert record["model_config_source_files"] == {
            "hf_quant_config.json": hashlib.sha256(sidecar.read_bytes()).hexdigest()
        }
        assert record["model_config_sha256"] == campaign.launch["model_config"]["sha256"]


def test_missing_model_sidecar_keeps_primary_config_evidence_unresolved(tmp_path, monkeypatch):
    campaign = _fake_campaign(tmp_path, monkeypatch)
    campaign.launch["model_config"]["source_files"] = {"hf_quant_config.json": "0" * 64}
    for record in campaign.run():
        assert record["unresolved_fields"]
        assert "hf_quant_config.json" in record["error"]
        assert record["model_config_sha256"] == campaign.launch["model_config"]["sha256"]


def _hub_loaded_config(campaign, monkeypatch, *, commit, text_commit):
    """Load the campaign's model by Hub ID through a config that forwards like vLLM 0.31.0 Glm5NextConfig.

    Once ``text_config`` is set, Glm5NextConfig reads every instance or class attribute of its text
    config from the text config, and writes the instance attributes there, except the keys of
    ``_UNMIRRORED_KEYS``. ``_commit_hash`` is not one of them (transformers_utils/configs/glm5_next.py:385-420).
    The base-class constructor runs first, so the top-level config keeps the ``_commit_hash`` that
    transformers gives it (:350-366). huggingface_hub is not installed in the test environment, so a
    stand-in module answers the cache lookups. Returns the cached config.json and the lookups.
    """
    import sys
    from types import ModuleType

    unmirrored = (
        "_name_or_path",
        "model_type",
        "dtype",
        "torch_dtype",
        "architectures",
        "_attn_implementation_internal",
    )

    class Config:
        # transformers PreTrainedConfig stores _commit_hash on every config that it builds.
        def __init__(self, commit_hash):
            self._commit_hash = commit_hash

    class TextConfig(Config):
        pass

    class ForwardingConfig(Config):
        def __init__(self):
            super().__init__(commit)
            self.text_config = TextConfig(text_commit)

        def __getattribute__(self, key):
            text = object.__getattribute__(self, "__dict__").get("text_config")
            if text is not None and key not in unmirrored and (key in vars(text) or key in vars(type(text))):
                return getattr(text, key)
            return object.__getattribute__(self, key)

        def __setattr__(self, key, value):
            text = object.__getattribute__(self, "__dict__").get("text_config")
            if text is not None and key not in unmirrored and key in vars(text):
                setattr(text, key, value)
            else:
                object.__setattr__(self, key, value)

    model = campaign.config.model_config
    model.model, model.hf_config = "zai-org/GLM-5.3-Flash", ForwardingConfig()
    cached = campaign.root / "hf-cache" / "config.json"
    cached.parent.mkdir()
    cached.write_text('{"model_type":"glm5_next"}')
    lookups = []

    def try_to_load_from_cache(repo_id, filename, *, revision=None):
        lookups.append((repo_id, filename, revision))
        return str(cached) if filename == "config.json" else None

    hub = ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = try_to_load_from_cache
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return cached, lookups


def test_hub_loaded_config_commit_is_read_from_the_config_itself(tmp_path, monkeypatch):
    import hashlib

    campaign = _glm_campaign(tmp_path, monkeypatch)
    # The top-level config holds the cache commit, and its text config holds None.
    cached, lookups = _hub_loaded_config(campaign, monkeypatch, commit="e" * 40, text_commit=None)
    digest = hashlib.sha256(cached.read_bytes()).hexdigest()
    for record in campaign.run():
        assert record["unresolved_fields"] == [], record.get("error")
        assert record["model_config_sha256"] == digest
        assert record["loaded_model_config"] == {"path": str(cached.resolve()), "sha256": digest}
    hub_id = "zai-org/GLM-5.3-Flash"
    assert lookups == [(hub_id, "params.json", "e" * 40), (hub_id, "config.json", "e" * 40)] * 2


@pytest.mark.parametrize("case", ["text_config_commit_only", "no_config_object"])
def test_hub_loaded_config_without_its_own_commit_is_unresolved(tmp_path, monkeypatch, case):
    campaign = _glm_campaign(tmp_path, monkeypatch)
    _, lookups = _hub_loaded_config(campaign, monkeypatch, commit=None, text_commit="e" * 40)
    if case == "no_config_object":
        # An object without an instance dictionary has no commit of its own either.
        campaign.config.model_config.hf_config = None
    for record in campaign.run():
        assert record["unresolved_fields"] == ["model_config_sha256"]
        assert "loaded remote config has no immutable cache commit" in record["error"]
        assert "model_config_sha256" not in record
    assert lookups == []


def _glm_campaign(tmp_path, monkeypatch):
    return _fake_campaign(tmp_path, monkeypatch, version="0.31.0", runner_version=2, layout="glm_tp4")


def test_v031_glm_worker_record_is_complete(tmp_path, monkeypatch):
    campaign = _glm_campaign(tmp_path, monkeypatch)
    for phase in ("prefill", "decode"):
        records = campaign.run(phase)
        assert len(records) == 2
        assert all(record["unresolved_fields"] == [] for record in records), [record.get("error") for record in records]
        worker, scheduler = (
            next(record for record in records if record["kind"] == kind) for kind in ("worker", "scheduler")
        )
        cache = worker["cache"]
        assert [group["kind"] for group in cache["groups"]] == ["attention"] + ["state"] * 5
        assert cache["prefix_caching"] == {"enable_prefix_caching": True, "mamba_cache_mode": "align"}
        views = cache["layer_tensors"]
        assert all(isinstance(layer, list) for layer in views.values())
        assert [len(layer) for name, layer in views.items() if name.endswith(".kda")] == [2] * 34
        kernels = {}
        for name, layer in views.items():
            kernels.setdefault(name.rsplit(".", 1)[1], set()).update(view["kernel_block_size_tokens"] for view in layer)
        assert kernels == {"mla": {64}, "indexer": {128}, "tail": {4}, "kda": {2176}}
        assert {view["block_axis"] for layer in views.values() for view in layer} == {0}
        # MLA and the indexer bind 3-D views: their bind_kv_cache squeezes H.
        latent, indexer = views["layers.3.mla"][0], views["layers.3.indexer"][0]
        assert (latent["shape"], latent["stride_bytes"]) == ([272, 64, 512], [32_768, 512, 1])
        assert (indexer["shape"], indexer["stride_bytes"]) == ([136, 32, 132], [4224, 132, 1])
        # The SSM state starts after the 36,864-byte convolution state of the page.
        assert [view["storage_offset_bytes"] for view in views["layers.0.kda"]] == [0, 36_864]
        allocation = cache["tensor_allocations"][0]
        assert (allocation["layers"], allocation["layer_stride"]) == (["layers.3.mla"], 8_912_896)
        # G0 lists each indexer before its MLA layer, and the scheduler copy holds the first inner spec.
        attention = cache["groups"][0]
        first = attention["layer_specs"][attention["layer_names"][0]]
        assert (attention["layer_names"][0], first["page_size_bytes"], first["cache_role"]) == (
            "layers.3.indexer",
            71_808,
            "INDEXER",
        )
        assert scheduler["cache"]["groups"][0]["spec_page_size_bytes"] == 71_808
        assert worker["raw_cache_config"]["kv_cache_layout"] == "LBHNC"
        assert worker["raw_cache_config"]["tensor_allocations"][0]["layer_stride"] == 8_912_896
        # A uniform group spec inherits the KVCacheSpec tokens_per_state, which raises NotImplementedError
        # (kv_cache_interface.py:185-187), so its raw record leaves that field out. The layer specs of its
        # group keep the value of each layer, and the raw records of other spec classes keep the field.
        raw_groups = worker["raw_cache_config"]["groups"]
        assert [set(group) for group in raw_groups[:2]] == [
            {"layer_names", "spec_type", "block_size", "page_size_bytes"}
        ] * 2
        assert {name: spec["tokens_per_state"] for name, spec in attention["layer_specs"].items()} == {
            name: 4 if name.endswith(".indexer") else 1 for name in attention["layer_names"]
        }
        assert [group["tokens_per_state"] for group in raw_groups[2:]] == [-1] * 4
        assert scheduler["raw_cache_config"]["groups"][0]["tokens_per_state"] == 4
        assert scheduler["cache"]["prefix_caching"] == {
            "enable_prefix_caching": True,
            "mamba_cache_mode": "align",
            "block_size": 2176,
            "hash_block_size": 2176,
        }
        # The startup capacity is engine-core evidence of the scheduler record only.
        assert (scheduler["cache"]["kv_cache_size_tokens"], scheduler["cache"]["kv_cache_max_concurrency"]) == (
            15_232,
            3.71875,
        )


def test_v031_glm_records_import_complete(tmp_path, monkeypatch):
    from collector.fpm_forward.runtime_observations import validate_observations

    from .test_runtime_observations import glm_tp4_cache_groups

    # The records that the 0.31.0 adapter writes go through the importer unchanged.
    campaign = _glm_campaign(tmp_path, monkeypatch)
    for phase in ("prefill", "decode"):
        records = campaign.run(phase)
        assert all(record["unresolved_fields"] == [] for record in records), [record.get("error") for record in records]
    result = validate_observations(tmp_path / "observations.json", {"worker": campaign.launch})["worker"]
    assert result["status"] == "complete", result["diagnostics"]
    assert result["resources"]["cache_groups"] == glm_tp4_cache_groups()
    memory = result["resources"]["runtime_memory"]
    assert memory["kv_cache_bytes"] == 7 * 13_045_120
    assert (memory["enable_prefix_caching"], memory["mamba_cache_mode"], memory["hash_block_size_tokens"]) == (
        True,
        "align",
        2176,
    )


def test_v031_view_shape_mismatch_is_unresolved(tmp_path, monkeypatch):
    campaign = _glm_campaign(tmp_path, monkeypatch)
    campaign.config.compilation_config.static_forward_context["layers.3.mla"].kv_cache.shape = [270, 64, 512]
    worker = next(record for record in campaign.run() if record["kind"] == "worker")
    assert worker["unresolved_fields"] == ["cache.layer_tensors"]
    assert "layers.3.mla" in worker["error"]


@pytest.mark.parametrize("change", [None, "latent_storage_block", "indexer_role"])
def test_v031_indexer_storage_block_is_the_kernel_block(tmp_path, monkeypatch, change):
    campaign = _glm_campaign(tmp_path, monkeypatch)
    specs = campaign.cache.kv_cache_groups[0].kv_cache_spec.kv_cache_specs
    # Only the sparse MLA indexer stores its tokens in blocks other than the cache block.
    if change == "latent_storage_block":
        specs["layers.3.mla"].storage_block_size = 128
    elif change == "indexer_role":
        specs["layers.3.indexer"].cache_role = type(specs["layers.3.indexer"].cache_role)["SPARSE"]
    records = campaign.run()
    worker, scheduler = (
        next(record for record in records if record["kind"] == kind) for kind in ("worker", "scheduler")
    )
    if change is None:
        assert worker["unresolved_fields"] == scheduler["unresolved_fields"] == []
        assert worker["cache"]["layer_tensors"]["layers.3.indexer"][0]["kernel_block_size_tokens"] == 128
        return
    # The scheduler copy of G0 holds the indexer spec only, so only the worker sees a latent storage block.
    flagged = [worker] if change == "latent_storage_block" else [worker, scheduler]
    for record in flagged:
        assert "cache.groups" in record["unresolved_fields"]
        assert "compressed cache storage" in record["error"]
    if change == "latent_storage_block":
        assert scheduler["unresolved_fields"] == []


def test_v031_backend_module_outside_the_manifest_is_unresolved(tmp_path, monkeypatch):
    campaign = _glm_campaign(tmp_path, monkeypatch)
    campaign.adapter.ObservedWorker().model_runner.attn_groups[2][0].backend.__module__ = "vendor.kda_backend"
    worker = next(record for record in campaign.run() if record["kind"] == "worker")
    assert "cache.layer_tensors" in worker["unresolved_fields"]
    assert "backend source" in worker["error"]


@pytest.mark.parametrize("change", ["host_resident", "role", "enable_kv_transfer", "tensor"])
def test_v031_hisparse_placement_is_unresolved(tmp_path, monkeypatch, change):
    campaign = _glm_campaign(tmp_path, monkeypatch)
    # The worker and scheduler cache configs share the Mamba groups and the tensors.
    group = campaign.cache.kv_cache_groups[2]
    if change == "host_resident":
        group.host_resident = True
    elif change == "role":
        group.role = type(group.role)["HISPARSE_SOURCE"]
    elif change == "enable_kv_transfer":
        group.enable_kv_transfer = False
    else:
        campaign.cache.kv_cache_tensors[0].host_resident = True
    name = "tensor placement for ['layers.3.mla']" if change == "tensor" else "group placement for ['layers.0.kda',"
    for record in campaign.run():
        assert "cache.groups" in record["unresolved_fields"]
        assert name in record["error"]


@pytest.mark.parametrize("runner_version", [1, 2])
def test_v031_plain_attention_record_is_complete(tmp_path, monkeypatch, runner_version):
    campaign = _fake_campaign(tmp_path, monkeypatch, version="0.31.0", runner_version=runner_version)
    records = campaign.run()
    assert len(records) == 2
    assert all(record["unresolved_fields"] == [] for record in records), [record.get("error") for record in records]
    worker, scheduler = (
        next(record for record in records if record["kind"] == kind) for kind in ("worker", "scheduler")
    )
    assert [group["kind"] for group in worker["cache"]["groups"]] == ["attention"]
    # A plain attention layer binds the [B, H, N, C] view unchanged.
    assert worker["cache"]["layer_tensors"]["layer0"] == [
        {
            "storage_id": "storage-0",
            "storage_offset_bytes": 0,
            "shape": [10, 1, 16, 128],
            "stride_bytes": [4096, 4096, 256, 2],
            "element_size_bytes": 2,
            "block_axis": 0,
            "kernel_block_size_tokens": 16,
        }
    ]
    assert scheduler["cache"]["prefix_caching"] == {
        "enable_prefix_caching": True,
        "mamba_cache_mode": "none",
        "block_size": 16,
        "hash_block_size": 16,
    }


def test_v031_only_an_mla_storage_block_sets_the_view_kernel_block(tmp_path, monkeypatch):
    campaign = _fake_campaign(tmp_path, monkeypatch, version="0.31.0", runner_version=2)
    # vLLM views a layer with its storage block only for an MLAAttentionSpec (worker/utils.py:449-450).
    campaign.cache.kv_cache_groups[0].kv_cache_spec.storage_block_size = 8
    worker = next(record for record in campaign.run() if record["kind"] == "worker")
    assert worker["unresolved_fields"] == ["cache.groups"]
    assert worker["cache"]["layer_tensors"]["layer0"][0]["kernel_block_size_tokens"] == 16
