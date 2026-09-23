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


@pytest.mark.parametrize("version", ["0.27.0", "0.28.0"])
def test_frozen_scheduler_export_preserves_native_benchmark_contract(tmp_path, monkeypatch, version):
    import importlib

    _fake_campaign(tmp_path, monkeypatch, version=version)
    frozen = load_instrumentation(tmp_path / "instrumentation/manifest.json")
    scheduler_name = frozen.manifest["scheduler_class"]
    # Both pinned Dynamo args.py consumers require this substring before importing
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


def _fake_campaign(tmp_path, monkeypatch, *, packed=False, planar=False, version="0.28.0"):
    """Synthetic vendor files and GPU objects; verifies mapping, never GPU support."""
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
    blocks, page = 10, 4096
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
        attn.append([SimpleNamespace(layer_names=[name], kv_cache_spec=spec, kv_cache_group_id=index, backend=Backend)])
    cache = SimpleNamespace(num_blocks=blocks, kv_cache_groups=groups, kv_cache_tensors=allocations)
    config.compilation_config.static_forward_context = layers
    runner = SimpleNamespace(
        kv_cache_config=cache,
        kv_caches=[layer.kv_cache for layer in layers.values()],
        shared_kv_cache_layers={},
        attn_groups=attn,
        _kernel_block_sizes=[16] * len(groups),
    )
    block_objects = [SimpleNamespace(block_id=index, is_null=index == 0) for index in range(blocks)]
    pool = SimpleNamespace(
        blocks=block_objects,
        num_gpu_blocks=blocks,
        null_block=block_objects[0],
        get_num_free_blocks=lambda: blocks - 1,
        free_block_queue=SimpleNamespace(get_all_free_blocks=lambda: block_objects[1:]),
    )
    manager = SimpleNamespace(
        block_pool=pool,
        watermark_blocks=0,
        coordinator=SimpleNamespace(single_type_managers=[SimpleNamespace(block_pool=pool) for group in groups]),
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
            assert supplied_config is config and supplied_cache is cache
            self.vllm_config, self.kv_cache_manager, self._fpm_dp_rank = config, manager, 0

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
            "kvcache_quant_mode": "bfloat16",
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
        adapter.ObservedInstrumentedScheduler(config, cache)
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
    result = validate_observations(tmp_path / "observations.json", {"worker": campaign.launch})["worker"]
    assert result["status"] == "complete", result["diagnostics"]


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
