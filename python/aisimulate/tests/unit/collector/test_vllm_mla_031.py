# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from collector.version_resolver import resolve_module
from collector.vllm.registry import REGISTRY

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3] / "collector/vllm"


def load(name, functions, namespace):
    path = ROOT / name
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in functions]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class Tensor(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype)


def test_metadata_builder_uses_serving_factory_for_backend_required_arguments():
    calls = []
    backend, spec, config, builder = object(), object(), object(), object()
    device = object()

    class NativeGroup:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def create_metadata_builders(self, actual_config, actual_device):
            assert actual_config is config and actual_device is device
            self.ready = True

        def get_metadata_builder(self):
            assert self.ready
            return builder

    module = load(
        "collect_mla_module_v2.py",
        {"create_metadata_builder"},
        {"AttentionGroup": NativeGroup, "torch": SimpleNamespace(device=lambda _: device)},
    )
    result = module.create_metadata_builder(backend, spec, ["indexer"], config, "cuda")
    assert result is builder
    assert calls == [dict(backend=backend, layer_names=["indexer"], kv_cache_spec=spec, kv_cache_group_id=0)]


def test_native_binding_and_history_writer_use_same_physical_page_table():
    writes, bound = [], []
    raw = np.zeros((8, 1, 2, 576))
    spec = SimpleNamespace(block_size=2, cache_dtype_str="fp8")
    layer = SimpleNamespace(hisparse_cache=None, _k_scale=object(), get_attn_backend=lambda: "native")

    def bind(cache):
        bound.append(cache)
        layer.kv_cache = cache.squeeze(1)

    layer.bind_kv_cache = bind
    layer.update_kv_cache = lambda *args: writes.append(args)
    namespace = {
        "torch": SimpleNamespace(int64=np.int64, arange=lambda count, **_: np.arange(count).view(Tensor)),
        "allocate_cache": lambda config, backend, actual_spec, blocks, device: raw,
    }
    module = load("collect_mla_module_v2.py", {"bind_cache", "populate_mla_cache"}, namespace)
    common = SimpleNamespace(block_table_tensor=np.array([[4, 5], [6, 0]]).view(Tensor))
    latent = [np.ones((3, 512)), np.empty((0, 512))]
    rope = [np.ones((3, 1, 64)), np.empty((0, 1, 64))]
    result = module.populate_mla_cache(object(), layer, spec, 8, common, latent, rope, "cpu")
    assert bound[0] is raw and result is layer.kv_cache
    assert len(writes) == 1  # Empty history has no write; the queued forward remains unchanged.
    assert writes[0][0] is latent[0]
    assert writes[0][1].shape == (3, 64)
    assert writes[0][2] is result
    np.testing.assert_array_equal(writes[0][3], [8, 9, 10])
    assert writes[0][4:] == (None, "fp8", layer._k_scale)
    layer.hisparse_cache = object()
    with pytest.raises(NotImplementedError, match="HiSparse"):
        module.populate_mla_cache(object(), layer, spec, 8, common, latent, rope, "cpu")


def test_worker_forwards_native_adapter_inside_inference_mode():
    calls = []
    runtime = object()
    module = load(
        "collect_mla_module.py",
        {"run_mla_module_worker"},
        {
            "torch": SimpleNamespace(inference_mode=nullcontext),
            "run_mla_module": lambda **kwargs: calls.append(kwargs),
        },
    )
    module.run_mla_module_worker(
        128, 1, 8, "fp8", "bfloat16", "bfloat16", "model", "dsa", perf_filename="dsa_context_perf.txt", _runtime=runtime
    )
    assert calls[0]["_runtime"] is runtime
    assert calls[0]["attn_type"] == "dsa"


def test_031_routes_preserve_old_family_versions():
    for op in ("mla_context_module", "mla_generation_module", "dsa_context_module", "dsa_generation_module"):
        entry = next(entry for entry in REGISTRY if entry.op == op)
        stem = "collect_mla_module" if op.startswith("mla") else "collect_dsa_module"
        assert resolve_module(entry, "0.31.0") == f"collector.vllm.{stem}_v2"
        assert resolve_module(entry, "0.25.0") == f"collector.vllm.{stem}_v1"
    calls = []
    module = load(
        "collect_mla_module_v2.py",
        {"create_common_metadata"},
        {"_create_common_metadata": lambda *args, **kwargs: calls.append((args, kwargs))},
    )
    module.create_common_metadata("batch", 64, 8192, "cpu", True)
    assert calls[0][1] == {"randomize_blocks": False}
