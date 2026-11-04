# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded native Attention boundary audit; diagnostic rows are never published."""

# ruff: noqa: E402
# Imports follow the mounted collector path setup.
import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import sys
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--kind", choices=["collector", "native"], required=True)
parser.add_argument("--context", type=int, required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path.cwd() / "source/python/aisimulate"))
import torch
from collector.vllm import collect_attn as c
from vllm.config import get_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.attention.backends import flash_attn as fa

out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
r = {
    "kind": args.kind,
    "context": bool(args.context),
    "shape": [1, 4096, 4, 1, 512, True, bool(args.context), 0],
    "publish_eligible": False,
    "source_ref": "c569abb1cd034c934f47974209515e3507f60b8a",
}


def describe(x):
    if isinstance(x, torch.Tensor):
        return {
            "shape": list(x.shape),
            "stride": list(x.stride()),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "values": x.detach().cpu().flatten().tolist() if x.numel() <= 8 else None,
        }
    if isinstance(x, (str, int, float, bool, type(None))):
        return x
    return str(x)


def fields(x):
    return {k: describe(v) for k, v in vars(x).items() if not callable(v)}


orig_kernel = fa._FA4_DENSE_ATTENTION_KERNEL


def kernel(**kwargs):
    r.setdefault("kernel_arguments", {k: describe(v) for k, v in kwargs.items()})
    return orig_kernel(**kwargs)


fa._FA4_DENSE_ATTENTION_KERNEL = kernel
orig_backend = c.get_attention_backend
orig_mock = c.MockAttentionLayer
holder = {}


def backend(name):
    builder, impl_cls = orig_backend(name)

    class Builder(builder):
        def build(self, *a, **kw):
            holder["slots"] = kw["common_attn_metadata"].slot_mapping
            result = super().build(*a, **kw)
            r["metadata"] = fields(result)
            return result

    def impl_factory(**kw):
        impl = impl_cls(**kw)
        r["collector_impl"] = fields(impl)
        r["collector_impl_class"] = str(type(impl))
        if args.kind == "native":
            cfg = get_current_vllm_config()
            previous = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                native = Attention(
                    num_heads=4,
                    head_size=512,
                    scale=512**-0.5,
                    num_kv_heads=1,
                    cache_config=cfg.cache_config,
                    prefix="audit_attention",
                ).to("cuda:0")
            finally:
                torch.set_default_dtype(previous)
            holder["native"] = native
            r["native_impl"] = fields(native.impl)
            r["native_impl_class"] = str(type(native.impl))
            r["native_backend"] = str(native.attn_backend)

            def forward(layer, q, k, v, cache, metadata, output=None, **kwargs):
                native.kv_cache = cache
                # Native Attention performs its own query quantization and unified ops.
                with set_forward_context(
                    {"audit_attention": metadata},
                    cfg,
                    num_tokens=q.shape[0],
                    slot_mapping={"audit_attention": holder["slots"]},
                ):
                    result = native(q.to(torch.bfloat16), k, v)
                output.copy_(result.view_as(output))
                return output

            impl.forward = forward
        return impl

    return Builder, impl_factory


c.get_attention_backend = backend


def layer(device):
    mock = orig_mock(device)
    r["mock_fields"] = fields(mock)
    if args.kind == "native":
        native = holder["native"]
        r["native_layer_fields"] = fields(native)
        r["native_buffers"] = {k: describe(v) for k, v in native.named_buffers()}
        return native
    return mock


c.MockAttentionLayer = layer
start = time.monotonic()
try:
    r["packages"] = {p: importlib.metadata.version(p) for p in ["vllm", "torch"]}
    r["gpu"] = {
        "name": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
    }
    assert tuple(r["gpu"]["capability"]) == (9, 0)
    assert r["packages"]["vllm"].split("+")[0] == "0.31.0"
    r["source_hashes"] = {
        str(Path(m.__file__).name): hashlib.sha256(
            Path(m.__file__).read_bytes()
        ).hexdigest()
        for m in [c, fa]
    }
    # Audit exactly the native layer fields accessed by the selected implementation.
    tree = ast.parse(inspect.getsource(fa))
    r["native_backend_layer_reads"] = sorted(
        {
            n.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "layer"
        }
    )
    c.run_attention_torch(*r["shape"], perf_filename=str(out / "diagnostic.csv"))
    torch.cuda.synchronize()
    r["status"] = "passed"
except BaseException as e:
    r.update(status="failed", error=str(e), traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    r["call_seconds"] = time.monotonic() - start
    r["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (out / "result.json").write_text(json.dumps(r, indent=2) + "\n")
