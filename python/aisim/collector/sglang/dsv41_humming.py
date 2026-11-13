# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify native Humming W4A16 without changing measured serving calls.

Uses the public serving APIs in sgl-project/sglang at
1aa0e962b206102b7c439a4a0c4981cfec6e87bc, paths
python/sglang/srt/layers/quantization/humming_utils.py and
python/sglang/srt/layers/moe/moe_runner/humming.py. No upstream implementation
is copied. The extra qualification forward is outside all timed samples.
"""

import hashlib
import inspect
import json
import sys
from pathlib import Path


def loaded_humming_geometry(experts):
    """Read the native weight layout; never add collector-side padding."""
    if getattr(experts, "_dsv4_mxfp4_backend", None) != "humming":
        raise RuntimeError("native baseline Humming MoE has not completed weight processing")
    metas = getattr(experts, "humming_metas", {})
    schemas = getattr(experts, "input_schemas", {})
    result = {}
    for name in ("w13", "w2"):
        schema = schemas.get(name)
        if (
            schema is None
            or getattr(schema, "a_dtype", "missing") is not None
            or getattr(schema, "input_scale_dtype", "missing") is not None
            or getattr(schema, "input_scale_group_size", None) != 0
        ):
            raise RuntimeError("native baseline Humming MoE requires unquantized activation schemas")
        try:
            meta = json.loads(metas[name]._config_str)
        except (KeyError, AttributeError, TypeError, ValueError) as error:
            raise RuntimeError("native baseline Humming MoE lacks native weight metadata") from error
        expected = {
            "a_dtype": "bfloat16",
            "b_dtype": "float4e2m1",
            "c_dtype": "bfloat16",
            "bs_dtype": "float8e8m0",
            "as_dtype": None,
            "input_scale_group_size": 0,
            "weight_scale_group_size": 32,
            "weight_scale_group_size_n": 0,
            "weight_scale_type": "group",
            "weight_scale_2_type": "none",
            "num_experts": 384,
        }
        if any(meta.get(key) != value for key, value in expected.items()):
            raise RuntimeError("native baseline Humming MoE weight/activation precision differs")
        result[name] = meta
    local = 2304 // experts.moe_tp_size
    expected_shapes = {"w13": (2 * local, 5120), "w2": (5120, local)}
    for name, (n, k) in expected_shapes.items():
        meta = result[name]
        if (meta["shape_n"] - meta["pad_shape_n"], meta["shape_k"] - meta["pad_shape_k"]) != (n, k):
            raise RuntimeError("native baseline Humming MoE geometry differs from V4.1 TP")
    return result


def validate_humming_observations(calls, configs):
    """Admit the actual post-quantization operands, not the requested dtype."""
    if {call.get("sublayer") for call in calls} != {"w13", "w2"}:
        raise RuntimeError("native Humming qualification missed an expert projection")
    if not configs or len({config["core_id"] for config in configs}) != 1:
        raise RuntimeError("native Humming qualification missed the executed fused runner")
    for call in calls:
        if call.get("input_dtype") != "torch.bfloat16" or call.get("has_input_scale") is not False:
            raise RuntimeError("native Humming qualification observed quantized activations")
        compute = call.get("compute_config")
        if isinstance(compute, str):
            compute = json.loads(compute)
        if not isinstance(compute, dict) or compute.get("use_f16_accum") is not False:
            raise RuntimeError("native Humming qualification requires full-precision accumulation")


def qualify_native_humming(experts, hidden, topk):
    """Observe one untimed native forward; preserve the serving implementations."""
    import torch
    from humming.layer import HummingMethod
    from sglang.srt.layers.moe.moe_runner.humming import HummingRunnerCore

    geometry = loaded_humming_geometry(experts)
    calls, configs = [], []
    forward_code = HummingMethod.forward_layer.__func__.__code__
    configs_code = HummingRunnerCore.get_humming_gemm_configs.__code__

    def observe(frame, event, value):
        if event != "return":
            return
        local = frame.f_locals
        if frame.f_code is forward_code:
            inputs = local.get("inputs")
            calls.append(
                {
                    "sublayer": local.get("sublayer_name"),
                    "input_dtype": str(getattr(inputs, "dtype", None)),
                    "input_shape": list(inputs.shape) if inputs is not None else None,
                    "has_input_scale": local.get("input_scale") is not None,
                    "compute_config": local.get("compute_config"),
                }
            )
        elif frame.f_code is configs_code:
            configs.append({"core_id": id(local.get("self")), "config": value})

    previous = sys.getprofile()
    if previous is not None:
        raise RuntimeError("native Humming qualification requires an unused Python profiler")
    try:
        sys.setprofile(observe)
        output = experts(hidden, topk)
    finally:
        sys.setprofile(previous)
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise RuntimeError("native Humming qualification returned non-finite output")
    validate_humming_observations(calls, configs)
    source = Path(inspect.getfile(HummingMethod))
    return {
        "state": "actual_bf16_native_humming_calls_verified",
        "humming_layer_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "geometry": geometry,
        "calls": calls,
        "configs": configs,
        "timed_samples_profiled": False,
        "native_functions_replaced": False,
    }
