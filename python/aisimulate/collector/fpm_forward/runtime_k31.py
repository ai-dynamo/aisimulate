# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate K3.1 native observations, never infer missing runtime evidence.

Source contract: runtime/instrumentation/vllm-kimi-k31.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

VERSION = "0.1.dev24216+g266e88c05"
LAYOUT = "kimi-k31-block-outer/v1"
SPEC = "vllm.v1.kv_cache_interface."
KDA = "vllm.models.kimi_k3.nvidia.kda.KimiK3DeltaAttention"
ATTENTION_CLASSES = {
    "vllm.model_executor.layers.attention.mla_attention.MLAAttention",
    "vllm.models.kimi_k3.common.dsa_cache.KimiK3IndexerCache",
}


def is_audited_runtime(runtime):
    """Exact inspected version and source map, shared by planning and import."""
    pinned = json.loads((Path(__file__).parent / "runtime/vllm-kimi-k31.example.json").read_text())["runtime"]
    return isinstance(runtime, dict) and {key: runtime.get(key) for key in pinned} == pinned


def _positive(value, label):
    if type(value) is not int or not 0 < value <= 2**53:
        raise ValueError(f"K3.1 {label} must be a positive integer")
    return value


def _nonnegative(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"K3.1 {label} must be a nonnegative integer")
    return value


def moe_precision(evidence):
    if not isinstance(evidence, dict) or evidence.get("enumeration") != "model.named_modules:RoutedExperts":
        raise ValueError("Kimi K3.1 runtime weight precision requires initialized MoE backend and activation evidence")
    layers = evidence.get("layers")
    expected = evidence.get("expected_layer_names")
    if not isinstance(layers, list) or not layers or not isinstance(expected, list) or not expected:
        raise ValueError("K3.1 initialized MoE layer coverage is incomplete")
    names = [layer.get("name") for layer in layers]
    if len(set(names)) != len(names) or set(names) != set(expected) or len(expected) != len(set(expected)):
        raise ValueError("K3.1 initialized MoE layer coverage is inconsistent")
    modes = set()
    for layer in layers:
        if (
            layer.get("method_type") != "vllm.model_executor.layers.quantization.mxfp4.Mxfp4MoEMethod"
            or layer.get("layer_type") != "vllm.model_executor.layers.fused_moe.routed_experts.RoutedExperts"
            or layer.get("kernel_type") != "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernel"
        ):
            raise ValueError("K3.1 initialized MoE method/kernel has no audited precision mapping")
        descriptors = layer.get("descriptors", {})
        if descriptors != {
            key: {"dtype": dtype}
            for key, dtype in (("_a1", "mxfp8"), ("_a2", "mxfp8"), ("_w1", "mxfp4"), ("_w2", "mxfp4"))
        }:
            raise ValueError("K3.1 initialized MoE activation/weight descriptors are unsupported")
        backend = layer.get("backend")
        if backend != "FLASHINFER_TRTLLM_MXFP4_MXFP8":
            raise ValueError("K3.1 initialized MoE backend has no audited precision mapping")
        modes.add("w4a8_mxfp4_mxfp8_trtllm")
        shapes = layer.get("weight_shapes", {})
        if set(shapes) != {"w13_weight", "w2_weight"}:
            raise ValueError("K3.1 initialized expert weights are incomplete")
        for shape in shapes.values():
            if not isinstance(shape, list) or not shape:
                raise ValueError("K3.1 initialized expert weight shape is missing")
            for size in shape:
                _positive(size, "expert weight dimension")
    if len(modes) != 1:
        raise ValueError("K3.1 initialized expert precision differs across layers")
    return modes.pop()


def normalize_groups(cache, scheduled, page_bytes):
    if cache.get("allocation_layout") != LAYOUT or scheduled.get("allocation_layout") != LAYOUT:
        raise ValueError("K3.1 worker/scheduler allocation mapping disagrees")
    raw, other = cache.get("groups"), scheduled.get("groups")
    if not isinstance(raw, list) or not raw or not isinstance(other, list) or len(raw) != len(other):
        raise ValueError("K3.1 cache group coverage is incomplete")
    names, result = set(), []
    for index, (group, scheduler) in enumerate(zip(raw, other, strict=True)):
        expected = {key: value for key, value in group.items() if key not in {"kind", "layer_classes"}}
        if (
            group.get("spec_type") == SPEC + "UniformTypeKVCacheSpecs"
            and scheduler.get("spec_type") != SPEC + "UniformTypeKVCacheSpecs"
        ):
            # Pinned kv_cache_utils.generate_scheduler_kv_cache_config selects
            # the first native child; JSON serialization may reorder its keys.
            children = expected.pop("layer_specs", None)
            representative = expected.pop("representative_layer", None)
            if not isinstance(children, dict) or not isinstance(representative, str) or representative not in children:
                raise ValueError("K3.1 scheduler projection requires the observed representative layer")
            expected.update(children[representative])
        if expected != scheduler:
            raise ValueError("K3.1 worker and scheduler cache group geometry disagrees")
        layers = group.get("layer_names")
        if (
            not isinstance(layers, list)
            or not layers
            or any(not isinstance(x, str) or not x for x in layers)
            or len(set(layers)) != len(layers)
            or names.intersection(layers)
        ):
            raise ValueError("K3.1 cache layer names must be nonempty and unique")
        names.update(layers)
        classes = group.get("layer_classes", {})
        if set(classes) != set(layers) or any(not isinstance(x, str) or not x for x in classes.values()):
            raise ValueError("K3.1 cache layer class coverage is incomplete")
        block = _positive(group.get("block_size_tokens"), "block size")
        _positive(group.get("spec_page_size_bytes"), "spec page size")
        kind = group.get("kind")
        item = {
            "name": f"runtime_group_{index}",
            "kind": kind,
            "num_layers": len(layers),
            "block_size_tokens": block,
            "page_size_bytes": page_bytes,
            "sliding_window": None,
        }
        if (
            group.get("sliding_window") is not None
            or group.get("attention_chunk_size") is not None
            or group.get("extra_retained_tokens") != 0
            or type(group.get("extra_retained_tokens")) is not int
        ):
            raise ValueError("K3.1 alternate cache retention is unsupported")
        if group.get("spec_type") == SPEC + "UniformTypeKVCacheSpecs":
            children = group.get("layer_specs")
            if not isinstance(children, dict) or set(children) != set(layers):
                raise ValueError("K3.1 uniform group requires every native per-layer spec")
            if any(child.get("spec_type") == SPEC + "UniformTypeKVCacheSpecs" for child in children.values()):
                raise ValueError("nested K3.1 uniform specs are unsupported")
            if (
                sum(_positive(child.get("spec_page_size_bytes"), "layer page bytes") for child in children.values())
                != group["spec_page_size_bytes"]
            ):
                raise ValueError("K3.1 uniform group page sum contradicts native per-layer specs")
            child_groups = [
                {**child, "layer_names": [name], "kind": kind, "layer_classes": {name: classes[name]}}
                for name, child in children.items()
            ]
            child_scheduled = [
                {key: value for key, value in child.items() if key not in {"kind", "layer_classes"}}
                for child in child_groups
            ]
            normalized = normalize_groups(
                {"allocation_layout": LAYOUT, "groups": child_groups},
                {"allocation_layout": LAYOUT, "groups": child_scheduled},
                page_bytes,
            )
            if any(child["block_size_tokens"] != block or child["kind"] != kind for child in normalized):
                raise ValueError("K3.1 uniform group has incompatible per-layer retention")
            if kind == "state":
                item["blocks_per_request"] = 1
            result.append(item)
            continue
        if group.get("spec_type") == SPEC + "MambaSpec" and kind == "state":
            if (
                set(classes.values()) != {KDA}
                or group.get("mamba_cache_mode") != "none"
                or group.get("num_speculative_blocks") != 0
                or type(group.get("num_speculative_blocks")) is not int
                or group.get("blocks_per_request") != 1
                or type(group.get("blocks_per_request")) is not int
                or block < _positive(group.get("max_model_len"), "maximum context")
            ):
                raise ValueError("K3.1 state requires native KDA, mamba none and one non-speculative state page")
            shapes, dtypes = group.get("state_shapes"), group.get("state_dtypes")
            if not isinstance(shapes, list) or len(shapes) != 2 or not isinstance(dtypes, list) or len(dtypes) != 2:
                raise ValueError("K3.1 KDA requires both convolution and recurrence components")
            content = 0
            for shape, dtype in zip(shapes, dtypes, strict=True):
                if not isinstance(shape, list) or not shape or dtype not in {"torch.bfloat16", "torch.float32"}:
                    raise ValueError("K3.1 state shape/dtype is unsupported")
                content += math.prod(_positive(x, "state dimension") for x in shape) * (
                    2 if dtype == "torch.bfloat16" else 4
                )
            if content != group.get("state_content_size_bytes") or content > group["spec_page_size_bytes"]:
                raise ValueError("K3.1 state payload differs from native page geometry")
            padded = group.get("page_size_padded")
            if (padded is None and content != group["spec_page_size_bytes"]) or (
                padded is not None and padded != group["spec_page_size_bytes"]
            ):
                raise ValueError("K3.1 state padding differs from native page geometry")
            item["blocks_per_request"] = 1
        elif (
            group.get("spec_type") != SPEC + "MLAAttentionSpec"
            or kind != "attention"
            or set(classes.values()) - ATTENTION_CLASSES
        ):
            raise ValueError("K3.1 cache spec/kind has no audited mapping")
        result.append(item)
    return result


def _kernel_view_geometry(mapping, group, view, count, page, width):
    """Check the observed AttentionGroup's manager-to-kernel address mapping."""
    if (
        not isinstance(mapping, dict)
        or mapping.get("attention_group_class") != "vllm.v1.worker.utils.AttentionGroup"
        or mapping.get("backend_class") != "vllm.models.kimi_k3.common.dsa_cache.KimiK3IndexerBackend"
        or type(mapping.get("maps_slot_mapping")) is not bool
        or mapping.get("writes_kv_cache_from_block_table") is not True
    ):
        raise ValueError("K3.1 kernel-block view has no audited native address mapping")
    kernel = _positive(mapping.get("kernel_block_size"), "kernel block size")
    stride = _positive(mapping.get("kernel_block_stride"), "kernel block stride")
    block = group["block_size_tokens"]
    if kernel >= block or block % kernel:
        raise ValueError("K3.1 kernel-block subdivision contradicts the native manager block")
    subdivisions = block // kernel
    offsets = mapping.get("kernel_block_offsets")
    if (
        not isinstance(offsets, list)
        or any(type(value) is not int for value in offsets)
        or offsets != list(range(subdivisions))
    ):
        raise ValueError("K3.1 native kernel-block offsets do not cover the manager block")
    table = mapping.get("kernel_block_table_shape")
    if (
        mapping.get("kernel_block_table_dtype") != "torch.int32"
        or not isinstance(table, list)
        or len(table) != 3
        or any(type(value) is not int or value < 1 for value in table)
        or table[-1] % subdivisions
    ):
        raise ValueError("K3.1 native kernel-block table shape/dtype is inconsistent")
    shape, strides = view["shape"], view["stride_bytes"]
    if len(shape) != 3 or shape[1] != kernel or strides[1:] != [shape[2], 1]:
        raise ValueError("K3.1 kernel-block component has no native contiguous token layout")
    kernel_bytes = kernel * shape[2]
    # AttentionGroup.map_kv_cache_to_kernel_blocks and map_to_kernel_block_table:
    # kernel page j of manager block b is b * stride + j. The strided tensor
    # spans other layers' gaps; only these addressed pages belong to this layer.
    if (
        strides[0] != kernel_bytes
        or stride * kernel_bytes != page
        or subdivisions * kernel_bytes != width
        or shape[0] != (count - 1) * stride + subdivisions
    ):
        raise ValueError("K3.1 kernel-block view contradicts native pool stride or layer page coverage")
    return kernel


def validate_views(cache, precision, count, page):
    """Verify block-outer native placements, including safe cross-group overlays.

    A block ID belongs to exactly one cache group. Different groups may overlap
    within that block's byte slot; layers in the same group may never overlap.
    """
    if precision != "fp8":
        raise ValueError("K3.1 native adapter requires FP8 attention/indexer cache")
    storages = cache.get("storages")
    if not isinstance(storages, list) or len(storages) != 1 or storages[0].get("size_bytes") != count * page:
        raise ValueError("K3.1 native allocator requires one exact shared storage")
    storage = storages[0].get("storage_id")
    if not isinstance(storage, str) or not storage:
        raise ValueError("K3.1 backing storage identity is missing")
    by_name = {}
    for index, group in enumerate(cache["groups"]):
        for name in group["layer_names"]:
            spec = {**group, **group.get("layer_specs", {}).get(name, {})}
            by_name[name] = (index, spec)
            if spec["kind"] == "attention" and spec.get("dtype") not in {"torch.uint8", "torch.float8_e4m3fn"}:
                raise ValueError("K3.1 attention storage contradicts FP8 precision")
    views = cache.get("layer_components")
    if not isinstance(views, dict) or set(views) != set(by_name):
        raise ValueError("K3.1 bound component coverage is incomplete")
    kernel_views = cache.get("kernel_block_views", {})
    if not isinstance(kernel_views, dict) or set(kernel_views) - set(by_name):
        raise ValueError("K3.1 kernel-block view layer coverage is inconsistent")
    allocations = cache.get("tensor_allocations")
    if not isinstance(allocations, list) or not allocations:
        raise ValueError("K3.1 native placements are missing")
    covered, intervals = set(), {}
    for allocation in allocations:
        if (
            allocation.get("storage_id") != storage
            or allocation.get("size") != count * page
            or allocation.get("block_stride") != page
            or allocation.get("host_resident") is not False
        ):
            raise ValueError("K3.1 native placement is not an exact device block-outer allocation")
        layers = allocation.get("layers")
        if (
            not isinstance(layers, list)
            or not layers
            or any(name not in by_name for name in layers)
            or len(set(layers)) != len(layers)
            or covered.intersection(layers)
        ):
            raise ValueError("K3.1 native placement layer coverage is inconsistent")
        step = _positive(allocation.get("layer_stride"), "layer stride")
        start = _nonnegative(allocation.get("offset"), "placement offset")
        for index, name in enumerate(layers):
            gid, group = by_name[name]
            offset, width = start + index * step, group["spec_page_size_bytes"]
            if offset + width > page:
                raise ValueError("K3.1 placement exceeds its pool block slot")
            for lo, hi in intervals.setdefault(gid, []):
                if offset < hi and lo < offset + width:
                    raise ValueError("K3.1 layers within one group overlap")
            intervals[gid].append((offset, offset + width))
            components = views[name]
            state = group["kind"] == "state"
            mapping = kernel_views.get(name)
            if name in kernel_views and (
                state or group["layer_classes"].get(name) != "vllm.models.kimi_k3.common.dsa_cache.KimiK3IndexerCache"
            ):
                raise ValueError("K3.1 kernel-block mapping is not an observed native indexer cache")
            if not isinstance(components, list) or len(components) != (2 if state else 1):
                raise ValueError("K3.1 layer component count contradicts native spec")
            component_offset = offset
            for component_index, view in enumerate(components):
                size = _positive(view.get("element_size_bytes"), "component element bytes")
                shape, strides = view.get("shape"), view.get("stride_bytes")
                if (
                    not isinstance(shape, list)
                    or not shape
                    or not isinstance(strides, list)
                    or len(shape) != len(strides)
                    or view.get("storage_id") != storage
                    or view.get("storage_offset_bytes") != component_offset
                ):
                    raise ValueError("K3.1 component block coverage, offsets or storage disagrees")
                for value in shape:
                    _positive(value, "component dimension")
                for value in strides:
                    if _nonnegative(value, "component stride") % size:
                        raise ValueError("K3.1 component stride is not element aligned")
                component_block = group["block_size_tokens"]
                if name in kernel_views:
                    component_block = _kernel_view_geometry(mapping, group, view, count, page, width)
                elif shape[0] != count or strides[0] != page:
                    raise ValueError("K3.1 component block coverage, offsets or storage disagrees")
                if state:
                    dtype = group["state_dtypes"][component_index]
                    if (
                        shape[1:] != group["state_shapes"][component_index]
                        or view.get("dtype") != dtype
                        or size != (2 if dtype == "torch.bfloat16" else 4)
                    ):
                        raise ValueError("K3.1 component shape/dtype differs from native state spec")
                elif size != 1 or view.get("dtype") != group["dtype"] or len(shape) != 3 or shape[1] != component_block:
                    raise ValueError("K3.1 attention component shape/dtype differs from native cache spec")
                span = size
                for stride, dimension in sorted(zip(strides[1:], shape[1:], strict=True)):
                    if dimension > 1:
                        if stride < span:
                            raise ValueError("K3.1 component dimensions overlap")
                        span += (dimension - 1) * stride
                if component_offset + span > offset + width:
                    raise ValueError("K3.1 component exceeds its native page interval")
                logical = math.prod(shape[1:]) * size
                if state and span != logical:
                    raise ValueError("K3.1 state components must retain native contiguous payloads")
                component_offset += logical
        covered.update(layers)
    if covered != set(by_name):
        raise ValueError("K3.1 native placements omit initialized layers")
