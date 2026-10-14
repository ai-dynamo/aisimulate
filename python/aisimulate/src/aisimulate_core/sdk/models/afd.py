# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pool-specific composition using agg model operators, without timing math.

A replica is one attention TP group. Shared experts use full weights and
rank-local tokens; dense layers retain ordinary attention TP. The resulting
source-derived plan is validated and evaluated by Rust.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from enum import Enum, auto

import aisimulate_core.sdk.operations as ops
from aisimulate_core.sdk import common
from aisimulate_core.sdk.models.helpers import _get_model_info, _moe_source_config, power_law_distribution


class _OpKind(Enum):
    ROUTED = auto()
    SKIP = auto()
    PREFIX = auto()
    SUFFIX = auto()
    ROUTER = auto()
    SHARED = auto()
    NORM = auto()
    DENSE = auto()
    ATTENTION = auto()


def _prefix_only_layout(layout):
    first_routed = next((i for i, (_, routed, _) in enumerate(layout) if routed), None)
    if first_routed is not None:
        for i in range(first_routed + 1, len(layout)):
            if not layout[i][1]:
                raise NotImplementedError(
                    f"AFD supports only a dense prefix followed by routed MoE layers: "
                    f"dense layer {i} follows routed layer {first_routed}"
                )
    return layout


def layer_layout(info):
    """Ordered (attention kind, routed, DSA indexer) source facts."""
    raw = _moe_source_config(info)
    n = info["layers"]
    extra = info["extra_params"]
    if isinstance(extra, common.HybridMoEConfig):
        if len(extra.moe_layer_freq) != n or len(extra.attn_layer_pattern) != n:
            raise ValueError("AFD source layer pattern length mismatch")
        return _prefix_only_layout(
            [
                ("global" if a else "swa", bool(m), True)
                for a, m in zip(extra.attn_layer_pattern, extra.moe_layer_freq, strict=True)
            ]
        )
    first = int(raw.get("first_k_dense_replace") or 0)
    freq = raw.get("moe_layer_freq", 1)
    if isinstance(freq, (list, tuple)):
        if len(freq) != n or any(type(v) is not int or v not in (0, 1) for v in freq):
            raise ValueError("AFD invalid MoE layer pattern")
        routed = [bool(v) for v in freq]
    else:
        freq = int(freq or 1)
        if not 0 <= first <= n or freq <= 0:
            raise ValueError("AFD invalid dense prefix/MoE frequency")
        routed = [i >= first and i % freq == 0 for i in range(n)]
    kinds = raw.get("layer_types")
    if kinds is not None:
        if len(kinds) != n or any(k not in ("full_attention", "sliding_attention") for k in kinds):
            raise NotImplementedError("AFD cannot partition this attention layer pattern")
        kinds = ["swa" if k == "sliding_attention" else "global" for k in kinds]
    elif info["architecture"] == "GptOssForCausalLM":
        kinds = ["swa" if i % 2 == 0 else "global" for i in range(n)]
    else:
        kinds = ["default"] * n
    index_freq = int(raw.get("index_topk_freq", 1) or 1)
    pattern = raw.get("index_topk_pattern")
    offset = raw.get("index_skip_topk_offset")
    full = [True] * n
    if pattern:
        if len(pattern) != n:
            raise ValueError("AFD DSA index pattern length mismatch")
        full = [v != "S" for v in pattern]
    elif index_freq > 1:
        full = [(max(i - offset + 1, 0) if offset is not None else max(i - 1, 0)) % index_freq == 0 for i in range(n)]
    return _prefix_only_layout(list(zip(kinds, routed, full, strict=True)))


def _flatten(items):
    for op in items:
        if isinstance(op, ops.OverlapOp):
            yield from _flatten(op._group_a)
            yield from _flatten(op._group_b)
        else:
            yield op


def _kind(op) -> _OpKind:
    """Explicit variants first; reject unknown model operations."""
    name = op._name.lower()
    spec = json.loads(op._spec_json())
    while "TokenScale" in spec:
        spec = spec["TokenScale"]["op"]
    if any(kind in spec for kind in ("Moe", "MoeExpertCompute")):
        return _OpKind.ROUTED
    if isinstance(op, (ops.GDNKernel, ops.Mamba2Kernel, ops.KDAKernel)) or any(
        marker in name for marker in ("mamba", "gdn", "kda")
    ):
        raise NotImplementedError(f"AFD cannot partition {name}")
    # Vocabulary-sharded embeddings need their own one-time TP reduction.
    # Layer reductions below are reconstructed by the AFD composition.
    if "embedding" in name:
        if isinstance(op, ops.CustomAllReduce) and spec["CustomAllReduce"]["tp_size"] == 1:
            return _OpKind.SKIP
        return _OpKind.PREFIX
    if "dispatch" in name or isinstance(op, (ops.CustomAllReduce, ops.NCCL, ops.P2P, ops.MoEDispatch, ops.MoEAllToAll)):
        return _OpKind.SKIP
    if "logits_gemm" in name:
        return _OpKind.SUFFIX
    if "router" in name:
        return _OpKind.ROUTER
    if any(s in name for s in ("reduce_add", "shared_merge")) or name.endswith("_combine"):
        return _OpKind.SKIP  # Explicit return-side combine replaces this operation.
    if "shared" in name:
        return _OpKind.SHARED
    if any(s in name for s in ("add_norm_2", "ffn_norm", "moe_norm", "mlp_norm")):
        return _OpKind.NORM
    if "dense" in name and not any(s in name for s in ("attn", "qkv", "proj_gemm")):
        return _OpKind.DENSE
    if any(
        s in name
        for s in (
            "ffn",
            "act_gate",
            "_act",
            "down_gemm",
            "gate_up",
            "up_gemm",
            "mlp",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    ):
        return _OpKind.DENSE
    if any(
        s in name
        for s in ("attention", "add_norm_1", "attn", "qkv", "mla", "bmm", "rope", "proj_gemm", "layernorm", "kv_b_proj")
    ):
        return _OpKind.ATTENTION
    raise NotImplementedError(f"AFD has no semantic composition rule for {name}")


def _clone(op, count, *, full_indexer=True):
    if isinstance(op, ops.FallbackOp):
        args, kwargs = op.__getnewargs_ex__()
        return type(op)(
            args[0],
            _clone(args[1], count, full_indexer=full_indexer),
            [_clone(child, count, full_indexer=full_indexer) for child in args[2]],
            **kwargs,
        )
    # Native constructors preserve the complete operator identity on copy.
    if isinstance(op, (ops.ContextDSAModule, ops.GenerationDSAModule)):
        args, kwargs = op.__getnewargs_ex__()
        kwargs["dsa_full_layer_fraction"] = 1.0 if full_indexer else 0.0
        op = type(op)(*args, **kwargs)
    else:
        op = copy.copy(op)
    op._scale_factor /= count
    return op


def _shared_copy(op, inter, hidden, count):
    args, kwargs = op.__getnewargs_ex__()
    args = list(args)
    args[1] /= count
    kwargs["scale_num_tokens"] = 1  # Rust queries the exact rank-local slice.
    variant = next(iter(json.loads(op._spec_json())))
    if variant == "Gemm":
        if "gate_up" in op._name or "ffn1" in op._name:
            args[2:4] = [2 * inter, hidden]
        elif "down" in op._name or "ffn2" in op._name:
            args[2:4] = [hidden, inter]
    elif variant == "Elementwise" and "act" in op._name:
        args[2:4] = [2 * inter, inter]
    return type(op)(*args, **kwargs)


def build_afd_model(model_path, config, backend, role):
    from aisimulate_core.sdk.models import get_model

    if role not in ("a", "f") or backend != "sglang":
        raise ValueError("AFD pool compilation requires role a/f and SGLang")
    info = _get_model_info(model_path)
    if info["architecture"] == "DeepseekV41ForCausalLM":
        raise NotImplementedError("AFD does not support DeepSeek-V4.1")
    layout = layer_layout(info)
    if sum(m for _, m, _ in layout) != info["num_moe_layers"] or not any(m for _, m, _ in layout):
        raise ValueError("AFD source layer composition disagrees with MoE layer count")
    # Build an agg reference with safe *unused* expert/attention geometry. F
    # never validates attention heads against its expert-pool GPU width.
    reference_tp = config.tp_size if role == "a" else 1
    reference_cfg = dataclasses.replace(
        config,
        tp_size=reference_tp,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=reference_tp,
        moe_comm_backend=None,
    )
    model = get_model(model_path, reference_cfg, backend)
    if role == "a" and hasattr(model, "_resolve_dims"):
        dimensions = model._resolve_dims(reference_tp)
        for key, value in dimensions.items():
            if key.endswith("_n_q") and value % reference_tp:
                raise ValueError("AFD attention head groups must divide attention TP")
    model.config = dataclasses.replace(
        model.config,
        tp_size=config.tp_size,
        attention_dp_size=config.attention_dp_size,
        moe_tp_size=config.moe_tp_size,
        moe_ep_size=config.moe_ep_size,
    )
    raw = _moe_source_config(info)
    inter = int(
        raw.get("shared_expert_intermediate_size")
        or raw.get("moe_shared_expert_intermediate_size")
        or info["num_shared_experts"] * info["moe_inter_size"]
    )
    # Source exclusions, rather than resolved hardware dtype, determine the
    # template grouping so heterogeneous A/F systems keep the same layer IDs.
    hybrid = isinstance(info["extra_params"], common.HybridMoEConfig)
    dense_modes = [(model.config.gemm_quant_mode, model.config.gemm_quant_mode)] * info["layers"]
    dense_tags = [0] * info["layers"]
    if not hybrid and hasattr(model, "_dense_mlp_ops") and any(not m for _, m, _ in layout):
        from aisimulate_core.sdk.models.deepseek_v32 import _dense_mlp_groups

        quant_source = {**raw, "first_k_dense_replace": info["layers"], "moe_layer_freq": 1}
        grouped = _dense_mlp_groups(quant_source, info["layers"], model.config.gemm_quant_mode, ordered=True)
        dense_modes = [pair for count, *pair in grouped for _ in range(count)]
        source_groups = _dense_mlp_groups(quant_source, info["layers"], common.GEMMQuantMode.fp8, ordered=True)
        source_pairs = list(dict.fromkeys((gate, down) for _, gate, down in source_groups))
        dense_tags = [source_pairs.index((gate, down)) for count, gate, down in source_groups for _ in range(count)]
    layout = [(*key, dense_tags[i] if not key[1] else 0) for i, key in enumerate(layout)]
    if info.get("architecture") == "Gemma4ForConditionalGeneration":
        inter = info["inter_size"]
    keys = list(dict.fromkeys(layout))
    plan = {
        "role": role,
        "model": model_path,
        "hidden_size": info["hidden_size"],
        "num_experts": info["num_experts"],
        "topk": info["topk"],
        "layers": [keys.index(k) for k in layout],
        "shared_inter_size": inter,
    }
    # Hybrid agg groups are rebuilt one type at a time so unnamed attention
    # ops retain the correct SWA/global dimensions and window.
    original = {"context": list(model.context_ops), "generation": list(model.generation_ops)}
    for phase, field in (("context", "context_ops"), ("generation", "generation_ops")):
        emitted, templates, prefix, suffix = [], [], [], []
        gated = True
        routed_quant = model.config.moe_quant_mode
        routed_distribution = power_law_distribution(
            model.config.workload_distribution, getattr(model, "_power_law_alpha", 1.2)
        )
        for source_op in _flatten(original[phase]):
            spec = json.loads(source_op._spec_json())
            while "TokenScale" in spec:
                spec = spec["TokenScale"]["op"]
            if "Moe" in spec:
                gated = spec["Moe"]["is_gated"]
                # Checkpoint exclusions can override the global model mode.
                # Preserve the precision and distribution of the actual agg op.
                routed_quant = getattr(common.MoEQuantMode, spec["Moe"]["quant_mode"])
                routed_distribution = spec["Moe"]["workload_distribution"]
                break
        for ti, (attention, routed, indexer, dense_tag) in enumerate(keys):
            source = model
            if hybrid:
                source = copy.copy(model)
                source._num_layers = 1
                source.set_hybrid_config(
                    dataclasses.replace(
                        info["extra_params"],
                        attn_layer_pattern=(int(attention == "global"),),
                        moe_layer_freq=(int(routed),),
                    )
                )
                source_items = getattr(source, field)
                denominator = 1
            else:
                source_items = original[phase]
                denominator = info["layers"]
            bucket, rank_bucket, post_attention = [], [], []
            for op in _flatten(source_items):
                name = op._name.lower()
                if (
                    not hybrid
                    and attention in ("global", "swa")
                    and (("_swa_" in name and attention != "swa") or ("_global_" in name and attention != "global"))
                ):
                    continue
                kind = _kind(op)
                if kind in (_OpKind.SKIP, _OpKind.ROUTED, _OpKind.DENSE):
                    continue
                if kind in (_OpKind.PREFIX, _OpKind.SUFFIX):
                    if role == "a" and ti == 0:
                        (prefix if kind is _OpKind.PREFIX else suffix).append(len(emitted))
                        emitted.append(copy.copy(op))
                    continue
                if role == "f":
                    continue
                if kind is _OpKind.NORM:
                    norm_count = (
                        sum(1 for a, *_ in layout if a == attention)
                        if not hybrid and ("_swa_" in name or "_global_" in name)
                        else denominator
                    )
                    post_attention.append(_clone(op, norm_count))
                    continue
                if kind in (_OpKind.SHARED, _OpKind.ROUTER):
                    if kind is _OpKind.SHARED and not inter:
                        raise ValueError("AFD shared branch has no source-derived intermediate size")
                    if not routed:
                        continue
                    rank_bucket.append(len(emitted))
                    count = denominator
                    if kind is _OpKind.SHARED:
                        count = op._scale_factor / (
                            getattr(model, "_mtp_scale_factor", 1.0) if phase == "generation" else 1.0
                        )
                        emitted.append(_shared_copy(op, inter, info["hidden_size"], count))
                    else:
                        count = op._scale_factor / (
                            getattr(model, "_mtp_scale_factor", 1.0) if phase == "generation" else 1.0
                        )
                        emitted.append(_clone(op, count))
                else:
                    if kind is _OpKind.ATTENTION and attention in ("global", "swa") and not hybrid:
                        # GPT-OSS grouped attention operators: the shared
                        # projections/norms occur on both layer kinds.
                        spec = json.loads(op._spec_json())
                        attention_spec = spec.get("ContextAttention") or spec.get("GenerationAttention")
                        if attention_spec is not None:
                            is_swa = bool(attention_spec["window_size"])
                            if is_swa != (attention == "swa"):
                                continue
                            denominator_op = sum(1 for a, *_ in layout if a == attention)
                        else:
                            denominator_op = denominator
                    else:
                        denominator_op = denominator
                    if not hybrid and ("_swa_" in name or "_global_" in name):
                        denominator_op = sum(1 for a, *_ in layout if a == attention)
                    bucket.append(len(emitted))
                    emitted.append(_clone(op, denominator_op, full_indexer=indexer))
            work_scale = getattr(model, "_mtp_scale_factor", 1.0) if phase == "generation" else 1.0
            if role == "a":
                if reference_tp > 1:
                    bucket.append(len(emitted))
                    emitted.append(
                        ops.CustomAllReduce(f"{phase}_afd_attn_ar", work_scale, info["hidden_size"], reference_tp)
                    )
                # Residual/FFN normalization consumes the reduced full
                # hidden, before rank-local routing/shared or dense compute.
                for op in post_attention:
                    bucket.append(len(emitted))
                    emitted.append(op)
                if not routed:
                    if hybrid:
                        dense = source._dense_ffn_ops(
                            f"{phase}_afd_dense_{ti}",
                            work_scale,
                            info["hidden_size"],
                            reference_tp,
                            source._resolve_dims(reference_tp)["dense_inter_per_tp"],
                            model.config.gemm_quant_mode,
                        )
                    else:
                        dense_inter = int(raw.get("intermediate_size") or info["inter_size"])
                        if dense_inter % reference_tp:
                            raise ValueError("AFD dense intermediate must divide attention TP")
                        layer = next(i for i, k in enumerate(layout) if k == keys[ti])
                        iq, down_q = dense_modes[layer]
                        d = dense_inter // reference_tp
                        dense = [
                            ops.GEMM(f"{phase}_afd_dense_gate_up", work_scale, 2 * d, info["hidden_size"], iq),
                            ops.ElementWise(f"{phase}_afd_dense_act", work_scale, 2 * d, d, 0.8),
                            ops.GEMM(
                                f"{phase}_afd_dense_down",
                                work_scale,
                                info["hidden_size"],
                                d,
                                down_q,
                                low_precision_input=True,
                            ),
                        ]
                    for op in dense:
                        bucket.append(len(emitted))
                        emitted.append(op)
                    if reference_tp > 1:
                        bucket.append(len(emitted))
                        emitted.append(
                            ops.CustomAllReduce(f"{phase}_afd_dense_ar", work_scale, info["hidden_size"], reference_tp)
                        )
            elif routed:
                bucket.append(len(emitted))
                emitted.append(
                    ops.MoE(
                        f"{phase}_afd_routed",
                        work_scale,
                        info["hidden_size"],
                        info["moe_inter_size"],
                        info["topk"],
                        info["num_experts"],
                        config.moe_tp_size,
                        config.moe_ep_size,
                        routed_quant,
                        routed_distribution,
                        1,
                        is_context=phase == "context",
                        is_gated=gated,
                        moe_kernel_source=config.moe_kernel_source,
                    )
                )
            if inter and routed and role == "a" and not any("shared" in emitted[i]._name for i in rank_bucket):
                raise NotImplementedError("AFD model descriptor declares shared experts but agg has no shared branch")
            templates.append(
                {
                    "attention_kind": attention + ("_index" if indexer else "_reuse") + f"_q{dense_tag}",
                    "routed": routed,
                    "ops": bucket,
                    "rank_ops": rank_bucket,
                    "work_scale": work_scale,
                }
            )
        if role == "a" and phase == "context" and not suffix:
            # The first generated token requires the same head as decode.
            # Some agg families omit it from the context approximation.
            for op in _flatten(original["generation"]):
                if _kind(op) is _OpKind.SUFFIX:
                    head = copy.copy(op)
                    head._name = head._name.replace("generation", "context", 1)
                    head._scale_factor = 1.0
                    suffix.append(len(emitted))
                    emitted.append(head)
        # Intern identical agg operators across layer types. Runtime queries
        # each index once per shape, while layer occurrences retain their
        # distinct physical weights and execution count.
        unique, indices, remap = [], {}, []
        for op in emitted:
            key = op._spec_json()
            if key not in indices:
                indices[key] = len(unique)
                unique.append(op)
            remap.append(indices[key])
        for template in templates:
            for bucket in ("ops", "rank_ops"):
                template[bucket] = [remap[i] for i in template[bucket]]
        prefix = [remap[i] for i in prefix]
        suffix = [remap[i] for i in suffix]
        setattr(model, field, unique)
        plan[phase] = {"templates": templates, "prefix": prefix, "suffix": suffix}
    model.afd_plan = plan
    return model
