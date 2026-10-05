# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint quantization metadata, without predictor imports or dtype guessing.

Profiles are the coarse GEMM/MoE inputs; exclusions and category algorithms must
also reach the predictor through the same checkpoint and inferred-mode marker.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from e2e_accuracy_source.defaults.sglang_additional_defaults import verified_sglang_fp4_expert_profile
from e2e_accuracy_source.recipes.inferencex_recipe import InferenceXRecipeError


class QuantizationMappingError(InferenceXRecipeError):
    """A parsed checkpoint cannot be represented by the reviewed profile mapping."""

    def __init__(self, message: str, *, kind: str = "unsupported_quantization", details: dict | None = None):
        super().__init__(message)
        self.kind = kind
        self.details = details or {}


def _algo(value: Any) -> str:
    name = str(value or "").lower().replace("-", "_")
    return {"fp4": "nvfp4", "mixedprecision": "mixed_precision", "e4m3": "fp8", "e5m2": "fp8"}.get(name, name)


def _category(target: str) -> str:
    target = target.lower()
    if "shared_expert" in target:
        return "shared_experts"
    if ".experts" in target or "routing_expert" in target:
        return "routed_experts"
    if "self_attn" in target or "linear_attn" in target:
        return "attention"
    if "mlp" in target or "feed_forward" in target:
        return "dense_mlp"
    return "other_gemm"


def _group_algo(group: dict) -> str:
    weights = group.get("weights") or {}
    explicit = weights.get("quant_algo") or weights.get("quantization_algo") or weights.get("quant_method")
    if explicit:
        return _algo(explicit)
    bits, dtype = weights.get("num_bits"), weights.get("type")
    activation = group.get("input_activations")
    if dtype == "int" and bits in (4, 8):
        if activation:
            raise QuantizationMappingError("integer activation quantization needs a separate execution profile")
        return f"int{bits}_wo"
    if dtype == "float" and bits == 8:
        if weights.get("strategy") == "block" or weights.get("block_structure"):
            return "fp8_block"
        return "mxfp8" if weights.get("group_size") == 32 else "fp8"
    if dtype == "float" and bits == 4:
        size = weights.get("group_size")
        if (
            size == 16
            and isinstance(activation, dict)
            and activation.get("num_bits") == 4
            and activation.get("group_size") == 16
        ):
            return "nvfp4"
        if size == 32 and not activation:
            return "w4a16_mxfp4"
        raise QuantizationMappingError(
            "FP4 group requires reviewed scale and activation metadata", details={"group": group}
        )
    raise QuantizationMappingError("unsupported quantization group", details={"group": group})


def _profile(algo: str, *, gemm: bool, dynamic: bool, approximations: list) -> str:
    if algo == "mxfp8":
        note = {
            "source_algorithm": "mxfp8",
            "profile": "fp8_block",
            "difference": "32-element MXFP8 scales versus 128-element FP8 block scales",
        }
        if note not in approximations:
            approximations.append(note)
        return "fp8_block"
    if algo == "fp8":
        return "fp8" if not gemm or dynamic else "fp8_static"
    if algo in {
        "bfloat16",
        "fp8_block",
        "nvfp4",
        "w4a16_nvfp4",
        "w4a16_mxfp4",
        "w4a8_mxfp4_mxfp8",
        "int4_wo",
        "int8_wo",
    }:
        return algo
    raise QuantizationMappingError(f"unsupported quantization algorithm: {algo!r}")


# Reviewed static-checkpoint dispatch; re-audit when adding a release.
_TRT_MXFP4_REVISIONS = {
    "1.2.0rc0.post1": "6632b4051ed2c67482fada3ecdd7aed08bbbce9e",
    "1.3.0rc14": "93cb6518b6d6dbd6095748189e626db731f44545",
    "1.3.0rc18": "15d06c0923b63ac1781784d5f59e1747bb47d5f1",
    "1.3.0rc20": "c25c23f71786bad54d192893d696ce8043426eca",
}


def _trt_mxfp4_profile(version: str | None, hardware: str, args: dict, evidence: dict) -> str:
    revision = _TRT_MXFP4_REVISIONS.get(version)
    if revision is None:
        raise QuantizationMappingError("TRT MXFP4 dispatch requires a reviewed runtime version")
    moe_backend = (args.get("moe_config") or {}).get("backend")
    override = (args.get("recipe_environment") or {}).get("OVERRIDE_QUANT_ALGO")
    if override:
        selected = str(override).lower()
        if selected not in {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8", "w4a8_mxfp4_fp8"}:
            raise QuantizationMappingError(f"unsupported TRT quantization override: {override!r}")
    elif hardware in {"b200", "b300", "gb200", "gb300"}:
        if moe_backend is None:
            raise QuantizationMappingError("TRT MXFP4 dispatch requires the effective MoE backend")
        selected = "w4a8_mxfp4_fp8" if moe_backend == "TRITON" else "w4a8_mxfp4_mxfp8"
    elif hardware in {"h100", "h200"}:
        selected = "w4a16_mxfp4"
    else:
        raise QuantizationMappingError(f"unreviewed TRT MXFP4 hardware: {hardware!r}")
    evidence["runtime_activation_quantization"] = {
        "framework_version": version,
        "hardware": hardware,
        "moe_backend": moe_backend,
        "override": override,
        "effective_algorithm": selected,
        "source": f"https://github.com/NVIDIA/TensorRT-LLM/blob/{revision}/tensorrt_llm/_torch/model_config.py",
        "resolver": "ModelConfig.get_mxfp4_quant_algo (static checkpoint)",
    }
    if selected == "w4a8_mxfp4_fp8":
        raise QuantizationMappingError(
            "TRT TRITON MXFP4/FP8 activations have no matching predictor profile",
            kind="unsupported_predictor_representation",
            details=evidence,
        )
    return selected


def resolve_checkpoint_quantization(
    checkpoint: dict,
    *,
    recipe_method: str | None = None,
    backend: str | None = None,
    hardware: str = "",
    framework_version: str | None = None,
    runtime_args: dict | None = None,
) -> dict:
    """Return profiles plus lossless metadata needed for checkpoint-derived splits.

    ``checkpoint['hf_quant_config']`` may contain the companion file loaded at
    the same resolved revision. Network fetching and content provenance belong
    to the caller. ``approximations`` never means an exact configuration match.
    """
    text = checkpoint.get("text_config") or checkpoint
    cfg = checkpoint.get("quantization_config") or text.get("quantization_config") or {}
    companion = checkpoint.get("hf_quant_config") or {}
    extra = companion.get("quantization") or {}
    if not isinstance(cfg, dict) or not isinstance(extra, dict):
        raise QuantizationMappingError("quantization metadata must be a mapping")
    inline_config = cfg
    sidecar_precedence = backend == "trtllm" and bool(companion)
    if sidecar_precedence:
        # TRT-LLM's ModelConfig.from_pretrained chooses the companion file as
        # a complete declaration. Inline scales, groups and exclusions do not
        # fill gaps in that declaration.
        cfg, extra = dict(extra), {}
    method = _algo(cfg.get("quant_method"))
    if sidecar_precedence and cfg.get("quant_algo") and not method:
        method = "modelopt"
    algorithm = _algo(extra.get("quant_algo") or cfg.get("quant_algo") or method)
    recipe = _algo(recipe_method)
    if recipe == "fbgemm_fp8" and not cfg and not extra:
        method = algorithm = recipe
    # Runtime spellings name a loader, whereas ModelOpt's algorithm names the
    # actual weight/activation format. They are compatible, not conflicting.
    if recipe and recipe not in {method, algorithm}:
        compatible = recipe in {"modelopt", "modelopt_fp4"} and (
            method == "modelopt" or algorithm in {"nvfp4", "mixed_precision"}
        )
        compatible |= recipe == "compressed_tensors" and bool(cfg.get("config_groups"))
        if not compatible:
            raise QuantizationMappingError("recipe and checkpoint quantization methods conflict")
    arch = (checkpoint.get("architectures") or text.get("architectures") or [None])[0]
    exclusions = []
    for section in (cfg, extra):
        for key in ("ignore", "exclude_modules", "modules_to_not_convert", "ignored_layers"):
            exclusions.extend(section.get(key) or [])
    detail = {
        "method": method,
        "algorithm": algorithm,
        "architecture": arch,
        "config": cfg,
        "hf_quant_config": companion,
        "excluded_modules": exclusions,
        "category_algorithms": {},
        "approximations": [],
        "requires_checkpoint_split": False,
        "gemm_profile_is_explicit": False,
        "checkpoint_quantization_precedence": "hf_quant_config.json" if sidecar_precedence else "checkpoint metadata",
        "ignored_inline_quantization_config": inline_config if sidecar_precedence else None,
        "predictor_requires_sidecar_precedence": sidecar_precedence and bool(inline_config),
        "predictor_snapshot_conflict": sidecar_precedence and bool(inline_config),
    }
    dynamic = cfg.get("activation_scheme") == "dynamic"
    approximations = detail["approximations"]
    groups = cfg.get("config_groups") or {}
    categories: dict[str, set[str]] = defaultdict(set)
    if not algorithm:
        dtype = text.get("dtype") or text.get("torch_dtype") or checkpoint.get("dtype") or checkpoint.get("torch_dtype")
        if dtype != "bfloat16":
            raise QuantizationMappingError("unquantized checkpoint dtype is unresolved", details=detail)
        gemm_algo = moe_algo = "bfloat16"
    elif algorithm in {"fp8", "fp8_block", "fbgemm_fp8"}:
        shape = cfg.get("weight_block_size")
        if shape is not None and shape != [128, 128]:
            raise QuantizationMappingError(f"unsupported FP8 block shape: {shape}", details=detail)
        gemm_algo = moe_algo = "fp8_block" if shape else "fp8"
        dynamic |= algorithm == "fbgemm_fp8"
        detail["gemm_profile_is_explicit"] = algorithm == "fbgemm_fp8"
    elif algorithm == "mxfp4":
        if arch != "GptOssForCausalLM":
            raise QuantizationMappingError("MXFP4 expert-only layout needs a reviewed architecture", details=detail)
        gemm_algo, moe_algo = "bfloat16", "w4a16_mxfp4"
        if backend == "trtllm":
            moe_algo = _trt_mxfp4_profile(framework_version, hardware, runtime_args or {}, detail)
    elif algorithm == "mxfp8":
        if cfg.get("weight_block_size") != [1, 32]:
            raise QuantizationMappingError("MXFP8 scale layout is unresolved", details=detail)
        gemm_algo = moe_algo = "mxfp8"
    elif algorithm in {"nvfp4", "w4a16_nvfp4"} and not groups:
        gemm_algo = moe_algo = algorithm
    elif algorithm in {"mixed_precision", "compressed_tensors", "nvfp4"} or groups:
        # Retain every category instead of selecting the first config group.
        for group in groups.values():
            algo = _group_algo(group)
            if algo == "fp8":
                dynamic |= bool((group.get("input_activations") or {}).get("dynamic", False))
            for target in group.get("targets") or []:
                categories["all_linear" if target == "Linear" else _category(target)].add(algo)
        for section in (extra, cfg):
            for target, metadata in (section.get("quantized_layers") or {}).items():
                algo = (
                    _algo(
                        metadata.get("quant_algo") or metadata.get("quant_method") or metadata.get("quantization_algo")
                    )
                    if isinstance(metadata, dict)
                    else _algo(metadata)
                )
                if not algo:
                    raise QuantizationMappingError("quantized layer algorithm is unresolved", details=detail)
                categories[_category(target)].add(algo)
        detail["category_algorithms"] = {k: sorted(v) for k, v in categories.items()}
        if not categories:
            raise QuantizationMappingError("mixed quantization lacks per-group or per-layer metadata", details=detail)
        if any(len(v) != 1 for v in categories.values()):
            raise QuantizationMappingError(
                "different algorithms within one predictor category",
                kind="unsupported_predictor_representation",
                details=detail,
            )
        default = categories.get("all_linear", set())
        dense = set().union(*(v for k, v in categories.items() if k not in {"routed_experts", "all_linear"})) or default
        expert = categories.get("routed_experts", default)
        if len(dense) != 1 or len(expert) != 1:
            raise QuantizationMappingError(
                "mixed GEMM categories need an explicit predictor adapter",
                kind="unsupported_predictor_representation",
                details=detail,
            )
        gemm_algo, moe_algo = next(iter(dense)), next(iter(expert))
        # Kimi-K2.5's INT4 export excludes every non-expert GEMM category.
        if algorithm == "compressed_tensors" and all(
            s in exclusions
            for s in ("re:.*self_attn.*", "re:.*shared_experts.*", "re:.*mlp\\.(gate|up|gate_up|down)_proj.*")
        ):
            gemm_algo = "bfloat16"
    else:
        raise QuantizationMappingError(f"unsupported weight quantization: {algorithm!r}", details=detail)
    if arch == "MiniMaxM2ForCausalLM" and any("self_attn" in p for p in exclusions):
        # Every non-expert projection in this all-MoE architecture is attention.
        # The generic MoE predictor does not consume checkpoint exclusions.
        excluded_layers = {
            int(match.group(1)) for p in exclusions if (match := re.fullmatch(r"model\.layers\.(\d+)\.self_attn\*", p))
        }
        if excluded_layers != set(range(text.get("num_hidden_layers", 0))) or not excluded_layers:
            raise QuantizationMappingError(
                "partial MiniMax-M2 attention exclusions need per-layer predictor support",
                kind="unsupported_predictor_representation",
                details=detail,
            )
        gemm_algo = "bfloat16"
        detail["gemm_profile_is_explicit"] = True
    gemm = _profile(gemm_algo, gemm=True, dynamic=dynamic, approximations=approximations)
    moe = _profile(moe_algo, gemm=False, dynamic=dynamic, approximations=approximations)
    if checkpoint.get("expert_dtype") == "fp4" and moe != "nvfp4":
        mapped, mapping_evidence = (None, {})
        if backend == "sglang":
            mapped, mapping_evidence = verified_sglang_fp4_expert_profile(
                checkpoint, runtime_args or {}, framework_version, hardware
            )
        if mapped is not None:
            moe = mapped
            detail["expert_kernel_mapping"] = mapping_evidence
        elif arch == "DeepseekV4ForCausalLM" and backend == "vllm" and hardware in {"b200", "b300", "gb200", "gb300"}:
            moe = "w4a8_mxfp4_mxfp8"
        else:
            raise QuantizationMappingError(
                "FP4 expert kernel selection needs a reviewed backend/hardware mapping",
                kind="unverified_kernel_mapping",
                details=detail,
            )
    if exclusions or detail["category_algorithms"]:
        detail["requires_checkpoint_split"] = True
    return {"gemm": gemm, "moe": moe, "evidence": detail}
