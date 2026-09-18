# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local config facts and explicitly bounded resource estimates for onboarding.

This module reads JSON and packaged hardware metadata. It never instantiates a
model, imports remote configuration code, or builds an operation graph. Tensor
storage estimates cover the declared vanilla decoder layouts, not arbitrary
implementations sharing their field names. Runtime reservations and activation
estimates are not measurements or a guarantee that a deployment fits.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from aisimulate import quantization
from aisimulate.fpm_profile import FpmModelProfile

from .schema import SupportRequest

_RESOURCE_FIELDS = (
    "weights_bytes",
    "activations_bytes",
    "runtime_overhead_bytes",
    "comm_overhead_bytes",
    "kv_bytes_per_token",
    "cache_layout",
    "max_num_tokens",
    "max_batch_size",
)
_MODE_ENUMS = {
    "gemm_quant_mode": quantization.GEMMQuantMode,
    "moe_quant_mode": quantization.MoEQuantMode,
    "fmha_quant_mode": quantization.FMHAQuantMode,
    "comm_quant_mode": quantization.CommQuantMode,
    "kv_cache_dtype": quantization.KVCacheQuantMode,
}
_DEPLOYMENT_FIELDS = (*_MODE_ENUMS, "moe_backend", "attention_backend")
_MODEL_FIELDS = ("architecture", "context_length", "num_experts")
OVERRIDE_FIELDS = frozenset((*_MODEL_FIELDS, *_RESOURCE_FIELDS, *_DEPLOYMENT_FIELDS, "provenance"))
INTEGER_FIELDS = frozenset(
    ("context_length", "num_experts", *(field for field in _RESOURCE_FIELDS if field != "cache_layout"))
)
FIELD_CHOICES = {field: tuple(enum.__members__) for field, enum in _MODE_ENUMS.items()}
FIELD_CHOICES["cache_layout"] = ("linear",)
_NONNEGATIVE_FIELDS = frozenset(
    ("num_experts", "weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes")
)
_BYTE_FIELDS = frozenset(field for field in INTEGER_FIELDS if "bytes" in field)

_ARCHITECTURES = {
    "llama": "LlamaForCausalLM",
    "mistral": "MistralForCausalLM",
    "mixtral": "MixtralForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "qwen3_moe": "Qwen3MoeForCausalLM",
    "minimax_m2": "MiniMaxM2ForCausalLM",
    "glm_moe_dsa": "GlmMoeDsaForCausalLM",
    "deepseek_v3": "DeepseekV3ForCausalLM",
    "deepseek_v32": "DeepseekV32ForCausalLM",
}
_DENSE = frozenset(("LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"))
_FULL_ATTENTION = _DENSE | {"MixtralForCausalLM", "Qwen3MoeForCausalLM", "MiniMaxM2ForCausalLM"}
_WEIGHT_LAYOUTS = _DENSE | {"MixtralForCausalLM"}
_EXPERT_ALIASES = ("num_local_experts", "n_routed_experts", "num_experts")
_INT_ALIASES = {
    "hidden_size": ("hidden_size", "n_embd", "d_model"),
    "intermediate_size": ("intermediate_size",),
    "num_hidden_layers": ("num_hidden_layers", "n_layer", "num_layers"),
    "num_attention_heads": ("num_attention_heads", "n_head"),
    "num_key_value_heads": ("num_key_value_heads", "num_kv_heads"),
    "head_dim": ("head_dim",),
    "vocab_size": ("vocab_size",),
    "context_length": ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "model_max_length"),
    "num_experts": _EXPERT_ALIASES,
}
_OPTIONAL_DIMENSIONS = frozenset(("num_key_value_heads", "head_dim", "intermediate_size"))
_EXTRA_DIMENSIONS = (
    "moe_intermediate_size",
    "num_experts_per_tok",
    "kv_lora_rank",
    "q_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "index_head_dim",
    "index_n_heads",
)
_AUXILIARY_DIMENSIONS = (
    "n_shared_experts",
    "shared_intermediate_size",
    "shared_expert_intermediate_size",
    "num_nextn_predict_layers",
    "num_mtp_modules",
    "mtp_transformer_layers",
    "first_k_dense_replace",
)


@dataclass(frozen=True)
class ModelConfig:
    raw: dict[str, Any]
    # Checkpoint/collector identity in suggestions may name a multimodal wrapper.
    decoder_architecture: str | None
    sha256: str
    suggestions: dict[str, Any]
    notes: dict[str, str]


@dataclass(frozen=True)
class ProfileDraft:
    profile: FpmModelProfile | None
    resolved: dict[str, Any]
    missing: dict[str, str]
    sources: dict[str, str]


class ProfileRequestError(ValueError):
    """Config facts conflict with a correctable onboarding request field."""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


def _integer(value: Any, field: str, *, minimum: int = 1, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise ValueError(f"{field} must be an integer {bound}, not a boolean, float or numeric string")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be a nonempty string without NUL characters")
    return value.strip()


def _aliased_int(raw: Mapping[str, Any], name: str) -> int | None:
    supplied = {}
    for key in _INT_ALIASES[name]:
        if key not in raw or (raw[key] is None and name in _OPTIONAL_DIMENSIONS):
            continue
        supplied[key] = _integer(raw[key], key, minimum=0 if name == "num_experts" else 1)
    if len(set(supplied.values())) > 1:
        raise ValueError(f"conflicting {name} fields: {supplied}")
    return next(iter(supplied.values()), None)


def _dtype(raw: Mapping[str, Any]) -> str | None:
    values = {
        key: _text(raw[key], key).removeprefix("torch.") for key in ("dtype", "torch_dtype") if raw.get(key) is not None
    }
    if len(set(values.values())) > 1:
        raise ValueError(f"conflicting dtype fields: {values}")
    return next(iter(values.values()), None)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate config JSON key: {key}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("config JSON numbers must be finite")
    return result


def _architecture(raw: Mapping[str, Any]) -> str | None:
    architectures = raw.get("architectures")
    architecture = None
    if architectures is not None:
        if not isinstance(architectures, list) or len(architectures) != 1:
            raise ValueError("architectures must identify exactly one decoder architecture")
        architecture = _text(architectures[0], "architectures[0]")
    model_type = _text(raw["model_type"], "model_type") if "model_type" in raw else None
    expected = _ARCHITECTURES.get(model_type)
    if expected and architecture and expected != architecture:
        raise ValueError(f"model_type={model_type!r} conflicts with architectures={architecture!r}")
    return architecture or expected


def _kv_cache_scheme(quant: Mapping[str, Any]) -> dict[str, Any]:
    scheme = quant.get("kv_cache_scheme")
    if scheme is None:
        return {}
    if isinstance(scheme, str):
        if scheme.strip().upper() == "FP8":
            return {"num_bits": 8, "type": "float"}
        raise ValueError("unsupported quantization_config.kv_cache_scheme string; supported value is FP8")
    if not isinstance(scheme, dict):
        raise ValueError("quantization_config.kv_cache_scheme must be an object or FP8")
    return scheme


def _validate_quantization(raw: Mapping[str, Any]) -> None:
    quant = raw.get("quantization_config")
    if quant is None:
        return
    if not isinstance(quant, dict):
        raise ValueError("quantization_config must be an object")
    for key in ("quant_method", "quant_algo", "activation_scheme"):
        if key in quant:
            _text(quant[key], f"quantization_config.{key}")
    if quant.get("quant_method") == "fp8" and quant.get("quant_algo") not in (None, "FP8"):
        raise ValueError("conflicting quantization_config quant_method and quant_algo")
    blocks = quant.get("weight_block_size")
    if blocks is not None:
        if not isinstance(blocks, list) or len(blocks) != 2:
            raise ValueError("quantization_config.weight_block_size must contain two positive integers")
        for block in blocks:
            _integer(block, "quantization_config.weight_block_size")
    for key in ("ignore", "modules_to_not_convert"):
        if key in quant:
            if not isinstance(quant[key], list):
                raise ValueError(f"quantization_config.{key} must be a list of module names")
            for name in quant[key]:
                _text(name, f"quantization_config.{key}")
    if quant.get("config_groups") is not None and not isinstance(quant["config_groups"], dict):
        raise ValueError("quantization_config.config_groups must be an object")
    for group_name, group in (quant.get("config_groups") or {}).items():
        if not isinstance(group, dict):
            raise ValueError(f"quantization_config.config_groups.{group_name} must be an object")
        for key in ("input_activations", "weights"):
            tensor = group.get(key)
            if tensor is None:
                continue
            if not isinstance(tensor, dict):
                raise ValueError(f"quantization_config.config_groups.{group_name}.{key} must be an object")
            if "dynamic" in tensor and type(tensor["dynamic"]) is not bool:
                raise ValueError(f"quantization_config.config_groups.{group_name}.{key}.dynamic must be a boolean")
            if "num_bits" in tensor:
                _integer(tensor["num_bits"], f"quantization_config.config_groups.{group_name}.{key}.num_bits")
    scheme = _kv_cache_scheme(quant)
    if scheme.get("type") is not None:
        _text(scheme["type"], "quantization_config.kv_cache_scheme.type")
    if "num_bits" in scheme:
        _integer(scheme["num_bits"], "quantization_config.kv_cache_scheme.num_bits")
    if "dynamic" in scheme and type(scheme["dynamic"]) is not bool:
        raise ValueError("quantization_config.kv_cache_scheme.dynamic must be a boolean")


def _validate_layout(raw: Mapping[str, Any], architecture: str | None) -> None:
    for key in ("attention_bias", "mlp_bias", "tie_word_embeddings", "is_encoder_decoder", "use_sliding_window"):
        if key in raw and type(raw[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if raw.get("is_encoder_decoder"):
        raise ValueError("unsupported encoder-decoder text layout; supply a decoder-only config or FPM profile")
    if any(raw.get(key) is not None for key in ("ssm_cfg", "mamba_d_state", "linear_conv_kernel_dim")) or any(
        name in (architecture or "").lower() for name in ("mamba", "jamba", "rwkv")
    ):
        raise ValueError("unsupported recurrent/hybrid state cannot be represented by this linear-cache FPM profile")
    window = raw.get("sliding_window")
    if window is not None:
        _integer(window, "sliding_window", minimum=0)
        if window and raw.get("use_sliding_window") is not False:
            raise ValueError("unsupported sliding-window cache; the FPM profile requires a full linear cache")
    layers = _aliased_int(raw, "num_hidden_layers")
    for key in ("layer_types", "attn_type_list"):
        if key not in raw:
            continue
        values = raw[key]
        if not isinstance(values, list) or (layers is not None and len(values) != layers):
            raise ValueError(f"{key} must have one entry per num_hidden_layers")
        expected = "full_attention" if key == "layer_types" else 1
        allowed = {expected}
        if key == "layer_types" and architecture in {"GlmMoeDsaForCausalLM", "DeepseekV32ForCausalLM"}:
            allowed.add("deepseek_sparse_attention")
        if any(type(value) is not type(expected) or value not in allowed for value in values):
            raise ValueError(f"unsupported mixed or non-full attention layout in {key}")
    if architecture in _FULL_ATTENTION and any(
        raw.get(key) for key in ("kv_lora_rank", "q_lora_rank", "compress_ratios", "attention_k_eq_v")
    ):
        raise ValueError("unsupported cache/projection modifiers conflict with the declared full-attention layout")


def _validate_architecture(raw: Mapping[str, Any], architecture: str | None, experts: int | None) -> None:
    _validate_layout(raw, architecture)
    experts_per_token = raw.get("num_experts_per_tok")
    if experts and experts_per_token is not None and experts_per_token > experts:
        raise ValueError("num_experts_per_tok cannot exceed the routed expert count")
    if architecture in _DENSE and experts not in (None, 0):
        raise ValueError("num_experts conflicts with the declared dense architecture")
    if architecture in set(_ARCHITECTURES.values()) - _DENSE and experts == 0:
        raise ValueError("num_experts must be positive for the declared MoE architecture")


def load_model_config(path: str | Path) -> ModelConfig:
    """Load a local JSON document, preserving its byte-level SHA-256 provenance."""
    payload = Path(path).expanduser().read_bytes()
    raw = json.loads(payload, object_pairs_hook=_unique_object, parse_float=_finite_float, parse_constant=_finite_float)
    if not isinstance(raw, dict):
        raise ValueError("model config JSON must be an object")
    document = raw
    notes = {
        "modeling_scope": (
            "Text decoder only. FPM excludes multimodal encoders, projectors, preprocessing and other non-text "
            "components and their resource costs. Full multimodal deployment memory and latency are not modeled."
        )
    }
    if "text_config" in document:
        text_config = document["text_config"]
        if not isinstance(text_config, dict) or not text_config:
            raise ValueError(
                "text_config must be a nonempty object describing one text decoder; "
                "supply its decoder config or a complete --fpm-profile"
            )
        if "text_config" in text_config:
            raise ValueError("ambiguous nested text_config; supply one text decoder config or a complete --fpm-profile")
        raw = dict(text_config)
        notes["decoder_config"] = "config text_config; outer model and encoder geometry are excluded"
        inherited = []
        for keys in (("dtype", "torch_dtype"), ("quantization_config", "hf_quant_config", "quant_algo")):
            if any(raw.get(key) is not None for key in keys):
                continue
            for key in keys:
                if document.get(key) is not None:
                    raw[key] = document[key]
                    inherited.append(key)
        if inherited:
            notes["shared_metadata"] = (
                f"config-level {', '.join(inherited)} inherited because text_config does not declare that metadata"
            )
    geometry = {key: _aliased_int(raw, key) for key in _INT_ALIASES}
    for key in _EXTRA_DIMENSIONS:
        if raw.get(key) is not None:
            _integer(raw[key], key)
    for key in _AUXILIARY_DIMENSIONS:
        if raw.get(key) is not None:
            _integer(raw[key], key, minimum=0)
    decoder_architecture = _architecture(raw)
    architecture = decoder_architecture
    if raw is not document:
        if raw.get("architectures") is None and (wrapper_architecture := _architecture(document)):
            architecture = wrapper_architecture
            notes["architecture"] = (
                f"config wrapper architecture={architecture} retained for checkpoint/collector identity"
            )
        notes["decoder_architecture"] = (
            f"text_config decoder architecture={decoder_architecture}; used for decoder validation/resource estimates"
            if decoder_architecture
            else "text_config does not identify a decoder architecture; decoder resource bounds require explicit input"
        )
    experts = geometry["num_experts"]
    _validate_architecture(raw, decoder_architecture, experts)
    _validate_quantization(raw)
    dtype = _dtype(raw)
    heads, kv_heads = geometry["num_attention_heads"], geometry["num_key_value_heads"]
    if heads and kv_heads and (kv_heads > heads or heads % kv_heads):
        raise ValueError("num_key_value_heads must divide num_attention_heads and cannot exceed it")
    dense_layers = raw.get("first_k_dense_replace")
    if geometry["num_hidden_layers"] and dense_layers is not None and dense_layers > geometry["num_hidden_layers"]:
        raise ValueError("first_k_dense_replace cannot exceed num_hidden_layers")
    if decoder_architecture in _DENSE:
        experts = 0
    suggestions: dict[str, Any] = {}
    notes.update({key: f"config {key}={value}" for key, value in geometry.items() if value is not None})
    for key in (*_EXTRA_DIMENSIONS, *_AUXILIARY_DIMENSIONS):
        if raw.get(key) is not None:
            notes[key] = f"config {key}={raw[key]}"
    names = {_text(document[key], key) for key in ("_name_or_path", "name_or_path") if document.get(key)}
    if len(names) > 1:
        raise ValueError("conflicting _name_or_path and name_or_path identity hints")
    if names:
        suggestions["model"] = names.pop()
        notes["model"] = "config _name_or_path/name_or_path hint; explicit deployment identity takes precedence"
    for key, value in (
        ("architecture", architecture),
        ("context_length", geometry["context_length"]),
        ("num_experts", experts),
    ):
        if value is not None:
            suggestions[key] = value
            notes.setdefault(
                key, f"config declared {key}={value}" if key != "num_experts" else "known dense architecture"
            )
    if experts is not None:
        suggestions["model_kind"] = "moe" if experts else "dense"
    if dtype:
        notes["dtype"] = (
            f"config tensor dtype={dtype}; checkpoint storage still needs verification; "
            "does not declare runtime attention or KV-cache precision"
        )
    if raw.get("auto_map") or document.get("auto_map"):
        notes["auto_map"] = "remote-code references present; read as metadata only and never executed"
    if raw.get("quantization_config") is not None:
        quant = raw["quantization_config"]
        summary = {
            key: quant[key]
            for key in ("quant_method", "quant_algo", "weight_block_size", "kv_cache_scheme")
            if key in quant
        }
        notes["quantization"] = (
            f"config quantization={json.dumps(summary, sort_keys=True)}; "
            "tensor scales/exclusions require explicit weight bytes"
        )
    for key in ("num_nextn_predict_layers", "num_mtp_modules", "mtp_transformer_layers", "use_mtp"):
        if key in raw:
            notes[key] = f"config {key}={raw[key]}; checkpoint capability does not enable runtime speculative decoding"
    return ModelConfig(raw, decoder_architecture, hashlib.sha256(payload).hexdigest(), suggestions, notes)


def validate_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate supplied values independently of a complete derived profile."""
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise ValueError("resource overrides must be a flat object")
    unknown = set(overrides) - OVERRIDE_FIELDS
    if unknown:
        raise ValueError(f"unknown resource override fields: {', '.join(sorted(map(str, unknown)))}")
    result = {}
    for key, value in overrides.items():
        if key in INTEGER_FIELDS:
            value = _integer(
                value,
                key,
                minimum=0 if key in _NONNEGATIVE_FIELDS else 1,
                maximum=2**53 if key in _BYTE_FIELDS else None,
            )
        else:
            value = _text(value, key)
            if key in FIELD_CHOICES and value not in FIELD_CHOICES[key]:
                raise ValueError(f"unknown {key}={value!r}; choose {', '.join(FIELD_CHOICES[key])}")
        result[key] = value
    if sum(result.get(key, 0) for key in _BYTE_FIELDS - {"kv_bytes_per_token"}) > 2**53:
        raise ValueError("total non-KV resource bytes must not exceed 2**53")
    return result


def _precision_facts(config: ModelConfig) -> dict[str, tuple[str, str]]:
    raw = config.raw
    quant = raw.get("quantization_config")
    if quant is None and (raw.get("hf_quant_config") is not None or raw.get("quant_algo") is not None):
        return {}  # Unrecognized sidecar/processed metadata must not become unquantized defaults.
    if quant is None:
        if _dtype(raw) == "bfloat16":
            return dict.fromkeys(
                ("gemm_quant_mode", "moe_quant_mode"),
                (
                    "bfloat16",
                    "assumption: bfloat16 tensor dtype with no inline quantization metadata; "
                    "verify deployed weight precision (sidecars and checkpoint tensors are not inspected)",
                ),
            )
        return {}
    result = {}
    method = str(quant.get("quant_method", "")).lower()
    algorithm = str(quant.get("quant_algo", method)).lower()
    source = (
        "config checkpoint/FPM quantization label, matching sdk/models/helpers.py; "
        "exclusions and tensor scales prevent inferring all tensors' storage from this label"
    )
    if algorithm == "fp8" and method in {"fp8", "modelopt", ""}:
        if quant.get("weight_block_size") == [128, 128]:
            result = dict.fromkeys(("gemm_quant_mode", "moe_quant_mode"), ("fp8_block", source))
        elif not quant.get("weight_block_size"):
            scheme = quant.get("activation_scheme")
            dynamic = scheme == "dynamic" or (
                scheme is None
                and any(
                    (group.get(key) or {}).get("dynamic") is True
                    for group in (quant.get("config_groups") or {}).values()
                    for key in ("input_activations", "weights")
                )
            )
            if scheme in (None, "static", "dynamic"):
                result["gemm_quant_mode"] = "fp8" if dynamic else "fp8_static", source
            result["moe_quant_mode"] = "fp8", source
    elif algorithm == "nvfp4" and method in {"nvfp4", "modelopt", ""}:
        result = dict.fromkeys(("gemm_quant_mode", "moe_quant_mode"), ("nvfp4", source))
    scheme = _kv_cache_scheme(quant)
    if scheme.get("num_bits") == 8 and scheme.get("type") in {"float", "int"}:
        result["kv_cache_dtype"] = (
            "fp8" if scheme["type"] == "float" else "int8",
            "explicit config quantization_config.kv_cache_scheme (independent of weight quantization)",
        )
    return result


def _geometry(config: ModelConfig, architecture: str | None, request: SupportRequest | None) -> dict[str, int]:
    dimensions = {key: value for key in _INT_ALIASES if (value := _aliased_int(config.raw, key)) is not None}
    if architecture not in _FULL_ATTENTION:
        return dimensions
    heads = dimensions.get("num_attention_heads")
    hidden = dimensions.get("hidden_size")
    if "head_dim" not in dimensions and heads and hidden:
        if hidden % heads:
            raise ValueError("hidden_size must divide into attention heads when head_dim is absent")
        dimensions["head_dim"] = hidden // heads
    if request is not None and heads:
        tp = request.search.tensor_parallel
        if heads % tp:
            raise ProfileRequestError(
                "tensor_parallel", "num_attention_heads must divide into the selected TP attention heads"
            )
        kv = dimensions.get("num_key_value_heads")
        if kv and ((kv >= tp and kv % tp) or (kv < tp and tp % kv)):
            raise ProfileRequestError(
                "tensor_parallel", "num_key_value_heads cannot be evenly sharded or replicated across the selected TP"
            )
        if kv:
            dimensions["local_kv_heads"] = max(1, kv // tp)
        intermediate = dimensions.get("intermediate_size")
        mlp_tp = request.parallelism()["moe_tensor"] if architecture == "MixtralForCausalLM" else tp
        if architecture in _WEIGHT_LAYOUTS and intermediate and intermediate % mlp_tp:
            raise ProfileRequestError(
                "moe_tensor_parallel" if architecture == "MixtralForCausalLM" else "tensor_parallel",
                "intermediate_size must be divisible by the selected tensor parallel dimension",
            )
    return dimensions


def _weight_estimate(
    config: ModelConfig,
    request: SupportRequest | None,
    values: dict[str, Any],
    geometry: dict[str, int],
    architecture: str | None,
) -> tuple[int | None, str]:
    raw = config.raw
    if any(raw.get(key) is not None for key in ("quantization_config", "hf_quant_config", "quant_algo")):
        return None, "quantized/mixed tensor storage, packing and scales require explicit rank-local weights_bytes"
    if architecture not in _WEIGHT_LAYOUTS:
        return (
            None,
            "tensor layout has no supported weight estimator; "
            "provide rank-local weights_bytes including auxiliary tensors",
        )
    if request is None:
        return None, "complete deployment identity and topology before estimating rank-local weights_bytes"
    if values.get("gemm_quant_mode") != "bfloat16" or (
        architecture == "MixtralForCausalLM" and values.get("moe_quant_mode") != "bfloat16"
    ):
        return None, "weight estimator supports unquantized bfloat16 deployment tensors only"
    needed = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "head_dim",
        "local_kv_heads",
        "vocab_size",
    )
    absent = [key for key in needed if key not in geometry]
    if absent:
        return None, f"weight geometry missing {', '.join(absent)}; provide rank-local weights_bytes"
    if any(raw.get(key) for key in (*_AUXILIARY_DIMENSIONS, "lm_head_bias")):
        return None, "additional prediction/shared-expert tensors require explicit rank-local weights_bytes"
    if raw.get("use_qk_norm") and architecture != "Qwen3ForCausalLM":
        return None, "nonstandard Q/K normalization tensors require explicit weights_bytes"
    if architecture == "Qwen2ForCausalLM" and "attention_bias" in raw:
        return None, "Qwen2 attention-bias override is outside the vanilla tensor layout; provide weights_bytes"
    if architecture in {"MistralForCausalLM", "MixtralForCausalLM"} and raw.get("attention_bias"):
        return None, "nonstandard attention biases require explicit weights_bytes"
    if architecture != "LlamaForCausalLM" and raw.get("mlp_bias"):
        return None, "nonstandard MLP biases require explicit weights_bytes"
    h, intermediate, layers, heads, head_dim, local_kv, vocab = (geometry[key] for key in needed)
    tp = request.search.tensor_parallel
    if vocab % 64 or vocab % tp:
        return None, "runtime vocabulary padding is not established for this vocabulary/TP; provide weights_bytes"
    q_width, kv_width = heads * head_dim // tp, local_kv * head_dim
    attention = 2 * h * (q_width + kv_width)
    qkv_bias = raw.get("attention_bias", architecture == "Qwen2ForCausalLM")
    output_bias = raw.get("attention_bias", False) if architecture == "LlamaForCausalLM" else False
    attention += (q_width + 2 * kv_width) * qkv_bias + h * output_bias
    norms = 2 * h + (2 * head_dim if architecture == "Qwen3ForCausalLM" else 0)
    if architecture == "MixtralForCausalLM":
        experts = values.get("num_experts")
        if experts is None:
            return None, "num_experts is required for the MoE weight estimate"
        if raw.get("mlp_bias"):
            return None, "biased MoE tensor layout requires explicit weights_bytes"
        parallel = request.parallelism()
        mlp = 3 * h * (intermediate // parallel["moe_tensor"]) * (experts // parallel["moe_expert"])
        router = h * experts  # replicated on every rank, including expert-parallel ranks
        if raw.get("moe_router_dtype") not in (None, "bfloat16"):
            return None, "mixed router dtype requires explicit weights_bytes"
    else:
        mlp = 3 * h * (intermediate // tp)
        mlp += (2 * intermediate // tp + h) * raw.get("mlp_bias", False)
        router = 0
    embeddings = (vocab // tp) * h * (1 if raw.get("tie_word_embeddings", False) else 2)
    result = 2 * (embeddings + layers * (attention + mlp + router + norms) + h)
    return result, (
        f"estimate: vanilla {architecture} bfloat16 tensors, TP={tp}, MoE TP={request.parallelism()['moe_tensor']}, "
        f"EP={request.parallelism()['moe_expert']}; "
        f"sharded embedding/head (tied={raw.get('tie_word_embeddings', False)}), "
        "Q/K/V/output and gated MLP, configured biases, replicated RMS norms/router; "
        "no quantization/padding/allocator extras"
    )


def _activation_estimate(
    values: dict[str, Any], geometry: dict[str, int], request: SupportRequest | None, architecture: str | None
) -> tuple[int | None, str]:
    if architecture not in _FULL_ATTENTION or request is None:
        return None, "activation/workspace estimate unavailable for this layout or topology; provide activations_bytes"
    if values.get("fmha_quant_mode") not in {"bfloat16", "float16"}:
        return None, "activation estimate requires an explicit 16-bit FMHA mode, otherwise provide activations_bytes"
    if "num_attention_heads" not in geometry or "head_dim" not in geometry:
        return None, "attention width missing; provide activations_bytes"
    tp = request.search.tensor_parallel
    # AISimulate's local vLLM backend shares the TRT-LLM activation estimate.
    # Keep this narrow subset independent of backend/model imports and do not
    # extrapolate the coefficient table to other TP sizes or DSA workspaces.
    coefficients = {1: 22, 2: 13, 4: 10, 8: 10} if architecture not in _DENSE else {1: 11, 2: 6.5, 4: 5, 8: 5}
    if tp not in coefficients:
        return None, f"no activation coefficient for TP={tp}; provide activations_bytes"
    tokens = values["max_num_tokens"]
    width = geometry["num_attention_heads"] * geometry["head_dim"]
    result = max(70 * 1024 * 1024, tokens * width * int(2 * coefficients[tp]))
    return result, (
        f"estimate: sdk/backends/base_backend.py + trtllm_backend.py vLLM policy; "
        f"max(70 MiB, 2 * max_num_tokens={tokens} * attention_width={width} * coefficient={coefficients[tp]}); "
        f"rank-local max_batch_size={values['max_batch_size']}; no speculative decoding; not memory qualification"
    )


def _hardware_estimates(request: SupportRequest | None) -> dict[str, tuple[int, str]]:
    if request is None:
        return {}
    resource = files("aiconfigurator_core").joinpath("systems", request.identity.gpu + ".yaml")
    if not resource.is_file():
        return {}
    payload = resource.read_bytes()
    spec = yaml.safe_load(payload)
    if not isinstance(spec, dict) or not isinstance(spec.get("misc", {}), dict):
        raise ValueError(f"invalid packaged hardware metadata for {request.identity.gpu}")
    misc = spec.get("misc", {})
    source = f"estimate: packaged systems/{request.identity.gpu}.yaml sha256={hashlib.sha256(payload).hexdigest()}"
    result = {}
    if "other_mem" in misc:
        value = _integer(misc["other_mem"], "system misc.other_mem", minimum=0, maximum=2**53)
        result["runtime_overhead_bytes"] = (
            value,
            source + " misc.other_mem per rank; excludes CUDA graph reservation; not a measurement",
        )
    parallel = request.parallelism()
    nccl = misc.get("nccl_mem", {})
    if not isinstance(nccl, dict):
        raise ValueError("system misc.nccl_mem must be a TP-indexed object")
    if parallel["attention_data"] == parallel["moe_expert"] == 1 and parallel["tensor"] in nccl:
        value = _integer(nccl[parallel["tensor"]], "system misc.nccl_mem", minimum=0, maximum=2**53)
        result["comm_overhead_bytes"] = (
            value,
            source + f" misc.nccl_mem[{parallel['tensor']}] for pure TP; not a measurement",
        )
    return result


def derive_profile(
    config: ModelConfig, request: SupportRequest | None = None, overrides: Mapping[str, Any] | None = None
) -> ProfileDraft:
    """Return every resolved value and actionable missing field, or a full profile.

    Invalid structural facts and overrides raise immediately. ``request=None``
    supports validating overrides and previewing config facts before collecting
    deployment identity; it can never produce a complete deployment profile.
    """
    supplied = validate_overrides(overrides)
    values = {key: config.suggestions[key] for key in _MODEL_FIELDS if key in config.suggestions}
    sources = {key: config.notes[key] for key in values}
    for key, (value, source) in _precision_facts(config).items():
        values[key], sources[key] = value, source
    for key in ("moe_backend", "attention_backend"):
        values[key], sources[key] = "auto", "existing FPM deployment schema default: automatic backend selection"
    values["max_num_tokens"] = 8192
    sources["max_num_tokens"] = "onboarding scheduler policy: rank-local max_batched_tokens=8192"
    values["max_batch_size"] = request.workload.concurrency if request else 256
    sources["max_batch_size"] = (
        "onboarding rank-local workload concurrency"
        if request
        else "initial onboarding max_sequences=256; resolved to workload concurrency after identity input"
    )
    for key, value in supplied.items():
        if key in ("architecture", "num_experts") and key in values and values[key] != value:
            raise ValueError(f"{key} override conflicts with the source config: {value!r} != {values[key]!r}")
        if key == "context_length" and key in values and value > values[key]:
            raise ValueError("context_length override exceeds the source config context limit")
        values[key], sources[key] = value, "user override; caller-declared value, not an inferred measurement"
    architecture = config.decoder_architecture
    if "architecture" not in config.suggestions:
        architecture = values.get("architecture")
    _validate_architecture(config.raw, architecture, values.get("num_experts"))
    geometry = _geometry(config, architecture, request)
    if request:
        if "num_experts" in values:
            experts = values["num_experts"]
            if (experts > 0) != (request.identity.model_kind == "moe"):
                raise ProfileRequestError("model_kind", "num_experts and request identity.model_kind disagree")
            if experts and experts % request.parallelism()["moe_expert"]:
                raise ProfileRequestError(
                    "moe_expert_parallel", "num_experts must be divisible by the selected moe_expert_parallel"
                )
        if "context_length" in values and request.search.context_length > values["context_length"]:
            raise ProfileRequestError(
                "context_length", "request search.context_length exceeds the configured profile context_length"
            )
    missing = {
        "architecture": "provide the decoder architecture; config architectures/model_type did not identify it",
        "context_length": "provide the supported model context length; not declared in config",
        "num_experts": "provide the routed expert count (0 for dense); model kind is unknown from this config",
        "gemm_quant_mode": (
            "declare the deployed GEMM quantization mode; config does not identify a supported unambiguous mode"
        ),
        "moe_quant_mode": "declare the deployed MoE quantization identity, also required for dense profiles",
        "fmha_quant_mode": "declare runtime attention precision; checkpoint weight dtype does not establish it",
        "comm_quant_mode": "declare runtime communication precision; it is not checkpoint metadata",
        "kv_cache_dtype": "declare runtime KV-cache dtype; weight precision does not establish it",
        "cache_layout": (
            "confirm linear cache layout; unknown cache semantics cannot be inferred from ordinary head counts"
        ),
        "kv_bytes_per_token": (
            "provide rank-local bytes per cached token for all persistent cache tensors; "
            "MLA/DSA and unknown layouts need explicit accounting"
        ),
        "runtime_overhead_bytes": "provide rank-local runtime reservation excluding separately configured CUDA graphs",
        "comm_overhead_bytes": (
            "provide rank-local communication reservation for this topology; no exact TP/DEP/TEP hardware entry"
        ),
    }
    if architecture in _FULL_ATTENTION:
        if "cache_layout" not in values:
            values["cache_layout"], sources["cache_layout"] = (
                "linear",
                "known full-attention GQA/MHA decoder layout; no sliding/recurrent state",
            )
        needed = ("num_hidden_layers", "local_kv_heads", "head_dim")
        kv_scheme = _kv_cache_scheme(config.raw.get("quantization_config") or {})
        ordinary_kv = kv_scheme.get("dynamic") is not True and kv_scheme.get("strategy") in (None, "tensor")
        if (
            "kv_bytes_per_token" not in values
            and all(key in geometry for key in needed)
            and "kv_cache_dtype" in values
            and ordinary_kv
        ):
            layers, local_kv, head_dim = (geometry[key] for key in needed)
            element_bytes = quantization.KVCacheQuantMode[values["kv_cache_dtype"]].value.memory
            values["kv_bytes_per_token"] = int(2 * layers * local_kv * head_dim * element_bytes)
            sources["kv_bytes_per_token"] = (
                f"exact linear tensor geometry: 2 K/V * {layers} layers * {local_kv} rank-local KV heads "
                f"* {head_dim} head_dim * {element_bytes} bytes; KV heads replicate when TP exceeds head count; "
                "attention DP does not divide a rank's cache"
            )
        missing["kv_bytes_per_token"] = (
            "linear KV calculation needs layer/head geometry, explicit runtime kv_cache_dtype and completed topology; "
            "otherwise provide bytes per token"
        )
        if not ordinary_kv:
            missing["kv_bytes_per_token"] = (
                "nonstandard KV scaling/packing needs explicit bytes per token including scales"
            )
    for key, (value, source) in _hardware_estimates(request).items():
        if key not in values:
            values[key], sources[key] = value, source
    for key, estimate in (("weights_bytes", _weight_estimate), ("activations_bytes", _activation_estimate)):
        if key in values:
            continue
        value, source = (
            estimate(config, request, values, geometry, architecture)
            if key == "weights_bytes"
            else estimate(values, geometry, request, architecture)
        )
        if value is None:
            missing[key] = source
        else:
            values[key], sources[key] = value, source
    missing = {key: value for key, value in missing.items() if key not in values}
    for key in _BYTE_FIELDS & values.keys():
        _integer(values[key], key, minimum=1 if key == "kv_bytes_per_token" else 0, maximum=2**53)
    if sum(values.get(key, 0) for key in _BYTE_FIELDS - {"kv_bytes_per_token"}) > 2**53:
        raise ValueError("total non-KV resource bytes must not exceed 2**53")
    if missing or request is None:
        return ProfileDraft(None, values, missing, sources)
    provenance = json.dumps(
        {
            "config_sha256": config.sha256,
            "method": (
                "local config facts and declared/estimated per-rank resources; no checkpoint inspection or measurements"
            ),
            "config_notes": config.notes,
            "fields": {key: {"value": values[key], "source": sources[key]} for key in sorted(values)},
            "deployment_identity": request.identity.model_dump(mode="json"),
            "parallelism": request.parallelism(),
        },
        sort_keys=True,
    )
    parallel = request.parallelism()
    profile = FpmModelProfile.model_validate(
        {
            "schema_version": 1,
            "model": request.identity.model,
            "model_revision": request.identity.model_revision,
            **{key: values[key] for key in _MODEL_FIELDS},
            "provenance": provenance,
            "deployments": [
                {
                    "system": request.identity.gpu,
                    "backend": request.identity.framework,
                    "backend_version": request.identity.framework_version,
                    "tp": parallel["tensor"],
                    "dp": parallel["attention_data"],
                    "moe_tp": parallel["moe_tensor"],
                    "moe_ep": parallel["moe_expert"],
                    **{key: values[key] for key in _DEPLOYMENT_FIELDS},
                    "resources": {**{key: values[key] for key in _RESOURCE_FIELDS}, "provenance": provenance},
                }
            ],
        }
    )
    return ProfileDraft(profile, values, {}, sources)
