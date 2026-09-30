# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint precision reconciliation shared by FPM onboarding and collection.

These functions do not change the general SDK's model-loading precedence.
"""

from __future__ import annotations

from typing import Any

from .utils import (
    _attach_hf_quant_config,
    _infer_quantization_fields,
    _normalize_kv_cache_algo,
    _normalize_quant_algo,
)


def inherit_decoder_precision(document: dict[str, Any], decoder: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Inherit a shared precision declaration only when the decoder lacks it."""
    raw = dict(decoder)
    inherited = []
    for keys, inferred_keys in (
        (("dtype", "torch_dtype"), ()),
        (("quantization_config", "hf_quant_config", "quant_algo"), ("quant_dynamic", "kv_cache_quant_algo")),
    ):
        if any(decoder.get(key) is not None for key in keys):
            continue
        for key in keys:
            if document.get(key) is not None:
                raw[key] = document[key]
                inherited.append(key)
        for key in inferred_keys:
            if decoder.get(key) is None and document.get(key) is not None:
                raw[key] = document[key]
                inherited.append(key)
    return raw, inherited


def attach_quantization_sidecar(raw: dict[str, Any], sidecar: dict[str, Any]) -> dict[str, Any]:
    """Reconcile one effective decoder with its adjacent ModelOpt sidecar."""
    section = sidecar.get("quantization")
    if section is not None and not isinstance(section, dict):
        raise ValueError("hf_quant_config.json quantization must be an object")
    for key in ("quant_algo", "quantization_algo", "kv_cache_quant_algo"):
        if section and section.get(key) is not None and (not isinstance(section[key], str) or not section[key].strip()):
            raise ValueError(f"hf_quant_config.json quantization.{key} must be a nonempty string")
    if (
        section
        and section.get("quant_algo")
        and section.get("quantization_algo")
        and _normalize_quant_algo(section["quant_algo"]) != _normalize_quant_algo(section["quantization_algo"])
    ):
        raise ValueError("conflicting quant_algo and quantization_algo in hf_quant_config.json")
    if raw.get("hf_quant_config") is not None and raw["hf_quant_config"] != sidecar:
        raise ValueError("conflicting inline hf_quant_config and adjacent hf_quant_config.json")
    # Validate the inline declaration independently: a repeated sidecar must not
    # hide contradictory quantization_config fields through SDK precedence.
    inline = _infer_quantization_fields({**raw, "hf_quant_config": None})
    merged = _attach_hf_quant_config(dict(raw), sidecar)
    # Preserve even unknown/empty sidecars so they cannot imply BF16 weights.
    merged["hf_quant_config"] = sidecar
    inferred = _infer_quantization_fields(merged)
    scheme = (raw.get("quantization_config") or {}).get("kv_cache_scheme")
    if isinstance(scheme, str) and scheme.strip().upper() == "FP8":
        inline.setdefault("kv_cache_quant_algo", "fp8")
    elif isinstance(scheme, dict) and scheme.get("num_bits") == 8 and scheme.get("type") in {"float", "int"}:
        inline.setdefault("kv_cache_quant_algo", "fp8" if scheme["type"] == "float" else "int8")
    if "kv_cache_quant_algo" in inline:
        inferred.setdefault("kv_cache_quant_algo", inline["kv_cache_quant_algo"])
    for key in ("quant_algo", "kv_cache_quant_algo"):
        normalize = _normalize_quant_algo if key == "quant_algo" else _normalize_kv_cache_algo
        after = normalize(inferred.get(key))
        for value in (raw.get(key), inline.get(key)):
            before = normalize(value)
            if key == "quant_algo" and (before in {"modelopt", "compressed-tensors"} or after == "mixed_precision"):
                continue  # Containers and per-layer layouts are not scalar algorithms.
            if key == "kv_cache_quant_algo":
                before = "bfloat16" if before == "none" else before
                after = "bfloat16" if after == "none" else after
            if before is not None and after is not None and before != after:
                raise ValueError(f"conflicting {key} in model config and hf_quant_config.json: {before!r} != {after!r}")
    dynamic = raw.get("quant_dynamic")
    if dynamic is not None and type(dynamic) is not bool:
        raise ValueError("quant_dynamic must be a boolean or null")
    if dynamic is not None and inferred.get("quant_dynamic") is not None and dynamic != inferred["quant_dynamic"]:
        raise ValueError("conflicting quant_dynamic in model config and quantization metadata")
    for key, value in inferred.items():
        # Null is unknown. Container markers can be refined by the sidecar;
        # actual conflicting scalar declarations were rejected above.
        if merged.get(key) is None or (key == "quant_algo" and merged[key] in {"modelopt", "compressed-tensors"}):
            merged[key] = value
    return merged
