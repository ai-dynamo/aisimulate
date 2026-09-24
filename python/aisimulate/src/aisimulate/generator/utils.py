# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for generator modules."""

from __future__ import annotations

import os
from functools import cache
from typing import Any, Optional

import yaml

DEFAULT_BACKEND = "trtllm"
GENERATOR_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
GENERATOR_FACTS_DIR = os.path.join(os.path.dirname(__file__), "facts")
DEFAULT_BACKEND_VERSION_MATRIX_PATH = os.path.join(GENERATOR_FACTS_DIR, "runtimes", "dynamo.yaml")


def normalize_backend(backend: Optional[str], default: str = DEFAULT_BACKEND) -> str:
    """Normalize backend names to lowercase strings with a fallback."""
    if backend:
        return str(backend).strip().lower()
    return default


def coerce_bool(value: Optional[Any]) -> Optional[bool]:
    """Best-effort conversion of user input into booleans."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def coerce_int(value: Optional[Any]) -> Optional[int]:
    """Convert values to ints while swallowing Type/Value errors."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_yaml_payload(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@cache
def load_backend_version_matrix(matrix_path: str) -> dict[str, dict[str, Any]]:
    payload = _load_yaml_payload(matrix_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Backend version matrix must be a YAML mapping: {matrix_path}")
    matrix = payload.get("matrix", payload)
    if not isinstance(matrix, dict):
        raise TypeError(f"Backend version matrix missing 'matrix' mapping: {matrix_path}")
    return matrix


def get_default_dynamo_version_mapping(
    matrix_path: str = DEFAULT_BACKEND_VERSION_MATRIX_PATH,
) -> tuple[str, dict[str, Any]]:
    """
    Return the default Dynamo version and its backend-version mapping.

    The default entry is the first item in backend_version_matrix.yaml.
    """
    matrix = load_backend_version_matrix(matrix_path)
    if not matrix:
        raise ValueError(f"Backend version matrix is empty: {matrix_path}")
    dynamo_version, entry = next(iter(matrix.items()))
    if not isinstance(entry, dict):
        raise TypeError(f"Invalid backend version entry for {dynamo_version}: {entry!r}")
    return str(dynamo_version), entry


def resolve_backend_version_for_dynamo(
    dynamo_version: str,
    backend: str | None = None,
    matrix_path: str = DEFAULT_BACKEND_VERSION_MATRIX_PATH,
) -> str:
    """
    Given a Dynamo (generator) version, look up the corresponding backend version for a specified backend.

    Parameters:
        dynamo_version (str): The target Dynamo generator release (e.g., "0.8.1").
        backend (str | None): Name of the backend to look up ("trtllm", "vllm", "sglang", or "auto").
        matrix_path (str): Path to the backend version matrix YAML file.

    Returns:
        str | dict: The backend version(str) for the given backend, or a dict of versions if backend is "auto" or None.

    Raises:
        ValueError: If the dynamo_version is missing or not present in the matrix.
        TypeError: If the loaded matrix or entry is invalid, or if no mapping exists for the given backend.
    """
    version_key = str(dynamo_version).strip()
    if version_key.lower().startswith("v") and len(version_key) > 1 and version_key[1].isdigit():
        version_key = version_key[1:]
    if not version_key:
        raise ValueError("dynamo_version must be a non-empty string.")
    matrix = load_backend_version_matrix(matrix_path)
    entry = matrix.get(version_key)
    if not isinstance(entry, dict):
        supported = ", ".join(sorted(matrix.keys()))
        raise TypeError(f"Unsupported dynamo_version '{version_key}'. Supported versions: {supported or 'none'}.")

    backend_key = normalize_backend(backend, DEFAULT_BACKEND)
    # return all backend versions for "auto" backend
    if not backend or backend_key == "auto":
        return entry

    backend_version = entry.get(backend_key)
    if backend_version is None:
        supported_backends = ", ".join(sorted(entry.keys()))
        raise ValueError(
            f"No backend version mapping for backend '{backend_key}' in dynamo '{version_key}'. "
            f"Supported backends: {supported_backends or 'none'}."
        )
    return str(backend_version)


def msa_sparse_implementation(backend_name: str, model_path: str, system_name: str) -> str | None:
    """MiniMax-M3 x TRT-LLM on the SM100 family: prescribe the msa
    (fmha_sm100) sparse-attention implementation.

    TRT-LLM 1.3.0rc23 serving DEFAULTS to the Triton reference path; the
    shipped SM100/103 MSA perf tables are collected with
    ``implementation="msa"`` (the performance path the config field exists
    for, hard-gated to those SMs by ``ensure_msa_available``). Emitting the
    knob makes generated deployments — BOTH the optimized (module_bridge)
    and the naive entry points — run exactly the configuration the perf
    data represents (PR #1507 review 4969690316). Keyed on the checkpoint
    ARCHITECTURE (never model-name patterns) and the system's sm_version
    fact; returns None everywhere else so the field is dropped.
    """
    if backend_name != "trtllm":
        return None
    from aisimulate.sdk.perf_database import load_system_spec
    from aisimulate.sdk.utils import get_model_config_from_model_path

    try:
        parsed = get_model_config_from_model_path(model_path)
        architecture = parsed.get("architecture")
    except (FileNotFoundError, KeyError, ValueError):
        # Unresolvable model config (e.g. a user-local checkpoint the SDK
        # does not bundle): leave the knob unset — serving falls back to its
        # own default rather than receiving a wrong prescription.
        return None
    # Both artifact forms of the same model: the BF16 bundle carries the
    # text-backbone architecture, the NVFP4 bundle the raw hub VL wrapper.
    if architecture not in ("MiniMaxM3ForCausalLM", "MiniMaxM3SparseForConditionalGeneration"):
        return None
    spec = load_system_spec(system_name)
    if int(spec.get("gpu", {}).get("sm_version", -1)) in (100, 103):
        return "msa"
    return None


def _model_architecture(model_path: str) -> str | None:
    """Architecture of a bundled checkpoint config, None when unresolvable
    (a user-local checkpoint the SDK does not bundle). Indirection so tests
    can stub the SDK lookup."""
    from aisimulate.sdk.utils import get_model_config_from_model_path

    try:
        return get_model_config_from_model_path(model_path).get("architecture")
    except (FileNotFoundError, KeyError, ValueError):
        return None


# DeepSeek sparse attention (DSA) architectures: the sparse-MLA selector is
# the one that rejects vLLM's auto-resolved kv dtype spelling (see below).
_DSA_ARCHITECTURES = ("GlmMoeDsaForCausalLM", "DeepseekV32ForCausalLM")


def _bundled_quantization(model_path: str) -> dict | None:
    """The artifact's quantization facts as the SDK loader exposes them:
    config.json ``quantization_config`` merged with the ``hf_quant_config``
    the loader attaches from the bundled ``<repo>_hf_quant_config.json`` (the
    modelopt artifacts keep ``kv_cache_quant_algo`` ONLY there — their
    config.json has no quantization block at all). None when the config is
    unresolvable or carries neither. Indirection so tests can stub it."""
    from aisimulate_core.sdk.utils import _load_model_config_from_model_path

    try:
        raw = _load_model_config_from_model_path(model_path)
    except (FileNotFoundError, KeyError, ValueError):
        return None
    merged: dict = {}
    hfq = raw.get("hf_quant_config")
    if isinstance(hfq, dict):
        merged.update(hfq.get("quantization") if isinstance(hfq.get("quantization"), dict) else hfq)
    if isinstance(raw.get("quantization_config"), dict):
        merged.update(raw["quantization_config"])
    return merged or None


def _artifact_pins_fp8_kv(quantization: dict | None) -> bool:
    """modelopt artifacts pin the KV cache either as ``kv_cache_quant_algo:
    FP8`` (hf_quant_config.json) or as ``kv_cache_scheme: {num_bits: 8,
    type: float}`` (config.json quantization block); either spelling counts."""
    if not quantization:
        return False
    if str(quantization.get("kv_cache_quant_algo") or "").upper() == "FP8":
        return True
    scheme = quantization.get("kv_cache_scheme")
    if isinstance(scheme, str):
        return scheme.upper() == "FP8"
    scheme = scheme or {}
    return scheme.get("num_bits") == 8 and str(scheme.get("type", "")).lower() == "float"


def vllm_dsa_kv_cache_dtype(backend_name: str, model_path: str, gemm_quant_mode: Any = None) -> str | None:
    """NVFP4 DSA checkpoints x vLLM: prescribe ``--kv-cache-dtype fp8``.

    The modelopt NVFP4 artifacts of the DSA models (GLM-5 / 5.1 / 5.2 / 5.3
    -NVFP4, DeepSeek-V3.2-NVFP4) pin the KV cache to FP8. vLLM 0.29.0
    resolves ``--kv-cache-dtype auto`` for them to the literal ``fp8_e4m3``
    and its sparse-MLA backend selector rejects that spelling
    (FLASHMLA_SPARSE: "kv_cache_dtype not supported"); the same engine with
    an explicit ``fp8`` loads FlashMLASparseImpl and serves. Serving-side
    the KV IS fp8 either way (checkpoint scheme), so the explicit value only
    says what ``auto`` means for these artifacts.

    Keyed on the checkpoint ARCHITECTURE (DSA) plus an artifact fact — never
    a model-name pattern: the task's ``nvfp4`` GEMM quant mode on the
    optimized path, or the bundled config's fp8 KV scheme on the naive
    ``cli generate`` path (which has no perf task). Bundled configs without a
    ``quantization_config`` (the Hub config of DeepSeek-V3.2-NVFP4 keeps its
    quantization only in hf_quant_config.json) get no prescription until the
    SDK bundles that fact — a known gap, recorded in the opharness findings.
    Evidence: findings ``vllm_fp8kv`` (0.29 addendum),
    ``glm53_onboarding_2026_09_24``; owner decision 2026-09-24. Returns None
    everywhere else so the task's own kv mode stands.
    """
    if backend_name != "vllm":
        return None
    if _model_architecture(model_path) not in _DSA_ARCHITECTURES:
        return None
    if str(gemm_quant_mode or "").lower() == "nvfp4":
        return "fp8"
    if _artifact_pins_fp8_kv(_bundled_quantization(model_path)):
        return "fp8"
    return None
