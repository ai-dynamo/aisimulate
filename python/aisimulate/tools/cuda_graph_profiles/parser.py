# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.cuda_graph_profiles.common import (
    GIB,
    SCHEMA_VERSION,
    model_revision_from_path,
    normalize_capture_sizes,
    normalize_model_id,
    normalize_system,
)


class ProfileParseError(ValueError):
    """Raised when an artifact cannot produce a trustworthy profile."""


@dataclass(frozen=True)
class ParsedMeasurement:
    identity: dict[str, Any]
    estimated_bytes_by_rank: dict[int, int]
    actual_bytes_by_rank: dict[int, int]
    available_kv_cache_bytes: int | None
    gpu_kv_cache_tokens: int | None
    profiled_full_count: int | None
    profiled_full_largest_capture_size: int | None
    profiled_piecewise_count: int | None
    profiled_piecewise_largest_capture_size: int | None
    graph_disabled: bool
    identity_sources: dict[str, str]


_RANK_RE = re.compile(r"(?:Worker(?:_(?:TP|DP))?|rank[ =])(\d+)", re.IGNORECASE)
_ESTIMATE_RE = re.compile(r"Estimated CUDA graph memory:\s*([0-9.]+)\s*GiB", re.IGNORECASE)
_POOL_RE = re.compile(
    r"CUDA graph pool memory:\s*([0-9.]+)\s*GiB\s*\(actual\),\s*([0-9.]+)\s*GiB\s*\(estimated\)",
    re.IGNORECASE,
)
_ACTUAL_RE = re.compile(r"Graph capturing finished.*?took\s*([0-9.]+)\s*GiB", re.IGNORECASE)
_AVAILABLE_KV_RE = re.compile(r"Available KV cache memory:\s*([0-9.]+)\s*GiB", re.IGNORECASE)
_KV_TOKENS_RE = re.compile(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", re.IGNORECASE)
_FULL_PROFILE_RE = re.compile(r"\bFULL=(\d+)\s*\(largest=(\d+)\)")
_PIECEWISE_PROFILE_RE = re.compile(r"\bPIECEWISE=(\d+)\s*\(largest=(\d+)\)")


def _bytes_from_gib(value: str) -> int:
    return round(float(value) * GIB)


def _rank(line: str) -> int:
    match = _RANK_RE.search(line)
    return int(match.group(1)) if match else 0


def _first(pattern: str, text: str, flags: int = 0) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1) if match else None


def _integer(pattern: str, text: str) -> int | None:
    value = _first(pattern, text)
    return int(value) if value is not None else None


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return None


def _consistent_profile_value(values: set[int], label: str) -> int | None:
    if len(values) > 1:
        raise ProfileParseError(f"incompatible rank-local CUDA graph {label}: {sorted(values)}")
    return next(iter(values), None)


def _engine_identity(text: str) -> tuple[dict[str, Any], dict[str, str]]:
    engine_line = next((line for line in text.splitlines() if "Initializing a V1 LLM engine" in line), "")
    args_line = next((line for line in text.splitlines() if "non-default args:" in line), "")
    combined = f"{args_line}\n{engine_line}"
    raw_model = _first(r"(?:with config: model=|'model':\s*)'([^']+)'", combined)
    precision = _first(r"'quantization':\s*'([^']+)'", args_line)
    quantization = _first(r"quantization=([^,]+)", engine_line) or precision
    revision = model_revision_from_path(raw_model)
    logged_revision = _first(r"\brevision=([^,]+)", engine_line)
    if not revision and logged_revision and logged_revision not in {"None", "main"}:
        revision = logged_revision.strip("'\"")

    capture_text = _first(r"cudagraph_capture_sizes'?:\s*\[([^]]*)\]", combined)
    capture_sizes = [] if capture_text is None else [int(value) for value in re.findall(r"\d+", capture_text)]
    spec_method = _first(r"SpeculativeConfig\(method='([^']+)'", engine_line)
    spec_tokens = _integer(r"num_spec_(?:tokens|ulative_tokens)[=']:\s*(\d+)", combined)
    if spec_tokens is None:
        spec_tokens = _integer(r"num_spec_tokens=(\d+)", engine_line)

    max_num_seqs = _integer(r"'max_num_seqs':\s*(\d+)", args_line)
    if max_num_seqs is None and spec_method == "eagle3" and capture_sizes and spec_tokens is not None:
        divisor = spec_tokens + 1
        if max(capture_sizes) % divisor == 0:
            max_num_seqs = max(capture_sizes) // divisor

    attention_backend = _first(r"'attention_backend':\s*'([^']+)'", args_line)
    if not attention_backend:
        attention_backend = _first(r"AttentionConfig\(backend=<AttentionBackendEnum\.([^:>]+)", combined)
    if not attention_backend:
        attention_backend = _first(r"Using\s+([A-Z0-9_]+)\s+attention backend", text)
    if not attention_backend and "flashinfer_sparse_mla" in text.lower():
        attention_backend = "FLASHINFER_SPARSE_MLA"

    graph_mode = _first(r"CUDAGraphMode\.([^:>]+)", combined)
    identity = {
        "model_id": normalize_model_id(raw_model, precision=quantization),
        "model_revision": revision,
        "model_config_sha256": None,
        "backend": "vllm",
        "backend_version": _first(r"Initializing a V1 LLM engine \(v([^)]+)\)", engine_line),
        "backend_build": None,
        "tp_size": _integer(r"tensor_parallel_size=(\d+)", engine_line),
        "pp_size": _integer(r"pipeline_parallel_size=(\d+)", engine_line),
        "attention_dp_size": _integer(r"data_parallel_size=(\d+)", engine_line),
        "dcp_size": _integer(r"decode_context_parallel_size=(\d+)", engine_line),
        "pcp_size": _integer(r"prefill_context_parallel_size=(\d+)", engine_line) or 1,
        "moe_tp_size": None,
        "moe_ep_size": None,
        "quantization": None if quantization in {None, "None"} else quantization.strip("'\""),
        "compute_dtype": _first(r"dtype=torch\.([^,]+)", engine_line),
        "kv_cache_dtype": _first(r"kv_cache_dtype=([^,]+)", engine_line),
        "cuda_graph_mode": graph_mode,
        "cuda_graph_capture_sizes": normalize_capture_sizes(capture_sizes),
        "compilation_mode": _first(r"CompilationMode\.([^:>]+)", combined),
        "compilation_backend": _first(r"compilation_config=.*?'backend':\s*'([^']+)'", engine_line),
        "moe_backend": _first(r"\bmoe_backend='([^']+)'", engine_line),
        "linear_backend": _first(r"\blinear_backend='([^']+)'", engine_line),
        "flashinfer_autotune": _boolean(
            _first(r"\benable_flashinfer_autotune=(True|False)", engine_line)
            or _first(r"'enable_flashinfer_autotune':\s*(True|False)", args_line)
        ),
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": _integer(r"max_num_batched_tokens=(\d+)", text),
        "max_model_len": _integer(r"max_seq_len=(\d+)", engine_line)
        or _integer(r"'max_model_len':\s*(\d+)", args_line),
        "attention_backend": attention_backend,
        "speculative_method": spec_method or "none",
        "speculative_tokens": spec_tokens or 0,
    }
    return identity, {field: "engine_log" for field, value in identity.items() if value is not None}


def _flatten_config(value: Any) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            flattened.setdefault(str(key), child)
            flattened.update(_flatten_config(child))
    elif isinstance(value, list):
        for child in value:
            flattened.update(_flatten_config(child))
    return flattened


def _config_identity(config: dict[str, Any]) -> dict[str, Any]:
    flat = _flatten_config(config)
    identity_section = config.get("identity", {})
    model_section = config.get("model", {})
    backend_section = config.get("backend", {})
    resources = config.get("resources", {})
    vllm_config = backend_section.get("vllm_config", {}).get("aggregated", {})
    pinned_model = identity_section.get("model", {}) if isinstance(identity_section, dict) else {}
    frameworks = identity_section.get("frameworks", {}) if isinstance(identity_section, dict) else {}
    raw_model = (
        pinned_model.get("repo")
        or vllm_config.get("served-model-name")
        or (model_section.get("path") if isinstance(model_section, dict) else None)
        or (model_section if isinstance(model_section, str) else None)
        or flat.get("model")
        or flat.get("model_path")
    )
    revision = (
        pinned_model.get("revision")
        or flat.get("revision")
        or model_revision_from_path(str(raw_model) if raw_model else None)
    )
    capture_sizes = flat.get("cudagraph_capture_sizes") or flat.get("cuda_graph_capture_sizes")
    if not capture_sizes:
        compilation_config = vllm_config.get("compilation-config")
        if isinstance(compilation_config, str):
            try:
                capture_sizes = json.loads(compilation_config).get("cudagraph_capture_sizes")
            except json.JSONDecodeError:
                capture_sizes = None
    speculative_config = vllm_config.get("speculative-config")
    speculative: dict[str, Any] = {}
    if isinstance(speculative_config, str):
        try:
            speculative = json.loads(speculative_config)
        except json.JSONDecodeError:
            speculative = {}
    elif isinstance(speculative_config, dict):
        speculative = speculative_config
    precision = model_section.get("precision") if isinstance(model_section, dict) else flat.get("precision")
    compilation_config = vllm_config.get("compilation-config")
    if isinstance(compilation_config, str):
        try:
            compilation_config = json.loads(compilation_config)
        except json.JSONDecodeError:
            compilation_config = {}
    if not isinstance(compilation_config, dict):
        compilation_config = {}
    return {
        "model_id": normalize_model_id(str(raw_model) if raw_model else None, precision=str(precision or "")),
        "model_revision": None if revision in {None, "main"} else str(revision),
        "model_config_sha256": flat.get("model_config_sha256"),
        "system": normalize_system(str(resources.get("gpu_type") or flat.get("hw") or flat.get("system") or "")),
        "backend": str(flat.get("framework") or backend_section.get("type") or "vllm").lower(),
        "backend_version": frameworks.get("vllm") or flat.get("backend_version") or flat.get("vllm_version"),
        "backend_build": pinned_model.get("image") or flat.get("image"),
        "tp_size": vllm_config.get("tensor-parallel-size") or flat.get("tp") or flat.get("tensor_parallel_size"),
        "pp_size": flat.get("pp") or flat.get("pipeline_parallel_size"),
        "attention_dp_size": vllm_config.get("data-parallel-size")
        or flat.get("attention_dp_size")
        or flat.get("data_parallel_size"),
        "dcp_size": flat.get("dcp_size") or flat.get("decode_context_parallel_size"),
        "pcp_size": flat.get("pcp_size") or flat.get("prefill_context_parallel_size"),
        "moe_tp_size": flat.get("moe_tp_size"),
        "moe_ep_size": flat.get("ep") or flat.get("moe_ep_size"),
        "quantization": flat.get("quantization") or precision,
        "compute_dtype": flat.get("dtype") or flat.get("compute_dtype"),
        "kv_cache_dtype": vllm_config.get("kv-cache-dtype") or flat.get("kv_cache_dtype"),
        "cuda_graph_mode": flat.get("cudagraph_mode") or flat.get("cuda_graph_mode"),
        "cuda_graph_capture_sizes": normalize_capture_sizes(capture_sizes) if capture_sizes is not None else None,
        "compilation_mode": compilation_config.get("mode") or flat.get("compilation_mode"),
        "compilation_backend": compilation_config.get("backend") or flat.get("compilation_backend"),
        "moe_backend": flat.get("moe_backend"),
        "linear_backend": flat.get("linear_backend"),
        "flashinfer_autotune": _boolean(flat.get("enable_flashinfer_autotune")),
        "max_num_seqs": vllm_config.get("max-num-seqs") or flat.get("max_num_seqs"),
        "max_num_batched_tokens": vllm_config.get("max-num-batched-tokens") or flat.get("max_num_batched_tokens"),
        "max_model_len": flat.get("max_model_len") or flat.get("max_seq_len"),
        "attention_backend": vllm_config.get("attention-backend") or flat.get("attention_backend"),
        "speculative_method": speculative.get("method") or flat.get("spec_decoding") or flat.get("speculative_method"),
        "speculative_tokens": speculative.get("num_speculative_tokens") or flat.get("num_speculative_tokens"),
    }


def _benchmark_identity(benchmark: dict[str, Any], engine_identity: dict[str, Any]) -> dict[str, Any]:
    engine_tp = int(engine_identity.get("tp_size") or 0)
    engine_attention_dp = int(engine_identity.get("attention_dp_size") or 0)
    benchmark_tp = int(benchmark.get("tp") or 1)
    moe_ep = int(benchmark.get("ep") or 1)
    dp_attention = _boolean(benchmark.get("dp_attention")) or False
    world_size = (engine_tp or benchmark_tp) * (engine_attention_dp or 1)
    if world_size % moe_ep:
        raise ProfileParseError(f"engine world size {world_size} is not divisible by benchmark ep={moe_ep}")
    return {
        "model_id": normalize_model_id(benchmark.get("model"), precision=benchmark.get("precision")),
        "system": normalize_system(benchmark.get("hw")),
        "backend": str(benchmark.get("framework", "vllm")).lower(),
        "backend_build": benchmark.get("image"),
        "tp_size": None if engine_tp else benchmark_tp,
        "pp_size": benchmark.get("pp"),
        "attention_dp_size": None if engine_attention_dp else (moe_ep if dp_attention else 1),
        "dcp_size": benchmark.get("dcp_size"),
        "pcp_size": benchmark.get("pcp_size"),
        "moe_tp_size": world_size // moe_ep,
        "moe_ep_size": moe_ep,
        "quantization": benchmark.get("precision"),
        "speculative_method": benchmark.get("spec_decoding"),
        "concurrency": benchmark.get("conc"),
        "recipe_fingerprint": benchmark.get("recipe_fingerprint"),
    }


def parse_log_text(
    text: str,
    *,
    benchmark: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    artifact_name: str | None = None,
) -> ParsedMeasurement:
    estimated: dict[int, int] = {}
    actual: dict[int, int] = {}
    available_kv: int | None = None
    kv_tokens: int | None = None
    full_counts: set[int] = set()
    full_largest: set[int] = set()
    piecewise_counts: set[int] = set()
    piecewise_largest: set[int] = set()
    for line in text.splitlines():
        rank = _rank(line)
        pool = _POOL_RE.search(line)
        if pool:
            actual[rank] = _bytes_from_gib(pool.group(1))
            estimated[rank] = _bytes_from_gib(pool.group(2))
            continue
        estimate = _ESTIMATE_RE.search(line)
        if estimate:
            estimated[rank] = _bytes_from_gib(estimate.group(1))
        measured = _ACTUAL_RE.search(line)
        if measured:
            actual[rank] = _bytes_from_gib(measured.group(1))
        kv_match = _AVAILABLE_KV_RE.search(line)
        if kv_match:
            available_kv = _bytes_from_gib(kv_match.group(1))
        tokens_match = _KV_TOKENS_RE.search(line)
        if tokens_match:
            kv_tokens = int(tokens_match.group(1).replace(",", ""))
        full_match = _FULL_PROFILE_RE.search(line)
        if full_match:
            full_counts.add(int(full_match.group(1)))
            full_largest.add(int(full_match.group(2)))
        piecewise_match = _PIECEWISE_PROFILE_RE.search(line)
        if piecewise_match:
            piecewise_counts.add(int(piecewise_match.group(1)))
            piecewise_largest.add(int(piecewise_match.group(2)))

    flat_config = _flatten_config(config or {})
    disabled = bool(
        re.search(r"enforce_eager(?:['\"]?\s*[:=]\s*|=)True", text, re.IGNORECASE)
        or re.search(r"--enforce-eager(?:\s|$)", text)
        or flat_config.get("enforce-eager") is True
        or flat_config.get("enforce_eager") is True
    )
    if disabled:
        estimated = {0: 0}
        actual = {0: 0}
    elif not estimated and not actual:
        raise ProfileParseError("log has no CUDA graph reservation, pool measurement, or explicit disabled marker")

    identity, sources = _engine_identity(text)
    fallback_system = normalize_system(artifact_name)
    if fallback_system:
        identity["system"] = fallback_system
        sources["system"] = "artifact_name"

    if benchmark:
        for field, value in _benchmark_identity(benchmark, identity).items():
            if value is not None:
                identity[field] = value
                sources[field] = "benchmark_json"
    if config:
        for field, value in _config_identity(config).items():
            if value is not None:
                identity[field] = value
                sources[field] = "config_yaml"

    if disabled:
        identity["cuda_graph_mode"] = "disabled"
        identity["cuda_graph_capture_sizes"] = "[]"

    identity["cuda_graph_capture_sizes"] = normalize_capture_sizes(identity.get("cuda_graph_capture_sizes"))
    identity["graph_disabled"] = disabled
    identity["schema_version"] = SCHEMA_VERSION
    return ParsedMeasurement(
        identity=identity,
        estimated_bytes_by_rank=estimated,
        actual_bytes_by_rank=actual,
        available_kv_cache_bytes=available_kv,
        gpu_kv_cache_tokens=kv_tokens,
        profiled_full_count=_consistent_profile_value(full_counts, "FULL count"),
        profiled_full_largest_capture_size=_consistent_profile_value(full_largest, "FULL largest capture"),
        profiled_piecewise_count=_consistent_profile_value(piecewise_counts, "PIECEWISE count"),
        profiled_piecewise_largest_capture_size=_consistent_profile_value(
            piecewise_largest, "PIECEWISE largest capture"
        ),
        graph_disabled=disabled,
        identity_sources=sources,
    )


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProfileParseError(f"expected a mapping in {path.name}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProfileParseError(f"expected an object in {path.name}")
    return value
