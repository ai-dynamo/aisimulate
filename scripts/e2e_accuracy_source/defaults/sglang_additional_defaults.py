# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-release SGLang page/chunk rules for the remaining InferenceX recipes.

This intentionally does not infer memory-profiled max_running_requests or use
nightly image tag SHAs as installed framework revisions.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import requests

from e2e_accuracy_source.recipes.inferencex_recipe import InferenceXRecipeError
from e2e_accuracy_source.sources import load_manifest, verify_sources

_SOURCES = load_manifest("sglang_additional_default_sources.json")
_QWEN = {"Qwen3_5MoeForConditionalGeneration", "Qwen3_5ForConditionalGeneration"}


def has_additional_sglang_defaults(version: str | None) -> bool:
    return version in _SOURCES


@cache
def _verified_sources(version: str) -> list[dict]:
    records = _SOURCES[version]["sources"]
    try:
        verify_sources(records)
    except requests.RequestException as error:
        raise InferenceXRecipeError(f"cannot verify SGLang defaults: {error}") from error
    return records


def verified_sglang_fp4_expert_profile(
    checkpoint: dict, args: dict, version: str | None, hardware: str | None
) -> tuple[str | None, dict]:
    """Map the measured DSV4 MXFP4/MXFP8 runner, never infer it from FP4 weights."""
    if (
        not _SOURCES.get(version, {}).get("fp4_moe_reviewed")
        or hardware not in {"b200", "b300", "gb200", "gb300"}
        or checkpoint.get("architectures") != ["DeepseekV4ForCausalLM"]
        or checkpoint.get("expert_dtype") != "fp4"
        or args.get("runtime_is_fp4_experts") is not True
        or args.get("moe_runner_backend") != "flashinfer_mxfp4"
        or args.get("flashinfer_mxfp4_moe_precision") != "default"
        or args.get("recipe_environment", {}).get("SGLANG_DSV4_FP4_DEQUANT") not in {None, "0"}
    ):
        return None, {}
    records = _verified_sources(version)
    return "w4a8_mxfp4_mxfp8", dict(
        kind="verified_runtime_kernel_mapping",
        version=version,
        source_revision=_SOURCES[version]["commit"],
        rule="detected FP4 experts -> Fp8Config -> Blackwell FlashInfer TRTLLM MXFP4 with default MXFP8 activations",
        sources=records,
    )


def apply_additional_sglang_defaults(
    args: dict[str, Any],
    version: str | None,
    *,
    hardware: str,
    checkpoint: dict | None,
) -> tuple[dict[str, Any], list[dict]]:
    """Resolve explicit TRT attention backends and explicit chunk budgets.

    Pass a checkpoint from the recorded model revision. Return the full effective
    args (including runtime overrides), then append evidence to role provenance.
    Unknown releases and model/backend modes remain unchanged.
    """
    result, evidence = dict(args), []
    if not has_additional_sglang_defaults(version) or hardware not in {"b200", "b300", "gb200", "gb300"}:
        return result, evidence
    architecture = next(iter((checkpoint or {}).get("architectures") or []), None)
    if _SOURCES[version].get("runtime_only"):
        # These identities come from checked worker fingerprints, not image tags.
        # The runtime adapter must establish this value from every worker's
        # scheduler summary. It already includes attention-DP division.
        chunk = args.get("effective_chunked_prefill_size")
        if (
            architecture not in _QWEN | {"DeepseekV4ForCausalLM"}
            or args.get("model_impl") not in {None, "auto", "sglang"}
            or args.get("speculative_algorithm")
            or args.get("enable_dynamic_chunking")
            or not isinstance(chunk, int)
            or isinstance(chunk, bool)
        ):
            return result, evidence
        records = _verified_sources(version)
        enabled = chunk > 0
        previous = result.get("enable_chunked_prefill")
        if previous != enabled:
            result["enable_chunked_prefill"] = enabled
            evidence.append(
                dict(
                    knob="enable_chunked_prefill",
                    value=enabled,
                    previous=previous,
                    version=version,
                    source_revision=_SOURCES[version]["commit"],
                    kind="runtime_override" if previous is not None else "verified_default",
                    rule="checked scheduler chunk -> native Scheduler.init_chunked_prefill positive-chunk branch",
                    sources=records,
                )
            )
        return result, evidence
    supported = {"DeepseekV3ForCausalLM"} | (_QWEN if version != "0.5.3rc1" else set())
    if (
        architecture not in supported
        or args.get("model_impl") not in {None, "auto", "sglang"}
        or any(
            args.get(key)
            for key in (
                "speculative_algorithm",
                "speculative_config",
                "enable_prefill_cp",
                "enable_dynamic_chunking",
                "enable_multi_item_scoring",
                "dllm_algorithm",
                "is_embedding",
                "json_model_override_args",
                "hf_overrides",
            )
        )
    ):
        return result, evidence
    # This Qwen path avoids Mamba cache strategies that constrain page_size.
    if architecture in _QWEN and args.get("disable_radix_cache") is not True:
        return result, evidence
    backend = args.get("attention_backend")
    expected = "trtllm_mha" if architecture in _QWEN else "trtllm_mla"
    if backend != expected or any(
        args.get(key) not in {None, backend} for key in ("prefill_attention_backend", "decode_attention_backend")
    ):
        return result, evidence
    records = _verified_sources(version)

    def put(key: str, value: Any, rule: str) -> None:
        previous = result.get(key)
        if previous != value:
            result[key] = value
            evidence.append(
                dict(
                    knob=key,
                    value=value,
                    previous=previous,
                    version=version,
                    source_revision=_SOURCES[version]["commit"],
                    kind="runtime_override" if previous is not None else "verified_default",
                    rule=rule,
                    sources=records,
                )
            )

    page = args.get("page_size", args.get("block_size"))
    supported_pages = {32, 64} if backend == "trtllm_mla" else {16, 32, 64}
    if backend == "trtllm_mha" and version == "0.5.19":
        supported_pages.add(128)
    effective_page = page if page in supported_pages else 64
    put("block_size", effective_page, "ServerArgs -> TRT attention page constraints -> KV allocator")
    if "page_size" in result:
        put("page_size", effective_page, "TRT attention backend overrides unsupported page_size")

    chunk = args.get("chunked_prefill_size")
    dp = args.get("dp_size", args.get("data_parallel_size", 1))
    if isinstance(chunk, int) and not isinstance(chunk, bool) and isinstance(dp, int) and dp > 0:
        effective_chunk = chunk // dp if dp > 1 and args.get("enable_dp_attention") else chunk
        # Preserve the CLI value so applying defaults twice cannot divide twice.
        put("effective_chunked_prefill_size", effective_chunk, "ServerArgs DP attention divides chunked_prefill_size")
        put("enable_chunked_prefill", effective_chunk > 0, "Scheduler.init_chunked_prefill disables nonpositive chunks")
        if effective_chunk > 0:
            put("max_num_batched_tokens", effective_chunk, "effective per-DP-rank chunked_prefill_size")
        elif result.get("max_num_batched_tokens") == chunk:
            # Normalization copies this CLI sentinel into the canonical field.
            # Disabled chunking supplies no hard batched-token cap.
            result.pop("max_num_batched_tokens", None)
            evidence.append(
                dict(
                    knob="max_num_batched_tokens",
                    value=None,
                    previous=chunk,
                    version=version,
                    source_revision=_SOURCES[version]["commit"],
                    kind="unresolved",
                    rule="disabled chunked prefill has no inferred hard batched-token cap",
                    sources=records,
                )
            )
    return result, evidence
