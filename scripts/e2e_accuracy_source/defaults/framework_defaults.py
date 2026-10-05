# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reviewed effective framework defaults, without importing GPU frameworks.

The source manifest pins the exact files reviewed for every supported tag.
This applies runtime mutations as well as missing defaults: SGLang's DP token
budget and forced page sizes must not be lost to a later ``defaults | args``.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import requests

from e2e_accuracy_source.recipes.inferencex_recipe import InferenceXRecipeError
from e2e_accuracy_source.sources import load_manifest, verify_sources

_SOURCES = load_manifest("framework_default_sources.json")
_BLACKWELL = {"b200", "b300", "gb200", "gb300"}
_HOPPER = {"h100", "h200"}
_KIMI_VLLM_REVISION = "e2fa28594f7baad142a426b0b6a2cfe2c79201c7"
_M3_VLLM_RUNTIME = "0.23.1rc1.dev231+g8b00f4123"


def has_reviewed_defaults(backend: str, version: str | None) -> bool:
    return version in _SOURCES.get(backend, {})


def verified_dynamo_usage_context(
    installation: dict, *, runtime_version: str | None = None
) -> tuple[str | None, list[dict]]:
    """Read-only source evidence; no Dynamo package is installed or imported."""
    if installation.get("install") is not True or installation.get("hash") or installation.get("top_of_tree"):
        return None, []
    requested = [installation[key] for key in ("wheel", "version") if installation.get(key) is not None]
    if not requested or any(not isinstance(value, str) or value != requested[0] for value in requested):
        return None, []
    version = requested[0]
    if version not in _SOURCES.get("dynamo", {}):
        return None, []
    if runtime_version is not None and runtime_version != version:
        return None, []
    records = _verified_sources("dynamo", version)
    return "openai_api_server", records


@cache
def _verified_sources(backend: str, version: str) -> list[dict]:
    records = _SOURCES.get(backend, {}).get(version, [])
    return verify_sources(records)


def _effective_auto_kv_dtype(args: dict, backend: str, version: str, checkpoint: dict) -> str | None:
    """Resolve auto only after backend checkpoint precedence is established."""
    text = checkpoint.get("text_config") or checkpoint
    inline = checkpoint.get("quantization_config") or text.get("quantization_config") or {}
    if backend == "trtllm":
        if "hf_quant_config" not in checkpoint and not checkpoint.get("_hf_quant_config_checked"):
            return None
        companion = checkpoint.get("hf_quant_config")
        if companion is not None:
            if not isinstance(companion, dict):
                return None
            if version == "1.3.0rc14" and "quantization" not in companion:
                return None  # This release's reader requires the legacy wrapper.
            quant = companion.get("quantization", companion)
        else:
            quant = inline
        if not isinstance(quant, dict) or quant.get("quant_algo") == "MIXED_PRECISION":
            return None  # quant_cfg.json can override the global KV quantization.
        algo = quant.get("kv_cache_quant_algo")
        if algo is None and "kv_cache_scheme" in quant:
            if version == "1.3.0rc14":
                return None
            scheme = quant["kv_cache_scheme"]
            if isinstance(scheme, str):
                algo = scheme.upper()
            elif isinstance(scheme, dict):
                algo = {("float", 8): "FP8", ("float", 4): "NVFP4", ("int", 8): "INT8"}.get(
                    (scheme.get("type"), scheme.get("num_bits"))
                )
            if algo is None and scheme is not None:
                return None
        if algo in {"FP8", "NVFP4", "INT8"}:
            return {"FP8": "fp8", "NVFP4": "nvfp4", "INT8": "int8"}[algo]
        if algo not in {None, "NO_QUANT"}:
            return None
        if not quant:
            return None  # dtypes.json is another source when inline metadata is absent.
        if quant.get("quant_method") not in {None, "fp8", "mxfp4", "mxfp8", "modelopt", "compressed-tensors", "nvfp4"}:
            return None
    elif backend == "sglang":
        # ModelOpt and compressed-tensors may carry a separate KV quantization
        # configuration. Only the native unquantized/FP8-weight cases are reviewed.
        if inline.get("quant_method") not in {None, "fp8"} or args.get("quantization") not in {None, "fp8"}:
            return None
        if inline.get("kv_cache_quant_algo") or inline.get("kv_cache_scheme"):
            return None
    else:
        return None
    dtype = args.get("dtype")
    if dtype in {None, "auto"}:
        dtype = text.get("dtype") or text.get("torch_dtype") or checkpoint.get("dtype") or checkpoint.get("torch_dtype")
        if dtype is None and backend == "trtllm":
            dtype = "bfloat16"  # ModelConfig.torch_dtype property, not a predictor guess.
    return {"bf16": "bf16", "bfloat16": "bf16", "fp16": "fp16", "float16": "fp16", "half": "fp16"}.get(dtype)


def apply_framework_defaults(
    args: dict[str, Any],
    backend: str,
    version: str | None,
    *,
    aggregated: bool,
    hardware: str = "",
    checkpoint: dict | None = None,
    checkpoint_weight_dtype: str | None = None,
    usage_context: str | None = None,
) -> tuple[dict[str, Any], list[dict]]:
    """Return effective args and per-knob evidence for reviewed exact versions.

    ``checkpoint`` is the already evidenced checkpoint config, not an inferred
    architecture. Unknown versions, hardware, and model-specific rules remain
    unresolved. Call once on parsed args, after the existing vLLM defaults.
    """
    result = dict(args)
    evidence: list[dict] = []
    if not version or not has_reviewed_defaults(backend, version) or hardware not in _BLACKWELL | _HOPPER:
        return result, evidence
    try:
        records = _verified_sources(backend, version)
    except requests.RequestException as error:
        raise InferenceXRecipeError(f"cannot verify framework defaults: {error}") from error

    def put(knob: str, value: Any, rule: str, *, force: bool = False) -> None:
        previous = result.get(knob)
        if (previous is None or force) and previous != value:
            result[knob] = value
            evidence.append(
                dict(
                    knob=knob,
                    version=version,
                    value=value,
                    previous=previous,
                    kind="runtime_override" if previous is not None else "verified_default",
                    rule=rule,
                    sources=records,
                )
            )

    config = checkpoint or {}
    architecture = next(iter(config.get("architectures") or []), None)
    text_config = config.get("text_config") or config
    if backend == "vllm":
        if version == _M3_VLLM_RUNTIME:
            # This source identity is obtained from matched worker logs. Keep
            # these rules separate from non-sparse MiniMax-M2 and MLA Kimi.
            if (
                architecture != "MiniMaxM3SparseForConditionalGeneration"
                or args.get("runtime_framework_version") != version
                or not aggregated
                or args.get("runner") not in {None, "auto", "generate"}
                or args.get("model_impl") not in {None, "auto", "vllm"}
                or args.get("hf_overrides")
                or args.get("speculative_config")
                or args.get("recipe_environment", {}).get("VLLM_PLUGINS")
                or args.get("performance_mode") not in {None, "balanced"}
            ):
                return result, evidence
            tokens = result.get("max_num_batched_tokens")
            if type(tokens) is int and tokens > 0:
                put("max_num_seqs", min(1024, tokens), "recorded M3 serve runtime -> >=70 GiB default -> token cap")
            if result.get("kv_cache_dtype") == "auto" and args.get("dtype") in {"bfloat16", "bf16"}:
                put(
                    "kv_cache_dtype",
                    "bf16",
                    "M3 sparse layer kv_cache_dtype_str_to_dtype(auto) -> recorded model dtype",
                    force=True,
                )
            return result, evidence
        kimi_mla = version == _KIMI_VLLM_REVISION and architecture == "KimiK25ForConditionalGeneration"
        if not kimi_mla and architecture not in {"MiniMaxM2ForCausalLM", "GptOssForCausalLM"}:
            return result, evidence
        environment = args.get("recipe_environment", {})
        if (
            args.get("runner") not in {None, "auto", "generate"}
            or args.get("task") not in {None, "auto", "generate"}
            or args.get("speculative_config")
            or args.get("model_impl") not in {None, "auto", "vllm"}
            or args.get("hf_overrides")
            or environment.get("VLLM_PLUGINS")
            or config.get("is_encoder_decoder")
            or (text_config.get("kv_lora_rank") and not kimi_mla)
            or (kimi_mla and (config.get("is_mm_prefix_lm") or text_config.get("is_mm_prefix_lm")))
        ):
            return result, evidence
        if version == "0.10.2":
            # Its V1 oracle falls back for these legacy execution options.
            # Scope this path to the ordinary CUDA GPT-OSS server only.
            unsupported = (
                "logits_processor_pattern",
                "preemption_mode",
                "disable_async_output_proc",
                "scheduler_delay_factor",
                "enable_prompt_embeds",
                "max_num_partial_prefills",
                "max_long_partial_prefills",
                "speculative_config",
            )
            if (
                architecture != "GptOssForCausalLM"
                or args.get("kv_cache_dtype") not in {None, "auto"}
                or environment.get("VLLM_USE_V1") not in {None, "1"}
                or args.get("load_format") == "sharded_state"
                or any(args.get(key) is not None for key in unsupported)
                or args.get("distributed_executor_backend") not in {None, "mp", "ray", "external_launcher"}
                or environment.get("VLLM_ATTENTION_BACKEND") not in {None, "FLASH_ATTN", "FLASHINFER"}
            ):
                return result, evidence
            put(
                "enable_chunked_prefill", True, "EngineArgs V1 oracle -> non-pooling forced chunked prefill", force=True
            )
        else:
            put("enable_chunked_prefill", True, "ModelConfig generative decoder -> EngineArgs default")
        dtype = result.get("kv_cache_dtype", "auto")
        quant = config.get("quantization_config") or {}
        if (
            dtype == "auto"
            and quant.get("quant_method") in {None, "fp8", "mxfp4"}
            and not any(quant.get(k) for k in ("kv_cache_scheme", "kv_cache_quant_algo", "quantization"))
        ):
            model_dtype = args.get("dtype")
            if model_dtype in {None, "auto"}:
                model_dtype = text_config.get("dtype") or text_config.get("torch_dtype") or checkpoint_weight_dtype
                if model_dtype == "float32":
                    model_dtype = "bfloat16"  # CUDA SM80+ preferred dtype for generation.
            effective_dtype = {
                "bfloat16": "bf16",
                "bf16": "bf16",
                "float16": "fp16",
                "half": "fp16",
                "fp16": "fp16",
            }.get(model_dtype)
            if effective_dtype:
                put(
                    "kv_cache_dtype",
                    effective_dtype,
                    "unquantized auto KV cache -> ModelConfig compute dtype",
                    force=True,
                )
        attention = args.get("attention_backend") or environment.get("VLLM_ATTENTION_BACKEND")
        attention_config = args.get("attention_config") or {}
        if kimi_mla:
            # The reviewed recipe explicitly uses FlashInfer prefill, proving
            # the dependency is present. SM100's first valid dense-MLA backend
            # is FLASHINFER_MLA; its supported pages are [32, 64], so inherited
            # get_preferred_block_size(16) selects 32. Prefill and decode backend
            # names are distinct controls.
            compute_dtype = args.get("dtype")
            if compute_dtype in {None, "auto"}:
                compute_dtype = text_config.get("dtype") or text_config.get("torch_dtype") or config.get("dtype")
            if (
                hardware in _BLACKWELL
                and attention in {None, "FLASHINFER_MLA"}
                and attention_config.get("backend") in {None, "auto", "FLASHINFER_MLA"}
                and not attention_config.get("backend_per_kind")
                and attention_config.get("mla_prefill_backend") == "FLASHINFER"
                and not attention_config.get("use_non_causal")
                and text_config.get("kv_lora_rank") == 512
                and text_config.get("qk_rope_head_dim") == 64
                and text_config.get("qk_nope_head_dim") in {64, 128, 192}
                and not text_config.get("index_topk")
                and compute_dtype in {"bfloat16", "bf16", "float16", "fp16", "half"}
                and result.get("kv_cache_dtype") in {"fp8", "fp8_e4m3", "bfloat16", "float16"}
                and not args.get("kv_cache_dtype_skip_layers")
            ):
                put("block_size", 32, "SM100 dense Kimi MLA -> FLASHINFER_MLA -> preferred supported page size 32")
        # Every standard non-MLA candidate for these models accepts the default
        # 16: FlashAttention, FlashInfer, Triton, FlexAttention. No hybrid alignment.
        if (
            not kimi_mla
            and attention in {None, "FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION"}
            and result.get("kv_cache_dtype") in {"bf16", "bfloat16", "fp16", "float16", "fp8", "fp8_e4m3", "fp8_e5m2"}
        ):
            put("block_size", 16, "CacheConfig -> CUDA backend preference -> non-hybrid KV cache spec")
        # v0.20.1 generation does not read tokenizer model_max_length. This
        # narrow MiniMax path has no RoPE scaling or sliding-window mutation.
        # Preserve explicit model lengths, including auto-fit (-1), untouched.
        if (
            version == "0.20.1"
            and architecture == "MiniMaxM2ForCausalLM"
            and result.get("max_model_len") is None
            and not args.get("disable_sliding_window")
            and not args.get("rope_scaling")
            and not text_config.get("rope_scaling")
            and not text_config.get("rope_parameters")
            and not text_config.get("sliding_window")
        ):
            length_keys = (
                "max_position_embeddings",
                "n_positions",
                "max_seq_len",
                "seq_length",
                "model_max_length",
                "max_target_positions",
                "max_sequence_length",
                "max_seq_length",
                "seq_len",
            )
            lengths = [text_config[key] for key in length_keys if text_config.get(key) is not None]
            if lengths and all(type(value) is int and value > 0 for value in lengths):
                max_len = text_config.get("model_max_length") or min(lengths)
                put(
                    "max_model_len",
                    max_len,
                    "v0.20.1 generative MiniMax ModelConfig -> checkpoint-derived model length",
                )
        # A worker may use the same serve context, but only source verification
        # of its exact installed package establishes that fact.
        if (
            (aggregated or usage_context == "openai_api_server")
            and args.get("performance_mode") in {None, "balanced"}
            and not (
                args.get("enable_expert_parallel")
                and int(args.get("data_parallel_size", 1)) > 1
                and args.get("all2all_backend") in {"deepep_low_latency", "nixl_ep"}
            )
        ):
            max_len = result.get("max_model_len")
            seqs = args.get("max_num_seqs", 1024)
            if isinstance(max_len, int) and max_len > 0 and isinstance(seqs, int) and seqs > 0:
                tokens = 8192 if result.get("enable_chunked_prefill") else max(8192, max_len)
                if version == "0.10.2":
                    put("max_num_batched_tokens", 8192, "V1 OpenAI serve >=70 GiB with forced chunked prefill")
                else:
                    put(
                        "max_num_batched_tokens",
                        min(tokens, seqs * max_len),
                        "OpenAI serve >=70 GiB -> model length cap",
                    )
            tokens = result.get("max_num_batched_tokens")
            if isinstance(tokens, int) and tokens > 0:
                put("max_num_seqs", min(1024, tokens), "OpenAI serve >=70 GiB -> token cap")
        return result, evidence
    if backend == "trtllm":
        # These rules cover the PyTorch API and trtllm-serve; legacy TensorRT
        # engines and custom architecture/plugin hooks require their own audit.
        if args.get("backend", "pytorch") != "pytorch" or args.get("speculative_config"):
            return result, evidence
        attn = args.get("attn_backend", "TRTLLM").upper()
        kv = args.get("kv_cache_config") or {}
        for field, target in (
            ("tokens_per_block", "block_size"),
            ("dtype", "kv_cache_dtype"),
            ("enable_block_reuse", "enable_prefix_caching"),
            ("free_gpu_memory_fraction", "free_gpu_memory_fraction"),
        ):
            if kv.get(field) is not None:
                put(target, kv[field], f"explicit kv_cache_config.{field}", force=True)
        for field, target in (("max_batch_size", "max_num_seqs"), ("max_num_tokens", "max_num_batched_tokens")):
            if args.get(field) is not None:
                put(target, args[field], f"explicit {field}", force=True)
        put("max_num_seqs", 2048, "serve BuildConfig/TorchLlmArgs -> get_runtime_sizes")
        put("max_num_batched_tokens", 8192, "serve BuildConfig/TorchLlmArgs -> get_runtime_sizes")
        put("enable_chunked_prefill", False, "serve CLI and BaseLlmArgs -> executor enable_chunked_context")
        put("free_gpu_memory_fraction", 0.9, "KvCacheConfig -> KV cache allocation")
        # auto follows checkpoint quantization metadata; preserving the sentinel
        # is intentional. Predictor mapping must resolve it independently.
        put("kv_cache_dtype", "auto", "KvCacheConfig.dtype follows checkpoint metadata")
        if result.get("kv_cache_dtype") == "auto" and checkpoint:
            dtype = _effective_auto_kv_dtype(result, backend, version, checkpoint)
            if dtype is not None:
                put(
                    "kv_cache_dtype",
                    dtype,
                    "model_loader auto preserves checkpoint KV quant; unquantized cache uses model dtype",
                    force=True,
                )
        sparse = args.get("sparse_attention_config")
        if attn == "TRTLLM" and not sparse and hardware in _BLACKWELL:
            put("block_size", 32, "KvCacheConfig -> executor; FlashMLA override only on SM90")
        elif attn == "TRTLLM" and not sparse and checkpoint:
            mla = text_config.get("kv_lora_rank") and text_config.get("qk_rope_head_dim")
            if mla and text_config["kv_lora_rank"] + text_config["qk_rope_head_dim"] == 576:
                put("block_size", 64, "SM90 MLA head_dim=576 -> enable_flash_mla -> executor", force=True)
            else:
                put("block_size", 32, "KvCacheConfig -> executor; no FlashMLA model override")
        # Prefix caching may be forcibly disabled for hybrid/disaggregated
        # models and some attention backends. Resolve only reviewed conditions.
        hybrid = architecture in {
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3_5MoeForCausalLM",
            "Qwen3NextForCausalLM",
            "Qwen3_5ForCausalLM",
        }
        if args.get("recipe_environment", {}).get("FORCE_DETERMINISTIC") == "1":
            put("enable_prefix_caching", False, "executor FORCE_DETERMINISTIC=1", force=True)
        elif hybrid and args.get("cache_transceiver_config", {}).get("backend"):
            put("enable_prefix_caching", False, "executor hybrid model with cache transceiver", force=True)
        elif attn in {"FLASHINFER", "FLASHINFER_STAR_ATTENTION"} and version != "1.3.0rc24":
            put("enable_prefix_caching", False, "executor disables block reuse for FlashInfer", force=True)
        elif (
            attn == "TRTLLM"
            and not sparse
            and checkpoint
            and not config.get("is_encoder_decoder")
            and (
                not (text_config.get("kv_lora_rank") and text_config.get("qk_rope_head_dim"))
                or result.get("kv_cache_dtype") in {"fp8", "fp8_e4m3", "bfloat16", "bf16", "float16", "fp16"}
            )
        ):
            put("enable_prefix_caching", True, "KvCacheConfig.enable_block_reuse; reviewed NVIDIA execution path")
        return result, evidence

    if backend != "sglang" or not checkpoint:
        return result, evidence
    # Imported model hooks and speculative/embedding paths need separate rules.
    reviewed_architectures = {
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
        "KimiK25ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3NextForCausalLM",
        "GptOssForCausalLM",
    }
    if architecture not in reviewed_architectures or args.get("speculative_algorithm") or args.get("is_embedding"):
        return result, evidence
    if (
        args.get("enable_multi_item_scoring")
        or args.get("enable_nsa_prefill_context_parallel")
        or args.get("model_impl") not in {None, "auto", "sglang"}
        or args.get("enable_dynamic_chunking")
    ):
        return result, evidence
    hybrid = architecture.startswith("Qwen3")
    dsa = architecture in {"GlmMoeDsaForCausalLM", "DeepseekV32ForCausalLM"} or bool(text_config.get("index_topk"))
    chunk = args.get("chunked_prefill_size")
    if chunk is None:
        chunk = 16384 if hardware in _BLACKWELL else 8192
        put("chunked_prefill_size", chunk, "ServerArgs GPU memory tier: >160 GiB / 60–160 GiB")  # noqa: RUF001
    if isinstance(chunk, int):
        dp = int(args.get("data_parallel_size", args.get("dp_size", 1))) if args.get("enable_dp_attention") else 1
        if dp <= 0:
            raise InferenceXRecipeError("SGLang DP size must be positive")
        effective_chunk = chunk // dp
        put("effective_chunked_prefill_size", effective_chunk, "ServerArgs._handle_data_parallelism: chunk // dp")
        put("enable_chunked_prefill", effective_chunk > 0, "scheduler uses positive chunked_prefill_size", force=True)
        if effective_chunk > 0:
            # max_prefill_tokens is a separate admission control, not a hard
            # min with this value: the first request may exceed it. Preserve it.
            put("max_prefill_tokens", 16384, "ServerArgs.max_prefill_tokens admission control")
            put(
                "max_num_batched_tokens",
                effective_chunk,
                "DP-adjusted chunked_prefill_size; separate max_prefill_tokens retained",
                force=True,
            )
        elif result.get("max_num_batched_tokens") == chunk:
            # The recipe adapter may have copied the -1 sentinel into this
            # canonical field. A disabled chunk is not a negative token cap.
            result.pop("max_num_batched_tokens", None)
            evidence.append(
                dict(
                    knob="max_num_batched_tokens",
                    version=version,
                    value=None,
                    previous=chunk,
                    kind="unresolved",
                    sources=records,
                    rule="disabled chunked prefill has no inferred hard batched-token cap",
                )
            )
    if args.get("max_running_requests") is not None:
        put("max_num_seqs", args["max_running_requests"], "explicit max_running_requests", force=True)
    # None is a runtime KV-capacity-derived request limit. Do not replace it
    # with benchmark concurrency or the often-quoted constant 4096.
    if not hybrid or args.get("disable_radix_cache") is not None:
        put("enable_prefix_caching", not args.get("disable_radix_cache", False), "ServerArgs.disable_radix_cache=False")
    if args.get("recipe_environment", {}).get("SGLANG_RADIX_FORCE_MISS") == "1":
        put("enable_prefix_caching", False, "SGLANG_RADIX_FORCE_MISS=1", force=True)
    dtype = args.get("kv_cache_dtype", "auto")
    if dsa and dtype in {"auto", "bf16"}:
        dtype = "bfloat16" if dtype == "bf16" or hardware in _HOPPER else "fp8_e4m3"
        put("kv_cache_dtype", dtype, "DSA _set_default_nsa_kv_cache_dtype by compute capability", force=True)
    else:
        put("kv_cache_dtype", "auto", "ServerArgs.kv_cache_dtype; no reviewed model override")
    if result.get("kv_cache_dtype") == "auto":
        effective_dtype = _effective_auto_kv_dtype(result, backend, version, checkpoint)
        if effective_dtype is not None:
            put(
                "kv_cache_dtype",
                effective_dtype,
                "ModelRunner.configure_kv_cache_dtype -> model compute dtype for native unquantized/FP8",
                force=True,
            )
    attention = args.get("attention_backend")
    prefill = args.get("prefill_attention_backend")
    decode = args.get("decode_attention_backend")
    if not any((attention, prefill, decode)) and hardware in _BLACKWELL and not hybrid:
        attention = "nsa" if dsa else "trtllm_mha" if architecture == "GptOssForCausalLM" else "trtllm_mla"
        put("attention_backend", attention, "ServerArgs model-specific Blackwell attention backend")
    page = args.get("page_size", args.get("block_size"))
    if dsa:
        page = 64
    if attention == "flashmla" or decode == "flashmla":
        page = 64
    if attention == "cutlass_mla" or decode == "cutlass_mla":
        page = 128
    if attention == "trtllm_mla" or decode == "trtllm_mla":
        page = page if page in {32, 64} else 64
    if "trtllm_mha" in {attention, prefill, decode}:
        page = page if page in {16, 32, 64} else 64
    if (
        page is None
        and (not hybrid or args.get("disable_radix_cache"))
        and attention in {"flashinfer", "fa3", "triton", "nsa"}
    ):
        page = 1
    if hybrid and not args.get("disable_radix_cache"):
        # Mamba radix strategies changed across these releases; do not apply
        # the ordinary attention page rule without reviewing that strategy.
        page = None
    if page is not None:
        put("block_size", page, "model/attention page-size overrides -> _handle_page_size", force=True)
    return result, evidence
