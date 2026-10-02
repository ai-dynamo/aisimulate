# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression cases derived from the September 14 resolved recipe audit."""

import pytest
from e2e_accuracy_source import framework_defaults as defaults
from e2e_accuracy_source.inferencex_recipe import InferenceXRecipeError


@pytest.fixture(autouse=True)
def pinned_sources(monkeypatch):
    monkeypatch.setattr(defaults, "_verified_sources", lambda backend, version: defaults._SOURCES[backend][version])


def apply(args, backend="sglang", version="0.5.12", architecture="DeepseekV3ForCausalLM", **kwargs):
    return defaults.apply_framework_defaults(
        args,
        backend,
        version,
        aggregated=kwargs.pop("aggregated", True),
        hardware=kwargs.pop("hardware", "b200"),
        checkpoint={"architectures": [architecture], **kwargs.pop("checkpoint", {})},
        **kwargs,
    )


def test_sglang_explicit_dp_chunk_is_runtime_adjusted():
    args = {
        "chunked_prefill_size": 32768,
        "max_num_batched_tokens": 32768,
        "max_running_requests": 512,
        "max_prefill_tokens": 32768,
        "enable_dp_attention": True,
        "data_parallel_size": 4,
        "disable_radix_cache": True,
        "attention_backend": "flashinfer",
    }
    resolved, evidence = apply(args)
    assert resolved["max_num_batched_tokens"] == 8192
    assert resolved["effective_chunked_prefill_size"] == 8192
    assert resolved["chunked_prefill_size"] == 32768  # retain original argument
    assert resolved["max_num_seqs"] == 512
    assert resolved["enable_chunked_prefill"] is True
    assert resolved["block_size"] == 1
    assert args["max_num_batched_tokens"] == 32768
    assert any(e["knob"] == "max_num_batched_tokens" and e["kind"] == "runtime_override" for e in evidence)


def test_prefill_admission_limit_is_not_a_hard_chunk_minimum():
    resolved, _ = apply({"chunked_prefill_size": 32768, "attention_backend": "flashinfer"})
    assert resolved["max_prefill_tokens"] == 16384
    assert resolved["max_num_batched_tokens"] == 32768


@pytest.mark.parametrize("hardware, expected", [("b200", 16384), ("h200", 8192)])
def test_memory_tier_default_and_dynamic_request_limit(hardware, expected):
    resolved, _ = apply({}, hardware=hardware)
    assert resolved["max_num_batched_tokens"] == expected
    assert "max_num_seqs" not in resolved  # runtime KV capacity, not concurrency
    assert "mem_fraction_static" not in resolved


@pytest.mark.parametrize("version", ["0.5.11", "0.5.12"])
def test_glm_dsa_runtime_overrides_explicit_page_and_auto_dtype(version):
    resolved, _ = apply(
        {"kv_cache_dtype": "auto", "block_size": 1},
        version=version,
        architecture="GlmMoeDsaForCausalLM",
        aggregated=False,
    )
    assert resolved["block_size"] == 64
    assert resolved["kv_cache_dtype"] == "fp8_e4m3"


def test_qwen_explicit_radix_disabled_trt_attention():
    resolved, _ = apply(
        {"disable_radix_cache": True, "attention_backend": "trtllm_mha", "chunked_prefill_size": 8192},
        version="0.5.14",
        architecture="Qwen3_5MoeForConditionalGeneration",
    )
    assert resolved["block_size"] == 64
    assert resolved["enable_chunked_prefill"] is True


def test_unknown_model_hook_does_not_receive_generic_defaults():
    resolved, evidence = apply({}, architecture="DeepseekV4ForCausalLM")
    assert resolved == {}
    assert evidence == []


def test_unknown_release_is_not_replaced_by_nearest_release():
    resolved, evidence = apply({}, version="0.5.13")
    assert resolved == {}
    assert evidence == []


def test_trt_blackwell_pytorch_defaults_and_explicit_values():
    resolved, _ = apply(
        {
            "max_num_tokens": 16640,
            "max_batch_size": 2,
            "kv_cache_config": {"dtype": "fp8", "enable_block_reuse": False, "free_gpu_memory_fraction": 0.6},
        },
        backend="trtllm",
        version="1.3.0rc18",
        aggregated=False,
    )
    assert resolved["block_size"] == 32
    assert resolved["enable_chunked_prefill"] is False
    assert resolved["max_num_batched_tokens"] == 16640
    assert resolved["max_num_seqs"] == 2
    assert resolved["free_gpu_memory_fraction"] == 0.6
    assert resolved["enable_prefix_caching"] is False


def test_trt_hopper_mla_forces_64_even_over_explicit_32():
    resolved, _ = apply(
        {"block_size": 32},
        backend="trtllm",
        version="1.3.0rc14",
        hardware="h100",
        checkpoint={"kv_lora_rank": 512, "qk_rope_head_dim": 64},
    )
    assert resolved["block_size"] == 64


def test_trt_sparse_block_size_stays_unknown():
    resolved, _ = apply({"sparse_attention_config": {"algorithm": "minimax_m3"}}, backend="trtllm", version="1.3.0rc24")
    assert "block_size" not in resolved


def test_trt_hybrid_disagg_forces_cache_reuse_off():
    resolved, _ = apply(
        {"enable_prefix_caching": True, "cache_transceiver_config": {"backend": "UCX"}},
        backend="trtllm",
        version="1.3.0rc18",
        architecture="Qwen3_5MoeForConditionalGeneration",
        aggregated=False,
    )
    assert resolved["enable_prefix_caching"] is False


def test_invalid_dp_is_not_silently_corrected():
    with pytest.raises(InferenceXRecipeError, match="DP size must be positive"):
        apply({"enable_dp_attention": True, "data_parallel_size": 0})


def test_disabled_chunk_does_not_become_negative_budget():
    resolved, evidence = apply({"chunked_prefill_size": -1, "max_num_batched_tokens": -1})
    assert resolved["enable_chunked_prefill"] is False
    assert "max_num_batched_tokens" not in resolved
    assert any(e["kind"] == "unresolved" for e in evidence)


def test_trt_rc21_defaults_are_reviewed_independently():
    resolved, evidence = apply(
        {"max_batch_size": 2, "max_num_tokens": 16640}, backend="trtllm", version="1.3.0rc21", aggregated=False
    )
    assert resolved["block_size"] == 32
    assert resolved["enable_chunked_prefill"] is False
    assert all("v1.3.0rc21/" in s["url"] for e in evidence for s in e["sources"])


def test_trt_auto_requires_sidecar_presence_check():
    resolved, _ = apply(
        {"kv_cache_dtype": "auto"},
        backend="trtllm",
        version="1.3.0rc21",
        checkpoint={"dtype": "bfloat16", "quantization_config": {"quant_method": "fp8"}},
    )
    assert resolved["kv_cache_dtype"] == "auto"


def test_trt_authoritative_sidecar_kv_overrides_inline_auto():
    resolved, _ = apply(
        {"kv_cache_config": {"dtype": "auto"}},
        backend="trtllm",
        version="1.3.0rc21",
        checkpoint={
            "dtype": "bfloat16",
            "quantization_config": {"quant_method": "mxfp4"},
            "hf_quant_config": {"quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8"}},
        },
    )
    assert resolved["kv_cache_dtype"] == "fp8"
    assert resolved["kv_cache_config"]["dtype"] == "auto"


def test_trt_gptoss_no_sidecar_uses_framework_bf16_fallback():
    resolved, _ = apply(
        {},
        backend="trtllm",
        version="1.3.0rc14",
        architecture="GptOssForCausalLM",
        checkpoint={"quantization_config": {"quant_method": "mxfp4"}, "hf_quant_config": None},
    )
    assert resolved["kv_cache_dtype"] == "bf16"


def test_trt_mixed_quant_cfg_precedence_is_unresolved():
    resolved, _ = apply(
        {},
        backend="trtllm",
        version="1.3.0rc21",
        checkpoint={
            "hf_quant_config": {"quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": "FP8"}}
        },
    )
    assert resolved["kv_cache_dtype"] == "auto"


@pytest.mark.parametrize("quantization", [{}, {"quant_method": "fp8", "weight_block_size": [128, 128]}])
def test_sglang_fp8_weights_do_not_imply_fp8_auto_kv(quantization):
    resolved, _ = apply({}, checkpoint={"dtype": "bfloat16", "quantization_config": quantization})
    assert resolved["kv_cache_dtype"] == "bf16"


@pytest.mark.parametrize("version", ["0.19.0", "0.20.1", "0.21.0", "0.22.0"])
def test_vllm_minimax_serve_effective_defaults(version):
    resolved, evidence = apply(
        {"max_model_len": 2048},
        backend="vllm",
        version=version,
        architecture="MiniMaxM2ForCausalLM",
        checkpoint={"torch_dtype": "bfloat16"},
    )
    assert resolved["block_size"] == 16
    assert resolved["kv_cache_dtype"] == "bf16"
    assert resolved["enable_chunked_prefill"] is True
    assert resolved["max_num_batched_tokens"] == 8192
    assert resolved["max_num_seqs"] == 1024
    assert all(record["sources"] for record in evidence)


def test_vllm_explicit_limits_and_cache_settings_survive():
    args = dict(
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=16384,
        block_size=64,
        kv_cache_dtype="fp8",
        enable_chunked_prefill=False,
    )
    resolved, _ = apply(args, backend="vllm", version="0.22.0", architecture="MiniMaxM2ForCausalLM")
    assert resolved == args


def test_vllm_default_tokens_follow_model_length_and_chunking():
    resolved, _ = apply(
        dict(max_model_len=16384, max_num_seqs=1, enable_chunked_prefill=False),
        backend="vllm",
        version="0.21.0",
        architecture="MiniMaxM2ForCausalLM",
    )
    assert resolved["max_num_batched_tokens"] == 16384
    resolved, _ = apply(
        dict(max_model_len=512, max_num_seqs=2),
        backend="vllm",
        version="0.21.0",
        architecture="MiniMaxM2ForCausalLM",
    )
    assert resolved["max_num_batched_tokens"] == 1024


def test_vllm_worker_does_not_inherit_openai_scheduler_defaults():
    resolved, _ = apply(
        dict(max_model_len=2048, kv_cache_dtype="fp8"),
        aggregated=False,
        backend="vllm",
        version="0.20.1",
        architecture="MiniMaxM2ForCausalLM",
    )
    assert resolved["block_size"] == 16
    assert "max_num_seqs" not in resolved
    assert "max_num_batched_tokens" not in resolved


@pytest.mark.parametrize(
    "quant",
    [
        {"quant_method": "modelopt"},
        {"quant_method": "mxfp4", "kv_cache_quant_algo": "FP8"},
    ],
)
def test_vllm_auto_kv_does_not_override_unreviewed_checkpoint_quantization(quant):
    resolved, _ = apply(
        {"kv_cache_dtype": "auto"},
        backend="vllm",
        version="0.22.0",
        architecture="GptOssForCausalLM",
        checkpoint={"torch_dtype": "bfloat16", "quantization_config": quant},
    )
    assert resolved["kv_cache_dtype"] == "auto"
    assert "block_size" not in resolved


def test_vllm_gptoss_weight_quantization_does_not_quantize_kv_cache():
    resolved, _ = apply(
        {"kv_cache_dtype": "auto"},
        backend="vllm",
        version="0.22.0",
        architecture="GptOssForCausalLM",
        hardware="h200",
        checkpoint={"torch_dtype": "bfloat16", "quantization_config": {"quant_method": "mxfp4"}},
    )
    assert resolved["kv_cache_dtype"] == "bf16"
    assert resolved["block_size"] == 16


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": "0.23.0"},
        {"architecture": "DeepseekV3ForCausalLM"},
        {"hardware": "a100"},
    ],
)
def test_vllm_review_scope_is_exact(overrides):
    options = dict(backend="vllm", version="0.22.0", architecture="MiniMaxM2ForCausalLM") | overrides
    resolved, evidence = apply({}, **options)
    assert resolved == {}
    assert evidence == []


def test_vllm_missing_config_dtype_uses_verified_weight_metadata():
    options = dict(
        backend="vllm",
        version="0.22.0",
        architecture="GptOssForCausalLM",
        checkpoint={"quantization_config": {"quant_method": "mxfp4"}},
    )
    unresolved, _ = apply({}, **options)
    assert "kv_cache_dtype" not in unresolved
    resolved, _ = apply({}, checkpoint_weight_dtype="bfloat16", **options)
    assert resolved["kv_cache_dtype"] == "bf16"
    assert resolved["block_size"] == 16


def test_vllm_minimax_fp8_weights_do_not_imply_fp8_kv():
    resolved, _ = apply(
        {},
        backend="vllm",
        version="0.21.0",
        architecture="MiniMaxM2ForCausalLM",
        checkpoint={"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}},
        checkpoint_weight_dtype="float32",
    )
    assert resolved["kv_cache_dtype"] == "bf16"
    assert resolved["block_size"] == 16


def test_vllm_0102_v1_forces_chunking_and_uses_historical_token_default():
    resolved, evidence = apply(
        dict(enable_chunked_prefill=False, max_model_len=512, max_num_seqs=1),
        backend="vllm",
        version="0.10.2",
        architecture="GptOssForCausalLM",
        checkpoint_weight_dtype="bfloat16",
    )
    assert resolved["enable_chunked_prefill"] is True
    assert resolved["block_size"] == 16
    assert resolved["kv_cache_dtype"] == "bf16"
    assert resolved["max_num_batched_tokens"] == 8192
    assert any(e["knob"] == "enable_chunked_prefill" and e["kind"] == "runtime_override" for e in evidence)


@pytest.mark.parametrize(
    "args",
    [
        {"recipe_environment": {"VLLM_USE_V1": "0"}},
        {"load_format": "sharded_state"},
        {"max_num_partial_prefills": 2},
    ],
)
def test_vllm_0102_does_not_apply_v1_rules_to_fallback_paths(args):
    resolved, evidence = apply(args, backend="vllm", version="0.10.2", architecture="GptOssForCausalLM")
    assert resolved == args
    assert evidence == []


def test_exact_dynamo_wheel_establishes_openai_worker_context():
    context, sources = defaults.verified_dynamo_usage_context({"install": True, "wheel": "1.2.0.dev20260526"})
    assert context == "openai_api_server"
    assert sources[0]["sha256"] == "7be70a20409532a1b3a2954c8df74d73a592e0ce8dff56c1e863d307647d7ec1"
    resolved, _ = apply(
        {"max_model_len": 2048, "max_num_batched_tokens": 2048},
        backend="vllm",
        version="0.20.1",
        architecture="MiniMaxM2ForCausalLM",
        aggregated=False,
        usage_context=context,
    )
    assert resolved["max_num_seqs"] == 1024
    assert defaults.verified_dynamo_usage_context({"install": True, "wheel": "1.2.0.dev20260527"}) == (None, [])
    assert defaults.verified_dynamo_usage_context({"install": False, "wheel": "1.2.0.dev20260526"}) == (None, [])


@pytest.mark.parametrize("version", ["1.3.0.dev20260710", "1.3.0.dev20260713"])
@pytest.mark.parametrize("key", ["wheel", "version"])
def test_reviewed_dynamo_installation_matches_runtime(version, key):
    context, sources = defaults.verified_dynamo_usage_context({"install": True, key: version}, runtime_version=version)
    assert context == "openai_api_server"
    assert sources == defaults._SOURCES["dynamo"][version]
    assert "dynamo/vllm/args.py" in sources[0]["reviewed_paths"]


@pytest.mark.parametrize(
    "installation,runtime_version",
    [
        ({"install": True, "version": "1.3.0.dev20260710"}, "1.3.0.dev20260713"),
        ({"install": True, "version": "1.3.0.dev20260710", "wheel": "1.3.0.dev20260713"}, None),
        ({"install": True, "version": "1.3.0.dev20260710", "hash": "a" * 40}, None),
        ({"install": True, "version": "1.3.0.dev20260710", "top_of_tree": True}, None),
        ({"install": True, "version": "1.3.0.dev20260711"}, None),
        ({"install": True, "version": ["1.3.0.dev20260710"]}, None),
        ({"install": True}, None),
    ],
)
def test_dynamo_context_requires_unambiguous_exact_installation(installation, runtime_version):
    assert defaults.verified_dynamo_usage_context(installation, runtime_version=runtime_version) == (None, [])


def test_matching_dynamo_version_and_wheel_are_unambiguous():
    context, _ = defaults.verified_dynamo_usage_context(
        {"install": True, "wheel": "1.3.0.dev20260710", "version": "1.3.0.dev20260710"}
    )
    assert context == "openai_api_server"


def kimi_defaults(args=None, **kwargs):
    return apply(
        {
            "kv_cache_dtype": "fp8",
            "max_model_len": 9472,
            "max_num_seqs": 4,
            "attention_config": {"mla_prefill_backend": "FLASHINFER"},
            **(args or {}),
        },
        backend="vllm",
        version=kwargs.pop("version", defaults._KIMI_VLLM_REVISION),
        architecture="KimiK25ForConditionalGeneration",
        checkpoint={
            "dtype": "bfloat16",
            "text_config": {
                "dtype": "bfloat16",
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "qk_nope_head_dim": 128,
            },
        },
        **kwargs,
    )


def test_kimi_nightly_resolves_mla_page_and_serve_scheduler_defaults():
    resolved, evidence = kimi_defaults({"data_parallel_size": 4, "enable_expert_parallel": True})
    assert resolved["block_size"] == 32
    assert resolved["max_num_batched_tokens"] == 8192
    assert resolved["max_num_seqs"] == 4
    assert resolved["enable_chunked_prefill"] is True
    assert all(item["version"] == defaults._KIMI_VLLM_REVISION for item in evidence)


def test_kimi_nightly_preserves_explicit_limits_and_page_size():
    resolved, _ = kimi_defaults({"block_size": 64, "max_num_batched_tokens": 4096})
    assert resolved["block_size"] == 64
    assert resolved["max_num_batched_tokens"] == 4096


def test_kimi_scheduler_caps_tokens_by_model_length_and_respects_chunking():
    resolved, _ = kimi_defaults({"max_num_seqs": 2, "max_model_len": 1024})
    assert resolved["max_num_batched_tokens"] == 2048
    resolved, _ = kimi_defaults({"enable_chunked_prefill": False})
    assert resolved["max_num_batched_tokens"] == 9472


@pytest.mark.parametrize(
    "overrides",
    [
        {"attention_backend": "CUTLASS_MLA"},
        {"attention_config": {"backend": "CUTLASS_MLA", "mla_prefill_backend": "FLASHINFER"}},
        {"attention_config": {"mla_prefill_backend": "FLASH_ATTN"}},
        {
            "attention_config": {
                "backend_per_kind": {"mla_attention": "CUTLASS_MLA"},
                "mla_prefill_backend": "FLASHINFER",
            }
        },
        {"kv_cache_dtype": "fp8_e5m2"},
        {"kv_cache_dtype_skip_layers": [1]},
    ],
)
def test_kimi_page_size_requires_reviewed_attention_path(overrides):
    resolved, _ = kimi_defaults(overrides)
    assert "block_size" not in resolved


def test_kimi_nightly_does_not_generalize_to_other_release_or_hopper():
    resolved, _ = kimi_defaults(version="0.20.1")
    assert "block_size" not in resolved and "max_num_batched_tokens" not in resolved
    resolved, _ = kimi_defaults(hardware="h200")
    assert "block_size" not in resolved


def test_kimi_batched_dp_scheduler_needs_its_own_trace():
    resolved, _ = kimi_defaults({"data_parallel_size": 4, "enable_expert_parallel": True, "all2all_backend": "nixl_ep"})
    assert "max_num_batched_tokens" not in resolved


def test_dynamo_minimax_derives_checkpoint_model_length_before_scheduler():
    resolved, evidence = apply(
        {"kv_cache_dtype": "fp8"},
        backend="vllm",
        version="0.20.1",
        architecture="MiniMaxM2ForCausalLM",
        aggregated=False,
        usage_context="openai_api_server",
        checkpoint={"max_position_embeddings": 196608},
    )
    assert resolved["max_model_len"] == 196608
    assert resolved["max_num_batched_tokens"] == 8192
    assert resolved["max_num_seqs"] == 1024
    assert next(x for x in evidence if x["knob"] == "max_model_len")["kind"] == "verified_default"


@pytest.mark.parametrize(
    "checkpoint, expected",
    [
        ({"max_position_embeddings": 196608, "seq_length": 4096}, 4096),
        ({"max_position_embeddings": 196608, "model_max_length": 262144}, 262144),
    ],
)
def test_minimax_model_length_uses_pinned_converter_precedence(checkpoint, expected):
    resolved, _ = apply(
        {}, backend="vllm", version="0.20.1", architecture="MiniMaxM2ForCausalLM", checkpoint=checkpoint
    )
    assert resolved["max_model_len"] == expected


@pytest.mark.parametrize(
    "args,checkpoint",
    [
        ({}, {}),
        ({}, {"max_position_embeddings": "196608"}),
        ({}, {"max_position_embeddings": 196608, "rope_scaling": {"factor": 2}}),
        ({"disable_sliding_window": True}, {"max_position_embeddings": 196608}),
    ],
)
def test_minimax_model_length_keeps_unknown_conditions_unresolved(args, checkpoint):
    resolved, _ = apply(
        args, backend="vllm", version="0.20.1", architecture="MiniMaxM2ForCausalLM", checkpoint=checkpoint
    )
    assert "max_model_len" not in resolved
    assert "max_num_batched_tokens" not in resolved


@pytest.mark.parametrize("max_len", [8192, -1])
def test_minimax_preserves_explicit_model_length_and_autofit(max_len):
    resolved, _ = apply(
        {"max_model_len": max_len},
        backend="vllm",
        version="0.20.1",
        architecture="MiniMaxM2ForCausalLM",
        checkpoint={"max_position_embeddings": 196608},
    )
    assert resolved["max_model_len"] == max_len
    if max_len == -1:
        assert "max_num_batched_tokens" not in resolved


def test_recorded_m3_runtime_resolves_requests_and_auto_kv_only():
    version = defaults._M3_VLLM_RUNTIME
    args = {
        "runtime_framework_version": version,
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "max_num_batched_tokens": 2048,
        "block_size": 128,
    }
    resolved, _ = apply(args, backend="vllm", version=version, architecture="MiniMaxM3SparseForConditionalGeneration")
    assert resolved["max_num_seqs"] == 1024
    assert resolved["kv_cache_dtype"] == "bf16"
    assert resolved["block_size"] == 128
    assert "enable_chunked_prefill" not in resolved
    resolved, _ = apply(
        args | {"runtime_framework_version": None},
        backend="vllm",
        version=version,
        architecture="MiniMaxM3SparseForConditionalGeneration",
    )
    assert "max_num_seqs" not in resolved
    assert resolved["kv_cache_dtype"] == "auto"
