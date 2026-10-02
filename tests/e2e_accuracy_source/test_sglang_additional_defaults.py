# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest
from e2e_accuracy_source import sglang_additional_defaults as defaults
from e2e_accuracy_source.checkpoint_quantization import QuantizationMappingError, resolve_checkpoint_quantization
from e2e_accuracy_source.inferencex_recipe import InferenceXRecipeError, _normalize_yaml_server_args


@pytest.fixture
def verified(monkeypatch):
    monkeypatch.setattr(defaults, "_verified_sources", lambda version: [{"url": "reviewed"}])


def resolve(version="0.5.16", *, qwen=False, **args):
    return defaults.apply_additional_sglang_defaults(
        dict(
            attention_backend="trtllm_mha" if qwen else "trtllm_mla",
            chunked_prefill_size=16384,
            disable_radix_cache=True,
        )
        | args,
        version,
        hardware="b200",
        checkpoint={"architectures": ["Qwen3_5MoeForConditionalGeneration" if qwen else "DeepseekV3ForCausalLM"]},
    )


@pytest.mark.parametrize("version", ["0.5.3rc1", "0.5.16", "0.5.19"])
def test_explicit_mla_chunk_and_page_defaults(version, verified):
    args, evidence = resolve(version, max_num_seqs=256)
    assert args["block_size"] == 64
    assert args["max_num_batched_tokens"] == 16384
    assert args["enable_chunked_prefill"] is True
    assert args["max_num_seqs"] == 256
    assert all(item["source_revision"] == defaults._SOURCES[version]["commit"] for item in evidence)


@pytest.mark.parametrize("version,page", [("0.5.16", 64), ("0.5.19", 128)])
def test_release_specific_mha_page_support(version, page, verified):
    args, _ = resolve(version, qwen=True, page_size=128, block_size=128)
    assert args["page_size"] == args["block_size"] == page


def test_supported_explicit_mla_page_preserved(verified):
    args, _ = resolve(page_size=32)
    assert args["block_size"] == 32


def test_request_limit_remains_runtime_dependent(verified):
    args, _ = resolve("0.5.19", qwen=True, cuda_graph_max_bs=4)
    assert "max_running_requests" not in args
    assert "max_num_seqs" not in args
    assert "kv_cache_dtype" not in args


def test_dp_chunk_is_effective_per_rank_and_idempotent(verified):
    args, _ = resolve(data_parallel_size=4, enable_dp_attention=True)
    assert args["chunked_prefill_size"] == 16384
    assert args["max_num_batched_tokens"] == 4096
    repeated, _ = defaults.apply_additional_sglang_defaults(
        args, "0.5.16", hardware="b200", checkpoint={"architectures": ["DeepseekV3ForCausalLM"]}
    )
    assert repeated == args


@pytest.mark.parametrize(
    "extra",
    [
        {"chunked_prefill_size": 0},
        {"chunked_prefill_size": -1},
    ],
)
def test_nonpositive_chunk_disables_chunking(extra, verified):
    args, _ = resolve(**extra)
    assert args["enable_chunked_prefill"] is False
    assert "max_num_batched_tokens" not in args


@pytest.mark.parametrize("chunk", [0, -1])
def test_normalized_disabled_chunk_does_not_become_a_token_cap(chunk, verified):
    parsed = _normalize_yaml_server_args(dict(attention_backend="trtllm_mla", chunked_prefill_size=chunk))
    args, evidence = defaults.apply_additional_sglang_defaults(
        parsed, "0.5.16", hardware="b200", checkpoint={"architectures": ["DeepseekV3ForCausalLM"]}
    )
    assert args["enable_chunked_prefill"] is False
    assert "max_num_batched_tokens" not in args
    assert any(item["knob"] == "max_num_batched_tokens" and item["kind"] == "unresolved" for item in evidence)


def test_omitted_chunk_is_not_guessed(verified):
    args, _ = resolve(chunked_prefill_size=None)
    assert "enable_chunked_prefill" not in args


@pytest.mark.parametrize(
    "extra",
    [
        {"speculative_algorithm": "EAGLE"},
        {"enable_prefill_cp": True},
        {"model_impl": "transformers"},
        {"decode_attention_backend": "flashmla"},
        {"enable_multi_item_scoring": True},
        {"enable_dynamic_chunking": True},
    ],
)
def test_unreviewed_modes_fail_closed(extra, verified):
    args, evidence = resolve(**extra)
    assert "block_size" not in args
    assert evidence == []


def test_qwen_mamba_radix_mode_remains_unresolved(verified):
    args, evidence = resolve("0.5.19", qwen=True, disable_radix_cache=False)
    assert "block_size" not in args
    assert evidence == []


@pytest.mark.parametrize("version", [None, "0.5.20", "303757cc", "nightly-dev-cu13-20260608-303757cc"])
def test_unknown_or_nightly_source_is_not_a_release(version, verified):
    args, evidence = resolve(version)
    assert "block_size" not in args
    assert evidence == []


def test_reviewed_source_hash_mismatch_rejected(monkeypatch):
    class Response:
        content = b"wrong source"

        def raise_for_status(self):
            pass

    defaults._verified_sources.cache_clear()
    monkeypatch.setattr(defaults.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(InferenceXRecipeError, match="source changed"):
        defaults._verified_sources("0.5.16")
    defaults._verified_sources.cache_clear()


def test_manifest_has_immutable_source_commits():
    for entry in defaults._SOURCES.values():
        assert len(entry["commit"]) == 40
        for source in entry["sources"]:
            assert f"/{entry['commit']}/" in source["url"]
            assert len(source["sha256"]) == hashlib.sha256().digest_size * 2


@pytest.mark.parametrize(
    "version",
    [
        "0.0.0.dev1+g2b3d9ad37",
        "0.0.0.dev1+g303757ccd",
        "0.0.0.dev1+g3cbb7568b",
        "0.0.0.dev1+gd6ef68881",
        "0.0.0.dev1+gf825d7293",
    ],
)
@pytest.mark.parametrize("architecture", ["Qwen3_5MoeForConditionalGeneration", "DeepseekV4ForCausalLM"])
def test_runtime_identity_uses_logged_chunk_without_dividing_again(version, architecture, verified):
    original = dict(
        effective_chunked_prefill_size=16384, chunked_prefill_size=65536, enable_dp_attention=True, data_parallel_size=4
    )
    args, evidence = defaults.apply_additional_sglang_defaults(
        original, version, hardware="gb300", checkpoint={"architectures": [architecture]}
    )
    assert args == original | {"enable_chunked_prefill": True}
    assert evidence[0]["source_revision"] == defaults._SOURCES[version]["commit"]
    assert "max_num_seqs" not in args
    assert "kv_cache_dtype" not in args
    assert "block_size" not in args


@pytest.mark.parametrize(
    "args",
    [
        {"chunked_prefill_size": 16384},
        {"effective_chunked_prefill_size": True},
        {"effective_chunked_prefill_size": 16384, "model_impl": "transformers"},
        {"effective_chunked_prefill_size": 16384, "enable_dynamic_chunking": True},
        {"effective_chunked_prefill_size": 16384, "speculative_algorithm": "EAGLE"},
    ],
)
def test_runtime_chunk_requires_reviewed_consumer_and_effective_value(args, verified):
    actual, evidence = defaults.apply_additional_sglang_defaults(
        args,
        "0.0.0.dev1+g2b3d9ad37",
        hardware="gb300",
        checkpoint={"architectures": ["Qwen3_5MoeForConditionalGeneration"]},
    )
    assert actual == args
    assert evidence == []


def test_runtime_nonpositive_chunk_forces_disabled(verified):
    args, _ = defaults.apply_additional_sglang_defaults(
        {"effective_chunked_prefill_size": -1, "enable_chunked_prefill": True},
        "0.0.0.dev1+g2b3d9ad37",
        hardware="gb300",
        checkpoint={"architectures": ["Qwen3_5MoeForConditionalGeneration"]},
    )
    assert args["enable_chunked_prefill"] is False


def fp4_checkpoint():
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "expert_dtype": "fp4",
        "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]},
    }


def fp4_runtime():
    return {
        "runtime_is_fp4_experts": True,
        "moe_runner_backend": "flashinfer_mxfp4",
        "flashinfer_mxfp4_moe_precision": "default",
    }


@pytest.mark.parametrize("version", ["0.0.0.dev1+g3cbb7568b", "0.0.0.dev1+gd6ef68881"])
def test_reviewed_dsv4_runtime_maps_experts_separately(version, verified):
    result = resolve_checkpoint_quantization(
        fp4_checkpoint(),
        backend="sglang",
        hardware="gb300",
        framework_version=version,
        runtime_args=fp4_runtime(),
    )
    assert result["gemm"] == "fp8_block"
    assert result["moe"] == "w4a8_mxfp4_mxfp8"
    assert result["evidence"]["expert_kernel_mapping"]["source_revision"] == defaults._SOURCES[version]["commit"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"runtime_is_fp4_experts": None},
        {"runtime_is_fp4_experts": False},
        {"moe_runner_backend": "auto"},
        {"moe_runner_backend": "marlin"},
        {"flashinfer_mxfp4_moe_precision": None},
        {"flashinfer_mxfp4_moe_precision": "bf16"},
        {"recipe_environment": {"SGLANG_DSV4_FP4_DEQUANT": "1"}},
    ],
)
def test_fp4_weights_do_not_establish_expert_kernel(overrides, verified):
    with pytest.raises(QuantizationMappingError, match="reviewed backend/hardware"):
        resolve_checkpoint_quantization(
            fp4_checkpoint(),
            backend="sglang",
            hardware="gb300",
            framework_version="0.0.0.dev1+g3cbb7568b",
            runtime_args=fp4_runtime() | overrides,
        )


@pytest.mark.parametrize(
    "version,hardware",
    [
        ("0.0.0.dev1+g303757ccd", "gb300"),
        ("0.0.0.dev1+g3cbb7568b", "h200"),
        (None, "gb300"),
        ("0.5.16", "gb300"),
    ],
)
def test_fp4_profile_is_scoped_to_reviewed_runtime_identity(version, hardware, verified):
    with pytest.raises(QuantizationMappingError, match="reviewed backend/hardware"):
        resolve_checkpoint_quantization(
            fp4_checkpoint(),
            backend="sglang",
            hardware=hardware,
            framework_version=version,
            runtime_args=fp4_runtime(),
        )
