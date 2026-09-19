# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config onboarding uses declared geometry, without analytical model construction."""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import pytest

from aisimulate.support.config_profile import derive_profile, load_model_config
from aisimulate.support.schema import SupportRequest

pytestmark = pytest.mark.unit


def _config(tmp_path, **updates):
    # Original small synthetic geometry makes every tensor count hand-checkable.
    raw = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 64,
        "max_position_embeddings": 2048,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
    }
    raw.update(updates)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_model_config(path)


def _request(kind="dense", **topology):
    return SupportRequest.model_validate(
        {
            "identity": {
                "model": "example/configured-model",
                "model_revision": "checkpoint-123",
                "model_kind": kind,
                "framework_version": "0.25.1",
                "gpu": "h200_sxm",
                "interconnect": "NVLink",
            },
            "search": {"context_length": 2048, **topology},
            "workload": {"concurrency": 2},
        }
    )


def _runtime(**updates):
    return {"kv_cache_dtype": "bfloat16", "fmha_quant_mode": "bfloat16", "comm_quant_mode": "half", **updates}


def test_config_hash_and_identity_are_content_facts_not_checkpoint_revision(tmp_path):
    config = _config(tmp_path, _name_or_path="source/repository", auto_map={"AutoConfig": "do_not_execute.Config"})
    assert config.sha256 == hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    assert config.suggestions == {
        "model": "source/repository",
        "architecture": "LlamaForCausalLM",
        "model_kind": "dense",
        "context_length": 2048,
        "num_experts": 0,
    }
    draft = derive_profile(config, _request(), _runtime())
    assert draft.profile is not None
    assert draft.profile.model == "example/configured-model"
    assert draft.profile.model_revision == "checkpoint-123"
    assert config.sha256 in draft.profile.provenance
    assert config.sha256 in draft.profile.deployments[0].resources.provenance
    assert "auto_map" in config.notes


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("declared_architecture", [False, True])
def test_nested_text_geometry_and_architecture_are_independent_of_wrapper_and_encoders(
    tmp_path, kind, declared_architecture
):
    text = _config(tmp_path).raw
    if kind == "moe":
        text.update(architectures=["MixtralForCausalLM"], model_type="mixtral", num_local_experts=4)
    architecture = text["architectures"][0]
    baseline = derive_profile(_config(tmp_path, **text), _request(kind), _runtime())
    if not declared_architecture:
        del text["architectures"]
    config = _config(
        tmp_path,
        architectures=["ExampleMultimodalForConditionalGeneration"],
        model_type="example_multimodal",
        _name_or_path="example/original-checkpoint",
        hidden_size=1024,
        n_layer=99,
        max_position_embeddings=8192,
        num_experts=64,
        text_config={**text, "_name_or_path": "example/decoder-base"},
        vision_config={"hidden_size": 512, "num_hidden_layers": 20, "num_experts": 8},
        audio_config={"hidden_size": 256, "num_hidden_layers": 10},
    )

    draft = derive_profile(config, _request(kind), _runtime())

    published_architecture = architecture if declared_architecture else "ExampleMultimodalForConditionalGeneration"
    assert draft.resolved == {**baseline.resolved, "architecture": published_architecture}
    assert draft.profile.architecture == published_architecture
    assert config.decoder_architecture == architecture
    assert config.suggestions["model"] == "example/original-checkpoint"
    assert config.sha256 == hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    notes = json.loads(draft.profile.provenance)["config_notes"]
    assert "text_config" in notes["decoder_config"]
    assert f"decoder architecture={architecture}" in notes["decoder_architecture"]
    assert "Text decoder only" in notes["modeling_scope"]
    assert "Full multimodal deployment memory and latency are not modeled" in notes["modeling_scope"]
    assert json.loads(draft.profile.deployments[0].resources.provenance)["config_notes"] == notes


def test_nested_config_inherits_only_shared_precision_metadata_and_original_identity(tmp_path):
    text = _config(tmp_path).raw
    del text["torch_dtype"]
    config = _config(
        tmp_path,
        _name_or_path="example/quantized-checkpoint",
        auto_map={"AutoConfig": "do_not_execute.Config"},
        text_config={**text, "_name_or_path": "example/decoder-base"},
        quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4", "kv_cache_scheme": "FP8"},
    )
    draft = derive_profile(config, _request(), _runtime(weights_bytes=1024))
    assert draft.profile is not None
    assert draft.resolved["gemm_quant_mode"] == draft.resolved["moe_quant_mode"] == "nvfp4"
    assert config.raw["torch_dtype"] == "bfloat16"
    assert config.raw["quantization_config"]["quant_algo"] == "NVFP4"
    assert config.suggestions["model"] == "example/quantized-checkpoint"
    assert "torch_dtype, quantization_config inherited" in config.notes["shared_metadata"]
    assert "auto_map" in config.notes
    assert config.sha256 in draft.profile.provenance


def test_nested_precision_metadata_takes_precedence_as_a_group(tmp_path):
    text = _config(tmp_path).raw
    del text["torch_dtype"]
    config = _config(
        tmp_path,
        text_config={**text, "dtype": "float16", "quantization_config": {"quant_method": "fp8"}},
        quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4"},
    )
    draft = derive_profile(config, _request())
    assert draft.resolved["gemm_quant_mode"] == "fp8_static"
    assert draft.resolved["moe_quant_mode"] == "fp8"
    assert "torch_dtype" not in config.raw
    assert "config tensor dtype=float16" in config.notes["dtype"]
    assert "shared_metadata" not in config.notes


@pytest.mark.parametrize("metadata", [{"hf_quant_config": {"format": "custom"}}, {"quant_algo": "CUSTOM"}])
def test_shared_unrecognized_quantization_prevents_bfloat16_storage_assumptions(tmp_path, metadata):
    config = _config(tmp_path, text_config=_config(tmp_path).raw, **metadata)
    draft = derive_profile(config, _request())
    assert {"gemm_quant_mode", "moe_quant_mode", "weights_bytes"} <= draft.missing.keys()
    for key, value in metadata.items():
        assert config.raw[key] == value
        assert key in config.notes["shared_metadata"]


@pytest.mark.parametrize("metadata", [{"hf_quant_config": {"format": "custom"}}, {"quant_algo": "CUSTOM"}])
def test_nested_unrecognized_quantization_is_not_replaced_by_shared_metadata(tmp_path, metadata):
    config = _config(
        tmp_path,
        text_config={**_config(tmp_path).raw, **metadata},
        quantization_config={"quant_method": "fp8"},
    )
    draft = derive_profile(config, _request())
    assert {"gemm_quant_mode", "moe_quant_mode", "weights_bytes"} <= draft.missing.keys()
    assert "quantization_config" not in config.raw


@pytest.mark.parametrize("model_type", [None, "example_custom_text"])
def test_unknown_nested_decoder_retains_wrapper_identity_and_requires_resource_bounds(tmp_path, model_type):
    text = _config(tmp_path).raw
    for key in ("architectures", "model_type", "max_position_embeddings"):
        del text[key]
    text.update(model_max_length=4096, n_routed_experts=4, local_layer_ids=[0], sliding_window_size=128)
    if model_type:
        text["model_type"] = model_type
    config = _config(
        tmp_path,
        architectures=["ExampleMultimodalForConditionalGeneration"],
        model_type="example_multimodal",
        text_config=text,
    )
    draft = derive_profile(config, _request("moe"), _runtime())
    assert draft.resolved["architecture"] == "ExampleMultimodalForConditionalGeneration"
    assert draft.resolved["context_length"] == 4096
    assert draft.resolved["num_experts"] == 4
    assert {"weights_bytes", "activations_bytes", "cache_layout", "kv_bytes_per_token"} <= draft.missing.keys()
    assert "wrapper architecture" in draft.sources["architecture"]
    assert "assumption" in draft.sources["gemm_quant_mode"]
    complete = derive_profile(
        config,
        _request("moe"),
        _runtime(weights_bytes=1024, activations_bytes=2048, kv_bytes_per_token=64, cache_layout="linear"),
    )
    assert complete.profile is not None
    assert "user override" in complete.sources["kv_bytes_per_token"]


def test_missing_text_geometry_does_not_fall_back_to_wrapper_dimensions(tmp_path):
    config = _config(tmp_path, text_config={"model_type": "llama", "model_max_length": 2048})
    draft = derive_profile(config, _request(), _runtime())
    assert {"weights_bytes", "activations_bytes", "kv_bytes_per_token"} <= draft.missing.keys()
    assert "hidden_size" not in config.raw
    assert "hidden_size" in draft.missing["weights_bytes"]


def test_known_wrapper_does_not_establish_unknown_nested_decoder_resource_layout(tmp_path):
    text = {key: value for key, value in _config(tmp_path).raw.items() if key not in ("architectures", "model_type")}
    config = _config(tmp_path, text_config=text)
    draft = derive_profile(config, _request(), _runtime())
    assert draft.resolved["architecture"] == "LlamaForCausalLM"
    assert config.decoder_architecture is None
    assert {"num_experts", "weights_bytes", "activations_bytes", "kv_bytes_per_token", "cache_layout"} <= (
        draft.missing.keys()
    )


@pytest.mark.parametrize("text", [None, {}, [], "decoder", [{"hidden_size": 16}], {"text_config": {"hidden_size": 16}}])
def test_malformed_or_ambiguous_text_sections_have_actionable_diagnostics(tmp_path, text):
    with pytest.raises(ValueError, match="text_config.*decoder.*--fpm-profile"):
        _config(tmp_path, text_config=text)


def test_nested_architecture_ambiguity_is_not_hidden_by_wrapper_fallback(tmp_path):
    text = {**_config(tmp_path).raw, "architectures": ["LlamaForCausalLM", "MixtralForCausalLM"]}
    with pytest.raises(ValueError, match="exactly one decoder architecture"):
        _config(tmp_path, text_config=text)


def test_flat_multimodal_metadata_does_not_change_decoder_estimates(tmp_path):
    baseline = derive_profile(_config(tmp_path), _request(), _runtime())
    config = _config(
        tmp_path,
        vision_config={"hidden_size": 512, "num_hidden_layers": 20},
        audio_config={"hidden_size": 256, "num_hidden_layers": 10},
    )
    draft = derive_profile(config, _request(), _runtime())
    assert draft.resolved == baseline.resolved
    assert "Text decoder only" in json.loads(draft.profile.provenance)["config_notes"]["modeling_scope"]


@pytest.mark.parametrize("tp,weights,kv", [(1, 13472, 64), (2, 6816, 32), (4, 3744, 32)])
def test_dense_tensor_counts_include_replicated_norms_and_kv_heads(tmp_path, tp, weights, kv):
    # TP1: 2 embedding/head matrices * 64*16 + 2 layers *
    # (Q/O 2*16*16 + K/V 2*16*8 + MLP 3*16*32 + norms 2*16) + final norm 16.
    draft = derive_profile(_config(tmp_path), _request(tensor_parallel=tp), _runtime())
    assert draft.missing == {}
    resources = draft.profile.deployments[0].resources
    assert resources.weights_bytes == weights
    assert resources.kv_bytes_per_token == kv
    assert resources.max_num_tokens == 8192
    assert resources.max_batch_size == 2
    assert "estimate" in draft.sources["weights_bytes"]
    assert "exact" in draft.sources["kv_bytes_per_token"]
    assert "estimate" in draft.sources["runtime_overhead_bytes"]
    assert "sha256" in draft.sources["runtime_overhead_bytes"]


def test_weight_tying_biases_and_qk_norm_have_independent_counts(tmp_path):
    request = _request(tensor_parallel=2)
    base = derive_profile(_config(tmp_path), request, _runtime()).resolved["weights_bytes"]
    tied = derive_profile(_config(tmp_path, tie_word_embeddings=True), request, _runtime())
    assert tied.resolved["weights_bytes"] == base - 64 // 2 * 16 * 2
    biased = derive_profile(_config(tmp_path, attention_bias=True, mlp_bias=True), request, _runtime())
    # Per layer: QKV (8+4+4), output (16), gated MLP (16+16+16), all two-byte.
    assert biased.resolved["weights_bytes"] == base + 2 * (16 + 16 + 48) * 2
    qwen = derive_profile(
        _config(tmp_path, architectures=["Qwen3ForCausalLM"], model_type="qwen3"), request, _runtime()
    )
    assert qwen.resolved["weights_bytes"] == base + 2 * (2 * 4) * 2


@pytest.mark.parametrize("tp,expected_bias_bytes", [(1, 128), (2, 64), (4, 48)])
def test_qwen3_attention_bias_counts_qkv_without_output_bias(tmp_path, tp, expected_bias_bytes):
    request = _request(tensor_parallel=tp)
    facts = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}
    base = derive_profile(_config(tmp_path, **facts), request, _runtime())
    biased = derive_profile(_config(tmp_path, **facts, attention_bias=True), request, _runtime())
    # Two layers, two-byte Q/K/V biases: widths (16,8,8), (8,4,4), or (4,4,4).
    # The output projection is bias-free; KV heads replicate at TP4.
    assert biased.resolved["weights_bytes"] - base.resolved["weights_bytes"] == expected_bias_bytes


@pytest.mark.parametrize(
    "topology,weights,kv,parallel",
    [
        ({"tensor_parallel": 2}, 16288, 32, (2, 1, 1, 2, 1, 1)),
        (
            {"attention_data_parallel": 2, "moe_tensor_parallel": 1, "moe_expert_parallel": 2},
            19872,
            64,
            (1, 1, 2, 1, 2, 1),
        ),
        ({"tensor_parallel": 2, "moe_tensor_parallel": 1, "moe_expert_parallel": 2}, 16288, 32, (2, 1, 1, 1, 2, 1)),
    ],
)
def test_moe_resources_are_rank_local_for_tp_dep_and_tep(tmp_path, topology, weights, kv, parallel):
    config = _config(tmp_path, architectures=["MixtralForCausalLM"], model_type="mixtral", num_local_experts=4)
    draft = derive_profile(config, _request("moe", **topology), _runtime(comm_overhead_bytes=321))
    assert draft.missing == {}
    deployment = draft.profile.deployments[0]
    assert deployment.parallel_tuple == parallel
    assert deployment.resources.weights_bytes == weights
    assert deployment.resources.kv_bytes_per_token == kv
    assert deployment.resources.comm_overhead_bytes == 321


def test_activation_estimate_uses_declared_scheduler_envelope(tmp_path):
    config = _config(tmp_path, hidden_size=4096, intermediate_size=8192, num_attention_heads=32, head_dim=128)
    small = derive_profile(config, _request(tensor_parallel=2), _runtime(max_num_tokens=16))
    large = derive_profile(config, _request(tensor_parallel=2), _runtime(max_num_tokens=4096))
    assert small.resolved["activations_bytes"] == 70 * 1024 * 1024
    assert large.resolved["activations_bytes"] == 2 * 4096 * 4096 * 6.5
    assert "estimate" in large.sources["activations_bytes"]
    assert "max_num_tokens=4096" in large.sources["activations_bytes"]


def test_mha_and_weight_padding_have_separate_supported_domains(tmp_path):
    draft = derive_profile(_config(tmp_path, num_key_value_heads=4, vocab_size=65), _request(), _runtime())
    assert draft.resolved["kv_bytes_per_token"] == 128
    assert "weights_bytes" in draft.missing
    assert "padding" in draft.missing["weights_bytes"]


@pytest.mark.parametrize("field", ["gemm_quant_mode", "kv_cache_dtype", "weights_bytes", "activations_bytes"])
def test_explicit_overrides_survive_derivation_and_are_recorded_per_field(tmp_path, field):
    value = "fp8" if "mode" in field or "dtype" in field else 123456
    draft = derive_profile(_config(tmp_path), _request(), _runtime(**{field: value}))
    assert draft.resolved[field] == value
    assert "user override" in draft.sources[field]


def test_hardware_defaults_do_not_reuse_tp_communication_for_dep(tmp_path):
    config = _config(tmp_path, architectures=["MixtralForCausalLM"], model_type="mixtral", num_local_experts=4)
    request = _request("moe", attention_data_parallel=2, moe_tensor_parallel=1, moe_expert_parallel=2)
    draft = derive_profile(config, request, _runtime())
    assert "comm_overhead_bytes" in draft.missing
    assert draft.resolved["runtime_overhead_bytes"] == 3758096384


def test_unknown_hardware_has_no_fallback_reservations(tmp_path):
    request = _request()
    request.identity.gpu = "custom_gpu"
    draft = derive_profile(_config(tmp_path), request, _runtime())
    assert {"comm_overhead_bytes", "runtime_overhead_bytes"} <= set(draft.missing)


def test_dynamic_kv_scales_need_explicit_accounting(tmp_path):
    config = _config(
        tmp_path,
        quantization_config={
            "quant_method": "fp8",
            "kv_cache_scheme": {"num_bits": 8, "type": "float", "dynamic": True},
        },
    )
    draft = derive_profile(config, _request())
    assert draft.resolved["kv_cache_dtype"] == "fp8"
    assert "kv_bytes_per_token" in draft.missing
    assert "scales" in draft.missing["kv_bytes_per_token"]


@pytest.mark.parametrize("cache_type", [[], {}, True, 8, 1.5, "", "\x00"])
def test_malformed_kv_cache_type_is_rejected_at_config_load(tmp_path, cache_type):
    with pytest.raises(ValueError, match=r"quantization_config\.kv_cache_scheme\.type"):
        _config(tmp_path, quantization_config={"kv_cache_scheme": {"num_bits": 8, "type": cache_type}})


@pytest.mark.parametrize(
    "scheme,dtype",
    [
        ("FP8", "fp8"),
        ({"num_bits": 8, "type": "float"}, "fp8"),
        ({"num_bits": 8, "type": "int"}, "int8"),
        ({"num_bits": 8, "type": None}, None),
        ({"num_bits": 8}, None),
        ({"num_bits": 8, "type": "unknown"}, None),
    ],
)
def test_kv_cache_type_preserves_known_precision_and_unresolved_metadata(tmp_path, scheme, dtype):
    config = _config(tmp_path, quantization_config={"kv_cache_scheme": scheme})
    draft = derive_profile(config, _request())
    if dtype is None:
        assert "kv_cache_dtype" not in draft.resolved
        assert "kv_cache_dtype" in draft.missing
    else:
        assert draft.resolved["kv_cache_dtype"] == dtype
        assert draft.resolved["kv_bytes_per_token"] == 32


def test_weight_quantization_does_not_select_runtime_attention_or_kv_dtype(tmp_path):
    config = _config(tmp_path, quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]})
    draft = derive_profile(config, _request())
    assert draft.resolved["gemm_quant_mode"] == "fp8_block"
    assert draft.resolved["moe_quant_mode"] == "fp8_block"
    assert {"fmha_quant_mode", "kv_cache_dtype", "weights_bytes", "comm_quant_mode"} <= draft.missing.keys()
    assert draft.profile is None
    assert "quantized" in draft.missing["weights_bytes"]


@pytest.mark.parametrize(
    "quant,gemm,moe",
    [
        ({"quant_method": "fp8"}, "fp8_static", "fp8"),
        ({"quant_method": "fp8", "activation_scheme": "static"}, "fp8_static", "fp8"),
        ({"quant_method": "fp8", "activation_scheme": "dynamic"}, "fp8", "fp8"),
        (
            {
                "quant_method": "modelopt",
                "quant_algo": "FP8",
                "config_groups": {"g": {"input_activations": {"dynamic": True}}},
            },
            "fp8",
            "fp8",
        ),
        ({"quant_method": "fp8", "weight_block_size": [128, 128]}, "fp8_block", "fp8_block"),
        ({"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["self_attn*"]}, "nvfp4", "nvfp4"),
    ],
)
def test_config_quantization_uses_existing_fpm_checkpoint_identity(tmp_path, quant, gemm, moe):
    from aisimulate.support.fpm import fpm_cli_args
    from aisimulate_core.sdk.models.helpers import _infer_quant_modes_from_raw_config
    from aisimulate_core.sdk.utils import _infer_quantization_fields

    config = _config(tmp_path, quantization_config=quant)
    draft = derive_profile(config, _request(), _runtime(weights_bytes=1024))
    assert draft.resolved["gemm_quant_mode"] == gemm
    assert draft.resolved["moe_quant_mode"] == moe
    # Compare the real collector's identity helper, not a duplicate test formula.
    existing = _infer_quant_modes_from_raw_config({**config.raw, **_infer_quantization_fields(config.raw)})
    assert existing["gemm_quant_mode"].name == gemm
    assert existing["moe_quant_mode"].name == moe
    request = SupportRequest.model_validate({**_request().model_dump(), "fpm_profile": draft.profile})
    command = fpm_cli_args(request, output_dir=tmp_path / "plan", plan_only=True)
    assert command[command.index("--fpm-weight-quantizations") + 1] == gemm
    assert draft.profile.deployments[0].match_identity()[:2] == [gemm, moe]
    assert "checkpoint/FPM" in draft.sources["gemm_quant_mode"]


def test_unknown_quantization_never_falls_back_to_residual_bfloat16(tmp_path):
    config = _config(tmp_path, quantization_config={"quant_method": "custom", "quant_algo": "custom"})
    draft = derive_profile(config, _request())
    assert {"gemm_quant_mode", "moe_quant_mode", "weights_bytes"} <= set(draft.missing)


def test_missing_fields_are_complete_and_unknown_layout_can_use_explicit_resources(tmp_path):
    config = _config(tmp_path, architectures=["NewDecoderForCausalLM"], model_type="new_decoder")
    assert "num_experts" not in config.suggestions
    draft = derive_profile(config, _request())
    assert {"num_experts", "weights_bytes", "activations_bytes", "kv_bytes_per_token", "cache_layout"} <= set(
        draft.missing
    )
    complete = derive_profile(
        config,
        _request(),
        _runtime(
            num_experts=0, weights_bytes=1000, activations_bytes=2000, kv_bytes_per_token=32, cache_layout="linear"
        ),
    )
    assert complete.profile is not None
    assert "user override" in complete.sources["weights_bytes"]


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"hidden_size": True}, "hidden_size"),
        ({"num_hidden_layers": 2.0}, "num_hidden_layers"),
        ({"num_attention_heads": 0}, "num_attention_heads"),
        ({"architectures": ["LlamaForCausalLM", "OtherForCausalLM"]}, "architectures"),
        ({"n_layer": 4}, "conflicting"),
        ({"model_max_length": 4096}, "conflicting context_length"),
        ({"dtype": "float32"}, "conflicting"),
        ({"num_key_value_heads": 3}, "num_key_value_heads"),
        ({"attention_bias": "false"}, "attention_bias"),
        ({"model_type": "qwen3"}, "model_type"),
        ({"quantization_config": []}, "quantization_config"),
        ({"num_mtp_modules": -1}, "num_mtp_modules"),
        (
            {"quantization_config": {"config_groups": {"g": {"weights": {"dynamic": "false"}}}}},
            "dynamic",
        ),
    ],
)
def test_corrupt_or_ambiguous_config_fails_before_profile_overrides(tmp_path, updates, match):
    with pytest.raises(ValueError, match=match):
        _config(tmp_path, **updates)


@pytest.mark.parametrize("raw", ['{"hidden_size":16,"hidden_size":32}', '{"hidden_size":NaN}', "[]"])
def test_json_must_be_finite_object_with_unique_keys(tmp_path, raw):
    path = tmp_path / "config.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_model_config(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"weights_byte": 7},
        {"weights_bytes": True},
        {"weights_bytes": 7.0},
        {"weights_bytes": -1},
        {"weights_bytes": 2**53 + 1},
        {"max_num_tokens": 0},
        {"kv_bytes_per_token": 0},
        {"num_experts": -1},
        {"cache_layout": "hybrid"},
        {"gemm_quant_mode": "fp16"},
        {"kv_cache_dtype": "auto"},
        {"provenance": " "},
        {"context_length": "2048"},
        {"weights_bytes": 2**53, "activations_bytes": 1},
    ],
)
def test_overrides_are_strict_even_before_identity_is_complete(tmp_path, overrides):
    with pytest.raises(ValueError):
        derive_profile(_config(tmp_path), None, overrides)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "updates",
    [
        {"sliding_window": 128},
        {"layer_types": ["full_attention", "linear_attention"]},
        {"is_encoder_decoder": True},
        {"ssm_cfg": {"d_state": 16}},
        {"kv_lora_rank": 8},
    ],
)
def test_known_incompatible_state_cannot_be_papered_over_with_overrides(tmp_path, updates, nested):
    if nested:
        updates = {"text_config": {**_config(tmp_path).raw, **updates}}
    with pytest.raises(ValueError, match="unsupported"):
        config = _config(tmp_path, **updates)
        derive_profile(config, _request(), _runtime(weights_bytes=1, kv_bytes_per_token=1, cache_layout="linear"))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("explicit_resources", [False, True])
@pytest.mark.parametrize(
    "facts,architecture,experts,match",
    [
        ({"num_experts": 4}, "LlamaForCausalLM", 4, "dense architecture"),
        ({}, "MixtralForCausalLM", 0, "MoE architecture"),
        ({"kv_lora_rank": 8}, "LlamaForCausalLM", 0, "cache/projection modifiers"),
        ({}, "MambaForCausalLM", 0, "recurrent/hybrid"),
        ({"num_experts_per_tok": 8}, "MixtralForCausalLM", 4, "routed expert count"),
    ],
)
def test_late_architecture_and_expert_inputs_cannot_bypass_structural_guards(
    tmp_path, preview, explicit_resources, facts, architecture, experts, match, nested
):
    config = _config(tmp_path, architectures=None, model_type="unknown", **facts)
    if nested:
        config = _config(tmp_path, architectures=None, model_type="unknown_wrapper", text_config=config.raw)
    overrides = _runtime(architecture=architecture, num_experts=experts)
    if explicit_resources:
        overrides.update(weights_bytes=1, kv_bytes_per_token=1, cache_layout="linear")
    with pytest.raises(ValueError, match=match):
        derive_profile(config, None if preview else _request("moe" if experts else "dense"), overrides)
    # Supplying the same effective facts in the original config also fails.
    with pytest.raises(ValueError, match=match):
        _config(tmp_path, **{**facts, "architectures": [architecture], "model_type": "unknown", "num_experts": experts})


@pytest.mark.parametrize("field", ["num_experts_per_tok", "first_k_dense_replace"])
@pytest.mark.parametrize("late_experts", [False, True])
@pytest.mark.parametrize("preview", [False, True])
def test_optional_null_moe_metadata_matches_absent_metadata(tmp_path, field, late_experts, preview):
    facts = {"architectures": ["MixtralForCausalLM"], "model_type": "mixtral"}
    overrides = _runtime()
    if late_experts:
        overrides["num_experts"] = 4
    else:
        facts["num_local_experts"] = 4
    request = None if preview else _request("moe")
    absent = derive_profile(_config(tmp_path, **facts), request, overrides)
    config = _config(tmp_path, **facts, **{field: None})

    draft = derive_profile(config, request, overrides)

    assert config.raw[field] is None
    assert field not in config.notes
    assert draft.resolved == absent.resolved
    assert draft.missing == absent.missing
    assert draft.resolved["num_experts"] == 4
    assert (draft.profile is None) == preview


@pytest.mark.parametrize("field", ["num_experts_per_tok", "first_k_dense_replace"])
@pytest.mark.parametrize("value", [True, False, -1, 1.5, "1"])
def test_optional_moe_metadata_rejects_noninteger_or_negative_values(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        _config(tmp_path, **{field: value})


@pytest.mark.parametrize(
    "facts,match",
    [
        ({"num_experts_per_tok": 0}, "num_experts_per_tok"),
        ({"num_experts_per_tok": 5}, "routed expert count"),
        ({"first_k_dense_replace": 3}, "num_hidden_layers"),
    ],
)
def test_optional_moe_metadata_preserves_count_validation(tmp_path, facts, match):
    with pytest.raises(ValueError, match=match):
        _config(tmp_path, architectures=["MixtralForCausalLM"], model_type="mixtral", num_local_experts=4, **facts)


def test_bundled_modelopt_string_kv_scheme_preserves_usable_geometry():
    path = (
        Path(__file__).parents[2] / "src/aisimulate_core/model_configs/Qwen--Qwen3-32B-FP8-Static-PerTensor_config.json"
    )
    config = load_model_config(path)
    draft = derive_profile(config, _request(tensor_parallel=4))
    assert config.raw["quantization_config"]["kv_cache_scheme"] == "FP8"
    assert config.suggestions["architecture"] == "Qwen3ForCausalLM"
    assert draft.resolved["gemm_quant_mode"] == "fp8_static"
    assert draft.resolved["moe_quant_mode"] == draft.resolved["kv_cache_dtype"] == "fp8"
    assert draft.resolved["kv_bytes_per_token"] == 2 * 64 * 2 * 128
    assert {"weights_bytes", "activations_bytes", "fmha_quant_mode"} <= draft.missing.keys()


def test_bundled_custom_layout_keeps_null_geometry_unresolved():
    path = (
        Path(__file__).parents[2]
        / "src/aisimulate_core/model_configs/nvidia--Llama-3_3-Nemotron-Super-49B-v1_config.json"
    )
    config = load_model_config(path)
    draft = derive_profile(config, _request(), _runtime())
    assert config.suggestions["architecture"] == "DeciLMForCausalLM"
    assert config.suggestions["context_length"] == 131072
    assert config.raw["intermediate_size"] is None
    assert "intermediate_size" not in config.notes
    assert {"weights_bytes", "activations_bytes", "kv_bytes_per_token", "num_experts"} <= draft.missing.keys()


def test_selected_topology_and_context_must_match_structural_config(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="attention heads"):
        derive_profile(config, _request(tensor_parallel=8), _runtime(weights_bytes=7, kv_bytes_per_token=7))
    with pytest.raises(ValueError, match="num_experts"):
        derive_profile(config, _request(), _runtime(num_experts=2))
    with pytest.raises(ValueError, match="context_length"):
        derive_profile(config, _request(), _runtime(context_length=4096))


@pytest.mark.parametrize(
    "filename,architecture,context,quant,kv",
    [
        ("MiniMaxAI--MiniMax-M2.7_config.json", "MiniMaxM2ForCausalLM", 204800, "fp8_block", None),
        ("nvidia--MiniMax-M2.7-NVFP4_config.json", "MiniMaxM2ForCausalLM", 196608, "nvfp4", "fp8"),
        ("zai-org--GLM-5.2_config.json", "GlmMoeDsaForCausalLM", 1048576, "bfloat16", None),
        ("nvidia--GLM-5.2-NVFP4_config.json", "GlmMoeDsaForCausalLM", 1048576, "nvfp4", "fp8"),
    ],
)
def test_real_minimax_and_glm_extract_independent_facts_without_naive_storage_rules(
    filename, architecture, context, quant, kv
):
    path = Path(__file__).parents[2] / "src/aisimulate_core/model_configs" / filename
    config = load_model_config(path)
    draft = derive_profile(config, _request("moe", tensor_parallel=4))
    assert config.suggestions["architecture"] == architecture
    assert config.suggestions["context_length"] == context
    assert config.suggestions["num_experts"] == 256
    assert config.suggestions["model_kind"] == "moe"
    assert draft.resolved["gemm_quant_mode"] == quant
    assert draft.resolved.get("kv_cache_dtype") == kv
    assert "fmha_quant_mode" in draft.missing
    assert "weights_bytes" in draft.missing
    if architecture == "GlmMoeDsaForCausalLM":
        assert "kv_bytes_per_token" in draft.missing
        assert "DSA" in draft.missing["kv_bytes_per_token"]
    else:
        finished_kv = derive_profile(config, _request("moe", tensor_parallel=4), {"kv_cache_dtype": "bfloat16"})
        assert finished_kv.resolved["kv_bytes_per_token"] == 62 * 2 * 2 * 128 * 2


@pytest.mark.parametrize("nested", [False, True])
def test_config_route_does_not_import_model_construction_or_remote_code(tmp_path, monkeypatch, nested):
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        assert not name.startswith(("transformers", "huggingface_hub", "collector")), name
        assert ".sdk.models" not in name, name
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)
    config = _config(tmp_path)
    if nested:
        config = _config(tmp_path, text_config=config.raw, vision_config={"hidden_size": 512})
    draft = derive_profile(config, _request(), _runtime())
    assert draft.profile is not None
