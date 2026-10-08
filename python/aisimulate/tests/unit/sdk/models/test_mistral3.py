# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Pixtral geometry cases derive from vLLM (Apache-2.0), modified for testing:
# Copyright contributors to the vLLM project.
# https://github.com/vllm-project/vllm/blob/ee0da84ab9e04ac7610e28580af62c365e898389/vllm/model_executor/models/pixtral.py
# https://github.com/vllm-project/vllm/blob/ee0da84ab9e04ac7610e28580af62c365e898389/vllm/model_executor/models/mistral3.py

"""Mistral-Medium-3.5 multimodal support.

The ``Mistral3ForConditionalGeneration`` architecture is a dense GQA text
decoder mapped to the LLAMA op graph plus a Pixtral (gated/SwiGLU) vision
encoder. Covers architecture routing, the Pixtral vision-config parsing
(including the top-level spatial_merge_size that the text_config flatten would
otherwise drop), the gated ViT FFN gate projection, and the assembled model
graph.
"""

import pytest

from aisimulate.sdk import common
from aisimulate.sdk import config as sdk_config
from aisimulate.sdk.backends.base_backend import BaseBackend
from aisimulate.sdk.models import get_model, get_model_family
from aisimulate.sdk.models.blocks.vit import build_encoder_ops
from aisimulate.sdk.utils import _parse_hf_config_json

pytestmark = pytest.mark.unit

_MODEL_PATH = "mistralai/Mistral-Medium-3.5-128B"


def _raw_config():
    """Minimal Mistral3 config with the fields the parser reads. spatial_merge_size
    sits at the TOP level (Pixtral), not inside vision_config."""
    return {
        "architectures": ["Mistral3ForConditionalGeneration"],
        "model_type": "mistral3",
        "spatial_merge_size": 2,
        "vision_feature_layer": -1,
        "text_config": {
            "head_dim": 128,
            "hidden_size": 12288,
            "intermediate_size": 28672,
            "max_position_embeddings": 262144,
            "num_attention_heads": 96,
            "num_hidden_layers": 88,
            "num_key_value_heads": 8,
            "vocab_size": 131072,
        },
        "vision_config": {
            "head_dim": 104,
            "hidden_size": 1664,
            "intermediate_size": 8192,
            "num_attention_heads": 16,
            "num_hidden_layers": 48,
            "patch_size": 14,
            "image_size": 1540,
        },
    }


class TestMistral3ConfigParsing:
    def test_architecture_maps_to_mistral3_family(self):
        assert common.ARCHITECTURE_TO_MODEL_FAMILY["Mistral3ForConditionalGeneration"] == "MISTRAL3"
        assert "MISTRAL3" in common.ModelFamily

    def test_text_decoder_fields_from_text_config(self):
        result = _parse_hf_config_json(_raw_config())
        assert result["architecture"] == "Mistral3ForConditionalGeneration"
        assert result["layers"] == 88
        assert result["n"] == 96
        assert result["n_kv"] == 8
        assert result["d"] == 128
        assert result["hidden_size"] == 12288
        assert result["inter_size"] == 28672
        assert result["vocab"] == 131072

    def test_vision_encoder_config(self):
        ep = _parse_hf_config_json(_raw_config())["extra_params"]
        assert isinstance(ep, common.VisionEncoderConfig)
        assert ep.depth == 48
        assert ep.hidden_size == 1664
        assert ep.num_heads == 16
        # head_size derives from hidden/heads and must match the HF head_dim (104).
        assert ep.hidden_size // ep.num_heads == 104
        assert ep.intermediate_size == 8192
        assert ep.patch_size == 14
        assert ep.temporal_patch_size == 1
        # Pulled from the TOP-LEVEL spatial_merge_size, not vision_config.
        assert ep.spatial_merge_size == 2
        assert ep.out_hidden_size == 12288
        assert ep.gated_mlp is True
        assert ep.projector_merger_replicated is True
        assert ep.partial_rotary_factor > 0

    def test_projector_dims_are_three_pixtral_gemms(self):
        ep = _parse_hf_config_json(_raw_config())["extra_params"]
        # patch_merger (merger_dim -> vit_hidden), linear_1 (vit_hidden -> text),
        # linear_2 (text -> text). merger_dim = vit_hidden * spatial_merge_size**2.
        assert ep.projector_dims == ((1664 * 4, 1664), (1664, 12288), (12288, 12288))

    @pytest.mark.parametrize("bad", [None, 0, -2, 2.5, True])
    def test_invalid_top_level_spatial_merge_raises(self, bad):
        # spatial_merge_size sizes the patch merger and image-token counts; a
        # silent default (or a truthy-but-invalid value: negative, non-int,
        # bool) would mispredict, so require a positive integer and fail loud.
        cfg = _raw_config()
        cfg["spatial_merge_size"] = bad
        with pytest.raises(ValueError, match="spatial_merge_size"):
            _parse_hf_config_json(cfg)

    def test_missing_top_level_spatial_merge_raises(self):
        cfg = _raw_config()
        del cfg["spatial_merge_size"]
        with pytest.raises(ValueError, match="spatial_merge_size"):
            _parse_hf_config_json(cfg)

    @pytest.mark.parametrize("bad", [{}, [], False, 0, "", [1]])
    def test_malformed_vision_config_raises(self, bad):
        cfg = _raw_config()
        cfg["vision_config"] = bad
        with pytest.raises(ValueError, match="Mistral3 vision_config must be a non-empty object"):
            _parse_hf_config_json(cfg)

    @pytest.mark.parametrize("layer", [-2, 0, 12, [-1], None, True])
    def test_unsupported_vision_feature_layer_raises(self, layer):
        cfg = _raw_config()
        cfg["vision_feature_layer"] = layer
        with pytest.raises(ValueError, match="vision_feature_layer=-1"):
            _parse_hf_config_json(cfg)

    @pytest.mark.parametrize("image_size", [None, 0, -1, True, 1540.0])
    def test_invalid_image_size_raises(self, image_size):
        cfg = _raw_config()
        cfg["vision_config"]["image_size"] = image_size
        with pytest.raises(ValueError, match="positive integer image_size"):
            _parse_hf_config_json(cfg)


class TestMistral3Workload:
    @pytest.mark.parametrize(
        ("height", "width", "patches", "embeddings", "context"),
        [(70, 70, 36, 9, 12), (70, 112, 48, 12, 15), (3080, 1540, 6160, 1540, 1595)],
    )
    def test_image_geometry_reaches_prefill(self, height, width, patches, embeddings, context):
        enc_cfg = _parse_hf_config_json(_raw_config())["extra_params"]
        runtime = sdk_config.RuntimeConfig(isl=100, image_height=height, image_width=width, num_images_per_request=2)
        workload = BaseBackend._encoder_workload_per_visual(runtime, enc_cfg)
        assert workload.patch_tokens_per_sequence == patches
        assert workload.transformer_tokens_per_sequence == patches
        assert workload.output_tokens_per_image == embeddings
        assert workload.context_tokens_per_image == context
        assert BaseBackend.effective_prefill_isl(_MODEL_PATH, runtime) == 100 + 2 * context

    @pytest.mark.parametrize(
        "video",
        [
            {"num_videos_per_request": 1, "video_frames": 2, "video_height": 70, "video_width": 70},
            {"num_videos_per_request": 1, "video_frames": 2, "num_video_tokens": 18},
            {"video_frames": 1},
        ],
    )
    def test_runtime_rejects_video_before_geometry(self, video):
        with pytest.raises(ValueError, match="Video workloads are not modeled for the Pixtral"):
            BaseBackend.effective_prefill_isl(_MODEL_PATH, sdk_config.RuntimeConfig(isl=100, **video))

    def test_image_token_override_needs_dimensions(self):
        with pytest.raises(ValueError, match="Pixtral requires image_height and image_width"):
            BaseBackend.effective_prefill_isl(_MODEL_PATH, sdk_config.RuntimeConfig(isl=100, num_image_tokens=9))

    def test_text_only_context_is_unchanged(self):
        assert BaseBackend.effective_prefill_isl(_MODEL_PATH, sdk_config.RuntimeConfig(isl=100)) == 100


class TestGatedViTBuilder:
    def _cfg(self, gated):
        return common.VisionEncoderConfig(
            depth=2,
            hidden_size=1664,
            num_heads=16,
            intermediate_size=8192,
            patch_size=14,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=12288,
            projector_dims=((6656, 1664), (1664, 12288), (12288, 12288)),
            gated_mlp=gated,
        )

    def test_gated_mlp_emits_gate_gemm(self):
        names = [op._name for op in build_encoder_ops(self._cfg(gated=True), tp_size=1)]
        assert "encoder_gate_gemm" in names
        # gate sits before the up projection.
        assert names.index("encoder_gate_gemm") < names.index("encoder_ffn1_gemm")

    def test_plain_mlp_has_no_gate_gemm(self):
        names = [op._name for op in build_encoder_ops(self._cfg(gated=False), tp_size=1)]
        assert "encoder_gate_gemm" not in names
        assert "encoder_ffn1_gemm" in names

    def test_projector_activation_selection_skips_patch_merger(self):
        names = [
            op._name for op in build_encoder_ops(self._cfg(gated=True), tp_size=1, projector_activation_indices=(1,))
        ]
        assert "encoder_projector_fc0_act" not in names
        assert "encoder_projector_fc1_act" in names


class TestMistral3ModelGraph:
    def test_builds_as_mistral3_model_with_encoder(self):
        assert get_model_family(_MODEL_PATH) == "MISTRAL3"
        model_config = sdk_config.ModelConfig(tp_size=1, attention_dp_size=1)
        model = get_model(_MODEL_PATH, model_config, backend_name="trtllm")

        assert model.model_family == "MISTRAL3"
        assert type(model).__name__ == "Mistral3Model"
        # Dense Mistral GQA has no per-layer q/k norm.
        assert model._use_qk_norm is False

        ctx = {op._name for op in model.context_ops}
        assert "context_attention" in ctx and "context_qkv_gemm" in ctx

        enc = [op._name for op in model.encoder_ops]
        assert "encoder_attention" in enc
        assert "encoder_gate_gemm" in enc
        assert "encoder_projector_fc0_gemm" in enc
        assert "encoder_projector_fc0_act" not in enc
        assert "encoder_projector_fc1_act" in enc
        assert "encoder_projector_fc2_gemm" in enc

    def test_patch_merger_stays_replicated_under_encoder_tp(self):
        model_config = sdk_config.ModelConfig(tp_size=4, attention_dp_size=1, enable_encoder_dp=False)
        model = get_model(_MODEL_PATH, model_config, backend_name="vllm")
        ops_by_name = {op._name: op for op in model.encoder_ops}

        assert "encoder_dp_all_gather" not in ops_by_name
        assert (ops_by_name["encoder_projector_fc0_gemm"]._n, ops_by_name["encoder_projector_fc0_gemm"]._k) == (
            1664,
            6656,
        )
        assert (ops_by_name["encoder_projector_fc1_gemm"]._n, ops_by_name["encoder_projector_fc1_gemm"]._k) == (
            12288 // 4,
            1664,
        )
        assert (ops_by_name["encoder_projector_fc2_gemm"]._n, ops_by_name["encoder_projector_fc2_gemm"]._k) == (
            12288,
            12288 // 4,
        )

    def test_fp8_static_text_gemm_from_checkpoint(self):
        model_config = sdk_config.ModelConfig(tp_size=1, attention_dp_size=1)
        get_model(_MODEL_PATH, model_config, backend_name="trtllm")
        assert model_config.gemm_quant_mode == common.GEMMQuantMode.fp8_static

    @pytest.mark.parametrize(
        ("tp", "encoder_dp", "expected_bytes"),
        [(1, False, 511_104_000), (4, False, 213_734_400), (4, True, 511_104_000), (8, False, 195_148_800)],
    )
    def test_encoder_memory_accounts_for_gated_swiglu_intermediate(self, tp, encoder_dp, expected_bytes):
        # A 1540x1540 image gives 12100 patches and 3025 merged embeddings.
        # BF16 residual: 1664 values/patch, replicated on every TP rank.
        # Gate+up: 16384 values/patch, sharded only in encoder TP mode.
        # Embeddings: 12288 values/token. At TP=8 the legacy 3*1664 floor wins.
        model = get_model(
            _MODEL_PATH,
            sdk_config.ModelConfig(tp_size=tp, attention_dp_size=1, enable_encoder_dp=encoder_dp),
            backend_name="sglang",
        )
        memory = BaseBackend()._get_encoder_component_memory(model, num_tokens=12100, embed_tokens=3025)
        assert memory["activations"] == pytest.approx(expected_bytes / (1 << 30))
