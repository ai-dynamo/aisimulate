# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architecture tests for the Gemma 4 vision tower and language adapter."""

from copy import deepcopy

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.utils import _parse_hf_config_json, get_model_config_from_model_path

pytestmark = pytest.mark.unit

MODEL = "google/gemma-4-26B-A4B"


@pytest.mark.parametrize("heads", [0, -1, False, True, None, 1.5, "16"])
def test_gemma4_parser_rejects_invalid_vision_head_count(heads):
    raw = deepcopy(get_model_config_from_model_path(MODEL)["raw_config"])
    raw["vision_config"]["num_attention_heads"] = heads
    with pytest.raises(ValueError, match="Gemma 4 vision num_attention_heads must be a positive integer"):
        _parse_hf_config_json(raw)


def _model_config(*, tp_size: int = 1, enable_encoder_dp: bool = True) -> config.ModelConfig:
    return config.ModelConfig(
        tp_size=tp_size,
        moe_tp_size=1,
        moe_ep_size=tp_size,
        enable_encoder_dp=enable_encoder_dp,
    )


def _op(model, name: str):
    return next(op for op in model.encoder_ops if op._name == name)


@pytest.mark.parametrize("language_only", [False, True])
def test_language_only_worker_retains_visual_attention_without_hosting_encoder(language_only):
    model_config = _model_config()
    model_config.language_only = language_only
    model = get_model(MODEL, model_config, "sglang")
    runtime = config.RuntimeConfig(isl=128, osl=1, image_height=448, image_width=448)
    backend = BaseBackend()

    assert isinstance(model.encoder_config, common.Gemma4VisionEncoderConfig)
    assert backend._visual_context_tokens(model, runtime) == 256
    assert bool(model.encoder_ops) is (not language_only)
    assert model.context_ops and model.generation_ops
    # Bidirectional attention over image embeddings belongs to the language
    # worker even when a separate worker runs the vision tower.
    assert backend._has_visual_context_work(model, runtime)
    assert [op._name for op in model.visual_context_ops] == ["context_swa_visual_block_attention"]
    if language_only:
        assert backend._get_encoder_component_memory_for_runtime(model, runtime, 1) == {}
        latency, energy, sources, _ = backend._run_encoder_phase(model, object(), runtime, 1)
        assert not latency and not energy and not sources


def test_gemma4_model_builds_checkpoint_accurate_vision_graph():
    model = get_model(MODEL, _model_config(), "trtllm")

    enc = model.encoder_config
    assert isinstance(enc, common.Gemma4VisionEncoderConfig)
    assert (enc.depth, enc.hidden_size, enc.num_heads, enc.head_dim) == (27, 1152, 16, 72)
    assert enc.intermediate_size == 4304
    assert enc.pooling_kernel_size == 3
    assert enc.soft_tokens_per_image == 280
    assert enc.projector_dims == ((1152, 2816),)
    assert model._gemma4_config.use_bidirectional_vision_attention is True

    names = {op._name for op in model.encoder_ops}
    assert {
        "encoder_patch_embed_gemm",
        "encoder_position_embedding",
        "encoder_qkv_norm_rope_2d",
        "encoder_attention",
        "encoder_ffn_gate_up_gemm",
        "encoder_ffn_act_mul",
        "encoder_ffn_down_gemm",
        "encoder_gemma4_pool_avg",
        "encoder_gemma4_pool_postprocess",
        "encoder_projector_pre_norm",
        "encoder_projector_fc0_gemm",
    } <= names

    # This is a gated Gemma MLP plus average pool + single adapter, not the
    # Qwen3-VL single-up-projection/PatchMerger/two-layer projector graph.
    assert "encoder_ffn1_gemm" not in names
    assert "encoder_projector_fc1_gemm" not in names
    assert _op(model, "encoder_patch_embed_gemm")._k == 3 * 16**2
    assert _op(model, "encoder_qkv_gemm")._n == 3 * 1152
    assert _op(model, "encoder_attention")._head_size == 72
    assert _op(model, "encoder_ffn_gate_up_gemm")._n == 2 * 4304
    assert _op(model, "encoder_gemma4_pool_avg")._scale_num_tokens == 9
    assert (_op(model, "encoder_projector_fc0_gemm")._n, _op(model, "encoder_projector_fc0_gemm")._k) == (
        2816,
        1152,
    )

    visual_attention = model.visual_context_ops
    assert len(visual_attention) == 1
    assert visual_attention[0]._name == "context_swa_visual_block_attention"
    assert visual_attention[0]._scale_factor == 25
    assert (visual_attention[0]._n, visual_attention[0]._head_size) == (16, 256)
    assert (visual_attention[0]._n_kv, visual_attention[0]._window_size) == (8, 1024)


def test_gemma4_encoder_weights_include_patch_position_vit_and_adapter():
    model = get_model(MODEL, _model_config(), "trtllm")

    expected_bf16_weights = (
        1152 * (3 * 16**2)  # patch projection
        + 2 * 10240 * 1152  # learned x/y position tables
        + 27
        * (
            (3 * 1152) * 1152  # QKV
            + 1152 * 1152  # attention output
            + (2 * 4304) * 1152  # gated MLP gate/up
            + 1152 * 4304  # gated MLP down
        )
        + 2816 * 1152  # vision-to-language projection
    )
    assert sum(op.get_weights() for op in model.encoder_ops) == expected_bf16_weights * 2


def test_gemma4_encoder_dp_replicates_compute_and_gathers_soft_tokens():
    model = get_model(MODEL, _model_config(tp_size=4, enable_encoder_dp=True), "trtllm")

    assert _op(model, "encoder_qkv_gemm")._n == 3 * 1152
    assert _op(model, "encoder_ffn_gate_up_gemm")._n == 2 * 4304
    gather = _op(model, "encoder_dp_all_gather")
    assert gather._num_gpus == 4
    # The backend supplies rank-local tokens; the NCCL table axis is the total
    # receive buffer, so the per-token width reconstructs all TP payloads.
    assert gather._num_elements_per_token == 2816 * 4


def test_gemma4_encoder_tp_shards_tower_but_replicates_adapter_and_communicates():
    model = get_model(MODEL, _model_config(tp_size=4, enable_encoder_dp=False), "trtllm")

    names = {op._name for op in model.encoder_ops}
    assert "encoder_dp_all_gather" not in names
    assert _op(model, "encoder_qkv_gemm")._n == 3 * 1152 // 4
    assert _op(model, "encoder_attention")._n == 16 // 4
    assert _op(model, "encoder_ffn_gate_up_gemm")._n == 2 * 4304 // 4
    # vLLM uses ReplicatedLinear for Gemma4MultimodalEmbedder and HF's
    # vision TP plan excludes this projection.
    assert _op(model, "encoder_projector_fc0_gemm")._n == 2816
    assert _op(model, "encoder_ar_1")._tp_size == 4
    assert _op(model, "encoder_ar_2")._tp_size == 4
    assert "encoder_projector_ar" not in names


class TestGemma4VisionRuntime:
    """Gemma 4 derives a pooled grid from its aspect-ratio resize contract."""

    @staticmethod
    def _model():
        return get_model(
            MODEL,
            config.ModelConfig(moe_tp_size=1, moe_ep_size=1),
            "trtllm",
        )

    def test_checkpoint_default_maps_one_image_to_280_soft_tokens_and_2520_patches(self):
        model = self._model()
        runtime = config.RuntimeConfig(
            batch_size=1,
            isl=512,
            osl=64,
            image_height=672,
            image_width=960,
            num_images_per_request=1,
        )

        post_pool, pre_pool, num_visuals = BaseBackend._encoder_pre_merge_per_visual(runtime, model.encoder_config)

        assert (post_pool, pre_pool, num_visuals) == (280, 2520, 1)
        assert BaseBackend._visual_context_tokens(model, runtime) == 280

    def test_square_image_keeps_padded_tower_budget_but_fewer_soft_tokens(self):
        model = self._model()
        runtime = config.RuntimeConfig(
            batch_size=1,
            isl=512,
            osl=64,
            image_height=448,
            image_width=448,
            num_images_per_request=1,
        )

        post_pool, pre_pool, num_visuals = BaseBackend._encoder_pre_merge_per_visual(runtime, model.encoder_config)

        assert (post_pool, pre_pool, num_visuals) == (256, 2520, 1)
        assert BaseBackend._visual_context_tokens(model, runtime) == 256

    def test_dynamic_soft_token_override_follows_supported_budget(self):
        model = self._model()
        runtime = config.RuntimeConfig(
            batch_size=1,
            isl=512,
            osl=64,
            image_height=672,
            image_width=960,
            num_images_per_request=2,
            num_image_tokens=560,
        )

        post_pool, pre_pool, num_visuals = BaseBackend._encoder_pre_merge_per_visual(runtime, model.encoder_config)

        assert (post_pool, pre_pool, num_visuals) == (532, 5040, 2)
        assert BaseBackend._visual_context_tokens(model, runtime) == 1064

    def test_invalid_soft_token_budget_is_rejected(self):
        model = self._model()
        runtime = config.RuntimeConfig(
            batch_size=1,
            isl=512,
            osl=64,
            image_height=672,
            image_width=960,
            num_image_tokens=281,
        )

        with pytest.raises(ValueError, match="must be one of"):
            BaseBackend._encoder_pre_merge_per_visual(runtime, model.encoder_config)

    def test_encoder_tp_memory_uses_rank_local_activation_widths(self):
        model = get_model(
            MODEL,
            config.ModelConfig(
                tp_size=4,
                moe_tp_size=1,
                moe_ep_size=4,
                enable_encoder_dp=False,
            ),
            "trtllm",
        )
        enc_cfg = model.encoder_config

        memory = BaseBackend()._get_encoder_component_memory(model, num_tokens=5040, embed_tokens=280)

        qkv_width = 3 * enc_cfg.hidden_size // model.config.tp_size
        gated_mlp_width = enc_cfg.hidden_size + (2 * enc_cfg.intermediate_size) // model.config.tp_size
        expected_bytes = 2 * 5040 * max(qkv_width, gated_mlp_width)
        expected_bytes += 2 * 280 * (2 * enc_cfg.hidden_size + enc_cfg.out_hidden_size)
        assert expected_bytes > 32 * 1024 * 1024
        assert memory["activations"] == pytest.approx(expected_bytes / (1 << 30))

    @pytest.mark.parametrize("include_energy", [True, False])
    def test_visual_attention_overlay_uses_native_kernel_without_fused_extras(self, include_energy):
        from aiconfigurator_core.sdk.engine import build_ops_json
        from aiconfigurator_core.sdk.perf_database import get_database
        from aiconfigurator_core.sdk.rust_engine_step import _cached_engine_handle

        model = get_model(MODEL, _model_config(), "vllm")
        database = get_database("b200_sxm", "vllm", "0.24.0", database_mode="SOL")
        runtime = config.RuntimeConfig(
            batch_size=2,
            isl=512,
            osl=2,
            image_height=672,
            image_width=960,
            num_images_per_request=2,
            seq_imbalance_correction_scale=1.25,
        )
        handle = _cached_engine_handle(model, database)
        ops_json = build_ops_json(model.visual_context_ops)
        kernel = handle.evaluate_context_attention_kernels_json(
            ops_json, batch_size=4, s=280, imbalance_correction_scale=1.25
        )
        full = handle.evaluate_ops_json(ops_json, is_context=True, batch_size=4, s=280, imbalance_correction_scale=1.25)
        assert len(kernel) == len(full) == 1
        name, kernel_latency, kernel_energy, kernel_source = kernel[0]
        assert full[0][1] > kernel_latency

        latency, energy, source = BaseBackend()._run_visual_context_phase(
            model, database, runtime, batch_size=2, include_energy=include_energy
        )

        assert latency == {name: pytest.approx(kernel_latency * 279 / 280)}
        assert energy == {name: pytest.approx(kernel_energy * 279 / 280) if include_energy else 0.0}
        assert source == {name: kernel_source}

    def test_visual_attention_overlay_uses_compiled_kernel_evaluation(self, monkeypatch):
        from aiconfigurator_core.sdk import engine as engine_module
        from aiconfigurator_core.sdk import rust_engine_step as rust_engine_module

        model = self._model()
        runtime = config.RuntimeConfig(
            batch_size=2,
            isl=512,
            osl=2,
            image_height=672,
            image_width=960,
            num_images_per_request=1,
            seq_imbalance_correction_scale=1.25,
        )
        captured = {}
        monkeypatch.setattr(engine_module, "build_ops_json", lambda ops: "visual-ops")

        def _evaluate(*args, **kwargs):
            captured.update(kwargs)
            return [("context_swa_visual_block_attention", 10.0, 20.0, "silicon")]

        monkeypatch.setattr(rust_engine_module, "evaluate_context_attention_kernels_with_rust", _evaluate)

        latency, energy, source = BaseBackend()._run_visual_context_phase(model, object(), runtime, batch_size=2)

        assert latency["context_swa_visual_block_attention"] == 10.0
        assert energy["context_swa_visual_block_attention"] == 20.0
        assert source == {"context_swa_visual_block_attention": "silicon"}
        assert captured["ops_json"] == "visual-ops"
        assert captured["batch_size"] == 2
        assert captured["s"] == 280
        assert captured["imbalance_correction_scale"] == 1.25
        assert captured["visual_block_upper_triangle"] is True
