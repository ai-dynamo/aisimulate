# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends import base_backend as base_backend_module
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.models import _get_model_info, get_model
from aiconfigurator.sdk.models.kimi_k3 import KimiK3Model
from aiconfigurator.sdk.models.vit_ops import build_kimi_k3_encoder_ops

pytestmark = pytest.mark.unit


def _model_config(**kwargs) -> config.ModelConfig:
    kwargs.setdefault("moe_tp_size", 1)
    kwargs.setdefault("moe_ep_size", 1)
    return config.ModelConfig(**kwargs)


@pytest.fixture
def kimi_k3_model() -> KimiK3Model:
    return get_model("moonshotai/Kimi-K3", _model_config(), "sglang")


class TestKimiK3VisionConfig:
    def test_checkpoint_preserves_language_and_vision_configs_together(self):
        extra = _get_model_info("moonshotai/Kimi-K3")["extra_params"]

        assert isinstance(extra, common.KimiK3Config)
        assert extra.layer_types.count("linear_attention") == 69
        assert extra.layer_types.count("full_attention") == 24
        assert isinstance(extra.vision_config, common.VisionEncoderConfig)

    def test_architecture_specific_vision_geometry(self):
        vision = _get_model_info("moonshotai/Kimi-K3")["extra_params"].vision_config

        assert vision.depth == 27
        assert vision.hidden_size == 1024
        assert vision.num_heads == 12
        assert vision.qkv_hidden_size == 1536
        assert vision.qkv_hidden_size // vision.num_heads == 128
        assert vision.patch_size == 14
        assert vision.temporal_patch_size == 1
        assert vision.spatial_merge_size == 2
        assert vision.temporal_pool_all
        assert vision.max_temporal_patches == 4
        assert vision.projector_dims == ((4096, 4096), (4096, 7168))
        assert vision.projector_post_norm

    def test_kimi_k25_does_not_inherit_k3_patchmergerv2_semantics(self):
        k25_info = _get_model_info("moonshotai/Kimi-K2.5")
        k25_model = get_model("moonshotai/Kimi-K2.5", _model_config(), "sglang")

        assert isinstance(k25_info["extra_params"], dict)
        assert not hasattr(k25_info["extra_params"], "vision_config")
        assert k25_model.encoder_ops == []


class TestKimiK3VisionModel:
    def test_model_keeps_language_and_dspark_paths(self, kimi_k3_model):
        context_names = {op._name for op in kimi_k3_model.context_ops}
        generation_names = {op._name for op in kimi_k3_model.generation_ops}

        assert "context_kda_scan" in context_names
        assert "context_mla_downscale_gemm" in context_names
        assert "generation_kda_recurrent" in generation_names

        dspark = get_model(
            "moonshotai/Kimi-K3",
            _model_config(nextn=7),
            "sglang",
        )
        assert "draft_attention" in {op._name for op in dspark.generation_ops}
        assert [op._name for op in dspark.encoder_ops] == [op._name for op in kimi_k3_model.encoder_ops]

    def test_encoder_ops_cover_moonvit3d_patchmergerv2_and_communication(self, kimi_k3_model):
        ops = {op._name: op for op in kimi_k3_model.encoder_ops}

        assert ops["encoder_patch_embed_gemm"]._n == 1024
        assert ops["encoder_patch_embed_gemm"]._k == 3 * 14 * 14
        assert ops["encoder_qkv_gemm"]._n == 3 * 1536
        assert ops["encoder_qkv_gemm"]._k == 1024
        assert ops["encoder_attention"]._n == 12
        assert ops["encoder_attention"]._head_size == 128
        assert ops["encoder_proj_gemm"]._n == 1024
        assert ops["encoder_proj_gemm"]._k == 1536
        assert ops["encoder_projector_fc0_gemm"]._n == 4096
        assert ops["encoder_projector_fc0_gemm"]._k == 4096
        assert ops["encoder_projector_fc1_gemm"]._n == 7168
        assert ops["encoder_projector_fc1_gemm"]._k == 4096
        assert "encoder_spatial_temporal_merge" in ops
        assert "encoder_projector_post_norm" in ops
        assert "encoder_ar_1" in ops
        assert "encoder_ar_2" in ops
        assert "encoder_projector_ar" in ops

    def test_encoder_dp_adds_embedding_all_gather(self):
        model = get_model(
            "moonshotai/Kimi-K3",
            _model_config(tp_size=2, moe_tp_size=2),
            "sglang",
        )

        assert "encoder_dp_all_gather" in {op._name for op in model.encoder_ops}

    @pytest.mark.parametrize(
        "disabled_flag",
        [
            "model_patch_embed",
            "model_pos_embed",
            "model_final_norm",
            "temporal_pool_all",
            "projector_post_norm",
        ],
    )
    def test_kimi_k3_builder_rejects_incomplete_encoder_semantics(self, disabled_flag):
        vision = _get_model_info("moonshotai/Kimi-K3")["extra_params"].vision_config
        incomplete = dataclasses.replace(vision, **{disabled_flag: False})

        with pytest.raises(ValueError, match="complete MoonViT3D"):
            build_kimi_k3_encoder_ops(incomplete, tp_size=1)


class _TestBackend(BaseBackend):
    def find_best_agg_result_under_constraints(self, model, database, runtime_config, **kwargs):
        raise NotImplementedError

    def _get_memory_usage(self, *args, **kwargs):
        return {"total": 1.0}


@pytest.fixture
def synthetic_database():
    return SimpleNamespace(
        backend="sglang",
        version="test",
        system="test",
        system_spec={"gpu": {"mem_capacity": 80 * (1 << 30)}},
    )


def _stub_compiled_encoder(backend, *, source_for_s=None):
    calls: dict[str, list[int]] = {}
    backend._require_rust_engine_step = lambda *args, **kwargs: None

    def _run(model, _database, _visuals_local, eff_s_of, *, include_energy):
        latency: dict[str, float] = {}
        energy: dict[str, float] = {}
        sources: dict[str, str] = {}
        for op in model.encoder_ops:
            eff_s = eff_s_of(op)
            calls.setdefault(op._name, []).append(eff_s)
            latency[op._name] = 1.0
            energy[op._name] = 2.0 if include_energy else 0.0
            sources[op._name] = source_for_s(eff_s) if source_for_s else "test"
        return latency, energy, sources

    backend._run_encoder_phase_with_rust = _run
    return calls


class TestKimiK3VisionRuntime:
    def test_image_workload_executes_nonzero_encoder_work(self, kimi_k3_model, synthetic_database):
        backend = _TestBackend()
        calls = _stub_compiled_encoder(backend)
        runtime = config.RuntimeConfig(
            batch_size=1,
            isl=64,
            osl=2,
            image_height=448,
            image_width=448,
            num_images_per_request=1,
            engine_step_backend="rust",
        )

        latency, energy, _, visual_tokens = backend._run_encoder_phase(
            kimi_k3_model, synthetic_database, runtime, batch_size=1
        )

        assert sum(latency.values()) > 0
        assert sum(energy.values()) > 0
        assert visual_tokens == 16 * 16
        assert calls["encoder_attention"] == [32 * 32]

    def test_video_attention_scales_with_frames_but_tpool_output_does_not(self, kimi_k3_model, synthetic_database):
        backend = _TestBackend()
        calls = _stub_compiled_encoder(backend)
        runtime = config.RuntimeConfig(
            batch_size=1,
            isl=64,
            osl=2,
            num_images_per_request=0,
            video_height=448,
            video_width=448,
            video_num_frames=4,
            num_videos_per_request=1,
            engine_step_backend="rust",
        )

        latency, _, _, visual_tokens = backend._run_encoder_phase(
            kimi_k3_model, synthetic_database, runtime, batch_size=1
        )

        assert sum(latency.values()) > 0
        assert visual_tokens == 16 * 16
        assert calls["encoder_attention"] == [4 * 32 * 32]
        assert calls["encoder_projector_fc0_gemm"] == [16 * 16]

    def test_mixed_image_video_workloads_preserve_mixed_source(self, kimi_k3_model, synthetic_database):
        backend = _TestBackend()
        _stub_compiled_encoder(backend, source_for_s=lambda eff_s: "silicon" if eff_s == 32 * 32 else "sol")
        runtime = config.RuntimeConfig(
            batch_size=1,
            isl=64,
            osl=2,
            image_height=448,
            image_width=448,
            num_images_per_request=1,
            video_height=448,
            video_width=448,
            video_num_frames=4,
            num_videos_per_request=1,
            engine_step_backend="rust",
        )

        _, _, sources, _ = backend._run_encoder_phase(kimi_k3_model, synthetic_database, runtime, batch_size=1)

        assert sources["encoder_attention"] == "mixed"
        assert sources["encoder_projector_fc0_gemm"] == "sol"

    def test_video_rejects_more_frames_than_k3_temporal_embedding(self, kimi_k3_model, synthetic_database):
        runtime = config.RuntimeConfig(
            batch_size=1,
            isl=64,
            osl=2,
            num_images_per_request=0,
            video_height=448,
            video_width=448,
            video_num_frames=5,
            num_videos_per_request=1,
            engine_step_backend="rust",
        )

        with pytest.raises(ValueError, match="supports at most 4 temporal patches"):
            _TestBackend()._run_encoder_phase(kimi_k3_model, synthetic_database, runtime, batch_size=1)

    def test_encoder_latency_memory_energy_and_ttft_reach_summary(self, kimi_k3_model, synthetic_database, monkeypatch):
        backend = _TestBackend()
        _stub_compiled_encoder(backend)
        monkeypatch.setattr(base_backend_module, "should_use_rust_engine_step", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            base_backend_module,
            "estimate_static_latency_breakdown_with_rust",
            lambda *args, **kwargs: (
                {"context_attention": 1.0},
                {"generation_attention": 1.0},
                {"context_attention": 2.0},
                {"generation_attention": 2.0},
                {"context_attention": "test"},
                {"generation_attention": "test"},
                (),
            ),
        )
        summary = backend.run_static(
            kimi_k3_model,
            synthetic_database,
            config.RuntimeConfig(
                batch_size=1,
                isl=64,
                osl=2,
                num_images_per_request=0,
                video_height=448,
                video_width=448,
                video_num_frames=4,
                num_videos_per_request=1,
                engine_step_backend="rust",
            ),
            mode="static",
            stride=1,
        )

        encoder_latency = sum(summary.get_encoder_latency_dict().values())
        context_latency = sum(summary.get_context_latency_dict().values())
        assert encoder_latency > 0
        assert sum(summary.get_encoder_energy_wms_dict().values()) > 0
        assert summary.get_encoder_memory()["weights"] > 0
        assert summary.get_encoder_memory()["activations"] > 0
        assert summary.get_summary_df().iloc[0]["ttft"] == pytest.approx(encoder_latency + context_latency)
