# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified processor/topology test derivatives (Apache-2.0), copyright 2026
# the HuggingFace Inc. team and HuggingFace Team, and copyright contributors
# to the vLLM project. Processor override dictionaries are synthetic fixtures.
# https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/image_processing_kimi_k25.py
# https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/video_processing_kimi_k25.py
# https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/kimi_k25_vit.py

"""Kimi K3 image/video encoder parsing, construction, and runtime tests."""

import dataclasses
import inspect
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends import base_backend as base_backend_module
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.models.vit_ops import build_kimi_k3_encoder_ops
from aiconfigurator.sdk.perf_database import get_database_view
from aiconfigurator.sdk.utils import _parse_hf_config_json, get_model_config_from_model_path

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("config_type", [common.VisionEncoderConfig, common.Gemma4VisionEncoderConfig])
def test_kimi_config_additions_preserve_existing_positional_constructors(config_type):
    positional_fields = (
        "depth",
        "hidden_size",
        "num_heads",
        "intermediate_size",
        "patch_size",
        "temporal_patch_size",
        "spatial_merge_size",
        "out_hidden_size",
        "deepstack_visual_indexes",
        "projector_dims",
        "projector_n_instances",
        "partial_rotary_factor",
        "in_channels",
        "image_size",
        "has_cls_token",
        "max_num_tiles",
        "resize_to_max_canvas",
        "add_global_tile",
        "prompt_image_tokens",
        "prompt_tokens_per_local_tile",
        "final_norm",
        "pool_temporal",
        "video_attention_type",
        "resize_mode",
        "image_max_patches",
        "video_max_patches",
        "max_patches_per_side",
        "max_video_frames",
        "projector_replicated",
    )
    if config_type is common.Gemma4VisionEncoderConfig:
        positional_fields += (
            "num_key_value_heads",
            "head_dim",
            "pooling_kernel_size",
            "position_embedding_size",
            "soft_tokens_per_image",
            "supported_soft_token_budgets",
            "standardize",
        )
    signature = inspect.signature(config_type)
    assert (
        tuple(
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        == positional_fields
    )
    existing = config_type(27, 1024, 16, 4096, 14, 1, 2, 7168)
    reconstructed = config_type(*(getattr(existing, name) for name in positional_fields))
    assert reconstructed == existing
    updated = dataclasses.replace(reconstructed, qkv_hidden_size=1536, projector_pre_norm=False)
    assert updated.qkv_hidden_size == 1536
    assert updated.projector_pre_norm is False
    assert all(getattr(updated, name) == getattr(existing, name) for name in positional_fields)


@pytest.mark.parametrize("language_only", [False, True])
@pytest.mark.parametrize("value", ["missing", None, 0, -1, False, 1.5, "4"])
def test_kimi_k3_rejects_invalid_temporal_limit_before_building_encoder(monkeypatch, language_only, value):
    from aiconfigurator_core.sdk import models as models_module

    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K3")["raw_config"])
    if value == "missing":
        raw["vision_config"].pop("init_pos_emb_time")
    else:
        raw["vision_config"]["init_pos_emb_time"] = value
    monkeypatch.setattr(models_module, "_get_model_info", lambda path: _parse_hf_config_json(raw))
    model_config = _model_config()
    model_config.language_only = language_only
    with pytest.raises(ValueError, match="init_pos_emb_time must be a positive integer"):
        get_model("moonshotai/Kimi-K3", model_config, "sglang")


def _model_config(tp_size: int = 1, *, enable_encoder_dp: bool = True, nextn: int = 0) -> config.ModelConfig:
    return config.ModelConfig(
        tp_size=tp_size,
        attention_dp_size=1,
        moe_tp_size=tp_size,
        moe_ep_size=1,
        enable_encoder_dp=enable_encoder_dp,
        nextn=nextn,
    )


@pytest.mark.parametrize("heads", [0, -1, False, True, None, 1.5, "12"])
def test_kimi_k3_parser_rejects_invalid_vision_head_count(heads):
    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K3")["raw_config"])
    raw["vision_config"]["vt_num_attention_heads"] = heads
    with pytest.raises(ValueError, match="vt_num_attention_heads must be a positive integer"):
        _parse_hf_config_json(raw)


@pytest.fixture
def kimi_k3_model():
    return get_model("moonshotai/Kimi-K3", _model_config(), "sglang")


@pytest.mark.parametrize("language_only", [False, True])
@pytest.mark.parametrize("video", [False, True])
def test_language_only_worker_retains_visual_tokens_without_hosting_encoder(language_only, video):
    model_config = _model_config()
    model_config.language_only = language_only
    model = get_model("moonshotai/Kimi-K3", model_config, "sglang")
    visual_fields = (
        dict(num_images_per_request=0, video_height=448, video_width=448, video_frames=4, num_videos_per_request=1)
        if video
        else dict(image_height=448, image_width=448)
    )
    runtime = config.RuntimeConfig(isl=128, osl=1, **visual_fields)
    backend = BaseBackend()

    assert isinstance(model.encoder_config, common.VisionEncoderConfig)
    assert backend._visual_context_tokens(model, runtime) == 256
    assert bool(model.encoder_ops) is (not language_only)
    assert model.context_ops and model.generation_ops
    if language_only:
        assert backend._get_encoder_component_memory_for_runtime(model, runtime, 1) == {}
        latency, energy, sources, _ = backend._run_encoder_phase(model, object(), runtime, 1)
        assert not latency and not energy and not sources


def test_checkpoint_preserves_language_and_vision_configs_together():
    info = get_model_config_from_model_path("moonshotai/Kimi-K3")
    extra = info["extra_params"]

    assert isinstance(extra, common.KimiK3Config)
    assert extra.layer_types.count("linear_attention") == 69
    assert extra.layer_types.count("full_attention") == 24
    assert isinstance(extra.vision_config, common.VisionEncoderConfig)


def test_architecture_specific_vision_geometry():
    vision = get_model_config_from_model_path("moonshotai/Kimi-K3")["extra_params"].vision_config

    assert (vision.depth, vision.hidden_size, vision.num_heads, vision.intermediate_size) == (27, 1024, 12, 4096)
    assert vision.qkv_hidden_size == 1536
    assert vision.qkv_hidden_size // vision.num_heads == 128
    assert (vision.patch_size, vision.temporal_patch_size, vision.spatial_merge_size) == (14, 1, 2)
    assert vision.projector_dims == ((4096, 4096), (4096, 7168))
    assert vision.out_hidden_size == 7168
    assert vision.pool_temporal is True
    assert vision.final_norm is True
    assert vision.projector_post_norm is True
    assert vision.max_temporal_patches == 4


def test_model_keeps_language_dspark_and_encoder_paths(kimi_k3_model):
    context_names = {op._name for op in kimi_k3_model.context_ops}
    generation_names = {op._name for op in kimi_k3_model.generation_ops}
    encoder_names = {op._name for op in kimi_k3_model.encoder_ops}

    assert "context_kda_scan" in context_names
    assert "context_mla_downscale_gemm" in context_names
    assert "generation_kda_recurrent" in generation_names
    assert {
        "encoder_patch_embed_gemm",
        "encoder_position_embed",
        "encoder_qkv_gemm",
        "encoder_attention",
        "encoder_rope_apply",
        "encoder_final_norm",
        "encoder_patch_merge_pool",
        "encoder_projector_fc0_gemm",
        "encoder_projector_fc1_gemm",
        "encoder_projector_post_norm",
    } <= encoder_names

    dspark = get_model("moonshotai/Kimi-K3", _model_config(nextn=7), "sglang")
    assert "draft_attention" in {op._name for op in dspark.generation_ops}
    assert [op._name for op in dspark.encoder_ops] == [op._name for op in kimi_k3_model.encoder_ops]


def test_encoder_qkv_and_projector_shapes_match_checkpoint(kimi_k3_model):
    encoder_ops = {op._name: op for op in kimi_k3_model.encoder_ops}

    assert (encoder_ops["encoder_qkv_gemm"]._n, encoder_ops["encoder_qkv_gemm"]._k) == (3 * 1536, 1024)
    assert (encoder_ops["encoder_attention"]._n, encoder_ops["encoder_attention"]._head_size) == (12, 128)
    assert (encoder_ops["encoder_proj_gemm"]._n, encoder_ops["encoder_proj_gemm"]._k) == (1024, 1536)
    assert (encoder_ops["encoder_projector_fc0_gemm"]._n, encoder_ops["encoder_projector_fc0_gemm"]._k) == (
        4096,
        4096,
    )
    assert (encoder_ops["encoder_projector_fc1_gemm"]._n, encoder_ops["encoder_projector_fc1_gemm"]._k) == (
        7168,
        4096,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("final_norm", False),
        ("pool_temporal", False),
        ("projector_post_norm", False),
        ("projector_replicated", False),
        ("projector_pre_norm", True),
        ("video_attention_type", ""),
        ("qkv_hidden_size", 0),
        ("max_temporal_patches", 0),
        ("encoder_type", "generic"),
    ],
)
def test_kimi_k3_builder_rejects_incomplete_encoder_semantics(field, value):
    vision = get_model_config_from_model_path("moonshotai/Kimi-K3")["extra_params"].vision_config

    with pytest.raises(ValueError):
        build_kimi_k3_encoder_ops(dataclasses.replace(vision, **{field: value}), tp_size=1)


def test_encoder_parallelism_models_required_communication():
    dp_model = get_model("moonshotai/Kimi-K3", _model_config(tp_size=2), "sglang")
    tp_model = get_model(
        "moonshotai/Kimi-K3",
        _model_config(tp_size=2, enable_encoder_dp=False),
        "sglang",
    )

    assert "encoder_dp_all_gather" in {op._name for op in dp_model.encoder_ops}
    tp_ops = {op._name: op for op in tp_model.encoder_ops}
    assert tp_ops["encoder_ar_1"]._tp_size == 2
    assert tp_ops["encoder_ar_2"]._tp_size == 2
    assert "encoder_projector_ar" not in tp_ops
    assert "encoder_merger_norm" not in tp_ops
    assert (tp_ops["encoder_projector_fc0_gemm"]._n, tp_ops["encoder_projector_fc0_gemm"]._k) == (4096, 4096)
    assert (tp_ops["encoder_projector_fc1_gemm"]._n, tp_ops["encoder_projector_fc1_gemm"]._k) == (7168, 4096)
    assert "encoder_projector_post_norm" in tp_ops


def test_video_temporal_pooling_keeps_context_spatial_and_attention_joint():
    vision = get_model_config_from_model_path("moonshotai/Kimi-K3")["extra_params"].vision_config
    runtime = config.RuntimeConfig(
        video_height=448,
        video_width=448,
        video_frames=4,
        num_videos_per_request=1,
    )

    assert BaseBackend._encoder_pre_merge_per_visual(runtime, vision) == (256, 4096, 1)


def test_video_rejects_more_frames_than_temporal_embedding():
    vision = get_model_config_from_model_path("moonshotai/Kimi-K3")["extra_params"].vision_config
    runtime = config.RuntimeConfig(
        video_height=448,
        video_width=448,
        video_frames=5,
        num_videos_per_request=1,
    )

    with pytest.raises(ValueError, match="supports at most 4 temporal patches"):
        BaseBackend._encoder_pre_merge_per_visual(runtime, vision)


@pytest.mark.parametrize(
    ("height", "width", "video", "expected"),
    [
        (448, 449, False, (272, 1088, 1)),
        (449, 448, False, (272, 1088, 1)),
        (4000, 4000, False, (4225, 16900, 1)),
        (1024, 1024, True, (1089, 17424, 1)),
        (1, 1, False, (1, 4, 1)),
    ],
)
def test_kimi_processor_geometry_reaches_context_and_encoder(kimi_k3_model, height, width, video, expected):
    # Pinned Transformers navit_resize: 448x449 pads to 448x476; 4000²
    # resizes then pads to 1820². Video's 4096 patch budget yields 924².
    visual_fields = (
        dict(num_images_per_request=0, video_height=height, video_width=width, video_frames=4, num_videos_per_request=1)
        if video
        else dict(image_height=height, image_width=width)
    )
    runtime = config.RuntimeConfig(isl=128, osl=1, **visual_fields)
    backend = BaseBackend()
    assert backend._encoder_pre_merge_per_visual(runtime, kimi_k3_model.encoder_config) == expected
    assert backend._visual_context_tokens(kimi_k3_model, runtime) == expected[0]


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("layout", ["native", "legacy-top-level", "legacy-nested"])
def test_checkpoint_processor_limits_reach_runtime(tmp_path, monkeypatch, remote, layout):
    from aiconfigurator_core.sdk import utils as utils_module

    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K3")["raw_config"])
    if layout == "native":
        image_processor = {"max_patches": 1024, "size": {"max_height": 16, "max_width": 16}}
    else:
        image_processor = {"in_patch_limit": 1024, "in_patch_limit_each_frame": 64, "patch_limit_on_one_side": 16}
        if layout == "legacy-nested":
            image_processor = {"media_proc_cfg": image_processor}
    files = {"config.json": raw, "preprocessor_config.json": image_processor}
    if layout == "native":
        files["video_preprocessor_config.json"] = {
            "max_patches": 64,
            "size": {"max_height": 16, "max_width": 16},
        }
    if remote:
        path = f"synthetic-kimi-k3/processor-{tmp_path.name}"
        monkeypatch.setattr(utils_module, "_download_hf_config", lambda model_path: raw)
        monkeypatch.setattr(
            utils_module, "_download_hf_json", lambda model_path, filename, **kwargs: files.get(filename)
        )
    else:
        for filename, content in files.items():
            (tmp_path / filename).write_text(json.dumps(content))
        path = str(tmp_path)
    model = get_model(path, _model_config(), "trtllm")
    # Image is side-limited to 224² (16² patches); video budget scales the
    # same input to 112² (8² patches), four frames pooled to 16 LM tokens.
    image_runtime = config.RuntimeConfig(batch_size=1, isl=128, osl=1, image_height=448, image_width=448)
    video_runtime = config.RuntimeConfig(
        batch_size=1,
        isl=128,
        osl=1,
        num_images_per_request=0,
        num_videos_per_request=1,
        video_frames=4,
        video_height=448,
        video_width=448,
    )
    assert BaseBackend._encoder_pre_merge_per_visual(image_runtime, model.encoder_config) == (64, 256, 1)
    assert BaseBackend._encoder_pre_merge_per_visual(video_runtime, model.encoder_config) == (16, 256, 1)
    database = get_database_view("b200_sxm", "trtllm", "current", database_mode="SOL", allow_missing_data=True)
    backend = TRTLLMBackend()
    for runtime, tokens in ((image_runtime, 64), (video_runtime, 16)):
        assert BaseBackend._visual_context_tokens(model, runtime) == tokens
        summary = backend.run_static(model, database, runtime, mode="static_ctx")
        override = deepcopy(runtime)
        override.image_height = override.image_width = override.video_height = override.video_width = 0
        if runtime.num_videos_per_request:
            override.num_video_tokens = tokens
        else:
            override.num_image_tokens = tokens
        expected = backend.run_static(model, database, override, mode="static_ctx")
        assert summary.get_encoder_latency_dict() == expected.get_encoder_latency_dict()
        assert summary.get_encoder_memory() == expected.get_encoder_memory()


@pytest.mark.parametrize(
    "processor,expected_side",
    [
        ({}, 512),
        ({"size": {"max_height": 512, "max_width": 512}}, 512),
        (
            {
                "size": {"max_height": 16, "max_width": 16},
                "video_processor": {"size": {"max_height": 16, "max_width": 16}},
            },
            16,
        ),
        ({"patch_limit_on_one_side": 16}, 16),
        ({"media_proc_cfg": {"patch_limit_on_one_side": 16}}, 16),
    ],
    ids=["native-defaults", "native-explicit-default", "native-matched", "legacy-top-level", "legacy-nested"],
)
def test_kimi_processor_preserves_native_and_shared_legacy_side_limits(processor, expected_side):
    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K3")["raw_config"])
    raw["preprocessor_config"] = processor
    enc = _parse_hf_config_json(raw)["extra_params"].vision_config
    assert enc.max_patches_per_side == expected_side


@pytest.mark.parametrize(
    "processor",
    [
        {"size": {"max_height": 16, "max_width": 16}},
        {"size": {"max_height": 16, "max_width": 16}, "video_processor": {}},
        {
            "size": {"max_height": 16, "max_width": 16},
            "video_processor": {"size": {"max_height": 32, "max_width": 32}},
        },
        {"video_processor": {"size": {"max_height": 16, "max_width": 16}}},
    ],
    ids=["image-only", "default-video", "explicit-mismatch", "video-only"],
)
def test_kimi_processor_rejects_different_effective_native_side_limits(processor):
    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K3")["raw_config"])
    raw["preprocessor_config"] = processor
    with pytest.raises(ValueError, match="image and video processor side limits must match"):
        _parse_hf_config_json(raw)


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize(
    "image_processor,video_processor",
    [
        ({"size": {"max_height": 16, "max_width": 16}}, None),
        ({"size": {"max_height": 16, "max_width": 16}}, {"max_patches": 4096}),
        ({}, {"size": {"max_height": 16, "max_width": 16}}),
    ],
    ids=["image-only", "default-video", "video-only"],
)
def test_loaded_native_side_override_cannot_change_other_processor_default(
    tmp_path, monkeypatch, remote, image_processor, video_processor
):
    from aiconfigurator_core.sdk import utils as utils_module

    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K3")["raw_config"])
    files = {
        "config.json": raw,
        "preprocessor_config.json": image_processor,
    }
    if video_processor is not None:
        files["video_preprocessor_config.json"] = video_processor
    if remote:
        path = f"synthetic-kimi/native-side-default-{tmp_path.name}"
        monkeypatch.setattr(utils_module, "_download_hf_config", lambda model_path: raw)
        monkeypatch.setattr(
            utils_module, "_download_hf_json", lambda model_path, filename, **kwargs: files.get(filename)
        )
    else:
        for filename, content in files.items():
            (tmp_path / filename).write_text(json.dumps(content))
        path = str(tmp_path)
    # Native image and video sizes each default to 512 independently.
    # A single shared side-limit field cannot model this pair accurately.
    with pytest.raises(ValueError, match="image and video processor side limits must match"):
        get_model(path, _model_config(), "trtllm")


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1024", None])
def test_kimi_processor_rejects_invalid_budget(value):
    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K3")["raw_config"])
    raw["preprocessor_config"] = {"max_patches": value}
    with pytest.raises(ValueError, match="positive integer"):
        _parse_hf_config_json(raw)


def test_token_override_cannot_bypass_kimi_video_temporal_limit(kimi_k3_model):
    runtime = config.RuntimeConfig(
        num_images_per_request=0, num_videos_per_request=1, video_frames=5, num_video_tokens=272
    )
    with pytest.raises(ValueError, match="supports at most 4 temporal patches"):
        BaseBackend._visual_context_tokens(kimi_k3_model, runtime)


@pytest.mark.parametrize("video", [False, True])
def test_kimi_rust_runtime_keeps_projector_replicated_and_accounts_for_weights(video):
    database = get_database_view("b200_sxm", "trtllm", "current", database_mode="SOL", allow_missing_data=True)
    backend = TRTLLMBackend()
    results = []
    for encoder_dp, batch in ((True, 4), (False, 2)):
        model = get_model("moonshotai/Kimi-K3", _model_config(tp_size=2, enable_encoder_dp=encoder_dp), "trtllm")
        runtime = config.RuntimeConfig(
            batch_size=batch,
            isl=128,
            osl=1,
            image_height=0 if video else 448,
            image_width=0 if video else 449,
            num_images_per_request=0 if video else 1,
            num_videos_per_request=1 if video else 0,
            video_height=448 if video else 0,
            video_width=449 if video else 0,
            video_frames=4 if video else 0,
            engine_step_backend="rust",
        )
        summary = backend.run_static(model, database, runtime, mode="static_ctx")
        latency = summary.get_encoder_latency_dict()
        memory = summary.get_encoder_memory()
        results.append(latency)
        assert backend._visual_context_tokens(model, runtime) == 272
        assert latency["encoder_attention"] > 0
        assert "encoder_projector_ar" not in latency
        assert "encoder_merger_norm" not in latency
        assert latency["encoder_projector_post_norm"] > 0
        assert summary.get_result_dict()["ttft"] > sum(latency.values())
        # Independent BF16 parameter count: input projection, 27 transformer
        # blocks (QKV width 1536, tower width 1024), two replicated linears.
        tower_tp = 1 if encoder_dp else 2
        expected_weights = 2 * (
            1024 * 3 * 14**2 + 27 * (4 * 1024 * 1536 + 2 * 1024 * 4096) // tower_tp + 4096**2 + 4096 * 7168
        )
        assert memory["weights"] == pytest.approx(expected_weights / (1 << 30))
        override = deepcopy(runtime)
        override.image_height = override.image_width = override.video_height = override.video_width = 0
        if video:
            override.num_video_tokens = 272
        else:
            override.num_image_tokens = 272
        override_summary = backend.run_static(model, database, override, mode="static_ctx")
        assert override_summary.get_encoder_latency_dict() == latency
        assert override_summary.get_encoder_memory() == memory
        assert override_summary.get_result_dict() == summary.get_result_dict()
    # Both configurations process two visuals per rank, so replicated
    # projector work is identical even when the tower uses TP.
    for name in ("encoder_projector_fc0_gemm", "encoder_projector_fc1_gemm", "encoder_projector_post_norm"):
        assert results[0][name] == pytest.approx(results[1][name])
    assert "encoder_dp_all_gather" in results[0]
    assert results[1]["encoder_ar_1"] > 0


def test_image_and_video_encoder_work_reaches_static_summary(kimi_k3_model, monkeypatch):
    backend = TRTLLMBackend()
    database = SimpleNamespace(
        backend="trtllm",
        version="structural-test",
        system="b200_sxm",
        system_spec={
            "gpu": {"mem_capacity": 1024 * (1 << 30)},
            "misc": {"nccl_mem": {1: 0}, "other_mem": 0},
        },
    )
    monkeypatch.setattr(base_backend_module, "should_use_rust_engine_step", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        base_backend_module,
        "estimate_static_latency_breakdown_with_rust",
        lambda *args, **kwargs: (
            {"context_attention": 1.0},
            {},
            {"context_attention": 2.0},
            {},
            {"context_attention": "structural-test"},
            {},
            (),
        ),
    )
    attention_shapes = []

    def _stub_encoder_phase(model, _database, shape_of, *, include_energy):
        attention = next(op for op in model.encoder_ops if op._name == "encoder_attention")
        shape = shape_of(attention)
        attention_shapes.append(shape)
        latency = shape[0] * shape[1] / 1_000.0
        return (
            {"encoder_attention": latency},
            {"encoder_attention": latency * 2 if include_energy else 0.0},
            {"encoder_attention": "structural-test"},
        )

    monkeypatch.setattr(backend, "_run_encoder_phase_with_rust", _stub_encoder_phase)
    image_runtime = config.RuntimeConfig(
        batch_size=1,
        isl=128,
        osl=1,
        image_height=896,
        image_width=896,
        engine_step_backend="rust",
    )
    video_runtime = config.RuntimeConfig(
        batch_size=1,
        isl=128,
        osl=1,
        num_images_per_request=0,
        video_height=896,
        video_width=896,
        video_frames=4,
        num_videos_per_request=1,
        engine_step_backend="rust",
    )

    image_summary = backend.run_static(kimi_k3_model, database, image_runtime, mode="static_ctx")
    video_summary = backend.run_static(kimi_k3_model, database, video_runtime, mode="static_ctx")

    assert attention_shapes == [(1, 4096), (1, 16384)]
    assert sum(image_summary.get_encoder_latency_dict().values()) > 0
    assert sum(image_summary.get_encoder_energy_wms_dict().values()) > 0
    assert video_summary.get_encoder_memory()["activations"] > image_summary.get_encoder_memory()["activations"]
    assert video_summary.get_result_dict()["ttft"] > image_summary.get_result_dict()["ttft"]
