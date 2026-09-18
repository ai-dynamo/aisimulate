# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified processor/topology test derivatives (Apache-2.0), copyright 2026
# the HuggingFace Inc. team and HuggingFace Team, and copyright contributors
# to the vLLM project. Processor override dictionaries are synthetic fixtures.
# https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/image_processing_kimi_k25.py
# https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/video_processing_kimi_k25.py
# https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/kimi_k25_vit.py

"""Kimi K2.5 image/video encoder parsing, construction, and runtime tests."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from aisimulate.sdk import common, config
from aisimulate.sdk.backends import base_backend as base_backend_module
from aisimulate.sdk.backends.base_backend import BaseBackend
from aisimulate.sdk.backends.trtllm_backend import TRTLLMBackend
from aisimulate.sdk.models import get_model
from aisimulate.sdk.perf_database import get_database_view
from aisimulate.sdk.utils import _parse_hf_config_json, get_model_config_from_model_path

pytestmark = pytest.mark.unit

_KIMI_MODELS = ("moonshotai/Kimi-K2.5", "nvidia/Kimi-K2.5-NVFP4")


def _model_config(tp_size: int = 1, *, enable_encoder_dp: bool = True) -> config.ModelConfig:
    return config.ModelConfig(
        tp_size=tp_size,
        attention_dp_size=1,
        moe_tp_size=tp_size,
        moe_ep_size=1,
        enable_encoder_dp=enable_encoder_dp,
    )


@pytest.mark.parametrize("model_id", _KIMI_MODELS)
def test_real_vision_config_is_preserved_alongside_language_config(model_id):
    info = get_model_config_from_model_path(model_id)

    assert "text_config" in info["raw_config"]
    assert info["raw_config"]["vision_config"]["video_attn_type"] == "spatial_temporal"
    assert info["extra_params"] == {
        "v_head_dim": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
    }
    assert isinstance(info["encoder_config"], common.VisionEncoderConfig)


@pytest.mark.parametrize("model_id", _KIMI_MODELS)
def test_kimi_vision_geometry_matches_published_checkpoint(model_id):
    enc = get_model_config_from_model_path(model_id)["encoder_config"]

    assert (enc.depth, enc.hidden_size, enc.num_heads, enc.intermediate_size) == (27, 1152, 16, 4304)
    assert (enc.patch_size, enc.temporal_patch_size, enc.spatial_merge_size) == (14, 1, 2)
    assert enc.projector_dims == ((4608, 4608), (4608, 7168))
    assert enc.out_hidden_size == 7168
    assert enc.partial_rotary_factor == 1.0
    assert enc.video_attention_type == "spatial_temporal"
    assert enc.pool_temporal is True
    assert enc.final_norm is True


@pytest.mark.parametrize("model_id", _KIMI_MODELS)
def test_kimi_builds_spatial_temporal_vit_patch_merger_and_projector(model_id):
    model = get_model(model_id, _model_config(), "trtllm")
    names = {op._name for op in model.encoder_ops}

    assert {
        "encoder_patch_embed_gemm",
        "encoder_position_embed",
        "encoder_qkv_gemm",
        "encoder_attention",
        "encoder_rope_apply",
        "encoder_final_norm",
        "encoder_merger_norm",
        "encoder_patch_merge_pool",
        "encoder_projector_fc0_gemm",
        "encoder_projector_fc1_gemm",
    } <= names
    assert "encoder_projector_ar" not in names


@pytest.mark.parametrize(
    "model_id,language_modes",
    [
        (
            "moonshotai/Kimi-K2.5",
            (
                common.GEMMQuantMode.bfloat16,
                common.MoEQuantMode.int4_wo,
                common.FMHAQuantMode.bfloat16,
                common.KVCacheQuantMode.bfloat16,
            ),
        ),
        (
            "nvidia/Kimi-K2.5-NVFP4",
            (
                common.GEMMQuantMode.nvfp4,
                common.MoEQuantMode.nvfp4,
                common.FMHAQuantMode.fp8,
                common.KVCacheQuantMode.fp8,
            ),
        ),
    ],
)
def test_language_checkpoint_modes_are_retained_while_encoder_stays_bf16(model_id, language_modes):
    model_cfg = _model_config()
    model = get_model(model_id, model_cfg, "trtllm")

    assert (
        model_cfg.gemm_quant_mode,
        model_cfg.moe_quant_mode,
        model_cfg.fmha_quant_mode,
        model_cfg.kvcache_quant_mode,
    ) == language_modes
    assert {op._quant_mode for op in model.encoder_ops if hasattr(op, "_quant_mode")} == {common.GEMMQuantMode.bfloat16}
    encoder_attention = next(op for op in model.encoder_ops if op._name == "encoder_attention")
    assert encoder_attention._fmha_quant_mode == common.FMHAQuantMode.bfloat16


def test_model_level_quantization_must_explicitly_exclude_kimi_vision_components():
    raw = deepcopy(get_model_config_from_model_path("nvidia/Kimi-K2.5-NVFP4")["raw_config"])
    raw["quantization_config"]["ignore"] = ["language_model.lm_head"]

    with pytest.raises(ValueError, match="does not explicitly exclude both vision_tower and mm_projector"):
        _parse_hf_config_json(raw)


@pytest.mark.parametrize("language_only", [False, True])
@pytest.mark.parametrize(
    "ignore,match",
    [
        (["language_model.lm_head"], "does not explicitly exclude both"),
        (["vision_tower.*", "mm_projector.linear_1"], "does not explicitly exclude both"),
        (["vision_tower.encoder.*", "mm_projector.*"], "does not explicitly exclude both"),
        (["re:vision_tower.*", "re:mm_projector.*"], "does not explicitly exclude both"),
        ("vision_tower mm_projector", "ignore must be a list or tuple of strings"),
        (["vision_tower", "mm_projector", None], "ignore must be a list or tuple of strings"),
    ],
)
def test_equal_root_and_text_quantization_still_requires_vision_exclusions(tmp_path, language_only, ignore, match):
    raw = deepcopy(get_model_config_from_model_path("nvidia/Kimi-K2.5-NVFP4")["raw_config"])
    raw["quantization_config"]["ignore"] = ignore
    raw["text_config"]["quantization_config"] = deepcopy(raw["quantization_config"])
    (tmp_path / "config.json").write_text(json.dumps(raw))
    model_cfg = _model_config()
    model_cfg.language_only = language_only

    with pytest.raises(ValueError, match=match):
        get_model(str(tmp_path), model_cfg, "trtllm")


@pytest.mark.parametrize("language_only", [False, True])
@pytest.mark.parametrize("ignore", [["*"], ["vision_tower", "mm_projector"], ["vision_tower.*", "mm_projector.*"]])
def test_equal_root_and_text_quantization_accepts_complete_vision_exclusions(tmp_path, language_only, ignore):
    raw = deepcopy(get_model_config_from_model_path("nvidia/Kimi-K2.5-NVFP4")["raw_config"])
    raw["quantization_config"]["ignore"] = ignore
    raw["text_config"]["quantization_config"] = deepcopy(raw["quantization_config"])
    (tmp_path / "config.json").write_text(json.dumps(raw))
    model_cfg = _model_config()
    model_cfg.language_only = language_only

    model = get_model(str(tmp_path), model_cfg, "trtllm")

    assert model_cfg.gemm_quant_mode == common.GEMMQuantMode.nvfp4
    assert model_cfg.moe_quant_mode == common.MoEQuantMode.nvfp4
    assert model_cfg.fmha_quant_mode == common.FMHAQuantMode.fp8
    assert model_cfg.kvcache_quant_mode == common.KVCacheQuantMode.fp8
    if language_only:
        assert not model.encoder_ops
    else:
        assert {op._quant_mode for op in model.encoder_ops if hasattr(op, "_quant_mode")} == {
            common.GEMMQuantMode.bfloat16
        }


@pytest.mark.parametrize("source", ["bundled", "local", "downloaded"])
@pytest.mark.parametrize("language_only", [False, True])
def test_repeated_text_only_quantized_loads_preserve_scope_and_language_modes(
    tmp_path, monkeypatch, source, language_only
):
    from aisimulate_core.sdk import utils as utils_module
    from aisimulate_core.sdk.models.helpers import _get_model_info

    model_id = "moonshotai/Kimi-K2.5"
    raw = utils_module._load_pre_downloaded_hf_config(model_id)
    text_quant = deepcopy(raw["text_config"]["quantization_config"])
    assert "quantization_config" not in raw
    if source == "local":
        (tmp_path / "config.json").write_text(json.dumps(raw))
        model_id = str(tmp_path)
    elif source == "downloaded":
        model_id = f"synthetic-kimi-k25/text-quant-{tmp_path.name}"
        monkeypatch.setattr(utils_module, "_download_hf_config", lambda model_path: deepcopy(raw))
        monkeypatch.setattr(utils_module, "_download_hf_json", lambda *args, **kwargs: None)

    _get_model_info.cache_clear()
    get_model_config_from_model_path.cache_clear()
    utils_module._load_model_config_from_model_path.cache_clear()
    for load_round in range(3):
        # Exercise cold loading, the parsed cache, then reparsing cached raw
        # metadata. None may promote text-only quantization into model scope.
        if load_round == 2:
            _get_model_info.cache_clear()
            get_model_config_from_model_path.cache_clear()
        model_cfg = _model_config()
        model_cfg.language_only = language_only
        model = get_model(model_id, model_cfg, "trtllm")
        loaded = get_model_config_from_model_path(model_id)["raw_config"]

        assert "quantization_config" not in loaded
        assert loaded["text_config"]["quantization_config"] == text_quant
        assert model_cfg.gemm_quant_mode == common.GEMMQuantMode.bfloat16
        assert model_cfg.moe_quant_mode == common.MoEQuantMode.int4_wo
        assert model_cfg.fmha_quant_mode == common.FMHAQuantMode.bfloat16
        assert model_cfg.kvcache_quant_mode == common.KVCacheQuantMode.bfloat16
        if language_only:
            assert not model.encoder_ops
        else:
            assert {op._quant_mode for op in model.encoder_ops if hasattr(op, "_quant_mode")} == {
                common.GEMMQuantMode.bfloat16
            }


@pytest.mark.parametrize("ignore", [None, "vision_tower mm_projector", 1, {}, ["vision_tower", "mm_projector", None]])
def test_kimi_quantization_rejects_invalid_ignore_container(ignore):
    raw = deepcopy(get_model_config_from_model_path("nvidia/Kimi-K2.5-NVFP4")["raw_config"])
    raw["quantization_config"]["ignore"] = ignore
    with pytest.raises(ValueError, match="quantization_config.ignore must be a list or tuple of strings"):
        _parse_hf_config_json(raw)


@pytest.mark.parametrize("container", [list, tuple])
def test_kimi_quantization_accepts_valid_ignore_container(container):
    raw = deepcopy(get_model_config_from_model_path("nvidia/Kimi-K2.5-NVFP4")["raw_config"])
    raw["quantization_config"]["ignore"] = container(["vision_tower.*", "mm_projector.*"])
    assert _parse_hf_config_json(raw)["encoder_config"].hidden_size == 1152


@pytest.mark.parametrize("suffix", ["", "*", ".*"])
def test_kimi_quantization_accepts_complete_modelopt_components(suffix):
    raw = deepcopy(get_model_config_from_model_path("nvidia/Kimi-K2.5-NVFP4")["raw_config"])
    raw["quantization_config"]["ignore"] = [f"vision_tower{suffix}", f"mm_projector{suffix}"]
    assert _parse_hf_config_json(raw)["encoder_config"].hidden_size == 1152


@pytest.mark.parametrize(
    "exclusions",
    [
        ["vision_tower.encoder.layers.0", "mm_projector.linear_1"],
        ["vision_tower.encoder.*", "mm_projector.*"],
        ["vision_tower.*", "mm_projector.linear_1"],
        ["other_vision_tower*", "other_mm_projector*"],
        ["vision_tower_extra*", "mm_projector_extra*"],
        ["VISION_TOWER*", "MM_PROJECTOR*"],
        ["re:vision_tower.*", "re:mm_projector.*"],
    ],
)
def test_kimi_quantization_rejects_partial_or_ambiguous_component_coverage(exclusions):
    raw = deepcopy(get_model_config_from_model_path("nvidia/Kimi-K2.5-NVFP4")["raw_config"])
    raw["quantization_config"]["ignore"] = exclusions
    with pytest.raises(ValueError, match="does not explicitly exclude both"):
        _parse_hf_config_json(raw)


def test_kimi_quantization_does_not_assume_other_methods_match_modelopt():
    raw = deepcopy(get_model_config_from_model_path("nvidia/Kimi-K2.5-NVFP4")["raw_config"])
    raw["quantization_config"]["quant_method"] = "compressed-tensors"
    with pytest.raises(ValueError, match="supported modelopt matching semantics"):
        _parse_hf_config_json(raw)


@pytest.mark.parametrize("heads", [0, -1, False, True, None, 1.5, "16"])
def test_kimi_k25_parser_rejects_invalid_vision_head_count(heads):
    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K2.5")["raw_config"])
    raw["vision_config"]["vt_num_attention_heads"] = heads
    with pytest.raises(ValueError, match="vt_num_attention_heads must be a positive integer"):
        _parse_hf_config_json(raw)


@pytest.mark.parametrize("model_id", _KIMI_MODELS)
@pytest.mark.parametrize("language_only", [False, True])
@pytest.mark.parametrize("heads", [5, 17, 19])
def test_kimi_local_checkpoint_rejects_incompatible_vision_head_count(tmp_path, model_id, language_only, heads):
    raw = deepcopy(get_model_config_from_model_path(model_id)["raw_config"])
    raw["vision_config"]["vt_num_attention_heads"] = heads
    (tmp_path / "config.json").write_text(json.dumps(raw))
    model_cfg = _model_config()
    model_cfg.language_only = language_only

    with pytest.raises(ValueError, match="vt_hidden_size must be divisible by vt_num_attention_heads"):
        get_model(str(tmp_path), model_cfg, "trtllm")


@pytest.mark.parametrize("model_id", _KIMI_MODELS)
@pytest.mark.parametrize("language_only", [False, True])
def test_kimi_local_checkpoint_preserves_valid_vision_attention_width(tmp_path, model_id, language_only):
    raw = deepcopy(get_model_config_from_model_path(model_id)["raw_config"])
    (tmp_path / "config.json").write_text(json.dumps(raw))
    model_cfg = _model_config()
    model_cfg.language_only = language_only

    model = get_model(str(tmp_path), model_cfg, "trtllm")

    assert (model.encoder_config.hidden_size, model.encoder_config.num_heads) == (1152, 16)
    if language_only:
        assert not model.encoder_ops
    else:
        encoder_ops = {op._name: op for op in model.encoder_ops}
        qkv = encoder_ops["encoder_qkv_gemm"]
        attention = encoder_ops["encoder_attention"]
        assert (qkv._n, qkv._k) == (3456, 1152)
        assert (attention._n, attention._head_size) == (16, 72)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("merge_kernel_size", [2, 3], "square merge_kernel_size"),
        ("mm_projector_type", "mlp", "mm_projector_type='patchmerger'"),
        ("merge_type", "sd2", "merge_type='sd2_tpool'"),
        ("video_attn_type", "spatial", "video_attn_type='spatial_temporal'"),
    ],
)
def test_kimi_vision_rejects_unsupported_encoder_topology(field, value, match):
    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K2.5")["raw_config"])
    raw["vision_config"][field] = value
    with pytest.raises(ValueError, match=match):
        _parse_hf_config_json(raw)


def test_encoder_parallelism_models_required_communication():
    dp_model = get_model("moonshotai/Kimi-K2.5", _model_config(tp_size=2), "trtllm")
    tp_model = get_model(
        "moonshotai/Kimi-K2.5",
        _model_config(tp_size=2, enable_encoder_dp=False),
        "trtllm",
    )

    dp_names = {op._name for op in dp_model.encoder_ops}
    tp_ops = {op._name: op for op in tp_model.encoder_ops}
    assert "encoder_dp_all_gather" in dp_names
    assert tp_ops["encoder_ar_1"]._tp_size == 2
    assert tp_ops["encoder_ar_2"]._tp_size == 2
    assert "encoder_projector_ar" not in tp_ops
    assert (tp_ops["encoder_projector_fc0_gemm"]._n, tp_ops["encoder_projector_fc0_gemm"]._k) == (4608, 4608)
    assert (tp_ops["encoder_projector_fc1_gemm"]._n, tp_ops["encoder_projector_fc1_gemm"]._k) == (7168, 4608)


def test_kimi_video_temporal_pooling_keeps_context_tokens_spatial():
    enc = get_model_config_from_model_path("moonshotai/Kimi-K2.5")["encoder_config"]
    runtime = config.RuntimeConfig(
        video_height=448,
        video_width=448,
        video_frames=4,
        num_videos_per_request=1,
    )

    assert BaseBackend._encoder_pre_merge_per_visual(runtime, enc) == (256, 4096, 1)


@pytest.mark.parametrize("frames", [5, 8])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("language_only", [False, True])
def test_kimi_rejects_multiple_video_chunks_through_runtime(frames, override, language_only):
    model_cfg = _model_config()
    model_cfg.language_only = language_only
    model = get_model("moonshotai/Kimi-K2.5", model_cfg, "trtllm")
    runtime = config.RuntimeConfig(
        num_images_per_request=0,
        num_videos_per_request=1,
        video_frames=frames,
        video_height=0 if override else 448,
        video_width=0 if override else 448,
        num_video_tokens=100 if override else 0,
    )
    # This public runtime rejects before any database/engine evaluation, even
    # on a language-only worker where visual tokens still extend the context.
    with pytest.raises(ValueError, match="at most 4 sampled frames"):
        TRTLLMBackend().run_static(model, None, runtime, mode="static_ctx")


@pytest.mark.parametrize(
    "height,width,video,expected",
    [
        (448, 449, False, (272, 1088, 1)),
        (449, 448, False, (272, 1088, 1)),
        (1, 1, False, (1, 4, 1)),
        (4000, 4000, False, (4225, 16900, 1)),
        (448, 449, True, (272, 4352, 1)),
        (4000, 4000, True, (1089, 17424, 1)),
        (28, 20000, False, (256, 1024, 1)),
    ],
)
def test_kimi_resize_matches_pinned_navit_processor(height, width, video, expected):
    # Oracle: navit_resize at Transformers cbc1651a, image/video processor
    # sources cited in THIRD_PARTY_NOTICES.md. Budgets apply before padding,
    # so a 4000-square image pads to 1820-square and exceeds 16384 patches.
    enc = get_model_config_from_model_path("moonshotai/Kimi-K2.5")["encoder_config"]
    if video:
        runtime = config.RuntimeConfig(
            num_images_per_request=0,
            num_videos_per_request=1,
            video_frames=4,
            video_height=height,
            video_width=width,
        )
    else:
        runtime = config.RuntimeConfig(image_height=height, image_width=width)
    assert BaseBackend._encoder_pre_merge_per_visual(runtime, enc) == expected


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("layout", ["native", "legacy-top-level", "legacy-nested"])
def test_checkpoint_processor_limits_reach_runtime(tmp_path, monkeypatch, remote, layout):
    from aisimulate_core.sdk import utils as utils_module

    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K2.5")["raw_config"])
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
        path = f"synthetic-kimi-k25/processor-{tmp_path.name}"
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
    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K2.5")["raw_config"])
    raw["preprocessor_config"] = processor
    enc = _parse_hf_config_json(raw)["encoder_config"]
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
    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K2.5")["raw_config"])
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
    from aisimulate_core.sdk import utils as utils_module

    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K2.5")["raw_config"])
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


@pytest.mark.parametrize("bad", [0, -1, True, None, "1024"])
def test_kimi_processor_rejects_invalid_patch_budgets(bad):
    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K2.5")["raw_config"])
    raw["preprocessor_config"] = {"max_patches": bad}
    with pytest.raises(ValueError, match="max_patches must be a positive integer"):
        _parse_hf_config_json(raw)


def test_pooled_video_token_override_reaches_encoder_shapes(monkeypatch):
    model = get_model("moonshotai/Kimi-K2.5", _model_config(tp_size=2), "trtllm")
    runtime = config.RuntimeConfig(
        batch_size=3,
        isl=128,
        osl=1,
        num_images_per_request=0,
        num_video_tokens=100,
        video_frames=4,
        num_videos_per_request=1,
    )
    backend = BaseBackend()
    assert BaseBackend._encoder_pre_merge_per_visual(runtime, model.encoder_config) == (100, 1600, 1)
    assert BaseBackend._visual_context_tokens(model, runtime) == 100
    captured = {}

    def evaluate(model_arg, database, shape_of, *, include_energy):
        captured.update({op._name: shape_of(op) for op in model_arg.encoder_ops})
        return {"encoder_attention": 1.0}, {"encoder_attention": 2.0}, {"encoder_attention": "silicon"}

    monkeypatch.setattr(backend, "_require_rust_engine_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "_run_encoder_phase_with_rust", evaluate)
    latency, energy, source, tokens = backend._run_encoder_phase(model, object(), runtime, batch_size=3)
    assert captured["encoder_attention"] == (2, 1600)
    # Projector keeps two independent pooled videos; GEMM later flattens b*s.
    assert captured["encoder_projector_fc1_gemm"] == (2, 100)
    assert tokens == 100
    assert latency and energy and source


@pytest.mark.parametrize("model_id", _KIMI_MODELS)
def test_kimi_image_and_video_runtime_cover_latency_memory_energy_and_ttft(model_id, monkeypatch):
    model = get_model(model_id, _model_config(), "trtllm")
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

    def _stub_encoder_phase(model_arg, _database, shape_of, *, include_energy):
        attention = next(op for op in model_arg.encoder_ops if op._name == "encoder_attention")
        attention_shape = shape_of(attention)
        attention_shapes.append(attention_shape)
        latency = attention_shape[0] * attention_shape[1] / 1_000.0
        return (
            {"encoder_attention": latency},
            {"encoder_attention": latency * 2 if include_energy else 0.0},
            {"encoder_attention": "structural-test"},
        )

    monkeypatch.setattr(backend, "_run_encoder_phase_with_rust", _stub_encoder_phase)

    image_runtime = config.RuntimeConfig(
        batch_size=2,
        isl=128,
        osl=1,
        image_height=448,
        image_width=448,
        engine_step_backend="rust",
    )
    video_runtime = config.RuntimeConfig(
        batch_size=2,
        isl=128,
        osl=1,
        num_images_per_request=0,
        video_height=448,
        video_width=448,
        video_frames=4,
        num_videos_per_request=1,
        engine_step_backend="rust",
    )

    enc = model.encoder_config
    assert BaseBackend._encoder_pre_merge_per_visual(image_runtime, enc) == (256, 1024, 1)
    assert BaseBackend._encoder_pre_merge_per_visual(video_runtime, enc) == (256, 4096, 1)
    assert BaseBackend._visual_context_tokens(model, image_runtime) == 256
    assert BaseBackend._visual_context_tokens(model, video_runtime) == 256

    image_summary = backend.run_static(model, database, image_runtime, mode="static_ctx")
    video_summary = backend.run_static(model, database, video_runtime, mode="static_ctx")

    assert attention_shapes == [(2, 1024), (2, 4096)]
    assert sum(image_summary.get_encoder_latency_dict().values()) > 0
    assert sum(image_summary.get_encoder_energy_wms_dict().values()) > 0
    assert sum(video_summary.get_encoder_latency_dict().values()) > sum(
        image_summary.get_encoder_latency_dict().values()
    )
    assert sum(video_summary.get_encoder_energy_wms_dict().values()) > sum(
        image_summary.get_encoder_energy_wms_dict().values()
    )
    assert video_summary.get_encoder_memory()["activations"] > image_summary.get_encoder_memory()["activations"]
    assert video_summary.get_result_dict()["ttft"] > image_summary.get_result_dict()["ttft"]


@pytest.mark.parametrize("video", [False, True])
def test_kimi_rust_runtime_preserves_replicated_projector_work_and_weights(video):
    database = get_database_view("b200_sxm", "trtllm", "current", database_mode="SOL", allow_missing_data=True)
    backend = TRTLLMBackend()
    summaries = []
    for encoder_dp, batch in ((True, 4), (False, 2)):
        model_cfg = _model_config(tp_size=2, enable_encoder_dp=encoder_dp)
        # Keep the published encoder intact; one LM layer makes this runtime
        # regression fit a small worker without changing vision computation.
        model_cfg.overwrite_num_layers = 1
        model = get_model("moonshotai/Kimi-K2.5", model_cfg, "trtllm")
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
        summaries.append(summary)
        assert BaseBackend._visual_context_tokens(model, runtime) == 272
        latency = summary.get_encoder_latency_dict()
        assert latency["encoder_attention"] > 0
        assert "encoder_projector_ar" not in latency
        assert summary.get_result_dict()["ttft"] > sum(latency.values())
        # Independent BF16 parameter count from published topology: input
        # projection + 27 attention/FFN blocks + two replicated projections.
        tower_tp = 1 if encoder_dp else 2
        expected_weights = 2 * (
            1152 * 3 * 14**2 + 27 * (4 * 1152**2 + 2 * 1152 * 4304) // tower_tp + 4608**2 + 4608 * 7168
        )
        assert summary.get_encoder_memory()["weights"] == pytest.approx(expected_weights / (1 << 30))
        # Explicit processor-output token counts reproduce the same actual
        # Rust operation queries as the corresponding unaligned dimensions.
        override = deepcopy(runtime)
        override.image_height = override.image_width = override.video_height = override.video_width = 0
        if video:
            override.num_video_tokens = 272
        else:
            override.num_image_tokens = 272
        override_summary = backend.run_static(model, database, override, mode="static_ctx")
        assert override_summary.get_encoder_latency_dict() == summary.get_encoder_latency_dict()
        assert override_summary.get_encoder_memory() == summary.get_encoder_memory()
    # Both schedules give two images per rank: replicated GEMMs must perform
    # identical work, although the ViT is sharded only in the second run.
    for name in ("encoder_projector_fc0_gemm", "encoder_projector_fc1_gemm"):
        assert summaries[0].get_encoder_latency_dict()[name] == pytest.approx(
            summaries[1].get_encoder_latency_dict()[name]
        )
    assert "encoder_dp_all_gather" in summaries[0].get_encoder_latency_dict()
    assert summaries[1].get_encoder_latency_dict()["encoder_ar_1"] > 0
