# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.resources as pkg_resources
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk import utils as utils_module
from aiconfigurator.sdk.backends import base_backend as base_backend_module
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.models import HybridMoEModel, get_model
from aiconfigurator.sdk.utils import _parse_hf_config_json, get_model_config_from_model_path

pytestmark = pytest.mark.unit

LLAMA4_CHECKPOINTS = (
    ("meta-llama/Llama-4-Scout-17B-16E-Instruct", 16, 48, 0),
    ("meta-llama/Llama-4-Maverick-17B-128E-Instruct", 128, 24, 24),
)
LLAMA4_MODEL_IDS = tuple(checkpoint[0] for checkpoint in LLAMA4_CHECKPOINTS)


@pytest.mark.parametrize("model_id", LLAMA4_MODEL_IDS)
@pytest.mark.parametrize("language_only", [False, True])
@pytest.mark.parametrize("source", ["local", "remote", "cached"])
@pytest.mark.parametrize("max_patches,expected_tokens", [(1, 147), (2, 437)])
def test_llama4_loads_separate_processor_metadata(
    tmp_path, monkeypatch, model_id, language_only, source, max_patches, expected_tokens
):
    raw = deepcopy(get_model_config_from_model_path(model_id)["raw_config"])
    raw.pop("image_processor_config")
    # Synthetic sidecar with the standard Transformers layout. The processor
    # supplies resize_to_max_canvas=False and the conditional global tile;
    # add_global_tile is not a serialized Transformers processor option.
    processor = {"image_processor_type": "Llama4ImageProcessorFast", "max_patches": max_patches}
    requested_files = []
    if source == "local":
        (tmp_path / "config.json").write_text(json.dumps(raw))
        (tmp_path / "preprocessor_config.json").write_text(json.dumps(processor))
        model_path = str(tmp_path)
    elif source == "cached":
        model_path = f"test/{tmp_path.name}"
        (tmp_path / f"{model_path.replace('/', '--')}_config.json").write_text(json.dumps(raw))
        (tmp_path / f"{model_path.replace('/', '--')}_preprocessor_config.json").write_text(json.dumps(processor))
        monkeypatch.setattr(utils_module, "DefaultHFModels", {model_path})
        monkeypatch.setattr(utils_module, "_get_model_config_path", lambda: tmp_path)
    else:
        model_path = f"test/{tmp_path.name}"

        def download(hf_id, filename, *, raise_on_404=True):
            assert hf_id == model_path
            requested_files.append(filename)
            return {"config.json": raw, "preprocessor_config.json": processor}.get(filename)

        monkeypatch.setattr(utils_module, "_download_hf_json", download)

    model = get_model(model_path, _model_config(language_only=language_only), "trtllm")
    runtime = RuntimeConfig(isl=128, osl=1, image_height=336, image_width=672)
    backend = BaseBackend()

    # One local tile gives 144 embeddings + 3 markers. Two local tiles also
    # get one global tile: 3 * 144 embeddings + 3 markers + 2 tile markers.
    assert backend._visual_context_tokens(model, runtime) == expected_tokens
    assert backend.effective_prefill_isl(model_path, runtime) == 128 + expected_tokens
    assert bool(model.encoder_ops) is (not language_only)
    assert model.context_ops and model.generation_ops
    assert model.encoder_config.max_num_tiles == max_patches
    # Both parsed and raw config caches must preserve the sidecar metadata.
    get_model_config_from_model_path.cache_clear()
    assert get_model_config_from_model_path(model_path)["extra_params"].vision_config == model.encoder_config
    if source == "remote":
        assert requested_files.count("preprocessor_config.json") == 1


@pytest.mark.parametrize(
    "processor,error,match",
    [
        ([], TypeError, "image_processor_config"),
        ({}, ValueError, "missing required fields"),
        ({"image_processor_type": "OtherProcessor"}, ValueError, "image_processor_type"),
        ({"image_processor_type": "Llama4ImageProcessorFast", "max_patches": True}, ValueError, "max_patches"),
        (
            {"image_processor_type": "Llama4ImageProcessorFast", "resize_to_max_canvas": "false"},
            TypeError,
            "resize_to_max_canvas",
        ),
        (
            {"image_processor_type": "Llama4ImageProcessorFast", "size": {"height": 224, "width": 224}},
            ValueError,
            "size",
        ),
    ],
)
def test_llama4_rejects_malformed_processor_sidecars(tmp_path, processor, error, match):
    raw = deepcopy(get_model_config_from_model_path(LLAMA4_MODEL_IDS[0])["raw_config"])
    raw.pop("image_processor_config")
    (tmp_path / "config.json").write_text(json.dumps(raw))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(processor))

    with pytest.raises(error, match=match):
        get_model(str(tmp_path), _model_config(), "trtllm")


def test_llama4_standard_processor_defaults(tmp_path):
    raw = deepcopy(get_model_config_from_model_path(LLAMA4_MODEL_IDS[0])["raw_config"])
    raw.pop("image_processor_config")
    (tmp_path / "config.json").write_text(json.dumps(raw))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"image_processor_type": "Llama4ImageProcessorFast"}))

    model = get_model(str(tmp_path), _model_config(), "trtllm")

    assert model.encoder_config.max_num_tiles == 16
    assert model.encoder_config.add_global_tile
    assert not model.encoder_config.resize_to_max_canvas
    assert BaseBackend._visual_context_tokens(model, RuntimeConfig(image_height=336, image_width=672)) == 437


def test_llama4_remote_processor_errors_are_not_silently_ignored(tmp_path, monkeypatch):
    raw = deepcopy(get_model_config_from_model_path(LLAMA4_MODEL_IDS[0])["raw_config"])
    raw.pop("image_processor_config")
    monkeypatch.setattr(utils_module, "_download_hf_config", lambda _: raw)

    def download(hf_id, filename, *, raise_on_404=True):
        assert filename == "preprocessor_config.json"
        raise utils_module.HuggingFaceDownloadError("processor access denied")

    monkeypatch.setattr(utils_module, "_download_hf_json", download)

    with pytest.raises(utils_module.HuggingFaceDownloadError, match="processor access denied"):
        get_model(f"test/{tmp_path.name}", _model_config(), "trtllm")


@pytest.mark.parametrize(
    "field",
    [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_channels",
        "intermediate_size",
        "image_size",
        "patch_size",
        "projector_input_dim",
        "projector_output_dim",
        "vision_output_dim",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "16", None])
def test_llama4_parser_rejects_invalid_integer_vision_fields(field, value):
    raw = deepcopy(get_model_config_from_model_path(LLAMA4_MODEL_IDS[0])["raw_config"])
    raw["vision_config"][field] = value
    with pytest.raises(ValueError, match=field + " must be a positive integer"):
        _parse_hf_config_json(raw)


def _model_config(**overrides):
    values = {
        "tp_size": 1,
        "pp_size": 1,
        "attention_dp_size": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 1,
    }
    values.update(overrides)
    return config.ModelConfig(**values)


def _stub_compiled_encoder(backend):
    shapes: dict[str, int] = {}
    backend._require_rust_engine_step = lambda *args, **kwargs: None

    def _run(model, _database, shape_of, *, include_energy):
        for op in model.encoder_ops:
            _batch, eff_s = shape_of(op)
            shapes[op._name] = eff_s
        return (
            {"encoder_attention": 1.0},
            {"encoder_attention": 2.0 if include_energy else 0.0},
            {"encoder_attention": "silicon"},
        )

    backend._run_encoder_phase_with_rust = _run
    return shapes


@pytest.mark.parametrize("model_id", LLAMA4_MODEL_IDS)
@pytest.mark.parametrize("language_only", [False, True])
def test_language_only_worker_retains_visual_tokens_without_hosting_encoder(model_id, language_only):
    model = get_model(model_id, _model_config(language_only=language_only), "sglang")
    runtime = RuntimeConfig(isl=128, osl=1, image_height=336, image_width=336)
    backend = BaseBackend()

    assert isinstance(model.encoder_config, common.VisionEncoderConfig)
    assert backend._visual_context_tokens(model, runtime) == 147
    assert bool(model.encoder_ops) is (not language_only)
    assert model.context_ops and model.generation_ops
    if language_only:
        assert backend._get_encoder_component_memory_for_runtime(model, runtime, 1) == {}
        latency, energy, sources, _ = backend._run_encoder_phase(model, object(), runtime, 1)
        assert not latency and not energy and not sources


@pytest.mark.parametrize("model_id,num_experts,moe_layers,dense_layers", LLAMA4_CHECKPOINTS)
def test_checkpoint_configs_preserve_text_and_exact_vision_shapes(model_id, num_experts, moe_layers, dense_layers):
    info = get_model_config_from_model_path(model_id)

    assert info["architecture"] == "Llama4ForConditionalGeneration"
    assert info["layers"] == 48
    assert info["hidden_size"] == 5120
    assert info["num_experts"] == num_experts

    hybrid = info["extra_params"]
    assert isinstance(hybrid, common.HybridMoEConfig)
    assert sum(hybrid.moe_layer_freq) == moe_layers
    assert hybrid.moe_layer_freq.count(0) == dense_layers
    # AIC-1740 adds the vision phase without changing the established Llama 4
    # hybrid-MoE text model (whose Rust parity goldens predate this feature).
    assert not hybrid.use_qk_norm

    vision = hybrid.vision_config
    assert isinstance(vision, common.VisionEncoderConfig)
    assert vision.depth == 34
    assert vision.hidden_size == 1408
    assert vision.num_heads == 16
    assert vision.intermediate_size == 5632
    assert vision.image_size == 336
    assert vision.patch_size == 14
    assert vision.spatial_merge_size == 2
    assert vision.out_hidden_size == 5120
    assert vision.projector_dims == ((5632, 4096), (4096, 4096), (4096, 5120))
    assert vision.in_channels == 3
    assert vision.has_cls_token
    assert vision.max_num_tiles == 16
    assert vision.add_global_tile
    assert vision.prompt_image_tokens == 3
    assert vision.prompt_tokens_per_local_tile == 1


@pytest.mark.parametrize("model_id", LLAMA4_MODEL_IDS)
def test_bundled_checkpoint_json_preserves_llama4_special_tokens_and_vision_metadata(model_id):
    config_path = (
        pkg_resources.files("aiconfigurator_core") / "model_configs" / (f"{model_id.replace('/', '--')}_config.json")
    )
    checkpoint = json.loads(config_path.read_text())

    assert checkpoint["boi_token_index"] == 200080
    assert checkpoint["eoi_token_index"] == 200081
    assert checkpoint["image_token_index"] == 200092
    assert checkpoint["image_processor_config"] == {
        "add_global_tile": True,
        "max_patches": 16,
        "resize_to_max_canvas": False,
    }
    assert checkpoint["vision_config"] == {
        "attention_dropout": 0.0,
        "hidden_act": "gelu",
        "hidden_size": 1408,
        "image_size": 336,
        "initializer_range": 0.02,
        "intermediate_size": 5632,
        "model_type": "llama4_vision_model",
        "multi_modal_projector_bias": False,
        "norm_eps": 1e-5,
        "num_channels": 3,
        "num_hidden_layers": 34,
        "num_attention_heads": 16,
        "patch_size": 14,
        "pixel_shuffle_ratio": 0.5,
        "projector_dropout": 0.0,
        "projector_input_dim": 4096,
        "projector_output_dim": 4096,
        "rope_theta": 10000,
        "torch_dtype": "bfloat16",
        "vision_feature_layer": -1,
        "vision_feature_select_strategy": "default",
        "vision_output_dim": 4096,
    }


@pytest.mark.parametrize("model_id", LLAMA4_MODEL_IDS)
def test_both_checkpoints_build_vision_and_hybrid_text_ops(model_id):
    model = get_model(model_id, _model_config(), "trtllm")

    assert isinstance(model, HybridMoEModel)
    assert model.encoder_ops
    assert model.context_ops
    assert model.generation_ops
    names = {op._name for op in model.encoder_ops}
    assert {
        "encoder_patch_embedding_gemm",
        "encoder_qkv_gemm",
        "encoder_attention",
        "encoder_ffn1_gemm",
        "encoder_ffn2_gemm",
        "encoder_projector_pixel_shuffle",
        "encoder_projector_adapter_fc0_gemm",
        "encoder_projector_adapter_fc1_gemm",
        "encoder_projector_adapter_ar",
        "encoder_projector_mm_gemm",
    } <= names


def test_scout_encoder_operation_shapes_match_engine_modules():
    model = get_model(LLAMA4_CHECKPOINTS[0][0], _model_config(), "trtllm")
    by_name = {op._name: op for op in model.encoder_ops}

    assert (by_name["encoder_patch_embedding_gemm"]._n, by_name["encoder_patch_embedding_gemm"]._k) == (
        1408,
        3 * 14 * 14,
    )
    assert (by_name["encoder_qkv_gemm"]._n, by_name["encoder_qkv_gemm"]._k) == (3 * 1408, 1408)
    assert by_name["encoder_qkv_gemm"]._scale_factor == 34
    assert by_name["encoder_attention"]._n == 16
    assert by_name["encoder_attention"]._head_size == 88
    assert by_name["encoder_attention"]._scale_factor == 34
    assert (by_name["encoder_ffn1_gemm"]._n, by_name["encoder_ffn1_gemm"]._k) == (5632, 1408)
    assert (by_name["encoder_projector_adapter_fc0_gemm"]._n, by_name["encoder_projector_adapter_fc0_gemm"]._k) == (
        4096,
        5632,
    )
    assert (by_name["encoder_projector_adapter_fc1_gemm"]._n, by_name["encoder_projector_adapter_fc1_gemm"]._k) == (
        4096,
        4096,
    )
    assert (by_name["encoder_projector_mm_gemm"]._n, by_name["encoder_projector_mm_gemm"]._k) == (5120, 4096)


def test_llama4_encoder_tp_models_both_engine_communication_boundaries():
    model = get_model(
        LLAMA4_CHECKPOINTS[0][0],
        _model_config(tp_size=8, moe_tp_size=1, moe_ep_size=8, enable_encoder_dp=False),
        "trtllm",
    )
    by_name = {op._name: op for op in model.encoder_ops}

    assert "encoder_patch_embedding_all_gather" in by_name
    assert by_name["encoder_projector_adapter_ar"]._tp_size == 8
    assert "encoder_projector_mm_all_gather" in by_name
    assert "encoder_dp_all_gather" not in by_name


def test_llama4_encoder_dp_models_exit_all_gather_only():
    model = get_model(
        LLAMA4_CHECKPOINTS[0][0],
        _model_config(tp_size=8, moe_tp_size=1, moe_ep_size=8, enable_encoder_dp=True),
        "trtllm",
    )
    names = {op._name for op in model.encoder_ops}

    assert "encoder_dp_all_gather" in names
    assert "encoder_patch_embedding_all_gather" not in names
    assert "encoder_projector_mm_all_gather" not in names


def test_single_tile_image_produces_nonzero_engine_and_text_tokens():
    enc_cfg = get_model_config_from_model_path(LLAMA4_CHECKPOINTS[0][0])["extra_params"].vision_config
    runtime_config = RuntimeConfig(image_height=336, image_width=336, num_images_per_request=1)
    workload = BaseBackend._encoder_workload_per_visual(runtime_config, enc_cfg)

    assert workload.patch_tokens_per_sequence == 576
    assert workload.transformer_tokens_per_sequence == 577
    assert workload.output_tokens_per_sequence == 144
    assert workload.output_tokens_per_image == 144
    assert workload.context_tokens_per_image == 147
    assert workload.sequences_per_image == 1
    assert BaseBackend._visual_context_tokens_from_encoder_config(enc_cfg, runtime_config) == 147


@pytest.mark.parametrize("encoder_dp", [False, True])
def test_llama4_parallel_operation_dimensions_reach_compiled_specs(encoder_dp):
    from aiconfigurator.sdk.engine import build_ops_json

    model = get_model(
        LLAMA4_MODEL_IDS[0],
        _model_config(
            tp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
            enable_encoder_dp=encoder_dp,
        ),
        "trtllm",
    )
    ops = {op._name: op for op in model.encoder_ops}
    tp = 1 if encoder_dp else 8
    assert (ops["encoder_patch_embedding_gemm"]._n, ops["encoder_patch_embedding_gemm"]._k) == (1408 // tp, 588)
    assert (ops["encoder_qkv_gemm"]._n, ops["encoder_qkv_gemm"]._k) == (4224 // tp, 1408)
    assert (ops["encoder_ffn1_gemm"]._n, ops["encoder_ffn1_gemm"]._k) == (5632 // tp, 1408)
    assert (ops["encoder_ffn2_gemm"]._n, ops["encoder_ffn2_gemm"]._k) == (1408, 5632 // tp)
    assert (ops["encoder_projector_adapter_fc0_gemm"]._n, ops["encoder_projector_adapter_fc0_gemm"]._k) == (
        4096 // tp,
        5632,
    )
    assert (ops["encoder_projector_adapter_fc1_gemm"]._n, ops["encoder_projector_adapter_fc1_gemm"]._k) == (
        4096,
        4096 // tp,
    )
    assert (ops["encoder_projector_mm_gemm"]._n, ops["encoder_projector_mm_gemm"]._k) == (5120 // tp, 4096)
    specs = {
        value["name"]: value for entry in json.loads(build_ops_json(model.encoder_ops)) for value in entry.values()
    }
    for name, width in [("encoder_ar_1", 1408), ("encoder_ar_2", 1408), ("encoder_projector_adapter_ar", 4096)]:
        assert specs[name]["hidden_size"] == width
        assert specs[name]["tp_size"] == tp
    payloads = (
        {"encoder_dp_all_gather": 5120 * 8}
        if encoder_dp
        else {
            "encoder_patch_embedding_all_gather": 1408,
            "encoder_projector_mm_all_gather": 5120,
        }
    )
    for name, width in payloads.items():
        assert ops[name]._num_elements_per_token == width
        assert ops[name]._num_gpus == 8


def test_llama4_encoder_dp_uses_busiest_rank_and_all_image_tiles(monkeypatch):
    model = get_model(
        LLAMA4_MODEL_IDS[0],
        _model_config(
            tp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
            enable_encoder_dp=True,
        ),
        "trtllm",
    )
    runtime = RuntimeConfig(batch_size=9, isl=128, osl=1, image_height=672, image_width=672)
    backend = BaseBackend()
    captured = {}

    def evaluate(model_arg, database, shape_of, *, include_energy):
        captured.update({op._name: shape_of(op) for op in model_arg.encoder_ops})
        return {"encoder_attention": 1.0}, {"encoder_attention": 2.0}, {"encoder_attention": "silicon"}

    monkeypatch.setattr(backend, "_require_rust_engine_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "_run_encoder_phase_with_rust", evaluate)
    latency, energy, source, tokens = backend._run_encoder_phase(model, object(), runtime, batch_size=9)
    # ceil(9 / 8) images on the busiest rank, each with four local tiles plus global.
    assert captured["encoder_patch_embedding_gemm"] == (10, 576)
    assert captured["encoder_attention"] == (10, 577)
    assert captured["encoder_projector_mm_gemm"] == (10, 144)
    assert captured["encoder_dp_all_gather"] == (10, 144)
    assert tokens == 727
    assert latency and energy and source


def test_four_local_tiles_add_engine_global_tile_and_720_text_tokens():
    enc_cfg = get_model_config_from_model_path(LLAMA4_CHECKPOINTS[0][0])["extra_params"].vision_config
    runtime_config = RuntimeConfig(image_height=672, image_width=672, num_images_per_request=1)
    workload = BaseBackend._encoder_workload_per_visual(runtime_config, enc_cfg)

    assert workload.sequences_per_image == 5
    assert workload.output_tokens_per_image == 5 * 144
    assert workload.context_tokens_per_image == 5 * 144 + 4 + 3
    assert BaseBackend._visual_context_tokens_from_encoder_config(enc_cfg, runtime_config) == 727


def test_token_only_override_preserves_non_chunk_aligned_sdk_workload():
    enc_cfg = get_model_config_from_model_path(LLAMA4_CHECKPOINTS[0][0])["extra_params"].vision_config
    runtime_config = RuntimeConfig(num_image_tokens=333, num_images_per_request=1)

    workload = BaseBackend._encoder_workload_per_visual(runtime_config, enc_cfg)

    assert workload.output_tokens_per_image == 333
    assert workload.context_tokens_per_image == 336
    assert workload.output_tokens_per_sequence == 333
    assert workload.patch_tokens_per_sequence == 1332
    assert workload.transformer_tokens_per_sequence == 1333
    assert workload.sequences_per_image == 1


@pytest.mark.parametrize("height,width", [(0, 336), (336, 0), (-1, 336)])
def test_tiled_sequence_count_rejects_nonpositive_image_geometry(height, width):
    enc_cfg = get_model_config_from_model_path(LLAMA4_CHECKPOINTS[0][0])["extra_params"].vision_config

    with pytest.raises(ValueError, match="requires positive geometry"):
        BaseBackend._tiled_encoder_sequence_count(height, width, enc_cfg)


def test_llama4_parser_rejects_missing_processor_metadata():
    model_id = LLAMA4_CHECKPOINTS[0][0]
    raw_config = dict(get_model_config_from_model_path(model_id)["raw_config"])
    raw_config.pop("image_processor_config")

    with pytest.raises(TypeError, match="must preserve image_processor_config metadata"):
        _parse_hf_config_json(raw_config)


def test_llama4_parser_rejects_missing_vision_metadata_schema_drift():
    model_id = LLAMA4_CHECKPOINTS[0][0]
    raw_config = dict(get_model_config_from_model_path(model_id)["raw_config"])
    raw_config.pop("vision_config")

    with pytest.raises(TypeError, match="must preserve vision_config metadata"):
        _parse_hf_config_json(raw_config)


@pytest.mark.parametrize("invalid_value", [None, [], "", 0])
def test_llama4_parser_rejects_nondict_vision_metadata(invalid_value):
    model_id = LLAMA4_CHECKPOINTS[0][0]
    raw_config = dict(get_model_config_from_model_path(model_id)["raw_config"])
    raw_config["vision_config"] = invalid_value

    with pytest.raises(TypeError, match="must preserve vision_config metadata"):
        _parse_hf_config_json(raw_config)


def test_llama4_parser_rejects_empty_vision_metadata():
    model_id = LLAMA4_CHECKPOINTS[0][0]
    raw_config = dict(get_model_config_from_model_path(model_id)["raw_config"])
    raw_config["vision_config"] = {}

    with pytest.raises(ValueError, match="vision_config is missing required fields"):
        _parse_hf_config_json(raw_config)


@pytest.mark.parametrize("model_id", LLAMA4_MODEL_IDS)
def test_nonzero_image_workload_reaches_encoder_and_text_context(model_id):
    model = get_model(model_id, _model_config(), "trtllm")
    database = SimpleNamespace(backend="trtllm", version="test", system="h200_sxm")
    runtime = RuntimeConfig(batch_size=1, isl=256, osl=16, image_height=336, image_width=336)
    backend = TRTLLMBackend()
    shapes = _stub_compiled_encoder(backend)

    latency, energy, source, image_tokens = backend._run_encoder_phase(model, database, runtime, 1)

    assert image_tokens == 147
    assert sum(latency.values()) > 0
    assert sum(energy.values()) > 0
    assert source
    assert shapes["encoder_patch_embedding_gemm"] == 576
    assert shapes["encoder_attention"] == 577
    assert shapes["encoder_projector_mm_gemm"] == 144


def test_static_ttft_memory_and_energy_include_llama4_encoder(monkeypatch):
    model = get_model(LLAMA4_CHECKPOINTS[0][0], _model_config(), "trtllm")
    database = SimpleNamespace(
        backend="trtllm",
        version="test",
        system="h200_sxm",
        system_spec={
            "gpu": {"mem_capacity": 16 * (1 << 40)},
            "misc": {"nccl_mem": {1: 0}, "other_mem": 0},
        },
    )
    runtime = RuntimeConfig(
        batch_size=1,
        isl=256,
        osl=16,
        image_height=336,
        image_width=336,
        engine_step_backend="rust",
    )
    context_isls = []

    def _static_breakdown(_model, _database, static_runtime, *_args, **_kwargs):
        context_isls.append(static_runtime.isl)
        return (
            {"context_attention": 1.0},
            {"generation_attention": 1.0},
            {"context_attention": 2.0},
            {"generation_attention": 2.0},
            {"context_attention": "silicon"},
            {"generation_attention": "silicon"},
            (),
        )

    monkeypatch.setattr(base_backend_module, "should_use_rust_engine_step", lambda *args, **kwargs: True)
    monkeypatch.setattr(base_backend_module, "estimate_static_latency_breakdown_with_rust", _static_breakdown)
    backend = TRTLLMBackend()
    _stub_compiled_encoder(backend)

    summary = backend.run_static(model, database, runtime, mode="static")
    row = summary.get_result_dict()

    assert row["encoder_latency"] > 0
    assert summary.get_encoder_power_avg() > 0
    assert row["encoder_memory"] > 0
    assert row["ttft"] == pytest.approx(row["encoder_latency"] + row["context_latency"])
    assert context_isls == [runtime.isl + 147]
