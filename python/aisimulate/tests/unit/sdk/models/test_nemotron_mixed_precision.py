# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
from importlib.resources import files

import pytest

from aisimulate_core.sdk import common
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.models.nemotron_h import _projection_modes
from aisimulate_core.sdk.speculation import SpeculationConfig

pytestmark = pytest.mark.unit


def quant_metadata():
    return {
        "quant_method": "modelopt_mixed",
        "ignore": ["*.fc1_latent_proj", "*.fc2_latent_proj"],
        "config_groups": {
            "dense": {"weights": {"num_bits": 8, "type": "float"}, "targets": ["*.in_proj", "*.out_proj"]},
            "experts": {
                "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                "targets": ["*.shared_experts.up_proj", "*.shared_experts.down_proj", "*.experts.*"],
            },
        },
    }


def test_resolves_shared_and_excluded_latent_projections():
    raw = {"quantization_config": quant_metadata()}
    modes = _projection_modes(raw, "MEME*", common.GEMMQuantMode.fp8_static)
    assert modes["shared_experts.up_proj"] == common.GEMMQuantMode.nvfp4
    assert modes["shared_experts.down_proj"] == common.GEMMQuantMode.nvfp4
    assert modes["fc1_latent_proj"] == common.GEMMQuantMode.bfloat16
    assert modes["fc2_latent_proj"] == common.GEMMQuantMode.bfloat16


def test_linear_class_target_respects_ignored_projections():
    raw = {
        "quantization_config": {
            "config_groups": {"all": {"weights": {"num_bits": 4, "type": "float"}, "targets": ["Linear"]}},
            "ignore": ["*.fc1_latent_proj", "*.fc2_latent_proj"],
        }
    }
    modes = _projection_modes(raw, "E", common.GEMMQuantMode.fp8_static)
    assert modes["shared_experts.up_proj"] == common.GEMMQuantMode.nvfp4
    assert modes["shared_experts.down_proj"] == common.GEMMQuantMode.nvfp4
    assert modes["fc1_latent_proj"] == common.GEMMQuantMode.bfloat16
    assert modes["fc2_latent_proj"] == common.GEMMQuantMode.bfloat16


def test_exact_group_target_does_not_imply_a_complete_layer_listing():
    raw = {
        "quantization_config": {
            "config_groups": {
                "one": {
                    "weights": {"num_bits": 4, "type": "float"},
                    "targets": ["model.layers.0.mixer.shared_experts.up_proj"],
                }
            }
        }
    }
    modes = _projection_modes(raw, "E", common.GEMMQuantMode.fp8_static)
    assert modes["shared_experts.up_proj"] == common.GEMMQuantMode.nvfp4
    assert modes["shared_experts.down_proj"] == common.GEMMQuantMode.fp8_static


def test_complete_layer_metadata_overrides_a_broad_group_declaration():
    raw = {
        "quantization_config": {
            "quantized_layers": {"model.layers.0.mixer.shared_experts.up_proj": {"quant_algo": "NVFP4"}},
            "config_groups": {"all": {"weights": {"num_bits": 8, "type": "float"}, "targets": ["Linear"]}},
        }
    }
    modes = _projection_modes(raw, "E", common.GEMMQuantMode.fp8_static)
    assert modes["shared_experts.up_proj"] == common.GEMMQuantMode.nvfp4
    assert modes["shared_experts.down_proj"] == common.GEMMQuantMode.bfloat16


@pytest.mark.parametrize("key", ["quant_algo", "quantization_algo", "quant_method"])
def test_projection_precision_accepts_modelopt_metadata_keys(key):
    raw = {
        "hf_quant_config": {
            "quantization": {"quantized_layers": {"model.layers.0.mixer.shared_experts.up_proj": {key: "NVFP4"}}}
        }
    }
    assert (
        _projection_modes(raw, "E", common.GEMMQuantMode.fp8_static)["shared_experts.up_proj"]
        == common.GEMMQuantMode.nvfp4
    )


def test_literal_layer_metadata_does_not_use_pattern_matching(monkeypatch):
    import aisimulate_core.sdk.models.nemotron_h as nemotron

    layers = {f"model.layers.0.mixer.experts.{i}.up_proj": {"quant_algo": "NVFP4"} for i in range(4096)}
    layers["model.layers.0.mixer.shared_experts.up_proj"] = {"quant_algo": "NVFP4"}

    def unexpected_pattern_match(*args):
        pytest.fail("Literal layer names must use indexed lookup")

    monkeypatch.setattr(nemotron, "_projection_matches", unexpected_pattern_match)
    modes = _projection_modes(
        {"quantization_config": {"quantized_layers": layers}}, "E", common.GEMMQuantMode.fp8_static
    )
    assert modes["shared_experts.up_proj"] == common.GEMMQuantMode.nvfp4


def test_native_backbone_names_and_unlisted_projections():
    raw = {
        "quantization_config": {
            "quantized_layers": {
                "backbone.layers.0.mixer.shared_experts.up_proj": {"quant_algo": "NVFP4"},
                "backbone.layers.0.mixer.shared_experts.down_proj": {"quant_algo": "NVFP4"},
            }
        }
    }
    modes = _projection_modes(raw, "E", common.GEMMQuantMode.fp8_static)
    assert modes["shared_experts.up_proj"] == common.GEMMQuantMode.nvfp4
    assert modes["fc1_latent_proj"] == common.GEMMQuantMode.bfloat16


def test_inactive_mamba_controls_preserve_configuration_identity():
    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    payload = ForwardPassPerfModelConfig(model="m", system="s", backend="vllm", worker_type="decode").to_dict()
    assert not any(key.startswith("mamba_") for key in payload)


def test_rejects_layer_dependent_precision_in_an_aggregated_operation():
    raw = {
        "quantization_config": {
            "quantized_layers": {
                "model.layers.0.mixer.shared_experts.up_proj": {"quant_algo": "NVFP4"},
                "model.layers.1.mixer.shared_experts.up_proj": {"quant_algo": "FP8"},
            }
        }
    }
    with pytest.raises(ValueError, match="different precision across layers"):
        _projection_modes(raw, "EE", common.GEMMQuantMode.fp8_static)


@pytest.fixture
def checkpoint(tmp_path):
    path = files("aisimulate_core") / "model_configs/nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4_config.json"
    data = json.loads(path.read_text())
    data["quantization_config"] = quant_metadata()
    (tmp_path / "config.json").write_text(json.dumps(data))
    return str(tmp_path)


@pytest.mark.parametrize("explicit", [None, common.GEMMQuantMode.bfloat16])
def test_model_uses_checkpoint_projection_precision_and_preserves_override(checkpoint, explicit):
    model = get_model(checkpoint, ModelConfig(tp_size=2, moe_tp_size=2, gemm_quant_mode=explicit), backend_name="vllm")
    for pipeline in (model.context_ops, model.generation_ops):
        by_name = {op._name.split("_", 1)[1]: op for op in pipeline}
        assert by_name["shared_up_gemm"]._quant_mode == (explicit or common.GEMMQuantMode.nvfp4)
        assert by_name["shared_down_gemm"]._quant_mode == (explicit or common.GEMMQuantMode.nvfp4)
        assert by_name["fc1_latent_proj_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
        assert by_name["fc2_latent_proj_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
        assert by_name["mamba_in_proj_gemm"]._quant_mode == (explicit or common.GEMMQuantMode.fp8_static)


@pytest.mark.parametrize("inferred_first", [True, False])
def test_engine_handle_cache_separates_inferred_and_explicit_projection_precision(
    checkpoint, monkeypatch, inferred_first
):
    from collections import OrderedDict
    from types import SimpleNamespace

    import aisimulate_core
    from aisimulate_core.sdk import engine as core_engine
    from aisimulate_core.sdk import rust_engine_step

    config = dict(tp_size=2, moe_tp_size=2)
    inferred = get_model(checkpoint, ModelConfig(**config), backend_name="vllm")
    explicit = get_model(
        checkpoint, ModelConfig(**config, gemm_quant_mode=common.GEMMQuantMode.fp8_static), backend_name="vllm"
    )
    # Same coarse mode, different shared-projection graph.
    assert inferred.config.gemm_quant_mode == explicit.config.gemm_quant_mode

    def shared_up(model):
        return next(op for op in model.context_ops if op._name.endswith("shared_up_gemm"))._quant_mode.name

    assert shared_up(inferred) != shared_up(explicit)
    # Real cache identity; only the compile tail is stubbed to record the graph.
    monkeypatch.setattr(rust_engine_step, "_ENGINE_HANDLE_CACHE", OrderedDict())
    monkeypatch.setattr(rust_engine_step, "_configure_default_data_roots", lambda: None)
    monkeypatch.setattr(core_engine, "build_engine_spec_json", lambda model, **kwargs: shared_up(model))
    monkeypatch.setattr(core_engine, "EngineHandle", lambda spec, systems_path=None: SimpleNamespace(spec=spec))
    monkeypatch.setattr(aisimulate_core, "engine_spec_bincode_from_json", lambda spec: spec.encode())
    database = SimpleNamespace(system="b200_sxm", backend="vllm", version="0.31.0")
    for model in (inferred, explicit) if inferred_first else (explicit, inferred):
        handle = rust_engine_step._cached_engine_handle(model, database)
        assert handle.spec == shared_up(model).encode()


def test_non_latent_nemotron_keeps_existing_calibration_policy():
    model = get_model(
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", ModelConfig(moe_tp_size=1), backend_name="vllm"
    )
    assert not model._projection_quant_modes
    for op in model.context_ops + model.generation_ops:
        if "shared_" in op._name and op._name.endswith("_gemm"):
            assert op._quant_mode == model.config.gemm_quant_mode


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_flashinfer_mamba_uses_rank_local_shapes_and_speculative_width(tp, depth):
    config = ModelConfig(
        tp_size=tp,
        moe_tp_size=tp,
        mamba_backend="flashinfer",
        mamba_ssm_cache_dtype="float16",
        mamba_stochastic_rounding=True,
        mamba_philox_rounds=5,
        speculation=SpeculationConfig(kind="mtp", params={"depth": depth}) if depth else None,
    )
    model = get_model("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", config, backend_name="vllm")
    op = next(op for op in model.generation_ops if op._name == "generation_mamba_ssm")
    # Pickle exercises the native constructor and all keyword-only identity fields.
    assert pickle.loads(pickle.dumps(op)).__getnewargs_ex__() == op.__getnewargs_ex__()
    args, kwargs = op.__getnewargs_ex__()
    assert args[2] == "vllm_flashinfer_selective_state_update"
    assert args[5] == 128 // tp
    assert args[9] == 8 // tp
    assert kwargs == dict(state_dtype="float16", stochastic_rounding=True, philox_rounds=5, token_width=depth + 1)


def test_mamba_controls_survive_canonical_compilation_and_saved_configuration(monkeypatch):
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    controls = dict(
        mamba_backend="flashinfer",
        mamba_ssm_cache_dtype="float16",
        mamba_stochastic_rounding=True,
        mamba_philox_rounds=5,
    )
    config = ForwardPassPerfModelConfig(
        model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        system="gb200",
        backend="vllm",
        backend_version="0.31.0",
        worker_type="decode",
        tp=2,
        moe_tp_size=2,
        moe_ep_size=1,
        nextn=3,
        database_mode="SOL",
        estimation_mode="op_level",
        **controls,
    )
    model = RustForwardPassPerfModel.best_available(config)
    try:
        saved = model.diagnostics()["provenance"]["config"]
        assert {key: saved[key] for key in controls} == controls
        restored = RustForwardPassPerfModel.best_available(json.loads(json.dumps(saved)))
        try:
            assert restored.diagnostics()["provenance"]["config"] == saved
            assert restored.static_phase_latency(batch_size=4, input_tokens=1024, output_tokens=8, prefill=False) > 0
        finally:
            restored.close()
    finally:
        model.close()


@pytest.mark.parametrize(
    "override", [{"backend": "sglang"}, {"estimation_mode": "fpm_regression"}, {"mamba_ssm_cache_dtype": "float32"}]
)
def test_canonical_mamba_controls_reject_incompatible_execution(override):
    import aisimulate_core
    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    values = dict(
        model="m",
        system="s",
        backend="vllm",
        worker_type="decode",
        estimation_mode="op_level",
        mamba_backend="flashinfer",
        mamba_ssm_cache_dtype="float16",
        mamba_stochastic_rounding=True,
    )
    with pytest.raises(ValueError, match="Mamba"):
        aisimulate_core.RustForwardPassPerfModel.normalize_config(
            json.dumps(ForwardPassPerfModelConfig(**(values | override)).to_dict())
        )


def test_mamba_controls_cannot_disappear_during_fallback(monkeypatch):
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    config = ForwardPassPerfModelConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        system="b200_sxm",
        backend="vllm",
        worker_type="decode",
        backend_version="0.24.0",
        estimation_mode="op_level",
        fallback_policy="allow",
        mamba_backend="flashinfer",
    )
    with pytest.raises(ValueError, match="Mamba"):
        RustForwardPassPerfModel.best_available(config)


def test_mamba_controls_reject_other_model_families():
    with pytest.raises(ValueError, match="Nemotron-H"):
        get_model("Qwen/Qwen3-30B-A3B", ModelConfig(mamba_backend="flashinfer"), backend_name="vllm")


def test_mamba_table_view_preserves_legacy_and_execution_keys():
    from aisimulate_core.sdk.engine_table_view import _mamba_tuple

    shape = "4096|128|64|128|8|4|128"
    assert _mamba_tuple(shape) == (4096, 128, 64, 128, 8, 4, 128)
    assert _mamba_tuple(shape + "|float16|1|5|4") == (4096, 128, 64, 128, 8, 4, 128, "float16", True, 5, 4)
    with pytest.raises(ValueError, match="Invalid Mamba timing key"):
        _mamba_tuple(shape + "|float16")
