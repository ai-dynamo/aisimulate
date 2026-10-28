# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from importlib.resources import files

import pytest

from aisimulate_core.sdk import common
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.models.nemotron_h import _projection_modes

pytestmark = pytest.mark.unit

FP8 = common.GEMMQuantMode.fp8_static
NVFP4 = common.GEMMQuantMode.nvfp4
BF16 = common.GEMMQuantMode.bfloat16
EXCLUDED_LATENT = ["*.fc1_latent_proj", "*.fc2_latent_proj"]


def group_metadata():
    return {
        "quant_method": "modelopt_mixed",
        "quant_algo": "MIXED_PRECISION",
        "ignore": EXCLUDED_LATENT,
        "config_groups": {
            "dense": {"weights": {"num_bits": 8, "type": "float"}, "targets": ["*.in_proj", "*.out_proj"]},
            "experts": {
                "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                "targets": ["*.shared_experts.up_proj", "*.shared_experts.down_proj", "*.experts.*"],
            },
        },
    }


def test_group_targets_resolve_shared_nvfp4_and_excluded_latent():
    modes = _projection_modes({"quantization_config": group_metadata()}, "MEME*", FP8)
    assert modes == {
        "shared_experts.up_proj": NVFP4,
        "shared_experts.down_proj": NVFP4,
        "fc1_latent_proj": BF16,
        "fc2_latent_proj": BF16,
    }


def test_linear_class_target_respects_exclusions():
    raw = {
        "quantization_config": {
            "config_groups": {"all": {"weights": {"num_bits": 4, "type": "float"}, "targets": ["Linear"]}},
            "ignore": EXCLUDED_LATENT,
        }
    }
    modes = _projection_modes(raw, "E", FP8)
    assert (modes["shared_experts.up_proj"], modes["fc1_latent_proj"]) == (NVFP4, BF16)


def test_exact_group_target_is_not_a_full_listing():
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
    modes = _projection_modes(raw, "E", FP8)
    assert (modes["shared_experts.up_proj"], modes["shared_experts.down_proj"]) == (NVFP4, FP8)


def test_full_listing_beats_groups_and_makes_unlisted_projections_bf16():
    raw = {
        "quantization_config": {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {"model.layers.0.mixer.shared_experts.up_proj": {"quant_algo": "NVFP4"}},
            "config_groups": {"all": {"weights": {"num_bits": 8, "type": "float"}, "targets": ["Linear"]}},
        }
    }
    modes = _projection_modes(raw, "E", FP8)
    assert (modes["shared_experts.up_proj"], modes["shared_experts.down_proj"]) == (NVFP4, BF16)


@pytest.mark.parametrize("prefix", ["model", "backbone", "language_model.backbone", "language_model.model"])
@pytest.mark.parametrize("key", ["quant_algo", "quantization_algo", "quant_method"])
def test_checkpoint_path_aliases_and_algo_keys(prefix, key):
    layers = {f"{prefix}.layers.0.mixer.shared_experts.{p}": {key: "NVFP4"} for p in ("up_proj", "down_proj")}
    raw = {"hf_quant_config": {"quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": layers}}}
    modes = _projection_modes(raw, "E", FP8)
    assert modes["shared_experts.up_proj"] == modes["shared_experts.down_proj"] == NVFP4
    assert modes["fc1_latent_proj"] == BF16


def test_unlisted_projection_keeps_the_global_mode_outside_the_mixed_precision_contract():
    # Without a MIXED_PRECISION marker the listing is not known to be complete: do not guess BF16.
    raw = {"quantization_config": {"quantized_layers": {"model.layers.0.mixer.shared_experts.up_proj": "NVFP4"}}}
    modes = _projection_modes(raw, "E", FP8)
    assert (modes["shared_experts.up_proj"], modes["shared_experts.down_proj"], modes["fc1_latent_proj"]) == (
        NVFP4,
        FP8,
        FP8,
    )


def test_exclusion_overrides_listed_layer():
    raw = {
        "hf_quant_config": {
            "quantization": {
                "quantized_layers": {
                    "backbone.layers.0.mixer.shared_experts.up_proj": "NVFP4",
                    "backbone.layers.0.mixer.fc1_latent_proj": "FP8",
                },
                "exclude_modules": ["backbone.layers.0.mixer.fc1_latent_proj"],
            }
        }
    }
    assert _projection_modes(raw, "E", FP8)["fc1_latent_proj"] == BF16


def test_literal_layer_names_use_indexed_lookup(monkeypatch):
    import aisimulate_core.sdk.models.nemotron_h as nemotron

    layers = {f"model.layers.0.mixer.experts.{i}.up_proj": {"quant_algo": "NVFP4"} for i in range(4096)}
    layers["model.layers.0.mixer.shared_experts.up_proj"] = {"quant_algo": "NVFP4"}
    monkeypatch.setattr(nemotron, "_projection_matches", lambda *a: pytest.fail("pattern match on literal names"))
    modes = _projection_modes({"quantization_config": {"quantized_layers": layers}}, "E", FP8)
    assert modes["shared_experts.up_proj"] == NVFP4


def test_rejects_layer_dependent_precision():
    layers = {f"model.layers.{i}.mixer.shared_experts.up_proj": {"quant_algo": "FP8"} for i in range(3)}
    layers["model.layers.3.mixer.shared_experts.up_proj"] = {"quant_algo": "NVFP4"}
    with pytest.raises(ValueError, match="different precision across layers"):
        _projection_modes({"quantization_config": {"quantized_layers": layers}}, "EEEE", FP8)


def partial_listing_checkpoint(tmp_path):
    """Synthetic MIXED_PRECISION listing shaped like a partially listed public export (40 MoE layers).

    shared up: FP8 x39, unlisted x1. shared down: FP8 x37, NVFP4 x1, unlisted x2.
    fc1: FP8 x3, unlisted x37. fc2: FP8 x1, unlisted x39. No exclusions.
    """
    path = files("aisimulate_core") / "model_configs/nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4_config.json"
    data = json.loads(path.read_text())
    e_layers = [i for i, kind in enumerate(data["hybrid_override_pattern"]) if kind == "E"]
    assert len(e_layers) == 40
    listed = {}
    for n, layer in enumerate(e_layers):
        base = f"backbone.layers.{layer}.mixer"
        if n < 39:
            listed[f"{base}.shared_experts.up_proj"] = {"quant_algo": "FP8"}
        if n < 37:
            listed[f"{base}.shared_experts.down_proj"] = {"quant_algo": "FP8"}
        elif n == 37:
            listed[f"{base}.shared_experts.down_proj"] = {"quant_algo": "NVFP4"}
        if n < 3:
            listed[f"{base}.fc1_latent_proj"] = {"quant_algo": "FP8"}
        if n == 0:
            listed[f"{base}.fc2_latent_proj"] = {"quant_algo": "FP8"}
    data["quantization_config"] = {"quant_algo": "MIXED_PRECISION", "quantized_layers": listed}
    (tmp_path / "config.json").write_text(json.dumps(data))
    return str(tmp_path)


def test_partial_listing_is_rejected_with_an_actionable_error(tmp_path):
    path = partial_listing_checkpoint(tmp_path)
    with pytest.raises(ValueError, match=r"shared_experts\.up_proj.*different precision.*gemm_quant_mode"):
        get_model(path, ModelConfig(tp_size=2, moe_tp_size=2), backend_name="vllm")


def test_explicit_gemm_mode_still_loads_a_partial_listing(tmp_path):
    path = partial_listing_checkpoint(tmp_path)
    model = get_model(path, ModelConfig(tp_size=2, moe_tp_size=2, gemm_quant_mode=FP8), backend_name="vllm")
    by_name = {op._name: op for op in model.context_ops}
    assert by_name["context_shared_up_gemm"]._quant_mode == FP8


def test_rejects_conflicting_or_unsupported_algorithms():
    path = "model.layers.0.mixer.shared_experts.up_proj"
    conflict = {
        "hf_quant_config": {"quantization": {"quantized_layers": {path: "NVFP4"}}},
        "quantization_config": {"quantized_layers": {path: "FP8"}},
    }
    with pytest.raises(ValueError, match="Unsupported mixed precision"):
        _projection_modes(conflict, "E", FP8)
    with pytest.raises(ValueError, match="Unsupported mixed precision"):
        _projection_modes({"quantization_config": {"quantized_layers": {path: "int4_awq"}}}, "E", FP8)


@pytest.fixture
def checkpoint(tmp_path):
    path = files("aisimulate_core") / "model_configs/nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4_config.json"
    data = json.loads(path.read_text())
    data["quantization_config"] = group_metadata()
    (tmp_path / "config.json").write_text(json.dumps(data))
    return str(tmp_path)


@pytest.mark.parametrize("explicit", [None, BF16, common.GEMMQuantMode.fp8])
def test_model_uses_checkpoint_projection_precision_and_preserves_override(checkpoint, explicit):
    model = get_model(checkpoint, ModelConfig(tp_size=2, moe_tp_size=2, gemm_quant_mode=explicit), backend_name="vllm")
    for pipeline in (model.context_ops, model.generation_ops):
        by_name = {op._name.split("_", 1)[1]: op for op in pipeline}
        shared = explicit or NVFP4
        latent = explicit or BF16
        assert by_name["shared_up_gemm"]._quant_mode == shared
        assert by_name["shared_down_gemm"]._quant_mode == shared
        assert by_name["fc1_latent_proj_gemm"]._quant_mode == latent
        assert by_name["fc2_latent_proj_gemm"]._quant_mode == latent
        # Dense GEMMs keep the global mode.
        assert by_name["mamba_in_proj_gemm"]._quant_mode == (explicit or FP8)


def test_non_latent_nemotron_keeps_existing_calibration_policy():
    model = get_model(
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", ModelConfig(moe_tp_size=1), backend_name="vllm"
    )
    assert not model._projection_quant_modes
    for op in model.context_ops + model.generation_ops:
        if "shared_" in op._name and op._name.endswith("_gemm"):
            assert op._quant_mode == model.config.gemm_quant_mode


@pytest.mark.parametrize(
    "name",
    ["nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"],
)
def test_bundled_nvfp4_metadata_has_no_per_layer_data_so_projections_use_the_global_mode(name):
    # The bundled copies carry only quant_algo: NVFP4; the resolver has nothing to refine.
    model = get_model(name, ModelConfig(tp_size=2, moe_tp_size=2), backend_name="vllm")
    assert model.config.gemm_quant_mode == NVFP4
    assert set(model._projection_quant_modes.values()) == {NVFP4}
    assert len(model._projection_quant_modes) == 4
    for op in model.context_ops + model.generation_ops:
        if op._name.endswith(("shared_up_gemm", "shared_down_gemm", "fc1_latent_proj_gemm", "fc2_latent_proj_gemm")):
            assert op._quant_mode == NVFP4


def _nvfp4_shared_checkpoint(tmp_path, *, mixed_dense_fp8):
    """Synthetic MIXED_PRECISION listing: NVFP4 shared experts, latent projections excluded."""
    path = files("aisimulate_core") / "model_configs/nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4_config.json"
    data = json.loads(path.read_text())
    pattern = data["hybrid_override_pattern"]
    layers = {}
    for index, kind in enumerate(pattern):
        base = f"backbone.layers.{index}.mixer"
        if kind == "E":
            layers[f"{base}.shared_experts.up_proj"] = {"quant_algo": "NVFP4"}
            layers[f"{base}.shared_experts.down_proj"] = {"quant_algo": "NVFP4"}
        elif mixed_dense_fp8 and kind == "M":
            layers[f"{base}.in_proj"] = {"quant_algo": "FP8"}
            layers[f"{base}.out_proj"] = {"quant_algo": "FP8"}
        elif mixed_dense_fp8 and kind == "*":
            layers.update(
                {f"{base}.{name}": {"quant_algo": "FP8"} for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
            )
    data["quantization_config"] = {
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": layers,
        "ignore": EXCLUDED_LATENT,
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    return str(tmp_path)


@pytest.mark.parametrize("mixed_dense_fp8", [False, True], ids=["uniform_nvfp4", "mixed_fp8_nvfp4"])
@pytest.mark.parametrize(
    ("system", "version", "native"),
    [("h200_sxm", "0.24.0", False), ("gb200", "0.31.0", True)],
    ids=["hopper", "blackwell"],
)
def test_task_path_resolves_nvfp4_projections_per_system(
    monkeypatch, tmp_path, system, version, native, mixed_dense_fp8
):
    from aisimulate.sdk.task_v2 import Task

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    task = Task(
        serving_mode="agg",
        model_path=_nvfp4_shared_checkpoint(tmp_path, mixed_dense_fp8=mixed_dense_fp8),
        system_name=system,
        backend_name="vllm",
        backend_version=version,
        database_mode="SOL",
    )
    model_config = task.build_model_config(role="agg", parallel=(1, 1, 1, 1, 1, 1))
    assert model_config._gemm_quant_mode_is_explicit is False
    model = get_model(task.model_path, model_config, "vllm")
    nvfp4 = NVFP4 if native else common.GEMMQuantMode.nvfp4_wo
    for pipeline in (model.context_ops, model.generation_ops):
        by_name = {op._name.split("_", 1)[1]: op._quant_mode for op in pipeline if hasattr(op, "_quant_mode")}
        assert by_name["shared_up_gemm"] == by_name["shared_down_gemm"] == nvfp4
        assert by_name["fc1_latent_proj_gemm"] == by_name["fc2_latent_proj_gemm"] == BF16
        if mixed_dense_fp8:
            assert by_name["mamba_in_proj_gemm"] == by_name["qkv_gemm"] == FP8
        else:
            assert by_name["mamba_in_proj_gemm"] == nvfp4


def test_explicit_gemm_mode_overrides_system_resolution(tmp_path):
    path = _nvfp4_shared_checkpoint(tmp_path, mixed_dense_fp8=True)
    config = ModelConfig(moe_tp_size=1, system="h200_sxm", gemm_quant_mode=common.GEMMQuantMode.fp8)
    model = get_model(path, config, "vllm")
    assert not model._projection_quant_modes
    assert {op._quant_mode for op in model.context_ops if op._name.endswith("shared_up_gemm")} == {
        common.GEMMQuantMode.fp8
    }
