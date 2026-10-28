# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from importlib.resources import files

import pytest

import aisimulate_core.sdk.operations as ops
from aisimulate_core.sdk import common
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.models.nemotron_h import _projection_modes

pytestmark = pytest.mark.unit

FP8 = common.GEMMQuantMode.fp8_static
NVFP4 = common.GEMMQuantMode.nvfp4
BF16 = common.GEMMQuantMode.bfloat16
EXCLUDED_LATENT = ["*.fc1_latent_proj", "*.fc2_latent_proj"]
PROJECTIONS = ("shared_up_gemm", "shared_down_gemm", "fc1_latent_proj_gemm", "fc2_latent_proj_gemm")


def _modes(raw, pattern, default, system=None):
    """One precision per projection: these fixtures give every MoE layer the same one."""
    buckets = _projection_modes(raw, pattern, default, system)
    assert all(len(layers) == 1 for layers in buckets.values()), buckets
    return {suffix: next(iter(layers)) for suffix, layers in buckets.items()}


def _projection_buckets(pipeline):
    """``{projection: {mode: scale factor}}`` over the ops that price each projection."""
    buckets = {}
    for op in pipeline:
        name = op._name.split("_", 1)[1]
        if name in PROJECTIONS:
            buckets.setdefault(name, {})[op._quant_mode] = op._scale_factor
    return buckets


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
    modes = _modes({"quantization_config": group_metadata()}, "MEME*", FP8)
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
    modes = _modes(raw, "E", FP8)
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
    modes = _modes(raw, "E", FP8)
    assert (modes["shared_experts.up_proj"], modes["shared_experts.down_proj"]) == (NVFP4, FP8)


def test_full_listing_beats_groups_and_makes_unlisted_projections_bf16():
    raw = {
        "quantization_config": {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {"model.layers.0.mixer.shared_experts.up_proj": {"quant_algo": "NVFP4"}},
            "config_groups": {"all": {"weights": {"num_bits": 8, "type": "float"}, "targets": ["Linear"]}},
        }
    }
    modes = _modes(raw, "E", FP8)
    assert (modes["shared_experts.up_proj"], modes["shared_experts.down_proj"]) == (NVFP4, BF16)


@pytest.mark.parametrize("prefix", ["model", "backbone", "language_model.backbone", "language_model.model"])
@pytest.mark.parametrize("key", ["quant_algo", "quantization_algo", "quant_method"])
def test_checkpoint_path_aliases_and_algo_keys(prefix, key):
    layers = {f"{prefix}.layers.0.mixer.shared_experts.{p}": {key: "NVFP4"} for p in ("up_proj", "down_proj")}
    raw = {"hf_quant_config": {"quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": layers}}}
    modes = _modes(raw, "E", FP8)
    assert modes["shared_experts.up_proj"] == modes["shared_experts.down_proj"] == NVFP4
    assert modes["fc1_latent_proj"] == BF16


def test_unlisted_projection_keeps_the_global_mode_outside_the_mixed_precision_contract():
    # Without a MIXED_PRECISION marker the listing is not known to be complete: do not guess BF16.
    raw = {"quantization_config": {"quantized_layers": {"model.layers.0.mixer.shared_experts.up_proj": "NVFP4"}}}
    modes = _modes(raw, "E", FP8)
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
    assert _modes(raw, "E", FP8)["fc1_latent_proj"] == BF16


def test_literal_layer_names_use_indexed_lookup(monkeypatch):
    import aisimulate_core.sdk.models.nemotron_h as nemotron

    layers = {f"model.layers.0.mixer.experts.{i}.up_proj": {"quant_algo": "NVFP4"} for i in range(4096)}
    layers["model.layers.0.mixer.shared_experts.up_proj"] = {"quant_algo": "NVFP4"}
    monkeypatch.setattr(nemotron, "_projection_matches", lambda *a: pytest.fail("pattern match on literal names"))
    modes = _modes({"quantization_config": {"quantized_layers": layers}}, "E", FP8)
    assert modes["shared_experts.up_proj"] == NVFP4


def test_layer_dependent_precision_counts_the_layers_at_each_precision():
    layers = {f"model.layers.{i}.mixer.shared_experts.up_proj": {"quant_algo": "FP8"} for i in range(3)}
    layers["model.layers.3.mixer.shared_experts.up_proj"] = {"quant_algo": "NVFP4"}
    buckets = _projection_modes({"quantization_config": {"quantized_layers": layers}}, "EEEE", FP8)
    assert buckets["shared_experts.up_proj"] == {FP8: 3, NVFP4: 1}


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


@pytest.mark.parametrize(("system", "nvfp4"), [("gb200", NVFP4), ("h200_sxm", common.GEMMQuantMode.nvfp4_wo)])
def test_partial_listing_prices_each_layer_at_its_own_precision(tmp_path, system, nvfp4):
    model = get_model(
        partial_listing_checkpoint(tmp_path), ModelConfig(tp_size=2, moe_tp_size=2, system=system), backend_name="vllm"
    )
    expected = {
        "shared_up_gemm": {FP8: 39, BF16: 1},
        "shared_down_gemm": {FP8: 37, nvfp4: 1, BF16: 2},
        "fc1_latent_proj_gemm": {FP8: 3, BF16: 37},
        "fc2_latent_proj_gemm": {FP8: 1, BF16: 39},
    }
    mtp = model._mtp_scale_factor
    for pipeline, scale in ((model.context_ops, 1), (model.generation_ops, mtp)):
        buckets = _projection_buckets(pipeline)
        assert buckets == {name: {m: n * scale for m, n in layers.items()} for name, layers in expected.items()}
        # Weight bytes equal a per-layer reference with one single-layer GEMM per MoE layer.
        for name, layers in expected.items():
            projection_ops = [op for op in pipeline if op._name.split("_", 1)[1] == name]
            n, k = projection_ops[0]._n, projection_ops[0]._k
            reference = sum(count * ops.GEMM(name, scale, n, k, mode).get_weights() for mode, count in layers.items())
            assert sum(op.get_weights() for op in projection_ops) == pytest.approx(reference)


def test_partial_listing_latency_sums_the_per_precision_layers(monkeypatch, tmp_path):
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    path = partial_listing_checkpoint(tmp_path)

    def latencies(gemm_quant_mode=None):
        config = ForwardPassPerfModelConfig(
            model=path,
            system="gb200",
            backend="vllm",
            backend_version="0.31.0",
            worker_type="decode",
            tp=1,
            moe_tp_size=1,
            moe_ep_size=1,
            database_mode="SOL",
            estimation_mode="op_level",
            **({"gemm_quant_mode": gemm_quant_mode} if gemm_quant_mode else {}),
        )
        model = RustForwardPassPerfModel.best_available(config)
        try:
            return {
                prefill: {
                    row["name"].split("_", 1)[1]: row["latency_ms"]
                    for row in model.static_phase_diagnostics(batch_size=1, context_length=1024, prefill=prefill)
                }
                for prefill in (True, False)
            }
        finally:
            model.close()

    mixed = latencies()
    uniform = {mode: latencies(mode.name) for mode in (FP8, NVFP4, BF16)}
    layers = {"shared_up_gemm": {FP8: 39, BF16: 1}, "shared_down_gemm": {FP8: 37, NVFP4: 1, BF16: 2}}
    for prefill in (True, False):
        for name, counts in layers.items():
            reference = sum(count / 40 * uniform[mode][prefill][name] for mode, count in counts.items())
            assert mixed[prefill][name] == pytest.approx(reference, rel=1e-9)


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
        assert sorted(
            op._name.split("_", 1)[1] for op in pipeline if op._name.split("_", 1)[1] in PROJECTIONS
        ) == sorted(PROJECTIONS)
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


@pytest.mark.parametrize(("system", "version"), [("gb200", "0.31.0"), ("h200_sxm", "0.24.0")])
@pytest.mark.parametrize(
    "name",
    ["nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"],
)
def test_packaged_latent_moe_configs_price_every_moe_layer(monkeypatch, name, system, version):
    from aisimulate.sdk.task_v2 import Task

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    task = Task(
        serving_mode="agg",
        model_path=name,
        system_name=system,
        backend_name="vllm",
        backend_version=version,
        database_mode="SOL",
    )
    model = get_model(name, task.build_model_config(role="agg", parallel=(1, 1, 1, 1, 1, 1)), "vllm")
    moe_layers = model._count_layer_types()["E"]
    for pipeline, scale in ((model.context_ops, 1), (model.generation_ops, model._mtp_scale_factor)):
        buckets = _projection_buckets(pipeline)
        assert set(buckets) == set(PROJECTIONS)
        for layers in buckets.values():
            assert sum(layers.values()) == pytest.approx(moe_layers * scale)
            if system == "h200_sxm":
                assert NVFP4 not in layers


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
