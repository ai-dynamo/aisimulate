# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from importlib.resources import files

import pytest

import aisim_core.sdk.operations as ops
from aisim_core.sdk import common
from aisim_core.sdk.config import ModelConfig
from aisim_core.sdk.models import get_model
from aisim_core.sdk.models.checkpoint_precision import checkpoint_precision
from aisim_core.sdk.models.nemotron_h import _moe_modes, _projection_modes

pytestmark = pytest.mark.unit

FP8 = common.GEMMQuantMode.fp8_static
NVFP4 = common.GEMMQuantMode.nvfp4
NVFP4_WO = common.GEMMQuantMode.nvfp4_wo
BF16 = common.GEMMQuantMode.bfloat16
MOE = common.MoEQuantMode
SUPER = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
ULTRA = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
PROJECTIONS = (
    "mamba_in_proj_gemm",
    "mamba_out_proj_gemm",
    "qkv_gemm",
    "proj_gemm",
    "shared_up_gemm",
    "shared_down_gemm",
    "fc1_latent_proj_gemm",
    "fc2_latent_proj_gemm",
)
SHARED_UP = "model.layers.0.mixer.shared_experts.up_proj"


def _buckets(pipeline, names=PROJECTIONS, scale=1):
    """``{op: {mode: layers}}`` over the ops that price each projection (or ``moe``)."""
    buckets = {}
    for op in pipeline:
        name = op._name.split("_", 1)[1]
        if name in names:
            buckets.setdefault(name, {})[op._quant_mode] = op._scale_factor / scale
    return buckets


def _experts(model):
    return _buckets(model.context_ops, ("moe",)).get("moe", {})


def _groups(**groups):
    """A compressed-tensors ``config_groups`` section: ``name=(num_bits, targets)``."""
    return {
        name: {"weights": {"num_bits": bits, "type": "float"}, "targets": targets}
        for name, (bits, targets) in groups.items()
    }


def _listing(layers, **extra):
    return {"quantization_config": {"quant_algo": "MIXED_PRECISION", "quantized_layers": layers, **extra}}


def _expert_listing(algos_by_layer):
    return _listing(
        {
            f"backbone.layers.{layer}.mixer.experts.0.{proj}": {"quant_algo": algo}
            for layer, algo in algos_by_layer.items()
            if algo
            for proj in ("up_proj", "down_proj")
        }
    )


def _write_super(tmp_path, edit):
    """The published Super NVFP4 export, with ``edit`` applied to both quantization sections."""
    root = files("aisim_core") / "model_configs"
    data = json.loads((root / f"{SUPER.replace('/', '--')}_config.json").read_text())
    hf_quant = json.loads((root / f"{SUPER.replace('/', '--')}_hf_quant_config.json").read_text())
    e_layers = [i for i, kind in enumerate(data["hybrid_override_pattern"]) if kind == "E"]
    for section in (data["quantization_config"], hf_quant["quantization"]):
        edit(section, e_layers)
    (tmp_path / "config.json").write_text(json.dumps(data))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(hf_quant))
    return str(tmp_path)


def _super_with_fp8_experts(tmp_path, fp8_layers=10):
    """Published Super with the routed experts of its first ``fp8_layers`` MoE layers listed FP8."""

    def edit(section, e_layers):
        for name, entry in section["quantized_layers"].items():
            if any(f".layers.{layer}.mixer.experts." in name for layer in e_layers[:fp8_layers]):
                entry["quant_algo"] = "FP8"

    return _write_super(tmp_path, edit)


def _task(model_path, system="gb200", version="0.31.0", **kwargs):
    from aisim.sdk.task_v2 import Task

    return Task(
        serving_mode="agg",
        model_path=model_path,
        system_name=system,
        backend_name="vllm",
        backend_version=version,
        database_mode="SOL",
        **kwargs,
    )


def _task_model(task, model_path, role="agg"):
    return get_model(model_path, task.build_model_config(role=role, parallel=(1, 1, 1, 1, 1, 1)), "vllm")


def _latencies(model, system="gb200", **modes):
    """``{prefill: {op: latency_ms}}`` from the canonical API at decode batch 1 and context 1024."""
    from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    perf_model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=model,
            system=system,
            backend="vllm",
            backend_version="0.30.0",
            worker_type="decode",
            tp=1,
            moe_tp_size=1,
            moe_ep_size=1,
            database_mode="SOL",
            estimation_mode="op_level",
            **{key: mode.name for key, mode in modes.items()},
        )
    )
    try:
        return {
            prefill: {
                row["name"].split("_", 1)[1]: row["latency_ms"]
                for row in perf_model.static_phase_diagnostics(batch_size=1, context_length=1024, prefill=prefill)
            }
            for prefill in (True, False)
        }
    finally:
        perf_model.close()


@pytest.mark.parametrize(
    ("raw", "pattern", "expected"),
    [
        pytest.param(
            {
                "quantization_config": {
                    "config_groups": _groups(
                        dense=(8, ["*.in_proj", "*.out_proj"]),
                        experts=(4, ["*.shared_experts.up_proj", "*.shared_experts.down_proj"]),
                    ),
                    "ignore": ["*.fc1_latent_proj", "*.fc2_latent_proj"],
                }
            },
            "ME*",
            # Group targets are not a listing, so untargeted attention keeps the default.
            {"mamba_in_proj_gemm": FP8, "shared_up_gemm": NVFP4, "fc1_latent_proj_gemm": BF16, "qkv_gemm": FP8},
            id="group-targets-and-exclusions",
        ),
        pytest.param(
            {"quantization_config": {"config_groups": _groups(all=(4, ["Linear"])), "ignore": ["*.fc1_latent_proj"]}},
            "E",
            {"shared_up_gemm": NVFP4, "fc1_latent_proj_gemm": BF16},
            id="linear-target-respects-exclusions",
        ),
        pytest.param(
            {"quantization_config": {"config_groups": _groups(one=(4, [SHARED_UP]))}},
            "E",
            {"shared_up_gemm": NVFP4, "shared_down_gemm": FP8},
            id="exact-group-target-is-not-a-listing",
        ),
        pytest.param(
            _listing({SHARED_UP: {"quant_algo": "NVFP4"}}, config_groups=_groups(all=(8, ["Linear"]))),
            "E",
            {"shared_up_gemm": NVFP4, "shared_down_gemm": BF16},
            id="listing-beats-groups-and-unlisted-is-bf16",
        ),
        pytest.param(
            {"quantization_config": {"quantized_layers": {SHARED_UP: "NVFP4"}}},
            "E",
            # Without a MIXED_PRECISION marker the listing is not known to be complete: do not guess BF16.
            {"shared_up_gemm": NVFP4, "shared_down_gemm": FP8, "fc1_latent_proj_gemm": FP8},
            id="unlisted-keeps-default-outside-the-contract",
        ),
        pytest.param(
            {
                "hf_quant_config": {
                    "quantization": {
                        "quantized_layers": {"backbone.layers.0.mixer.fc1_latent_proj": "FP8"},
                        "exclude_modules": ["backbone.layers.0.mixer.fc1_latent_proj"],
                    }
                }
            },
            "E",
            {"fc1_latent_proj_gemm": BF16},
            id="exclusion-beats-listing",
        ),
        pytest.param(
            {
                "quantization_config": {
                    "config_groups": _groups(all=(8, ["Linear"])),
                    "ignore": ["backbone.layers.0.mixer"],
                }
            },
            "MM",
            {"mamba_in_proj_gemm": {BF16: 1, FP8: 1}},
            id="parent-exclusion-covers-its-projections",
        ),
        pytest.param(
            _listing(
                {
                    **{f"model.layers.{i}.mixer.shared_experts.up_proj": {"quant_algo": "FP8"} for i in range(3)},
                    "model.layers.3.mixer.shared_experts.up_proj": {"quant_algo": "NVFP4"},
                }
            ),
            "EEEE",
            {"shared_up_gemm": {FP8: 3, NVFP4: 1}},
            id="layers-are-counted-per-precision",
        ),
        pytest.param(
            _listing(
                {
                    # A listed descendant declares its module; an exact listing wins over descendants.
                    "backbone.layers.0.mixer.in_proj.weight": {"quant_algo": "FP8"},
                    "backbone.layers.0.mixer.out_proj": {"quant_algo": "FP8"},
                    "backbone.layers.0.mixer.out_proj.weight": {"quant_algo": "NVFP4"},
                }
            ),
            "M",
            {"mamba_in_proj_gemm": FP8, "mamba_out_proj_gemm": FP8},
            id="listed-descendants",
        ),
    ],
)
def test_projection_modes(raw, pattern, expected):
    buckets = _projection_modes(raw, pattern, FP8)
    got = {key: buckets[key] if isinstance(want, dict) else set(buckets[key]) for key, want in expected.items()}
    assert got == {key: want if isinstance(want, dict) else {want} for key, want in expected.items()}


@pytest.mark.parametrize("prefix", ["model", "backbone", "language_model.backbone", "language_model.model"])
@pytest.mark.parametrize("key", ["quant_algo", "quantization_algo", "quant_method"])
def test_checkpoint_path_aliases_and_algo_keys(prefix, key):
    raw = _listing({f"{prefix}.layers.0.mixer.shared_experts.up_proj": {key: "NVFP4"}})
    assert _projection_modes(raw, "E", FP8)["shared_up_gemm"] == {NVFP4: 1}


@pytest.mark.parametrize(
    ("raw", "pattern", "match"),
    [
        (
            {
                "hf_quant_config": {"quantization": {"quantized_layers": {SHARED_UP: "NVFP4"}}},
                "quantization_config": {"quantized_layers": {SHARED_UP: "FP8"}},
            },
            "E",
            "Conflicting checkpoint precisions",
        ),
        (
            _listing(
                {
                    "backbone.layers.0.mixer.in_proj.weight": {"quant_algo": "NVFP4"},
                    "backbone.layers.0.mixer.in_proj.bias": {"quant_algo": "FP8"},
                }
            ),
            "M",
            "Conflicting checkpoint precisions",
        ),
        ({"quantization_config": {"quantized_layers": {SHARED_UP: "int4_awq"}}}, "E", "Unsupported mixed precision"),
        (
            _listing({f"backbone.layers.0.mixer.{p}_proj": {"quant_algo": "FP8"} for p in "qk"}),
            "*",
            "disagree on precision",
        ),
    ],
    ids=["two-sources", "descendants", "unsupported", "fused-qkv"],
)
def test_rejects_inconsistent_metadata(raw, pattern, match):
    with pytest.raises(ValueError, match=match):
        _projection_modes(raw, pattern, FP8)


@pytest.mark.parametrize(
    ("raw", "default", "system", "expected"),
    [
        (
            _expert_listing({1: "NVFP4", 3: "FP8", 5: "NVFP4", 7: None}),
            MOE.nvfp4,
            "gb200",
            {MOE.nvfp4: 2, MOE.fp8: 1, MOE.bfloat16: 1},
        ),
        # A uniform listing keeps the global, system-resolved mode.
        (_expert_listing({1: "NVFP4", 3: "NVFP4", 5: "NVFP4", 7: "NVFP4"}), MOE.nvfp4, "gb200", {}),
        # Non-Blackwell NVFP4 runs weight-only, as the global mode does.
        (
            _expert_listing({1: "NVFP4", 3: "FP8", 5: "FP8", 7: "FP8"}),
            MOE.nvfp4_wo,
            "h200_sxm",
            {MOE.nvfp4_wo: 1, MOE.fp8: 3},
        ),
        # An untargeted layer keeps the global mode outside a targets-only export.
        (
            {"quantization_config": {"config_groups": _groups(a=(4, ["backbone.layers.1.mixer.experts.*"]))}},
            MOE.fp8,
            "gb200",
            {MOE.nvfp4: 1, MOE.fp8: 3},
        ),
        # An exclusion is explicit BF16, unlike an untargeted layer.
        (
            {
                "quantization_config": {
                    "config_groups": _groups(a=(8, ["Linear"])),
                    "ignore": ["backbone.layers.1.mixer.experts.*"],
                }
            },
            MOE.fp8,
            "gb200",
            {MOE.fp8: 3, MOE.bfloat16: 1},
        ),
    ],
)
def test_moe_modes(raw, default, system, expected):
    assert _moe_modes(raw, "MEMEMEME", default, system) == expected


def test_checkpoint_parse_is_indexed_and_cached(monkeypatch):
    import aisim_core.sdk.models.checkpoint_precision as precision

    layers = {f"model.layers.0.mixer.experts.{i}.up_proj": {"quant_algo": "NVFP4"} for i in range(4096)}
    raw = {"quantization_config": {"quantized_layers": {**layers, SHARED_UP: {"quant_algo": "NVFP4"}}}}
    # Literal names are dict lookups, not pattern matches.
    monkeypatch.setattr(precision, "_target_regex", lambda t: pytest.fail("pattern match on literals") if t else None)
    parsed = checkpoint_precision(raw)
    assert _projection_modes(raw, "E", FP8)["shared_up_gemm"] == {NVFP4: 1}
    assert checkpoint_precision(raw) is parsed
    # The cache keys on identity, not content, and keeps the latest 8 configs.
    assert checkpoint_precision(json.loads(json.dumps(raw))) is not parsed
    for _ in range(8):
        checkpoint_precision(_expert_listing({1: "FP8"}))
    assert checkpoint_precision(raw) is not parsed


_PUBLISHED = {
    SUPER: lambda nvfp4: {
        "mamba_in_proj_gemm": {FP8: 28, BF16: 12},
        "mamba_out_proj_gemm": {FP8: 29, BF16: 11},
        "qkv_gemm": {BF16: 8},
        "proj_gemm": {FP8: 2, BF16: 6},
        "shared_up_gemm": {FP8: 39, BF16: 1},
        "shared_down_gemm": {FP8: 37, nvfp4: 1, BF16: 2},
        "fc1_latent_proj_gemm": {FP8: 3, BF16: 37},
        "fc2_latent_proj_gemm": {FP8: 1, BF16: 39},
    },
    ULTRA: lambda nvfp4: {
        "mamba_in_proj_gemm": {FP8: 48},
        "mamba_out_proj_gemm": {FP8: 48},
        "qkv_gemm": {BF16: 12},
        "proj_gemm": {BF16: 12},
        "shared_up_gemm": {FP8: 48},
        "shared_down_gemm": {FP8: 48},
        "fc1_latent_proj_gemm": {BF16: 48},
        "fc2_latent_proj_gemm": {BF16: 48},
    },
}


@pytest.mark.parametrize("model_path", sorted(_PUBLISHED))
@pytest.mark.parametrize(("system", "nvfp4"), [("gb200", NVFP4), ("h200_sxm", NVFP4_WO)])
def test_published_exports_price_each_layer_at_its_listed_precision(model_path, system, nvfp4):
    model = get_model(model_path, ModelConfig(tp_size=2, moe_tp_size=2, system=system), backend_name="vllm")
    expected = _PUBLISHED[model_path](nvfp4)
    for pipeline, scale in ((model.context_ops, 1), (model.generation_ops, model._mtp_scale_factor)):
        assert _buckets(pipeline, scale=scale) == expected
        # Weight bytes equal a per-layer reference with one single-layer GEMM per layer.
        for name, layers in expected.items():
            gemms = [op for op in pipeline if op._name.split("_", 1)[1] == name]
            n, k = gemms[0]._n, gemms[0]._k
            reference = sum(count * ops.GEMM(name, scale, n, k, mode).get_weights() for mode, count in layers.items())
            assert sum(op.get_weights() for op in gemms) == pytest.approx(reference)
        # Every routed-expert layer is NVFP4: one MoE op on the system's lane, weight-only on Hopper.
        lane = MOE.nvfp4 if system == "gb200" else MOE.nvfp4_wo
        assert {op._quant_mode for op in pipeline if op._name.split("_", 1)[1] == "moe"} == {lane}
    # Explicit modes price every projection and expert with one mode.
    config = ModelConfig(tp_size=2, moe_tp_size=2, system=system, gemm_quant_mode=BF16, moe_quant_mode=MOE.fp8)
    explicit = get_model(model_path, config, backend_name="vllm")
    assert all(set(modes) == {BF16} for modes in _buckets(explicit.context_ops).values())
    assert set(_experts(explicit)) == {MOE.fp8}


@pytest.mark.parametrize(
    "model_path",
    [
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8",
        "nvidia/Nemotron-H-56B-Base-8K",
        # Lightning lists W4A16_NVFP4 layers, which no packaged table measures, so it keeps the global modes.
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    ],
)
def test_models_without_usable_per_layer_metadata_keep_the_global_modes(model_path):
    model = get_model(model_path, ModelConfig(moe_tp_size=1, system="gb200"), backend_name="vllm")
    assert not model._moe_quant_modes
    for name, modes in _buckets(model.context_ops).items():
        assert set(modes) == {model.config.gemm_quant_mode}, name


def test_mixed_precision_latency_sums_the_per_precision_layers(monkeypatch, tmp_path):
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    path = _super_with_fp8_experts(tmp_path)
    mixed = _latencies(path)
    projections = _PUBLISHED[SUPER](NVFP4)
    experts = {MOE.fp8: 10, MOE.nvfp4: 30}
    uniform = {mode: _latencies(path, gemm_quant_mode=mode) for mode in (FP8, NVFP4, BF16)}
    uniform |= {mode: _latencies(path, moe_quant_mode=mode) for mode in experts}
    for prefill in (True, False):
        for names, counts in [((name,), layers) for name, layers in projections.items()] + [
            (("moe", "moe_pre_dispatch", "moe_post_dispatch"), experts)
        ]:
            total = sum(counts.values())
            for name in names:
                reference = sum(count / total * uniform[mode][prefill][name] for mode, count in counts.items())
                assert mixed[prefill][name] == pytest.approx(reference, rel=1e-9), (prefill, name)


@pytest.mark.parametrize("system", ["h200_sxm", "gb200"])
@pytest.mark.parametrize("mixed", [False, True])
def test_entry_points_agree_on_per_layer_experts(monkeypatch, tmp_path, system, mixed):
    from aisim_core.sdk.models.helpers import resolve_nvfp4_for_system

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    path = _super_with_fp8_experts(tmp_path) if mixed else SUPER
    # The canonical API sets the system but leaves the inferred global MoE mode native.
    canonical = get_model(path, ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, system=system), "vllm")
    # The legacy CLI estimate path infers the global mode (nvfp4_wo on Hopper) before get_model.
    legacy_config = ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
    resolve_nvfp4_for_system(legacy_config, system, path, backend_name="vllm")
    legacy = get_model(path, legacy_config, "vllm")
    task = _task_model(_task(path, system=system, version="0.30.0"), path)
    nvfp4 = MOE.nvfp4_wo if system == "h200_sxm" else MOE.nvfp4
    expected = {MOE.fp8: 10, nvfp4: 30} if mixed else {nvfp4: 40}
    for model in (canonical, legacy, task):
        assert _experts(model) == pytest.approx(expected)
        assert {op._quant_mode for op in model.context_ops if op._name == "context_moe_pre_dispatch"} == set(expected)
        assert _buckets(model.context_ops) == _buckets(canonical.context_ops)
    weights = [sum(op.get_weights() for op in m.context_ops if op._name == "context_moe") for m in (canonical, legacy)]
    assert weights[0] == weights[1]
    # The canonical API prices the per-layer lanes; native FP4 on Hopper would raise MissingSystemFlopsError.
    assert _latencies(path, system=system)


@pytest.mark.parametrize("serving_mode", ["agg", "disagg"])
@pytest.mark.parametrize("explicit", [False, True])
def test_task_yaml_round_trip_and_recommend_rebuild_keep_inferred_precision(
    monkeypatch, tmp_path, serving_mode, explicit
):
    import yaml

    from aisim.legacy_cli.api import _build_recommend_tasks
    from aisim.sdk.task_v2 import Task

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    path = _super_with_fp8_experts(tmp_path)
    overrides = {"gemm_quant_mode": BF16, "moe_quant_mode": MOE.fp8} if explicit else {}
    if serving_mode == "agg":
        roles = ["agg"]
        task = _task(path, **overrides)
    else:
        roles = ["prefill", "decode"]
        kwargs = {"total_gpus": 2}
        for role in roles:
            kwargs |= {f"{role}_model_path": path, f"{role}_system_name": "gb200", f"{role}_backend_name": "vllm"}
            kwargs |= {f"{role}_{key}": value for key, value in overrides.items()}
        task = Task(serving_mode="disagg", backend_version="0.31.0", database_mode="SOL", **kwargs)
    reloaded = Task.from_yaml(yaml.safe_load(task.to_yaml()))
    rebuilt = _build_recommend_tasks({serving_mode: task}, 8)[serving_mode]
    assert rebuilt.total_gpus == 8
    for role in roles:
        model = _task_model(task, path, role)
        before = _buckets(model.context_ops), _experts(model)
        for other in (reloaded, rebuilt):
            other_model = _task_model(other, path, role)
            assert (_buckets(other_model.context_ops), _experts(other_model)) == before
        if explicit:
            assert all(set(modes) == {BF16} for modes in before[0].values())
            assert set(before[1]) == {MOE.fp8}
        else:
            assert any(len(modes) > 1 for modes in before[0].values())
            assert set(before[1]) == {MOE.fp8, MOE.nvfp4}


def test_config_only_modelopt_mixed_listing_prices_unlisted_layers_bf16(monkeypatch, tmp_path):
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")

    def edit(section, e_layers):
        # One Mamba in_proj FP8 and one MoE layer's experts NVFP4; the rest unlisted.
        section.clear()
        section["quant_method"] = "modelopt_mixed"
        section["quant_algo"] = "MIXED_PRECISION"
        section["quantized_layers"] = {
            "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
            **{
                f"backbone.layers.{e_layers[0]}.mixer.experts.0.{p}_proj": {"quant_algo": "NVFP4"}
                for p in ("up", "down")
            },
        }

    path = _write_super(tmp_path, edit)
    (tmp_path / "hf_quant_config.json").unlink()  # self-contained config.json
    pattern = json.loads((tmp_path / "config.json").read_text())["hybrid_override_pattern"]
    m_layers, e_layers = pattern.count("M"), pattern.count("E")
    for model in (
        get_model(path, ModelConfig(tp_size=2, moe_tp_size=2, system="gb200"), "vllm"),
        _task_model(_task(path), path),
    ):
        buckets = _buckets(model.context_ops)
        assert buckets["mamba_in_proj_gemm"] == {FP8: 1, BF16: m_layers - 1}
        assert set(buckets["mamba_out_proj_gemm"]) == set(buckets["qkv_gemm"]) == {BF16}
        assert _experts(model) == pytest.approx({MOE.nvfp4: 1, MOE.bfloat16: e_layers - 1})


def test_canonical_api_resolves_nvfp4_lanes_under_the_selected_systems_root(monkeypatch, tmp_path):
    from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, engine

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    # A Blackwell alias that only the explicit systems root defines.
    packaged = files("aisim_core") / "systems"
    (tmp_path / "custom_bw.yaml").write_text((packaged / "gb200.yaml").read_text())
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "custom_bw").symlink_to(packaged / "data" / "gb200")
    built = []
    monkeypatch.setattr(engine, "get_model", lambda *args: built.append(get_model(*args)) or built[-1])
    RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=SUPER,
            system="custom_bw",
            systems_paths=(str(tmp_path),),
            backend="vllm",
            backend_version="0.30.0",
            worker_type="decode",
            tp=1,
            moe_tp_size=1,
            moe_ep_size=1,
            database_mode="SOL",
            estimation_mode="op_level",
        )
    ).close()
    (model,) = built
    assert _buckets(model.context_ops)["shared_down_gemm"] == _PUBLISHED[SUPER](NVFP4)["shared_down_gemm"]
    assert set(_experts(model)) == {MOE.nvfp4}


def test_rebuilding_from_a_config_tracks_its_current_quant_modes(monkeypatch):
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    import dataclasses

    def build(config):
        model = get_model(SUPER, config, backend_name="vllm")
        return _buckets(model.context_ops), _experts(model), sum(op.get_weights() for op in model.context_ops)

    def assert_uniform(built):
        assert all(set(modes) == {BF16} for modes in built[0].values()) and set(built[1]) == {MOE.fp8}

    direct = ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, system="gb200")
    # A Task-built config carries explicit=False flags for its inferred modes.
    task = _task(SUPER).build_model_config(role="agg", parallel=(1, 1, 1, 1, 1, 1))
    for config in (direct, task):
        inferred = build(config)
        # Rebuilding the same config keeps the checkpoint split (Mamba in_proj: FP8 and BF16 layers).
        assert build(config) == inferred and len(inferred[0]["mamba_in_proj_gemm"]) == 2
        # Overrides on a copy, or assigned to the config itself, are explicit on every later build.
        assert_uniform(build(dataclasses.replace(config, gemm_quant_mode=BF16, moe_quant_mode=MOE.fp8)))
        config.gemm_quant_mode, config.moe_quant_mode = BF16, MOE.fp8
        for _ in range(3):
            assert_uniform(build(config))
        # Clearing the overrides restores inference.
        config.gemm_quant_mode = config.moe_quant_mode = None
        assert build(config) == inferred
        assert build(config) == inferred
