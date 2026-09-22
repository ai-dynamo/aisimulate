# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FPM profiles keep identity, memory admission and compilation graph-independent."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, engine, memory
from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from aisimulate_core.sdk.fpm_profile import FpmModelProfile, load_fpm_profile

pytestmark = pytest.mark.unit


@pytest.fixture
def profile_dict():
    return {
        "schema_version": 1,
        "model": "test/unknown-decoder",
        "model_revision": "profile-test-fixture-v1",
        "architecture": "UnregisteredDecoderForCausalLM",
        "context_length": 4096,
        "num_experts": 64,
        "provenance": "Synthetic metadata for SDK behavior tests; not silicon qualification.",
        "deployments": [
            {
                "system": "test_gpu",
                "backend": "vllm",
                "backend_version": "0.25.1",
                "tp": 2,
                "dp": 1,
                "moe_tp": 2,
                "moe_ep": 1,
                "gemm_quant_mode": "nvfp4",
                "moe_quant_mode": "nvfp4",
                "fmha_quant_mode": "fp8",
                "comm_quant_mode": "half",
                "kv_cache_dtype": "fp8",
                "resources": {
                    "weights_bytes": 100,
                    "activations_bytes": 20,
                    "runtime_overhead_bytes": 30,
                    "comm_overhead_bytes": 50,
                    "kv_bytes_per_token": 10,
                    "cache_layout": "linear",
                    "max_num_tokens": 8192,
                    "max_batch_size": 256,
                    "provenance": "Declared per-rank arithmetic fixture; excludes CUDA graphs.",
                },
            }
        ],
    }


def _fail_graph(*_args, **_kwargs):
    raise AssertionError("analytical model construction or timing database was accessed")


def _request(profile, method="auto", **overrides):
    profile = json.loads(profile) if isinstance(profile, str) else profile
    deployment = profile["deployments"][0] if profile else {}
    config = {
        "model": profile["model"] if profile else "test/unknown-decoder",
        "system": deployment.get("system", "test_gpu"),
        "backend": "vllm",
        "backend_version": deployment.get("backend_version", "0.25.1"),
        "worker_type": "aggregated",
        "tp": deployment.get("tp", 2),
        "attention_dp": deployment.get("dp", 1),
        "moe_tp_size": deployment.get("moe_tp", 2),
        "moe_ep_size": deployment.get("moe_ep", 1),
        "estimation_mode": "fpm_interpolation",
        "fallback_policy": "deny",
        "fpm_profile": profile,
        "estimator_config": {"fpm_interpolation": {"method": method}},
    }
    config.update(overrides)
    return config


def _normalize(config):
    return json.loads(engine.aisimulate_core.RustForwardPassPerfModel.normalize_config(json.dumps(config)))


@pytest.fixture
def direct_compile(monkeypatch):
    monkeypatch.setattr(engine, "get_model", _fail_graph)
    monkeypatch.setattr(engine, "build_model_config", _fail_graph)
    monkeypatch.setattr(engine, "_maybe_load_database", _fail_graph)
    monkeypatch.setattr(engine, "_literal_backend_version", lambda _s, _b, version, *_args: version)
    monkeypatch.setattr(engine.aisimulate_core, "engine_spec_bincode_from_json", lambda value: value.encode())

    def compile_profile(profile, **kwargs):
        config = _normalize(_request(profile, kwargs.pop("fpm_interpolation", "auto"), **kwargs))
        return json.loads(
            engine.compile_engine(
                config["model"],
                config["system"],
                config["backend"],
                config["backend_version"],
                tp_size=config["tp"],
                moe_tp_size=config["moe_tp_size"],
                moe_ep_size=config["moe_ep_size"],
                forward_model="fpm",
                fpm_profile=config["fpm_profile"],
                fpm_interpolation=config["estimator_config"]["fpm_interpolation"]["method"],
                **kwargs,
            )
        )

    return compile_profile


def test_unknown_profile_auto_compiles_without_operations(profile_dict, direct_compile):
    spec = direct_compile(profile_dict)
    profile = FpmModelProfile.model_validate(profile_dict)
    deployment = profile.deployments[0]
    assert json.loads(spec["engine"]["extra"]["fpm_profile"]) == profile.model_dump(mode="json")
    assert json.loads(spec["engine"]["extra"]["estimator_config"])["fpm_interpolation"]["method"] == "direct"
    assert spec["engine"]["activation_dtype"] == "fp8"
    for key, phase in (("context_ops", "prefill"), ("generation_ops", "decode")):
        assert len(spec[key]) == 1
        op = spec[key][0]["FpmForward"]
        assert op["phase"] == phase
        assert op["model_path"] == profile.model
        assert op["match_identity"] == deployment.match_identity()
        assert op["interpolation"] == "direct"
        assert op["sol_ops"] == []
        assert op["weight_bytes"] == 100


def test_direct_json_transport_preserves_all_identity_fields(profile_dict, direct_compile):
    profile_dict["deployments"][0].update(moe_backend="pinned_moe", attention_backend="pinned_attention")
    spec = direct_compile(json.dumps(profile_dict), fpm_interpolation="direct")
    assert spec["context_ops"][0]["FpmForward"]["match_identity"] == [
        "nvfp4",
        "nvfp4",
        "fp8",
        "half",
        "fp8",
        "2",
        "1",
        "1",
        "2",
        "1",
        "1",
        "pinned_moe",
        "pinned_attention",
        "False",
        "False",
    ]


@pytest.mark.parametrize("override", [{"fmha_quant_mode": "bfloat16"}, {"gemm_quant_mode": "fp8"}])
def test_precision_conflicts_fail_before_build(profile_dict, direct_compile, override):
    with pytest.raises(ValueError, match="FPM profile identity conflict"):
        direct_compile(profile_dict, **override)


@pytest.mark.parametrize("controls", [{"enable_eplb": True}, {"wideep_num_slots": 256}, {"moe_backend": "megamoe"}])
def test_profile_engine_controls_fail_before_model_lookup(profile_dict, direct_compile, profile_memory, controls):
    with pytest.raises(ValueError):
        _normalize(_request(profile_dict, "direct", estimation_mode="auto", **controls))
    with pytest.raises(ValueError, match="FPM profiles do not support"):
        engine.compile_engine(
            "test/unknown-decoder",
            "test_gpu",
            "vllm",
            "0.25.1",
            tp_size=2,
            moe_tp_size=2,
            forward_model="fpm",
            fpm_profile=profile_dict,
            fpm_interpolation="direct",
            **controls,
        )
    with pytest.raises(ValueError, match="FPM profiles do not support"):
        profile_memory(**controls)


@pytest.mark.parametrize("cp_size", [2, 0, True, 1.0, "1", None])
@pytest.mark.parametrize("with_profile", [False, True])
def test_sdk_entry_points_reject_unsupported_cp_before_construction(profile_dict, monkeypatch, cp_size, with_profile):
    monkeypatch.setattr(engine, "build_model_config", _fail_graph)
    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", _fail_graph)
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "from_model_path", _fail_graph)
    kwargs = {"cp_size": cp_size, "fpm_profile": profile_dict if with_profile else None}
    with pytest.raises(ValueError, match="cp_size must be the integer 1"):
        engine.compile_engine("test/unknown-decoder", "test_gpu", "vllm", "0.25.1", forward_model="fpm", **kwargs)
    with pytest.raises(ValueError, match="cp_size must be the integer 1"):
        memory.estimate_kv_cache(
            "test/unknown-decoder",
            "test_gpu",
            "vllm",
            "0.25.1",
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.8,
            allow_naive_fallback=True,
            **kwargs,
        )


@pytest.mark.parametrize("kwargs", [{"nextn": 1}, {"cp_size": 2}, {"database_mode": "SOL"}])
def test_direct_rejects_unsupported_execution(profile_dict, direct_compile, kwargs):
    with pytest.raises(ValueError):
        direct_compile(profile_dict, **kwargs)


def test_direct_requires_explicit_resources(direct_compile):
    with pytest.raises(ValueError, match="requires an fpm_profile"):
        direct_compile(None, fpm_interpolation="direct")


def test_explicit_sol_does_not_fall_back_on_unknown_architecture(profile_dict, direct_compile):
    with pytest.raises(ValueError, match="requires a registered analytical model class"):
        direct_compile(profile_dict, fpm_interpolation="sol")


def test_registered_auto_uses_registry_presence_without_constructing(profile_dict, monkeypatch):
    from aisimulate_core.sdk.models.base import _MODEL_REGISTRY

    profile = load_fpm_profile(profile_dict)
    monkeypatch.setitem(_MODEL_REGISTRY, profile.architecture, object())
    assert _normalize(_request(profile_dict))["estimator_config"]["fpm_interpolation"]["method"] == "sol"
    assert _normalize(_request(profile_dict, "direct"))["estimator_config"]["fpm_interpolation"]["method"] == "direct"
    assert _normalize(_request(None))["estimator_config"]["fpm_interpolation"]["method"] == "sol"


def test_registered_build_failure_is_not_direct_fallback(profile_dict, monkeypatch):
    from aisimulate_core.sdk.models.base import _MODEL_REGISTRY

    monkeypatch.setitem(_MODEL_REGISTRY, profile_dict["architecture"], object())
    monkeypatch.setattr(engine, "_literal_backend_version", lambda _s, _b, version, *_args: version)
    observed = []

    def broken_model(_path, config, _backend):
        observed.append(config.fmha_quant_mode.name)
        raise RuntimeError("registered model has an unrelated implementation error")

    monkeypatch.setattr(engine, "get_model", broken_model)
    monkeypatch.setattr(engine, "_direct_fpm_spec_json", _fail_graph)
    with pytest.raises(ValueError, match="unrelated implementation error"):
        RustForwardPassPerfModel.best_available(_request(profile_dict))
    assert observed == ["fp8"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(unrecognized_field=1),
        lambda p: p.update(model_revision="main"),
        lambda p: p.update(model_revision="unknown"),
        lambda p: p.update(context_length=True),
        lambda p: p.update(schema_version=True),
        lambda p: p.update(schema_version=1.0),
        lambda p: p.update(provenance="   "),
        lambda p: p.update(deployments=[]),
        lambda p: p["deployments"].append(copy.deepcopy(p["deployments"][0])),
        lambda p: p["deployments"][0].update(tp="2"),
        lambda p: p["deployments"][0].update(pp=2),
        lambda p: p["deployments"][0].update(cp=2),
        lambda p: p["deployments"][0].update(cp=True),
        lambda p: p["deployments"][0].update(pp=1.0),
        lambda p: p["deployments"][0].update(dp=2),
        lambda p: p["deployments"][0].update(backend_version="current"),
        lambda p: p["deployments"][0].update(fmha_quant_mode="bf16"),
        lambda p: p["deployments"][0].update(enable_wideep=True),
        lambda p: p["deployments"][0].update(enable_eplb=0),
        lambda p: p["deployments"][0]["resources"].update(cache_layout="sliding_window"),
        lambda p: p["deployments"][0]["resources"].update(kv_bytes_per_token=0),
        lambda p: p["deployments"][0]["resources"].update(weights_bytes=-1),
        lambda p: p["deployments"][0]["resources"].update(weights_bytes=1.5),
        lambda p: p["deployments"][0]["resources"].update(weights_bytes=2**53),
    ],
)
def test_profiles_reject_ambiguous_or_unsupported_metadata(profile_dict, mutation):
    mutation(profile_dict)
    with pytest.raises(ValidationError):
        load_fpm_profile(profile_dict)


def test_topologies_have_separate_resources_and_expert_admission(profile_dict):
    tep = profile_dict["deployments"][0]
    tep.update(tp=8, dp=1, moe_tp=1, moe_ep=8)
    dep = copy.deepcopy(tep)
    dep.update(tp=1, dp=8)
    dep["resources"]["weights_bytes"] = 180
    profile_dict["deployments"].append(dep)
    profile = load_fpm_profile(profile_dict)
    selected = profile.select(
        model=profile.model,
        system="test_gpu",
        backend="vllm",
        backend_version="0.25.1",
        tp_size=1,
        attention_dp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
    )
    assert selected.resources.weights_bytes == 180
    assert profile.deployments[0].resources.weights_bytes == 100
    with pytest.raises(ValueError, match="no matching FPM deployment"):
        profile.select(model=profile.model, system="test_gpu", backend="vllm", backend_version="0.25.1")
    with pytest.raises(ValueError, match="model identity mismatch"):
        profile.select(model="other/model", system="test_gpu", backend="vllm", backend_version="0.25.1")
    profile_dict["num_experts"] = 63
    with pytest.raises(ValidationError, match="divisible by moe_ep"):
        load_fpm_profile(profile_dict)


def test_dense_tp_does_not_require_expert_tensor_parallelism(profile_dict):
    profile_dict["num_experts"] = 0
    profile_dict["deployments"][0]["moe_tp"] = 1
    assert load_fpm_profile(profile_dict).deployments[0].parallel_tuple == (2, 1, 1, 1, 1, 1)


@pytest.mark.parametrize("kwargs", [{"moe_tp_size": 0}, {"moe_ep_size": False}, {"cp_size": True}])
def test_select_rejects_zero_or_boolean_dimensions(profile_dict, kwargs):
    profile = load_fpm_profile(profile_dict)
    with pytest.raises(ValueError, match="positive integers"):
        profile.select(model=profile.model, system="test_gpu", backend="vllm", backend_version="0.25.1", **kwargs)


@pytest.fixture
def profile_memory(profile_dict, monkeypatch):
    monkeypatch.setattr(memory, "get_model", _fail_graph)
    monkeypatch.setattr(memory, "build_model_config", _fail_graph)
    monkeypatch.setattr(memory.perf_database, "get_database", _fail_graph)
    monkeypatch.setattr(memory.perf_database, "resolve_query_version", _fail_graph)
    monkeypatch.setattr(memory.perf_database, "load_system_spec", lambda *_a, **_k: {"gpu": {"mem_capacity": 1000}})
    monkeypatch.setattr(engine, "_literal_backend_version", lambda _s, _b, version, *_args: version)

    def estimate(**overrides):
        args = dict(
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.8,
            tp_size=2,
            moe_tp_size=2,
            fpm_profile=profile_dict,
        )
        args.update(overrides)
        return memory.estimate_kv_cache("test/unknown-decoder", "test_gpu", "vllm", "0.25.1", **args)

    return estimate


def test_profile_memory_uses_declared_bounds_without_graph_or_timing_data(profile_memory):
    result = profile_memory(cuda_graph_reserved_bytes=20, tolerance_fraction=0.1)
    assert result["source"] == "profile"
    assert result["total_gpu_capacity_bytes"] == 1000
    assert result["total_kv_size_bytes"] == 580  # 1000 * .8 - 200 - 20
    assert result["kv_size_per_token_bytes"] == 10
    assert result["total_kv_size_tokens"] == 58
    assert result["tolerance_adjusted"]["total_kv_size_bytes"] == 522
    assert result["tolerance_adjusted"]["total_kv_size_tokens"] == 52
    assert result["memory_breakdown"]["cuda_graph_reserved_bytes"] == 20
    assert "Declared per-rank" in result["resource_provenance"]
    json.dumps(result)  # The result is replayable; it contains no callables.


def test_profile_memory_capacity_override_does_not_read_hardware(profile_memory, monkeypatch):
    monkeypatch.setattr(memory.perf_database, "load_system_spec", _fail_graph)
    result = profile_memory(gpu_memory_capacity_bytes_override=2000)
    assert result["total_kv_size_bytes"] == 1400


def test_profile_block_budget_preserves_resource_provenance(profile_dict, profile_memory):
    diagnostics = {}
    blocks = memory.estimate_num_gpu_blocks(
        "test/unknown-decoder",
        "test_gpu",
        "vllm",
        "0.25.1",
        scheduler_block_size=4,
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.8,
        tp_size=2,
        moe_tp_size=2,
        fpm_profile=profile_dict,
        diagnostics=diagnostics,
    )
    assert blocks == 15  # (1000 * .8 - 200) / 10 / 4
    assert diagnostics["source"] == "profile"
    assert diagnostics["resource_provenance"] == profile_dict["deployments"][0]["resources"]["provenance"]
    with pytest.raises(ValueError, match="positive integer"):
        memory.estimate_num_gpu_blocks(
            "test/unknown-decoder",
            "test_gpu",
            "vllm",
            "0.25.1",
            scheduler_block_size=4,
            max_num_tokens=True,
            max_batch_size=256,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.8,
            tp_size=2,
            moe_tp_size=2,
            fpm_profile=profile_dict,
        )


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"max_num_tokens": 8193}, "resource envelope exceeded"),
        ({"max_batch_size": 257}, "resource envelope exceeded"),
        ({"max_num_tokens": True}, "positive integer"),
        ({"fmha_quant_mode": "bfloat16"}, "identity conflict"),
        ({"attention_backend": "different"}, "identity conflict"),
        ({"cp_size": 2}, "cp_size must be the integer 1"),
        ({"nextn": 1}, "nextn must be 0"),
        ({"gpu_memory_capacity_bytes_override": 200}, "no KV budget"),
        ({"memory_fraction_kind": "of_free"}, "incompatible memory fraction"),
    ],
)
def test_profile_memory_fails_closed(profile_memory, kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        profile_memory(**kwargs)


@pytest.fixture
def native_profile_config(profile_dict, monkeypatch):
    # Use a bundled system/version for native engine construction. Resource and
    # precision variants below are transport fixtures, not measured timing cells.
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    profile_dict.update(model="MiniMaxAI/MiniMax-M2.7")
    profile_dict["deployments"][0].update(system="h200_sxm")
    monkeypatch.setattr(engine, "get_model", _fail_graph)
    specs = []
    encode = engine.aisimulate_core.engine_spec_bincode_from_json

    def record_spec(value):
        specs.append(json.loads(value))
        return encode(value)

    monkeypatch.setattr(engine.aisimulate_core, "engine_spec_bincode_from_json", record_spec)

    def compile_config():
        engine.compile_engine(
            profile_dict["model"],
            "h200_sxm",
            "vllm",
            "0.25.1",
            tp_size=2,
            moe_tp_size=2,
            forward_model="fpm",
            fpm_profile=profile_dict,
            fpm_interpolation="direct",
        )
        return specs[-1]["engine"]

    return compile_config, specs


@pytest.mark.parametrize(
    ("gemm_mode", "moe_mode"),
    [
        ("fp8_block", "nvfp4"),
        ("int8_wo", "nvfp4"),
        ("int4_wo", "nvfp4"),
        ("sq", "nvfp4"),
        ("fp8_ootb", "nvfp4"),
        ("nvfp4", "int4_wo"),
    ],
)
def test_legacy_migration_preserves_profile_precision(profile_dict, native_profile_config, gemm_mode, moe_mode):
    profile_dict["deployments"][0].update(gemm_quant_mode=gemm_mode, moe_quant_mode=moe_mode)
    compile_config, _ = native_profile_config
    canonical = ForwardPassPerfModelConfig.from_legacy_engine_config(compile_config(), "aggregated")
    resolved = _normalize(canonical.to_dict())
    assert resolved["gemm_quant_mode"] == gemm_mode
    assert resolved["moe_quant_mode"] == moe_mode
    assert resolved["estimator_config"]["fpm_interpolation"]["method"] == "direct"
    assert resolved["fpm_profile"] == load_fpm_profile(profile_dict).model_dump(mode="json")
    assert "extra" not in resolved


@pytest.mark.parametrize("field", ["weight_dtype", "moe_dtype", "activation_dtype", "kv_cache_dtype"])
def test_native_profile_rejects_different_wire_dtype(native_profile_config, field):
    compile_config, _ = native_profile_config
    config = compile_config()
    config[field] = "bfloat16"
    with pytest.raises(ValueError, match="FPM profile identity conflict"):
        ForwardPassPerfModelConfig.from_legacy_engine_config(config, "prefill")


@pytest.mark.parametrize("mode", ["auto", "fpm_interpolation", "fpm_regression"])
@pytest.mark.parametrize(
    "case", ["malformed_profile", "invalid_method", "missing_profile", "precision", "missing_version", "version_alias"]
)
def test_invalid_profile_controls_fail_before_any_fallback(profile_dict, monkeypatch, mode, case):
    monkeypatch.setattr(engine, "compile_engine", _fail_graph)
    config = _request(profile_dict, "direct", estimation_mode=mode, fallback_policy="allow")
    if case == "malformed_profile":
        config["fpm_profile"] = {}
    elif case == "invalid_method":
        config["estimator_config"]["fpm_interpolation"]["method"] = "dierct"
    elif case == "missing_profile":
        config["fpm_profile"] = None
    elif case == "precision":
        config["gemm_quant_mode"] = "bfloat16"
    elif case == "missing_version":
        config["backend_version"] = None
    else:
        config["backend_version"] = "current"
    with pytest.raises(ValueError):
        RustForwardPassPerfModel.best_available(config)


def test_legacy_selector_migrates_to_nested_control(native_profile_config):
    compile_config, _ = native_profile_config
    config = compile_config()
    config["extra"].pop("estimator_config")
    config["extra"]["fpm_interpolation"] = "direct"
    canonical = ForwardPassPerfModelConfig.from_legacy_engine_config(config, "prefill").to_dict()
    assert canonical["estimator_config"]["fpm_interpolation"]["method"] == "direct"
    assert "fpm_interpolation" not in canonical
    assert isinstance(canonical["fpm_profile"], dict)


def _write_external_profile_pair(profile, directory):
    """Generate routing/identity evidence; these 2/3 ms rows are not silicon data."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    deployment = load_fpm_profile(profile).deployments[0]
    identity = deployment.model_dump(mode="json", exclude={"resources"})
    rows = [
        {
            **identity,
            "cell_id": f"synthetic-{phase}-{kv}",
            "model_path": profile["model"],
            "weight_quantization": "synthetic",
            "workload_kind": phase,
            "partition_policy": "balanced_v1",
            "batch_size": 1,
            "total_prefill_tokens": 1 if phase == "prefill" else 0,
            "total_kv_read_tokens": kv,
            "latency_ms": latency,
            "kv_seed_regime": "real_kv",
        }
        for phase, kv, latency in [("prefill", 0, 2.0), ("decode", 1, 3.0), ("decode", 64, 3.0)]
    ]
    path = directory / "synthetic-profile.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "schema_name": "aic_fpm_forward_perf",
                "schema_version": 6,
                "coordinate_system": "iteration_totals_balanced_v1",
                "measurement_policy": "dynamo_native_single_sample_v1",
                "system": deployment.system,
                "backend": deployment.backend,
                "backend_version": deployment.backend_version,
                "row_count": len(rows),
                "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    )
    return path


@pytest.mark.parametrize(
    ("model", "system", "tp", "dp", "moe_tp", "moe_ep", "gemm", "fmha"),
    [
        ("MiniMaxAI/MiniMax-M2.7", "h200_sxm", 4, 1, 4, 1, "fp8_block", "bfloat16"),
        ("nvidia/GLM-5.2-NVFP4", "b200_sxm", 1, 8, 1, 8, "nvfp4", "fp8"),
        ("nvidia/GLM-5.2-NVFP4", "b200_sxm", 8, 1, 1, 8, "nvfp4", "fp8"),
    ],
)
def test_external_fpm_cells_query_with_model_construction_disabled(
    profile_dict, monkeypatch, tmp_path, model, system, tp, dp, moe_tp, moe_ep, gemm, fmha
):
    # Synthetic timing and resource declarations test the real checkpoint
    # identities through external storage with all graph constructors disabled.
    # Accuracy against retained silicon measurements is separate validation.
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    profile_dict.update(model=model)
    profile_dict["deployments"][0].update(
        system=system,
        tp=tp,
        dp=dp,
        moe_tp=moe_tp,
        moe_ep=moe_ep,
        gemm_quant_mode=gemm,
        moe_quant_mode=gemm,
        fmha_quant_mode=fmha,
    )
    monkeypatch.setattr(engine, "get_model", _fail_graph)
    monkeypatch.setattr(engine, "build_model_config", _fail_graph)
    monkeypatch.setattr(engine, "_maybe_load_database", _fail_graph)
    path = _write_external_profile_pair(profile_dict, tmp_path)
    forward = RustForwardPassPerfModel.best_available(
        _request(
            profile_dict,
            estimator_config={"fpm_interpolation": {"method": "direct", "fpm_parquet_path": str(path)}},
        )
    )
    resolved = forward.diagnostics()["provenance"]["config"]
    assert resolved["estimation_mode"] == "fpm_interpolation"
    assert resolved["estimator_config"]["fpm_interpolation"]["method"] == "direct"
    assert resolved["fpm_profile"] == load_fpm_profile(profile_dict).model_dump(mode="json")
    assert resolved["gemm_quant_mode"] == gemm
    assert resolved["fmha_quant_mode"] == fmha
    prefill_metrics = {"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 1}}
    decode_metrics = {
        "scheduled_requests": {
            "num_decode_requests": 1,
            "sum_decode_kv_tokens": 1,
        }
    }
    # Exact expectations are the hand-declared synthetic rows in the external pair.
    assert forward.estimate_forward_pass_time_ms(prefill_metrics) == pytest.approx(2.0)
    assert forward.estimate_forward_pass_time_ms(decode_metrics) == pytest.approx(3.0)
    reloaded = RustForwardPassPerfModel.best_available(json.loads(json.dumps(resolved)))
    assert reloaded.estimate_forward_pass_time_ms(prefill_metrics) == pytest.approx(2.0)
    assert reloaded.diagnostics()["provenance"]["config"] == resolved
    with pytest.raises(PerfDataNotAvailableError, match="direct"):
        forward.estimate_forward_pass_time_ms(
            {
                "scheduled_requests": {
                    "num_prefill_requests": 1,
                    "sum_prefill_tokens": 1_000_000,
                }
            }
        )
    assert forward.diagnostics()["provenance"]["config"] == resolved


def test_registered_glm_auto_retains_sol_with_explicit_fp8_fmha(profile_dict, monkeypatch, tmp_path):
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    profile_dict.update(model="nvidia/GLM-5.2-NVFP4", architecture="GlmMoeDsaForCausalLM")
    profile_dict["deployments"][0].update(system="b200_sxm", tp=1, dp=8, moe_tp=1, moe_ep=8)
    real_get_model = engine.get_model
    built_models = []

    def record_model(*args):
        model = real_get_model(*args)
        built_models.append(model)
        return model

    monkeypatch.setattr(engine, "get_model", record_model)
    monkeypatch.setattr(engine, "_direct_fpm_spec_json", _fail_graph)
    path = _write_external_profile_pair(profile_dict, tmp_path)
    forward = RustForwardPassPerfModel.best_available(
        _request(
            profile_dict,
            estimator_config={"fpm_interpolation": {"method": "auto", "fpm_parquet_path": str(path)}},
        )
    )
    resolved = forward.diagnostics()["provenance"]["config"]
    assert resolved["estimator_config"]["fpm_interpolation"]["method"] == "sol"
    assert len(built_models) == 1
    assert built_models[0].config.fmha_quant_mode.name == "fp8"
    assert built_models[0].context_ops[0]._sol_ops
    assert (
        forward.estimate_forward_pass_time_ms(
            {"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 1}}
        )
        > 0
    )


def test_profile_topology_accepts_dense_tp_without_moe_partition(profile_dict, monkeypatch):
    monkeypatch.setattr(engine, "compile_engine", _fail_graph)
    profile_dict["num_experts"] = 0
    profile_dict["deployments"][0]["moe_tp"] = 1
    resolved = _normalize(_request(profile_dict))
    assert resolved["tp"] == 2
    assert resolved["moe_tp_size"] == 1
    assert resolved["estimator_config"]["fpm_interpolation"]["method"] == "direct"


@pytest.mark.parametrize("registered", [False, True])
def test_profile_auto_preserves_global_priority_with_deny(profile_dict, monkeypatch, registered):
    from aisimulate_core.sdk.models.base import _MODEL_REGISTRY

    if registered:
        monkeypatch.setitem(_MODEL_REGISTRY, profile_dict["architecture"], object())
    attempts = []

    def unavailable(*_args, **kwargs):
        attempts.append(kwargs["forward_model"])
        raise RuntimeError("fixture native timing unavailable")

    monkeypatch.setattr(engine, "compile_engine", unavailable)
    forward = RustForwardPassPerfModel.best_available(_request(profile_dict, "direct", estimation_mode="auto"))
    provenance = forward.diagnostics()["provenance"]
    assert attempts == (["op_level", "fpm"] if registered else ["fpm"])
    assert len(provenance["selection_failures"]) == 2
    assert provenance["selected_estimation_mode"] == "fpm_regression"
    assert provenance["config"]["estimator_config"]["fpm_interpolation"]["method"] == "direct"
    assert (
        forward.estimate_forward_pass_time_ms(
            {"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 1}}
        )
        is None
    )


def test_canonical_profile_round_trip_can_size_memory_without_timing(profile_dict, profile_memory, monkeypatch):
    monkeypatch.setattr(engine, "compile_engine", _fail_graph)
    resolved = json.loads(json.dumps(_normalize(_request(profile_dict))))
    result = memory.estimate_kv_cache(
        resolved["model"],
        resolved["system"],
        resolved["backend"],
        resolved["backend_version"],
        tp_size=resolved["tp"],
        pp_size=resolved["pp"],
        attention_dp_size=resolved["attention_dp"],
        moe_tp_size=resolved["moe_tp_size"],
        moe_ep_size=resolved["moe_ep_size"],
        fpm_profile=resolved["fpm_profile"],
        gemm_quant_mode=resolved["gemm_quant_mode"],
        moe_quant_mode=resolved["moe_quant_mode"],
        fmha_quant_mode=resolved["fmha_quant_mode"],
        kvcache_quant_mode=resolved["kvcache_quant_mode"],
        comm_quant_mode=resolved["comm_quant_mode"],
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.8,
    )
    assert result["source"] == "profile"
    assert result["total_kv_size_bytes"] == 600  # 1000 * 0.8 - 200 declared non-KV bytes.
    assert result["total_kv_size_tokens"] == 60  # 600 bytes / 10 declared bytes per token.


@pytest.fixture
def measured_profile_roots(profile_dict, tmp_path, monkeypatch):
    """Small synthetic cells isolate availability from phase/domain coverage."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import aisimulate_core

    profile_dict["deployments"][0].update(system="h200_sxm", attention_backend="future_attention")
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    monkeypatch.setattr(engine, "get_model", _fail_graph)
    monkeypatch.setattr(engine, "build_model_config", _fail_graph)

    def make_root(name, kind):
        root = tmp_path / name
        root.mkdir()
        packaged = Path(aisimulate_core.__file__).parent / "systems/h200_sxm.yaml"
        (root / packaged.name).write_bytes(packaged.read_bytes())
        identity = load_fpm_profile(profile_dict).deployments[0].model_dump(mode="json", exclude={"resources"})
        coordinates = []
        if kind in {"genuine", "mixed"}:
            coordinates.extend([("prefill", 0, "real_kv", 2.0), ("decode", 1, "real_kv", 3.0)])
        if kind in {"fake", "mixed"}:
            coordinates.append(("decode", 64, "fake_fallback", 99.0))
        rows = [
            {
                **identity,
                "model_path": profile_dict["model"],
                "cell_id": f"synthetic-{phase}-{kv}",
                "weight_quantization": "synthetic",
                "workload_kind": phase,
                "partition_policy": "balanced_v1",
                "batch_size": 1,
                "total_prefill_tokens": 1 if phase == "prefill" else 0,
                "total_kv_read_tokens": kv,
                "latency_ms": latency,
                "kv_seed_regime": regime,
            }
            for phase, kv, regime, latency in coordinates
        ]
        path = root / "data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
        path.with_suffix(".metadata.json").write_text(
            json.dumps(
                {
                    "schema_name": "aic_fpm_forward_perf",
                    "schema_version": 6,
                    "coordinate_system": "iteration_totals_balanced_v1",
                    "measurement_policy": "dynamo_native_single_sample_v1",
                    "system": "h200_sxm",
                    "backend": "vllm",
                    "backend_version": "0.25.1",
                    "row_count": len(rows),
                    "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        )
        return str(root)

    return make_root


@pytest.mark.parametrize("fallback", ["deny", "allow"])
def test_direct_fake_only_cell_is_unavailable(profile_dict, measured_profile_roots, fallback):
    root = measured_profile_roots("fake", "fake")
    config = _request(profile_dict, "direct", systems_paths=[root], fallback_policy=fallback)
    if fallback == "deny":
        with pytest.raises(PerfDataNotAvailableError, match="no genuine measurements"):
            RustForwardPassPerfModel.best_available(config)
        return
    model = RustForwardPassPerfModel.best_available(config)
    diagnostics = model.diagnostics()
    assert diagnostics["readiness"] == "unsupported_config"
    assert diagnostics["provenance"]["selected_estimation_mode"] == "fpm_regression"
    assert diagnostics["provenance"]["selected_systems_root"] is None
    assert "no genuine measurements" in diagnostics["provenance"]["selection_failures"][0]
    assert (
        model.estimate_forward_pass_time_ms(
            {"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 64}}
        )
        is None
    )


@pytest.mark.parametrize("fallback", ["deny", "allow"])
@pytest.mark.parametrize(
    "kinds,selected",
    [(["genuine"], 0), (["mixed"], 0), (["fake", "genuine"], 1), (["fake", "mixed"], 1), (["genuine", "fake"], 0)],
)
def test_direct_availability_uses_first_genuine_root_and_pins_queries(
    profile_dict, measured_profile_roots, fallback, kinds, selected
):
    roots = [measured_profile_roots(str(index), kind) for index, kind in enumerate(kinds)]
    model = RustForwardPassPerfModel.best_available(
        _request(profile_dict, "direct", systems_paths=roots, fallback_policy=fallback)
    )
    before = model.diagnostics()
    assert before["readiness"] == "ready"
    assert before["provenance"]["selected_systems_root"] == roots[selected]
    assert before["provenance"]["config"]["systems_paths"] == [roots[selected]]
    # This is the exact, hand-declared genuine decode timing above.
    assert (
        model.estimate_forward_pass_time_ms(
            {"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 1}}
        )
        == 3.0
    )
    with pytest.raises(PerfDataNotAvailableError, match="direct"):
        model.estimate_forward_pass_time_ms(
            {"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 64}}
        )
    assert model.diagnostics() == before


def test_unknown_profile_outer_auto_skips_analytical_backend_parsing(profile_dict, measured_profile_roots):
    root = measured_profile_roots("genuine", "genuine")
    model = RustForwardPassPerfModel.best_available(
        _request(profile_dict, systems_paths=[root], estimation_mode="auto")
    )
    provenance = model.diagnostics()["provenance"]
    assert provenance["selected_estimation_mode"] == "fpm_interpolation"
    assert provenance["config"]["estimator_config"]["fpm_interpolation"]["method"] == "direct"
    assert provenance["config"]["attention_backend"] == "future_attention"
    assert len(provenance["selection_failures"]) == 1
    assert "registered architecture" in provenance["selection_failures"][0]
    assert (
        model.estimate_forward_pass_time_ms(
            {"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 1}}
        )
        == 3.0
    )


def test_registered_profile_outer_auto_keeps_op_level_priority_with_inner_direct(profile_dict, monkeypatch):
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    profile_dict.update(model="nvidia/GLM-5.2-NVFP4", architecture="GlmMoeDsaForCausalLM")
    # Op-level priority uses a currently shipped op database; FPM0.25.1 pairs
    # are external inputs and do not establish op-level data availability.
    profile_dict["deployments"][0].update(system="b200_sxm", tp=1, dp=8, moe_tp=1, moe_ep=8, backend_version="0.24.0")
    monkeypatch.setattr(engine, "_direct_fpm_spec_json", _fail_graph)
    model = RustForwardPassPerfModel.best_available(_request(profile_dict, "direct", estimation_mode="auto"))
    provenance = model.diagnostics()["provenance"]
    assert provenance["selected_estimation_mode"] == "op_level"
    assert provenance["selection_failures"] == []
    assert provenance["config"]["estimator_config"]["fpm_interpolation"]["method"] == "direct"
