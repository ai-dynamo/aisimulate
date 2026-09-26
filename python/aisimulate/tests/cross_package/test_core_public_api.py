# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the unified wheel's performance-model facade."""

from __future__ import annotations

import ast
import importlib.resources
import inspect
import json
import subprocess
import sys

import pytest

import aisimulate_core
import aisimulate_core.sdk as sdk
from aisimulate_core.sdk.common import AttentionBackend, MoEBackend
from aisimulate_core.sdk.config import ModelConfig, RuntimeConfig
from aisimulate_core.sdk.engine import EngineHandle, compile_engine
from aisimulate_core.sdk.memory import estimate_kv_cache, estimate_num_gpu_blocks
from aisimulate_core.sdk.operations import ElementWise, Embedding, MoEDispatch
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

EXPECTED_FACADE = {
    "AttentionBackend",
    "EngineHandle",
    "ForwardPassPerfModelConfig",
    "ForwardPassPerfOptions",
    "ModelConfig",
    "MoEBackend",
    "RuntimeConfig",
    "RustForwardPassPerfModel",
    "compile_engine",
    "estimate_kv_cache",
    "estimate_num_gpu_blocks",
}


def _raw_regression_model(worker_type, options_json=None, *, cls=None):
    cls = cls or aisimulate_core.RustForwardPassPerfModel
    config = {
        "model": "test/model",
        "system": "test",
        "backend": "vllm",
        "worker_type": worker_type,
        "estimation_mode": "fpm_regression",
    }
    if options_json is not None:
        config["estimator_config"] = json.loads(cls.legacy_estimator_config(options_json))
    return cls.best_available(json.dumps(config))


def test_sdk_facade_import_is_lazy_in_a_fresh_interpreter() -> None:
    script = """
import sys

import aisimulate_core.sdk

protected_modules = {
    "aisimulate_core.sdk.engine",
    "aisimulate_core.sdk.memory",
    "aisimulate_core.sdk.rust_engine_step",
}
loaded_modules = protected_modules.intersection(sys.modules)
assert not loaded_modules, f"SDK facade eagerly loaded: {sorted(loaded_modules)}"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_sdk_facade_exports_the_canonical_objects() -> None:
    assert set(sdk.__all__) == EXPECTED_FACADE
    assert sdk.AttentionBackend is AttentionBackend
    assert sdk.EngineHandle is EngineHandle
    assert sdk.ModelConfig is ModelConfig
    assert sdk.MoEBackend is MoEBackend
    assert sdk.RuntimeConfig is RuntimeConfig
    assert sdk.RustForwardPassPerfModel is RustForwardPassPerfModel
    assert sdk.compile_engine is compile_engine
    assert sdk.estimate_kv_cache is estimate_kv_cache
    assert sdk.estimate_num_gpu_blocks is estimate_num_gpu_blocks


def test_native_and_ergonomic_fpm_classes_are_deliberately_distinct() -> None:
    assert aisimulate_core.RustForwardPassPerfModel is not RustForwardPassPerfModel
    assert RustForwardPassPerfModel.__module__ == "aisimulate_core.sdk.rust_engine_step"


def test_stable_function_signatures() -> None:
    assert str(inspect.signature(compile_engine)) == (
        "(model_path: 'str', system: 'str', backend: 'str', backend_version: 'str | None' = None, *, "
        "tp_size: 'int' = 1, pp_size: 'int' = 1, attention_dp_size: 'int' = 1, "
        "moe_tp_size: 'int | None' = None, moe_ep_size: 'int | None' = None, "
        "gemm_quant_mode: 'str | None' = None, moe_quant_mode: 'str | None' = None, "
        "kvcache_quant_mode: 'str | None' = None, fmha_quant_mode: 'str | None' = None, "
        "fpm_fmha_quant_mode: 'str | None' = None, "
        "comm_quant_mode: 'str | None' = None, attention_backend: 'str | None' = None, "
        "moe_kernel_source: 'str | None' = None, "
        "moe_backend: 'str | None' = None, enable_eplb: 'bool' = False, wideep_num_slots: 'int | None' = None, "
        "nextn: 'int' = 0, "
        "speculation: 'dict | None' = None, "
        "kv_block_size: 'int | None' = None, "
        "systems_path: 'str | None' = None, "
        "forward_model: 'str | None' = None, "
        "decoder_replay: 'bool' = False, "
        "database_mode: 'str | None' = None, shared_layer: 'bool | None' = None, "
        "transfer_policy: 'str | list[str] | None' = None, "
        "strict_provenance: 'bool | None' = None, fpm_parquet_path: 'str | None' = None) -> 'bytes'"
    )
    assert "scheduler_block_size" in inspect.signature(estimate_num_gpu_blocks).parameters
    assert "memory_fraction_kind" in inspect.signature(estimate_kv_cache).parameters
    assert list(inspect.signature(RustForwardPassPerfModel.best_available).parameters) == ["config"]
    assert not hasattr(RustForwardPassPerfModel, "from_regression")
    assert not hasattr(RustForwardPassPerfModel, "from_native")


@pytest.mark.parametrize("worker_type", ["agg", "Prefill", "AGGREGATED", ""])
def test_raw_fpm_binding_rejects_worker_type_aliases(worker_type: str) -> None:
    with pytest.raises(ValueError, match="worker_type"):
        _raw_regression_model(worker_type)


def test_raw_fpm_binding_requires_worker_type() -> None:
    with pytest.raises(ValueError, match="worker_type"):
        aisimulate_core.RustForwardPassPerfModel.best_available(
            json.dumps({"model": "m", "system": "s", "backend": "vllm"})
        )


@pytest.mark.parametrize("worker_type", ["prefill", "decode", "aggregated"])
def test_raw_fpm_binding_constructs_every_worker_type_with_weights(worker_type: str) -> None:
    options = json.dumps(
        {
            "regression_attention_kv_weight": 2.0,
            "regression_prefill_attention_pair_weight": 3.0,
            "regression_ffn_token_weight": 4.0,
        }
    )
    model = _raw_regression_model(worker_type, options)
    diagnostics = json.loads(model.diagnostics())
    assert {key: value for key, value in diagnostics.items() if key != "provenance"} == {
        "source": "fallback_regression",
        "readiness": "insufficient_data",
        "retained_observations": 0,
        "correction_ready_buckets": 0,
        "last_warning": None,
    }


def _raw_regression_iteration(worker_type: str, index: int) -> dict[str, object]:
    if worker_type == "prefill":
        num_prefill_requests = index % 3 + 1
        sum_prefill_tokens = index * 5 + 3
        sum_prefill_kv_tokens = index * index * 7
        scheduled_requests = {
            "num_prefill_requests": num_prefill_requests,
            "sum_prefill_tokens": sum_prefill_tokens,
            "sum_prefill_kv_tokens": sum_prefill_kv_tokens,
        }
        attention = (
            sum_prefill_kv_tokens
            + sum_prefill_kv_tokens * sum_prefill_tokens / num_prefill_requests
            + sum_prefill_tokens**2 / (2 * num_prefill_requests)
            + sum_prefill_tokens / 2
        )
        ffn = sum_prefill_tokens
    elif worker_type == "decode":
        num_decode_requests = index + 1
        sum_decode_kv_tokens = index * index * 17 + 11
        scheduled_requests = {
            "num_decode_requests": num_decode_requests,
            "sum_decode_kv_tokens": sum_decode_kv_tokens,
        }
        attention = sum_decode_kv_tokens
        ffn = num_decode_requests
    else:
        num_prefill_requests = index % 3 + 1
        sum_prefill_tokens = index * 4 + 2
        sum_prefill_kv_tokens = index * index * 5
        num_decode_requests = index + 1
        sum_decode_kv_tokens = index * index * 13 + 7
        scheduled_requests = {
            "num_prefill_requests": num_prefill_requests,
            "sum_prefill_tokens": sum_prefill_tokens,
            "sum_prefill_kv_tokens": sum_prefill_kv_tokens,
            "num_decode_requests": num_decode_requests,
            "sum_decode_kv_tokens": sum_decode_kv_tokens,
        }
        attention = (
            sum_prefill_kv_tokens
            + sum_decode_kv_tokens
            + sum_prefill_kv_tokens * sum_prefill_tokens / num_prefill_requests
            + sum_prefill_tokens**2 / (2 * num_prefill_requests)
            + sum_prefill_tokens / 2
        )
        ffn = sum_prefill_tokens + num_decode_requests

    observed_ms = 1.0 + 0.01 * attention + 0.1 * ffn
    return {
        "version": 1,
        "wall_time": observed_ms / 1000.0,
        "scheduled_requests": scheduled_requests,
    }


@pytest.mark.parametrize("worker_type", ["prefill", "decode", "aggregated"])
def test_raw_fpm_binding_regression_round_trip(worker_type: str) -> None:
    model = _raw_regression_model(
        worker_type,
        '{"min_observations":5}',
    )
    iterations = [[_raw_regression_iteration(worker_type, index)] for index in range(1, 7)]

    assert model.estimate_forward_pass_time_ms(json.dumps(iterations[-1])) is None
    model.tune_with_fpms(json.dumps(iterations))

    diagnostics = json.loads(model.diagnostics())
    assert {key: value for key, value in diagnostics.items() if key != "provenance"} == {
        "source": "fallback_regression",
        "readiness": "ready",
        "retained_observations": 6,
        "correction_ready_buckets": 0,
        "last_warning": None,
    }
    prediction = model.estimate_forward_pass_time_ms(json.dumps(iterations[-1]))
    assert prediction is not None and prediction > 0.0


@pytest.mark.parametrize(
    ("fit", "expected_interval"),
    [
        ({}, None),
        ({"rebuild_interval": 31}, 31),
        ({"rebuild_interval": 4096}, 4096),
        ({"rebuild_interval": None}, None),
    ],
)
def test_raw_canonical_regression_rebuild_interval_normalizes_and_reloads(fit, expected_interval):
    raw = aisimulate_core.RustForwardPassPerfModel
    request = {
        "model": "test/model",
        "system": "test",
        "backend": "vllm",
        "worker_type": "decode",
        "estimation_mode": "fpm_regression",
        "estimator_config": {"fpm_regression": {"fit": fit}},
    }
    normalized = json.loads(raw.normalize_config(json.dumps(request)))
    assert normalized["estimator_config"]["fpm_regression"]["fit"] == {
        "kind": "standardized_nnls",
        "singular_ridge_scale": 1e-9,
        "rebuild_interval": expected_interval,
    }
    model = raw.best_available(json.dumps(normalized))
    resolved = json.loads(model.diagnostics())["provenance"]["config"]
    assert resolved["estimator_config"] == normalized["estimator_config"]
    restored = raw.best_available(json.dumps(resolved))
    assert json.loads(restored.diagnostics())["provenance"]["config"] == resolved


@pytest.mark.parametrize("entrypoint", ["normalize_config", "best_available"])
@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, 4096.0, "4096", [], {}])
def test_raw_canonical_regression_rebuild_interval_rejects_invalid_values_with_path(entrypoint, invalid):
    request = {
        "model": "test/model",
        "system": "test",
        "backend": "vllm",
        "worker_type": "decode",
        "estimation_mode": "auto",
        "fallback_policy": "allow",
        "estimator_config": {"fpm_regression": {"fit": {"rebuild_interval": invalid}}},
    }
    # Invalid configuration fails before automatic selection or fallback.
    with pytest.raises(ValueError, match=r"estimator_config\.fpm_regression\.fit\.rebuild_interval"):
        getattr(aisimulate_core.RustForwardPassPerfModel, entrypoint)(json.dumps(request))


def test_legacy_options_inherit_rust_rebuild_default_without_a_new_flat_control():
    migrated = json.loads(aisimulate_core.RustForwardPassPerfModel.legacy_estimator_config("{}"))
    assert migrated["fpm_regression"]["fit"]["rebuild_interval"] is None
    assert migrated["fpm_regression"]["sampling"] == {"bins_per_axis": [4, 4], "max_observations": 64}
    assert "rebuild_interval" not in sdk.ForwardPassPerfOptions.__dataclass_fields__
    assert "regression_rebuild_interval" not in sdk.ForwardPassPerfOptions.__dataclass_fields__
    assert (
        sdk.ForwardPassPerfModelConfig(
            model="test/model", system="test", backend="vllm", worker_type="decode"
        ).estimator_config
        == {}
    )


def test_raw_fpm_binding_validates_regression_weights() -> None:
    with pytest.raises(ValueError, match="regression_attention_kv_weight"):
        _raw_regression_model(
            "prefill",
            '{"regression_attention_kv_weight": 0.0}',
        )


@pytest.mark.parametrize(
    "field",
    [
        "regression_attention_kv_weight",
        "regression_prefill_attention_pair_weight",
        "regression_ffn_token_weight",
    ],
)
@pytest.mark.parametrize("sentinel", ["NaN", "Infinity", "-Infinity"])
def test_raw_fpm_binding_decodes_exact_nonfinite_weight_sentinels(
    field: str,
    sentinel: str,
) -> None:
    """Raw ``options_json`` callers use quoted sentinels rather than non-JSON numbers."""
    with pytest.raises(ValueError, match=field):
        _raw_regression_model(
            "aggregated",
            json.dumps({field: sentinel}),
        )


def test_raw_fpm_binding_rejects_unknown_nonfinite_weight_sentinel() -> None:
    with pytest.raises(ValueError, match="invalid legacy options"):
        _raw_regression_model(
            "decode",
            '{"regression_ffn_token_weight":"Inf"}',
        )


def test_native_operation_constructors_preserve_legacy_keyword_names() -> None:
    Embedding("embedding", 1.0, 1024, 128, empirical_bw_scaling_factor=0.4)
    ElementWise("elementwise", 1.0, 128, 64, empirical_bw_scaling_factor=0.6)
    MoEDispatch(
        "dispatch",
        1.0,
        7168,
        8,
        256,
        1,
        16,
        1,
        False,
        enable_fp4_all2all=False,
        backend="sglang",
        reduce_results=False,
    )


def test_distribution_carries_typing_contract() -> None:
    root = importlib.resources.files("aisimulate_core")
    assert (root / "py.typed").is_file()
    assert (root / "_native.pyi").is_file()


@pytest.mark.unit
@pytest.mark.parametrize("namespace", ["aisimulate_core", "aisimulate_core"])
def test_regression_bucket_diagnostics_stub_matches_native_contract(namespace: str) -> None:
    root = importlib.resources.files("aisimulate_core")
    stub = ast.parse((root / "_native.pyi").read_text(encoding="utf-8"))
    model = next(
        node for node in stub.body if isinstance(node, ast.ClassDef) and node.name == "RustForwardPassPerfModel"
    )
    method_name = "regression_store_diagnostics"
    method = next(
        (node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == method_name),
        None,
    )
    assert method is not None, f"The shipped RustForwardPassPerfModel stub omits {method_name}"
    assert [argument.arg for argument in method.args.args] == ["self"]
    assert ast.unparse(method.returns) == "str"

    native_model = _raw_regression_model("aggregated", cls=importlib.import_module(namespace).RustForwardPassPerfModel)
    diagnostics = getattr(native_model, method_name)()
    assert isinstance(diagnostics, str)
    assert json.loads(diagnostics) == [
        {"workload_kind": kind, "ready": False, "retained_observations": 0}
        for kind in ("pure_decode", "contains_locally_mixed", "cross_rank_aggregated", "pure_prefill")
    ]


@pytest.mark.unit
@pytest.mark.parametrize("namespace", ["aisimulate_core", "aisimulate_core"])
def test_context_attention_kernel_stub_matches_native_contract(namespace: str) -> None:
    root = importlib.resources.files("aisimulate_core")
    stub = ast.parse((root / "_native.pyi").read_text(encoding="utf-8"))
    engine = next(node for node in stub.body if isinstance(node, ast.ClassDef) and node.name == "AicEngine")
    method_name = "evaluate_context_attention_kernels_json"
    method = next(
        (node for node in engine.body if isinstance(node, ast.FunctionDef) and node.name == method_name),
        None,
    )
    assert method is not None, f"The shipped AicEngine stub omits {method_name}"

    native_engine = importlib.import_module(namespace).AicEngine
    parameters = list(inspect.signature(getattr(native_engine, method_name)).parameters.values())
    arguments = method.args
    stub_parameters = [*arguments.posonlyargs, *arguments.args]
    assert [argument.arg for argument in stub_parameters] == [parameter.name for parameter in parameters]
    assert [parameter.kind for parameter in parameters] == [
        *[inspect.Parameter.POSITIONAL_ONLY] * len(arguments.posonlyargs),
        *[inspect.Parameter.POSITIONAL_OR_KEYWORD] * len(arguments.args),
    ]
    assert arguments.vararg is None and arguments.kwarg is None and not arguments.kwonlyargs
    assert [ast.literal_eval(default) for default in arguments.defaults] == [
        parameter.default for parameter in parameters if parameter.default is not inspect.Parameter.empty
    ]
    assert {argument.arg: ast.unparse(argument.annotation) for argument in stub_parameters[1:]} == {
        "ops_json": "str",
        "batch_size": "int",
        "s": "int",
        "prefix": "int",
        "imbalance_correction_scale": "float",
        "visual_block_upper_triangle": "bool",
    }
    assert ast.unparse(method.returns) == "list[tuple[str, float, float, str]]"


@pytest.mark.unit
def test_static_phase_diagnostics_stub_matches_native_contract() -> None:
    root = importlib.resources.files("aisimulate_core")
    stub = ast.parse((root / "_native.pyi").read_text(encoding="utf-8"))
    model = next(
        node for node in stub.body if isinstance(node, ast.ClassDef) and node.name == "RustForwardPassPerfModel"
    )
    method = next(
        node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "static_phase_diagnostics"
    )
    native_class = importlib.import_module("aisimulate_core").RustForwardPassPerfModel
    parameters = inspect.signature(native_class.static_phase_diagnostics).parameters
    assert [arg.arg for arg in method.args.args] == list(parameters)
    assert {arg.arg: ast.unparse(arg.annotation) for arg in method.args.args[1:]} == {
        "batch_size": "int",
        "context_length": "int",
        "prefix": "int",
        "prefill": "bool",
    }
    assert ast.unparse(method.returns) == "str"
    model = native_class.best_available(
        json.dumps(
            {
                "model": "Qwen/Qwen3-32B",
                "system": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.24.0",
                "worker_type": "aggregated",
                "estimation_mode": "op_level",
                "tp": 2,
            }
        )
    )
    # Zero scheduled work is an empty native JSON array, not a Python list.
    result = model.static_phase_diagnostics(0, 128, 0, True)
    assert isinstance(result, str)
    assert json.loads(result) == []
