# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Includes changes adapted from:
# https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/test_fpm_forward.py

"""Unit tests for forward_model="fpm": FPMForwardOp construction, the
centralized model rewrite, and static/mixed-step routing through the compiled
Rust engine (``Op::FpmForward``; the loader/query machinery lives in
``perf_database/fpm_forward.rs`` with its own tests).

Synthetic parquet/metadata pairs are written directly from the documented
``aic_fpm_forward_perf`` schema — deliberately NOT via collector code, so
this suite doubles as the producer/consumer contract test on the modeling
side of the module boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.main import main
from aisimulate.runner import AICAFDCompanionPerformanceModel, EngineReplayRunnerFactory
from aisimulate.sdk import common, models
from aisimulate.sdk import config as sdk_config
from aisimulate.sdk.backends.factory import get_backend
from aisimulate.sdk.operations import FPMForwardOp
from aisimulate.sdk.perf_database import PerfDatabase
from aisimulate.sweeper import AFDLayerTimes, AFDTopology, BackendDeploymentSpec, ReplaySpec
from aisimulate.sweeper.replay import ReplayOutputRequirements
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel
from aisimulate_core.sdk.engine import EngineHandle, compile_engine
from aisimulate_core.sdk.operations.fpm_forward import _CELL_MATCH_COLUMNS

pytestmark = pytest.mark.unit

SYSTEM = "h200_sxm"
BACKEND = "vllm"
VERSION = "test-fpm-version"
MODEL_PATH = "test-org/test-model"

import aisimulate_core

_CORE_SYSTEMS = os.path.join(os.path.dirname(aisimulate_core.__file__), "systems")


def _row(
    workload_kind: str,
    batch_size: int,
    total_prefill_tokens: int,
    total_kv_read_tokens: int,
    latency_ms: float,
    *,
    model_path: str = MODEL_PATH,
    identity: dict | None = None,
) -> dict:
    base_identity = {
        "gemm_quant_mode": "bfloat16",
        "moe_quant_mode": "",
        "fmha_quant_mode": "",
        "comm_quant_mode": "half",
        "kv_cache_dtype": "bfloat16",
        "tp": "1",
        "pp": "1",
        "dp": "1",
        "moe_tp": "1",
        "moe_ep": "1",
        "cp": "1",
        "moe_backend": "auto",
        "attention_backend": "auto",
        "enable_wideep": False,
        "enable_eplb": False,
    }
    if identity:
        base_identity.update(identity)
    # Identity dicts built from an op's _match_identity carry str(bool)
    # ("True"/"False"); the parquet columns are REAL booleans.
    for knob in ("enable_wideep", "enable_eplb"):
        if isinstance(base_identity[knob], str):
            base_identity[knob] = base_identity[knob] == "True"
    return {
        "cell_id": f"fpm-test-{workload_kind}",
        "model_path": model_path,
        "system": SYSTEM,
        "backend": BACKEND,
        "backend_version": VERSION,
        "weight_quantization": base_identity["gemm_quant_mode"],
        "gemm_quant_mode": base_identity["gemm_quant_mode"],
        "moe_quant_mode": base_identity["moe_quant_mode"] or None,
        "fmha_quant_mode": base_identity["fmha_quant_mode"] or None,
        "comm_quant_mode": base_identity["comm_quant_mode"] or None,
        "kv_cache_dtype": base_identity["kv_cache_dtype"],
        "tp": int(base_identity["tp"]),
        "pp": int(base_identity["pp"]),
        "dp": int(base_identity["dp"]),
        "moe_tp": int(base_identity["moe_tp"]),
        "moe_ep": int(base_identity["moe_ep"]),
        "cp": int(base_identity["cp"]),
        "moe_backend": base_identity["moe_backend"],
        "attention_backend": base_identity["attention_backend"],
        "enable_wideep": base_identity["enable_wideep"],
        "enable_eplb": base_identity["enable_eplb"],
        "workload_kind": workload_kind,
        "batch_size": batch_size,
        "total_prefill_tokens": total_prefill_tokens,
        "total_kv_read_tokens": total_kv_read_tokens,
        "partition_policy": "balanced_v1",
        "latency_ms": latency_ms,
    }


def _write_pair(data_dir: str, rows: list[dict], *, sidecar_overrides: dict | None = None) -> str:
    os.makedirs(data_dir, exist_ok=True)
    parquet_path = os.path.join(data_dir, "fpm_forward_perf.parquet")
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    with open(parquet_path, "rb") as handle:
        parquet_sha = hashlib.sha256(handle.read()).hexdigest()
    metadata = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "coordinate_system": "iteration_totals_balanced_v1",
        "measurement_policy": "dynamo_native_single_sample_v1",
        "row_count": len(rows),
        "parquet_sha256": parquet_sha,
        "system": SYSTEM,
        "backend": BACKEND,
        "backend_version": VERSION,
    }
    metadata.update(sidecar_overrides or {})
    with open(os.path.join(data_dir, "fpm_forward_perf.metadata.json"), "w") as handle:
        json.dump(metadata, handle)
    return parquet_path


def _model_config(**overrides) -> sdk_config.ModelConfig:
    defaults = dict(
        tp_size=1,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
    )
    defaults.update(overrides)
    return sdk_config.ModelConfig(**defaults)


def _median_pair_request(tmp_path, backend, *, model_path="zai-org/GLM-5.3-Flash", version=None, tp=2, points=None):
    """Synthetic measured medians, independent of the production writer.

    There are no operator tables, so the public SILICON query must use these
    exact already-reduced millisecond values, without SOL or per-sample scaling.
    """
    system = "gb300"
    version = version or ("0.30.0" if backend == "vllm" else "0.5.20")
    systems_root = tmp_path / "systems"
    systems_root.mkdir()
    shutil.copy(Path(_CORE_SYSTEMS) / f"{system}.yaml", systems_root / f"{system}.yaml")
    model = models.get_model(
        model_path, sdk_config.ModelConfig(tp_size=tp, moe_tp_size=tp, moe_ep_size=1, forward_model="fpm"), backend
    )
    identity = dict(zip(_CELL_MATCH_COLUMNS, model.context_ops[0]._match_identity, strict=True))
    boundary = "vllm_native_scheduler_output_interval" if backend == "vllm" else "sglang_native_forward_device_timer"
    rows = [
        _row(phase, batch, q, kv, latency, model_path=model_path, identity=identity)
        | {key: identity[key] for key in _CELL_MATCH_COLUMNS[-4:]}
        | dict(
            system=system,
            backend=backend,
            backend_version=version,
            measurement_policy=f"{backend}_native_real_hybrid_median_v1",
            global_warmup_iterations=0,
            warmup_repeats=5,
            measurement_repeats=10,
            state_protocol="glm53flash_same_request_real_hybrid_v1",
            timing_boundary=boundary,
            kv_seed_regime="n/a" if phase == "prefill" and kv == 0 else "real_kv",
        )
        for phase, batch, q, kv, latency in (
            points
            if points is not None
            else [
                ("prefill", 1, 512, 0, 13.25),
                ("prefill", 1, 1024, 0, 26.5),
                ("decode", 1, 0, 512, 7.125),
                ("decode", 1, 0, 1024, 14.25),
            ]
        )
    ]
    metadata = dict(
        schema_version=7,
        system=system,
        backend=backend,
        backend_version=version,
        measurement_policy=f"{backend}_native_real_hybrid_median_v1",
        warmup_repeats=5,
        measurement_repeats=10,
    )
    path = _write_pair(str(tmp_path / "measured"), rows, sidecar_overrides=metadata)
    config = ForwardPassPerfModelConfig(
        model=model_path,
        system=system,
        backend=backend,
        backend_version=version,
        worker_type="aggregated",
        tp=tp,
        moe_tp_size=tp,
        moe_ep_size=1,
        database_mode="SILICON",
        strict_provenance=True,
        estimation_mode="fpm_interpolation",
        fallback_policy="deny",
        systems_paths=(str(systems_root),),
        estimator_config={"fpm_interpolation": {"fpm_parquet_path": path}},
    )
    return config, rows, metadata


# Own observed failure geometries; all values below are TEST_ONLY invented data,
# never copied measured latencies or campaign acceptance evidence.
_TAIL_PREFIXES = (122, 125, 131, 134, 4346, 4349, 4355, 4358)
_TAIL_VERSION = "0.30.0+glm53tail.eb4704514fdf"


def _tail_metrics(batch, query, prefix):
    metrics = _median_metrics("prefill", batch * query)
    metrics["scheduled_requests"].update(num_prefill_requests=batch, sum_prefill_kv_tokens=batch * prefix)
    return metrics


@pytest.mark.parametrize("model_path", ["zai-org/GLM-5.3-Flash", "nvidia/GLM-5.3-Flash-NVFP4"])
@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("bracket", [False, True])
def test_public_tail_fpm_qualified_unaligned_exact_and_bracket(tmp_path, model_path, tp, bracket):
    from collector import glm53flash_runtime_identity as identity

    # This public Rust behavior must agree with the actual immutable packaged
    # four-cell qualification, not a monkeypatched admission or version prefix.
    assert identity.ADMITTED_VLLM_REPAIRS == {
        _TAIL_VERSION: "8fc691d6054f48741c248eb7937b7b4db6220ff1ea337b968ff656c56ba8cf45"
    }
    assert identity.vllm_unaligned_prefill_admitted(_TAIL_VERSION)
    endpoints = ((-1, 13.0), (1, 17.0)) if bracket else ((0, 15.0),)
    points = [
        ("prefill", batch, batch * 32, batch * (prefix + delta), latency)
        for batch in (1, 4, 32)
        for prefix in _TAIL_PREFIXES
        for delta, latency in endpoints
    ]
    config, _, _ = _median_pair_request(
        tmp_path, "vllm", model_path=model_path, version=_TAIL_VERSION, tp=tp, points=points
    )
    predictor = RustForwardPassPerfModel.best_available(config)
    try:
        resolved = predictor.diagnostics()["provenance"]["config"]
        assert resolved["backend_version"] == _TAIL_VERSION
        assert ForwardPassPerfModelConfig(**resolved).to_dict() == resolved
        for batch in (1, 4, 32):
            for prefix in _TAIL_PREFIXES:
                value = predictor.estimate_forward_pass_time_ms(_tail_metrics(batch, 32, prefix))
                # Exact lookup preserves the invented 15ms; a bracket must
                # stay strictly between its independently specified endpoints.
                if bracket:
                    assert 13.0 < value < 17.0
                else:
                    assert value == 15.0
    finally:
        predictor.close()


@pytest.mark.parametrize(
    "version",
    [
        "0.30.0",
        "0.30.0+unknown",
        _TAIL_VERSION + ".other",
        "0.30.0+glm53tail.eb4704514fde",
        "0.30.0+glm53kpool.bf5f6b0e689d",
    ],
)
def test_public_tail_fpm_rejects_unqualified_exact_data(tmp_path, version):
    points = [("prefill", 1, 32, p, 15.0) for p in _TAIL_PREFIXES]
    config, _, _ = _median_pair_request(tmp_path, "vllm", version=version, points=points)
    predictor = RustForwardPassPerfModel.best_available(config)
    try:
        for prefix in _TAIL_PREFIXES:
            with pytest.raises(Exception, match="cached-prefill start is unqualified|runtime quarantined"):
                predictor.estimate_forward_pass_time_ms(_tail_metrics(1, 32, prefix))
    finally:
        predictor.close()


def _median_metrics(phase, tokens):
    return dict(
        version=1,
        wall_time=1.0,
        scheduled_requests=dict(
            num_prefill_requests=int(phase == "prefill"),
            num_decode_requests=int(phase == "decode"),
            sum_prefill_tokens=tokens if phase == "prefill" else 0,
            sum_prefill_kv_tokens=0,
            sum_decode_kv_tokens=tokens if phase == "decode" else 0,
            var_prefill_length=0.0,
            var_decode_kv_tokens=0.0,
        ),
    )


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("model_path", ["zai-org/GLM-5.3-Flash", "nvidia/GLM-5.3-Flash-NVFP4"])
def test_public_median_fpm_preserves_ms_interpolation_and_saved_identity(tmp_path, backend, model_path):
    config, rows, metadata = _median_pair_request(tmp_path, backend, model_path=model_path)
    median = RustForwardPassPerfModel.best_available(config)
    resolved = median.diagnostics()["provenance"]["config"]
    restored = RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig(**resolved))
    # The legacy table has the identical numerical data. Only its measurement
    # contract differs; this anchors unchanged interpolation without copying math.
    legacy_rows = [
        {
            k: v
            for k, v in row.items()
            if k
            not in {
                "measurement_policy",
                "global_warmup_iterations",
                "warmup_repeats",
                "measurement_repeats",
                "state_protocol",
                "timing_boundary",
            }
        }
        for row in rows
    ]
    legacy_path = _write_pair(
        str(tmp_path / "legacy"),
        legacy_rows,
        sidecar_overrides=metadata | {"measurement_policy": "dynamo_native_single_sample_v1"},
    )
    legacy_config = config.to_dict() | {"estimator_config": {"fpm_interpolation": {"fpm_parquet_path": legacy_path}}}
    legacy = RustForwardPassPerfModel.best_available(legacy_config)
    try:
        assert restored.diagnostics()["provenance"]["config"] == resolved
        for phase, endpoints in [("prefill", (13.25, 26.5)), ("decode", (7.125, 14.25))]:
            for tokens, expected in [(512, endpoints[0]), (1024, endpoints[1]), (768, None)]:
                metrics = _median_metrics(phase, tokens)
                value = median.estimate_forward_pass_time_ms(metrics)
                assert value == restored.estimate_forward_pass_time_ms(metrics)
                assert value == legacy.estimate_forward_pass_time_ms(metrics)
                if expected is not None:
                    assert value == expected
                else:
                    assert endpoints[0] < value < endpoints[1]
    finally:
        median.close()
        restored.close()
        legacy.close()


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("measurement_policy", "dynamo_native_single_sample_v1"),
        ("measurement_policy", None),
        ("measurement_policy", "MISSING"),
        ("global_warmup_iterations", 1),
        ("warmup_repeats", 4),
        ("measurement_repeats", 1),
        ("measurement_repeats", 10.0),
        ("measurement_repeats", "10"),
        ("measurement_repeats", True),
        ("measurement_repeats", None),
        ("measurement_repeats", "MISSING"),
        ("state_protocol", "wrong"),
        ("state_protocol", None),
        ("state_protocol", "MISSING"),
        ("timing_boundary", "wrong"),
        ("timing_boundary", None),
        ("timing_boundary", "MISSING"),
        ("kv_seed_regime", "fake_fallback"),
    ],
)
def test_public_median_fpm_rejects_row_contract_before_prediction(tmp_path, backend, key, value):
    config, rows, metadata = _median_pair_request(tmp_path, backend)
    # A whole-column type change is valid parquet but invalid protocol. A
    # single contradictory row tests that valid siblings cannot hide a mix.
    changed = rows if value == "MISSING" or type(value) in (float, bool) or value == "10" else [rows[1]]
    for row in changed:
        if value == "MISSING":
            row.pop(key)
        else:
            row[key] = value
    _write_pair(str(tmp_path / "measured"), rows, sidecar_overrides=metadata)
    with pytest.raises((ValueError, RuntimeError), match="fake_fallback" if key == "kv_seed_regime" else key):
        model = RustForwardPassPerfModel.best_available(config)
        try:
            model.estimate_forward_pass_time_ms(_median_metrics("prefill", 512))
        finally:
            model.close()


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", 6),
        ("measurement_policy", "per_row"),
        ("measurement_policy", "OTHER_BACKEND"),
        ("warmup_repeats", 4),
        ("warmup_repeats", None),
        ("warmup_repeats", "MISSING"),
        ("measurement_repeats", 1),
        ("measurement_repeats", 10.0),
        ("measurement_repeats", "10"),
        ("measurement_repeats", True),
        ("measurement_repeats", None),
        ("measurement_repeats", "MISSING"),
    ],
)
def test_public_median_fpm_rejects_sidecar_contract(tmp_path, backend, key, value):
    config, rows, metadata = _median_pair_request(tmp_path, backend)
    if value == "OTHER_BACKEND":
        value = f"{'sglang' if backend == 'vllm' else 'vllm'}_native_real_hybrid_median_v1"
    if value == "MISSING":
        metadata.pop(key)
    else:
        metadata[key] = value
    _write_pair(str(tmp_path / "measured"), rows, sidecar_overrides=metadata)
    with pytest.raises((ValueError, RuntimeError), match="measurement_policy|median sidecar"):
        model = RustForwardPassPerfModel.best_available(config)
        try:
            model.estimate_forward_pass_time_ms(_median_metrics("prefill", 512))
        finally:
            model.close()


@pytest.mark.parametrize("optional_policy", [None, "historical-label"])
def test_legacy_single_sample_optional_row_policy_retains_prior_behavior(tmp_path, optional_policy):
    config, rows, metadata = _median_pair_request(tmp_path, "vllm")
    for row in rows:
        row["measurement_policy"] = optional_policy
        for key in (
            "state_protocol",
            "timing_boundary",
            "global_warmup_iterations",
            "warmup_repeats",
            "measurement_repeats",
        ):
            row.pop(key)
    _write_pair(
        str(tmp_path / "measured"),
        rows,
        sidecar_overrides=metadata | {"measurement_policy": "dynamo_native_single_sample_v1"},
    )
    model = RustForwardPassPerfModel.best_available(config)
    try:
        assert model.estimate_forward_pass_time_ms(_median_metrics("prefill", 512)) == 13.25
    finally:
        model.close()


# ---------------------------------------------------------------------------
# Centralized model rewrite
# ---------------------------------------------------------------------------


class TestForwardModelRewrite:
    def test_default_keeps_op_level_lists(self):
        model = models.get_model("Qwen/Qwen3-0.6B", _model_config(), "vllm")
        assert model.forward_model == "op_level"
        assert len(model.context_ops) > 1
        assert not any(isinstance(op, FPMForwardOp) for op in model.context_ops)

    def test_fpm_rewrite_yields_exactly_one_op_per_phase(self):
        baseline = models.get_model("Qwen/Qwen3-0.6B", _model_config(), "vllm")
        expected_weights = float(sum(op.get_weights() for op in baseline.context_ops))

        model = models.get_model("Qwen/Qwen3-0.6B", _model_config(forward_model="fpm"), "vllm")
        assert model.forward_model == "fpm"
        assert [op._name for op in model.context_ops] == ["fpm_forward_prefill"]
        assert [op._name for op in model.generation_ops] == ["fpm_forward_decode"]
        assert all(isinstance(op, FPMForwardOp) for op in (*model.context_ops, *model.generation_ops))
        # Weight bytes captured from the original lists keep memory estimation intact.
        assert model.context_ops[0].get_weights() == pytest.approx(expected_weights)

    def test_fpm_spec_preserves_explicit_attention_backend_for_nested_ops(self):
        from aisimulate.sdk.engine import build_engine_spec_json
        from aisimulate.sdk.perf_database import get_database

        attention_backend = "trtllm_mha"
        model = models.get_model(
            "Qwen/Qwen3-0.6B",
            _model_config(forward_model="fpm", attention_backend=attention_backend),
            "sglang",
        )
        database = get_database("b200_sxm", "sglang", "0.5.14")

        spec = json.loads(
            build_engine_spec_json(
                model,
                model_path="Qwen/Qwen3-0.6B",
                system="b200_sxm",
                backend="sglang",
                backend_version="0.5.14",
                kv_block_size=None,
                systems_path=None,
                nextn=0,
                database=database,
            )
        )
        context_lane_order = next(
            op["ContextAttention"]["lane_order"]
            for op in spec["context_ops"][0]["FpmForward"]["sol_ops"]
            if "ContextAttention" in op
        )
        generation_lane_order = next(
            op["GenerationAttention"]["lane_order"]
            for op in spec["generation_ops"][0]["FpmForward"]["sol_ops"]
            if "GenerationAttention" in op
        )

        assert context_lane_order[0] == attention_backend
        assert generation_lane_order[0] == attention_backend

    def test_fpm_spec_carries_external_parquet_path(self):
        from aisimulate_core.sdk.engine import build_engine_spec_json

        model = models.get_model(
            "Qwen/Qwen3-0.6B",
            _model_config(forward_model="fpm"),
            "sglang",
        )
        external_path = "/artifacts/reviewed-fpm.parquet"

        spec = json.loads(
            build_engine_spec_json(
                model,
                model_path="Qwen/Qwen3-0.6B",
                system="b200_sxm",
                backend="sglang",
                backend_version="0.5.14",
                kv_block_size=None,
                systems_path=None,
                nextn=0,
                fpm_parquet_path=external_path,
            )
        )

        assert spec["engine"]["fpm_parquet_path"] == external_path

    def test_fpm_rejects_construction_without_sol_ops(self):
        # Legacy "exactly one of sol_fn/sol_ops" contract, minus the retired
        # half: omitting sol_ops keeps raising (main's ValueError), with the
        # message pointing at the surviving parameter.
        with pytest.raises(ValueError, match="provide sol_ops"):
            FPMForwardOp("prefill", _model_config(), MODEL_PATH, weight_bytes=1.0)

    def test_fpm_sol_fn_raises_targeted_migration_error(self):
        # The legacy sol_fn slot stays in the signature (so positional
        # weight_bytes/sol_ops callers keep their meaning) but a callback
        # cannot cross the compiled boundary: passing one must fail with
        # migration guidance, not silently rebind or get ignored.
        with pytest.raises(TypeError, match="sol_ops"):
            FPMForwardOp("prefill", _model_config(), MODEL_PATH, lambda **kwargs: 1.0)

    def test_fpm_legacy_positional_layout_preserved(self):
        # main exposed (phase, model_config, model_path, sol_fn=None,
        # weight_bytes=0.0, sol_ops=None): the 5th positional is
        # weight_bytes and the 6th is sol_ops.
        op = FPMForwardOp("prefill", _model_config(), MODEL_PATH, None, 123.0, [])
        assert op.get_weights() == 123.0
        assert op._sol_ops == []

    def test_fpm_rejects_unknown_phase(self):
        with pytest.raises(ValueError, match="unknown FPM phase"):
            FPMForwardOp("mixed", _model_config(), MODEL_PATH, weight_bytes=1.0, sol_ops=[])

    def test_unknown_forward_model_rejected(self):
        with pytest.raises(ValueError, match="Unknown forward_model"):
            models.get_model("Qwen/Qwen3-0.6B", _model_config(forward_model="banana"), "vllm")

    def test_encoder_model_rejected(self):
        cfg = _model_config(forward_model="fpm")
        with pytest.raises(NotImplementedError, match="encoder"):
            models.get_model("Qwen/Qwen3-VL-2B-Instruct", cfg, "vllm")

    def test_mtp_rejected(self):
        cfg = _model_config(forward_model="fpm", nextn=1)
        with pytest.raises(NotImplementedError, match="MTP"):
            models.get_model("Qwen/Qwen3-0.6B", cfg, "vllm")

    # -- Hybrid speculative shape (verify-on-FPM) --

    _EAGLE3_CONFIG: ClassVar[dict] = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "model_type": "llama",
        "num_hidden_layers": 1,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "draft_vocab_size": 32000,
        "sliding_window": None,
        "use_sliding_window": False,
    }

    def test_fpm_hybrid_ngram_sets_verify_width_without_draft_ops(self):
        from aisimulate_core.sdk.speculation import SpeculationConfig

        cfg = _model_config(
            forward_model="fpm",
            speculation=SpeculationConfig(kind="ngram", params={"num_speculative_tokens": 3}),
        )
        model = models.get_model("Qwen/Qwen3-0.6B", cfg, "vllm")
        # ngram drafts on the host: no draft ops, pure verify-width change.
        assert [op._name for op in model.generation_ops] == ["fpm_forward_decode"]
        assert model.generation_ops[0]._verify_width == 4
        assert model.context_ops[0]._verify_width == 1  # prefill untouched
        assert model._nextn == 3  # engine widening channel stays consistent

    def test_fpm_standalone_draft_keeps_native_ops(self):
        from aisimulate_core.sdk.speculation import SpeculationConfig

        cfg = _model_config(
            forward_model="fpm",
            speculation=SpeculationConfig(
                kind="draft_model", params={"num_speculative_tokens": 3}, draft_model_path="Qwen/Qwen3-0.6B"
            ),
        )
        model = models.get_model("Qwen/Qwen3-8B", cfg, "vllm")
        assert model.generation_ops[0]._verify_width == 4
        assert model.generation_ops[1:]
        assert all(not isinstance(op, FPMForwardOp) for op in model.generation_ops[1:])
        assert all(not isinstance(op, FPMForwardOp) for op in model.context_ops[1:])
        draft = model.spec_scheme._draft_model
        assert draft.config.forward_model == "op_level"
        assert draft._nextn == 0

    def test_fpm_hybrid_eagle3_keeps_draft_ops_op_level(self):
        from aisimulate_core.sdk.engine import _fpm_spec_dict
        from aisimulate_core.sdk.speculation import SpeculationConfig

        cfg = _model_config(
            forward_model="fpm",
            speculation=SpeculationConfig(
                kind="eagle3",
                params={"num_speculative_tokens": 3},
                draft_config=self._EAGLE3_CONFIG,
            ),
        )
        baseline = models.get_model("Qwen/Qwen3-0.6B", _model_config(), "vllm")
        target_weights = float(sum(op.get_weights() for op in baseline.context_ops))
        model = models.get_model("Qwen/Qwen3-0.6B", cfg, "vllm")

        # Phase lists LEAD with the whole-model op; draft ops follow op-level.
        assert isinstance(model.generation_ops[0], FPMForwardOp)
        gen_tail = model.generation_ops[1:]
        assert gen_tail and all(op._name.startswith("draft_") for op in gen_tail)
        assert model.generation_ops[0]._verify_width == model.verify_width == 4

        # The whole-model fold covers the TARGET only: sol_ops and the weight
        # inventory exclude draft ops (draft memory rides the scheme hooks).
        assert not any(op._name.startswith("draft_") for op in model.generation_ops[0]._sol_ops)
        assert model.context_ops[0].get_weights() == pytest.approx(target_weights)

        # Wire: verify_width rides the FpmForward opspec.
        spec = _fpm_spec_dict(model.generation_ops[0])
        assert spec["FpmForward"]["verify_width"] == 4


# ---------------------------------------------------------------------------
# Static + mixed-step integration through a real PerfDatabase/backend
# ---------------------------------------------------------------------------


@pytest.fixture()
def fpm_session(tmp_path):
    """A real PerfDatabase over a temp systems root holding ONLY fpm data,
    plus an fpm-mode model whose identity the rows are written to match."""
    systems_root = tmp_path / "systems"
    os.makedirs(systems_root, exist_ok=True)
    shutil.copy(os.path.join(_CORE_SYSTEMS, f"{SYSTEM}.yaml"), systems_root / f"{SYSTEM}.yaml")

    model = models.get_model("Qwen/Qwen3-0.6B", _model_config(forward_model="fpm"), BACKEND)
    identity = dict(zip(_CELL_MATCH_COLUMNS, model.context_ops[0]._match_identity, strict=True))

    isl, osl = 512, 2
    rows = [
        # static_ctx: batch=2 x isl new tokens, and the mixed ctx component (batch=1 x isl)
        _row("prefill", 2, 2 * isl, 0, 40.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, isl, 0, 22.0, model_path=model.model_path, identity=identity),
        # static_gen with osl=2 runs one decode step at s = isl+1.
        _row("decode", 2, 0, 2 * (isl + 1), 6.0, model_path=model.model_path, identity=identity),
        # mixed/genonly route through run_static(isl=isl+osl//2, osl=2), whose
        # decode step lands at s = isl + osl//2 + 1.
        _row("decode", 2, 0, 2 * (isl + osl // 2 + 1), 7.0, model_path=model.model_path, identity=identity),
        # Totals-coordinate rows for the mixed-step composition: the prefill
        # component queries (batch, chunk + gen_tokens, past_kv).
        _row("prefill", 1, isl + 2, 0, 23.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, 258, 0, 11.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, 258, 256, 13.0, model_path=model.model_path, identity=identity),
        # Prefill-only chunk averages used by the hybrid attribution probe.
        _row("prefill", 1, 256, 0, 10.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, 256, 256, 12.0, model_path=model.model_path, identity=identity),
        # CUDA-graph cliff pair at capture=2048 plus the eager plateau: the
        # regime is encoded in the data, the formula only addresses it.
        _row("prefill", 1, 2048, 0, 47.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, 2049, 0, 99.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, 4096, 0, 99.0, model_path=model.model_path, identity=identity),
        # Decode coverage for the cliff test (gen=8 at isl=2048, osl=2).
        _row("decode", 8, 0, 1026, 6.5, model_path=model.model_path, identity=identity),
        _row("decode", 8, 0, 16400, 9.5, model_path=model.model_path, identity=identity),
    ]
    # Separate selector lane makes swapped warning values observable.
    rows += [
        dict(row, fmha_quant_mode="fp8", cell_id=row["cell_id"] + "-fp8", latency_ms=row["latency_ms"] + 1.0)
        for row in rows
    ]
    # data_dir comes from the system yaml ("data/h200_sxm").
    data_dir = os.path.join(systems_root, "data", SYSTEM, BACKEND, VERSION)
    _write_pair(data_dir, rows)

    database = PerfDatabase(SYSTEM, BACKEND, VERSION, systems_root=str(systems_root))
    backend = get_backend(BACKEND)
    return model, database, backend, isl, osl


def test_afd_companion_packaged_fpm_selector_reaches_native_loader(fpm_session):
    _, database, _, isl, osl = fpm_session
    topology = AFDTopology(
        n_a_nodes=1,
        n_f_nodes=1,
        gpus_per_node=1,
        tp_a=1,
        a_batch_size=1,
        num_microbatches=1,
        phase="decode",
        combined_with_pd=True,
    )
    spec = ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode=topology.adapter_topology,
            backend=BACKEND,
            backend_version=VERSION,
            parallel_config={
                "prefill_tp": 1,
                "prefill_pp": 1,
                "prefill_attention_dp": 1,
                "prefill_moe_tp": 1,
                "prefill_moe_ep": 1,
            },
            prefill_engine_args={
                "max_num_batched_tokens": isl,
                "max_num_seqs": 1,
                "aic_model_path": "Qwen/Qwen3-0.6B",
                "aic_system": SYSTEM,
                "aic_forward_model": "fpm",
                "aic_fpm_fmha_dtype": "fp8",
                "systems_path": database.systems_root,
            },
            num_prefill_workers=1,
        ),
        workload={"isl": isl, "osl": osl},
        goal={"target": "throughput", "sla": None},
        concurrency=1,
    )

    timing = AICAFDCompanionPerformanceModel().measure(spec)

    # The data tree has a separate fp8 selector row with latency 22.0 + 1.0.
    assert timing.latency_ms == pytest.approx(23.0)
    assert timing.provenance["source"] == "aisimulate_core.sdk.rust_engine_step.RustForwardPassPerfModel"
    assert timing.provenance["fpm_fmha_dtype"] == "fp8"


class TestFPMStaticAndMixed:
    @pytest.mark.parametrize("ctx_tokens", [256, 512])
    @pytest.mark.parametrize("gen_requests", [0, 2])
    def test_public_mixed_hybrid_keeps_native_component_sources(self, fpm_session, ctx_tokens, gen_requests):
        from aisimulate.sdk.config import RuntimeConfig
        from aisimulate.sdk.inference_session import InferenceSession
        from aisimulate_core.sdk.operations.elementwise import ElementWise
        from aisimulate_core.sdk.rust_engine_step import _cached_engine_handle
        from aisimulate_core.sdk.speculation import SpeculationConfig
        from aisimulate_core.sdk.speculation.materialize import _fold_width
        from aisimulate_core.sdk.step_estimate import MixedStepInput

        baseline, database, backend, isl, osl = fpm_session
        model = models.get_model(
            baseline.model_path,
            _model_config(
                forward_model="fpm",
                speculation=SpeculationConfig(kind="ngram", params={"num_speculative_tokens": 3}),
            ),
            BACKEND,
        )
        # A synthetic materialized draft graph runs real native empirical ops
        # alongside the fixture's silicon-tagged FPM table components.
        model.context_ops.append(ElementWise("draft_context", 1.0, 4096, 4096, 0.8))
        generation = ElementWise("draft_generation", 1.0, 4096, 4096, 0.8)
        _fold_width(generation, 1, 4)
        model.generation_ops.append(generation)
        handle = _cached_engine_handle(model, database)
        native = handle._mixed_step_breakdown_per_op_with_metadata(ctx_tokens, gen_requests, isl, osl, 0)
        estimate = InferenceSession(model, database, backend).run_mixed(
            RuntimeConfig(isl=isl, osl=osl), MixedStepInput(ctx_tokens, gen_requests)
        )
        shared, context, decode = native
        assert {row[0] for row in shared} == {"fpm_forward_prefill", "draft_context"}
        assert {row[0] for row in decode} == ({"fpm_forward_decode", "draft_generation"} if gen_requests else set())
        rows = {row[0]: row for group in native for row in group}
        # Independent phase queries establish draft values, energy and source;
        # the reporting path must not absorb these into target FPM rows.
        expected_context = handle.evaluate_context_ops([1], batch_size=1, s=ctx_tokens)[0]
        assert rows["draft_context"][1:3] == pytest.approx(expected_context[1:3])
        assert rows["draft_context"][3] == expected_context[3] == "empirical"
        if gen_requests:
            expected_generation = handle.evaluate_generation_ops(
                [1], batch_size=gen_requests * 4, s=isl + osl // 2 + 1
            )[0]
            assert rows["draft_generation"][1:3] == pytest.approx(expected_generation[1:3])
            assert rows["draft_generation"][3] == expected_generation[3] == "empirical"
        assert rows["fpm_forward_prefill"][3] == "silicon"
        public_rows = {
            "generation_attention" if name == "fpm_forward_decode" else name: row for name, row in rows.items()
        }
        expected_latency = {name: pytest.approx(row[1]) for name, row in public_rows.items()}
        expected_latency.setdefault("generation_attention", 0.0)
        expected_latency["context_attention (scaled)"] = 0.0
        assert estimate.per_op_latency_ms == expected_latency
        expected_source = {name: row[3] for name, row in public_rows.items()}
        expected_source.setdefault("generation_attention", "silicon")
        expected_source["context_attention (scaled)"] = "silicon"
        assert estimate.per_op_source == expected_source
        for name, group in zip(("shared_non_attention", "context_attention", "decode_attention"), native, strict=True):
            assert estimate.component_latency_ms[name] == pytest.approx(sum(row[1] for row in group))
            assert estimate.component_energy_wms[name] == pytest.approx(sum(row[2] for row in group))
        assert estimate.latency_ms == pytest.approx(handle.mixed_step_latency(ctx_tokens, gen_requests, isl, osl, 0))
        assert estimate.latency_ms == pytest.approx(sum(row[1] for row in rows.values()))
        assert estimate.energy_wms == pytest.approx(sum(row[2] for row in rows.values()))

    def test_ngram_verify_width_reaches_native_fpm_query(self, fpm_session):
        from aisimulate.sdk.config import RuntimeConfig
        from aisimulate.sdk.inference_session import InferenceSession
        from aisimulate_core.sdk.speculation import SpeculationConfig

        baseline, database, backend, isl, osl = fpm_session
        model = models.get_model(
            baseline.model_path,
            _model_config(
                forward_model="fpm",
                speculation=SpeculationConfig(kind="ngram", params={"num_speculative_tokens": 3}),
            ),
            BACKEND,
        )
        summary = InferenceSession(model, database, backend).run_static(
            RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl), mode="static_gen"
        )
        # Two requests verify four queries each; shared KV remains 2 * 513.
        # The synthetic fixture has an exact (batch=8, KV=1026) row at 6.5 ms.
        assert summary.get_generation_latency_dict() == {"fpm_forward_decode": pytest.approx(6.5)}

    def test_static_ctx_uses_fpm_row(self, fpm_session):
        from aisimulate.sdk.config import RuntimeConfig
        from aisimulate.sdk.inference_session import InferenceSession

        model, database, backend, isl, osl = fpm_session
        session = InferenceSession(model, database, backend)
        summary = session.run_static(
            runtime_config=RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl),
            mode="static_ctx",
        )
        latency_dict = summary.get_context_latency_dict()
        assert list(latency_dict) == ["fpm_forward_prefill"]
        assert latency_dict["fpm_forward_prefill"] == pytest.approx(40.0)

    def test_static_gen_uses_fpm_row(self, fpm_session):
        from aisimulate.sdk.config import RuntimeConfig
        from aisimulate.sdk.inference_session import InferenceSession

        model, database, backend, isl, osl = fpm_session
        session = InferenceSession(model, database, backend)
        summary = session.run_static(
            runtime_config=RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl),
            mode="static_gen",
        )
        latency_dict = summary.get_generation_latency_dict()
        assert list(latency_dict) == ["fpm_forward_decode"]
        # osl=2 -> one decode step at s=isl+1, repeat_count 1.
        assert latency_dict["fpm_forward_decode"] == pytest.approx(6.0)

    def test_mixed_step_is_prefill_plus_marginal_decode(self, fpm_session):
        from aisimulate.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl)
        total, energy, per_op, per_src = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=isl, gen_tokens=2, isl=isl, osl=osl, prefix=0
        )
        # ctx component prices the step's SCHEDULED TOTAL: one whole prefill
        # (isl tokens) plus 2 decode riders -> totals (1, isl+2, 0) = 23.0.
        # gen component rides the prefill pass, so only its marginal counts:
        # full decode at s=isl+osl//2+1 (7.0) minus the pass baseline at the
        # KV-domain floor (the 2*(isl+1)=1026 row, 6.0) -> 1.0. The compiled
        # engine reports the decode marginal under the mixed breakdown's
        # uniform "generation_attention" component key.
        assert per_op["fpm_forward_prefill"] == pytest.approx(23.0)
        assert per_op["generation_attention"] == pytest.approx(1.0)
        assert total == pytest.approx(24.0)
        assert energy == 0.0
        assert set(per_src.values()) == {"silicon"}

    def test_mixed_step_total_crosses_the_graph_cliff(self, fpm_session):
        # Spec tests 1+2: the engine picks its regime from the step's TOTAL
        # scheduled tokens. ctx=2048 alone sits ON the capture boundary
        # (graph side, 47 ms); the same chunk with 8 decode riders crosses
        # it and must price on the eager plateau (99 ms).
        from aisimulate.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=2048, osl=osl)
        _, _, graph_ops, _ = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=2048, gen_tokens=0, isl=2048, osl=osl, prefix=0
        )
        assert graph_ops["fpm_forward_prefill"] == pytest.approx(47.0)
        _, _, eager_ops, _ = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=2048, gen_tokens=8, isl=2048, osl=osl, prefix=0
        )
        assert eager_ops["fpm_forward_prefill"] == pytest.approx(99.0)
        assert eager_ops["fpm_forward_prefill"] > 2 * graph_ops["fpm_forward_prefill"]

    def test_mixed_step_chunks_average_exact_coordinates(self, fpm_session):
        # Spec tests 3+4: a chunked request prices each chunk at its own
        # (chunk + gen, past_kv) coordinates — the exact rows (1, 258, 0)=11.0
        # and (1, 258, 256)=13.0 for ctx=256 of isl=512 — and the component is
        # their per-iteration average, identical to pricing the chunks
        # independently (no double billing, no averaging artifacts).
        from aisimulate.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl)
        total, _, per_op, _ = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=256, gen_tokens=2, isl=isl, osl=osl, prefix=0
        )
        assert per_op["fpm_forward_prefill"] == pytest.approx((11.0 + 13.0) / 2) == pytest.approx(12.0)
        assert total == pytest.approx(12.0 + 1.0)

    def test_mixed_step_gen_zero_prices_pure_chunk(self, fpm_session):
        # Spec test 5 (gen=0 degenerate): a pure-prefill step prices its own
        # totals with no decode marginal term.
        from aisimulate.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl)
        total, _, per_op, _ = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=isl, gen_tokens=0, isl=isl, osl=osl, prefix=0
        )
        assert per_op == {
            "fpm_forward_prefill": pytest.approx(22.0),
            # The mixed breakdown's uniform component keys are always present
            # (zero when the pass contributed nothing).
            "context_attention (scaled)": 0,
            "generation_attention": 0,
        }
        assert total == pytest.approx(22.0)

    def test_genonly_step_keeps_full_decode_pass(self, fpm_session):
        # With no prefill work in the step there is no pass to ride on: the
        # decode component must keep its full standalone latency. A step
        # without prefill work is a GENONLY step now — `MixedStepInput`
        # requires context_tokens > 0, so the gen-only contract lives behind
        # `_get_genonly_step_latency` (the mixed entry raises instead of
        # silently rerouting).
        from aisimulate.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl)
        total, _energy, per_op, _src, _fallbacks = backend._get_genonly_step_latency(
            model, database, runtime_config, gen_tokens=2, isl=isl, osl=osl
        )
        assert per_op["fpm_forward_decode"] == pytest.approx(7.0)
        assert total == pytest.approx(7.0)
        with pytest.raises(ValueError, match="context_tokens must be positive"):
            backend._get_mix_step_latency(
                model, database, runtime_config, ctx_tokens=0, gen_tokens=2, isl=isl, osl=osl, prefix=0
            )

    def test_genonly_step_works_with_single_op(self, fpm_session):
        from aisimulate.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl)
        total, energy, per_op, _, _fallbacks = backend._get_genonly_step_latency(
            model, database, runtime_config, gen_tokens=2, isl=isl, osl=osl
        )
        assert per_op["fpm_forward_decode"] == pytest.approx(7.0)
        assert total == pytest.approx(7.0)


def test_explicit_selector_emits_matched_cell_warning_once(fpm_session, capfd):
    from aisimulate_core.sdk.rust_engine_step import _cached_engine_handle

    baseline, database, _backend, _isl, _osl = fpm_session
    selected = models.get_model(
        baseline.model_path,
        _model_config(forward_model="fpm", fpm_fmha_quant_mode=common.FMHAQuantMode.fp8),
        BACKEND,
    )
    original = baseline.config.fmha_quant_mode.name
    selector = common.FMHAQuantMode.fp8.name
    assert original != selector
    capfd.readouterr()
    handle = _cached_engine_handle(selected, database)
    first = handle.evaluate_context_ops([0], batch_size=1, s=512)
    warning = capfd.readouterr().err
    assert first[0][1] == 23.0
    assert "WARNING: FPM table FMHA selector" in warning
    assert f'original_model_mode="{original}"' in warning
    assert f'selector="{selector}"' in warning
    assert "matched_cell_ids=" in warning
    assert "fpm-test-prefill-fp8" in warning and "fpm-test-decode-fp8" in warning
    assert "does not independently verify runtime attention precision" in warning
    assert handle.evaluate_context_ops([0], batch_size=1, s=512) == first
    assert "FPM table FMHA selector" not in capfd.readouterr().err


@pytest.mark.parametrize(
    "forward_model,path", [(None, "/missing/fpm.parquet"), ("op_level", "/missing/fpm.parquet"), ("fpm", "")]
)
@pytest.mark.parametrize("compile_fn", [compile_engine, EngineHandle.compile])
def test_compile_rejects_invalid_external_fpm_path(compile_fn, forward_model, path):
    with pytest.raises(ValueError, match="fpm_parquet_path"):
        compile_fn(
            "Qwen/Qwen3-0.6B",
            SYSTEM,
            BACKEND,
            backend_version=VERSION,
            forward_model=forward_model,
            fpm_parquet_path=path,
        )


@pytest.fixture
def external_fpm_config(tmp_path, monkeypatch):
    # Synthetic exact anchors: one 512-token prefill costs 22 ms, and the
    # decode anchors around that prompt cost 6 ms. No SOL/op data exists.
    systems_root = tmp_path / "systems"
    systems_root.mkdir()
    shutil.copy(Path(_CORE_SYSTEMS) / f"{SYSTEM}.yaml", systems_root / f"{SYSTEM}.yaml")
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(systems_root))
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    model = models.get_model("Qwen/Qwen3-0.6B", _model_config(forward_model="fpm"), BACKEND)
    identity = dict(zip(_CELL_MATCH_COLUMNS, model.context_ops[0]._match_identity, strict=True))
    rows = [
        _row("prefill", 1, 512, 0, 22.0, model_path=model.model_path, identity=identity),
        *[
            _row("decode", 1, 0, kv, 6.0, model_path=model.model_path, identity=identity)
            for kv in (511, 512, 513, 1024)
        ],
    ]
    parquet = Path(_write_pair(str(tmp_path / "external"), rows))
    external = parquet.with_name("reviewed-fpm.parquet")
    parquet.rename(external)
    parquet.with_suffix(".metadata.json").rename(external.with_suffix(".metadata.json"))
    config = CorePredictionConfig.model_validate(
        yaml.safe_load(f"""
engine:
  model: Qwen/Qwen3-0.6B
  hardware: {SYSTEM}
  backend: {BACKEND}
  backend_version: {VERSION}
  context_length: 1024
  workers:
    aggregated:
      kv_cache:
        prefix_caching: false
        capacity: {{type: fixed, blocks: 128}}
      timing:
        type: default
        forward_model: fpm
        fpm_parquet_path: {external}
traffic:
  source: {{type: synthetic, input_tokens: 512, output_tokens: 2}}
  load: {{type: concurrency, concurrency: 1}}
  stop: {{requests: 1}}
""")
    )
    return config, systems_root


@pytest.mark.parametrize("canonical", [False, True])
def test_external_fpm_pair_drives_yaml_replay_without_backend_data(external_fpm_config, canonical):
    config, systems_root = external_fpm_config
    if canonical:
        raw = config.model_dump(mode="json", exclude_none=True)
        timing = raw["engine"]["workers"]["aggregated"]["timing"]
        timing["estimator_config"] = {"fpm_interpolation": {"fpm_parquet_path": timing.pop("fpm_parquet_path")}}
        config = CorePredictionConfig.model_validate(yaml.safe_load(yaml.safe_dump(raw)))
    report = (
        EngineReplayRunnerFactory()
        .create(0)
        .run(
            prediction_to_replay_spec(config),
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True),
        )
    )
    assert not (systems_root / "data").exists()
    assert report.metrics["completed_requests"] == 1
    record = report.metadata["native_report"]["per_request"][0]
    # The final prefill forward produces the first token; only later tokens decode.
    assert record["first_token_ms"] - record["arrival_time_ms"] == pytest.approx(22.0)
    assert record["last_token_ms"] - record["first_token_ms"] == pytest.approx(6.0)


@pytest.mark.parametrize("query_before_chdir", [False, True], ids=["lazy_load", "live_cache"])
@pytest.mark.parametrize("api", ["legacy", "canonical", "sweeper"])
def test_external_fpm_relative_path_is_bound_at_engine_construction(
    external_fpm_config, tmp_path, monkeypatch, query_before_chdir, api
):
    config, systems_root = external_fpm_config
    original = Path(config.engine.workers.aggregated.timing.fpm_parquet_path)
    rows = pq.read_table(original).to_pylist()
    first_pair = Path(_write_pair(str(tmp_path / "first"), rows))
    for row in rows:
        if row["workload_kind"] == "prefill":
            row["latency_ms"] = 99.0
    second_pair = Path(_write_pair(str(tmp_path / "second"), rows))
    from aisimulate.sweeper.config import SearchSpace
    from aisimulate.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver

    resolver = ForwardPassEstimatorResolver(SearchSpace(model_name="Qwen/Qwen3-0.6B", hardware_sku=SYSTEM))

    def compile_relative():
        if api != "legacy":
            request = ForwardPassPerfModelConfig(
                model="Qwen/Qwen3-0.6B",
                system=SYSTEM,
                backend=BACKEND,
                backend_version=VERSION,
                worker_type="aggregated",
                estimation_mode="fpm_interpolation",
                estimator_config={"fpm_interpolation": {"fpm_parquet_path": first_pair.name}},
                systems_paths=(str(systems_root),),
            )
            if api == "sweeper":
                request = resolver._resolve(request, "agg").config
            return RustForwardPassPerfModel.best_available(request)
        return EngineHandle.compile(
            "Qwen/Qwen3-0.6B",
            SYSTEM,
            BACKEND,
            backend_version=VERSION,
            forward_model="fpm",
            fpm_parquet_path=first_pair.name,
            systems_path=str(systems_root),
        )

    def prefill(engine):
        if api != "legacy":
            return engine.static_phase_latency(batch_size=1, input_tokens=512, output_tokens=4, prefill=True)
        return engine.predict_prefill_latency(1, 512)

    monkeypatch.chdir(first_pair.parent)
    first = compile_relative()
    if query_before_chdir:
        assert prefill(first) == pytest.approx(22.0)
    monkeypatch.chdir(second_pair.parent)
    second = compile_relative()
    assert prefill(first) == pytest.approx(22.0)
    assert prefill(second) == pytest.approx(99.0)
    if api != "legacy":
        resolved = first.diagnostics()["provenance"]["config"]
        assert resolved["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] == str(first_pair)
        assert first.static_phase_latency(
            batch_size=1, input_tokens=512, output_tokens=4, prefill=False
        ) == pytest.approx(18.0)
        first.close()
        second.close()


def test_canonical_external_fpm_rejects_empty_path_before_fallback():
    with pytest.raises(ValueError, match="fpm_parquet_path cannot be empty"):
        RustForwardPassPerfModel.best_available(
            ForwardPassPerfModelConfig(
                model="Qwen/Qwen3-0.6B",
                system=SYSTEM,
                backend=BACKEND,
                worker_type="aggregated",
                fallback_policy="allow",
                estimator_config={"fpm_interpolation": {"fpm_parquet_path": ""}},
            )
        )


def test_canonical_config_retains_inactive_fpm_controls_when_reloaded(tmp_path):
    path = str(tmp_path / "unselected.parquet")
    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model="Qwen/Qwen3-0.6B",
            system=SYSTEM,
            backend=BACKEND,
            worker_type="aggregated",
            estimation_mode="fpm_regression",
            estimator_config={"fpm_interpolation": {"fpm_parquet_path": path}},
        )
    )
    resolved = model.diagnostics()["provenance"]["config"]
    reloaded = RustForwardPassPerfModel.best_available(resolved)
    assert reloaded.diagnostics()["provenance"]["config"] == resolved
    assert resolved["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] == path
    model.close()
    reloaded.close()


@pytest.mark.parametrize(
    ("phase", "companion_role", "latency"), [("decode", "prefill", 22.0), ("prefill", "decode", 6.0)]
)
@pytest.mark.parametrize("output_tokens", [2, 4])
@pytest.mark.parametrize("identity_fields", ["default", "prefixed", "plain"])
def test_external_fpm_pair_drives_afd_companion_replay(
    external_fpm_config, phase, companion_role, latency, output_tokens, identity_fields, tmp_path, monkeypatch
):
    config, systems_root = external_fpm_config
    raw = config.model_dump(mode="json", exclude_none=True)
    raw["traffic"]["source"]["output_tokens"] = output_tokens
    engine = raw["engine"]
    engine["mode"] = "afd"
    engine["afd"] = {
        "phase": phase,
        "combined_with_pd": True,
        "n_a_nodes": 1,
        "n_f_nodes": 1,
        "tp_a": 1,
        "a_batch_size": 1,
        "num_microbatches": 1,
    }
    worker = engine["workers"].pop("aggregated")
    worker["scheduler"] = {"max_sequences": 1, "max_batched_tokens": 512}
    engine["workers"] = {companion_role: worker}

    class AFDPerformanceFixture:
        def measure(self, request):
            return (
                AFDLayerTimes(
                    phase=request.topology.phase.value,
                    attention_ms=1.0,
                    ffn_ms=2.0,
                    a_to_f_ms=0.1,
                    f_to_a_ms=0.1,
                    num_layers=2,
                    provenance={"provider": "test"},
                ),
            )

    spec = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(raw), afd_performance_model=AFDPerformanceFixture()
    )
    expected_path = worker["timing"]["fpm_parquet_path"]
    if identity_fields != "default":
        # Neither model-default precision nor the built-in systems directory
        # can satisfy this cell. Exercise the real companion compiler/loader.
        custom_system = "custom_fpm_companion"
        shutil.copy(systems_root / f"{SYSTEM}.yaml", systems_root / f"{custom_system}.yaml")
        monkeypatch.delenv("AICONFIGURATOR_SYSTEMS_PATH")
        model = models.get_model(
            engine["model"],
            _model_config(
                forward_model="fpm",
                gemm_quant_mode=common.GEMMQuantMode.fp8,
                moe_quant_mode=common.MoEQuantMode.fp8,
                fmha_quant_mode=common.FMHAQuantMode.fp8,
                fpm_fmha_quant_mode=common.FMHAQuantMode.bfloat16,
                kvcache_quant_mode=common.KVCacheQuantMode.fp8,
                comm_quant_mode=common.CommQuantMode.fp8,
            ),
            BACKEND,
        )
        identity = dict(zip(_CELL_MATCH_COLUMNS, model.context_ops[0]._match_identity, strict=True))
        rows = [
            _row(
                row["workload_kind"],
                row["batch_size"],
                row["total_prefill_tokens"],
                row["total_kv_read_tokens"],
                row["latency_ms"],
                model_path=model.model_path,
                identity=identity,
            )
            | {"system": custom_system}
            for row in pq.read_table(expected_path).to_pylist()
        ]
        expected_path = _write_pair(str(tmp_path / "custom-pair"), rows, sidecar_overrides={"system": custom_system})
        args = getattr(spec.backend_deployment, f"{companion_role}_engine_args")
        args.update(aic_system=custom_system, systems_path=str(systems_root), aic_fpm_parquet_path=expected_path)
        prefix = "aic_" if identity_fields == "prefixed" else ""
        args.update({f"{prefix}{field}_dtype": "fp8" for field in ("gemm", "moe", "fmha", "kv_cache", "comm")})
        args[f"{prefix}fpm_fmha_dtype"] = "bfloat16"
        args.update(
            {
                "aic_pp_size": 1,
                f"{prefix}moe_tp_size": 1,
                f"{prefix}moe_ep_size": 1,
            }
        )
        if identity_fields == "plain":
            args["backend_version"] = args.pop("aic_backend_version")
            args["forward_model"] = args.pop("aic_forward_model")
            args["fpm_parquet_path"] = args.pop("aic_fpm_parquet_path")
    report = EngineReplayRunnerFactory().create(0).run(spec)
    assert not (systems_root / "data").exists()
    assert report.metrics["completed_requests"] == 1
    companion = report.metadata["afd_replay"]["companion"]
    assert companion["source"] == "aisimulate_core.sdk.rust_engine_step.RustForwardPassPerfModel"
    assert companion["fpm_parquet_path"] == expected_path
    assert report.metrics["mean_ttft_ms" if companion_role == "prefill" else "mean_tpot_ms"] == pytest.approx(latency)


def test_fpm_detail_distinguishes_memory_budget_from_runtime_capacity(external_fpm_config, tmp_path, capsys):
    config, _systems_root = external_fpm_config
    path = tmp_path / "fpm.yaml"
    raw = config.model_dump(mode="json", exclude_none=True)
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["capacity"] = {"type": "default"}
    path.write_text(yaml.safe_dump(raw))
    assert (
        main(["predict", "-c", str(path), "--detail", "all", "--format", "json", "--output-dir", str(tmp_path / "out")])
        == 0
    )
    sections = json.loads(capsys.readouterr().out)["details"]["sections"]
    memory = sections["memory"]["roles"]["aggregated"]
    assert memory["scope"] == "capacity_estimate_per_rank"
    assert memory["stage"] == "before_native_capacity_adjustments"
    assert memory["estimated_num_gpu_blocks"] > 0
    assert "num_gpu_blocks" not in memory
    assert set(sections) == {"summary", "memory", "time", "energy", "source"}
    assert sections["time"]["diagnostics"]["status"] == "unavailable"
    assert sections["source"]["status"] == "unavailable"
    assert "whole-model FPM" in sections["source"]["unavailable_reason"]
    assert sections["time"]["serving_metrics"]["mean_ttft_ms"] > 0


def test_fpm_selector_allows_fallback_to_untrained_regression(tmp_path):
    systems = tmp_path / "systems"
    systems.mkdir()
    shutil.copy(Path(_CORE_SYSTEMS) / f"{SYSTEM}.yaml", systems / f"{SYSTEM}.yaml")
    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model="Qwen/Qwen3-0.6B",
            system=SYSTEM,
            backend=BACKEND,
            backend_version=VERSION,
            worker_type="aggregated",
            systems_paths=(str(systems),),
            estimation_mode="fpm_interpolation",
            fallback_policy="allow",
            fpm_fmha_quant_mode="fp8",
        )
    )
    diagnostics = model.diagnostics()
    assert diagnostics["provenance"]["selected_estimation_mode"] == "fpm_regression"
    model.close()


def test_canonical_config_preserves_positional_quantization_fields():
    config = ForwardPassPerfModelConfig(
        "model",
        "system",
        "sglang",
        "aggregated",
        "version",
        4,
        1,
        1,
        4,
        1,
        "fp8",
        "fp8",
        "bfloat16",
        "fp8",
        "bfloat16",
        0,
        fpm_fmha_quant_mode="fp8",
    )
    assert config.kvcache_quant_mode == "fp8"
    assert config.comm_quant_mode == "bfloat16"
    assert config.nextn == 0
    assert config.fpm_fmha_quant_mode == "fp8"
