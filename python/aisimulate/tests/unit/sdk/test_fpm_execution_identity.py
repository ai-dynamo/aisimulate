# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from aiconfigurator_core.sdk.fpm_identity import LEGACY_EXECUTION_IDENTITY, execution_identity
from aiconfigurator_core.sdk.utils import _attach_inferred_quant_fields, _get_model_config_path

pytestmark = pytest.mark.unit


def config():
    return json.loads((Path(_get_model_config_path()) / "deepseek-ai--DeepSeek-V4.1-Flash_config.json").read_text())


def test_v41_config_and_execution_cannot_borrow_a_table():
    raw = config()
    off = execution_identity(raw, engram_cpu_offload=False, input_modality="text")
    on = execution_identity(raw, decoder_replay=True, backend="sglang", engram_cpu_offload=False, input_modality="text")
    assert off[1:] == ("full", "hbm_tp_sharded", "text")
    assert on[0] == off[0] and on[1] == "decoder_bounded"
    altered = copy.deepcopy(raw)
    altered["text_config"]["kv_source_layer_ids"] = [2, 8, 14]
    assert execution_identity(altered, engram_cpu_offload=False, input_modality="text")[0] != off[0]
    assert (
        execution_identity(
            _attach_inferred_quant_fields(copy.deepcopy(raw)), engram_cpu_offload=False, input_modality="text"
        )
        == off
    )
    assert raw == config()
    with pytest.raises(NotImplementedError, match="not verified for vllm"):
        execution_identity(raw, decoder_replay=True, engram_cpu_offload=False, input_modality="text")


def test_existing_model_identity_stays_legacy():
    assert execution_identity({"architectures": ["LlamaForCausalLM"]}) == LEGACY_EXECUTION_IDENTITY


def test_native_v41_requires_measured_execution_and_text_evidence():
    from types import SimpleNamespace

    from collector.fpm_forward.native_artifact import _validate_execution_provenance

    identity = execution_identity(config(), engram_cpu_offload=False, input_modality="text")
    cell = SimpleNamespace(execution_identity=identity, input_text_sha256="a" * 64)
    fields = ("model_config_sha256", "execution_profile", "engram_residency", "input_modality")
    payload = {
        "execution_mode": "eager",
        "execution_identity": dict(zip(fields, identity, strict=True)),
        "input_provenance": {
            "source": "tokenizer_text",
            "text_sha256": "a" * 64,
            "token_ids_sha256": "b" * 64,
            "tokenizer_revision": "pinned",
            "token_count": 100,
            "unique_token_count": 20,
        },
    }
    assert _validate_execution_provenance(cell, payload, Path("artifact")) == payload["input_provenance"]
    for mode in (None, "cuda_graph"):
        corrupt = copy.deepcopy(payload)
        corrupt["execution_mode"] = mode
        with pytest.raises(ValueError, match="verified eager execution"):
            _validate_execution_provenance(cell, corrupt, Path("artifact"))
    corrupt = copy.deepcopy(payload)
    corrupt["execution_identity"]["execution_profile"] = "decoder_bounded"
    with pytest.raises(ValueError, match="execution identity"):
        _validate_execution_provenance(cell, corrupt, Path("artifact"))
    corrupt = copy.deepcopy(payload)
    corrupt["input_provenance"]["unique_token_count"] = 1
    with pytest.raises(ValueError, match="multiple tokenizer-generated"):
        _validate_execution_provenance(cell, corrupt, Path("artifact"))
    assert (
        _validate_execution_provenance(
            SimpleNamespace(execution_identity=LEGACY_EXECUTION_IDENTITY, input_text_sha256=""), {}, Path("legacy")
        )
        is None
    )


@pytest.mark.parametrize("replay,backend", [(False, "vllm"), (True, "sglang")])
def test_v41_fpm_wrap_retains_resident_inventory_and_serialized_stages(replay, backend):
    from aiconfigurator_core.sdk.config import ModelConfig
    from aiconfigurator_core.sdk.deepseek_v41 import MODEL_PATH
    from aiconfigurator_core.sdk.engine import build_ops_json
    from aiconfigurator_core.sdk.models import get_model

    kwargs = dict(tp_size=4, pp_size=1, attention_dp_size=1, moe_tp_size=4, moe_ep_size=1, decoder_replay=replay)
    granular = get_model(MODEL_PATH, ModelConfig(**kwargs), backend)
    wrapped = get_model(MODEL_PATH, ModelConfig(**kwargs, forward_model="fpm"), backend)
    for ops in (wrapped.context_ops, wrapped.generation_ops):
        assert len(ops) == 1
        assert ops[0].get_weights() == granular.get_resident_weights_bytes()
        assert ops[0]._match_identity[-4:] == execution_identity(
            config(), decoder_replay=replay, backend=backend, engram_cpu_offload=False, input_modality="text"
        )
        native = json.loads(build_ops_json(ops))[0]["FpmForward"]
        assert len(native["match_identity"]) == 19
        stages = [item["Dsv41Stage"] for item in native["sol_ops"] if "Dsv41Stage" in item]
        assert len(stages) == 40
        assert all(stage["decoder_replay"] == replay for stage in stages)
        assert '"Dsv41Linear"' in json.dumps(native["sol_ops"])


def test_table_selector_preserves_checkpoint_graph_residency_and_cache_identity():
    from types import SimpleNamespace

    from aiconfigurator_core.sdk.common import FMHAQuantMode
    from aiconfigurator_core.sdk.config import ModelConfig
    from aiconfigurator_core.sdk.deepseek_v41 import MODEL_PATH
    from aiconfigurator_core.sdk.engine import build_ops_json
    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.rust_engine_step import _engine_config_json

    kwargs = dict(tp_size=4, pp_size=1, attention_dp_size=1, moe_tp_size=4, moe_ep_size=1)
    granular = get_model(MODEL_PATH, ModelConfig(**kwargs), "vllm")
    native = get_model(MODEL_PATH, ModelConfig(**kwargs, forward_model="fpm"), "vllm")
    selected = get_model(
        MODEL_PATH,
        ModelConfig(**kwargs, forward_model="fpm", fpm_fmha_quant_mode=FMHAQuantMode.fp8),
        "vllm",
    )
    overridden = get_model(
        MODEL_PATH, ModelConfig(**kwargs, forward_model="fpm", fmha_quant_mode=FMHAQuantMode.fp8), "vllm"
    )
    assert native.config.fmha_quant_mode == selected.config.fmha_quant_mode == FMHAQuantMode.bfloat16
    assert overridden.config.fmha_quant_mode == FMHAQuantMode.fp8
    for phase in ("context_ops", "generation_ops"):
        direct = json.loads(build_ops_json(getattr(granular, phase)))
        baseline = json.loads(build_ops_json(getattr(native, phase)))[0]["FpmForward"]
        query = json.loads(build_ops_json(getattr(selected, phase)))[0]["FpmForward"]
        arithmetic = json.loads(build_ops_json(getattr(overridden, phase)))[0]["FpmForward"]
        assert query["sol_ops"] == baseline["sol_ops"] == direct
        assert query["sol_ops"] != arithmetic["sol_ops"]
        assert query["match_identity"][2] == arithmetic["match_identity"][2] == "fp8"
        assert baseline["match_identity"][2] == "bfloat16"
        assert query["original_fmha_quant_mode"] == "bfloat16"
        assert baseline["original_fmha_quant_mode"] is None
    for model in (native, selected):
        assert model.get_resident_weights_bytes() == granular.get_resident_weights_bytes()
        assert model.get_additional_activation_bytes(512) == granular.get_additional_activation_bytes(512)
        assert model.get_kvcache_bytes_per_sequence(2048) == granular.get_kvcache_bytes_per_sequence(2048)
    # Same arithmetic with a different table selector must not reuse a compiled
    # handle that selected another FPM identity.
    database = SimpleNamespace(system="gb200", backend="vllm", version="test")
    assert _engine_config_json(native, database) != _engine_config_json(selected, database)


@pytest.mark.parametrize("forward_model", [None, "op_level"])
def test_table_selector_requires_fpm_before_model_resolution(forward_model):
    from aiconfigurator_core.sdk.common import FMHAQuantMode
    from aiconfigurator_core.sdk.config import ModelConfig
    from aiconfigurator_core.sdk.models import get_model

    with pytest.raises(ValueError, match="requires forward_model='fpm'"):
        get_model(
            "deliberately-unresolved-model",
            ModelConfig(forward_model=forward_model, fpm_fmha_quant_mode=FMHAQuantMode.fp8),
            "vllm",
        )


def _synthetic_v41_fpm(tmp_path, *, replay=False):
    """Exercise the native loader with four invented timings, without campaign files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aiconfigurator_core.sdk.deepseek_v41 import MODEL_PATH

    backend = "sglang" if replay else "vllm"
    version = "test-v41-fpm"
    systems = tmp_path / "systems"
    data = systems / "data" / "gb200" / backend / version
    data.mkdir(parents=True)
    packaged_systems = Path(_get_model_config_path()).parent / "systems"
    shutil.copyfile(packaged_systems / "gb200.yaml", systems / "gb200.yaml")
    fields = ("model_config_sha256", "execution_profile", "engram_residency", "input_modality")
    identity = execution_identity(
        config(), decoder_replay=replay, backend=backend, engram_cpu_offload=False, input_modality="text"
    )
    rows = []
    for phase, new, kv, latency in (
        ("prefill", 32, 128, 10.0),
        ("prefill", 64, 128, 12.0),
        ("decode", 0, 128, 6.0),
        ("decode", 0, 256, 8.0),
    ):
        rows.append(
            dict(
                cell_id=f"synthetic-{phase}",
                model_path=MODEL_PATH,
                system="gb200",
                backend=backend,
                backend_version=version,
                weight_quantization="fp8_block",
                gemm_quant_mode="fp8_block",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm" if backend == "sglang" else "w4a8_mxfp4_mxfp8",
                fmha_quant_mode="fp8",
                comm_quant_mode="half",
                kv_cache_dtype="fp8",
                tp=4,
                pp=1,
                dp=1,
                moe_tp=4,
                moe_ep=1,
                cp=1,
                moe_backend="auto",
                attention_backend="auto",
                enable_wideep=False,
                enable_eplb=False,
                workload_kind=phase,
                batch_size=1,
                total_prefill_tokens=new,
                total_kv_read_tokens=kv,
                partition_policy="balanced_v1",
                kv_seed_regime="real_kv",
                latency_ms=latency,
            )
            | dict(zip(fields, identity, strict=True))
        )
    parquet = data / "fpm_forward_perf.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    metadata = dict(
        schema_name="aic_fpm_forward_perf",
        schema_version=7,
        coordinate_system="iteration_totals_balanced_v1",
        measurement_policy="dynamo_native_single_sample_v1",
        row_count=len(rows),
        parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
        system="gb200",
        backend=backend,
        backend_version=version,
    )
    parquet.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    native = dict(
        schema_version=1,
        model_name=MODEL_PATH,
        system_name="gb200",
        backend=backend,
        backend_version=version,
        systems_path=str(systems),
        enable_shared_layer=False,
        strict_provenance=True,
        tp_size=4,
        pp_size=1,
        moe_tp_size=4,
        moe_ep_size=1,
        attention_dp_size=1,
        database_mode="SILICON",
        decoder_replay=replay,
        forward_model="fpm",
        fpm_fmha_dtype="fp8",
    )
    return native, rows


def test_v41_fpm_selector_roundtrip_and_interpolation(tmp_path):
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    selected, rows = _synthetic_v41_fpm(tmp_path)
    legacy = dict(selected)
    legacy["activation_dtype"] = legacy.pop("fpm_fmha_dtype")
    old = RustForwardPassPerfModel.from_native(legacy)
    new = RustForwardPassPerfModel.from_native(selected)
    # Two exact endpoints and one interpolation point per phase. The fixture
    # values are arbitrary test data, not calibration or validation measurements.
    for phase in ("prefill", "decode"):
        exact_rows = [row for row in rows if row["workload_kind"] == phase]
        points = [(row["total_prefill_tokens"], row["total_kv_read_tokens"], row["latency_ms"]) for row in exact_rows]
        points.append((48, 128, None) if phase == "prefill" else (0, 192, None))
        for q, k, expected in points:
            scheduled = dict(
                num_prefill_requests=1 if phase == "prefill" else 0,
                num_decode_requests=1 if phase == "decode" else 0,
                sum_prefill_tokens=q,
                sum_prefill_kv_tokens=k if phase == "prefill" else 0,
                sum_decode_kv_tokens=k if phase == "decode" else 0,
                var_prefill_length=0.0,
                var_decode_kv_tokens=0.0,
            )
            fpm = dict(version=1, wall_time=1.0, scheduled_requests=scheduled)
            prediction = new.estimate_forward_pass_time_ms(fpm)
            assert prediction == old.estimate_forward_pass_time_ms(fpm)
            if expected is not None:
                assert prediction == expected
            else:
                assert exact_rows[0]["latency_ms"] < prediction < exact_rows[1]["latency_ms"]
    with pytest.raises(ValueError, match="requires forward_model='fpm'"):
        RustForwardPassPerfModel.from_native(selected | {"forward_model": "op_level"})


def test_v41_fpm_rejects_ambiguous_aggregates_but_keeps_identifiable_inputs(tmp_path):
    """The public whole-forward path must reject before a balanced lookup."""
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    native, rows = _synthetic_v41_fpm(tmp_path, replay=True)
    predictor = RustForwardPassPerfModel.from_native(native)

    # Equal complete prompts can have unequal current extends: (1, 1023)
    # and (1023, 1) in (new, prefix) coordinates. Prompt variance is zero.
    scheduled = dict(
        num_prefill_requests=2,
        sum_prefill_tokens=1024,
        sum_prefill_kv_tokens=1024,
        var_prefill_length=0.0,
        num_decode_requests=0,
        sum_decode_kv_tokens=0,
        var_decode_kv_tokens=0.0,
    )
    for decode_batch in (0, 1):
        metrics = dict(
            version=1,
            wall_time=1.0,
            scheduled_requests=scheduled
            | dict(num_decode_requests=decode_batch, sum_decode_kv_tokens=decode_batch * 129),
        )
        with pytest.raises(ValueError, match="multiple prefill requests"):
            predictor.estimate_forward_pass_time_ms(metrics)

    for phase in ("prefill", "decode"):
        row = next(row for row in rows if row["workload_kind"] == phase and row["batch_size"] == 1)
        batch, new, kv = row["batch_size"], row["total_prefill_tokens"], row["total_kv_read_tokens"]
        exact = dict(
            num_prefill_requests=batch if phase == "prefill" else 0,
            sum_prefill_tokens=new,
            sum_prefill_kv_tokens=kv if phase == "prefill" else 0,
            var_prefill_length=0.0,
            num_decode_requests=batch if phase == "decode" else 0,
            sum_decode_kv_tokens=kv if phase == "decode" else 0,
            var_decode_kv_tokens=0.0,
        )
        metrics = dict(version=1, wall_time=1.0, scheduled_requests=exact)
        assert predictor.estimate_forward_pass_time_ms(metrics) == row["latency_ms"]
        if phase == "decode":
            # Fully cached prefill metadata schedules no new prefill compute.
            exact.update(num_prefill_requests=2, sum_prefill_kv_tokens=1024)
            assert predictor.estimate_forward_pass_time_ms(metrics) == row["latency_ms"]


@pytest.mark.parametrize("offload", [None, True, 0, "false"])
def test_v41_identity_rejects_missing_or_unverified_residency(offload):
    with pytest.raises(ValueError, match="explicit engram_cpu_offload=False"):
        execution_identity(config(), engram_cpu_offload=offload, input_modality="text")


@pytest.mark.parametrize("modality", [None, "image", "multimodal", ""])
def test_v41_identity_rejects_missing_or_nontext_input(modality):
    with pytest.raises(ValueError, match="explicit input_modality='text'"):
        execution_identity(config(), engram_cpu_offload=False, input_modality=modality)
