# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
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


def test_real_v41_fpm_selector_roundtrip_and_frozen_interpolation_are_unchanged():
    """126 exact cells and 38 geometry-only queries; no heldout timing is read."""
    import pyarrow.parquet as pq

    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    root = Path(__file__).resolve().parents[5]
    data = root / "data/experimental/deepseek-v41"
    calibration = data / "gb200-fpm/calibration-v1"
    legacy = json.loads((calibration / "prediction-config.json").read_text())
    legacy["systems_path"] = str(calibration / "systems")
    selected = dict(legacy)
    selected["fpm_fmha_dtype"] = selected.pop("activation_dtype")
    old = RustForwardPassPerfModel.from_native(legacy)
    new = RustForwardPassPerfModel.from_native(selected)
    native_rows = pq.read_table(next((calibration / "systems").rglob("fpm_forward_perf.parquet"))).to_pylist()
    observed = {
        (r["workload_kind"], r["batch_size"], r["total_prefill_tokens"], r["total_kv_read_tokens"]): r["latency_ms"]
        for r in native_rows
    }
    assert len(observed) == 126
    count = 0
    for role in ("calibration", "heldout"):
        manifest = json.loads((data / f"verification-plan/{role}.json").read_text())
        for phase in ("prefill", "decode"):
            for point in manifest[phase]:
                b, q, k = point["batch_size"], point.get("total_prefill_tokens", 0), point["total_kv_read_tokens"]
                scheduled = dict(
                    num_prefill_requests=b if phase == "prefill" else 0,
                    num_decode_requests=b if phase == "decode" else 0,
                    sum_prefill_tokens=q,
                    sum_prefill_kv_tokens=k if phase == "prefill" else 0,
                    sum_decode_kv_tokens=k if phase == "decode" else 0,
                    var_prefill_length=0.0,
                    var_decode_kv_tokens=0.0,
                )
                fpm = dict(version=1, wall_time=1.0, scheduled_requests=scheduled)
                prediction = new.estimate_forward_pass_time_ms(fpm)
                assert prediction == old.estimate_forward_pass_time_ms(fpm)
                if role == "calibration":
                    assert prediction == observed[(phase, b, q, k)]
                count += 1
    assert count == 164
    with pytest.raises(ValueError, match="requires forward_model='fpm'"):
        RustForwardPassPerfModel.from_native(selected | {"forward_model": "op_level"})


def test_real_v41_fpm_rejects_ambiguous_aggregates_but_keeps_identifiable_inputs():
    """The public whole-forward path must reject before a balanced lookup."""
    import pyarrow.parquet as pq

    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    root = Path(__file__).resolve().parents[5]
    packet = root / "data/experimental/deepseek-v41/gb300-fpm/on-union-v3-tracewait2"
    config = json.loads((packet / "prediction-config.json").read_text())
    config["systems_path"] = str(packet / "systems")
    predictor = RustForwardPassPerfModel.from_native(config)

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

    rows = pq.read_table(next((packet / "systems").rglob("fpm_forward_perf.parquet"))).to_pylist()
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
