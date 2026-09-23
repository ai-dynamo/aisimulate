# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

import aisimulate_core
from aisimulate.config import EnginePredictionConfig
from aisimulate.config.engine import TimingConfig
from aisimulate_core.sdk.config_builders import build_model_config
from aisimulate_core.sdk.engine import InvalidEngineConfigurationError, compile_engine
from aisimulate_core.sdk.fastafd_backend import apply_fastafd_moe_profile
from aisimulate_core.sdk.fastafd_profile import FASTAFD_OFFICIAL_REPOSITORY, FASTAFD_PROFILE_SCHEMA
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.models.helpers import resolve_dsv4_moe_arch

pytestmark = pytest.mark.unit


def _profile(tmp_path, *, system="b200_sxm", latency_ms=5.25, mtp_nextn=0):
    entry = {
        "key": {
            "model_path": "deepseek-ai/DeepSeek-V4-Flash",
            "system": system,
            "stage": "agg",
            "topology": "ep8",
            "logical_batch_per_source_rank": 8,
            "mtp_nextn": mtp_nextn,
            "microbatches": 1,
            "moe_layers": 43,
            "routed_topk": 6,
            "moe_precision": "w4a8_mxfp4_mxfp8",
            "moe_backend": "megamoe",
        },
        "model_profile": "deepseek_v4_flash_fp4",
        "latency_ms": latency_ms,
        "validation": {"stable": True, "correctness": True, "evidence": "test"},
        "measurement": {
            "scope": "complete_moe_stage",
            "method": "nsight-systems-cupti",
            "method_version": "2025.5.2",
            "statistic": "p50",
            "sample_count": 10,
            "procedure": "docs/profile-procedure.md",
            "procedure_sha256": "a" * 64,
            "raw_artifact": "raw/trace.nsys-rep",
            "raw_sha256": "b" * 64,
        },
    }
    path = tmp_path / "fastafd.json"
    path.write_text(
        json.dumps(
            {
                "schema": FASTAFD_PROFILE_SCHEMA,
                "lookup_policy": "exact-only",
                "source": {
                    "repository": FASTAFD_OFFICIAL_REPOSITORY,
                    "commit": "3c7161949310b6d59d6b4cf9bf997a4935c8113b",
                },
                "entries": [entry],
            }
        )
    )
    return path


def _compile(profile, *, nextn=0):
    return compile_engine(
        "deepseek-ai/DeepSeek-V4-Flash",
        "b200_sxm",
        "sglang",
        tp_size=1,
        attention_dp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
        nextn=nextn,
        forward_model="op_level",
        fastafd_profile_path=str(profile),
        fastafd_moe_backend="megamoe",
    )


def test_fastafd_stage_runs_end_to_end_and_matches_composition(tmp_path):
    engine = aisimulate_core.AicEngine.from_spec(_compile(_profile(tmp_path)))

    rows = engine.decode_step_per_op(8, 1024, 2)
    names = [name for name, *_ in rows]
    stage = next(row for row in rows if row[0] == "generation_fastafd_moe_stage")

    assert stage[1] == pytest.approx(5.25)
    assert names.index("generation_router_gemm") < names.index("generation_fastafd_moe_stage")
    assert "generation_moe_overlap" not in names
    assert engine.decode_step_latency(8, 1024, 2) == pytest.approx(sum(row[1] for row in rows))
    with pytest.raises(ValueError, match="no exact FastAFD MoE stage measurement for 7 tokens"):
        engine.decode_step_latency(7, 1024, 2)


def test_fastafd_profile_rejects_cross_system_use(tmp_path):
    with pytest.raises(InvalidEngineConfigurationError, match="no FastAFD AGG measurements match"):
        _compile(_profile(tmp_path, system="gb200"))


def test_compiled_spec_records_source_and_method(tmp_path, monkeypatch):
    import aisimulate_core.sdk.engine as core_engine

    captured = {}

    def capture(spec_json):
        captured.update(json.loads(spec_json))
        return b""

    monkeypatch.setattr(core_engine.aisimulate_core, "engine_spec_bincode_from_json", capture)
    _compile(_profile(tmp_path))

    extra = captured["engine"]["extra"]
    assert extra["source_repository"] == FASTAFD_OFFICIAL_REPOSITORY
    assert extra["source_commit"] == "3c7161949310b6d59d6b4cf9bf997a4935c8113b"
    assert extra["measurement_method"] == "nsight-systems-cupti"
    assert extra["measurement_method_version"] == "2025.5.2"


def test_fastafd_mtp_keeps_unmeasured_layer(tmp_path):
    engine = aisimulate_core.AicEngine.from_spec(_compile(_profile(tmp_path, latency_ms=6.5, mtp_nextn=1), nextn=1))

    rows = engine.decode_step_per_op(8, 1024, 2)
    by_name = {name: latency for name, latency, *_ in rows}

    assert by_name["generation_fastafd_moe_stage"] == pytest.approx(6.5)
    assert "generation_unmeasured_moe_approximation" in by_name
    assert engine.decode_step_latency(8, 1024, 2) == pytest.approx(sum(row[1] for row in rows))


def test_fastafd_stage_preserves_generation_weights(tmp_path):
    config = build_model_config(
        tp_size=1,
        pp_size=1,
        attention_dp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
        forward_model="op_level",
    )
    model_path = "deepseek-ai/DeepSeek-V4-Flash"
    resolve_dsv4_moe_arch(config, model_path, system_name="b200_sxm", backend_name="sglang")
    model = get_model(model_path, config, "sglang")
    before = sum(op.get_weights() for op in model.generation_ops)

    metadata = apply_fastafd_moe_profile(
        model,
        model_path=model_path,
        system="b200_sxm",
        backend="sglang",
        nextn=0,
        profile_path=_profile(tmp_path),
        profile_backend="megamoe",
    )

    stage = next(op for op in model.generation_ops if op._name == "generation_fastafd_moe_stage")
    assert stage.get_weights() > 0
    assert sum(op.get_weights() for op in model.generation_ops) == pytest.approx(before)
    assert metadata["source_repository"] == FASTAFD_OFFICIAL_REPOSITORY
    assert metadata["source_commit"] == "3c7161949310b6d59d6b4cf9bf997a4935c8113b"
    assert metadata["measurement_method"] == "nsight-systems-cupti"
    assert metadata["measurement_method_version"] == "2025.5.2"


def test_fastafd_timing_requires_a_backend_and_op_level():
    with pytest.raises(ValueError, match="configured together"):
        TimingConfig(fastafd_profile_path="profile.json")
    with pytest.raises(ValueError, match="op_level"):
        TimingConfig(
            fastafd_profile_path="profile.json",
            fastafd_moe_backend="megamoe",
            estimation_mode="fpm_interpolation",
        )


def test_fastafd_profile_is_aggregated_sglang_only():
    timing = {"fastafd_profile_path": "profile.json", "fastafd_moe_backend": "megamoe"}
    worker = {"parallelism": {"tensor": 1, "attention_data": 8, "moe_tensor": 1, "moe_expert": 8}, "timing": timing}
    with pytest.raises(ValueError, match="backend='sglang'"):
        EnginePredictionConfig(
            mode="aggregated",
            model="deepseek-ai/DeepSeek-V4-Flash",
            hardware="b200_sxm",
            backend="vllm",
            workers={"aggregated": worker},
        )

    with pytest.raises(ValueError, match="pipeline=1"):
        EnginePredictionConfig(
            mode="aggregated",
            model="deepseek-ai/DeepSeek-V4-Flash",
            hardware="b200_sxm",
            backend="sglang",
            workers={"aggregated": {**worker, "parallelism": {**worker["parallelism"], "pipeline": 2}}},
        )
