# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

import aisimulate_core
from aisimulate.config import EnginePredictionConfig
from aisimulate.config.engine import TimingConfig
from aisimulate_core.sdk.engine import InvalidEngineConfigurationError, compile_engine
from aisimulate_core.sdk.fastafd_profile import FASTAFD_PROFILE_SCHEMA

pytestmark = pytest.mark.unit


def _profile(tmp_path, *, system="b200_sxm", latency_ms=5.25):
    entry = {
        "model_path": "deepseek-ai/DeepSeek-V4-Flash",
        "model_profile": "deepseek_v4_flash_fp4",
        "system": system,
        "stage": "agg",
        "topology": "ep8",
        "logical_batch_per_source_rank": 8,
        "mtp_nextn": 0,
        "microbatches": 1,
        "moe_layers": 43,
        "routed_topk": 6,
        "moe_precision": "w4a8_mxfp4_mxfp8",
        "moe_backend": "megamoe",
        "latency_ms": latency_ms,
        "validation": {"stable": True, "correctness": True, "evidence": "test"},
        "source": {
            "commit": "e507eacf858d2046bdc2cca02ed86c0e58bd6c60",
            "source_tree_sha256": "a" * 64,
            "result": "raw/point.json",
        },
    }
    path = tmp_path / "fastafd.json"
    path.write_text(json.dumps({"schema": FASTAFD_PROFILE_SCHEMA, "lookup_policy": "exact-only", "entries": [entry]}))
    return path


def _compile(profile):
    return compile_engine(
        "deepseek-ai/DeepSeek-V4-Flash",
        "b200_sxm",
        "sglang",
        tp_size=1,
        attention_dp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
        moe_quant_mode="w4a8_mxfp4_mxfp8",
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
