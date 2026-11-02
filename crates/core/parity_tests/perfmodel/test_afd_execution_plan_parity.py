# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-source AFD layouts, separately pinned from the legacy numerical goldens."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from test_afd_parity import _run_native, afd_parity

GOLDEN = Path(__file__).parent / "goldens" / "afd_execution_plan.json"
CASES = json.loads(GOLDEN.read_text())["cases"]


def case_config(case, directory):
    return afd_parity.layout_config(case, directory)


@pytest.mark.parametrize("name", list(CASES))
def test_real_layout_native_replay_matches_independent_golden(name, tmp_path):
    case = CASES[name]
    raw = case_config(case, tmp_path)
    config = tmp_path / "prediction.yaml"
    config.write_text(yaml.safe_dump(raw))
    prediction = _run_native(config)
    summary = prediction.get("summary", prediction)
    assert summary["completed_requests"] == summary["num_requests"] == 4
    expected = case["expected"]
    for field, value in expected.items():
        assert summary[field] == pytest.approx(value, rel=1e-6, abs=1e-9), field
    assert summary["committed_prefill_tokens"] <= summary["total_input_tokens"]
    if "cached_chunked" in name:
        assert summary["committed_prefill_tokens"] == 512 + 3 * 256
        assert summary["first_admission_prefix_cache_reused_ratio"] > 0


def test_new_layout_native_plan_survives_engine_json_and_binary_round_trip(monkeypatch):
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate_core._native import AicEngine, engine_spec_bincode_from_json
    from aisimulate_core.sdk import engine

    compiled = []
    build = engine.build_engine_spec_json

    def capture(*args, **kwargs):
        text = build(*args, **kwargs)
        spec = json.loads(text)
        if spec.get("afd_plan"):
            AicEngine.from_spec(engine_spec_bincode_from_json(text))
            compiled.append(spec)
        return text

    monkeypatch.setattr(engine, "build_engine_spec_json", capture)
    raw = copy.deepcopy(CASES["deepseek_v32_tp2"]["config"])
    raw["traffic"]["stop"]["requests"] = 1
    runner = EngineReplayRunnerFactory().create(0)
    try:
        report = runner.run(
            prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
        )
    finally:
        runner.close()
    assert report.metrics["completed_requests"] == 1
    assert {spec["afd_plan"]["role"] for spec in compiled} == {"a", "f"}
    for spec in compiled:
        plan = spec["afd_plan"]
        assert [plan["context"]["templates"][i]["routed"] for i in plan["layers"]] == [
            False
        ] * 3 + [True] * 58
        assert plan["shared_inter_size"] == 2048


def test_native_cli_rejects_interleaved_dense_model_with_a_nonzero_exit():
    config = GOLDEN.parent.parent / "configs" / "afd-predict-pure.yaml"
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run_native(
            config, "engine.model=meta-llama/Llama-4-Maverick-17B-128E-Instruct"
        )
    assert error.value.returncode != 0
    assert "dense layer 2 follows routed layer 1" in error.value.stderr


def test_dense_prefix_full_batch_activation_reaches_memory_fit_and_kv_budget(
    monkeypatch,
):
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate.sdk import afd_memory

    captured = {}
    for name in ("check_afd_memory_fit", "estimate_afd_kv_budget_bytes"):
        original = getattr(afd_memory, name)

        def capture(*args, _name=name, _original=original, **kwargs):
            captured[_name] = kwargs["a_activation_gb"]
            return _original(*args, **kwargs)

        monkeypatch.setattr(afd_memory, name, capture)
    raw = copy.deepcopy(CASES["deepseek_v32_tp2"]["config"])
    raw["engine"]["workers"]["aggregated"]["afd"].update(
        a_batch_size=8, num_microbatches=3
    )
    raw["traffic"]["stop"]["requests"] = 1
    runner = EngineReplayRunnerFactory().create(0)
    try:
        report = runner.run(
            prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
        )
    finally:
        runner.close()
    assert report.metrics["completed_requests"] == 1
    # DeepSeek: dense gate/up width 2*18432/TP2, 3 bytes per
    # intermediate element, plus BF16 hidden width 7168. Full B=8,
    # never the padded 3*3 or the routed microbatch B=3.
    for name, phases in captured.items():
        for phase, sequence in (("prefill", 128), ("decode", 1)):
            expected_bytes = 8 * sequence * (18432 * 3 + 7168 * 2)
            assert phases[phase] * (1 << 30) == expected_bytes, (name, phase)
    assert len(captured) == 2


@pytest.mark.parametrize("variable", [False, True])
def test_single_request_prefill_is_independent_of_configured_batch_capacity(
    variable, tmp_path
):
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory

    raw = copy.deepcopy(CASES["qwen_tp2"]["config"])
    raw["traffic"]["source"].update(input_tokens=128, output_tokens=1)
    if variable:
        trace = tmp_path / "one-request.jsonl"
        trace.write_text(
            json.dumps(
                {
                    "timestamp": 0,
                    "input_length": 128,
                    "output_length": 1,
                    "hash_ids": [42],
                }
            )
            + "\n"
        )
        raw["traffic"]["source"] = {
            "type": "trace",
            "format": "mooncake",
            "paths": [str(trace)],
        }
        raw["traffic"]["stop"] = {}
    else:
        raw["traffic"]["stop"]["requests"] = 1
    raw["traffic"]["load"]["concurrency"] = 1
    results = []
    for capacity in [4, 8]:
        raw["engine"]["workers"]["aggregated"]["afd"]["a_batch_size"] = capacity
        runner = EngineReplayRunnerFactory(determinism="canonical_v1").create(0)
        try:
            results.append(
                runner.run(
                    prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
                ).metrics
            )
        finally:
            runner.close()
    for result in results:
        assert result["completed_requests"] == result["total_output_tokens"] == 1
        assert result["total_input_tokens"] == 128
    assert results[0]["mean_ttft_ms"] == results[1]["mean_ttft_ms"]


def test_canonical_f_ep64_constructor_builds_only_routed_compute(monkeypatch):
    from aisimulate_core.sdk import (
        ForwardPassPerfModelConfig,
        RustForwardPassPerfModel,
        engine,
    )

    compiled = []
    build = engine.build_engine_spec_json

    def capture(*args, **kwargs):
        raw = build(*args, **kwargs)
        compiled.append(json.loads(raw))
        return raw

    monkeypatch.setattr(engine, "build_engine_spec_json", capture)
    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model="Qwen/Qwen3-30B-A3B",
            system="h200_sxm",
            backend="sglang",
            backend_version="0.5.14",
            worker_type="aggregated",
            estimation_mode="op_level",
            tp=64,
            moe_tp_size=1,
            moe_ep_size=64,
            afd_pool_role="f",
        )
    )
    model.close()
    spec = compiled[-1]
    assert spec["afd_plan"]["role"] == "f"
    assert spec["afd_plan"]["layers"] == [0] * 48
    for phase in ["context", "generation"]:
        assert len(spec["afd_plan"][phase]["templates"]) == 1
        assert len(spec[phase + "_ops"]) == 1
        assert spec[phase + "_ops"][0]["Moe"]["moe_ep_size"] == 64
