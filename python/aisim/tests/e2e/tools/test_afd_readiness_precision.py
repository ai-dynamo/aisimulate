# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-boundary regressions with real A/F compiled plans and data identities."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from aisim.compiler import prediction_to_replay_spec
from aisim.config import CorePredictionConfig
from aisim.runner import EngineReplayRunnerFactory, _materialize_engine_execution_spec
from aisim.sdk import afd_memory
from aisim.sweeper.replay import ReplayOutputRequirements
from aisim_core.sdk import engine

pytestmark = pytest.mark.build
SYSTEMS = Path(engine.__file__).parents[1] / "systems"
ROOT = Path(__file__).resolve().parents[5]


def _raw():
    return yaml.safe_load((ROOT / "crates/core/parity_tests/perfmodel/configs/afd-predict-pure.yaml").read_text())


def test_native_afd_reports_request_weighted_user_throughput(tmp_path):
    from aisim.aiperf_metrics import AIPerfMetric, metric_value

    trace = tmp_path / "requests.jsonl"
    trace.write_text(
        "".join(
            json.dumps(
                {
                    "timestamp": 0,
                    "input_length": inputs,
                    "output_length": outputs,
                    # Mooncake hashes each cover 512 tokens; keep the requests' prefixes disjoint.
                    "hash_ids": list(range(index * 16, index * 16 + (inputs + 511) // 512)),
                }
            )
            + "\n"
            for index, (inputs, outputs) in enumerate(((64, 3), (8192, 7)))
        )
    )
    raw = _raw()
    raw["engine"].update(database_mode="SOL", context_length=16384)
    raw["traffic"]["source"] = {"type": "trace", "format": "mooncake", "paths": [str(trace)]}
    raw["traffic"]["load"]["concurrency"] = 1
    raw["traffic"]["stop"] = {}
    replay = replace(
        prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)),
        goal={"target": "output_token_throughput_per_user.avg"},
    )
    runner = EngineReplayRunnerFactory(determinism="canonical_v1").create(0)
    try:
        report = runner.run(replay, output_requirements=ReplayOutputRequirements(capture_per_request=True))
    finally:
        runner.close()

    records = report.metadata["native_report"]["per_request"]
    assert len(records) == report.metrics["completed_requests"] == 2
    assert {record["input_length"] for record in records} == {64, 8192}
    itls = [record["itl_ms"] for record in records]
    assert max(itls) > min(itls) > 0
    # AIPerf averages each request's reciprocal ITL; the retired analytical
    # runner used the reciprocal of mean ITL, which differs on these requests.
    expected = sum(1000 / itl for itl in itls) / len(itls)
    actual = metric_value(report.metrics, AIPerfMetric.USER_TPS_AVG)
    assert actual == pytest.approx(expected)
    assert actual > 1000 / (sum(itls) / len(itls))


@pytest.mark.parametrize(
    "lengths,prefill_limit,expected_bound",
    [([64], 1, 64), ([31, 32, 32], 64, 95)],
)
def test_native_afd_memory_covers_prefill_budget_overshoot(monkeypatch, lengths, prefill_limit, expected_bound):
    from aisim import _runtime

    raw = _raw()
    raw["engine"].update(database_mode="SOL", context_length=128)
    raw["traffic"]["source"].update(input_tokens=max(lengths), output_tokens=1)
    replay = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    execution = _materialize_engine_execution_spec(replay, trace_block_size=1, record_per_request=True)
    execution.pop("traffic")
    rank = execution["spec"]["engine"]["rank"]
    rank.update(enable_prefix_caching=False, max_num_seqs=len(lengths))
    rank["sglang"].update(chunked_prefill_size=128, max_prefill_tokens=prefill_limit)
    execution["spec"]["max_in_flight"] = len(lengths)
    execution["spec"]["requests"] = [
        {
            "id": str(i),
            "arrival_time_ms": 0,
            "input_tokens": length,
            "input_token_ids": list(range(i * 128, i * 128 + length)),
            "output_tokens": 1,
        }
        for i, length in enumerate(lengths)
    ]
    observed = []

    def capture(function):
        def call(**kwargs):
            observed.append((function.__name__, kwargs["prefill_tokens"]))
            return function(**kwargs)

        return call

    for name in ("check_afd_memory_fit", "estimate_afd_kv_budget_bytes"):
        monkeypatch.setattr(afd_memory, name, capture(getattr(afd_memory, name)))

    report = json.loads(_runtime.run_replay_json(json.dumps(execution)))
    # SGLang admits before checking its soft prefill budget: a 64-token
    # request crosses 1; 31 + 32 leaves one token before the last 32 reaches 95.
    assert report["completed_requests"] == len(lengths)
    assert report["committed_prefill_tokens"] == sum(lengths) == expected_bound
    assert observed == [
        ("check_afd_memory_fit", expected_bound),
        ("estimate_afd_kv_budget_bytes", expected_bound),
    ]


@pytest.mark.parametrize("f_pool_fits", [True, False])
def test_sol_replay_without_measured_data_still_checks_pool_memory(tmp_path, f_pool_fits):
    for name in ("h200_sxm.yaml", "b200_sxm.yaml", "query_versions.yaml"):
        (tmp_path / name).write_text((SYSTEMS / name).read_text())
    if not f_pool_fits:
        spec = yaml.safe_load((tmp_path / "b200_sxm.yaml").read_text())
        # The model's routed weights and fixed overhead cannot fit in 1 GiB.
        spec["gpu"]["mem_capacity"] = 1 << 30
        (tmp_path / "b200_sxm.yaml").write_text(yaml.safe_dump(spec))
    raw = _raw()
    raw["engine"].update(database_mode="SOL", systems_paths=[str(tmp_path)], context_length=2048)
    raw["engine"]["workers"]["aggregated"]["afd"]["f_system"] = "b200_sxm"
    raw["traffic"]["stop"]["requests"] = 1
    # Exercise the saved-config boundary as well as native mode passthrough.
    config = CorePredictionConfig.model_validate_json(CorePredictionConfig.model_validate(raw).model_dump_json())
    replay = prediction_to_replay_spec(config)
    runner = EngineReplayRunnerFactory().create(0)
    try:
        if not f_pool_fits:
            with pytest.raises(RuntimeError, match="out_of_memory:.*F pool"):
                runner.run(replay)
        else:
            report = runner.run(replay, output_requirements=ReplayOutputRequirements(include_raw_report=True))
            assert report.metrics["completed_requests"] == 1
            memory = report.metadata["native_report"]["afd_memory"]["aggregated"]
            assert memory["status"] == "fits"
            assert memory["kv_budget_bytes_per_a_gpu"] > 0
            assert memory["kv_bytes_per_token_per_a_gpu"] > 0
            for phase in ("prefill", "decode"):
                assert memory["pools"][phase]["attention"]["system"] == "h200_sxm"
                assert memory["pools"][phase]["ffn"]["system"] == "b200_sxm"
    finally:
        runner.close()


@pytest.mark.parametrize("system,expected", [("h200_sxm", "nvfp4_wo"), ("b200_sxm", "nvfp4")])
def test_pool_precision_matches_memory_and_preserves_request(system, expected, monkeypatch):
    captured = []
    build = engine.build_engine_spec_json

    def capture(*args, **kwargs):
        raw = build(*args, **kwargs)
        captured.append(json.loads(raw))
        return raw

    monkeypatch.setattr(engine, "build_engine_spec_json", capture)
    engine.compile_engine(
        "Qwen/Qwen3-30B-A3B",
        system,
        "sglang",
        "0.5.14",
        afd_pool_role="a",
        gemm_quant_mode="nvfp4",
        moe_quant_mode="nvfp4",
    )
    modes = {v["quant_mode"] for op in captured[-1]["context_ops"] for k, v in op.items() if k == "Gemm"}
    assert expected in modes
    # Checkpoint exclusions may retain BF16 ops; they must not retain a
    # native-FP4 execution lane on Hopper.
    if expected == "nvfp4_wo":
        assert "nvfp4" not in modes
    overrides = {"gemm_quant_mode": "nvfp4", "moe_quant_mode": "nvfp4"}
    _, model = afd_memory._build_pool(
        afd_memory.get_backend("sglang"),
        "sglang",
        "0.5.14",
        "0.5.14",
        "Qwen/Qwen3-30B-A3B",
        8,
        "both",
        {"system": system, "a_tp_size": 1},
        is_a_pool=True,
        quant_overrides=overrides,
    )
    assert model.config.gemm_quant_mode.name == expected
    assert overrides == {"gemm_quant_mode": "nvfp4", "moe_quant_mode": "nvfp4"}


@pytest.mark.parametrize("missing_pool", ["A", "F"])
@pytest.mark.parametrize("missing_kind", ["table", "bucket"])
def test_missing_communication_rejects_before_memory_or_execution(tmp_path, monkeypatch, missing_pool, missing_kind):
    missing_system = "h200_sxm" if missing_pool == "A" else "b200_sxm"
    for source in SYSTEMS.iterdir():
        if source.is_file():
            (tmp_path / source.name).symlink_to(source)
    for system in ("h200_sxm", "b200_sxm"):
        for source in (SYSTEMS / "data" / system).rglob("*"):
            if not source.is_file():
                continue
            target = tmp_path / source.relative_to(SYSTEMS)
            target.parent.mkdir(parents=True, exist_ok=True)
            if system == missing_system and source.name in {"nccl_perf.parquet", "oneccl_perf.parquet"}:
                if missing_kind == "bucket" and source.name == "nccl_perf.parquet":
                    import pyarrow.compute as pc
                    import pyarrow.parquet as pq

                    table = pq.read_table(source)
                    wanted = pc.and_(pc.equal(table["nccl_dtype"], "half"), pc.equal(table["op_name"], "all_gather"))
                    pq.write_table(table.filter(pc.invert(wanted)), target)
                continue
            target.symlink_to(source)
    raw = _raw()
    raw["engine"].update(systems_paths=[str(tmp_path)], enable_shared_layer=False)
    raw["engine"]["workers"]["aggregated"]["afd"].update(tp_a=2, f_system="b200_sxm")

    def unreachable(**kwargs):
        raise AssertionError("communication readiness must precede memory and first pass")

    monkeypatch.setattr(afd_memory, "check_afd_memory_fit", unreachable)
    runner = EngineReplayRunnerFactory().create(0)
    try:
        with pytest.raises(RuntimeError, match=f"AFD {missing_pool} pool .*communication preflight"):
            runner.run(prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)))
    finally:
        runner.close()


@pytest.mark.parametrize(
    "model,n_f,tp,positive",
    [
        ("Qwen/Qwen3-30B-A3B", 2, 1, {"afd_a_side_combine"}),
        ("deepseek-ai/DeepSeek-V3.2", 1, 1, {"afd_a_shared_merge"}),
        ("Qwen/Qwen3-30B-A3B", 1, 2, {"afd_a_token_allgather"}),
        ("deepseek-ai/DeepSeek-V3.2", 2, 2, {"afd_a_side_combine", "afd_a_shared_merge", "afd_a_token_allgather"}),
    ],
)
def test_restore_operations_are_exercised_separately_and_together(tmp_path, model, n_f, tp, positive, capsys):
    from aisim.main import main

    raw = _raw()
    raw["engine"].update(model=model, context_length=2048)
    raw["engine"]["workers"]["aggregated"]["afd"].update(n_f_nodes=n_f, tp_a=tp, a_batch_size=4, num_microbatches=2)
    raw["traffic"]["source"].update(input_tokens=128, output_tokens=8)
    raw["traffic"]["load"]["concurrency"] = 2
    raw["traffic"]["stop"]["requests"] = 2
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    assert (
        main(
            [
                "predict",
                "--stack",
                "engine",
                "--config",
                str(config),
                "--output-dir",
                str(tmp_path / "out"),
                "--format",
                "json",
            ]
        )
        == 0
    )
    output = json.loads((tmp_path / "out/prediction.json").read_text())
    summary = output.get("summary", output)
    assert summary["completed_requests"] == 2
    restore = {"afd_a_side_combine", "afd_a_shared_merge", "afd_a_token_allgather"}
    phases = summary["power_diagnostics"]["phases"]
    for phase in phases:
        active = {op["name"] for op in phase["operations"] if op["name"] in restore and op["latency_ms"] > 0}
        assert active == positive


def test_precision_rejection_identifies_pool_gpu_and_request(tmp_path):
    from aisim_core.sdk.errors import InvalidEngineConfigurationError

    spec = yaml.safe_load((SYSTEMS / "h200_sxm.yaml").read_text())
    spec["gpu"].pop("sm_version")
    (tmp_path / "h200_sxm.yaml").write_text(yaml.safe_dump(spec))
    with pytest.raises(InvalidEngineConfigurationError, match="AFD A pool.*h200_sxm.*nvfp4"):
        engine.compile_engine(
            "Qwen/Qwen3-30B-A3B",
            "h200_sxm",
            "sglang",
            "0.5.14",
            afd_pool_role="a",
            gemm_quant_mode="nvfp4",
            moe_quant_mode="nvfp4",
            systems_path=str(tmp_path),
        )


def test_prediction_discloses_requested_and_effective_pool_precision(tmp_path):
    from aisim.main import main

    raw = _raw()
    raw["engine"].update(gemm_quant_mode="bfloat16", context_length=2048)
    raw["traffic"]["source"].update(input_tokens=128, output_tokens=2)
    raw["traffic"]["stop"]["requests"] = 1
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    assert (
        main(
            [
                "predict",
                "--stack",
                "engine",
                "--config",
                str(config),
                "--output-dir",
                str(tmp_path / "out"),
                "--format",
                "json",
            ]
        )
        == 0
    )
    output = json.loads((tmp_path / "out/prediction.json").read_text())
    summary = output.get("summary", output)
    identity = summary["prediction_provenance"]["aggregated"]["data_identity"]
    for pool in identity.values():
        assert pool["requested_quantization"]["gemm_quant_mode"] == "bfloat16"
        assert pool["requested_quantization"]["moe_quant_mode"] is None
    assert identity["a_pool"]["effective_quantization"]["gemm"] == ["bfloat16"]
    assert identity["f_pool"]["effective_quantization"]["moe"] == ["bfloat16"]
    assert raw["engine"]["gemm_quant_mode"] == "bfloat16"
