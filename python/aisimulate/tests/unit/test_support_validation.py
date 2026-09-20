# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Onboarding validates actual replay lookups, including failed and partial runs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from aisimulate import main as cli
from aisimulate import supervision
from aisimulate.support.plan import create_plan
from aisimulate.support.schema import SupportRequest

pytestmark = pytest.mark.unit


@pytest.fixture
def validation_case(tmp_path, monkeypatch, request):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk import engine

    def reject_graph(*_args, **_kwargs):
        raise AssertionError("FPM replay must not construct an analytical model")

    monkeypatch.setattr(engine, "get_model", reject_graph)
    monkeypatch.setattr(engine, "build_model_config", reject_graph)
    profile = {
        "schema_version": 1,
        "model": "test/onboarding-coverage",
        "model_revision": "synthetic-v1",
        "architecture": "UnregisteredDecoderForCausalLM",
        "context_length": 512,
        "num_experts": 0,
        "provenance": "Synthetic test metadata; not silicon qualification.",
        "deployments": [
            {
                "system": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.25.1",
                "tp": 2,
                "dp": 1,
                "moe_tp": 1,
                "moe_ep": 1,
                "gemm_quant_mode": "fp8",
                "moe_quant_mode": "fp8",
                "fmha_quant_mode": "bfloat16",
                "comm_quant_mode": "half",
                "kv_cache_dtype": "fp8",
                "resources": {
                    "weights_bytes": 100,
                    "activations_bytes": 20,
                    "runtime_overhead_bytes": 30,
                    "comm_overhead_bytes": 50,
                    "kv_bytes_per_token": 10,
                    "cache_layout": "linear",
                    "max_num_tokens": 128,
                    "max_batch_size": 2,
                    "provenance": "Declared synthetic rank-local bounds.",
                },
            }
        ],
    }
    topology = getattr(request, "param", "tp")
    parallel = {"tensor_parallel": 2, "context_length": 512}
    if topology in {"dep", "tep"}:
        profile["num_experts"] = 4
        profile["deployments"][0].update(tp=1 if topology == "dep" else 2, dp=2 if topology == "dep" else 1, moe_ep=2)
        parallel.update(
            tensor_parallel=profile["deployments"][0]["tp"],
            attention_data_parallel=profile["deployments"][0]["dp"],
            moe_tensor_parallel=1,
            moe_expert_parallel=2,
        )
    request = SupportRequest.model_validate(
        {
            "identity": {
                "model": profile["model"],
                "model_revision": profile["model_revision"],
                "model_kind": "dense" if topology == "tp" else "moe",
                "framework_version": "0.25.1",
                "gpu": "h200_sxm",
                "interconnect": "nvswitch",
            },
            "search": parallel,
            "collection": {"max_num_tokens": 128, "max_batch_size": 2, "gpu_memory_utilization": 0.73},
            "workload": {"input_tokens": 64, "output_tokens": 3, "request_count": 1},
            "fpm_profile": profile,
        }
    )
    plan = tmp_path / "collection"
    create_plan(request, plan)
    deployment = request.profile_deployment().model_dump(mode="json", exclude={"resources"})
    rows = []
    for batch in (1, 2):
        for phase, tokens, kv_values in (
            ("prefill", (1, 64, 128, 256), (0, 64, 128, 512)),
            ("decode", (0,), (1, 2, 64, 128, 1024)),
        ):
            for tokens_in_batch in tokens:
                for kv in kv_values:
                    rows.append(
                        {
                            **deployment,
                            "model_path": profile["model"],
                            "cell_id": f"synthetic-{phase}",
                            "weight_quantization": "synthetic",
                            "workload_kind": phase,
                            "partition_policy": "balanced_v1",
                            "batch_size": batch,
                            "total_prefill_tokens": tokens_in_batch,
                            "total_kv_read_tokens": kv,
                            "latency_ms": 1 + batch + tokens_in_batch / 128 + kv / 1024,
                            "kv_seed_regime": "real_kv",
                        }
                    )
    table = plan / "systems/data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet"
    table.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), table)
    metadata = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "coordinate_system": "iteration_totals_balanced_v1",
        "measurement_policy": "dynamo_native_single_sample_v1",
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.25.1",
        "row_count": len(rows),
        "parquet_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
    }
    table.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    trace = tmp_path / "trace.jsonl"
    play = {
        "id": "synthetic-play",
        "models": ["trace/source-model"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {"t": 0.0, "type": "s", "model": "trace/source-model", "in": 64, "out": 3, "hash_ids": [1]},
            {
                "t": 0.1,
                "type": "s",
                "model": "trace/source-model",
                "in": 128,
                "out": 3,
                "hash_ids": [1, 2],
            },
        ],
    }
    trace.write_text(json.dumps(play) + "\n")
    output = tmp_path / "validation"
    args = [
        "onboard",
        "validate-fpm",
        "--config",
        str(plan / "request.yaml"),
        "--output-dir",
        str(plan),
        "--trace",
        str(trace),
        "--validation-output-dir",
        str(output),
    ]
    # Exercise the public CLI compiler/runner/native runtime in-process so the
    # no-analytical-graph assertion above also covers estimator construction.
    monkeypatch.setattr(supervision, "main", cli.main)
    return args, request, plan, trace, output


@pytest.mark.parametrize("validation_case", ["tp", "dep", "tep"], indirect=True)
def test_onboard_validation_replays_profile_without_graph_and_preserves_timing(validation_case, tmp_path):
    args, request, plan, _trace, output = validation_case
    saved_inputs = {path: path.read_bytes() for path in plan.rglob("*") if path.is_file()}
    assert cli.main(args) == 0
    validation = json.loads((output / "validation.json").read_text())
    covered = json.loads((output / "prediction/prediction.json").read_text())
    assert validation["status"] == "covered"
    assert validation["accuracy"] == "not_assessed"
    assert validation["fpm_query_coverage"]["queries"]["measured"] > 0
    assert validation["fpm_query_coverage"]["queries"]["interpolated"] > 0
    assert validation["fpm_query_coverage"]["queries"]["unsupported"] == 0
    assert covered["completed_requests"] == 2
    assert covered["agentic_model_projection"]["target_model"] == request.identity.model
    assert covered["agentic_graph"]["source_models"] == ["trace/source-model"]
    prediction = yaml.safe_load((output / "predict.yaml").read_text())
    engine = prediction["engine"]
    assert engine["fpm_profile"] == request.fpm_profile.model_dump(mode="json")
    assert engine["workers"]["aggregated"]["scheduler"] == request.scheduler_limits()
    assert engine["workers"]["aggregated"]["kv_cache"]["capacity"]["memory_fraction"] == 0.73
    assert saved_inputs == {path: path.read_bytes() for path in saved_inputs}

    # Coverage is diagnostic only: cold/warm prefix behavior and every native
    # request output remain identical when collection is disabled.
    ordinary = copy.deepcopy(prediction)
    ordinary["engine"]["workers"]["aggregated"]["timing"]["estimator_config"]["fpm_interpolation"].pop(
        "collect_coverage"
    )
    ordinary_path = tmp_path / "ordinary.yaml"
    ordinary_path.write_text(yaml.safe_dump(ordinary))
    ordinary_output = tmp_path / "ordinary"
    assert (
        cli.main(
            ["predict", "--config", str(ordinary_path), "--output-dir", str(ordinary_output), "--capture-per-request"]
        )
        == 0
    )
    uncollected = json.loads((ordinary_output / "prediction.json").read_text())
    assert "fpm_query_coverage" not in uncollected
    for key in ("per_request", "agentic_graph", "agentic_play_outcomes", "prefix_cache_reused_ratio"):
        assert covered[key] == uncollected[key]


def test_onboard_validation_keeps_missing_query_evidence_on_native_failure(validation_case):
    import pyarrow as pa
    import pyarrow.parquet as pq

    args, _request, plan, trace, output = validation_case
    play = json.loads(trace.read_text())
    play["requests"][0]["in"] = 384
    play["requests"][0]["hash_ids"] = list(range(6))
    trace.write_text(json.dumps(play) + "\n")
    table = plan / "systems/data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet"
    rows = [
        row
        for row in pq.read_table(table).to_pylist()
        if row["workload_kind"] != "prefill" or row["total_kv_read_tokens"] <= 128
    ]
    pq.write_table(pa.Table.from_pylist(rows), table)
    metadata_path = table.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata.update(row_count=len(rows), parquet_sha256=hashlib.sha256(table.read_bytes()).hexdigest())
    metadata_path.write_text(json.dumps(metadata))
    # The third prefill chunk needs a longer KV bracket than this table has.
    assert cli.main(args) != 0
    validation = json.loads((output / "validation.json").read_text())
    assert validation["status"] == "incomplete"
    assert validation["issues"]
    assert validation["prediction_exit_code"] != 0
    coverage = validation["fpm_query_coverage"]
    assert coverage["queries"]["unsupported"] > 0
    assert coverage["queries"]["measured"] > 0
    assert coverage["roles"][0]["coverage"]["gaps"][0]["coordinates"]["total_kv_read_tokens"] == 256


def test_onboard_validation_rejects_changed_collection_identity_before_replay(validation_case, monkeypatch):
    args, _request, plan, _trace, output = validation_case
    config = yaml.safe_load((plan / "request.yaml").read_text())
    config["identity"]["model_revision"] = "different-revision"
    supplied = plan.parent / "different.yaml"
    supplied.write_text(yaml.safe_dump(config))
    args[args.index("--config") + 1] = str(supplied)
    monkeypatch.setattr(supervision, "main", lambda _args: pytest.fail("replay must not start"))
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("flag", ["--trace", "--config"])
def test_validation_overwrite_preserves_inputs_in_output_directory(validation_case, flag):
    args, _request, _plan, _trace, output = validation_case
    source = Path(args[args.index(flag) + 1])
    before = source.read_bytes()
    output.mkdir()
    input_path = output / "prediction.json"
    input_path.write_bytes(before)
    args[args.index(flag) + 1] = str(input_path)
    with pytest.raises(SystemExit) as error:
        cli.main([*args, "--overwrite"])
    assert error.value.code == 2
    assert input_path.read_bytes() == before


@pytest.mark.parametrize("failure", ["failed_request", "incomplete_play", "truncated_output", "zero_queries"])
def test_validation_never_accepts_partial_replay(validation_case, monkeypatch, failure):
    args, _request, _plan, _trace, output = validation_case

    def partial_prediction(command):
        assert command[0] == "predict"
        assert json.loads((output / "validation.json").read_text())["status"] == "incomplete"
        report = {
            "num_requests": 1,
            "completed_requests": 1,
            "agentic_graph": {"node_count": 1, "play_count": 1},
            "per_request": [{"terminal_status": "completed", "output_length": 2, "requested_output_length": 2}],
            "agentic_play_outcomes": [{"status": "completed", "settled_at_ms": 5}],
        }
        coverage = {"status": "covered", "queries": {"measured": 1, "interpolated": 0, "unsupported": 0}}
        if failure == "failed_request":
            report["per_request"][0]["terminal_status"] = "failed"
        elif failure == "incomplete_play":
            report["agentic_play_outcomes"][0]["status"] = "incomplete"
        elif failure == "truncated_output":
            report["per_request"][0]["output_length"] = 1
        else:
            coverage.update(status="incomplete", queries={"measured": 0, "interpolated": 0, "unsupported": 0})
        destination = Path(command[command.index("--output-dir") + 1])
        (destination / "prediction.json").write_text(json.dumps(report))
        (destination / "fpm-coverage.json").write_text(json.dumps(coverage))
        return 0

    monkeypatch.setattr(supervision, "main", partial_prediction)
    assert cli.main(args) == 1
    report = json.loads((output / "validation.json").read_text())
    assert report["status"] == "incomplete"
    assert report["issues"]


def test_validation_persists_incomplete_status_before_interrupted_prediction(validation_case, monkeypatch):
    args, _request, _plan, _trace, output = validation_case

    def interrupt(_command):
        assert json.loads((output / "validation.json").read_text())["status"] == "incomplete"
        raise KeyboardInterrupt

    monkeypatch.setattr(supervision, "main", interrupt)
    assert cli.main(args) == 130
    report = json.loads((output / "validation.json").read_text())
    assert report["status"] == "incomplete"
    assert "interrupted" in report["issues"][0]
