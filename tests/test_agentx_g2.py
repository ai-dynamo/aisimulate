# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Weka -> public API -> native G2 acceptance, using original fixtures."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from aisim import _runtime
from aisim import main as cli
from aisim.compiler import prediction_to_replay_spec
from aisim.config.cli import CorePredictionConfig
from aisim.runner import (
    EngineReplayRunnerFactory,
    _materialize_engine_execution_spec,
)
from aisim.sweeper.replay import AdapterReplaySpec, ReplayOutputRequirements, RuntimeHookSpec
from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

ROOT = Path(__file__).resolve().parents[1]


def _config(name: str) -> dict:
    config = yaml.safe_load((ROOT / "examples/cli" / name).read_text())
    config["traffic"]["source"]["paths"] = [
        str(ROOT / "tests/e2e/configs/unified_cli/fixtures/traces/weka-g2-restore.jsonl")
    ]
    config["engine"]["model"] = str(ROOT / "tests/e2e/configs/unified_cli/fixtures/tiny-model")
    return config


def _run(config: dict) -> dict:
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    result = (
        EngineReplayRunnerFactory()
        .create(0)
        .run(
            spec,
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True),
        )
    )
    return result.metadata["native_report"]


def _restored(report: dict) -> dict:
    # The original fixture's third outer event resumes the parent after the
    # child evicts its prefix. Weka currently encodes that event as :outer:2;
    # keep this selector tied to that fixture if request-ID encoding changes.
    return next(row for row in report["per_request"] if row["request_id"].endswith(":outer:2"))


def _evidence(report: dict) -> dict:
    fields = (
        "first_admission_g1_reused_input_tokens",
        "first_admission_host_reused_input_tokens",
        "first_admit_ms",
        "ttft_ms",
        "admission_history",
    )
    return {
        "completed_requests": report["completed_requests"],
        "committed_prefill_tokens": report["committed_prefill_tokens"],
        "g2_domains": report.get("g2_domains", []),
        "qualification": report["agentic_qualification"],
        "restore": {key: _restored(report)[key] for key in fields},
    }


@pytest.mark.parametrize("name", ["agentx-g2-local.yaml", "agentx-g2-shared-pd.yaml"])
def test_weka_g2_cli_python_and_native_agree(name, tmp_path, capsys) -> None:
    config = _config(name)
    python_report = _run(config)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    native_payload = _materialize_engine_execution_spec(spec, trace_block_size=4, record_per_request=True)
    native_report = json.loads(_runtime.run_replay_json(json.dumps(native_payload)))
    path = tmp_path / "agentx.yaml"
    path.write_text(yaml.safe_dump(config))
    assert (
        cli.main(
            [
                "predict",
                "--stack",
                "engine",
                "--config",
                str(path),
                "--capture-per-request",
                "--output-dir",
                str(tmp_path / "out"),
                "--format",
                "json",
            ]
        )
        == 0
    )
    cli_report = json.loads(capsys.readouterr().out)
    assert _evidence(cli_report) == _evidence(python_report) == _evidence(native_report)
    restored = _restored(python_report)
    assert restored["first_admission_g1_reused_input_tokens"] == 0
    assert restored["first_admission_host_reused_input_tokens"] == 8
    assert python_report["agentic_qualification"] == "functional_only"
    assert python_report["completed_requests"] == 4
    assert sum(row["first_admission_host_reused_input_tokens"] for row in python_report["per_request"]) == 8
    if config["engine"]["mode"] == "disaggregated":
        assert {row["pool"] for row in restored["admission_history"]} == {
            "prefill",
            "decode",
        }
        assert len(python_report["g2_domains"]) == 1
        assert python_report["g2_domains"][0]["capacity_blocks"] == 8


@pytest.mark.parametrize("name", ["agentx-g2-local.yaml", "agentx-g2-shared-pd.yaml"])
def test_weka_g2_restore_has_causal_bandwidth_and_capacity_controls(name) -> None:
    config = _config(name)
    fast = _run(config)
    slow_config, small_config, hbm_config = (copy.deepcopy(config) for _ in range(3))
    # YAML aliases can share objects; assign constants rather than relative edits.
    for worker in slow_config["engine"]["workers"].values():
        worker["kv_cache"]["host_offload"]["h2d_bandwidth_gbps"] = 0.01
    for worker in small_config["engine"]["workers"].values():
        worker["kv_cache"]["host_offload"]["num_host_blocks"] = 1
    for worker in hbm_config["engine"]["workers"].values():
        worker["kv_cache"].pop("host_offload", None)
    slow, small, hbm = (_run(candidate) for candidate in (slow_config, small_config, hbm_config))
    assert _restored(slow)["first_admit_ms"] >= _restored(fast)["first_admit_ms"] + 100
    assert _restored(slow)["ttft_ms"] >= _restored(fast)["ttft_ms"] + 100
    assert _restored(small)["first_admission_host_reused_input_tokens"] == 0
    assert _restored(hbm)["reused_input_tokens"] == 0
    assert fast["committed_prefill_tokens"] < hbm["committed_prefill_tokens"]


@pytest.mark.parametrize("field,value", [("num_host_blocks", 16), ("shared_h2d_bandwidth_gbps", 2)])
def test_weka_g2_rejects_inconsistent_shared_pool(field, value) -> None:
    config = _config("agentx-g2-shared-pd.yaml")
    config["engine"]["workers"]["decode"] = copy.deepcopy(config["engine"]["workers"]["decode"])
    config["engine"]["workers"]["decode"]["kv_cache"]["host_offload"][field] = value
    with pytest.raises(RuntimeError, match="cluster_shared host_offload participants are incompatible"):
        _run(config)


def test_weka_g2_rejects_incompatible_shared_layout() -> None:
    config = _config("agentx-g2-shared-pd.yaml")
    config["engine"]["workers"]["decode"] = copy.deepcopy(config["engine"]["workers"]["decode"])
    config["engine"]["workers"]["decode"]["parallelism"]["tensor"] = 2
    with pytest.raises(RuntimeError, match="cluster_shared host_offload participants are incompatible"):
        _run(config)


@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
@pytest.mark.parametrize(
    "fault,message",
    [
        ("backend", "agentic host offload requires backend=vllm on every role"),
        ("aic_nextn", "requires canonical mtp speculation"),
        ("nextn", "requires canonical mtp speculation"),
        ("speculation", "requires canonical mtp speculation"),
        ("g3", "agentic replay does not support G3 offload"),
        ("scope", "dp_rank_local"),
        ("scaling", "agentic host offload requires static worker pools without a scaling policy"),
    ],
)
def test_agentic_invalid_configuration_matrix_rejects_python_and_native(role, fault, message) -> None:
    """The same semantic failures must fail both API boundaries.

    Start from one valid lowered payload, then express each fault in the public
    and native schemas. Python aliases map to native aic_nextn; scaling hooks
    map to the native scaling provider. Native scope remains a typed enum, so
    its invalid-variant diagnostic can differ from Python's early scope gate.
    """
    aggregated = role == "aggregated"
    config = _config("agentx-g2-local.yaml" if aggregated else "agentx-g2-shared-pd.yaml")
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    payload = _materialize_engine_execution_spec(spec, trace_block_size=4, record_per_request=True)
    args_field = "agg_engine_args" if aggregated else f"{role}_engine_args"
    args = getattr(spec.backend_deployment, args_field)
    engine = payload["spec"]["engine"] if aggregated else payload["spec"]["engine"][role]
    rank = engine["rank"]
    if fault == "backend":
        args["engine_type"] = "sglang"
        rank["backend"] = "sglang"
    elif fault in {"aic_nextn", "nextn", "speculation"}:
        args[fault] = {"num_speculative_tokens": 1} if fault == "speculation" else 1
        rank["aic_nextn"] = 1
    elif fault == "g3":
        args["g3_offload"] = rank["g3_offload"] = {"scope": "worker_local", "num_g3_blocks": 8}
    elif fault == "scope":
        args["native_host_offload"]["scope"] = "unsupported_scope"
        rank["native_host_offload"]["scope"] = "unsupported_scope"
    elif fault == "scaling":
        hook = RuntimeHookSpec(provider="test.planner", kind="scaling_policy", api_version=1)
        spec = replace(spec, adapters={"planner": AdapterReplaySpec(runtime_hooks=(hook,))})
        payload["spec"]["adapters"]["scaling"]["provider"] = hook.provider
    else:
        raise AssertionError(f"unhandled semantic fault: {fault}")
    with pytest.raises(ValueError, match=message):
        EngineReplayRunnerFactory().capabilities().require_compatible(spec)
    native_message = "requires.*canonical" if fault in {"aic_nextn", "nextn", "speculation"} else message
    with pytest.raises(RuntimeError, match=native_message):
        _runtime.run_replay_json(json.dumps(payload))


def test_agentic_g2_runs_multi_worker_attention_dp_pd_with_mtp_on_one_shared_pool(tmp_path) -> None:
    """Two DP2 prefill workers and one DP2 decode worker share one G2 pool under MTP.

    The resumed parent is the third request, so round-robin places it on a prefill
    rank that never held its prefix: it restores from the shared pool. Six output
    tokens per request make every decode run draft rounds.
    """
    config = _canonical_mtp(_config("agentx-g2-shared-pd.yaml"))
    source = config["traffic"]["source"]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(Path(source["paths"][0]).read_text().replace('"out":1', '"out":6'))
    source["paths"] = [str(trace)]
    prefill = config["engine"]["workers"]["prefill"]
    prefill["kv_cache"]["capacity"]["blocks"] = 4  # 9 input and 6 output tokens
    decode = config["engine"]["workers"]["decode"] = copy.deepcopy(prefill)
    prefill["parallelism"].update(replicas=2, attention_data=2)
    decode["parallelism"].update(attention_data=2)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    lowered = _materialize_engine_execution_spec(spec, trace_block_size=4, record_per_request=True)["spec"]
    assert lowered["topology"]["prefill"]["initial_workers"] == 2
    for role in ("prefill", "decode"):
        assert lowered["engine"][role]["dp_size"] == 2
        assert lowered["engine"][role]["rank"]["aic_nextn"] == 2
    report = _run(config)
    assert report["completed_requests"] == 4
    assert report["total_output_tokens"] == 24
    assert [domain["capacity_blocks"] for domain in report["g2_domains"]] == [8]
    restored = _restored(report)
    assert restored["first_admission_g1_reused_input_tokens"] == 0
    assert restored["first_admission_host_reused_input_tokens"] > 0


def test_agentic_g2_runs_on_several_aggregated_workers_with_private_pools() -> None:
    """Several aggregated workers with a private G2 each run through both boundaries."""
    config = _config("agentx-g2-local.yaml")
    config["engine"]["workers"]["aggregated"]["parallelism"]["replicas"] = 2
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    EngineReplayRunnerFactory().capabilities().require_compatible(spec)
    payload = _materialize_engine_execution_spec(spec, trace_block_size=4, record_per_request=True)
    assert payload["spec"]["topology"]["workers"]["initial_workers"] == 2
    report = json.loads(_runtime.run_replay_json(json.dumps(payload)))
    assert report["completed_requests"] == 4
    assert report["agentic_qualification"] == "functional_only"
    workers = {entry["logical_worker_id"] for row in report["per_request"] for entry in row["routing_history"]}
    assert workers == {0, 1}


def _planner_pass(prefills: int, prefill_tokens: int, decodes: int, kv: int) -> list[dict]:
    ms = 2.0 + 0.004 * prefill_tokens + 0.3 * decodes + 2e-5 * kv
    counters = {"num_prefill_requests": prefills, "sum_prefill_tokens": prefill_tokens}
    counters |= {"num_decode_requests": decodes, "sum_decode_kv_tokens": kv}
    return [{"version": 1, "wall_time": ms / 1000, "scheduled_requests": counters}]


def _planner_snapshot(path: Path, worker_type: str, kinds: tuple[str, ...]) -> str:
    """A planner fits its regression for an unknown deployment, so the snapshot records attention DP as unknown."""
    unknown = dict.fromkeys(("system", "backend_version", "tp", "pp", "attention_dp"), "unknown")
    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=str(ROOT / "tests/e2e/configs/unified_cli/fixtures/tiny-model"),
            backend="vllm",
            worker_type=worker_type,
            kv_block_size=4,
            estimation_mode="fpm_regression",
            **unknown,
        )
    )
    passes = []
    for i in range(48):
        decodes, kv = 1 + i % 8, (1 + i % 8) * (128 + 37 * (i % 11))
        shapes = {
            "prefill": (1 + i % 3, 128 * (1 + i % 5), 0, 0),
            "decode": (0, 0, decodes, kv),
            "mixed": (1, 32 * (1 + i % 6), decodes, kv),
        }
        passes += [_planner_pass(*shapes[kind]) for kind in kinds]
    model.tune_with_fpms(passes)
    model.save_state_snapshot(str(path))
    return str(path)


@pytest.mark.parametrize(
    "name,kinds",
    [
        ("agentx-g2-local.yaml", {"aggregated": ("prefill", "decode", "mixed")}),
        ("agentx-g2-shared-pd.yaml", {"prefill": ("prefill",), "decode": ("decode",)}),
    ],
)
def test_agentic_g2_replays_planner_snapshots_with_an_unknown_attention_dp(name, kinds, tmp_path) -> None:
    config = _config(name)
    workers = config["engine"]["workers"]
    for role, role_kinds in kinds.items():
        snapshot = _planner_snapshot(tmp_path / role, role, role_kinds)
        timing = {"estimation_mode": "fpm_regression", "estimator_config": {"state_snapshot": {"path": snapshot}}}
        workers[role] = {**workers[role], "timing": timing}  # The P/D example aliases one worker block.
    report = _run(config)
    assert report["completed_requests"] == 4
    # The snapshots, not the example's fixed timing, price the passes.
    fixed = _run(_config(name))
    assert [row["ttft_ms"] for row in report["per_request"]] != [row["ttft_ms"] for row in fixed["per_request"]]


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("with_g2", [False, True])
def test_agentic_g3_rejection_matches_without_requiring_g2(backend, with_g2) -> None:
    config = _config("agentx-g2-local.yaml")
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    payload = _materialize_engine_execution_spec(spec, trace_block_size=4, record_per_request=True)
    spec = replace(spec, backend_deployment=replace(spec.backend_deployment, backend=backend))
    args = spec.backend_deployment.agg_engine_args
    rank = payload["spec"]["engine"]["rank"]
    args["engine_type"] = rank["backend"] = backend
    args["g3_offload"] = rank["g3_offload"] = {"scope": "worker_local", "num_g3_blocks": 8}
    if not with_g2:
        args.pop("native_host_offload")
        rank.pop("native_host_offload")
    message = "agentic replay does not support G3 offload"
    with pytest.raises(ValueError, match=message):
        EngineReplayRunnerFactory().capabilities().require_compatible(spec)
    with pytest.raises(RuntimeError, match=message):
        _runtime.run_replay_json(json.dumps(payload))


def _canonical_mtp(config: dict) -> dict:
    config["engine"].update(
        model="meta-llama/Meta-Llama-3.1-8B",
        backend_version="0.24.0",
        estimation_mode="op_level",
        speculation={
            "kind": "mtp",
            "params": {"depth": 2},
            "expected_accepted_tokens": 1.5,
            "seed": 17,
        },
    )
    for worker in config["engine"]["workers"].values():
        worker["timing"] = {"type": "default"}
    return config


def test_agentic_canonical_mtp_executes_with_local_g2() -> None:
    config = _canonical_mtp(_config("agentx-g2-local.yaml"))
    config["engine"]["workers"]["aggregated"]["kv_cache"]["capacity"]["blocks"] = 4
    report = _run(config)
    assert report["completed_requests"] == 4
    assert report["total_output_tokens"] == 4
    assert report["agentic_qualification"] == "functional_only"
    assert _restored(report)["first_admission_host_reused_input_tokens"] > 0
