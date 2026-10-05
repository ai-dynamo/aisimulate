# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only end-to-end coverage for the public engine-stack CLI."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from aisimulate import EngineReplayRunnerFactory, ReplayOutputRequirements
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.sweeper import SweepResult
from aisimulate.sweeper.result import CandidateStatus

pytestmark = [
    pytest.mark.integration,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_ROOT = Path("tests/e2e/configs/unified_cli")
_EXPECTED_PREDICT_CASES = (
    "01-synthetic-default.yaml",
    "02-synthetic-poisson.yaml",
    "03-synthetic-session-constant.yaml",
    "04-trace-mooncake-speedup.yaml",
    "05-trace-mooncake-delta-concurrency.yaml",
    "06-trace-agentic-mooncake.yaml",
    "07-trace-applied-compute-agentic.yaml",
    "08-trace-dynamo-standard.yaml",
    "09-trace-dynamo-agentic.yaml",
    "10-trace-dynamo-standard-disagg.yaml",
    "11-synthetic-afd.yaml",
    "11-trace-weka-agentic-lane.yaml",
    "12-trace-weka-jsonl-agentic-lane.yaml",
)
_EXPECTED_RECOMMEND_CASES = (
    "01-default-preset-throughput.yaml",
    "02-custom-preset-throughput-per-gpu.yaml",
    "03-preset-off-ttft.yaml",
    "04-mixed-disagg-pareto.yaml",
    "05-kv-fraction-goodput.yaml",
    "06-override-parallel-mappings-agg-disagg.yaml",
    "07-afd-plus-pd.yaml",
    "08-heterogeneous-pd.yaml",
)
_PREDICT_CASES = tuple(sorted((_REPO_ROOT / _CONFIG_ROOT / "predict/engine").glob("*.yaml")))
_RECOMMEND_CASES = tuple(sorted((_REPO_ROOT / _CONFIG_ROOT / "recommend/engine").glob("*.yaml")))


def _native_policy_config(tmp_path: Path, topology: str, backend: str, affinity: str) -> dict:
    """An authored fork/join workload with repeated, physically reusable prefixes."""
    config = yaml.safe_load((_REPO_ROOT / _CONFIG_ROOT / "predict/engine/09-trace-dynamo-agentic.yaml").read_text())
    records = [
        {
            "schema": "dynamo.agentic_mooncake",
            "version": 2,
            "block_size": 64,
            "hash_id_scope": "local",
            "source": {"format": "aisimulate-e2e", "digest": "native-policy-v1"},
        }
    ]
    # Both plays deliberately reuse conversation names. Their binding keys must
    # stay distinct, while each pair of siblings shares a key in sibling mode.
    conversations = ["parent", "left", "right", "left", "right", "parent"]
    predecessors = [None, 0, 0, 1, 2, 0]
    for play in ("alpha", "beta"):
        for index, (conversation, previous) in enumerate(zip(conversations, predecessors, strict=True)):
            relation = "spawn" if index in (1, 2) else "sequence"
            records.append(
                {
                    "request_id": f"{play}/{index}",
                    "play_id": play,
                    "session_id": conversation,
                    "model": "tiny-model",
                    "input_length": 192,
                    "output_length": 3,
                    "hash_ids": [17, 19, 23],
                    "not_before_ms": 25 * index,
                    "recorded_api_time_ms": 3,
                    "dependencies": []
                    if previous is None
                    else [
                        {
                            "request_id": f"{play}/{previous}",
                            "relation": relation,
                            "trigger": "dispatch" if relation == "spawn" else "completion",
                            "delay_ms": 25,
                        }
                    ],
                }
            )
    trace = tmp_path / "policy-trace.jsonl"
    trace.write_text("".join(json.dumps(record) + "\n" for record in records))
    config["traffic"] = {
        "source": {"type": "trace", "format": "agentic_mooncake", "paths": [str(trace)], "block_size": 64},
        "load": {"type": "trace_timestamps"},
    }
    worker = config["engine"]["workers"]["aggregated"]
    worker["parallelism"].update(replicas=2, attention_data=2)
    roles = ["aggregated"] if topology == "aggregated" else ["prefill", "decode"]
    config["engine"].update(mode=topology, backend=backend, workers={role: copy.deepcopy(worker) for role in roles})
    config["router"] = {"policy": "kv_router", "affinity": {"mode": affinity, "ttl_seconds": 3600}}
    return config


def _run_native_policy_cli(tmp_path: Path, config: dict, name: str) -> dict:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / name
    _run_cli("predict", "--config", str(path), "--capture-per-request", "--output-dir", str(output), "--format", "json")
    report = json.loads((output / "prediction.json").read_text())
    assert report["per_request"] == [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    return report


def _assert_native_policy_execution(report: dict, topology: str, affinity: str) -> None:
    expected_roles = {"aggregated"} if topology == "aggregated" else {"prefill", "decode"}
    evidence = report["routing_policy"]
    assert evidence["api_version"] == 1
    assert set(evidence["roles"]) == expected_roles
    native_decisions = []
    for role, data in evidence["roles"].items():
        assert data["native_policy"] == "dynamo.SelectionCore"
        assert data["physical_kv_events"] > 0
        native_decisions.extend({**item, "role": role} for item in data["decisions"])
    decisions = {(item["request_id"], item["role"]): item for item in native_decisions}
    bound: dict[tuple, set] = {}
    keys: dict[tuple, set] = {}
    for request in report["per_request"]:
        assert request["terminal_status"] == "completed"
        identity = request["agentic"]
        lineage = identity["lineage"]
        conversation = identity["conversation_id"]
        parent = lineage.get("parent_conversation_id")
        group = (
            (identity["play_id"], conversation)
            if affinity == "session"
            else (identity["play_id"], lineage["root_conversation_id"], parent or conversation, parent is not None)
        )
        roles = set()
        for route in request["routing_history"]:
            role = "aggregated" if route["pool"] == "agg" else route["pool"]
            roles.add(role)
            selected = decisions[(request["uuid"], role)]
            target = selected["worker_id"], selected["dp_rank"]
            assert target == (route["logical_worker_id"], route["dp_rank"])
            assert all(value in (0, 1) for value in target)
            bound.setdefault((group, role), set()).add(target)
            keys.setdefault(group, set()).add(selected["group_key"])
        assert roles == expected_roles
    assert bound and all(len(targets) == 1 for targets in bound.values())
    assert all(len(values) == 1 for values in keys.values())
    assert len({next(iter(values)) for values in keys.values()}) == len(keys)
    assert any(item["binding_reused"] for item in native_decisions)
    assert report["first_admission_prefix_cache_reused_ratio"] > 0
    assert any(request["reused_input_tokens"] > 0 for request in report["per_request"])


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("affinity", ["session", "sibling_group"])
def test_installed_native_policy_yaml(tmp_path: Path, backend: str, topology: str, affinity: str) -> None:
    """Run with the existing Dynamo runtime wheel; no Dynamo replay adapter is required."""
    runtime = pytest.importorskip("dynamo._core", reason="optional Dynamo runtime wheel is not installed")
    assert runtime.NativeReplayPolicy.contract()["api_version"] == 1
    config = _native_policy_config(tmp_path, topology, backend, affinity)
    report = _run_native_policy_cli(tmp_path, config, "native")
    assert report["completed_requests"] == 12
    _assert_native_policy_execution(report, topology, affinity)
    for worker in config["engine"]["workers"].values():
        worker["kv_cache"]["prefix_caching"] = False
    cold = _run_native_policy_cli(tmp_path, config, "uncached")
    assert cold["completed_requests"] == report["completed_requests"]
    assert cold["first_admission_prefix_cache_reused_ratio"] == 0
    assert all(request["reused_input_tokens"] == 0 for request in cold["per_request"])


@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
def test_installed_native_policy_duration(tmp_path: Path, topology: str) -> None:
    runtime = pytest.importorskip("dynamo._core", reason="optional Dynamo runtime wheel is not installed")
    assert runtime.NativeReplayPolicy.contract()["api_version"] == 1
    config = _native_policy_config(tmp_path, topology, "vllm", "sibling_group")
    config["traffic"]["load"].update(
        agentic_lanes=2,
        agentic_snapshot={"seed": 42},
        agentic_warmup=True,
        agentic_profile={"duration_seconds": 0.5},
    )
    report = _run_native_policy_cli(tmp_path, config, "duration")
    _assert_native_policy_execution(report, topology, "sibling_group")
    profile, phases = report["agentic_profile"], report["agentic_phases"]
    assert profile["admission_cutoff_ms"] - profile["profile_start_ms"] == pytest.approx(500)
    assert profile["profile_start_ms"] == phases["profile_start_ms"]
    assert profile["plays_started"] > 2
    assert profile["unsettled_server_requests"] == 0 and not profile["cancel_drain_timed_out"]
    prepared = {request["uuid"] for request in phases["requests"]}
    measured = {request["uuid"] for request in report["per_request"]}
    assert prepared - measured
    assert prepared | measured <= {
        decision["request_id"] for role in report["routing_policy"]["roles"].values() for decision in role["decisions"]
    }


def _run_cli(
    *args: str,
    timeout: float = 120.0,
    env: dict[str, str] | None = None,
    expected_returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "aisimulate", *args],
        cwd=_REPO_ROOT,
        env=None if env is None else {**os.environ, **env},
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == expected_returncode, (
        f"aisimulate {' '.join(args)} failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _assert_concrete(value: Any, *, path: str = "config") -> None:
    if isinstance(value, dict):
        assert "choices" not in value, f"{path} still contains a choices domain"
        assert "range" not in value, f"{path} still contains a range domain"
        assert "preset" not in value, f"{path} still contains a preset"
        for key, child in value.items():
            _assert_concrete(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_concrete(child, path=f"{path}[{index}]")


def _check_documented_candidate_renderer(recommendation_output: Path, tmp_path: Path) -> None:
    guide = (_REPO_ROOT / "python/aisimulate/docs/dynamo_deployment_guide.md").read_text()
    script = guide.split("```python\n", 1)[1].split("\n```", 1)[0]
    original = (recommendation_output / "recommendation.json").read_text()
    selected_id = json.loads(original)["views"]["top_n"][0]
    # Keep the real feasible candidate; vary selection metadata to exercise
    # the documentation's scalar/Pareto choice and rejection paths.
    cases = (
        ("scalar", [], None),
        ("pareto", ["--candidate-id", selected_id], None),
        ("missing", [], "No scalar selection"),
        ("unknown", ["--candidate-id", "not-a-candidate"], "Unknown candidate ID"),
        ("infeasible", ["--candidate-id", selected_id], "must be feasible"),
    )
    for name, args, error in cases:
        root = tmp_path / f"render-{name}"
        study = root / "deployment-study"
        ledger_dir = study / "recommendation"
        ledger_dir.mkdir(parents=True)
        ledger = json.loads(original)
        if name in {"pareto", "missing"}:
            ledger["views"] = {"top_n": [], "pareto_front": [selected_id]}
        if name == "infeasible":
            for candidate in ledger["candidates"]:
                if candidate["candidate_id"] == selected_id:
                    candidate["status"] = "failed"
        (ledger_dir / "recommendation.json").write_text(json.dumps(ledger))
        script_path = study / "render.py"
        script_path.write_text(script)
        result = subprocess.run(
            [sys.executable, str(script_path), *args],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if error:
            assert result.returncode == 2, (name, result.stdout, result.stderr)
            assert error in result.stderr
            assert not (study / "generated").exists()
        else:
            assert result.returncode == 0, (name, result.stdout, result.stderr)
            assert json.loads((study / "selected-candidate.json").read_text())["candidate_id"] == selected_id
            assert (study / "generated/run_0.sh").is_file()
            assert (study / "generated/bench_run.sh").is_file()


def test_engine_cli_case_matrix_is_complete() -> None:
    assert tuple(path.name for path in _PREDICT_CASES) == _EXPECTED_PREDICT_CASES
    assert tuple(path.name for path in _RECOMMEND_CASES) == _EXPECTED_RECOMMEND_CASES


@pytest.mark.parametrize(
    "state_enabled,expected_duration_ms",
    [(False, 2.0), (True, 4.0), ("auto-k3", 4.0)],
)
def test_manual_state_cache_runs_through_native_engine(
    tmp_path: Path, state_enabled: bool | str, expected_duration_ms: float
) -> None:
    config = tmp_path / "state-cache.yaml"
    config.write_text(
        """engine:
  mode: aggregated
  backend: vllm
  model: manual-state-smoke
  hardware: h200_sxm
  context_length: 2048
  workers:
    aggregated:
      scheduler: {max_batched_tokens: 256, max_sequences: 2}
      kv_cache:
        prefix_caching: false
        block_size: 64
        bytes_per_token: 16
        capacity: {type: fixed, bytes: 6144}
        state_cache:
          bytes_per_request: 1500
      timing: {type: fixed, prefill_ms: 1, decode_ms: 1}
traffic:
  source: {type: synthetic, input_tokens: 128, output_tokens: 1}
  load: {type: concurrency, concurrency: 2}
  stop: {requests: 4}
""",
        encoding="utf-8",
    )
    if not state_enabled:
        payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        del payload["engine"]["workers"]["aggregated"]["kv_cache"]["state_cache"]
        config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    if state_enabled == "auto-k3":
        payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        payload["engine"]["model"] = "moonshotai/Kimi-K3"
        worker = payload["engine"]["workers"]["aggregated"]
        worker["parallelism"] = {"tensor": 8}
        worker["kv_cache"].update(
            block_size=None, bytes_per_token="auto", state_cache={}, capacity={"type": "fixed", "blocks": 6}
        )
        config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    output = tmp_path / "state-cache"
    result = _run_cli(
        "predict",
        "--stack",
        "engine",
        "--config",
        str(config),
        "--output-dir",
        str(output),
        "--capture-per-request",
        "--format",
        "json",
        timeout=30.0,
    )
    summary = json.loads(result.stdout)
    report = json.loads((output / "prediction.json").read_text(encoding="utf-8"))
    assert summary["completed_requests"] == 4
    assert report.get("summary", report)["completed_requests"] == 4
    if state_enabled:
        state = report["state_cache"]["aggregated"]
        assert state["source"] == ("inferred" if state_enabled == "auto-k3" else "overridden")
    else:
        assert "state_cache" not in report
    if state_enabled == "auto-k3":
        assert state["bytes_per_request"] == 61046784
        assert state["block_size"] == 768
        assert state["kv_bytes_per_token"] == 27648
    # Six blocks fit two token-only requests, but only one with its state allocation.
    assert summary["duration_ms"] == pytest.approx(expected_duration_ms)
    assert report.get("summary", report)["duration_ms"] == pytest.approx(expected_duration_ms)


@pytest.mark.parametrize("config_path", _PREDICT_CASES, ids=lambda path: path.stem)
def test_engine_predict_cli_cases(config_path: Path, tmp_path: Path) -> None:
    output = tmp_path / config_path.stem
    result = _run_cli(
        "predict",
        "--stack",
        "engine",
        "--config",
        str(config_path.relative_to(_REPO_ROOT)),
        "--output-dir",
        str(output),
        "--format",
        "json",
    )

    summary = json.loads(result.stdout)
    report = json.loads((output / "prediction.json").read_text(encoding="utf-8"))
    saved_summary = report.get("summary", report)
    assert summary["completed_requests"] > 0
    assert saved_summary["completed_requests"] == summary["completed_requests"]
    if config_path.name == "11-synthetic-afd.yaml":
        replay_spec = json.loads((output / "afd-replay-spec.json").read_text(encoding="utf-8"))
        qualification = json.loads((output / "afd-qualification.json").read_text(encoding="utf-8"))
        assert replay_spec["backend_deployment"]["deployment_mode"] == "afd"
        assert qualification["qualification"] == {
            "execution": "analytical_foreground",
            "native_deployment_reason": (
                "AISimulate does not provide a physical AFD serving adapter or launch renderer"
            ),
            "native_deployment_supported": False,
            "status": "qualified_for_analytical_replay",
        }
    if "-trace-weka-" in config_path.name:
        assert "heuristically resolved one nested timestamp basis" in result.stderr
        assert "complete Weka corpus" in result.stderr
        assert "requested='auto', resolved='absolute'" in result.stderr


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("trace", ["weka-two-plays.jsonl", "weka-relative.json"])
def test_agentx_replay_cli_matches_public_python(backend: str, trace: str, tmp_path: Path) -> None:
    config = yaml.safe_load(
        (_REPO_ROOT / _CONFIG_ROOT / "predict/engine/12-trace-weka-jsonl-agentic-lane.yaml").read_text()
    )
    config["engine"]["backend"] = backend
    config["traffic"]["source"]["paths"] = [str(_REPO_ROOT / _CONFIG_ROOT / "fixtures/traces" / trace)]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "result"
    result = _run_cli(
        "predict",
        "--config",
        str(config_path),
        "--output-dir",
        str(output),
        "--capture-per-request",
        "--format",
        "json",
    )
    cli_report = json.loads(result.stdout)
    saved = json.loads((output / "prediction.json").read_text())
    assert cli_report == {key: value for key, value in saved.items() if key != "power_diagnostics"}
    assert saved["power_diagnostics"]["publication_status"] == "unsupported"
    assert cli_report["agentic_qualification"] == "functional_only"
    assert cli_report["agentic_lanes"] == 1
    assert cli_report["completed_requests"] == cli_report["agentic_graph"]["node_count"]
    assert all(row["status"] == "completed" for row in cli_report["agentic_play_outcomes"])
    assert [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()] == cli_report[
        "per_request"
    ]

    runner = EngineReplayRunnerFactory().create(0)
    try:
        spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
        python_report = runner.run(
            spec,
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True),
        ).metadata["native_report"]
    finally:
        runner.close()
    for key in (
        "agentic_graph",
        "agentic_lifecycle_digest",
        "agentic_play_outcomes",
        "per_request",
    ):
        assert cli_report[key] == python_report[key], key


def test_agentx_replay_cli_table_and_help_identify_qualification(tmp_path: Path) -> None:
    result = _run_cli(
        "predict",
        "--config",
        str(_CONFIG_ROOT / "predict/engine/12-trace-weka-jsonl-agentic-lane.yaml"),
        "--output-dir",
        str(tmp_path / "result"),
    )
    assert "AgentX functional replay only; not an AgentX benchmark result." in result.stdout
    help_text = " ".join(_run_cli("predict", "--help").stdout.split())
    for expected in (
        "weka",
        "agentic_mooncake",
        "HBM-only",
        "vLLM/SGLang",
        "speculative decoding disabled",
        "functional_only",
    ):
        assert expected in help_text


@pytest.mark.parametrize(
    "config_path",
    (*_RECOMMEND_CASES, _REPO_ROOT / "examples/cli/dynamo-deployment-recommend.yaml"),
    ids=lambda path: path.stem,
)
def test_engine_recommend_cli_cases_round_trip(config_path: Path, tmp_path: Path) -> None:
    CoreRecommendationConfig.from_yaml(config_path)
    output = tmp_path / config_path.stem
    result = _run_cli(
        "recommend",
        "--stack",
        "engine",
        "--config",
        str(config_path.relative_to(_REPO_ROOT)),
        "--output-dir",
        str(output),
        "--format",
        "json",
    )

    rows = json.loads(result.stdout)
    recommendation_paths = sorted((output / "recommendations").glob("*.yaml"))
    assert rows
    assert len(recommendation_paths) == len(rows)
    assert len({path.read_bytes() for path in recommendation_paths}) == len(recommendation_paths)

    generated_modes = set()
    for index, recommendation_path in enumerate(recommendation_paths):
        raw = yaml.safe_load(recommendation_path.read_text(encoding="utf-8"))
        _assert_concrete(raw)
        concrete = CorePredictionConfig.model_validate(raw)
        generated_modes.add(raw["engine"]["mode"])

        prediction_output = tmp_path / f"{config_path.stem}-predict-{index}"
        prediction = _run_cli(
            "predict",
            "--stack",
            "engine",
            "--config",
            str(recommendation_path),
            "--output-dir",
            str(prediction_output),
            "--format",
            "json",
        )
        assert json.loads(prediction.stdout)["completed_requests"] > 0
        if config_path.name == "dynamo-deployment-recommend.yaml":
            assert json.loads(prediction.stdout)["completed_requests"] == 40
            assert raw["engine"]["model"] == "Qwen/Qwen3-32B-FP8"
            assert raw["engine"]["backend_version"] == "0.24.0"
        if config_path.name == "08-heterogeneous-pd.yaml":
            candidate = SweepResult.from_json((output / "recommendation.json").read_text()).selected_candidates[index]
            deployment = prediction_to_replay_spec(concrete).backend_deployment
            assert raw["engine"]["hardware"] == "h200_sxm"
            assert raw["engine"]["workers"]["decode"]["hardware"] == "gb200"
            for role, hardware in (("prefill", "h200_sxm"), ("decode", "gb200")):
                engine_args = getattr(deployment, f"{role}_engine_args")
                assert "aic_system" not in engine_args
                timing = engine_args["timing_model"]
                assert timing["type"] == "external"
                assert timing["provider"] == "aic"
                assert timing["config"]["system"] == hardware
            assert candidate.config["prefill_hardware_sku"] == "h200_sxm"
            assert candidate.config["decode_hardware_sku"] == "gb200"
            assert candidate.used_gpus == 2
            metrics = json.loads(prediction.stdout)
            assert metrics["completed_requests"] == 6
            for key in ("mean_ttft_ms", "mean_tpot_ms", "output_throughput_tok_s"):
                assert metrics[key] == pytest.approx(candidate.metrics[key])
        if config_path.name == "07-afd-plus-pd.yaml":
            qualification = json.loads((prediction_output / "afd-qualification.json").read_text(encoding="utf-8"))
            assert qualification["identity"]["deployment_mode"] == "afd+pd"
            assert qualification["deployment_plan"]["pools"]["companion"]["role"] in {
                "prefill",
                "decode",
            }
            assert qualification["deployment_plan"]["launch"]["supported"] is False

    if config_path.name == "06-override-parallel-mappings-agg-disagg.yaml":
        assert generated_modes == {"aggregated", "disaggregated"}
        assert [row["score"] for row in rows] == sorted((row["score"] for row in rows), reverse=True)
    if config_path.name == "dynamo-deployment-recommend.yaml":
        _check_documented_candidate_renderer(output, tmp_path)


@pytest.mark.parametrize("load_type", ["concurrency", "constant_rate", "poisson"])
def test_min_gpus_real_engine_ranks_and_round_trips(load_type: str, tmp_path: Path) -> None:
    data = yaml.safe_load((_REPO_ROOT / _CONFIG_ROOT / "recommend/engine/03-preset-off-ttft.yaml").read_text())
    data["traffic"]["load"] = (
        {"type": "concurrency", "concurrency": 2}
        if load_type == "concurrency"
        else {"type": load_type, "requests_per_second": 10}
    )
    data["traffic"]["stop"] = {"requests": 20}
    data["evaluation"] = {"sla": {"e2e_ms": 100}}
    data["optimization"] = {"target": "min_gpus", "constraints": {"max_candidate_gpus": 2}}
    if load_type != "concurrency":
        data["optimization"]["constraints"]["min_goodput_rps"] = 5
    data["optimizer"]["max_trials"] = 8
    config = tmp_path / "minimum.yaml"
    config.write_text(yaml.safe_dump(data))
    output = tmp_path / "recommend"
    _run_cli("recommend", "--config", str(config), "--output-dir", str(output))
    result = SweepResult.from_json((output / "recommendation.json").read_text())
    counts = [candidate.used_gpus for candidate in result.selected_candidates]
    assert counts == sorted(counts)
    assert set(counts) == {1, 2}
    candidate = result.selected_candidates[0]
    assert candidate.used_gpus == 1
    assert candidate.score == -1
    assert all(c.metrics["mean_e2e_latency_ms"] <= 100 for c in result.selected_candidates)
    assert {row.used_gpus for row in result.candidates} == {1, 2}
    if load_type != "concurrency":
        assert all(c.metrics["goodput_request_throughput_rps"] >= 5 for c in result.selected_candidates)
    selected_path = sorted((output / "recommendations").glob("*.yaml"))[0]
    prediction = _run_cli(
        "predict", "--config", str(selected_path), "--output-dir", str(tmp_path / "predict"), "--format", "json"
    )
    assert json.loads(prediction.stdout)["completed_requests"] == 20

    # All candidates miss this SLA: fail explicitly without manufacturing a smallest result.
    no_result = tmp_path / "no-result"
    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisimulate",
            "recommend",
            "--config",
            str(config),
            "--set",
            "evaluation.sla.e2e_ms=0.001",
            "--output-dir",
            str(no_result),
        ],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert rejected.returncode == 1, rejected.stderr
    empty = SweepResult.from_json((no_result / "recommendation.json").read_text())
    assert empty.selected_candidates == []
    assert empty.counts.infeasible > 0
    infeasible = [c for c in empty.candidates if c.status is CandidateStatus.INFEASIBLE]
    assert len(infeasible) == empty.counts.infeasible
    assert all(c.metrics["mean_e2e_latency_ms"] > 0.001 for c in infeasible)
    assert not list((no_result / "recommendations").glob("*.yaml"))


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("warmup", [False, True])
@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
def test_agentic_snapshot_prediction_and_recommendation_keep_identical_evidence(
    tmp_path: Path, backend: str, warmup: bool, mode: str
) -> None:
    config_path = _REPO_ROOT / _CONFIG_ROOT / "predict/engine/12-trace-weka-jsonl-agentic-lane.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["engine"]["backend"] = backend
    config["engine"]["workers"]["aggregated"]["kv_cache"]["block_size"] = 2
    config["engine"]["workers"]["aggregated"]["kv_cache"]["capacity"]["blocks"] = 2048
    config["engine"]["mode"] = mode
    if mode == "disaggregated":
        worker = config["engine"]["workers"].pop("aggregated")
        config["engine"]["workers"] = {role: copy.deepcopy(worker) for role in ("prefill", "decode")}
    config_path = tmp_path / "predict.yaml"
    config_path.write_text(yaml.safe_dump(config))
    config["traffic"]["load"]["agentic_snapshot"] = {"seed": 42}
    config["traffic"]["load"]["agentic_warmup"] = warmup
    runner = EngineReplayRunnerFactory().create(0)
    try:
        report = runner.run(
            prediction_to_replay_spec(CorePredictionConfig.model_validate(config)),
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True),
        ).metadata["native_report"]
    finally:
        runner.close()
    output = tmp_path / "snapshot"
    _run_cli(
        "predict",
        "--stack",
        "engine",
        "--config",
        str(config_path),
        "--set",
        "traffic.load.agentic_snapshot.seed=42",
        "--set",
        f"traffic.load.agentic_warmup={str(warmup).lower()}",
        "--capture-per-request",
        "--output-dir",
        str(output),
        "--format",
        "json",
    )
    saved = json.loads((output / "prediction.json").read_text())
    assert saved["agentic_snapshots"] == report["agentic_snapshots"]
    assert saved.get("agentic_phases") == report.get("agentic_phases")
    if warmup:
        assert report["agentic_phases"]["phase"] == "profile"
        assert report["agentic_phases"]["profile_start_ms"] is not None
        assert report["agentic_phases"]["failure_request_id"] is None
        assert all(lane["warmup_completed"] == 10 for lane in report["agentic_phases"]["lanes"])
    else:
        assert "agentic_phases" not in report
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert records == report["per_request"]
    first_request = min(records, key=lambda record: record["first_admit_ms"])
    assert first_request["admission_history"][0]["reused_input_tokens"] == (2 if warmup else 0)

    for worker in config["engine"]["workers"].values():
        worker["parallelism"]["preset"] = False
    config["optimization"] = {
        "target": "throughput",
        "constraints": {"max_candidate_gpus": len(config["engine"]["workers"])},
    }
    config["optimizer"] = {"algorithm": "random", "max_trials": 1, "parallelism": 1, "seed": 11}
    recommendation_config = tmp_path / "recommend.yaml"
    recommendation_config.write_text(yaml.safe_dump(config))
    recommendation_output = tmp_path / "recommendation"
    _run_cli(
        "recommend",
        "--stack",
        "engine",
        "--config",
        str(recommendation_config),
        "--output-dir",
        str(recommendation_output),
        "--format",
        "json",
    )
    result = json.loads((recommendation_output / "recommendation.json").read_text())
    [candidate] = result["candidates"]
    assert candidate["status"] == "feasible", candidate.get("reason")
    assert candidate["provenance"]["workload"]["agentic_snapshot"] == {"seed": 42}
    assert candidate["provenance"]["workload"].get("agentic_warmup", False) is warmup
    assert candidate["provenance"]["runner_metadata"].get("agentic_phases") == report.get("agentic_phases")
    evidence = candidate["provenance"]["runner_metadata"]["agentic_snapshots"]
    assert evidence == report["agentic_snapshots"]
    assert [snapshot["seed"] for snapshot in evidence] == [42]
    assert candidate["metrics"]["completed_requests"] == report["completed_requests"]
    [recommended_path] = (recommendation_output / "recommendations").glob("*.yaml")
    recommended = yaml.safe_load(recommended_path.read_text())
    assert recommended["traffic"]["load"]["agentic_snapshot"] == {"seed": 42}
    assert recommended["traffic"]["load"].get("agentic_warmup", False) is warmup


@pytest.mark.parametrize("target", ["throughput", "ttft", "e2e_latency", "pareto"])
@pytest.mark.parametrize("output_format", ["json", "table"])
def test_agentic_warmup_rejection_exports_evidence_without_ranking(
    tmp_path: Path, target: str, output_format: str
) -> None:
    config = yaml.safe_load(
        (_REPO_ROOT / _CONFIG_ROOT / "predict/engine/12-trace-weka-jsonl-agentic-lane.yaml").read_text()
    )
    config["traffic"]["load"].update(agentic_snapshot={"seed": 42}, agentic_warmup=True)
    # The retained prompt has four tokens; vLLM rejects its preparation request
    # against this context limit before any profile request can start.
    config["engine"]["context_length"] = 2
    config["engine"]["workers"]["aggregated"]["kv_cache"]["block_size"] = 2
    runner = EngineReplayRunnerFactory().create(0)
    try:
        report = runner.run(
            prediction_to_replay_spec(CorePredictionConfig.model_validate(config)),
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True),
        )
    finally:
        runner.close()
    phases = report.metadata["agentic_phases"]
    assert phases["phase"] == "aborted"
    assert phases["profile_start_ms"] is None
    assert phases["failure_request_id"] == phases["requests"][0]["uuid"]
    assert phases["requests"][0]["terminal_status"] == "rejected"
    assert "Rejected" in phases["failure_reason"]
    assert report.metrics["completed_requests"] == 0
    assert report.metrics["mean_e2e_latency_ms"] == report.metrics["mean_ttft_ms"] == 0
    assert report.metadata["native_report"]["per_request"] == []

    config_path = tmp_path / "rejected.yaml"
    config_path.write_text(yaml.safe_dump(config))
    prediction_output = tmp_path / "prediction"
    predicted = _run_cli(
        "predict",
        "--stack",
        "engine",
        "--config",
        str(config_path),
        "--capture-per-request",
        "--output-dir",
        str(prediction_output),
        "--format",
        output_format,
        expected_returncode=1,
    )
    assert predicted.stdout == ""
    assert "agentic preparation aborted" in predicted.stderr
    prediction = json.loads((prediction_output / "prediction.json").read_text())
    assert prediction["agentic_phases"] == phases
    assert prediction["completed_requests"] == 0
    assert prediction["per_request"] == []
    assert (prediction_output / "requests.jsonl").read_text() == ""

    config["engine"]["workers"]["aggregated"]["parallelism"]["preset"] = False
    config["optimization"] = {"target": target, "constraints": {"max_candidate_gpus": 1}}
    config["optimizer"] = {"algorithm": "random", "max_trials": 1, "parallelism": 1, "seed": 11}
    config_path.write_text(yaml.safe_dump(config))
    recommendation_output = tmp_path / "recommendation"
    recommended = _run_cli(
        "recommend",
        "--stack",
        "engine",
        "--config",
        str(config_path),
        "--output-dir",
        str(recommendation_output),
        "--format",
        "json",
        expected_returncode=1,
    )
    assert "no feasible candidate found" in recommended.stderr
    result = json.loads((recommendation_output / "recommendation.json").read_text())
    [candidate] = result["candidates"]
    assert candidate["status"] == "failed"
    assert candidate["reason_category"] == "replay_runtime"
    assert candidate["score"] is None and candidate["objectives"] is None
    assert candidate["provenance"]["runner_metadata"]["agentic_phases"] == phases
    assert candidate["metrics"]["completed_requests"] == 0
    assert result["counts"]["failed"] == 1 and result["counts"]["feasible"] == 0
    assert result["views"] == {"top_n": [], "pareto_front": []}
    assert not list((recommendation_output / "recommendations").glob("*.yaml"))
