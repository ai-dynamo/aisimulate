# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only end-to-end coverage for the public engine-stack CLI."""

from __future__ import annotations

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


def _run_cli(*args: str, timeout: float = 120.0, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "aisimulate", *args],
        cwd=_REPO_ROOT,
        env=None if env is None else {**os.environ, **env},
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, (
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
    for key in ("agentic_graph", "agentic_lifecycle_digest", "agentic_play_outcomes", "per_request"):
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
            assert deployment.prefill_engine_args["aic_system"] == "h200_sxm"
            assert deployment.decode_engine_args["aic_system"] == "gb200"
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


_FPM_CASE = _CONFIG_ROOT / "predict/fpm/01-minimax-m27-h200-tp4-fpm.yaml"


def test_engine_predict_accepts_forward_model_from_yaml_and_set(tmp_path: Path) -> None:
    # The bundled FPM cell is collected outside the queryable version slots.
    env = {"AIC_ALLOW_UNLISTED_VERSIONS": "1"}
    fpm = json.loads(
        _run_cli(
            "predict",
            "--stack",
            "engine",
            "--config",
            str(_FPM_CASE),
            "--output-dir",
            str(tmp_path / "fpm"),
            "--format",
            "json",
            env=env,
        ).stdout
    )
    op_level = json.loads(
        _run_cli(
            "predict",
            "--stack",
            "engine",
            "--config",
            str(_FPM_CASE),
            "--set",
            "engine.workers.aggregated.timing.forward_model=op_level",
            "--output-dir",
            str(tmp_path / "op-level"),
            "--format",
            "json",
            env=env,
        ).stdout
    )

    # Both runs prove CLI plumbing only (the YAML field and the --set path are accepted and the
    # replay completes). Whether the FPM data path is actually engaged is proven in-process by
    # tests/test_unified_traffic_runtime.py (fail-closed on an uncovered identity); accuracy is a
    # FPM-vs-silicon question and is not asserted anywhere in the test suite.
    assert fpm["completed_requests"] == 8
    assert op_level["completed_requests"] == 8


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_agentic_snapshot_prediction_and_recommendation_keep_identical_evidence(tmp_path: Path, backend: str) -> None:
    config_path = _REPO_ROOT / _CONFIG_ROOT / "predict/engine/12-trace-weka-jsonl-agentic-lane.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["engine"]["backend"] = backend
    config["traffic"]["load"]["agentic_snapshot"] = {"seed": 42}
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
        f"engine.backend={backend}",
        "--set",
        "traffic.load.agentic_snapshot.seed=42",
        "--capture-per-request",
        "--output-dir",
        str(output),
        "--format",
        "json",
    )
    saved = json.loads((output / "prediction.json").read_text())
    assert saved["agentic_snapshots"] == report["agentic_snapshots"]
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert records == report["per_request"]

    config["engine"]["workers"]["aggregated"]["parallelism"]["preset"] = False
    config["optimization"] = {"target": "throughput", "constraints": {"max_candidate_gpus": 1}}
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
    evidence = candidate["provenance"]["runner_metadata"]["agentic_snapshots"]
    assert evidence == report["agentic_snapshots"]
    assert [snapshot["seed"] for snapshot in evidence] == [42]
    assert candidate["metrics"]["completed_requests"] == report["completed_requests"]
    [recommended_path] = (recommendation_output / "recommendations").glob("*.yaml")
    recommended = yaml.safe_load(recommended_path.read_text())
    assert recommended["traffic"]["load"]["agentic_snapshot"] == {"seed": 42}
