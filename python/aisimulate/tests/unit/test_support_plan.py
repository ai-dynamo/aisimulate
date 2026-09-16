# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The onboarding plan stays usable before model integration or GPU collection."""

from __future__ import annotations

import builtins
import fcntl
import json
import os
import select
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import ValidationError

from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.support.fpm import run_fpm
from aisimulate.support.plan import create_plan, plan_lock, request_id
from aisimulate.support.schema import SupportRequest

pytestmark = pytest.mark.unit


class _OfflineRunner:
    def run(self, spec):
        from aisimulate.sweeper.replay import ReplayReport

        return ReplayReport(metrics={"output_throughput_tok_s": 10.0, "gpu_hours": 1.0})

    def close(self):
        pass


class _OfflineRunnerFactory:
    def capabilities(self):
        from aisimulate.sweeper.replay import RunnerCapabilities

        return RunnerCapabilities(supported_backend_topologies=(("vllm", "agg"),))

    def create(self, worker_id):
        return _OfflineRunner()


def _request(**updates) -> SupportRequest:
    payload = {
        "identity": {
            "model": "unregistered/Example Model",
            "model_revision": "checkpoint-2026-09-14",
            "model_kind": "moe",
            "framework_version": "0.25.1",
            "gpu": "h200_sxm",
            "gpu_count": 12,
            "node_count": 2,
            "gpus_per_node": 6,
            "interconnect": "NVLink",
        },
        "search": {"tensor_parallel": 4},
    }
    for section, values in updates.items():
        payload.setdefault(section, {}).update(values)
    return SupportRequest.model_validate(payload)


@pytest.mark.parametrize(
    "updates",
    [
        {"identity": {"model_kind": "auto"}},
        {"identity": {"model_revision": "main"}},
        {"identity": {"framework_version": " "}},
        {"identity": {"gpu": "../h200_sxm"}},
        {"identity": {"gpu_count": 13}},
        {"search": {"tensor_parallel": 8}},
        {"search": {"tensor_parallel": True}},
        {"search": {"max_candidates": 3}},
        {"search": {"context_length": 1151}},
        {"workload": {"input_tokens": 0}},
        {"workload": {"request_count": 0}},
        {"workload": {"concurrency": 5}},
        {"workload": {"slo": {"ttft_ms": float("inf"), "tpot_ms": 1.0}}},
        {"workload": {"slo": {"ttft_ms": 1.0, "tpot_ms": True}}},
    ],
)
def test_request_rejects_invalid_allocation_workload_and_identity(updates):
    with pytest.raises(ValidationError):
        _request(**updates)


def test_plan_does_not_resolve_unknown_model_and_reloads_public_configs(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def no_model_or_collector_import(name, *args, **kwargs):
        assert not name.startswith(("collector", "huggingface_hub", "transformers")), name
        assert ".sdk.models" not in name, name
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_model_or_collector_import)
    request = _request()
    root = tmp_path / "plan with 'quotes'"
    plan = create_plan(request, root)
    prediction = CorePredictionConfig.from_yaml(root / "predict/pilot.yaml")
    recommendation = CoreRecommendationConfig.from_yaml(root / "recommend/pilot.yaml")
    assert SupportRequest.from_yaml(root / "request.yaml") == request
    assert json.loads((root / "support-plan.json").read_text()) == plan
    assert prediction.engine.systems_path == str(root / "systems")
    assert recommendation.engine.systems_path == prediction.engine.systems_path
    assert prediction.engine.model == "unregistered/Example Model"
    worker = prediction.engine.workers.aggregated
    assert worker.timing.type == "default"
    assert worker.timing.forward_model == "fpm"
    assert worker.parallelism.model_dump() == {
        "replicas": 1,
        "tensor": 4,
        "pipeline": 1,
        "attention_data": 1,
        "moe_tensor": 4,
        "moe_expert": 1,
    }
    assert recommendation.engine.workers.aggregated.timing.forward_model == "fpm"
    assert recommendation.optimizer.max_trials == 1
    assert recommendation.optimizer.parallelism == 1
    assert plan["search"]["candidate_count"] == 1
    assert "single" in plan["search"]["detail"]
    assert {item["status"] for item in plan["prerequisites"]} == {"not_checked"}
    assert plan["accuracy"]["status"] == "not_assessed"
    assert (root / "systems/h200_sxm.yaml").is_file()
    assert not list((root / "systems/data").iterdir())
    assert not (root / "evidence.yaml").exists()


@pytest.mark.parametrize("model_kind,preset,moe_tensor", [("dense", "tp", 1), ("moe", "pure_tp", 4)])
def test_plan_collects_one_chosen_worker_and_bounds_recommendation(tmp_path, model_kind, preset, moe_tensor):
    from aisimulate.recommend import recommendation_to_sweeper

    request = _request(identity={"model_kind": model_kind}, search={"max_candidates": 2})
    plan = create_plan(request, tmp_path)
    configs = [CoreRecommendationConfig.from_yaml(path) for path in plan["outputs"]["recommendation_configs"]]
    candidates = [candidate for config in configs for candidate in config.engine.workers.aggregated.parallelism.preset]
    assert [(c.replicas, c.tensor, c.moe_tensor) for c in candidates] == [(1, 4, moe_tensor), (2, 4, moe_tensor)]
    assert all(config.optimizer.max_trials == 1 for config in configs)
    assert plan["search"]["candidate_count"] == 2
    for config in configs:
        search = recommendation_to_sweeper(config).search_space
        assert search.agg_max_num_batched_tokens == [8192]
        assert search.agg_max_num_seqs == [256]
    commands = json.loads((tmp_path / "commands.json").read_text())
    command = commands["fpm_plan_local"]
    assert command[command.index("--fpm-gpu-counts") + 1] == "4"
    assert command[command.index("--fpm-max-gpus") + 1] == "4"
    assert command[command.index("--fpm-parallel-presets") + 1] == preset
    assert command[command.index("--fpm-max-prefill-isl") + 1] == "1024"
    assert command[command.index("--fpm-max-prefill-batch-size") + 1] == "1"
    assert "--plan-only" in command
    assert "--execute" in commands["fpm_run_local"]


def test_one_worker_allocation_keeps_one_pilot_even_with_two_candidate_limit(tmp_path):
    request = _request(identity={"node_count": 1, "gpus_per_node": 4, "gpu_count": 4}, search={"max_candidates": 2})
    plan = create_plan(request, tmp_path)
    commands = json.loads((tmp_path / "commands.json").read_text())

    assert plan["search"]["candidate_count"] == 1
    assert plan["outputs"]["recommendation_configs"] == [str(tmp_path / "recommend/pilot.yaml")]
    assert len(commands["recommend"]) == 1
    assert "single" in plan["search"]["detail"]


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("model_kind,moe_tensor", [("dense", 1), ("moe", 4)])
def test_emitted_recommendations_evaluate_both_replica_choices(tmp_path, monkeypatch, seed, model_kind, moe_tensor):
    from aisimulate.main import build_parser
    from aisimulate.recommend import run_recommendation
    from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig

    # Supply only offline model/runtime/data legality. Both replica choices are
    # legal for every command; the real config lowering, sampler and budget run.
    legal = [
        ReplicaParallelConfig(shape=ParallelShape(tp=4, pp=1, dp=1, moe_tp=moe_tensor, moe_ep=1), replicas=count)
        for count in (1, 2)
    ]
    monkeypatch.setattr("aisimulate.sweeper.search_space.parallel_configs_for", lambda *args, **kwargs: legal)
    request = _request(identity={"model_kind": model_kind}, search={"max_candidates": 2, "seed": seed})
    plan = create_plan(request, tmp_path)
    commands = json.loads((tmp_path / "commands.json").read_text())["recommend"]
    evaluated = []
    results = []
    configs = []
    outputs = []
    for command in commands:
        args = build_parser().parse_args(command[1:])
        config = CoreRecommendationConfig.from_yaml(args.config)
        result = run_recommendation(config, stack="engine", runner_factory=_OfflineRunnerFactory(), show_progress=False)
        evaluated.extend(candidate.config["replicas"] for candidate in result.candidates)
        results.append(result)
        configs.append(args.config)
        outputs.append(args.output_dir)
    assert evaluated == [1, 2]
    assert configs == plan["outputs"]["recommendation_configs"]
    assert outputs == plan["outputs"]["recommendation_results"]
    assert len(set(outputs)) == 2
    assert all(result.counts.evaluated == 1 and result.counts.cache_hits == 0 for result in results)


@pytest.mark.parametrize(
    "max_candidates,missing_pilot_samples,statuses",
    [(1, False, [0]), (2, False, [0, 0]), (2, True, [1, 0])],
)
@pytest.mark.parametrize("tampered_commands", [False, True])
def test_documented_recommendation_loop_attempts_every_candidate(
    tmp_path, max_candidates, missing_pilot_samples, statuses, tampered_commands
):
    guide = Path(__file__).resolve().parents[4] / "docs/self-service-support.md"
    snippet = guide.read_text().split("python3 - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    workdir = tmp_path / "work with 'quotes'; $(literal)"
    workdir.mkdir()
    request = _request(
        identity={"model_kind": "dense", "gpu_count": 8, "node_count": 1, "gpus_per_node": 8},
        search={"max_candidates": max_candidates, "objective": "ttft"},
    )
    plan = create_plan(request, workdir / "aisimulate-support")
    commands = json.loads(Path(plan["outputs"]["commands"]).read_text())["recommend"]
    injected_marker = tmp_path / "injected-command-ran"
    if tampered_commands:
        command_file = Path(plan["outputs"]["commands"])
        payload = json.loads(command_file.read_text())
        payload["recommend"].insert(
            0,
            [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('unexpected execution')",
                str(injected_marker),
            ],
        )
        command_file.write_text(json.dumps(payload))
    evaluated = tmp_path / "evaluated.txt"
    wrapper = tmp_path / "bin/aisimulate"
    wrapper.parent.mkdir()
    # Substitute only offline legality and replay metrics. The snippet executes
    # real subprocesses, public CLI dispatch, sampling, scoring and output export.
    wrapper.write_text(
        f"#!{sys.executable}\n"
        f"""
import os
import sys
from pathlib import Path

import aisimulate.main as cli
import aisimulate.sweeper.search_space as search_space
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import ReplayReport, RunnerCapabilities

legal = [
    ReplicaParallelConfig(shape=ParallelShape(tp=4, pp=1, dp=1, moe_tp=1, moe_ep=1), replicas=count)
    for count in (1, 2)
]
search_space.parallel_configs_for = lambda *args, **kwargs: legal

class Runner:
    def run(self, spec):
        replicas = spec.backend_deployment.num_workers
        with Path(os.environ["SUPPORT_TEST_EVALUATED"]).open("a") as stream:
            stream.write(str(replicas) + "\\n")
        samples = 0.0 if {missing_pilot_samples!r} and replicas == 1 else 4.0
        return ReplayReport(metrics={{"mean_ttft_ms": 10.0, "num_ttft_samples": samples}})

    def close(self):
        pass

class Factory:
    def capabilities(self):
        return RunnerCapabilities(supported_backend_topologies=(("vllm", "agg"),))

    def create(self, worker_id):
        return Runner()

cli.resolve_runner_factory = lambda stack: Factory()
if __name__ == "__main__":
    raise SystemExit(cli.main(sys.argv[1:]))
"""
    )
    wrapper.chmod(0o755)
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=workdir,
        env={
            **os.environ,
            "PATH": str(wrapper.parent) + os.pathsep + os.environ.get("PATH", ""),
            "SUPPORT_TEST_EVALUATED": str(evaluated),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert not injected_marker.exists()
    assert result.returncode == (1 if any(statuses) else 0), result.stderr
    assert evaluated.read_text().splitlines() == [str(index) for index in range(1, max_candidates + 1)]
    for index, (output, status) in enumerate(zip(plan["outputs"]["recommendation_results"], statuses, strict=True), 1):
        root = Path(output)
        payload = json.loads((root / "recommendation.json").read_text())
        assert payload["counts"]["evaluated"] == 1
        assert payload["counts"]["feasible"] == (0 if status else 1)
        assert payload["counts"]["infeasible"] == (1 if status else 0)
        assert payload["candidates"][0]["config"]["replicas"] == index
        exported = root / "recommendations/0001.yaml"
        if status:
            assert "no qualifying samples" in payload["candidates"][0]["reason"]
            assert not exported.exists()
        else:
            prediction = CorePredictionConfig.from_yaml(exported)
            assert prediction.engine.workers.aggregated.parallelism.replicas == index
            assert prediction.engine.systems_path == str(workdir / "aisimulate-support/systems")
    assert [line for line in result.stdout.splitlines() if line.startswith("exit ")] == [
        f"exit {status}: {shlex.join(command)}" for command, status in zip(commands, statuses, strict=True)
    ]


def test_prefill_bounds_follow_concurrent_workload_and_collector_minimum(tmp_path):
    request = _request(workload={"input_tokens": 10, "concurrency": 3, "request_count": 4})
    command = create_plan(request, tmp_path)["fpm"]["plan_command"]
    assert command[command.index("--fpm-max-prefill-isl") + 1] == "30"
    assert command[command.index("--fpm-max-prefill-batch-size") + 1] == "3"
    small = _request(workload={"input_tokens": 1})
    command = create_plan(small, tmp_path / "small")["fpm"]["plan_command"]
    assert command[command.index("--fpm-max-prefill-isl") + 1] == "2"


@pytest.mark.parametrize(
    "updates",
    [
        {"search": {"tensor_parallel": 2}},
        {"search": {"max_candidates": 2}},
        {"search": {"seed": 7}},
        {"workload": {"input_tokens": 64}},
        {"identity": {"model_revision": "other-checkpoint"}},
        {"identity": {"framework_version": "0.26.0"}},
        {"identity": {"tokenizer_revision": "tokenizer-v2"}},
    ],
)
def test_overwrite_rejects_different_requests_without_touching_data(tmp_path, updates):
    request = _request()
    create_plan(request, tmp_path)
    data = tmp_path / "systems/data/collected.parquet"
    data.write_bytes(b"timings")
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    changed = _request(**updates)
    assert request_id(changed) != request_id(request)
    with pytest.raises(ValueError, match="identity|different request|mix"):
        create_plan(changed, tmp_path, overwrite=True)
    after = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_same_request_overwrite_preserves_timings_and_rejects_modified_inputs(tmp_path):
    request = _request()
    plan = create_plan(request, tmp_path)
    data = tmp_path / "systems/data/collected.parquet"
    data.write_bytes(b"timings")
    with pytest.raises(ValueError, match="nonempty|already exists|overwrite"):
        create_plan(request, tmp_path)
    assert create_plan(request, tmp_path, overwrite=True) == plan
    assert data.read_bytes() == b"timings"
    (tmp_path / "request.yaml").write_text("modified: true\n")
    with pytest.raises(ValueError, match="modified|request|identity"):
        create_plan(request, tmp_path, overwrite=True)
    assert data.read_bytes() == b"timings"


@pytest.mark.parametrize("modified_command", [False, True])
def test_pre_onboard_plan_repairs_only_exact_legacy_commands(tmp_path, modified_command):
    request = _request()
    plan = create_plan(request, tmp_path)
    command_file = tmp_path / "commands.json"
    commands = json.loads(command_file.read_text())
    commands["fpm_run_local"][1] = "support"
    if modified_command:
        commands["fpm_run_local"].remove("--execute")
    command_file.write_text(json.dumps(commands, indent=2, sort_keys=True) + "\n")
    _seed_campaign(tmp_path)
    original = _file_contents(tmp_path)
    prediction = tmp_path / "predict/pilot.yaml"
    prediction.unlink()
    before_repair = _file_contents(tmp_path)

    if modified_command:
        with pytest.raises(ValueError, match="generated plan input .*commands.json was modified"):
            create_plan(request, tmp_path, overwrite=True)
        assert _file_contents(tmp_path) == before_repair
    else:
        assert create_plan(request, tmp_path, overwrite=True) == plan
        assert _file_contents(tmp_path) == original


def test_new_gpu_is_rejected_before_creating_outputs(tmp_path):
    root = tmp_path / "new-plan"
    with pytest.raises(ValueError, match="packaged.*system|system specification"):
        create_plan(_request(identity={"gpu": "unreleased_gpu"}), root)
    assert not root.exists()


def test_preview_is_shell_safe_and_does_not_create_outputs_or_import_collector(tmp_path, monkeypatch, capsys):
    request = _request()
    root = tmp_path / "plan with 'quotes'"
    real_import = builtins.__import__

    def no_collector_import(name, *args, **kwargs):
        assert not name.startswith("collector"), name
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_collector_import)
    assert run_fpm(request, output_dir=root) == 0
    command = shlex.split(capsys.readouterr().out)
    assert command[1:3] == ["-m", "collector.fpm_forward"]
    assert command[command.index("--model-path") + 1] == request.identity.model
    assert command[command.index("--fpm-database-root") + 1] == str(root / "systems/data")
    assert "--plan-only" in command
    assert not root.exists()


def test_execute_delegates_only_for_matching_plan(tmp_path, monkeypatch):
    from collector.fpm_forward import cli

    calls = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(argv) or 7)
    request = _request()
    with pytest.raises(ValueError, match="plan"):
        run_fpm(request, execute=True, output_dir=tmp_path)
    assert not calls
    create_plan(request, tmp_path)
    assert run_fpm(request, execute=True, output_dir=tmp_path, smoke=True, limit=1) == 7
    assert calls[0][calls[0].index("--limit") + 1] == "1"
    assert "--smoke" in calls[0]
    assert "--plan-only" not in calls[0]
    with pytest.raises(ValueError, match="identity|different request|mix"):
        run_fpm(_request(search={"tensor_parallel": 2}), execute=True, resume=True, output_dir=tmp_path)
    assert len(calls) == 1


def _seed_campaign(root, *, smoke=False, checkpoint_dir=None):
    sha = "a" * 64
    artifact = root / "fpm-artifacts" / sha[:16]
    if smoke:
        artifact /= "smoke"
    timing = artifact / "cells/cell-prefill/raw/existing-timing.json"
    timing.parent.mkdir(parents=True)
    timing.write_bytes(b"previous measured data")
    checkpoint = (checkpoint_dir or root / "fpm-checkpoint") / (
        "fpm_forward_smoke.json" if smoke else "fpm_forward.json"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({"schema": "aic-fpm-collector-checkpoint-v3", "plan_sha256": sha, "cells": {}}))
    if not smoke:
        (root / "systems/data/collected.parquet").write_bytes(b"formal timings")
    return checkpoint, timing


def _file_contents(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _reject_collector_import(monkeypatch):
    real_import = builtins.__import__

    def no_collector_import(name, *args, **kwargs):
        assert not name.startswith("collector"), "occupied campaigns must be rejected before collector import"
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_collector_import)


@pytest.mark.parametrize("command_name", ["onboard", "support"])
def test_published_execution_command_rejects_an_occupied_campaign(tmp_path, monkeypatch, capsys, command_name):
    from aisimulate.main import main

    create_plan(_request(), tmp_path)
    _seed_campaign(tmp_path)
    before = _file_contents(tmp_path)
    command = json.loads((tmp_path / "commands.json").read_text())["fpm_run_local"]
    assert command[:3] == ["aisimulate", "onboard", "collect-fpm"]
    command[1] = command_name
    _reject_collector_import(monkeypatch)

    with pytest.raises(SystemExit) as error:
        main(command[1:])

    assert error.value.code == 2
    assert "require --resume" in capsys.readouterr().err
    assert _file_contents(tmp_path) == before


@pytest.mark.parametrize("smoke", [False, True])
def test_repeated_execute_requires_resume_without_touching_campaign(tmp_path, monkeypatch, smoke):
    create_plan(_request(), tmp_path)
    _seed_campaign(tmp_path, smoke=smoke)
    before = _file_contents(tmp_path)
    _reject_collector_import(monkeypatch)

    with pytest.raises(ValueError, match="resume.*new output|new output.*resume"):
        run_fpm(SupportRequest.from_yaml(tmp_path / "request.yaml"), output_dir=tmp_path, execute=True, smoke=smoke)

    assert _file_contents(tmp_path) == before
    with plan_lock(tmp_path):
        pass


@pytest.mark.parametrize(
    "smoke,relative",
    [
        (False, "systems/data/collected.parquet"),
        (False, "fpm-checkpoint/custom/fpm_forward.json"),
        (True, "fpm-checkpoint/custom/fpm_forward_smoke.json"),
        (False, "fpm-artifacts/aaaaaaaaaaaaaaaa/cells/raw.json"),
        (True, "fpm-artifacts/aaaaaaaaaaaaaaaa/smoke/cells/raw.json"),
    ],
)
def test_partial_campaign_output_alone_prevents_restart(tmp_path, monkeypatch, smoke, relative):
    request = _request()
    create_plan(request, tmp_path)
    sentinel = tmp_path / relative
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_bytes(b"interrupted campaign")
    before = _file_contents(tmp_path)
    _reject_collector_import(monkeypatch)

    with pytest.raises(ValueError, match="resume"):
        run_fpm(request, output_dir=tmp_path, execute=True, smoke=smoke)

    assert _file_contents(tmp_path) == before


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("checkpoint_state", ["missing", "invalid_json", "empty_object", "new_directory"])
def test_resume_occupied_campaign_requires_usable_selected_checkpoint(tmp_path, monkeypatch, smoke, checkpoint_state):
    request = _request()
    create_plan(request, tmp_path)
    checkpoint, _ = _seed_campaign(tmp_path, smoke=smoke)
    selected_dir = checkpoint.parent
    if checkpoint_state == "missing":
        checkpoint.unlink()
    elif checkpoint_state == "new_directory":
        selected_dir /= "new-campaign"
    else:
        checkpoint.write_text("invalid JSON" if checkpoint_state == "invalid_json" else "{}")
    before = _file_contents(tmp_path)
    _reject_collector_import(monkeypatch)

    with pytest.raises(ValueError, match="checkpoint"):
        run_fpm(request, output_dir=tmp_path, execute=True, smoke=smoke, resume=True, checkpoint_dir=selected_dir)

    assert _file_contents(tmp_path) == before
    with plan_lock(tmp_path):
        pass


@pytest.mark.parametrize("smoke", [False, True])
def test_matching_explicit_resume_delegates_and_preserves_campaign(tmp_path, monkeypatch, smoke):
    from collector.fpm_forward import cli

    request = _request()
    create_plan(request, tmp_path)
    checkpoint, _ = _seed_campaign(tmp_path, smoke=smoke, checkpoint_dir=tmp_path / "fpm-checkpoint/custom")
    before = _file_contents(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(argv) or 7)

    assert (
        run_fpm(request, output_dir=tmp_path, execute=True, smoke=smoke, resume=True, checkpoint_dir=checkpoint.parent)
        == 7
    )

    assert "--resume" in calls[0]
    assert calls[0][calls[0].index("--checkpoint-dir") + 1] == str(checkpoint.parent)
    assert _file_contents(tmp_path) == before


@pytest.mark.parametrize("smoke", [False, True])
def test_fresh_smoke_and_formal_campaigns_are_independent(tmp_path, monkeypatch, smoke):
    from collector.fpm_forward import cli

    request = _request()
    create_plan(request, tmp_path)
    _seed_campaign(tmp_path, smoke=not smoke)
    before = _file_contents(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(argv) or 7)

    assert run_fpm(request, output_dir=tmp_path, execute=True, smoke=smoke) == 7

    assert len(calls) == 1
    assert _file_contents(tmp_path) == before


def test_execute_checks_for_campaign_outputs_while_holding_lock(tmp_path, monkeypatch):
    from aisimulate.support import plan

    request = _request()
    create_plan(request, tmp_path)
    real_lock = plan.plan_lock

    @contextmanager
    def raced_lock(root):
        with real_lock(root):
            _seed_campaign(root)
            yield

    monkeypatch.setattr(plan, "plan_lock", raced_lock)
    _reject_collector_import(monkeypatch)

    with pytest.raises(ValueError, match="resume"):
        run_fpm(request, output_dir=tmp_path, execute=True)

    assert (tmp_path / "systems/data/collected.parquet").read_bytes() == b"formal timings"
    with plan_lock(tmp_path):
        pass


@pytest.mark.parametrize("smoke,limit", [(False, 1), (True, 0), (True, -1), (True, True)])
def test_invalid_limited_collection_fails_before_preview_or_execution(tmp_path, smoke, limit):
    with pytest.raises(ValueError, match="limit"):
        run_fpm(_request(), output_dir=tmp_path, smoke=smoke, limit=limit)


def test_plan_rejects_symlinked_output_without_touching_target(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    data = external / "collected.parquet"
    data.write_bytes(b"timings")
    root = tmp_path / "plan"
    create_plan(_request(), root)
    (root / "systems/data").rmdir()
    (root / "systems/data").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        create_plan(_request(), root, overwrite=True)
    assert data.read_bytes() == b"timings"


@pytest.mark.parametrize("relative", ["../shared", "fpm-checkpoint/../../shared"])
def test_checkpoint_override_cannot_escape_request_directory(tmp_path, relative):
    with pytest.raises(ValueError, match="checkpoint_dir"):
        run_fpm(_request(), output_dir=tmp_path, checkpoint_dir=tmp_path / relative)


@pytest.mark.parametrize("kill_owner", [False, True])
def test_plan_lock_recovers_after_owner_exit_and_preserves_existing_files(tmp_path, kill_owner):
    request = _request()
    create_plan(request, tmp_path)
    before = (tmp_path / "support-plan.json").read_bytes()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; from aisimulate.support.plan import plan_lock; "
            "lock = plan_lock(Path(sys.argv[1])); lock.__enter__(); "
            "print('locked', flush=True); sys.stdin.read(); lock.__exit__(None, None, None)",
            str(tmp_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert select.select([process.stdout], [], [], 10)[0], "lock owner did not become ready"
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(ValueError, match="another onboarding operation"):
            create_plan(request, tmp_path, overwrite=True)
        if kill_owner:
            process.kill()
        process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)

    (tmp_path / "predict/pilot.yaml").unlink()
    create_plan(request, tmp_path, overwrite=True)
    assert (tmp_path / "predict/pilot.yaml").is_file()
    assert (tmp_path / "support-plan.json").read_bytes() == before


def test_plan_lock_does_not_unlink_the_inode_used_by_a_waiting_process(tmp_path):
    create_plan(_request(), tmp_path)
    with (tmp_path / ".support.lock").open("r") as waiting:
        with plan_lock(tmp_path):
            pass
        fcntl.flock(waiting, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another onboarding operation"):
            create_plan(_request(), tmp_path, overwrite=True)


@pytest.mark.parametrize("existing_target", [False, True])
def test_plan_lock_does_not_follow_a_symlink(tmp_path, existing_target):
    root = tmp_path / "plan"
    root.mkdir()
    external = tmp_path / "external"
    if existing_target:
        external.write_text("preserve this")
    (root / ".support.lock").symlink_to(external)

    with pytest.raises(OSError):
        create_plan(_request(), root)

    assert not (root / "request.yaml").exists()
    assert external.read_text() == "preserve this" if existing_target else not external.exists()


def test_racing_writer_cannot_be_overwritten_or_reported_as_success(tmp_path, monkeypatch):
    original_open = Path.open
    destination = tmp_path / "predict/pilot.yaml"

    def racing_open(path, mode="r", *args, **kwargs):
        if path == destination and mode == "xb":
            with original_open(path, "wb") as handle:
                handle.write(b"another writer's config")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    with pytest.raises(FileExistsError):
        create_plan(_request(), tmp_path)
    assert destination.read_bytes() == b"another writer's config"
    assert not (tmp_path / "support-plan.json").exists()
    with plan_lock(tmp_path):
        pass
