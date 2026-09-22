# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import psutil
import pytest
import yaml

from aisimulate.config.common import ResourceConfig
from aisimulate.resources import GB, MIB
from aisimulate.supervision import run_process


def _policy(extra_mib=64, **kwargs):
    # Account for the pytest coordinator rather than allocating a large fixture.
    return ResourceConfig(
        memory_limit_gb=(psutil.Process().memory_info().rss + extra_mib * MIB) / GB,
        cpu_limit=1,
        reserve_memory_gb=0.0,
        reserve_memory_fraction=0.0,
        **kwargs,
    )


def _run_script(tmp_path, source, *, policy=None, timeout=5):
    script = tmp_path / "worker.py"
    script.write_text(source)
    return run_process([sys.executable, str(script)], policy=policy or _policy(), timeout=timeout)


def test_watchdog_stops_small_allocation_and_reaps_child(tmp_path):
    pidfile = tmp_path / "pid"
    result = _run_script(
        tmp_path,
        f"""
import os, time
from pathlib import Path
Path({str(pidfile)!r}).write_text(str(os.getpid()))
chunks = []
for _ in range(32):
    chunks.append(bytearray(4 * 1024 * 1024))
    time.sleep(0.05)
""",
        policy=_policy(extra_mib=48),
    )
    assert result["status"] == "resource_limited"
    assert result["peak_observed_rss_bytes"] > result["budget"]["memory_limit_bytes"]
    assert result["termination_complete"] is True
    assert not psutil.pid_exists(int(pidfile.read_text()))


def test_timeout_kills_term_resistant_descendant(tmp_path):
    parent_pid = tmp_path / "parent"
    child_pid = tmp_path / "child"
    child_source = f"""import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path({str(child_pid)!r}).write_text(str(os.getpid()))
time.sleep(30)
"""
    result = _run_script(
        tmp_path,
        f"""
import os, signal, subprocess, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path({str(parent_pid)!r}).write_text(str(os.getpid()))
subprocess.Popen([sys.executable, "-c", {child_source!r}])
time.sleep(30)
""",
        timeout=0.5,
    )
    assert result["status"] == "timed_out"
    assert result["termination_complete"]
    assert not psutil.pid_exists(int(parent_pid.read_text()))
    child = int(child_pid.read_text())
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE


def test_runtime_threads_are_limited_before_import(tmp_path):
    output = tmp_path / "threads.json"
    result = _run_script(
        tmp_path,
        f"""
import json, os
from pathlib import Path
keys = ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS")
Path({str(output)!r}).write_text(json.dumps({{key: os.environ[key] for key in keys}}))
""",
    )
    assert result["status"] == "completed"
    assert set(json.loads(output.read_text()).values()) == {"1"}


def test_initialization_has_a_separate_deadline(tmp_path):
    result = _run_script(tmp_path, "import time; time.sleep(30)", policy=_policy(initialization_timeout_seconds=0.2))
    assert result["status"] == "timed_out"
    assert result["reason"] == "execution initialization timed out"


def test_child_exit_code_is_preserved(tmp_path):
    result = _run_script(tmp_path, "raise SystemExit(3)")
    assert result["status"] == "resource_limited"
    assert result["exit_code"] == 3


@pytest.mark.parametrize("failure", ["missing", "malformed", "override", "resources"])
@pytest.mark.parametrize("stack", ["engine", "missing-stack"])
def test_invalid_cli_input_preserves_outputs_and_stack_error_order(tmp_path, failure, stack):
    import subprocess

    config = tmp_path / "config.yaml"
    if failure != "missing":
        config.write_text("[" if failure == "malformed" else "execution:\n  resources:\n    memory_limit_gb: -1\n")
    output = tmp_path / "output"
    output.mkdir()
    original = {"prediction.json": '{"old":true}', "resource-runtime.json": '{"old_runtime":true}'}
    for name, contents in original.items():
        (output / name).write_text(contents)
    command = [
        sys.executable,
        "-m",
        "aisimulate",
        "predict",
        "--config",
        str(config),
        "--stack",
        stack,
        "--output-dir",
        str(output),
        "--overwrite",
    ]
    if failure == "override":
        command += ["--set", "invalid-assignment"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2, result.stderr
    assert {path.name: path.read_text() for path in output.iterdir()} == original
    if stack == "missing-stack":
        assert "missing-stack" in result.stderr
        assert "could not read configuration" not in result.stderr
        assert "malformed YAML" not in result.stderr
        assert "invalid --set" not in result.stderr


@pytest.mark.parametrize(
    "routing_yaml,duplicate",
    [
        ("router: {policy: round_robin}\nrouter: {policy: kv_router}\n", "router"),
        ("router: {policy: round_robin, policy: kv_router}\n", "policy"),
        ("router: {policy: kv_router, affinity: null, affinity: {mode: session}}\n", "affinity"),
        ("router: {policy: kv_router, affinity: {mode: session, mode: sibling_group}}\n", "mode"),
        ("router: {<<: {policy: round_robin, policy: kv_router}}\n", "policy"),
        ("router: {policy: kv_router, extra: [{mode: session, mode: sibling_group}]}\n", "mode"),
    ],
)
@pytest.mark.parametrize("input_source", ["config", "override"])
def test_duplicate_yaml_keys_fail_before_cli_outputs(tmp_path, routing_yaml, duplicate, input_source):
    import subprocess

    config = tmp_path / "config.yaml"
    config.write_text("engine: {model: example/model, hardware: h200_sxm, workers: {aggregated: {}}}\n")
    output = tmp_path / "output"
    output.mkdir()
    original = {"prediction.json": '{"old":true}', "resource-runtime.json": '{"old_runtime":true}'}
    for name, contents in original.items():
        (output / name).write_text(contents)
    command = [
        sys.executable,
        "-m",
        "aisimulate",
        "predict",
        "--config",
        str(config),
        "--output-dir",
        str(output),
        "--overwrite",
    ]
    if input_source == "config":
        config.write_text(config.read_text() + routing_yaml)
    else:
        command += ["--set", f"router={routing_yaml}"]

    result = subprocess.run(command, capture_output=True, text=True, timeout=15)

    assert result.returncode == 2, result.stderr
    assert f"duplicate mapping key {duplicate!r}" in result.stderr
    assert "Traceback" not in result.stderr
    assert {path.name: path.read_text() for path in output.iterdir()} == original


@pytest.mark.parametrize("input_source", ["config", "override"])
def test_cli_yaml_preserves_aliases_and_explicit_merge_overrides(tmp_path, input_source):
    from aisimulate.cli_args import _apply_overrides, _load_mapping

    source = """
first: &first {policy: kv_router, ttl_seconds: 60}
second: &second {policy: round_robin, mode: session}
router:
  <<: [*first, *second]
  ttl_seconds: 120
aliases: [*first, *first]
literal_keys: {=: value, '<<': literal}
"""
    if input_source == "config":
        config = tmp_path / "config.yaml"
        config.write_text(source)
        value = _load_mapping(str(config))
    else:
        data = {}
        _apply_overrides(data, [f"router={source}"], command="predict")
        value = data["router"]

    assert value["router"] == {"policy": "kv_router", "ttl_seconds": 120, "mode": "session"}
    assert value["aliases"][0] is value["first"]
    assert value["aliases"][1] is value["first"]
    assert value["literal_keys"] == {"=": "value", "<<": "literal"}
    _apply_overrides(value, ["router.mode=session", "router.mode=sibling_group"], command="predict")
    assert value["router"]["mode"] == "sibling_group"


def _cli_arguments(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("engine:\n  model: example/model\n  hardware: h200_sxm\n  workers:\n    aggregated: {}\n")
    return ["predict", "--config", str(config), "--output-dir", str(tmp_path / "output")]


@pytest.mark.parametrize("probe", ["discover_host", "resolve_budget"])
@pytest.mark.parametrize("detailed_plan", [False, True])
def test_early_resource_failure_keeps_a_complete_runtime_envelope(tmp_path, monkeypatch, probe, detailed_plan):
    from aisimulate import supervision
    from aisimulate.resources import HostResources, ResourceLimitError

    details = {"budget": {"memory_limit_bytes": 100}, "peak_observed_rss_bytes": 90} if detailed_plan else {}

    def refuse(*args, **kwargs):
        raise ResourceLimitError(
            "resource probe failed", plan={"status": "resource_limited", "reason": "probe", **details}
        )

    monkeypatch.setattr(supervision, "discover_host", lambda: HostResources(16 * GB, 8 * GB, 4))
    monkeypatch.setattr(supervision, probe, refuse)
    assert supervision.main(_cli_arguments(tmp_path)) == 3
    plan = json.loads((tmp_path / "output/resource-plan.json").read_text())
    assert plan["schema_version"] == 1
    assert plan["status"] == "resource_limited"
    assert plan["reason"] == "probe"
    assert plan["estimate"] is None
    assert plan["budget"] == details.get("budget")
    assert plan["requested_resources"] == ResourceConfig().model_dump(mode="json")
    if probe == "discover_host":
        assert plan["host"] is None
    else:
        assert plan["host"]["available_memory_bytes"] == 8 * GB
    report = json.loads((tmp_path / "output/resource-runtime.json").read_text())
    assert report["schema_version"] == 1
    assert report["status"] == "resource_limited"
    assert report["reason"] == "probe"
    assert report["requested_resources"] == ResourceConfig().model_dump(mode="json")
    for key in (
        "exit_code",
        "budget",
        "peak_observed_rss_bytes",
        "wall_seconds",
        "thread_limit_per_runtime",
        "termination_complete",
    ):
        assert report[key] == details.get(key)


def test_supervisor_reports_auto_selected_routing_stack(tmp_path, monkeypatch):
    from aisimulate import supervision
    from aisimulate.resources import ResourceLimitError

    args = _cli_arguments(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(config.read_text() + "router: {policy: kv_router}\n")

    def refuse(*args, **kwargs):
        raise ResourceLimitError("resource probe failed")

    monkeypatch.setattr(supervision, "run_process", refuse)
    assert supervision.main(args) == 3
    plan = json.loads((tmp_path / "output/resource-plan.json").read_text())
    assert plan["stack"] == "dynamo-policy"


@pytest.mark.parametrize("child_code,expected", [(-9, 1), (-15, 1), (2, 2)])
def test_cli_normalizes_signal_exit_but_retains_raw_diagnostics(tmp_path, monkeypatch, child_code, expected):
    from aisimulate import supervision

    monkeypatch.setattr(
        supervision,
        "run_process",
        lambda *args, **kwargs: {"status": "failed", "reason": "child failed", "exit_code": child_code},
    )
    assert supervision.main(_cli_arguments(tmp_path)) == expected
    report = json.loads((tmp_path / "output/resource-runtime.json").read_text())
    assert report["exit_code"] == child_code


def test_overwrite_clears_stale_results_before_early_resource_refusal(tmp_path):
    import subprocess

    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "engine": {
                    "mode": "aggregated",
                    "model": "example/model",
                    "hardware": "h200_sxm",
                    "workers": {"aggregated": {}},
                },
                "optimization": {"target": "throughput"},
                "execution": {"resources": {"memory_limit_gb": 0.000001}},
            }
        )
    )
    output = tmp_path / "output"
    recommendations = output / "recommendations"
    recommendations.mkdir(parents=True)
    (output / "recommendation.json").write_text('{"old":true}')
    (recommendations / "0001.yaml").write_text("old: true\n")
    (output / "notes.txt").write_text("keep")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisimulate",
            "recommend",
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--overwrite",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 3, result.stderr
    assert not (output / "recommendation.json").exists()
    assert not (recommendations / "0001.yaml").exists()
    assert (output / "notes.txt").read_text() == "keep"
    assert json.loads((output / "resource-runtime.json").read_text())["status"] == "resource_limited"
    plan = json.loads((output / "resource-plan.json").read_text())
    assert plan["status"] == "resource_limited"
    assert plan["budget"]["memory_limit_bytes"] == 1000
    assert plan["host"]["process_memory_bytes"] > 1000
    assert plan["estimate"] is None


def test_public_cli_runs_small_native_prediction_with_resource_evidence(tmp_path):
    import subprocess

    config = tmp_path / "predict.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "engine": {
                    "model": "example/model",
                    "hardware": "h200_sxm",
                    "context_length": 1024,
                    "workers": {
                        "aggregated": {
                            "kv_cache": {"capacity": {"type": "fixed", "blocks": 128}},
                            "timing": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
                        }
                    },
                },
                "traffic": {
                    "source": {"type": "synthetic", "input_tokens": 8, "output_tokens": 2},
                    "load": {"type": "concurrency", "concurrency": 2},
                    "stop": {"requests": 4},
                },
                "execution": {"resources": {"cpu_limit": 1}},
            }
        )
    )
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisimulate",
            "predict",
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "prediction.json").is_file()
    runtime = json.loads((output / "resource-runtime.json").read_text())
    assert runtime["status"] == "completed"
    assert runtime["budget"]["cpu_limit"] == 1
    assert runtime["peak_observed_rss_bytes"] > 0
    events = [json.loads(line) for line in (output / "execution-events.jsonl").read_text().splitlines()]
    assert any(event["event"] == "resource_plan" for event in events)


@pytest.mark.parametrize("profile", [{}, {"duration_seconds": 86400}])
def test_public_cli_profile_uses_unqualified_supervised_resource_plan(tmp_path, profile):
    import subprocess

    trace = tmp_path / "play.json"
    trace.write_text(
        json.dumps(
            {
                "id": "play",
                "models": ["model"],
                "block_size": 4,
                "hash_id_scope": "local",
                "requests": [{"t": 0, "type": "s", "model": "model", "in": 4, "out": 1, "hash_ids": [1]}],
            }
        )
    )
    config = tmp_path / "profile.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "engine": {
                    "model": "example/model",
                    "hardware": "h200_sxm",
                    "context_length": 1024,
                    "workers": {
                        "aggregated": {
                            "kv_cache": {"block_size": 4, "capacity": {"type": "fixed", "blocks": 128}},
                            "timing": {"type": "fixed", "prefill_ms": 600000, "decode_ms": 1},
                        }
                    },
                },
                "traffic": {
                    "source": {"type": "trace", "format": "weka", "paths": [str(trace)]},
                    "load": {
                        "type": "trace_timestamps",
                        "agentic_lanes": 1,
                        "agentic_snapshot": {"seed": 42},
                        "agentic_profile": profile,
                    },
                },
            }
        )
    )
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisimulate",
            "predict",
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    runtime = json.loads((output / "resource-runtime.json").read_text())
    assert runtime["status"] == "completed"
    assert runtime["peak_observed_rss_bytes"] > 0
    events = [json.loads(line) for line in (output / "execution-events.jsonl").read_text().splitlines()]
    plan = next(event["value"] for event in events if event["event"] == "resource_plan")
    assert plan["effective_parallelism"] == 1
    assert plan["estimate"]["estimated_peak_bytes"] is None
    assert plan["estimate"]["allocation_model"] == "agentic-profile-unqualified-v1"
    report = json.loads((output / "prediction.json").read_text())["agentic_profile"]
    assert report["options"]["duration_seconds"] == profile.get("duration_seconds", 3600)
    assert report["plays_started"] > 1
    assert report["successful_responses"] > 0


def test_sdk_recommendation_is_supervised_and_refuses_oversized_input():
    from aisimulate.config import CoreRecommendationConfig
    from aisimulate.recommend import run_recommendation
    from aisimulate.runner import EngineReplayRunnerFactory

    raw = yaml.safe_load(
        Path("tests/e2e/configs/unified_cli/recommend/engine/01-default-preset-throughput.yaml").read_text()
    )
    raw["traffic"] = {
        "source": {"type": "synthetic", "input_tokens": 10240, "output_tokens": 1024},
        "load": {"type": "concurrency", "concurrency": 64512},
        "stop": {"requests_per_load_unit": 100},
    }
    raw["optimizer"].update(max_trials=1, parallelism=1)
    raw["execution"] = {"resources": {"memory_limit_gb": 2.0}}
    config = CoreRecommendationConfig.model_validate(raw)
    result = run_recommendation(config, stack="engine", runner_factory=EngineReplayRunnerFactory(), show_progress=False)
    assert result.counts.resource_limited == 1
    assert result.counts.evaluated == 0
    assert result.candidates[0].status == "resource_limited"
    assert result.execution_resources["termination_complete"]
    assert result.execution_resources["peak_observed_rss_bytes"] > 0


def test_pool_termination_reaps_before_shutdown_returns():
    import multiprocessing
    import time

    from aisimulate.supervision import terminate_pool

    worker = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(30,))
    worker.start()

    class Pool:
        def __init__(self):
            self._processes = {worker.pid: worker}
            self.closed = False

        def shutdown(self, *, wait, cancel_futures):
            assert wait and cancel_futures
            assert not worker.is_alive()
            assert worker.exitcode is not None
            self.closed = True

    pool = Pool()
    terminate_pool(pool)
    assert pool.closed
    assert not psutil.pid_exists(worker.pid)


def test_completed_events_survive_interruption(tmp_path):
    output = tmp_path / "events.jsonl"
    script = tmp_path / "worker.py"
    script.write_text("""
import json, os, time
from pathlib import Path
budget = json.loads(os.environ["_AISIMULATE_SUPERVISED_BUDGET"])
Path(budget["ready_path"]).touch()
Path(budget["events_path"]).write_text('{"event":"candidate_completed","value":{"score":1}}\\n' + '{"event":')
time.sleep(30)
""")
    report = run_process([sys.executable, str(script)], policy=_policy(), timeout=0.5, events_output=str(output))
    assert report["status"] == "timed_out"
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"event": "candidate_completed", "value": {"score": 1}}
    ]


def test_cancellation_cleans_only_the_owned_process_tree(tmp_path, monkeypatch):
    import subprocess

    import aisimulate.supervision as supervision

    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    original = supervision.discover_host
    calls = 0
    owned = tmp_path / "pid"

    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 5:
            raise KeyboardInterrupt
        return original()

    monkeypatch.setattr(supervision, "discover_host", interrupt)
    try:
        result = _run_script(
            tmp_path,
            f"""
import os, time
from pathlib import Path
Path({str(owned)!r}).write_text(str(os.getpid()))
time.sleep(30)
""",
        )
        assert result["status"] == "cancelled"
        assert not psutil.pid_exists(int(owned.read_text()))
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait()


def test_sdk_completes_real_recommendation_and_preserves_resource_evidence():
    from aisimulate.config import CoreRecommendationConfig
    from aisimulate.recommend import run_recommendation
    from aisimulate.runner import EngineReplayRunnerFactory

    path = Path("tests/e2e/configs/unified_cli/recommend/engine/01-default-preset-throughput.yaml")
    raw = yaml.safe_load(path.read_text())
    raw["optimizer"].update(max_trials=1, parallelism=1)
    result = run_recommendation(
        CoreRecommendationConfig.model_validate(raw),
        stack="engine",
        runner_factory=EngineReplayRunnerFactory(),
        show_progress=False,
    )
    assert result.counts.feasible == 1
    assert result.execution_resources["status"] == "completed"
    assert result.execution_resources["termination_complete"]
    assert result.execution_resources["peak_observed_rss_bytes"] > 0
    assert result.selected_candidates


def test_shrinking_host_headroom_stops_new_work(tmp_path, monkeypatch):
    import dataclasses

    import aisimulate.supervision as supervision

    original = supervision.discover_host
    calls = 0

    def pressure():
        nonlocal calls
        calls += 1
        host = original()
        return dataclasses.replace(host, available_memory_bytes=0) if calls >= 5 else host

    monkeypatch.setattr(supervision, "discover_host", pressure)
    policy = _policy().model_copy(update={"reserve_memory_gb": 0.01})
    report = _run_script(tmp_path, "import time; time.sleep(30)", policy=policy)
    assert report["status"] == "resource_limited"
    assert "reserved headroom" in report["reason"]
    assert report["termination_complete"]


def test_shutdown_deadline_stops_a_hung_finalizer(tmp_path):
    result = _run_script(
        tmp_path,
        """
import json, os, time
from pathlib import Path
budget = json.loads(os.environ["_AISIMULATE_SUPERVISED_BUDGET"])
Path(budget["ready_path"]).write_text("shutdown")
time.sleep(30)
""",
        policy=_policy(shutdown_timeout_seconds=0.2),
    )
    assert result["status"] == "timed_out"
    assert result["reason"] == "execution shutdown timed out"
    assert result["termination_complete"]


def test_sdk_accepts_a_factory_defined_in_a_guarded_script(tmp_path):
    import subprocess

    script = tmp_path / "sdk.py"
    script.write_text("""
import json
from pathlib import Path
import yaml
from aisimulate.config import CoreRecommendationConfig
from aisimulate.recommend import run_recommendation
from aisimulate.runner import EngineReplayRunnerFactory

class ScriptFactory(EngineReplayRunnerFactory):
    pass

if __name__ == "__main__":
    path = Path("tests/e2e/configs/unified_cli/recommend/engine/01-default-preset-throughput.yaml")
    raw = yaml.safe_load(path.read_text())
    raw["optimizer"].update(max_trials=1, parallelism=1)
    result = run_recommendation(CoreRecommendationConfig.model_validate(raw), stack="engine",
                                runner_factory=ScriptFactory(), show_progress=False)
    print(json.dumps({"feasible": result.counts.feasible, "status": result.execution_resources["status"]}))
""")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"feasible": 1, "status": "completed"}


def test_supervisor_argument_and_output_setup_do_not_import_runtime():
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import aisimulate.supervision, aisimulate.cli_args, aisimulate.output; "
                "assert not any(name in sys.modules for name in "
                "('aisimulate._runtime', 'aisimulate.sweeper', 'numpy', 'pandas', 'aisimulate_core'))"
            ),
        ],
        check=True,
        timeout=15,
    )
