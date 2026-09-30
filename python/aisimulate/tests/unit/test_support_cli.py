# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public onboarding CLI behavior without a model download or GPU collection."""

from __future__ import annotations

import builtins
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import aisimulate.main as cli
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.support.schema import SupportRequest

pytestmark = pytest.mark.unit

_REQUIRED = {
    "model": "example/unintegrated-model",
    "model_revision": "revision-123",
    "model_kind": "dense",
    "framework_version": "0.24.0",
    "gpu": "h200_sxm",
    "gpu_count": 4,
    "interconnect": "nvswitch",
}
_PILOT = {
    "tensor_parallel": 2,
    "input_tokens": 1024,
    "output_tokens": 128,
    "concurrency": 1,
    "context_length": 16384,
    "ttft_ms": 1000,
    "tpot_ms": 100,
}


def _init_args(output: Path, *, full: bool = False, **changes) -> list[str]:
    options = {**_REQUIRED, **(_PILOT if full else {}), **changes}
    return ["onboard", "init", "--output", str(output)] + [
        part
        for name, value in options.items()
        if value is not None
        for part in ("--" + name.replace("_", "-"), str(value))
    ]


def _terminal(monkeypatch, answers=()) -> list[str]:
    prompts = []
    remaining = iter(answers)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def respond(prompt: str) -> str:
        prompts.append(prompt)
        try:
            answer = next(remaining)
        except StopIteration:
            pytest.fail(f"unexpected prompt: {prompt}")
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr("builtins.input", respond)
    return prompts


def test_supervised_entrypoint_runs_onboarding_before_prediction_config_handling(tmp_path, monkeypatch):
    from aisimulate import supervision

    def unexpected_supervision(*args, **kwargs):
        pytest.fail("interactive onboarding must retain the calling terminal")

    monkeypatch.setattr(supervision, "run_process", unexpected_supervision)
    output = tmp_path / "request.yaml"
    assert supervision.main(_init_args(output)) == 0
    assert SupportRequest.from_yaml(output).identity.model == _REQUIRED["model"]


@pytest.mark.parametrize("command", [[], ["onboard"]])
def test_help_explains_model_fpm_and_target_hardware_scope(capsys, command) -> None:
    with pytest.raises(SystemExit) as result:
        cli.main([*command, "--help"])

    assert result.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "Onboard a model for FPM simulation on a target hardware platform." in output
    assert "aisimulate" in output


@pytest.mark.parametrize("action", ["init", "plan", "collect-fpm"])
def test_retired_support_command_is_rejected_before_dispatch(tmp_path, monkeypatch, capsys, action) -> None:
    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("the retired command must be rejected before setup or collector dispatch")

    monkeypatch.setattr(cli, "run_support_command", unexpected_dispatch)
    monkeypatch.setattr(cli, "resolve_runner_factory", unexpected_dispatch)
    request = tmp_path / "request.yaml"
    if action == "init":
        args = _init_args(request)[1:]
    else:
        args = [action, "--config", str(request), "--output-dir", str(tmp_path / "plan")]
        if action == "collect-fpm":
            args.append("--execute")

    with pytest.raises(SystemExit) as result:
        cli.main(["support", *args])

    assert result.value.code == 2
    assert "invalid choice: 'support'" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_guided_and_scripted_setup_produce_the_same_request(tmp_path, monkeypatch, capsys) -> None:
    guided = tmp_path / "guided request.yaml"
    scripted = tmp_path / "scripted.yaml"
    prompts = _terminal(
        monkeypatch,
        ["example/unintegrated-model", "revision-123", "dense", "0.24.0", "h200_sxm", "4", "nvswitch", "2"] + [""] * 6,
    )

    assert cli.main(["onboard", "init", "--interactive", "--output", str(guided)]) == 0
    assert cli.main(_init_args(scripted, tensor_parallel=2)) == 0

    assert SupportRequest.from_yaml(guided) == SupportRequest.from_yaml(scripted)
    assert any("Input tokens per request [1024]" in prompt for prompt in prompts)
    assert any("Target time to first token (ms) [1000.0]" in prompt for prompt in prompts)
    request = SupportRequest.from_yaml(guided)
    assert request.workload.request_count == 4
    assert request.identity.gpus_per_node == 4
    assert request.identity.aisimulate_revision is None
    assert request.identity.tokenizer_revision is None
    output = capsys.readouterr().out
    assert "accuracy is not assessed" in output
    assert "are unchecked" in output


def test_supplied_options_skip_prompts_and_onboarding_alias_is_optional(tmp_path, monkeypatch) -> None:
    prompts = _terminal(monkeypatch)
    guided = tmp_path / "guided.yaml"
    scripted = tmp_path / "scripted.yaml"
    options = {"request_count": 8, "seed": 123, "max_candidates": 2, "objective": "goodput"}

    assert cli.main(_init_args(guided, full=True, **options) + ["--interactive", "--profile", "onboarding"]) == 0
    assert cli.main(_init_args(scripted, full=True, **options)) == 0

    assert prompts == []
    assert SupportRequest.from_yaml(guided) == SupportRequest.from_yaml(scripted)


def test_guided_setup_recovers_numeric_input_and_shared_field_validation(tmp_path, monkeypatch, capsys) -> None:
    output = tmp_path / "request.yaml"
    prompts = _terminal(monkeypatch, ["many", "4", "revision-123"])

    assert cli.main(_init_args(output, full=True, model_revision="main", gpu_count=None) + ["--interactive"]) == 0

    request = SupportRequest.from_yaml(output)
    assert request.identity.gpu_count == 4
    assert request.identity.model_revision == "revision-123"
    assert len(prompts) == 3
    transcript = capsys.readouterr().out
    assert "Enter a valid integer" in transcript
    assert "declare a pinned revision" in transcript


@pytest.mark.parametrize(
    ("changes", "answers", "expected"),
    [
        ({"tensor_parallel": 8}, ["unknown-option", "--tensor-parallel", "2"], "tensor_parallel"),
        ({"context_length": 100}, ["context-length", "4096"], "context_length"),
        ({"concurrency": 8}, ["request-count", "8"], "request_count"),
        ({"node_count": 2, "gpus_per_node": 4}, ["gpu-count", "8"], "gpu_count"),
    ],
)
def test_guided_setup_recovers_cross_field_validation(
    tmp_path, monkeypatch, capsys, changes, answers, expected
) -> None:
    output = tmp_path / "request.yaml"
    prompts = _terminal(monkeypatch, answers)

    assert cli.main(_init_args(output, full=True, **changes) + ["--interactive"]) == 0

    SupportRequest.from_yaml(output)
    assert any("Option to correct" in prompt for prompt in prompts)
    assert expected in capsys.readouterr().out


@pytest.mark.parametrize(
    "changes",
    [
        {"model": None},
        {"model_revision": "latest"},
        {"gpu_count": 0},
        {"ttft_ms": "nan"},
        {"request_count": 0},
        {"context_length": 100},
        {"tensor_parallel": 8},
        {"max_candidates": 3},
        {"objective": "invented"},
        {"node_count": 2},
    ],
)
def test_scripted_setup_uses_shared_validation_and_never_writes_invalid_requests(tmp_path, capsys, changes) -> None:
    output = tmp_path / "new" / "request.yaml"

    with pytest.raises(SystemExit) as error:
        cli.main(_init_args(output, **changes))

    assert error.value.code == 2
    assert not output.parent.exists()
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt])
@pytest.mark.parametrize("existing", [False, True])
def test_cancelled_setup_preserves_files_and_creates_no_partial_request(
    tmp_path, monkeypatch, capsys, interruption, existing
) -> None:
    output = tmp_path / "new" / "request.yaml"
    if existing:
        output.parent.mkdir()
        output.write_text("existing contents\n")
    _terminal(monkeypatch, ["example/unintegrated-model", interruption()])

    assert cli.main(["onboard", "init", "--interactive", "--output", str(output), "--overwrite"]) == 130

    if existing:
        assert output.read_text() == "existing contents\n"
        assert list(output.parent.iterdir()) == [output]
    else:
        assert not output.parent.exists()
    assert "Setup cancelled" in capsys.readouterr().err


def test_interactive_requires_a_terminal_without_reading_input(tmp_path, monkeypatch, capsys) -> None:
    _terminal(monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    output = tmp_path / "request.yaml"

    with pytest.raises(SystemExit) as error:
        cli.main(["onboard", "init", "--interactive", "--output", str(output)])

    assert error.value.code == 2
    assert "requires a terminal" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("target_kind", ["existing_file", "directory", "parent_file"])
def test_invalid_output_is_rejected_before_prompting(tmp_path, monkeypatch, target_kind) -> None:
    _terminal(monkeypatch)
    target = tmp_path / "request.yaml"
    if target_kind == "existing_file":
        target.write_text("keep me")
    elif target_kind == "directory":
        target.mkdir()
    else:
        parent = tmp_path / "file"
        parent.write_text("keep me")
        target = parent / "request.yaml"

    with pytest.raises(SystemExit) as error:
        cli.main(["onboard", "init", "--interactive", "--output", str(target)])

    assert error.value.code == 2


def test_output_install_does_not_overwrite_a_file_created_during_setup(tmp_path, monkeypatch) -> None:
    output = tmp_path / "request.yaml"
    real_link = os.link

    def competing_writer(source, destination) -> None:
        Path(destination).write_text("concurrent writer\n")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", competing_writer)

    with pytest.raises(SystemExit) as error:
        cli.main(_init_args(output))

    assert error.value.code == 2
    assert output.read_text() == "concurrent writer\n"
    assert list(tmp_path.iterdir()) == [output]


def test_explicit_overwrite_replaces_a_symlink_without_modifying_its_target(tmp_path) -> None:
    existing = tmp_path / "existing.yaml"
    existing.write_text("keep me\n")
    output = tmp_path / "request.yaml"
    output.symlink_to(existing)

    assert cli.main(_init_args(output) + ["--overwrite"]) == 0

    assert not output.is_symlink()
    SupportRequest.from_yaml(output)
    assert existing.read_text() == "keep me\n"


def test_init_next_command_quotes_the_request_path(tmp_path, capsys) -> None:
    output = tmp_path / "request 'with spaces'; $(unused).yaml"

    assert cli.main(_init_args(output)) == 0

    next_command = next(
        line.removeprefix("next: ") for line in capsys.readouterr().out.splitlines() if line.startswith("next: ")
    )
    assert shlex.split(next_command) == ["aisimulate", "onboard", "plan", "--config", str(output)]


def test_next_command_handles_a_relative_filename_starting_with_a_dash(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    args = _init_args(Path("unused.yaml"))
    args[2:4] = ["--output=-request.yaml"]

    assert cli.main(args) == 0

    next_command = next(
        line.removeprefix("next: ") for line in capsys.readouterr().out.splitlines() if line.startswith("next: ")
    )
    parsed = cli.build_parser().parse_args(shlex.split(next_command)[1:])
    assert Path(parsed.config) == tmp_path / "-request.yaml"


@pytest.mark.parametrize("output_dir", ["-plan", "plan 'quoted'; $(unused)"])
def test_printed_plan_next_command_runs_in_a_shell(tmp_path, monkeypatch, capsys, output_dir) -> None:
    monkeypatch.chdir(tmp_path)
    request = tmp_path / "request.yaml"
    assert cli.main(_init_args(request)) == 0
    capsys.readouterr()
    assert cli.main(["onboard", "plan", "-c", str(request), f"--output-dir={output_dir}", "--format", "json"]) == 0
    summary = json.loads(capsys.readouterr().out)

    result = subprocess.run(
        ["/bin/sh", "-c", summary["next"]],
        cwd=tmp_path,
        env={**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert Path(summary["plan"]) == tmp_path / output_dir / "support-plan.json"
    assert "collector.fpm_forward" in result.stdout
    assert "--plan-only" in result.stdout


@pytest.mark.parametrize("model_kind", ["dense", "moe"])
def test_init_plan_and_preview_use_real_public_configs_without_launching_collection(
    tmp_path, monkeypatch, capsys, model_kind
) -> None:
    def unexpected_launch(*args, **kwargs):
        pytest.fail("setup, plan, and preview must not launch a collector")

    monkeypatch.setattr(subprocess, "run", unexpected_launch)
    monkeypatch.setitem(sys.modules, "collector.fpm_forward.cli", SimpleNamespace(main=unexpected_launch))
    monkeypatch.setattr(cli, "resolve_runner_factory", unexpected_launch)
    request_path = tmp_path / "request 'quoted'.yaml"
    output = tmp_path / "plan 'quoted'"
    assert cli.main(_init_args(request_path, model_kind=model_kind, tensor_parallel=2)) == 0
    capsys.readouterr()

    assert (
        cli.main(["onboard", "plan", "--config", str(request_path), "--output-dir", str(output), "--format", "json"])
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    plan = json.loads((output / "support-plan.json").read_text())
    assert summary["request_id"] == plan["request_id"]
    assert summary["candidate_count"] == 1
    prediction = CorePredictionConfig.from_yaml(plan["outputs"]["prediction_configs"][0])
    recommendation = CoreRecommendationConfig.from_yaml(plan["outputs"]["recommendation_configs"][0])
    assert prediction.engine.workers.aggregated.timing.estimation_mode == "fpm_interpolation"
    assert recommendation.engine.workers.aggregated.timing.estimation_mode == "fpm_interpolation"
    assert prediction.engine.systems_paths == [plan["outputs"]["systems_root"]]
    assert recommendation.engine.systems_paths == prediction.engine.systems_paths
    assert prediction.traffic.source.input_tokens == 1024

    next_command = shlex.split(summary["next"])
    assert next_command[:3] == ["aisimulate", "onboard", "collect-fpm"]
    assert cli.main(next_command[1:]) == 0
    preview = capsys.readouterr().out
    assert "collector.fpm_forward" in preview
    assert "--fpm-max-gpus 2 --fpm-gpu-counts 2" in preview
    assert "--fpm-parallel-presets " + ("pure_tp" if model_kind == "moe" else "tp") in preview


def test_explicit_execute_forwards_diagnostic_options_and_exit_status(tmp_path, monkeypatch) -> None:
    calls = []

    def collector_main(argv):
        calls.append(argv)
        return 7

    monkeypatch.setitem(sys.modules, "collector.fpm_forward.cli", SimpleNamespace(main=collector_main))
    request_path = tmp_path / "request.yaml"
    output = tmp_path / "plan"
    assert cli.main(_init_args(request_path, tensor_parallel=2)) == 0
    assert cli.main(["onboard", "plan", "-c", str(request_path), "--output-dir", str(output)]) == 0

    assert (
        cli.main(
            [
                "onboard",
                "collect-fpm",
                "-c",
                str(request_path),
                "--output-dir",
                str(output),
                "--execute",
                "--smoke",
                "--limit",
                "1",
                "--resume",
            ]
        )
        == 7
    )

    assert len(calls) == 1
    assert "--plan-only" not in calls[0]
    assert "--smoke" in calls[0]
    assert "--resume" in calls[0]
    assert calls[0][calls[0].index("--limit") + 1] == "1"


def _local_collection_command(tmp_path, **changes):
    model = tmp_path / "model with spaces"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "num_hidden_layers": 2,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 512,
                "max_position_embeddings": 16384,
                "torch_dtype": "bfloat16",
            }
        )
    )
    request_path = tmp_path / "request.yaml"
    root = tmp_path / "plan"
    assert cli.main(_init_args(request_path, model=model, **changes)) == 0
    assert cli.main(["onboard", "plan", "-c", str(request_path), "--output-dir", str(root)]) == 0
    return ["onboard", "collect-fpm", "-c", str(request_path), "--output-dir", str(root), "--execute"]


def _synthetic_collector(monkeypatch, version):
    """Exercise real collection, provenance validation, and publication without GPUs."""
    from collector.fpm_forward import planner, runner

    executions = []

    def render_cell(_plan, _cell, cell_dir, _overrides, **_kwargs):
        (cell_dir / "k8s_deploy.yaml").write_text("apiVersion: v1\nkind: Pod\nmetadata:\n  name: test-cell\n")
        for name in ("run.sh", "fpm_env.sh", "collector-runtime-env.sh"):
            (cell_dir / name).write_text("#!/bin/sh\n")

    class LocalResource:
        def __init__(self, _manifest, cell_dir):
            self.cell = json.loads((cell_dir / "cell.json").read_text())
            self.raw = cell_dir / "raw/pod-0"

        def apply(self):
            self.raw.mkdir(parents=True)

        def wait_ready(self, _expected_nodes):
            return ["pod-0"]

        def stage(self, _pods, _files):
            pass

        def prepare_attempt(self, _pods, **identity):
            (self.raw / "collector-provenance.json").write_text(
                json.dumps(
                    {
                        "schema_name": "aic_fpm_collector_provenance",
                        "schema_version": 1,
                        **identity,
                        "runtime": {"backend": "vllm", "backend_version": version},
                    }
                )
            )

        def execute(self, _pods):
            executions.append(self.cell["cell_id"])
            phase = self.cell["workload_kind"]
            prefill = int(phase == "prefill")
            point = {
                "point_type": phase,
                "benchmark_id": 1,
                "batch_size": 1,
                "total_prefill_tokens": prefill,
                "total_kv_read_tokens": 1,
            }
            fpm = {
                "counter_id": 1,
                "dp_rank": 0,
                "wall_time": 0.01,
                "scheduled_requests": {
                    "num_prefill_requests": prefill,
                    "sum_prefill_tokens": prefill,
                    "sum_prefill_kv_tokens": prefill,
                    "num_decode_requests": 1 - prefill,
                    "sum_decode_kv_tokens": 1 - prefill,
                },
            }
            (self.raw / "benchmark.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "artifact_type": "rank",
                        "status": "complete",
                        "valid": True,
                        "usable": True,
                        "timing_valid": True,
                        "run_id": "synthetic-run",
                        "grid_digest": "synthetic-grid",
                        "config": {"mode": phase},
                        "coverage": {"expected_points": 1, "completed_points": 1, "skipped_points": 0},
                        "dp": {"rank": 0, "size": 1},
                        "results": [{"point": point, "fpms": [fpm]}],
                        "iteration_groups": [
                            {
                                "benchmark_id": 1,
                                "point": point,
                                "expected_dp_ranks": [0],
                                "complete": True,
                                "wall_time": 0.01,
                                "rank_results": [{"dp_rank": 0, "fpms": [fpm]}],
                            }
                        ],
                        "skipped_points": [],
                        "missing_phases": [],
                        "timing": {"benchmark_elapsed_seconds": 1.0, "measured_iteration_seconds": 0.01},
                        "kvwarm": {"enabled": True, "warm_eligible": True, "skip_reason": None},
                    }
                )
            )

        def collect(self, _pods, *, require_benchmark=True):
            pass

        def cleanup(self):
            pass

    monkeypatch.setattr(planner, "_git_revision", lambda: "test-source-revision")
    monkeypatch.setattr(runner, "_render_cell", render_cell)
    monkeypatch.setattr(runner, "KubernetesCellRunner", LocalResource)
    return executions


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize(
    "declared,observed",
    [
        ("0.25.1", "0.25.1"),
        ("0.25.1", "0.25.1+cu128"),
        ("0.25.1", "0.26.0"),
        ("0.25.1+cu128", "0.25.1+cu128"),
    ],
)
def test_formal_collection_checks_published_version_and_preserves_evidence_on_resume(
    tmp_path, monkeypatch, capsys, smoke, declared, observed
):
    executions = _synthetic_collector(monkeypatch, observed)
    command = _local_collection_command(tmp_path, framework_version=declared)
    root = tmp_path / "plan"
    if declared != observed:
        command += ["--checkpoint-dir", str(root / "fpm-checkpoint/custom")]
    if smoke:
        command += ["--smoke", "--limit", "1"]
    capsys.readouterr()
    expected = int(not smoke and declared != observed)

    assert cli.main(command) == expected

    stderr = capsys.readouterr().err
    if expected:
        assert f"pod-reported vllm version {observed!r}" in stderr
        assert f"framework_version {declared!r}" in stderr
        assert "preserved" in stderr
        assert "new plan" in stderr
        assert "Traceback" not in stderr
    provenance = list((root / "fpm-artifacts").rglob("collector-provenance.json"))
    assert provenance
    assert {json.loads(path.read_text())["runtime"]["backend_version"] for path in provenance} == {observed}
    metadata = root / "systems/data/h200_sxm/vllm" / observed / "fpm_forward_perf.metadata.json"
    if smoke:
        assert not list((root / "systems/data").iterdir())
    else:
        assert json.loads(metadata.read_text())["backend_version"] == observed
        if declared != observed:
            # A stale matching directory must not hide this campaign's different runtime.
            (root / "systems/data/h200_sxm/vllm" / declared).mkdir()
    preserved = {
        path: path.read_bytes() for path in root.rglob("*") if path.is_file() and path.name not in {"run-manifest.json"}
    }
    initial_executions = executions.copy()

    assert cli.main([*command, "--resume"]) == expected
    assert executions == initial_executions
    assert all(path.read_bytes() == contents for path, contents in preserved.items())
    if not smoke:
        # A completed formal checkpoint remains resumable after raw-artifact reclamation.
        for path in provenance:
            shutil.rmtree(path.parent.parent)
        assert cli.main([*command, "--resume"]) == expected
        assert executions == initial_executions
        assert metadata.read_bytes() == preserved[metadata]


def test_formal_collection_requires_publication_evidence(tmp_path, monkeypatch, capsys):
    from collector.fpm_forward import runner

    command = _local_collection_command(tmp_path)
    monkeypatch.setattr(runner, "run_collection", lambda *args, **kwargs: [])
    capsys.readouterr()

    assert cli.main(command) == 1
    stderr = capsys.readouterr().err
    assert "aisimulate onboard collect-fpm failed:" in stderr
    assert "Traceback" not in stderr


def test_deployment_options_reach_frozen_collector_plan(tmp_path, monkeypatch):
    from collector.fpm_forward import runner

    calls = []
    monkeypatch.setattr(runner, "run_collection", lambda plan, **kwargs: calls.append((plan, kwargs)) or [])
    command = _local_collection_command(tmp_path)
    command += [
        "--smoke",
        "--dynamo-version",
        "1.2.0",
        "--image",
        "registry.example/fpm@sha256:" + "a" * 64,
        "--namespace",
        "pilot-collection",
        "--model-cache",
        "model-cache:/models:checkpoint",
        "--transport",
        "nvlink",
        "--image-pull-secret",
        "registry-secret",
    ]

    assert cli.main(command) == 0

    [(plan, kwargs)] = calls
    assert kwargs["generator_overrides"] == {
        "generator_dynamo_version": "1.2.0",
        "K8sConfig": {
            "k8s_image": "registry.example/fpm@sha256:" + "a" * 64,
            "k8s_namespace": "pilot-collection",
            "k8s_pvc_name": "model-cache",
            "k8s_pvc_mount_path": "/models",
            "k8s_model_path_in_pvc": "checkpoint",
            "transport": "nvlink",
            "k8s_image_pull_secret": "registry-secret",
        },
    }
    assert plan.cells
    assert kwargs["artifact_root"] == str(tmp_path / "plan/fpm-artifacts")
    assert kwargs["database_root"] == str(tmp_path / "plan/systems/data")


@pytest.mark.parametrize(
    "option,value",
    [
        ("--dynamo-version", "latest"),
        ("--image", "bad image"),
        ("--image", ""),
        ("--namespace", "invalid namespace"),
        ("--model-cache", ":/models"),
        ("--model-cache", "cache:a:b:c"),
        ("--model-cache", "cache:relative"),
        ("--transport", "pcie"),
        ("--image-pull-secret", "invalid secret"),
        ("--generator-set", "params.agg.tp=8"),
        ("--generator-config", "unrestricted.yaml"),
    ],
)
def test_deployment_options_reject_invalid_or_unrestricted_inputs(tmp_path, capsys, option, value):
    command = _local_collection_command(tmp_path)
    command.remove("--execute")

    with pytest.raises(SystemExit) as error:
        cli.main([*command, option, value])

    assert error.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
    assert not (tmp_path / "plan/fpm-checkpoint").exists()
    assert not (tmp_path / "plan/fpm-artifacts").exists()


@pytest.mark.parametrize(
    "option,first,changed",
    [
        ("--dynamo-version", "1.2.0", "1.3.0"),
        ("--image", "registry.example/fpm:first", "registry.example/fpm:second"),
        ("--namespace", "first", "second"),
        ("--model-cache", "cache:/models:first", "cache:/models:second"),
        ("--transport", "nvlink", "ib"),
        ("--image-pull-secret", "first", "second"),
    ],
)
def test_deployment_resume_requires_the_same_frozen_identity(tmp_path, monkeypatch, capsys, option, first, changed):
    from collector.fpm_forward import planner, runner

    command = [*_local_collection_command(tmp_path), "--smoke", option, first]
    checkpoint = tmp_path / "plan/fpm-checkpoint/fpm_forward_smoke.json"
    plans = []
    actual_run = runner.run_collection

    def checkpoint_only(plan, **kwargs):
        # No silicon data is produced: exercise planning and checkpoint guards
        # with a synthetic empty campaign checkpoint in place of GPU execution.
        plans.append(plan)
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_text(json.dumps({"schema": runner.CHECKPOINT_SCHEMA, "plan_sha256": plan.sha256, "cells": {}}))
        return []

    monkeypatch.setattr(planner, "_git_revision", lambda: "test-source-revision")
    monkeypatch.setattr(runner, "run_collection", checkpoint_only)
    assert cli.main(command) == 0
    saved = checkpoint.read_bytes()
    with pytest.raises(SystemExit) as error:
        cli.main(command)
    assert error.value.code == 2
    assert len(plans) == 1
    assert cli.main([*command, "--resume"]) == 0
    assert plans[0].sha256 == plans[1].sha256
    assert checkpoint.read_bytes() == saved

    # The real collector must reject a changed identity before GPU execution.
    monkeypatch.setattr(runner, "run_collection", actual_run)
    monkeypatch.setattr(runner, "KubernetesCellRunner", lambda *args, **kwargs: pytest.fail("must not launch GPU work"))
    capsys.readouterr()
    assert cli.main([*command, "--resume", option, changed]) == 1
    assert "checkpoint does not match the current frozen plan" in capsys.readouterr().err
    assert checkpoint.read_bytes() == saved


@pytest.mark.parametrize("stage", ["input", "execution"])
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, OSError, KeyboardInterrupt])
def test_collector_failures_keep_public_cli_exit_codes(tmp_path, monkeypatch, capsys, stage, error_type):
    from collector.fpm_forward import cli as collector_cli

    request_path = tmp_path / "request.yaml"
    root = tmp_path / "plan"
    assert cli.main(_init_args(request_path)) == 0
    assert cli.main(["onboard", "plan", "-c", str(request_path), "--output-dir", str(root)]) == 0
    capsys.readouterr()

    def fail(*args, **kwargs):
        raise error_type("test collection failure")

    monkeypatch.setattr(collector_cli, "build_collection_case_plan", lambda **kwargs: SimpleNamespace(model_path=None))
    monkeypatch.setattr(collector_cli, "resolve_run_inputs", fail if stage == "input" else lambda *args: None)
    monkeypatch.setattr(collector_cli, "run_resolved", fail)
    command = ["onboard", "collect-fpm", "-c", str(request_path), "--output-dir", str(root), "--execute"]

    if error_type is KeyboardInterrupt:
        assert cli.main(command) == 130
    elif stage == "input":
        with pytest.raises(SystemExit) as error:
            cli.main(command)
        assert error.value.code == 2
    else:
        assert cli.main(command) == 1
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    if stage == "execution" and error_type is not KeyboardInterrupt:
        assert "aisimulate onboard collect-fpm failed: test collection failure" in stderr


@pytest.mark.parametrize("error_type", [ImportError, KeyboardInterrupt])
def test_collector_import_failures_keep_public_cli_exit_codes(tmp_path, monkeypatch, capsys, error_type):
    command = _local_collection_command(tmp_path)
    real_import = builtins.__import__

    def fail_collector_import(name, *args, **kwargs):
        if name == "collector.fpm_forward.cli":
            raise error_type("collector dependency unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_collector_import)
    capsys.readouterr()

    assert cli.main(command) == (130 if error_type is KeyboardInterrupt else 1)
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    if error_type is ImportError:
        assert "aisimulate onboard collect-fpm failed: collector dependency unavailable" in stderr


def test_malformed_resumed_checkpoint_fails_cleanly_without_launching_collection(tmp_path, capsys):
    from collector.fpm_forward.runner import CHECKPOINT_SCHEMA

    command = _local_collection_command(tmp_path)
    capsys.readouterr()
    assert cli.main([argument for argument in command if argument != "--execute"]) == 0
    preview = shlex.split(capsys.readouterr().out)
    launch_marker = tmp_path / "cluster-command-launched"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl = bin_dir / "kubectl"
    kubectl.write_text('#!/bin/sh\n: > "$FPM_TEST_LAUNCH_MARKER"\nexit 99\n')
    kubectl.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
        "FPM_KUBECTL": "kubectl",
        "FPM_TEST_LAUNCH_MARKER": str(launch_marker),
    }
    planned = subprocess.run(
        [sys.executable, *preview[1:]],
        cwd=tmp_path,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert planned.returncode == 0, planned.stderr
    frozen = json.loads(planned.stdout)
    root = tmp_path / "plan"
    checkpoint = root / "fpm-checkpoint/fpm_forward.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text(
        json.dumps(
            {
                "schema": CHECKPOINT_SCHEMA,
                "plan_sha256": frozen["sha256"],
                "cells": {frozen["cells"][0]["cell_id"]: []},
            }
        )
    )
    original = checkpoint.read_bytes()

    resumed = subprocess.run(
        [sys.executable, "-m", "aisimulate", *command, "--resume"],
        cwd=tmp_path,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert resumed.returncode == 1, resumed.stderr
    assert "Traceback" not in resumed.stderr
    assert "aisimulate onboard collect-fpm failed:" in resumed.stderr
    assert checkpoint.read_bytes() == original
    assert not launch_marker.exists()
    assert not list((root / "fpm-artifacts").rglob("*.yaml"))


def test_real_collector_artifact_symlink_rejected_before_first_write(tmp_path, monkeypatch, capsys):
    from collector.fpm_forward import runner

    command = _local_collection_command(tmp_path)
    capsys.readouterr()
    assert cli.main([argument for argument in command if argument != "--execute"]) == 0
    preview = shlex.split(capsys.readouterr().out)
    planned = subprocess.run(
        [sys.executable, *preview[1:]],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert planned.returncode == 0, planned.stderr
    frozen = json.loads(planned.stdout)
    root = tmp_path / "plan"
    external = tmp_path / "external"
    external.mkdir()
    (root / "fpm-artifacts").mkdir()
    (root / "fpm-artifacts" / frozen["sha256"][:16]).symlink_to(external, target_is_directory=True)
    reached_checkpoint = []

    def stop_before_gpu(*args, **kwargs):
        reached_checkpoint.append(True)
        raise RuntimeError("collector reached checkpoint loading")

    # The real collector writes collection-plan.json before loading its
    # checkpoint. Keep that path intact and stop before any GPU work.
    monkeypatch.setattr(runner, "_load_checkpoint", stop_before_gpu)

    with pytest.raises(SystemExit) as error:
        cli.main(command)

    assert error.value.code == 2
    assert "refusing symlinked plan output" in capsys.readouterr().err
    assert not reached_checkpoint
    assert not list(external.iterdir())
    assert not (root / "fpm-checkpoint").exists()


@pytest.mark.parametrize("contents", ["[one, two]\n", "identity: [\n", "identity: {}\n"])
def test_bad_request_yaml_is_reported_without_a_traceback(tmp_path, capsys, contents) -> None:
    request = tmp_path / "bad.yaml"
    request.write_text(contents)

    with pytest.raises(SystemExit) as error:
        cli.main(["onboard", "plan", "--config", str(request), "--output-dir", str(tmp_path / "plan")])

    assert error.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
    assert not (tmp_path / "plan").exists()


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_ordinary_cli_options_and_numerical_defaults_are_preserved(command) -> None:
    args = cli.build_parser().parse_args([command, "--config", "ordinary.yaml"])
    assert args.command == command
    assert args.stack == "engine"
    assert args.output_dir == "./aisimulate-output"
    assert args.overrides == []
    assert args.overwrite is False
    assert args.format == "table"
    config = CorePredictionConfig.model_validate(
        {"engine": {"model": "example/model", "hardware": "h200_sxm", "workers": {"aggregated": {}}}}
    )
    assert config.engine.workers.aggregated.timing.forward_model == "op_level"
    assert config.traffic.load.concurrency == 10
    assert config.traffic.stop.requests == 100


def test_installed_module_scripted_init_works_without_a_terminal(tmp_path) -> None:
    output = tmp_path / "request.yaml"
    result = subprocess.run(
        [sys.executable, "-m", "aisimulate.main", *_init_args(output)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert SupportRequest.from_yaml(output).identity.model == _REQUIRED["model"]
    assert "next:" in result.stdout
