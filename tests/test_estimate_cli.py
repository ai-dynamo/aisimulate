# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public fixed-point estimates must preserve the compatibility estimator's inputs and reports."""

from __future__ import annotations

import argparse
import subprocess
import sys

import pytest

import aiconfigurator.cli.api as estimator
import aiconfigurator.cli.main as compatibility
import aisimulate.main as cli
from aiconfigurator.cli.api import EstimateResult
from aiconfigurator.sdk.errors import PerfDataNotAvailableError

pytestmark = pytest.mark.unit

_BASE = [
    "estimate",
    "--model-path",
    "meta-llama/Meta-Llama-3.1-8B",
    "--system",
    "h200_sxm",
    "--backend",
    "vllm",
    "--backend-version",
    "0.24.0",
    "--batch-size",
    "4",
    "--tp-size",
    "2",
    "--isl",
    "128",
    "--osl",
    "16",
    "--no-color",
    "--log-level",
    "ERROR",
]


def _result(kwargs):
    return EstimateResult(
        ttft=10.0,
        tpot=2.0,
        power_w=None,
        isl=kwargs["isl"],
        osl=kwargs["osl"],
        batch_size=kwargs["batch_size"],
        ctx_tokens=kwargs["isl"],
        tp_size=kwargs["tp_size"],
        pp_size=1,
        model_path=kwargs["model_path"],
        system_name=kwargs["system_name"],
        backend_name=kwargs["backend_name"],
        backend_version=kwargs["backend_version"],
        mode=kwargs["mode"],
        raw={"ttft": 10.0, "tpot": 2.0, "memory": 12.0, "generation_latency": 2.0},
        per_ops_data={"generation": {"gemm": 2.0}},
        per_ops_source={"generation": {"gemm": "silicon"}},
    )


@pytest.mark.parametrize("mode", ["agg", "disagg", "afd", "static", "static_ctx", "static_gen"])
def test_estimates_match_compatibility_inputs_and_reports(mode, monkeypatch, capsys):
    calls = []

    def estimate(**kwargs):
        calls.append(kwargs)
        return _result(kwargs)

    monkeypatch.setattr(estimator, "cli_estimate", estimate)
    monkeypatch.setattr(compatibility.perf_database, "set_systems_paths", lambda paths: None)
    monkeypatch.setattr(
        cli,
        "resolve_runner_factory",
        lambda *_: pytest.fail("estimate must not resolve a stack"),
    )
    arguments = [
        *_BASE,
        "--estimate-mode",
        mode,
        "--detail",
        "all",
        "--systems-paths",
        "default",
    ]
    if mode == "disagg":
        arguments += [
            "--prefill-batch-size",
            "1",
            "--prefill-num-workers",
            "2",
            "--decode-batch-size",
            "4",
            "--decode-num-workers",
            "3",
            "--decode-system",
            "gb200",
        ]
    if mode == "afd":
        arguments += ["--n-a-nodes", "1", "--n-f-nodes", "2", "--a-batch-size", "4"]

    assert cli.main(arguments) == 0
    actual = capsys.readouterr().out
    actual_calls = list(calls)
    calls.clear()
    parser = argparse.ArgumentParser()
    compatibility.configure_parser(parser)
    compatibility.main(parser.parse_args(arguments))
    expected = capsys.readouterr().out

    assert actual == expected
    assert actual_calls == calls
    assert actual_calls[0]["mode"] == mode
    assert actual_calls[0]["batch_size"] == 4
    assert actual_calls[0]["tp_size"] == 2
    assert "Performance Estimate" in actual
    assert "Detailed Breakdown (all)" in actual
    assert "deprecated" not in actual.lower()


def test_estimate_defaults_and_runtime_options(monkeypatch):
    received = []
    paths = []
    monkeypatch.setattr(compatibility, "run_estimate", received.append)
    monkeypatch.setattr(compatibility.perf_database, "set_systems_paths", paths.append)
    assert (
        cli.main(
            [
                *_BASE,
                "--systems-paths",
                "default,/tmp/profiles",
                "--forward-model",
                "fpm",
                "--engine-step-backend",
                "rust",
            ]
        )
        == 0
    )
    args = received[0]
    assert args.estimate_mode == "agg"
    assert args.detail is None
    assert args.forward_model == "fpm"
    assert args.engine_step_backend == "rust"
    assert paths == ["default,/tmp/profiles"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--estimate-mode", "unknown"],
        ["--config", "prediction.yaml"],
        ["--stack", "engine"],
        ["--format", "json"],
        ["--output-dir", "results"],
        ["--save-dir", "results"],
        ["--top-n", "3"],
        ["--deployment-target", "dynamo-j2"],
    ],
)
def test_estimate_rejects_unsupported_options(arguments, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([*_BASE, *arguments])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_serving_commands_reject_estimate_mode(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([command, "--config", "unused.yaml", "--estimate-mode", "static_gen"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --estimate-mode" in capsys.readouterr().err


@pytest.mark.parametrize("detail", ["unknown", "time,unknown"])
def test_invalid_detail_fails_before_estimator_execution(detail, monkeypatch):
    monkeypatch.setattr(
        estimator,
        "cli_estimate",
        lambda **_: pytest.fail("invalid detail reached estimator"),
    )
    with pytest.raises(SystemExit, match="Unknown"):
        cli.main([*_BASE, "--detail", detail])


@pytest.mark.parametrize(
    "error",
    [ValueError("invalid configuration"), PerfDataNotAvailableError("missing profile")],
)
def test_expected_estimator_errors_remain_concise(error, monkeypatch, capsys):
    def fail(**kwargs):
        raise error

    monkeypatch.setattr(estimator, "cli_estimate", fail)
    with pytest.raises(SystemExit, match=str(error)):
        cli.main(_BASE)
    assert "Performance Estimate" not in capsys.readouterr().out


@pytest.mark.parametrize("error,code", [(RuntimeError("failed estimator"), 1), (KeyboardInterrupt(), 130)])
def test_execution_failure_and_interrupt_exit_codes(error, code, monkeypatch):
    def fail(args):
        raise error

    monkeypatch.setattr(compatibility, "run_estimate", fail)
    assert cli.main(_BASE) == code


def test_estimate_preserves_epd_dispatch_and_validation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        compatibility,
        "_run_estimate_epd",
        lambda args, mode: calls.append((args, mode)),
    )
    assert (
        cli.main(
            [
                *_BASE,
                "--enable-epd",
                "--encoder-tp",
                "2",
                "--encoder-batch-size",
                "1",
                "--encoder-num-workers",
                "3",
            ]
        )
        == 0
    )
    args, mode = calls[0]
    assert (
        mode,
        args.encoder_tp,
        args.encoder_batch_size,
        args.encoder_num_workers,
    ) == ("agg", 2, 1, 3)
    with pytest.raises(SystemExit, match="require --enable-epd"):
        cli.main([*_BASE, "--encoder-tp", "2"])


@pytest.mark.parametrize("arguments", [["--help"], ["predict", "--help"], ["recommend", "--help"]])
def test_serving_help_keeps_estimator_imports_lazy(arguments):
    script = """
import sys
from aisimulate.main import main
try:
    main(sys.argv[1:])
except SystemExit as exc:
    assert exc.code == 0
assert 'aiconfigurator.cli.main' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, *arguments],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    if arguments == ["--help"]:
        assert "estimate" in result.stdout
