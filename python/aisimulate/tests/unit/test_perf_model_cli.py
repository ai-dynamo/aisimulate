# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``aisimulate perf-model`` command: its help, its three actions and its exit codes.

The tests call ``aisimulate.supervision.main``, the function the ``aisimulate``
console script runs, and read what the command prints and what it leaves on
disk. ``aisimulate.main.main`` has a second copy of the ``perf-model`` branch;
a few tests run it once for each exit code. ``check`` builds real performance
models from the data shipped with the package. ``fetch`` runs against a
replaced profile store, a manifest written for the test or a profile whose
files are served from memory, so nothing is downloaded.
"""

from __future__ import annotations

import datetime
import errno
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

import aisimulate.main as cli
from aisimulate import supervision
from aisimulate.config import CorePredictionConfig
from aisimulate.config import ais_perf_model_migrate as migrate_module
from aisimulate_core.ais_perf_model import AisPerfModelDeprecationWarning, load
from aisimulate_core.sdk import fpm_dataset

from .sdk.test_dataset_contracts import dataset  # noqa: F401
from .test_fpm_config_imports import _IMPORT_GUARD

pytestmark = pytest.mark.unit

MODEL, SYSTEM = "Qwen/Qwen3-0.6B", "h200_sxm"
TRAFFIC = {
    "source": {"type": "synthetic", "input_tokens": 128, "output_tokens": 16},
    "load": {"type": "concurrency", "concurrency": 4},
    "stop": {"requests": 8},
}
# A config whose performance model builds from the data shipped with the package.
BUILDS = {"model": MODEL, "system": SYSTEM, "backend": "vllm", "kv_block_size": 16, "estimation_mode": "op_level"}
# The package has no data for this system, and op_level may not fall back, so this config does not build.
DOES_NOT_BUILD = {"model": MODEL, "system": "no_such_system", "backend": "vllm", "estimation_mode": "op_level"}
# A regression needs no data to build; it has seen no observations yet, so it cannot estimate.
UNTRAINED = {"model": "test/model", "system": "test", "backend": "vllm", "estimation_mode": "fpm_regression"}
# The performance model rejects the mode; the command itself does not validate the values of an entry.
UNKNOWN_MODE = {**UNTRAINED, "estimation_mode": "no_such_mode"}
# Old estimator fields at the engine level; every value differs from the default of its field.
OLD_FIELDS = {"estimation_mode": "op_level", "fallback_policy": "allow", "database_mode": "HYBRID"}
# The engine keys of ``_predict`` that are not estimator fields, in the order the file states them.
ENGINE_KEYS = ["model", "hardware", "backend", "mode", "context_length", "workers"]
# The two sections only a recommend config has.
OPTIMIZATION = {"target": "throughput"}
OPTIMIZER = {"algorithm": "random", "max_trials": 1}
WROTE = "Wrote {} (comments and formatting of the original file are not kept)."
PROFILES = [
    {"manifest": "example-manifest", "profile": "h200-tp1", "admission": "serving", "repo_id": "example/fpm"},
    {"manifest": "example-manifest", "profile": "h200-tp2", "admission": "rejected", "repo_id": "example/fpm"},
]


def _exit_code(entry_point: Any, argv: Sequence[object]) -> int:
    try:
        return entry_point(["perf-model", *(str(item) for item in argv)])
    except SystemExit as stop:
        # argparse reports a usage error, and prints help, by exiting.
        return stop.code


def _run(*argv: object) -> int:
    """Run ``aisimulate perf-model`` as the console script does and return its exit code."""
    return _exit_code(supervision.main, argv)


def _predict(**engine_fields: Any) -> dict[str, Any]:
    """A predict config for one aggregated worker, as its file states it."""
    engine = {
        "model": MODEL,
        "hardware": SYSTEM,
        "backend": "vllm",
        "mode": "aggregated",
        "context_length": 256,
        "workers": {"aggregated": {}},
        **engine_fields,
    }
    return {"engine": engine, "traffic": TRAFFIC}


def _unsorted_predict() -> dict[str, Any]:
    """A predict config with old fields whose keys are not in alphabetical order and whose model name is not ASCII."""
    config = _predict(model="模型/测试", **OLD_FIELDS)
    return {"traffic": config["traffic"], "engine": config["engine"]}


def _recommend(**sections: Any) -> dict[str, Any]:
    """A recommend config with old fields; ``sections`` are the sections only this kind of config has."""
    config = _predict(**OLD_FIELDS)
    del config["engine"]["mode"]
    config["engine"]["workers"] = {"aggregated": {}, "prefill": {}, "decode": {}}
    return config | sections


def _search(**space_fields: Any) -> dict[str, Any]:
    """A search config for one aggregated deployment, as its file states it."""
    space = {"model_name": MODEL, "hardware_sku": SYSTEM, "backend": ["vllm"], "deployment_mode": ["agg"]}
    workload = {"isl": 128, "osl": 16, "request_rate": 2, "num_request_ratio": 3}
    return {"search_space": {**space, **space_fields}, "workload": workload}


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _one_line(text: str) -> str:
    """Help text without the line breaks argparse adds to fit the terminal."""
    return " ".join(text.split())


# --- Help and command line --------------------------------------------------------------------


def test_help_names_the_three_actions(capsys):
    assert _run("--help") == 0

    text = _one_line(capsys.readouterr().out)
    assert "usage: aisimulate perf-model [-h] {check,migrate,fetch}" in text
    assert "check Build the performance model of every entry and report whether it builds." in text
    assert "migrate Convert the old estimator fields of a config file to ais_perf_model." in text
    assert "fetch Download a published FPM profile and print the folder to name in systems_paths." in text


@pytest.mark.parametrize(
    ("action", "options"),
    [
        ("check", ["VALUE", "--worker-type TYPE", "--fill KEY=VALUE", "--workspace DIR", "--format {table,json}"]),
        (
            "migrate",
            ["CONFIG", "--kind {predict,recommend,search}", "--output FILE", "--overwrite", "--format {table,json}"],
        ),
        (
            "fetch",
            ["[MANIFEST] [PROFILE]", "--list", "--cache-dir DIR", "--local-files-only", "--allow-unqualified"],
        ),
    ],
)
def test_help_of_an_action_names_its_arguments(action, options, capsys):
    assert _run(action, "--help") == 0

    text = _one_line(capsys.readouterr().out)
    assert f"usage: aisimulate perf-model {action}" in text
    for option in options:
        assert option in text


def test_migrate_help_says_what_to_know_before_converting_a_file(capsys):
    assert _run("migrate", "--help") == 0

    text = _one_line(capsys.readouterr().out)
    assert "Without --output the plan is printed and nothing is written." in text
    assert "run this command from the directory the config is run from" in text
    assert "A relative path is rewritten to start at the folder of FILE, or of CONFIG without --output." in text
    assert "is a symbolic link to a file in another folder, or CONFIG is such a link to FILE, a relative path" in text
    assert "is written as an absolute path" in text
    assert "as JSON when FILE ends with .json, as YAML otherwise" in text
    assert "another file that exists is replaced only with --overwrite" in text
    assert "Comments and formatting of the original file are not kept." in text
    assert "Do not convert a replay config that 'aisimulate onboard' wrote" in text


def test_check_help_says_what_ok_not_ready_means(capsys):
    assert _run("check", "--help") == 0

    text = _one_line(capsys.readouterr().out)
    assert "A model that is built but cannot estimate yet is reported as 'ok, not ready'" in text
    assert "The exit code is 1 when an entry does not build." in text


def test_an_action_is_required(capsys):
    assert _run() == 2

    assert "usage: aisimulate perf-model [-h] {check,migrate,fetch}" in capsys.readouterr().err


def test_parsing_the_command_imports_no_runtime_module():
    probe = (
        _IMPORT_GUARD
        + """
from aisimulate.cli_args import build_parser
import aisimulate.perf_model_cli

parser = build_parser()
args = parser.parse_args(['perf-model', 'migrate', 'config.yaml', '--output', 'new.yaml'])
assert (args.command, args.perf_model_action, args.config, args.output) == (
    'perf-model', 'migrate', 'config.yaml', 'new.yaml')
for action in ('check', 'migrate', 'fetch'):
    try:
        parser.parse_args(['perf-model', action, '--help'])
    except SystemExit as exc:
        assert exc.code == 0
assert_lightweight()
"""
    )

    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False, timeout=15)

    assert result.returncode == 0, result.stdout + result.stderr


def test_the_supervised_entry_point_runs_the_command_itself(tmp_path, monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("perf-model must not start a supervised child process")

    monkeypatch.setattr(supervision, "run_process", unexpected)
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))

    assert supervision.main(["perf-model", "migrate", str(config)]) == 0
    assert "new key engine.ais_perf_model:" in capsys.readouterr().out

    with pytest.raises(SystemExit) as stop:
        supervision.main(["perf-model", "migrate", str(tmp_path / "missing.yaml")])
    assert stop.value.code == 2
    assert "could not read configuration" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        (["check", json.dumps(UNTRAINED), "--worker-type", "decode"], 0),
        (["check", json.dumps(UNKNOWN_MODE), "--worker-type", "decode"], 1),
        (["check", json.dumps(BUILDS)], 2),
    ],
    ids=["nothing wrong", "a problem with the input", "a usage error"],
)
def test_the_entry_point_of_the_main_module_gives_the_same_result(arguments, code, capsys):
    expected = (_run(*arguments), *capsys.readouterr())

    assert (_exit_code(cli.main, arguments), *capsys.readouterr()) == expected
    assert expected[0] == code


def test_an_interrupt_exits_130_from_both_entry_points(monkeypatch, capsys):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(fpm_dataset, "list_fpm_profiles", interrupted)

    assert _run("fetch", "--list") == 130
    assert _exit_code(cli.main, ["fetch", "--list"]) == 130
    assert tuple(capsys.readouterr()) == ("", "")


# --- check ------------------------------------------------------------------------------------


def test_check_builds_every_entry_and_exits_1_when_one_does_not_build(tmp_path, capsys):
    value = _write(
        tmp_path / "model.yaml",
        [{"worker_type": "aggregated", **BUILDS}, {"worker_type": "decode", **DOES_NOT_BUILD}],
    )

    assert _run("check", value, "--format", "json") == 1

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    built, failed = report["entries"]
    assert (built["worker_type"], built["estimation_mode"], built["ok"], built["error"]) == (
        "aggregated",
        "op_level",
        True,
        None,
    )
    assert built["config"] == {"worker_type": "aggregated", **BUILDS}
    assert built["readiness"]["ready"] is True
    assert (failed["worker_type"], failed["estimation_mode"], failed["ok"]) == ("decode", "op_level", False)
    assert failed["error"].startswith("ValueError: ")
    assert "no_such_system" in failed["error"]
    assert failed["readiness"] is None


def test_check_prints_one_row_per_entry_with_ok_or_the_error(tmp_path, capsys):
    value = _write(
        tmp_path / "model.yaml",
        [
            {"worker_type": "aggregated", **BUILDS},
            {"worker_type": "prefill", **UNTRAINED},
            {"worker_type": "decode", **UNKNOWN_MODE},
        ],
    )

    assert _run("check", value) == 1

    header, built, untrained, failed = capsys.readouterr().out.splitlines()
    assert header.split() == ["worker", "type", "estimation_mode", "result"]
    assert built.split() == ["aggregated", "op_level", "ok"]
    assert untrained.split()[:5] == ["prefill", "fpm_regression", "ok,", "not", "ready:"]
    assert "(insufficient_data)" in untrained
    # The error is the one the performance model raises for the value.
    assert failed.split()[:4] == ["decode", "no_such_mode", "failed:", "ValueError:"]
    assert "estimation_mode: unknown variant `no_such_mode`" in failed


def test_check_builds_only_the_worker_types_that_are_asked_for(tmp_path, capsys):
    value = _write(
        tmp_path / "model.yaml",
        [{"worker_type": "aggregated", **BUILDS}, {"worker_type": "decode", **DOES_NOT_BUILD}],
    )

    assert _run("check", value, "--worker-type", "aggregated", "--format", "json") == 0

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert [entry["worker_type"] for entry in report["entries"]] == ["aggregated"]


def test_check_of_a_model_that_builds_but_cannot_estimate_yet_exits_0_and_says_so(capsys):
    assert _run("check", json.dumps(UNTRAINED), "--worker-type", "decode") == 0
    row = capsys.readouterr().out.splitlines()[1]
    assert row.startswith("decode       fpm_regression   ok, not ready: ")
    assert "pure_decode (insufficient_data)" in row

    assert _run("check", json.dumps(UNTRAINED), "--worker-type", "decode", "--format", "json") == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["entries"][0]["readiness"]["ready"] is False


def test_check_of_one_config_without_worker_type_needs_the_worker_types(capsys):
    assert _run("check", json.dumps(BUILDS)) == 2

    assert "name the worker types to check with --worker-type" in capsys.readouterr().err


@pytest.mark.parametrize("options", [[], ["--worker-type", "aggregated"]], ids=["no worker type", "a worker type"])
def test_check_of_a_whole_config_says_which_value_it_takes(options, tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", _predict(ais_perf_model=BUILDS))

    assert _run("check", config, *options) == 2

    error = capsys.readouterr().err
    assert "VALUE is a whole config" in error
    assert "the value of engine.ais_perf_model" in error
    # The file is not one entry that leaves worker_type out, so the command must not ask for the option.
    assert "--worker-type" not in error


@pytest.mark.parametrize("other_sections", [{}, {"engine": {}}], ids=["search_space", "search_space and engine"])
def test_check_of_a_whole_search_config_names_the_key_of_the_search_space(other_sections, tmp_path, capsys):
    config = _write(tmp_path / "search.yaml", _search(ais_perf_model=BUILDS) | other_sections)

    assert _run("check", config) == 2

    error = capsys.readouterr().err
    assert "VALUE is a whole config" in error
    # A search_space section makes a search config, whatever else the file has.
    assert "the value of search_space.ais_perf_model" in error
    assert "engine.ais_perf_model" not in error


def test_check_builds_a_worker_type_once_when_it_is_named_twice(capsys):
    assert _run("check", json.dumps(UNTRAINED), "--worker-type", "decode", "--worker-type", "decode") == 0

    header, *rows = capsys.readouterr().out.splitlines()
    assert header.split() == ["worker", "type", "estimation_mode", "result"]
    assert [row.split()[0] for row in rows] == ["decode"]


def test_check_reports_the_mode_that_a_template_states(tmp_path, capsys):
    _write(tmp_path / "template.yaml", BUILDS)
    # The entry itself writes no estimation_mode; the mode comes from the config it starts from.
    value = _write(tmp_path / "model.yaml", {"from": "template.yaml"})

    assert _run("check", value, "--worker-type", "aggregated") == 0
    assert capsys.readouterr().out.splitlines()[1].split() == ["aggregated", "op_level", "ok"]

    assert _run("check", value, "--worker-type", "aggregated", "--format", "json") == 0
    [checked] = json.loads(capsys.readouterr().out)["entries"]
    assert checked["estimation_mode"] == "op_level"


def test_check_fill_adds_a_value_the_entry_leaves_out(capsys):
    entry = {key: value for key, value in BUILDS.items() if key not in ("model", "estimation_mode")}

    assert _run("check", json.dumps(entry), "--worker-type", "aggregated", "--fill", f"model={MODEL}") == 0
    row = capsys.readouterr().out.splitlines()[1]
    # The entry writes no estimation_mode, and the command does not guess which estimator was selected.
    assert row.split() == ["aggregated", "(not", "set)", "ok"]

    arguments = ["--worker-type", "aggregated", "--fill", f"model={MODEL}", "--fill", "tp=2", "--format", "json"]
    assert _run("check", json.dumps(entry), *arguments) == 0
    [checked] = json.loads(capsys.readouterr().out)["entries"]
    assert checked["estimation_mode"] is None
    # The value of --fill is read as YAML, so tp is a number.
    assert checked["config"] == {"worker_type": "aggregated", **entry, "model": MODEL, "tp": 2}


def test_check_fill_that_differs_from_a_written_value_fails_the_entry(capsys):
    arguments = ["--worker-type", "decode", "--fill", "model=other/model"]

    assert _run("check", json.dumps(BUILDS), *arguments) == 1

    row = capsys.readouterr().out.splitlines()[1]
    assert "failed: AisPerfModelError: " in row
    assert f'model is "{MODEL}" but the caller requires "other/model"' in row


@pytest.mark.parametrize(
    ("fill", "message"),
    [
        (["--fill", "model"], "--fill takes KEY=VALUE, and 'model' has no key or no '='"),
        (["--fill", "=name"], "--fill takes KEY=VALUE, and '=name' has no key or no '='"),
        (["--fill", "tp=1", "--fill", "tp=2"], "--fill sets tp more than once"),
        (["--fill", "tp=[1"], "--fill tp: the value '[1' is not valid YAML"),
    ],
)
def test_check_rejects_a_fill_option_it_cannot_read(fill, message, capsys):
    assert _run("check", json.dumps(BUILDS), "--worker-type", "decode", *fill) == 2

    assert message in capsys.readouterr().err


def test_check_of_a_worker_type_the_value_has_no_entry_for_is_a_usage_error(tmp_path, capsys):
    value = _write(tmp_path / "model.yaml", [{"worker_type": "decode", **BUILDS}])

    assert _run("check", value, "--worker-type", "prefill") == 2

    assert "no entry for worker type 'prefill'; it has: decode" in capsys.readouterr().err


def test_check_of_a_value_that_cannot_be_read_is_a_usage_error(tmp_path, capsys):
    missing = tmp_path / "missing.yaml"

    assert _run("check", missing) == 2

    assert f"cannot read {missing}" in capsys.readouterr().err


def test_check_prepares_a_dataset_entry_in_the_workspace(dataset, tmp_path, capsys):  # noqa: F811
    root, manifest, _ = dataset
    value = _write(tmp_path / "model.yaml", {"from": {"dataset": {"root": str(root), "manifest": manifest}}})
    fill = {"model": "meta-llama/Meta-Llama-3.1-8B", "system": SYSTEM, "backend": "vllm", "tp": 1}
    arguments = ["--worker-type", "prefill", "--format", "json"]
    for key, text in fill.items():
        arguments += ["--fill", f"{key}={text}"]
    workspace = tmp_path / "workspace"

    assert _run("check", value, *arguments, "--workspace", workspace) == 0

    [checked] = json.loads(capsys.readouterr().out)["entries"]
    [prepared] = workspace.iterdir()
    assert checked["ok"] is True
    assert checked["config"]["systems_paths"] == [str(prepared / "systems")]

    # Without --workspace the dataset is prepared in a temporary folder that is removed afterwards.
    assert _run("check", value, *arguments) == 0
    [checked] = json.loads(capsys.readouterr().out)["entries"]
    assert checked["ok"] is True
    assert not Path(checked["config"]["systems_paths"][0]).exists()


# --- migrate ----------------------------------------------------------------------------------


def test_migrate_prints_the_plan_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    text = config.read_text(encoding="utf-8")
    assert _run("migrate", "config.yaml", "--format", "json") == 0
    report = json.loads(capsys.readouterr().out)

    assert _run("migrate", "config.yaml") == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        "config: config.yaml",
        "kind: predict",
        f"working directory: {tmp_path} (relative paths in the old fields start here)",
    ]
    block = lines[lines.index("new key engine.ais_perf_model:") + 1 : lines.index("removed fields:")]
    assert yaml.safe_load("\n".join(block)) == {"engine": {"ais_perf_model": OLD_FIELDS}}
    # The plan lists the removed fields of the report, one on a line.
    removed = lines[lines.index("removed fields:") + 1 :]
    assert report["removed"]
    assert removed[: len(report["removed"])] == [f"  {name}" for name in report["removed"]]
    assert lines[-1] == "Nothing was written. Pass --output FILE to write the converted config."
    assert config.read_text(encoding="utf-8") == text
    assert list(tmp_path.iterdir()) == [config]


def test_migrate_output_loads_without_the_deprecation_warning_and_means_the_same(tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    text = config.read_text(encoding="utf-8")
    output = tmp_path / "converted.yaml"
    with pytest.warns(AisPerfModelDeprecationWarning):
        original = CorePredictionConfig.from_yaml(config)

    assert _run("migrate", config, "--output", output) == 0

    assert capsys.readouterr().out.splitlines()[-1] == WROTE.format(output)
    written = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert written["engine"]["ais_perf_model"] == OLD_FIELDS
    assert not set(OLD_FIELDS) & set(written["engine"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converted = CorePredictionConfig.from_yaml(output)
    assert converted.model_dump(mode="json") == original.model_dump(mode="json")
    assert config.read_text(encoding="utf-8") == text


def test_migrate_output_may_be_the_config_itself(tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))

    assert _run("migrate", config, "--output", config) == 0
    capsys.readouterr()

    assert yaml.safe_load(config.read_text(encoding="utf-8"))["engine"]["ais_perf_model"] == OLD_FIELDS
    # The converted file has no old fields left, so a second run changes nothing.
    text = config.read_text(encoding="utf-8")
    assert _run("migrate", config, "--output", config) == 0
    last = capsys.readouterr().out.splitlines()[-1]
    assert last == "The config has no old estimator fields; there is nothing to convert."
    assert config.read_text(encoding="utf-8") == text


def test_migrate_keeps_the_order_of_the_keys_and_text_that_is_not_ascii(tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", _unsorted_predict())
    output = tmp_path / "converted.yaml"

    assert _run("migrate", config, "--output", output) == 0
    capsys.readouterr()

    text = output.read_text(encoding="utf-8")
    written = yaml.safe_load(text)
    assert list(written) == ["traffic", "engine"]
    assert list(written["traffic"]) == list(TRAFFIC)
    assert [key for key in written["engine"] if key != "ais_perf_model"] == ENGINE_KEYS
    assert "模型/测试" in text


def test_migrate_writes_json_to_a_file_whose_name_ends_with_json(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_unsorted_predict(), indent=2, ensure_ascii=False), encoding="utf-8")
    with pytest.warns(AisPerfModelDeprecationWarning):
        original = CorePredictionConfig.from_yaml(config)
    assert _run("migrate", config, "--format", "json") == 0
    plan = json.loads(capsys.readouterr().out)

    assert _run("migrate", config, "--output", config, "--format", "json") == 0

    # The report is the one of the plan; only the file is in another format.
    assert json.loads(capsys.readouterr().out) == plan | {"output": str(config)}
    text = config.read_text(encoding="utf-8")
    written = json.loads(text)
    # Two spaces to a level, text that is not ASCII as it is, and a newline at the end.
    assert text == json.dumps(written, indent=2, ensure_ascii=False) + "\n"
    assert "模型/测试" in text
    assert list(written) == ["traffic", "engine"]
    assert [key for key in written["engine"] if key != "ais_perf_model"] == ENGINE_KEYS
    assert written["engine"]["ais_perf_model"] == OLD_FIELDS
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converted = CorePredictionConfig.from_yaml(config)
    assert converted.model_dump(mode="json") == original.model_dump(mode="json")


def test_migrate_writes_json_to_a_file_whose_name_ends_with_json_in_capital_letters(tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    output = tmp_path / "CONVERTED.JSON"

    assert _run("migrate", config, "--output", output) == 0
    capsys.readouterr()

    text = output.read_text(encoding="utf-8")
    written = json.loads(text)
    assert text == json.dumps(written, indent=2, ensure_ascii=False) + "\n"
    assert written["engine"]["ais_perf_model"] == OLD_FIELDS


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (datetime.date(2024, 1, 1), "date"),
        ({1: "first"}, "extra.values[1] is {1: 'first'} and would be read back from the file as {'1': 'first'}"),
        (float("inf"), "inf"),
        (1e-05, "extra.values[1] is 1e-05 and would be read back from the file as '1e-05'"),
    ],
    ids=["a date", "a key that is not text", "a number that is not finite", "a number that YAML reads as text"],
)
def test_migrate_refuses_to_write_json_that_does_not_read_back_as_the_config(
    value, reason, tmp_path, monkeypatch, capsys
):
    converted = _predict(ais_perf_model=OLD_FIELDS) | {"extra": {"values": [0.5, value]}}
    # YAML holds values that JSON does not, and a config file is read as YAML. The conversion is replaced
    # so that the config to write has such a value whatever the conversion accepts.
    monkeypatch.setattr(migrate_module, "migrate_mapping", lambda *args, **kwargs: (converted, ["removed"], []))
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    output = tmp_path / "converted.json"

    assert _run("migrate", config, "--output", output) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"cannot write {output} as JSON: " in captured.err
    assert reason in captured.err
    assert "name a file that does not end with .json to write YAML" in captured.err
    assert list(tmp_path.iterdir()) == [config]
    # A YAML file holds the same config.
    assert _run("migrate", config, "--output", tmp_path / "converted.yaml") == 0
    assert yaml.safe_load((tmp_path / "converted.yaml").read_text(encoding="utf-8")) == converted


def test_migrate_refuses_to_replace_another_file_unless_overwrite_is_given(tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    other = tmp_path / "other.yaml"
    other.write_text("kept: true\n", encoding="utf-8")

    assert _run("migrate", config, "--output", other) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"output file {other} exists; pass --overwrite to replace it" in captured.err
    assert other.read_text(encoding="utf-8") == "kept: true\n"

    assert _run("migrate", config, "--output", other, "--overwrite") == 0

    assert capsys.readouterr().out.splitlines()[-1] == WROTE.format(other)
    assert yaml.safe_load(other.read_text(encoding="utf-8"))["engine"]["ais_perf_model"] == OLD_FIELDS


@pytest.mark.parametrize(
    "spelling", ["configs/../configs/config.yaml", "link.yaml"], ids=["another spelling of the path", "a symlink"]
)
def test_migrate_output_that_names_the_config_another_way_needs_no_overwrite(spelling, tmp_path, capsys):
    config = _write(tmp_path / "configs" / "config.yaml", _predict(**OLD_FIELDS))
    link = tmp_path / "link.yaml"
    link.symlink_to(config)

    assert _run("migrate", config, "--output", tmp_path / spelling) == 0
    capsys.readouterr()

    assert link.is_symlink()
    assert not config.is_symlink()
    assert yaml.safe_load(config.read_text(encoding="utf-8"))["engine"]["ais_perf_model"] == OLD_FIELDS


def test_migrate_of_a_config_given_through_a_symlink_changes_the_file_and_keeps_the_symlink(tmp_path, capsys):
    real = _write(tmp_path / "real" / "config.yaml", _predict(**OLD_FIELDS))
    link = tmp_path / "config.yaml"
    link.symlink_to(real)

    assert _run("migrate", link, "--output", link) == 0
    capsys.readouterr()

    assert link.is_symlink()
    assert not real.is_symlink()
    assert yaml.safe_load(real.read_text(encoding="utf-8"))["engine"]["ais_perf_model"] == OLD_FIELDS
    assert sorted(tmp_path.rglob("*")) == [link, real.parent, real]


@pytest.fixture
def umask_027():
    """New files of the test are not readable by others, whatever the umask of the test run is."""
    previous = os.umask(0o027)
    yield
    os.umask(previous)


def test_migrate_keeps_the_permissions_of_the_file_it_replaces(umask_027, tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    config.chmod(0o664)

    assert _run("migrate", config, "--output", config) == 0
    capsys.readouterr()

    assert stat.S_IMODE(config.stat().st_mode) == 0o664


def test_migrate_gives_a_new_file_the_permissions_of_any_new_file(umask_027, tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    output = tmp_path / "converted.yaml"

    assert _run("migrate", config, "--output", output) == 0
    capsys.readouterr()

    assert stat.S_IMODE(output.stat().st_mode) == 0o640


def test_migrate_writes_an_output_file_whose_name_is_240_bytes_long(tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    output = tmp_path / f"{'c' * 235}.yaml"
    try:
        output.touch()
    except OSError:
        pytest.skip("the file system does not take a file name of 240 bytes")
    output.unlink()

    assert _run("migrate", config, "--output", output) == 0

    assert capsys.readouterr().out.splitlines()[-1] == WROTE.format(output)
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["engine"]["ais_perf_model"] == OLD_FIELDS
    assert sorted(tmp_path.iterdir()) == sorted([config, output])


def test_migrate_makes_the_new_file_in_the_folder_of_the_file_it_replaces(tmp_path, monkeypatch, capsys):
    config = _write(tmp_path / "configs" / "config.yaml", _predict(**OLD_FIELDS))
    elsewhere = tmp_path / "temporary-files"
    elsewhere.mkdir()
    moved_from = []
    replace = os.replace

    def watched(source, target):
        moved_from.append(os.path.dirname(source))
        replace(source, target)

    with monkeypatch.context() as patch:
        # A file made where temporary files go can be on another file system, and a move from there is not one step.
        patch.setattr(tempfile, "tempdir", str(elsewhere))
        patch.setattr(os, "replace", watched)
        assert _run("migrate", config, "--output", config) == 0
    capsys.readouterr()

    assert moved_from == [str(config.parent)]
    assert list(elsewhere.iterdir()) == []


def test_migrate_twice_into_one_folder_makes_the_new_files_under_two_names(tmp_path, monkeypatch, capsys):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    moved_from = []
    replace = os.replace

    def watched(source, target):
        moved_from.append(source)
        replace(source, target)

    with monkeypatch.context() as patch:
        # Two commands that run at the same time in one folder must not make their new file under the same name.
        patch.setattr(os, "replace", watched)
        assert _run("migrate", config, "--output", tmp_path / "first.yaml") == 0
        assert _run("migrate", config, "--output", tmp_path / "second.yaml") == 0
    capsys.readouterr()

    first, second = moved_from
    assert first != second


def test_migrate_stores_the_whole_new_text_before_it_replaces_the_file(tmp_path, monkeypatch, capsys):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    sizes_when_stored = []
    fsync = os.fsync

    def watched(descriptor):
        sizes_when_stored.append(os.fstat(descriptor).st_size)
        fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", watched)
        assert _run("migrate", config, "--output", config) == 0
    capsys.readouterr()

    assert yaml.safe_load(config.read_text(encoding="utf-8"))["engine"]["ais_perf_model"] == OLD_FIELDS
    assert sizes_when_stored == [config.stat().st_size]


def test_migrate_that_cannot_build_the_text_leaves_the_file_as_it_was(tmp_path, monkeypatch):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    before = config.read_bytes()

    def fails(*args, **kwargs):
        raise yaml.YAMLError("the value cannot be written")

    monkeypatch.setattr(yaml, "safe_dump", fails)

    with pytest.raises(yaml.YAMLError, match="the value cannot be written"):
        _run("migrate", config, "--output", config)

    assert config.read_bytes() == before
    assert list(tmp_path.iterdir()) == [config]


def test_migrate_that_runs_into_the_file_size_limit_leaves_the_config_as_it_was(tmp_path):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    before = config.read_bytes()
    # The limit is smaller than the converted file, so the write stops partway, as it does on a full disk.
    script = """
import resource, sys
sys.dont_write_bytecode = True
from aisimulate import supervision

resource.setrlimit(resource.RLIMIT_FSIZE, (64, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))
sys.exit(supervision.main(['perf-model', 'migrate', sys.argv[1], '--output', sys.argv[1]]))
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(config)], capture_output=True, text=True, check=False, timeout=60
    )

    assert config.read_bytes() == before
    assert list(tmp_path.iterdir()) == [config]
    assert result.returncode == 2, result.stdout + result.stderr
    assert result.stdout == ""
    assert f"could not write {config}: " in result.stderr
    assert os.strerror(errno.EFBIG) in result.stderr


@pytest.mark.parametrize(
    ("step", "failure", "code"),
    [
        ("fsync", OSError(errno.EIO, os.strerror(errno.EIO)), 2),
        ("replace", OSError(errno.EBUSY, os.strerror(errno.EBUSY)), 2),
        ("fsync", KeyboardInterrupt(), 130),
    ],
    ids=["the new text cannot be stored", "the new file cannot be put in place", "the write is interrupted"],
)
def test_migrate_write_that_stops_leaves_the_file_as_it_was_and_no_other_file(
    step, failure, code, tmp_path, monkeypatch, capsys
):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    before = config.read_bytes()

    def stops(*args, **kwargs):
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(os, step, stops)
        assert _run("migrate", config, "--output", config) == code

    captured = capsys.readouterr()
    assert captured.out == ""
    if isinstance(failure, KeyboardInterrupt):
        assert captured.err == ""
    else:
        assert f"could not write {config}: {failure}" in captured.err
    assert config.read_bytes() == before
    assert list(tmp_path.iterdir()) == [config]


@pytest.mark.parametrize(
    ("name", "reason"),
    [("missing/converted.yaml", errno.ENOENT), ("folder", errno.EISDIR)],
    ids=["a folder that does not exist", "a folder"],
)
def test_migrate_output_that_cannot_be_a_file_is_a_usage_error(name, reason, tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    folder = tmp_path / "folder"
    folder.mkdir()
    output = tmp_path / name

    assert _run("migrate", config, "--output", output) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"{os.strerror(reason)}: '{output}'" in captured.err
    assert sorted(tmp_path.rglob("*")) == [config, folder]


def test_migrate_output_that_is_a_read_only_file_is_a_usage_error_and_the_file_stays(tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    config.chmod(0o444)
    if os.access(config, os.W_OK):
        pytest.skip("this user may write a read-only file")
    before = config.read_bytes()

    assert _run("migrate", config, "--output", config) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"{os.strerror(errno.EACCES)}: '{config}'" in captured.err
    assert config.read_bytes() == before
    assert list(tmp_path.iterdir()) == [config]


def test_migrate_output_without_a_folder_is_written_in_the_working_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))

    assert _run("migrate", "config.yaml", "--output", "converted.yaml") == 0

    assert capsys.readouterr().out.splitlines()[-1] == WROTE.format("converted.yaml")
    assert sorted(file.name for file in tmp_path.iterdir()) == ["config.yaml", "converted.yaml"]
    written = yaml.safe_load((tmp_path / "converted.yaml").read_text(encoding="utf-8"))
    assert written["engine"]["ais_perf_model"] == OLD_FIELDS


def test_migrate_that_cannot_convert_one_field_converts_nothing_and_exits_1(tmp_path, capsys):
    fields = {**OLD_FIELDS, "systems_path": "data/a", "systems_paths": ["data/b"]}
    config = _write(tmp_path / "config.yaml", _predict(**fields))
    output = tmp_path / "converted.yaml"
    assert _run("migrate", config, "--output", output, "--format", "json") == 1
    problems = json.loads(capsys.readouterr().out)["problems"]

    assert _run("migrate", config, "--output", output) == 1

    # After the three lines that name the config, the plan prints the problems of the report.
    assert problems
    assert capsys.readouterr().out.splitlines()[3:] == [*problems, "Nothing was converted."]
    assert not output.exists()


def _afd_engine(afd: dict[str, Any]) -> dict[str, Any]:
    """Engine fields for an AFD decode worker next to an ordinary prefill worker, both with old fields."""
    old_timing = {"estimation_mode": "op_level"}
    workers = {"prefill": {"timing": old_timing}, "decode": {"type": "afd", "afd": afd, "timing": old_timing}}
    return {"backend": "sglang", "mode": "disaggregated", "workers": workers, **OLD_FIELDS}


AFD_ROLE = {"type": "afd", "afd": {"a_batch_size": 1}}
AFD_SPACE = {"backend": ["sglang"], "deployment_mode": ["disagg"], "afd_roles": {"decode": AFD_ROLE}}


@pytest.mark.parametrize(
    ("config", "kind", "reason"),
    [
        (
            _predict(**_afd_engine({"n_a_nodes": 1, "n_f_nodes": 1, "tp_a": 1, "a_batch_size": 8})),
            "predict",
            "engine.workers.decode.type is afd",
        ),
        (
            _predict(**_afd_engine({"a_batch_size": 8})) | {"optimization": OPTIMIZATION, "optimizer": OPTIMIZER},
            "recommend",
            "engine.workers.decode.type is afd",
        ),
        (
            _search(**AFD_SPACE, **OLD_FIELDS, role_estimator_controls={"decode": {"estimation_mode": "op_level"}}),
            "search",
            "search_space.afd_roles is set",
        ),
    ],
    ids=["predict", "recommend", "search"],
)
def test_migrate_of_a_config_with_afd_says_why_and_leaves_the_file_as_it_was(config, kind, reason, tmp_path, capsys):
    path = _write(tmp_path / "config.yaml", config)
    written = path.read_bytes()

    assert _run("migrate", path, "--output", path) == 1

    out = capsys.readouterr().out
    assert f"kind: {kind}" in out
    assert out.splitlines()[3:] == [
        f"cannot convert: {reason}, and ais_perf_model does not cover AFD yet",
        "Nothing was converted.",
    ]
    assert path.read_bytes() == written
    # check takes the value of the key, never a whole config, with AFD or without.
    assert _run("check", path) == 2
    assert "VALUE is a whole config" in capsys.readouterr().err
    assert path.read_bytes() == written


@pytest.mark.parametrize(
    ("config", "kind", "key"),
    [
        (_recommend(optimization=OPTIMIZATION, optimizer=OPTIMIZER), "recommend", "engine.ais_perf_model"),
        (_recommend(optimizer=OPTIMIZER), "recommend", "engine.ais_perf_model"),
        (_recommend(optimization=OPTIMIZATION), "recommend", "engine.ais_perf_model"),
        (_search(**OLD_FIELDS), "search", "search_space.ais_perf_model"),
        # A search_space section makes a search config, whatever else the file has.
        (_search(**OLD_FIELDS) | {"engine": {}}, "search", "search_space.ais_perf_model"),
    ],
    ids=[
        "optimization and optimizer",
        "optimizer only",
        "optimization only",
        "search_space",
        "search_space and engine",
    ],
)
def test_migrate_reads_the_kind_of_config_from_its_sections(config, kind, key, tmp_path, capsys):
    assert _run("migrate", _write(tmp_path / "config.yaml", config)) == 0

    out = capsys.readouterr().out
    assert f"kind: {kind}" in out
    assert f"new key {key}:" in out


def test_migrate_of_a_file_that_is_no_known_kind_asks_for_the_kind(tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", {"traffic": TRAFFIC})

    assert _run("migrate", config) == 2

    error = capsys.readouterr().err
    assert f"cannot tell what kind of config {config} is" in error
    assert "name the kind with --kind (predict, recommend or search)" in error


def test_migrate_kind_option_is_used_instead_of_the_sections(tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))

    assert _run("migrate", config, "--kind", "search") == 2

    assert "a search config needs a top-level search_space mapping" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("fields", "status", "code"),
    [
        (OLD_FIELDS, "converted", 0),
        ({}, "nothing_to_convert", 0),
        ({"systems_path": "data/a", "systems_paths": ["data/b"]}, "cannot_convert", 1),
    ],
)
def test_migrate_json_report_states_the_outcome(fields, status, code, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path / "config.yaml", _predict(**fields))

    assert _run("migrate", config, "--format", "json") == code

    report = json.loads(capsys.readouterr().out)
    removed, notes, problems = (report.pop(name) for name in ("removed", "notes", "problems"))
    converted = status == "converted"
    assert report == {
        "config": str(config),
        "kind": "predict",
        "working_directory": str(tmp_path),
        "status": status,
        "ais_perf_model": OLD_FIELDS if converted else None,
        "output": None,
    }
    # The conversion words the removed fields, the notes and the problems; the report lists them as text.
    assert all(isinstance(line, str) for line in [*removed, *notes, *problems])
    assert bool(removed) is converted
    assert bool(problems) is (status == "cannot_convert")


def test_migrate_json_report_names_the_file_it_wrote(tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    output = tmp_path / "converted.yaml"

    assert _run("migrate", config, "--output", output, "--format", "json") == 0

    assert json.loads(capsys.readouterr().out)["output"] == str(output)
    assert output.exists()


def test_migrate_rewrites_a_relative_path_for_the_folder_of_the_config(tmp_path, monkeypatch, capsys):
    run_folder = tmp_path / "run"
    (run_folder / "data" / "systems").mkdir(parents=True)
    monkeypatch.chdir(run_folder)
    config = _write(tmp_path / "configs" / "config.yaml", _predict(systems_paths=["data/systems"]))
    output = tmp_path / "configs" / "converted.yaml"

    assert _run("migrate", config, "--output", output) == 0

    plan = capsys.readouterr().out.splitlines()
    written = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert written["engine"]["ais_perf_model"] == {"systems_paths": ["../run/data/systems"]}
    # The plan prints the notes of the report, one on a line.
    assert _run("migrate", config, "--format", "json") == 0
    notes = json.loads(capsys.readouterr().out)["notes"]
    assert notes
    assert all(f"  {note}" in plan for note in notes)
    # Read from any working directory, the converted file names the folder the old field named from this one.
    monkeypatch.chdir(tmp_path)
    assert CorePredictionConfig.from_yaml(output).engine.systems_paths == [str(run_folder / "data" / "systems")]


def test_migrate_rewrites_a_relative_path_for_the_folder_of_the_output_file(tmp_path, monkeypatch, capsys):
    run_folder = tmp_path / "run"
    (run_folder / "data" / "systems").mkdir(parents=True)
    monkeypatch.chdir(run_folder)
    config = _write(tmp_path / "configs" / "config.yaml", _predict(systems_paths=["data/systems"]))
    # The output file is in another folder, at another depth, than the config.
    output = tmp_path / "converted" / "nightly" / "config.yaml"
    output.parent.mkdir(parents=True)
    with pytest.warns(AisPerfModelDeprecationWarning):
        original = CorePredictionConfig.from_yaml(config)

    assert _run("migrate", config, "--output", output) == 0
    capsys.readouterr()
    # Without --output the paths are still written for the folder of the config.
    assert _run("migrate", config, "--format", "json") == 0
    assert json.loads(capsys.readouterr().out)["ais_perf_model"] == {"systems_paths": ["../run/data/systems"]}

    # Read from any working directory, the file names the folder the old field named.
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converted = CorePredictionConfig.from_yaml(output)
    assert converted.engine.systems_paths == [str(run_folder / "data" / "systems")]
    assert converted.model_dump(mode="json") == original.model_dump(mode="json")


def _run_folder_with_a_linked_config(tmp_path: Path) -> Path:
    """A folder to run from: ``data``, a config in ``configs`` whose old field names ``data``, and links to files there.

    ``link.yaml`` names the config. ``new.yaml`` names ``converted/new.yaml``, a file that does not exist yet.
    """
    run_folder = tmp_path / "run"
    (run_folder / "data").mkdir(parents=True)
    (run_folder / "converted").mkdir()
    _write(run_folder / "configs" / "predict.yaml", _predict(systems_paths=["data"]))
    (run_folder / "link.yaml").symlink_to(Path("configs", "predict.yaml"))
    (run_folder / "new.yaml").symlink_to(Path("converted", "new.yaml"))
    return run_folder


@pytest.mark.parametrize(
    ("config", "output", "link", "written"),
    [
        ("configs/predict.yaml", "link.yaml", "link.yaml", "configs/predict.yaml"),
        ("link.yaml", "link.yaml", "link.yaml", "configs/predict.yaml"),
        ("link.yaml", "configs/predict.yaml", "link.yaml", "configs/predict.yaml"),
        ("configs/predict.yaml", "new.yaml", "new.yaml", "converted/new.yaml"),
    ],
    ids=[
        "the file to its link",
        "the link in place",
        "the link to its file",
        "a link to a file that does not exist yet",
    ],
)
def test_migrate_through_a_link_in_another_folder_reads_the_same_folder_by_both_names(
    config, output, link, written, tmp_path, monkeypatch, capsys
):
    run_folder = _run_folder_with_a_linked_config(tmp_path)
    data = [str(run_folder / "data")]
    monkeypatch.chdir(run_folder)
    with pytest.warns(AisPerfModelDeprecationWarning):
        originals = [CorePredictionConfig.from_yaml(name) for name in ("configs/predict.yaml", "link.yaml")]
    assert [original.engine.systems_paths for original in originals] == [data, data]

    assert _run("migrate", config, "--output", output) == 0
    capsys.readouterr()

    link, file = run_folder / link, run_folder / written
    assert link.is_symlink()
    assert not file.is_symlink()
    # A relative path would start at the folder of the name the file is read by, so the path is written in full.
    assert yaml.safe_load(file.read_text(encoding="utf-8"))["engine"]["ais_perf_model"] == {"systems_paths": data}
    # Read by either name, from any working directory, the file names the folder the old field named.
    monkeypatch.chdir(tmp_path)
    for name in (link, file):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            converted = CorePredictionConfig.from_yaml(name)
        assert converted.engine.systems_paths == data
        assert converted.model_dump(mode="json") == originals[0].model_dump(mode="json")


def test_migrate_through_a_link_in_another_folder_says_why_the_path_is_absolute(tmp_path, monkeypatch, capsys):
    run_folder = _run_folder_with_a_linked_config(tmp_path)
    data = str(run_folder / "data")
    monkeypatch.chdir(run_folder)

    # Without --output the plan is the one for the config itself, here the link.
    assert _run("migrate", "link.yaml", "--format", "json") == 0
    report = json.loads(capsys.readouterr().out)

    assert report["ais_perf_model"] == {"systems_paths": [data]}
    [note] = report["notes"]
    assert f"path 'data' (relative to the working directory) rewritten as {data!r} (an absolute path: " in note
    assert f"has names in {run_folder} and {run_folder / 'configs'}" in note
    assert _run("migrate", "link.yaml") == 0
    assert f"  {note}" in capsys.readouterr().out.splitlines()
    # The command is given one name of the file here, so the plan keeps a relative path.
    assert _run("migrate", "configs/predict.yaml", "--format", "json") == 0
    assert json.loads(capsys.readouterr().out)["ais_perf_model"] == {"systems_paths": ["../data"]}


def test_migrate_through_a_link_in_the_folder_of_its_file_keeps_relative_paths(tmp_path, monkeypatch, capsys):
    run_folder = _run_folder_with_a_linked_config(tmp_path)
    config = run_folder / "configs" / "predict.yaml"
    link = run_folder / "configs" / "current.yaml"
    link.symlink_to("predict.yaml")
    monkeypatch.chdir(run_folder)

    assert _run("migrate", "configs/predict.yaml", "--output", "configs/current.yaml") == 0
    capsys.readouterr()

    assert link.is_symlink()
    assert not config.is_symlink()
    written = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert written["engine"]["ais_perf_model"] == {"systems_paths": ["../data"]}
    monkeypatch.chdir(tmp_path)
    for name in (config, link):
        assert CorePredictionConfig.from_yaml(name).engine.systems_paths == [str(run_folder / "data")]


def test_migrate_through_links_counts_the_folder_of_a_link_between_the_first_link_and_the_file(
    tmp_path, monkeypatch, capsys
):
    run_folder = tmp_path / "run"
    (run_folder / "data").mkdir(parents=True)
    config = _write(run_folder / "predict.yaml", _predict(systems_paths=["data"]))
    # The first link and the file are in one folder; the link between them is in another.
    between = tmp_path / "shared" / "predict.yaml"
    between.parent.mkdir()
    between.symlink_to(config)
    first = run_folder / "link.yaml"
    first.symlink_to(between)
    monkeypatch.chdir(run_folder)

    assert _run("migrate", "predict.yaml", "--output", "link.yaml") == 0
    capsys.readouterr()

    assert first.is_symlink()
    assert between.is_symlink()
    assert not config.is_symlink()
    monkeypatch.chdir(tmp_path)
    for name in (config, between, first):
        assert CorePredictionConfig.from_yaml(name).engine.systems_paths == [str(run_folder / "data")]


def test_migrate_of_a_config_that_cannot_be_read_is_a_usage_error(tmp_path, capsys):
    assert _run("migrate", tmp_path / "missing.yaml") == 2
    assert "could not read configuration" in capsys.readouterr().err

    broken = tmp_path / "broken.yaml"
    broken.write_text("engine: [", encoding="utf-8")
    assert _run("migrate", broken) == 2
    assert f"malformed YAML in {broken}" in capsys.readouterr().err

    not_utf8 = tmp_path / "latin1.yaml"
    not_utf8.write_bytes("engine:\n  model: café\n".encode("latin-1"))
    assert _run("migrate", not_utf8) == 2
    assert f"could not read configuration {not_utf8}: " in capsys.readouterr().err

    invalid = _write(tmp_path / "invalid.yaml", _predict(estimation_mode="no_such_mode"))
    assert _run("migrate", invalid, "--output", tmp_path / "converted.yaml") == 2
    assert "the engine section is not a valid predict config" in capsys.readouterr().err
    assert not (tmp_path / "converted.yaml").exists()


def test_migrate_reports_a_conversion_that_fails_its_own_check_and_writes_nothing(tmp_path, monkeypatch, capsys):
    def changes_the_meaning(*args, **kwargs):
        raise migrate_module.MigrationCheckError("the converted config does not read back as the original")

    monkeypatch.setattr(migrate_module, "migrate_mapping", changes_the_meaning)
    config = _write(tmp_path / "config.yaml", _predict(**OLD_FIELDS))
    output = tmp_path / "converted.yaml"

    assert _run("migrate", config, "--output", output) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("internal error: the converted config does not read back as the original.")
    assert "Nothing was converted and nothing was written" in captured.err
    assert not output.exists()


# --- fetch ------------------------------------------------------------------------------------


@pytest.fixture
def profile_store(monkeypatch, tmp_path):
    """Replace the profile store by two profiles of one manifest, and record what the command asks of it."""
    calls: dict[str, list[Any]] = {"listed": [], "fetched": []}
    folder = tmp_path / "cache" / "h200-tp1" / "systems"

    def list_profiles(manifest=None):
        calls["listed"].append(manifest)
        return PROFILES

    def materialize(manifest, profile, **options):
        calls["fetched"].append((manifest, profile, options))
        return folder

    monkeypatch.setattr(fpm_dataset, "list_fpm_profiles", list_profiles)
    monkeypatch.setattr(fpm_dataset, "materialize_fpm_profile", materialize)
    calls["folder"] = [folder]
    return calls


def test_fetch_list_prints_the_profiles_and_fetches_nothing(profile_store, capsys):
    assert _run("fetch", "--list") == 0

    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert rows == [
        ["profile", "admission", "manifest"],
        ["h200-tp1", "serving", "example-manifest"],
        ["h200-tp2", "rejected", "example-manifest"],
    ]
    # No MANIFEST asks for the profiles of every manifest shipped with the package.
    assert profile_store["listed"] == [None]
    assert profile_store["fetched"] == []


def test_fetch_list_of_one_manifest_as_json(profile_store, capsys):
    assert _run("fetch", "--list", "example-manifest", "--format", "json") == 0

    assert json.loads(capsys.readouterr().out) == PROFILES
    assert profile_store["listed"] == ["example-manifest"]


def test_fetch_list_says_when_there_are_no_profiles(monkeypatch, capsys):
    monkeypatch.setattr(fpm_dataset, "list_fpm_profiles", lambda manifest=None: [])

    assert _run("fetch", "--list") == 0

    assert capsys.readouterr().out == "No profiles.\n"


def test_fetch_list_reads_the_manifests_shipped_with_the_package(monkeypatch, capsys):
    def no_download(*args, **kwargs):
        pytest.fail("listing profiles must not fetch one")

    monkeypatch.setattr(fpm_dataset, "materialize_fpm_profile", no_download)

    assert _run("fetch", "--list", "--format", "json") == 0

    profiles = json.loads(capsys.readouterr().out)
    assert profiles
    assert all({"manifest", "profile", "admission"} <= set(record) for record in profiles)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--list", "example-manifest", "h200-tp1"], "--list takes at most MANIFEST; leave PROFILE out"),
        ([], "fetch takes MANIFEST and PROFILE; use --list to see the profiles"),
        (["example-manifest"], "fetch takes MANIFEST and PROFILE; use --list to see the profiles"),
    ],
)
def test_fetch_with_the_wrong_arguments_is_a_usage_error(arguments, message, profile_store, capsys):
    assert _run("fetch", *arguments) == 2

    assert message in capsys.readouterr().err
    assert profile_store["fetched"] == []


def test_fetch_prints_the_folder_and_the_line_to_write_in_the_entry(profile_store, capsys):
    [folder] = profile_store["folder"]

    assert _run("fetch", "example-manifest", "h200-tp1") == 0

    assert capsys.readouterr().out.splitlines() == [
        f"profile h200-tp1 is at {folder}",
        "Name the folder in the ais_perf_model entry that uses the profile:",
        f'  systems_paths: ["{folder}"]',
    ]
    options = {"cache_dir": None, "local_files_only": False, "allow_unqualified": False}
    assert profile_store["fetched"] == [("example-manifest", "h200-tp1", options)]


def test_fetch_passes_the_cache_folder_and_the_download_options_on(profile_store, tmp_path, capsys):
    [folder] = profile_store["folder"]
    cache = tmp_path / "cache"
    arguments = ["--cache-dir", cache, "--local-files-only", "--allow-unqualified", "--format", "json"]

    assert _run("fetch", "example-manifest", "h200-tp2", *arguments) == 0

    assert json.loads(capsys.readouterr().out) == {
        "manifest": "example-manifest",
        "profile": "h200-tp2",
        "path": str(folder),
        "systems_paths": [str(folder)],
    }
    options = {"cache_dir": str(cache), "local_files_only": True, "allow_unqualified": True}
    assert profile_store["fetched"] == [("example-manifest", "h200-tp2", options)]


@pytest.fixture
def published_profile(tmp_path, monkeypatch):
    """A manifest file with one profile; the files of the profile are served from memory in place of the download."""
    bodies = {
        "system.yaml": b"data_dir: data\n",
        "data/fpm_forward_perf.parquet": b"the bytes of a table",
        "data/fpm_forward_perf.metadata.json": b'{"schema_version": 7}',
    }
    files = [
        {"path": name, "target": name, "sha256": hashlib.sha256(body).hexdigest()} for name, body in bodies.items()
    ]
    profiles = {"admitted": {"admission": "serving", "files": files}}
    manifest = {"format_version": 1, "repo_id": "example/fpm", "revision": "a" * 40, "profiles": profiles}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    def served(request, **options):
        return io.BytesIO(bodies[request.full_url.partition(f"/resolve/{manifest['revision']}/")[2]])

    monkeypatch.setattr(fpm_dataset, "urlopen", served)
    # Without this the profile store adds the access token of the user to the request.
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    return path


@pytest.mark.parametrize(
    ("options", "environment", "cache"),
    [(["--cache-dir", "cache"], {}, "cache"), ([], {"XDG_CACHE_HOME": "xdg"}, "xdg/aisimulate/fpm")],
    ids=["--cache-dir", "XDG_CACHE_HOME"],
)
def test_fetch_with_a_relative_cache_folder_prints_an_absolute_folder(
    options, environment, cache, published_profile, tmp_path, monkeypatch, capsys
):
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    monkeypatch.chdir(run_folder)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    # The first command stores the profile and the second finds it in the cache. The profile store
    # returns a folder that starts at the cache folder as it was given, so a relative one both times.
    assert _run("fetch", published_profile, "admitted", *options) == 0
    table = capsys.readouterr().out.splitlines()
    assert _run("fetch", published_profile, "admitted", *options, "--local-files-only", "--format", "json") == 0
    report = json.loads(capsys.readouterr().out)

    [stored] = (run_folder / cache).iterdir()
    assert (stored / "data" / "fpm_forward_perf.parquet").is_file()
    assert report == {
        "manifest": str(published_profile),
        "profile": "admitted",
        "path": str(stored),
        "systems_paths": [str(stored)],
    }
    assert table == [
        f"profile admitted is at {stored}",
        "Name the folder in the ais_perf_model entry that uses the profile:",
        f'  systems_paths: ["{stored}"]',
    ]
    # The printed line names the folder of the profile from an entry file in any folder.
    entry = _write(tmp_path / "configs" / "perf-model.yaml", {"systems_paths": report["systems_paths"]})
    monkeypatch.chdir(tmp_path)
    assert load(entry).entries[0]["systems_paths"] == [str(stored)]


def test_fetch_with_a_cache_folder_that_goes_up_from_a_linked_folder_prints_the_folder_the_system_reaches(
    published_profile, tmp_path, monkeypatch, capsys
):
    run_folder = tmp_path / "run"
    (run_folder / "elsewhere" / "deep").mkdir(parents=True)
    (run_folder / "link").symlink_to(Path("elsewhere", "deep"), target_is_directory=True)
    monkeypatch.chdir(run_folder)
    assert _run("fetch", published_profile, "admitted", "--cache-dir", "elsewhere/cache") == 0
    capsys.readouterr()
    [stored] = (run_folder / "elsewhere" / "cache").iterdir()
    # From the linked folder, ".." is the parent of the folder the link names. Dropped as text,
    # "link/.." would leave "cache" in the working directory, which is not where the profile is.
    again = ["--cache-dir", "link/../cache", "--local-files-only", "--format", "json"]

    assert _run("fetch", published_profile, "admitted", *again) == 0

    assert json.loads(capsys.readouterr().out)["systems_paths"] == [str(stored)]


def test_fetch_of_a_profile_the_manifest_does_not_have_is_a_usage_error(profile_store, capsys):
    assert _run("fetch", "example-manifest", "h200-tp8") == 2

    error = capsys.readouterr().err
    assert "manifest example-manifest has no profile 'h200-tp8'; it has: h200-tp1, h200-tp2" in error
    assert profile_store["fetched"] == []


@pytest.mark.parametrize(
    "failure", [ValueError("a file of the profile is damaged"), FileNotFoundError("not in the cache")]
)
def test_fetch_that_fails_exits_1_and_prints_the_reason(failure, profile_store, monkeypatch, capsys):
    def fails(*args, **kwargs):
        raise failure

    monkeypatch.setattr(fpm_dataset, "materialize_fpm_profile", fails)

    assert _run("fetch", "example-manifest", "h200-tp1") == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"could not fetch profile 'h200-tp1': {failure}\n"


@pytest.fixture
def local_manifest(tmp_path):
    """A manifest file with a profile admitted for serving and one that is not; no file of either is in a cache."""
    names = ["system.yaml", "data/fpm_forward_perf.parquet", "data/fpm_forward_perf.metadata.json"]
    files = [{"path": name, "target": name, "sha256": "0" * 64} for name in names]
    profiles = {
        "admitted": {"admission": "serving", "files": files},
        "development-only": {"admission": "development", "admission_reason": "not reviewed yet", "files": files},
    }
    manifest = {"format_version": 1, "repo_id": "example/fpm", "revision": "a" * 40, "profiles": profiles}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_fetch_of_a_profile_not_admitted_for_serving_names_the_option_that_allows_it(local_manifest, tmp_path, capsys):
    cache = tmp_path / "cache"

    assert _run("fetch", local_manifest, "development-only", "--cache-dir", cache, "--local-files-only") == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    reason, *hint = captured.err.splitlines()
    assert reason.startswith("could not fetch profile 'development-only': ")
    assert hint == ["Pass --allow-unqualified to fetch the profile anyway."]
    assert not cache.exists()


def test_fetch_that_fails_for_another_reason_does_not_name_the_option(local_manifest, tmp_path, capsys):
    offline = ["--cache-dir", tmp_path / "cache", "--local-files-only"]

    # The profile is admitted; it is only not in the cache.
    assert _run("fetch", local_manifest, "admitted", *offline) == 1
    assert "--allow-unqualified" not in capsys.readouterr().err

    # The option is given, so the profile fails for the same reason as the admitted one.
    with pytest.warns(UserWarning, match="development-only"):
        assert _run("fetch", local_manifest, "development-only", "--allow-unqualified", *offline) == 1
    error = capsys.readouterr().err
    assert error.startswith("could not fetch profile 'development-only': ")
    assert "--allow-unqualified" not in error
