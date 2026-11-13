# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relative paths, repeated keys and the deprecation warning where config files are read.

Three places read a prediction or recommendation config file: the parent
CLI, the supervised child it starts, and the ``from_yaml`` loaders. Each one
starts the relative paths inside ``engine.ais_perf_model`` at the folder of
the file, and rejects a key the file writes twice inside it. The parent CLI
and the loaders warn once when the config states old estimator fields; the
child and plain validation never warn.
"""

from __future__ import annotations

import contextlib
import copy
import os
import shlex
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import aisim.main as cli
from aisim import supervision
from aisim.config.cli import CorePredictionConfig, CoreRecommendationConfig
from aisim.supervision import main as supervised_main
from aisim.sweeper.replay import RunnerCapabilities
from aisim_core.ais_perf_model import AisPerfModelDeprecationWarning, AisPerfModelError

# Onboarding fixtures: a reviewed profile, the replay config onboarding writes from it, and serving preparation.
from .test_support_serving_validation import materialized_workload, serving_case  # noqa: F401
from .test_support_validation import validation_case as _replay_validation_case  # noqa: F401

pytestmark = pytest.mark.unit

MODEL, SYSTEM = "Qwen/Qwen3-0.6B", "h200_sxm"
TRAFFIC = {
    "source": {"type": "synthetic", "input_tokens": 128, "output_tokens": 16},
    "load": {"type": "concurrency", "concurrency": 4},
    "stop": {"requests": 8},
}
SEARCH = {
    "optimization": {"target": "throughput", "constraints": {"max_candidate_gpus": 2}},
    "optimizer": {"algorithm": "random", "max_trials": 1, "parallelism": 1},
}
ENGINE = {
    "model": MODEL,
    "hardware": SYSTEM,
    "backend": "vllm",
    "context_length": 256,
    "mode": "aggregated",
    "workers": {"aggregated": {}},
}
LOADERS = {"predict": CorePredictionConfig, "recommend": CoreRecommendationConfig}
# The estimator settings every test uses: written as old engine fields, or kept in a template file.
OLD_FIELDS = {"estimation_mode": "op_level", "fallback_policy": "allow"}
OLD_PATHS = "engine.estimation_mode, engine.fallback_policy"
# One relative path to a template file and one relative path to a perf data folder.
ENTRY = {"from": "templates/perf.yaml", "systems_paths": ["perf-data"]}
FIXED_TIMING = {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}


def _mapping(command: str, engine_fields: dict[str, Any]) -> dict[str, Any]:
    """A whole config mapping for ``predict`` or ``recommend`` with extra engine fields."""
    sections = SEARCH if command == "recommend" else {}
    return {"traffic": TRAFFIC, "engine": {**ENGINE, **engine_fields}, **sections}


def _write(path: Path, engine_fields: dict[str, Any], command: str = "predict") -> Path:
    path.write_text(yaml.safe_dump(_mapping(command, engine_fields)))
    return path


def _argv(command: str, config: Path, output: Path, overrides: tuple[str, ...]) -> list[str]:
    argv = [command, "--config", str(config), "--output-dir", str(output)]
    for override in overrides:
        argv += ["--set", override]
    return argv


def _supervised(argv: list[str]) -> list[str]:
    """The command the parent CLI starts for ``argv``: the child gets the same arguments, unchanged."""
    return [sys.executable, "-m", "aisim.resource_worker", "cli", *argv]


def _dump(config: CorePredictionConfig | CoreRecommendationConfig) -> dict[str, Any]:
    return config.model_dump(mode="json", exclude_none=True)


def _old_spelling(config_dir: Path) -> dict[str, Any]:
    """The engine fields that ``ENTRY`` stands for when the config file is in ``config_dir``."""
    return {**OLD_FIELDS, "systems_paths": [str(config_dir / "perf-data")]}


def _migrate_hint(config: Path) -> str:
    """Return the hint a warning gives for the file ``config``, which names the file as a shell reads it."""
    return f"run 'aisim perf-model migrate {shlex.quote(str(config))}' to convert the file"


@contextlib.contextmanager
def _deprecations():
    """Yield a list that holds, after the block, the ``ais_perf_model`` deprecation warnings raised in it."""
    found: list[warnings.WarningMessage] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield found
    found.extend(warning for warning in caught if issubclass(warning.category, AisPerfModelDeprecationWarning))


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """A folder for config files, with a template and a perf data folder; the working directory is elsewhere."""
    folder = Path(os.path.realpath(tmp_path)) / "real" / "configs"
    (folder / "templates").mkdir(parents=True)
    (folder / "perf-data").mkdir()
    (folder / "templates" / "perf.yaml").write_text(yaml.safe_dump(OLD_FIELDS))
    elsewhere = folder.parent.parent / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return folder


class _Parent:
    """The parent CLI with the supervised child process replaced by a recorder."""

    def __init__(self, monkeypatch, output: Path) -> None:
        self.output = output
        self.child_exit_code = 0
        self.child_commands: list[list[str]] = []
        monkeypatch.setattr(supervision, "run_process", self._child)

    def _child(self, command, *, policy, events_output):
        del policy, events_output
        self.child_commands.append(command)
        return {"status": "completed", "exit_code": self.child_exit_code}

    def run(self, command: str, config: Path, *overrides: str) -> int:
        return supervision.main(_argv(command, config, self.output, overrides))

    def prepared_output(self) -> bool:
        """Whether the parent created the output directory and saved its own report there."""
        return (self.output / "resource-runtime.json").is_file()


@pytest.fixture
def parent(monkeypatch, tmp_path):
    return _Parent(monkeypatch, tmp_path / "parent-output")


class _RecordingRunner:
    """Keeps the replay request the child compiled and stops before any simulation runs."""

    def __init__(self) -> None:
        self.compiled = None

    def run(self, compiled, *, output_requirements=None):
        del output_requirements
        self.compiled = compiled
        raise RuntimeError("stopped after the request was compiled")

    def close(self) -> None:
        pass


class _Factory:
    def __init__(self, runner: _RecordingRunner) -> None:
        self.runner = runner

    def capabilities(self):
        return RunnerCapabilities(supported_backend_topologies=(("vllm", "agg"),))

    def create(self, worker_id: int):
        del worker_id
        return self.runner


@pytest.fixture
def child(monkeypatch, tmp_path):
    """Run the supervised child CLI in process and return the replay request it compiled."""

    def run(config: Path, *overrides: str):
        runner = _RecordingRunner()
        monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: _Factory(runner))
        # The recording runner stops the run, which the child reports as exit code 1.
        assert cli.main(_argv("predict", config, tmp_path / "child-output", overrides)) == 1
        assert runner.compiled is not None, "the child stopped before it compiled a request"
        return runner.compiled

    return run


def _estimator(compiled) -> dict[str, Any]:
    """The estimator settings of the aggregated worker in a compiled replay request."""
    config = compiled.backend_deployment.agg_engine_args["timing_model"]["config"]
    return {key: config[key] for key in ("estimation_mode", "fallback_policy", "systems_paths")}


def _child_error(capsys, config: Path, *overrides: str, command: str = "predict") -> str:
    """Run the child on a config it must reject and return what it wrote to stderr."""
    with pytest.raises(SystemExit) as stopped:
        cli.main(_argv(command, config, config.parent / "never-created", overrides))
    assert stopped.value.code == 2
    assert not (config.parent / "never-created").exists()
    return capsys.readouterr().err


# --- Old estimator fields, with ais_perf_model absent ----------------------------------------


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_parent_warns_once_about_old_fields_in_the_file(config_dir, parent, command):
    path = _write(config_dir / "old.yaml", OLD_FIELDS, command)
    parent.child_exit_code = 5

    with _deprecations() as found:
        exit_code = parent.run(command, path)

    assert exit_code == 5, "the warning must not change the exit code, which is the child's"
    (warning,) = found
    assert f"{OLD_PATHS} are deprecated and will be removed in the next release" in str(warning.message)
    assert f"use engine.ais_perf_model ({_migrate_hint(path)})" in str(warning.message)
    assert warning.filename == __file__, "the warning points at the code that called the CLI"
    (child_command,) = parent.child_commands
    assert child_command == _supervised(_argv(command, path, parent.output, ()))
    assert parent.prepared_output()


def test_parent_hint_is_set_when_only_an_override_states_old_fields(config_dir, parent):
    path = _write(config_dir / "plain.yaml", {})

    with _deprecations() as found:
        assert parent.run("predict", path, "engine.estimation_mode=op_level") == 0

    (warning,) = found
    assert "engine.estimation_mode is deprecated" in str(warning.message)
    assert "use engine.ais_perf_model (set engine.ais_perf_model.<key> with --set instead)" in str(warning.message)


def test_parent_hint_is_migrate_when_the_file_and_an_override_state_old_fields(config_dir, parent):
    path = _write(config_dir / "old.yaml", OLD_FIELDS)

    with _deprecations() as found:
        assert parent.run("predict", path, "engine.estimation_mode=fpm_interpolation") == 0

    (warning,) = found
    assert OLD_PATHS in str(warning.message)
    assert _migrate_hint(path) in str(warning.message)


@pytest.mark.parametrize(
    "file_name",
    ["old config.yaml", "it's old.yaml", "old$(whoami);.yaml"],
    ids=["space", "quote", "shell syntax"],
)
def test_parent_hint_is_a_command_that_a_shell_reads_as_naming_the_config_file(config_dir, parent, file_name):
    path = _write(config_dir / file_name, OLD_FIELDS)

    with _deprecations() as found:
        assert parent.run("predict", path) == 0

    (warning,) = found
    command = str(warning.message).split("(run '", 1)[1].rsplit("' to convert the file)", 1)[0]
    assert shlex.split(command) == ["aisim", "perf-model", "migrate", str(path)]


def test_parent_does_not_warn_about_a_config_it_rejects(config_dir, parent, capsys):
    path = _write(config_dir / "invalid.yaml", {**OLD_FIELDS, "no_such_field": 1})

    with _deprecations() as found, pytest.raises(SystemExit) as stopped:
        parent.run("predict", path)

    assert stopped.value.code == 2
    assert "no_such_field" in capsys.readouterr().err
    assert found == []
    assert not parent.output.exists()


def test_child_does_not_warn_about_old_fields(config_dir, child):
    path = _write(config_dir / "old.yaml", OLD_FIELDS)

    with _deprecations() as found:
        compiled = child(path)

    assert found == []
    assert _estimator(compiled)["estimation_mode"] == "op_level"
    assert _estimator(compiled)["fallback_policy"] == "allow"


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_from_yaml_warns_about_old_fields_and_model_validate_does_not(config_dir, command):
    path = _write(config_dir / "old.yaml", OLD_FIELDS, command)

    with _deprecations() as from_file:
        loaded = LOADERS[command].from_yaml(path)
    with _deprecations() as from_data:
        validated = LOADERS[command].model_validate(yaml.safe_load(path.read_text()))

    (warning,) = from_file
    assert OLD_PATHS in str(warning.message)
    assert _migrate_hint(path) in str(warning.message)
    assert warning.filename == __file__, "the warning points at the code that called from_yaml"
    assert from_data == []
    assert _dump(loaded) == _dump(validated)


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_from_yaml_does_not_warn_about_a_file_it_rejects(config_dir, command):
    path = _write(config_dir / "invalid.yaml", {**OLD_FIELDS, "no_such_field": 1}, command)

    with _deprecations() as found, pytest.raises(ValidationError, match="no_such_field"):
        LOADERS[command].from_yaml(path)

    assert found == []


def test_default_values_written_out_do_not_warn(config_dir, parent):
    plain = CorePredictionConfig.model_validate(_mapping("predict", {}))
    saved = _dump(plain)
    assert saved["engine"]["estimation_mode"] == "auto"
    assert saved["engine"]["fallback_policy"] == "deny"
    path = config_dir / "saved.yaml"
    path.write_text(yaml.safe_dump(saved))

    with _deprecations() as found:
        loaded = CorePredictionConfig.from_yaml(path)
        assert parent.run("predict", path) == 0

    assert found == []
    assert _dump(loaded) == saved


def test_old_field_does_not_warn_in_a_config_the_new_key_cannot_describe(config_dir, parent):
    # ais_perf_model does not cover fixed timing yet, so there is nothing to convert this file to.
    fields = {"systems_paths": [str(config_dir / "perf-data")], "workers": {"aggregated": {"timing": FIXED_TIMING}}}
    path = _write(config_dir / "fixed-timing.yaml", fields)

    with _deprecations() as found:
        loaded = CorePredictionConfig.from_yaml(path)
        assert parent.run("predict", path) == 0

    assert found == []
    assert loaded.engine.systems_paths == [str(config_dir / "perf-data")]


# A decode worker that runs attention and FFN on separate pools (AFD), as each command writes it.
AFD_DECODE = {
    "predict": {"type": "afd", "afd": {"n_a_nodes": 1, "n_f_nodes": 1, "tp_a": 1, "a_batch_size": 8}},
    "recommend": {"type": "afd", "afd": {"a_batch_size": 8}},
}


def _afd_fields(command: str, place: str) -> dict[str, Any]:
    """Engine fields for an AFD decode worker next to an ordinary prefill worker, with ``OLD_FIELDS`` in one place.

    ``place`` is ``engine`` for the engine section, or the worker whose timing holds the fields.
    """
    workers: dict[str, Any] = {"prefill": {}, "decode": copy.deepcopy(AFD_DECODE[command])}
    fields = {"backend": "sglang", "mode": "disaggregated", "workers": workers}
    if place == "engine":
        return {**fields, **OLD_FIELDS}
    workers[place]["timing"] = dict(OLD_FIELDS)
    return fields


@pytest.mark.parametrize("place", ["engine", "decode", "prefill"], ids=["engine section", "AFD worker", "other worker"])
@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_old_fields_do_not_warn_in_a_config_with_an_afd_worker(config_dir, parent, command, place):
    # ais_perf_model does not cover AFD yet, so there is nothing to convert this file to.
    path = _write(config_dir / "afd.yaml", _afd_fields(command, place), command)

    with _deprecations() as found:
        loaded = LOADERS[command].from_yaml(path)
        assert parent.run(command, path) == 0

    assert found == []
    stored = _dump(loaded)["engine"]
    assert (stored if place == "engine" else stored["workers"][place]["timing"])["fallback_policy"] == "allow"
    assert _dump(loaded) == _dump(LOADERS[command].model_validate(yaml.safe_load(path.read_text())))


# --- ais_perf_model in a config file: relative paths -----------------------------------------


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_from_yaml_starts_relative_paths_at_the_folder_of_the_file(config_dir, command):
    path = _write(config_dir / "new.yaml", {"ais_perf_model": ENTRY}, command)
    old = LOADERS[command].model_validate(_mapping(command, _old_spelling(config_dir)))

    with _deprecations() as found:
        loaded = LOADERS[command].from_yaml(path)

    assert found == []
    assert _dump(loaded) == _dump(old)
    assert loaded.engine.systems_paths == [str(config_dir / "perf-data")]


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_parent_starts_relative_paths_at_the_folder_of_the_file(config_dir, parent, command):
    path = _write(config_dir / "new.yaml", {"ais_perf_model": ENTRY}, command)

    with _deprecations() as found:
        # Read against the working directory, the template would be missing and the parent would stop with 2.
        assert parent.run(command, path) == 0

    assert found == []
    (child_command,) = parent.child_commands
    assert child_command == _supervised(_argv(command, path, parent.output, ()))
    assert parent.prepared_output()


def test_child_starts_relative_paths_at_the_folder_of_the_file(config_dir, child):
    new = _write(config_dir / "new.yaml", {"ais_perf_model": ENTRY})
    old = _write(config_dir / "old.yaml", _old_spelling(config_dir))

    with _deprecations() as found:
        compiled = child(new)

    assert found == []
    assert _estimator(compiled) == _old_spelling(config_dir)
    assert compiled == child(old)


def test_config_opened_through_a_link_starts_paths_at_the_real_folder(config_dir, parent, child):
    # "configs" is reached as "link"; ".." must mean the parent of the real folder, not of the link.
    shared = config_dir.parent / "shared"
    (shared / "perf-data").mkdir(parents=True)
    (shared / "perf.yaml").write_text(yaml.safe_dump(OLD_FIELDS))
    link = config_dir.parent.parent / "link"
    link.symlink_to(config_dir, target_is_directory=True)
    entry = {"from": "../shared/perf.yaml", "systems_paths": ["../shared/perf-data"]}
    _write(config_dir / "new.yaml", {"ais_perf_model": entry})
    expected = {**OLD_FIELDS, "systems_paths": [str(shared / "perf-data")]}

    loaded = CorePredictionConfig.from_yaml(link / "new.yaml")

    assert {key: getattr(loaded.engine, key) for key in expected} == expected
    assert parent.run("predict", link / "new.yaml") == 0
    assert _estimator(child(link / "new.yaml")) == expected


def test_config_path_that_goes_up_from_a_link_starts_paths_at_the_folder_that_holds_the_file(config_dir, parent, child):
    # "link/.." opens the parent of the real "configs" folder. Read as text it would be the folder
    # next to the link, which holds neither the config file nor the template.
    link = config_dir.parent.parent / "link"
    link.symlink_to(config_dir, target_is_directory=True)
    entry = {"from": "configs/templates/perf.yaml", "systems_paths": ["configs/perf-data"]}
    _write(config_dir.parent / "new.yaml", {"ais_perf_model": entry})
    path = link / ".." / "new.yaml"
    expected = _old_spelling(config_dir)

    loaded = CorePredictionConfig.from_yaml(path)

    assert {key: getattr(loaded.engine, key) for key in expected} == expected
    assert parent.run("predict", path) == 0
    assert _estimator(child(path)) == expected


def test_path_in_the_value_that_goes_up_from_a_link_names_the_folder_the_system_opens(config_dir, child):
    # "hop" names far/deep, so for the system "hop/.." is "far". Read as text it would be the folder of
    # the config file, which has a "perf-data" folder of its own and no "perf.yaml".
    far = config_dir / "far"
    (far / "deep").mkdir(parents=True)
    (far / "perf-data").mkdir()
    (far / "perf.yaml").write_text(yaml.safe_dump(OLD_FIELDS))
    (config_dir / "hop").symlink_to(Path("far", "deep"), target_is_directory=True)
    entry = {"from": "hop/../perf.yaml", "systems_paths": ["hop/../perf-data"]}
    new = _write(config_dir / "new.yaml", {"ais_perf_model": entry})
    # The old field names the same folder with the same text, as an absolute path.
    old_fields = {**OLD_FIELDS, "systems_paths": [str(config_dir / "hop" / ".." / "perf-data")]}
    old = _write(config_dir / "old.yaml", old_fields)
    expected = {**OLD_FIELDS, "systems_paths": [str(far / "perf-data")]}

    with _deprecations() as found:
        loaded = CorePredictionConfig.from_yaml(new)
        compiled = child(new)

    assert found == []
    assert {key: getattr(loaded.engine, key) for key in expected} == expected
    assert _estimator(compiled) == expected
    assert compiled == child(old)


# --- ais_perf_model and --set ----------------------------------------------------------------


def test_set_edits_an_entry_after_the_paths_of_the_file_were_anchored(config_dir, child):
    path = _write(config_dir / "new.yaml", {"ais_perf_model": ENTRY})

    compiled = child(path, "engine.ais_perf_model.fallback_policy=deny")

    assert _estimator(compiled) == {**_old_spelling(config_dir), "fallback_policy": "deny"}


def test_set_creates_an_entry_and_its_paths_start_at_the_working_directory(config_dir, child):
    path = _write(config_dir / "plain.yaml", {})
    (Path.cwd() / "cwd-data").mkdir()

    compiled = child(
        path,
        "engine.ais_perf_model.estimation_mode=op_level",
        "engine.ais_perf_model.systems_paths=[cwd-data]",
    )

    assert _estimator(compiled) == {
        "estimation_mode": "op_level",
        "fallback_policy": "deny",
        "systems_paths": [str(Path.cwd() / "cwd-data")],
    }


def test_model_validate_starts_relative_paths_at_the_working_directory(config_dir):
    (Path.cwd() / "cwd-data").mkdir()
    entry = {"systems_paths": ["cwd-data"]}

    config = CorePredictionConfig.model_validate(_mapping("predict", {"ais_perf_model": entry}))

    assert config.engine.systems_paths == [str(Path.cwd() / "cwd-data")]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            "engine.ais_perf_model.fallback_policy=allow",
            "--set path 'engine.ais_perf_model.fallback_policy' crosses a non-mapping value",
        ),
        (
            "engine.ais_perf_model.0.fallback_policy=allow",
            "invalid --set path 'engine.ais_perf_model.0.fallback_policy'; sequence indexes are unsupported",
        ),
    ],
)
def test_set_cannot_edit_inside_a_list_of_entries(config_dir, capsys, override, message):
    entries = [{"worker_type": "aggregated", "estimation_mode": "op_level"}]
    path = _write(config_dir / "list.yaml", {"ais_perf_model": entries})

    assert f"{path}: {message}" in _child_error(capsys, path, override)


def test_set_replaces_a_list_of_entries_whole(config_dir, child):
    entries = [{"worker_type": "aggregated", "estimation_mode": "op_level"}]
    path = _write(config_dir / "list.yaml", {"ais_perf_model": entries})

    compiled = child(path, "engine.ais_perf_model=[{worker_type: aggregated, fallback_policy: allow}]")

    assert _estimator(compiled)["estimation_mode"] == "auto"
    assert _estimator(compiled)["fallback_policy"] == "allow"


def test_set_null_removes_the_entry(config_dir, child):
    new = _write(config_dir / "new.yaml", {"ais_perf_model": ENTRY})
    plain = _write(config_dir / "plain.yaml", {})

    assert child(new, "engine.ais_perf_model=null") == child(plain)


def test_parent_hint_is_to_remove_the_old_fields_when_set_null_removes_the_key_of_the_file(config_dir, parent):
    # The file states every setting twice. Without the key the run uses the old fields, and migrate
    # does not convert a file that sets the key, so the hint must not send the author there.
    path = _write(config_dir / "both.yaml", {**OLD_FIELDS, "ais_perf_model": OLD_FIELDS})

    with _deprecations() as found:
        assert parent.run("predict", path, "engine.ais_perf_model=null") == 0

    (warning,) = found
    assert str(warning.message).endswith(
        f"{OLD_PATHS} are deprecated and will be removed in the next release; "
        "use engine.ais_perf_model (the file already sets it, so remove the old fields from the file)"
    )
    assert len(parent.child_commands) == 1


@pytest.mark.parametrize("given_with", ["file", "--set"])
def test_string_value_is_rejected_the_same_way_from_a_file_and_from_set(config_dir, capsys, given_with):
    if given_with == "file":
        path, overrides = _write(config_dir / "string.yaml", {"ais_perf_model": "templates/perf.yaml"}), ()
    else:
        path, overrides = _write(config_dir / "plain.yaml", {}), ("engine.ais_perf_model=templates/perf.yaml",)

    error = _child_error(capsys, path, *overrides)

    assert "engine.ais_perf_model cannot be a string; write the mapping or 'from: <file>'" in error


# --- ais_perf_model the config file gets wrong -----------------------------------------------


def test_missing_template_stops_the_parent_before_it_prepares_output(config_dir, parent, capsys):
    path = _write(config_dir / "missing.yaml", {"ais_perf_model": {"from": "templates/absent.yaml"}})

    with pytest.raises(SystemExit) as stopped:
        parent.run("predict", path)

    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert f"error: {path}: " in error
    assert f"cannot read {config_dir / 'templates' / 'absent.yaml'}" in error
    assert parent.child_commands == []
    assert not parent.output.exists()


@pytest.mark.parametrize("given_with", ["file", "--set"])
def test_misspelled_entry_key_stops_the_parent_before_overwrite_removes_the_last_prediction(
    config_dir, parent, capsys, given_with
):
    if given_with == "file":
        path, overrides = _write(config_dir / "misspelled.yaml", {"ais_perf_model": {"estimaton_mode": "op_level"}}), ()
    else:
        path, overrides = _write(config_dir / "plain.yaml", {}), ("engine.ais_perf_model.estimaton_mode=op_level",)
    parent.output.mkdir()
    last_prediction = parent.output / "prediction.json"
    last_prediction.write_text('{"from": "the run before"}')

    with pytest.raises(SystemExit) as stopped:
        supervision.main([*_argv("predict", path, parent.output, overrides), "--overwrite"])

    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert f"error: {path}: " in error
    assert (
        "engine.ais_perf_model entry for aggregated: "
        "estimaton_mode: not a perf model field (did you mean estimation_mode?)"
    ) in error
    assert parent.child_commands == []
    assert last_prediction.read_text() == '{"from": "the run before"}'


def test_missing_template_is_a_config_error_in_the_child_and_in_from_yaml(config_dir, capsys):
    path = _write(config_dir / "missing.yaml", {"ais_perf_model": {"from": "templates/absent.yaml"}})
    missing = f"cannot read {config_dir / 'templates' / 'absent.yaml'}"

    assert missing in _child_error(capsys, path)
    with pytest.raises(ValidationError, match="cannot use the template named by from"):
        CorePredictionConfig.from_yaml(path)


def test_entry_the_loader_rejects_is_reported_by_the_child(config_dir, parent, capsys):
    path = _write(config_dir / "rejected.yaml", {"ais_perf_model": {"worker_type": "mixed"}})
    problem = 'ais_perf_model: worker_type is "mixed"; use one of: prefill, decode, aggregated'

    # The parent treats the file like one it cannot read: it prepares nothing and lets the child report.
    assert parent.run("predict", path) == 0
    assert len(parent.child_commands) == 1
    assert not parent.output.exists()
    assert f"error: {path}: {problem}" in _child_error(capsys, path)
    with pytest.raises(AisPerfModelError, match="worker_type is"):
        CorePredictionConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("entry", "problem"),
    [
        pytest.param({"from": "templates/absent.yaml"}, "cannot use the template named by from", id="missing template"),
        pytest.param({"worker_type": "mixed"}, 'worker_type is "mixed"', id="entry the loader rejects"),
        pytest.param({"estimaton_mode": "auto"}, "estimaton_mode: not a perf model field", id="misspelled key"),
    ],
)
def test_command_line_reports_a_bad_entry_without_a_traceback(config_dir, tmp_path, entry, problem):
    path = _write(config_dir / "bad.yaml", {"ais_perf_model": entry})
    output = tmp_path / "output"

    finished = subprocess.run(
        [sys.executable, "-m", "aisim", *_argv("predict", path, output, ())],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert finished.returncode == 2
    assert problem in finished.stderr
    assert "Traceback" not in finished.stderr
    assert not output.exists()


# --- A key the config file writes twice ------------------------------------------------------

REPEATED_KEY = "the key 'estimation_mode' is written twice (lines 10 and 12); write it once"


def _write_text(path: Path, command: str, *engine_lines: str) -> Path:
    """Write a config file with extra lines in its engine section. A dump cannot write a key twice."""
    rest = {name: value for name, value in _mapping(command, {}).items() if name != "engine"}
    engine = yaml.safe_dump({"engine": ENGINE}) + "".join(f"  {line}\n" for line in engine_lines)
    path.write_text(engine + yaml.safe_dump(rest))
    return path


def _write_repeated_key(path: Path, command: str = "predict") -> Path:
    """Write a config whose ``engine.ais_perf_model`` gives ``estimation_mode`` twice, on lines 10 and 12."""
    lines = ("ais_perf_model:", "  estimation_mode: op_level", "  fallback_policy: allow", "  estimation_mode: auto")
    _write_text(path, command, *lines)
    # The YAML reader keeps the last value, and the run would follow it without a word.
    assert yaml.safe_load(path.read_text())["engine"]["ais_perf_model"] == {
        "estimation_mode": "auto",
        "fallback_policy": "allow",
    }
    return path


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_from_yaml_rejects_a_key_written_twice_inside_the_key(config_dir, command):
    path = _write_repeated_key(config_dir / "twice.yaml", command)

    with _deprecations() as found, pytest.raises(AisPerfModelError) as failure:
        LOADERS[command].from_yaml(path)

    assert str(failure.value) == f"engine.ais_perf_model in {path}: {REPEATED_KEY}"
    assert found == []


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_key_written_twice_inside_the_key_is_reported_by_the_child(config_dir, parent, capsys, command):
    path = _write_repeated_key(config_dir / "twice.yaml", command)

    # The parent treats the file like one it cannot read: it prepares nothing and lets the child report.
    assert parent.run(command, path) == 0
    assert len(parent.child_commands) == 1
    assert not parent.output.exists()
    error = _child_error(capsys, path, command=command)
    assert f"error: {path}: engine.ais_perf_model in {path}: {REPEATED_KEY}" in error


def test_command_line_reports_a_key_written_twice_without_a_traceback(config_dir, tmp_path):
    path = _write_repeated_key(config_dir / "twice.yaml")
    output = tmp_path / "output"

    finished = subprocess.run(
        [sys.executable, "-m", "aisim", *_argv("predict", path, output, ())],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert finished.returncode == 2
    assert f"engine.ais_perf_model in {path}: {REPEATED_KEY}" in finished.stderr
    assert "Traceback" not in finished.stderr
    assert not output.exists()


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_key_repeated_outside_the_key_keeps_its_last_value(config_dir, parent, command):
    # The engine section already gives context_length. It is given again, next to a key that repeats nothing.
    lines = ("ais_perf_model:", "  estimation_mode: op_level", "context_length: 512")
    path = _write_text(config_dir / "outside.yaml", command, *lines)
    assert path.read_text().count("  context_length: ") == 2

    with _deprecations() as found:
        loaded = LOADERS[command].from_yaml(path)
        assert parent.run(command, path) == 0

    assert found == []
    assert (loaded.engine.estimation_mode, loaded.engine.context_length) == ("op_level", 512)
    assert parent.prepared_output()


# --- Configs that AISim writes itself ---------------------------------------------------


def test_saved_dump_of_a_config_with_the_new_key_warns_when_read_back(config_dir):
    # A dump holds the old fields the entry was lowered to, so reading it back counts as an old file.
    path = _write(config_dir / "new.yaml", {"ais_perf_model": ENTRY})
    saved = config_dir / "saved.yaml"
    saved.write_text(yaml.safe_dump(_dump(CorePredictionConfig.from_yaml(path))))

    with _deprecations() as found:
        CorePredictionConfig.from_yaml(saved)

    (warning,) = found
    assert f"{OLD_PATHS}, engine.systems_paths" in str(warning.message)


def _child_in_process(command, *, policy, events_output):
    """Stand in for the supervised process: run the child CLI here and report its exit code."""
    del policy, events_output
    return {"status": "completed", "exit_code": cli.main(command[command.index("cli") + 1 :])}


def test_onboarding_does_not_warn_about_the_replay_configs_it_writes(request, monkeypatch, tmp_path):
    *_, validation_output = request.getfixturevalue("_replay_validation_case")
    predictions: list[str] = []

    def parent_cli(argv):
        predictions.append(argv[argv.index("--config") + 1])
        return supervised_main(argv)

    # Onboarding predicts in process through the parent CLI, which is where the warning is raised.
    monkeypatch.setattr(supervision, "main", parent_cli)
    monkeypatch.setattr(supervision, "run_process", _child_in_process)

    with _deprecations() as from_onboarding:
        # Replay validation, then serving preparation: each predicts from a config that onboarding wrote.
        request.getfixturevalue("serving_case")

    replay_config = validation_output / "predict.yaml"
    assert len(predictions) == 2
    assert predictions[0] == str(replay_config)
    assert from_onboarding == []

    # The same files still warn when a person gives one to the CLI or to from_yaml. The warning does not
    # send that person to migrate: onboarding compares each file with its record and rejects a converted one.
    matched_config = Path(predictions[1])
    assert matched_config.parent.name == "matched"
    with _deprecations() as from_person:
        assert parent_cli(_argv("predict", replay_config, tmp_path / "by-hand", ())) == 0
        CorePredictionConfig.from_yaml(replay_config)
        CorePredictionConfig.from_yaml(matched_config)
    assert len(from_person) == 3
    for warning in from_person:
        assert "engine.workers.aggregated.timing.estimation_mode" in str(warning.message)
        assert "perf-model migrate" not in str(warning.message)
        assert str(warning.message).endswith(
            "use engine.ais_perf_model "
            "('aisim onboard' wrote this file and rejects it once it is changed, so leave it as it is)"
        )

    # A copy in a folder of the person's own is a file onboarding does not look at, so it can be converted.
    own_copy = tmp_path / "own" / "predict.yaml"
    own_copy.parent.mkdir()
    shutil.copy(replay_config, own_copy)
    with _deprecations() as from_copy:
        CorePredictionConfig.from_yaml(own_copy)
    (warning,) = from_copy
    assert str(warning.message).endswith(f"use engine.ais_perf_model ({_migrate_hint(own_copy)})")
