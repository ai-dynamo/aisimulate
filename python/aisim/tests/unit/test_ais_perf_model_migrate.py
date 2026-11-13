# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``migrate_mapping`` moves the old estimator fields of a config into ``ais_perf_model``.

The converted config must mean what the original means. So most tests here
convert a config written with the old fields and compare the two: the dump,
the settings every worker type gets, the perf model configs the compiler
builds, or the estimator requests a search builds.
"""

from __future__ import annotations

import copy
import datetime
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

import aisim.main as cli
from aisim import supervision
from aisim.compiler import prediction_to_replay_spec
from aisim.config import CorePredictionConfig, CoreRecommendationConfig
from aisim.config import ais_perf_model_migrate as migrate_module
from aisim.config.ais_perf_model import ESTIMATOR_KEYS, anchor_paths
from aisim.config.ais_perf_model_migrate import CannotConvert, MigrationCheckError, migrate_mapping
from aisim.config.cli import config_folder
from aisim.support.schema import SupportRequest
from aisim.sweeper.config import SearchSpace, SmartSearchConfig
from aisim.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver

from .test_onboard_model_config import _args as _onboard_args
from .test_onboard_model_config import _files as _onboard_files

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
WORKLOAD = {"isl": 128, "osl": 16, "request_rate": 2, "num_request_ratio": 3}
# "both" is a recommend config that leaves the mode open, so it runs all three worker types.
ROLES = {
    "aggregated": ("aggregated",),
    "disaggregated": ("prefill", "decode"),
    "both": ("aggregated", "prefill", "decode"),
}
# The search space calls the aggregated worker "agg" in its field names.
SEARCH_ROLES = {"agg": ("agg",), "disagg": ("prefill", "decode")}
# Every value differs from the default of its field, so every key counts as written at either level.
EST = {
    "estimation_mode": "op_level",
    "fallback_policy": "allow",
    "database_mode": "HYBRID",
    "transfer_policy": "xshape",
    "estimator_config": {"correction": {"mode": "fixed", "min_observations": 3}},
}
# Settings of one worker that differ from EST in every key they share.
OTHER = {
    "estimation_mode": "auto",
    "fallback_policy": "deny",
    "database_mode": "SILICON",
    "estimator_config": {"correction": {"mode": "fixed", "min_observations": 5}},
}
MODELS = {"predict": CorePredictionConfig, "recommend": CoreRecommendationConfig}


def _engine(mode: str, timing: dict[str, dict[str, Any]] | None = None, **fields: Any) -> dict[str, Any]:
    """A raw engine mapping; ``timing`` gives the old role-level fields of a worker type."""
    workers = {role: {"timing": copy.deepcopy(timing[role])} if role in (timing or {}) else {} for role in ROLES[mode]}
    engine: dict[str, Any] = {
        "model": MODEL,
        "hardware": SYSTEM,
        "backend": "vllm",
        "context_length": 256,
        "workers": workers,
    }
    if mode != "both":
        engine["mode"] = mode
    engine.update(copy.deepcopy(fields))
    return engine


def _config(kind: str, engine: dict[str, Any]) -> dict[str, Any]:
    """A whole predict or recommend config as its file states it."""
    return {"engine": engine, "traffic": TRAFFIC, **(SEARCH if kind == "recommend" else {})}


def _space(mode: str, **fields: Any) -> dict[str, Any]:
    """A whole search config for one model on one system."""
    space = {"model_name": MODEL, "hardware_sku": SYSTEM, "backend": ["vllm"], "deployment_mode": [mode]}
    space.update(copy.deepcopy(fields))
    return {"search_space": space, "workload": WORKLOAD}


def _migrate(
    raw: dict[str, Any], kind: str = "predict", *, config_dir: Any = os.curdir, cwd: Any = os.curdir, **options: Any
):
    return migrate_mapping(raw, kind, config_dir=config_dir, cwd=cwd, **options)


def _validated(kind: str, raw: dict[str, Any], folder: Any = os.curdir) -> Any:
    """Read a config the way the command line reads its file: the key starts a path at the file's folder."""
    return MODELS[kind].model_validate(anchor_paths(raw, os.fspath(folder), kind))


def _command_line(*argv: object) -> int:
    """Run ``aisim perf-model`` as the console script does and return its exit code."""
    try:
        return supervision.main(["perf-model", *(str(item) for item in argv)])
    except SystemExit as stop:
        # argparse reports a usage error by exiting.
        return stop.code


def _write(path: Path, raw: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _rewrite_note(old: str, new: str, folder: Any) -> str:
    """The note for a relative path that was rewritten for a config file in ``folder``."""
    return (
        f"path {old!r} (relative to the working directory) rewritten as {new!r} "
        f"(relative to the folder of the converted config, {os.path.realpath(folder)})"
    )


def _absolute_note(old: str, new: str, *folders: Any) -> str:
    """The note for a relative path that was written in full for a config file with a name in each of ``folders``."""
    *first, last = (os.path.realpath(folder) for folder in folders)
    return (
        f"path {old!r} (relative to the working directory) rewritten as {new!r} "
        f"(an absolute path: the converted config has names in {', '.join(first)} and {last}, "
        "and a relative path would start at a different folder for each name)"
    )


def _dump(config: Any) -> str:
    """The config as a dump writes it. It is compared as text, so the order of keys counts too."""
    return json.dumps(config.model_dump(mode="json", exclude_none=True), indent=1)


def _effective(config: Any, role: str) -> dict[str, Any]:
    """The estimator settings one worker type gets: its own timing value when set, else the engine value."""
    engine = config.engine
    timing = getattr(engine.workers, role).timing
    values = {
        name: getattr(timing, name) or getattr(engine, name)
        for name in ("estimation_mode", "fallback_policy", "systems_paths", "database_mode")
    }
    for name in ("estimator_config", "transfer_policy"):
        own = getattr(timing, name)
        values[name] = own if own is not None else getattr(engine, name)
    if timing.fpm_parquet_path is not None:
        # The legacy field names the file that estimator_config.fpm_interpolation.fpm_parquet_path names.
        interpolation = {**values["estimator_config"].get("fpm_interpolation", {})}
        interpolation["fpm_parquet_path"] = timing.fpm_parquet_path
        values["estimator_config"] = {**values["estimator_config"], "fpm_interpolation": interpolation}
    return values


def _perf_model_configs(config: CorePredictionConfig) -> dict[str, dict[str, Any]]:
    """The perf model config the compiler builds for each worker type."""
    deployment = prediction_to_replay_spec(config).backend_deployment
    names = {"aggregated": "agg_engine_args", "prefill": "prefill_engine_args", "decode": "decode_engine_args"}
    return {role: args["timing_model"]["config"] for role, name in names.items() if (args := getattr(deployment, name))}


def _requests(raw: dict[str, Any], mode: str) -> dict[str, dict[str, Any]]:
    """The estimator request the search builds for a candidate, for each role the mode runs."""
    space = SearchSpace.model_validate(raw["search_space"])
    resolver = ForwardPassEstimatorResolver(space)
    requests = {}
    for role in SEARCH_ROLES[mode]:
        prefix = "" if role == "agg" else f"{role}_"
        parallelism = {"tp": 2, "pp": 1, "attention_dp": 1, "moe_tp": 1, "moe_ep": 1}
        candidate = {
            "backend": "vllm",
            "hardware_sku": SYSTEM,
            f"{role}_block_size": 64,
            # The search copies the legacy file field of the space into every candidate.
            f"{role}_fpm_parquet_path": getattr(space, f"{role}_fpm_parquet_path"),
            **{f"{prefix}{name}": value for name, value in parallelism.items()},
        }
        requests[role] = resolver._request(candidate, role).to_dict()
    return requests


def _assert_same_meaning(kind: str, mode: str, raw: dict[str, Any], converted: dict[str, Any]) -> None:
    """Every worker type gets the same settings, and a second conversion finds nothing left to move."""
    before, after = _validated(kind, raw), _validated(kind, converted)
    for role in ROLES[mode]:
        assert _effective(after, role) == _effective(before, role)
    if kind == "predict":
        assert _perf_model_configs(after) == _perf_model_configs(before)
    assert _migrate(converted, kind) == (converted, [], [])


# --- Old fields at the engine level -----------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "mode"), [("predict", "aggregated"), ("predict", "disaggregated"), ("recommend", "both")]
)
def test_engine_level_fields_become_one_config_without_worker_type(kind, mode):
    raw = _config(kind, _engine(mode, **EST))
    original = copy.deepcopy(raw)

    converted, moved, notes = _migrate(raw, kind)

    assert converted["engine"]["ais_perf_model"] == EST
    assert sorted(moved) == sorted(f"engine.{name}" for name in EST)
    assert notes == []
    assert not set(EST) & set(converted["engine"])
    assert _dump(_validated(kind, converted)) == _dump(_validated(kind, raw))
    assert _migrate(converted, kind) == (converted, [], [])
    assert raw == original


def test_a_field_that_states_its_default_is_not_an_old_field_and_stays_in_the_file():
    raw = _config("predict", _engine("aggregated", estimation_mode="op_level", fallback_policy="deny"))

    converted, moved, _ = _migrate(raw)

    assert moved == ["engine.estimation_mode"]
    assert converted["engine"]["ais_perf_model"] == {"estimation_mode": "op_level"}
    assert converted["engine"]["fallback_policy"] == "deny"
    assert _dump(_validated("predict", converted)) == _dump(_validated("predict", raw))


def test_config_without_old_fields_comes_back_as_a_copy_with_nothing_moved():
    raw = _config("predict", _engine("aggregated"))

    converted, moved, notes = _migrate(raw)

    assert (converted, moved, notes) == (raw, [], [])
    assert converted is not raw and converted["engine"] is not raw["engine"]


# --- Old fields of a worker -------------------------------------------------------------------


def test_role_level_fields_that_differ_become_one_entry_per_worker_type():
    raw = _config("predict", _engine("disaggregated", timing={"prefill": EST, "decode": OTHER}))

    converted, moved, notes = _migrate(raw)

    assert converted["engine"]["ais_perf_model"] == [
        {"worker_type": "prefill", **EST},
        {"worker_type": "decode", **OTHER},
    ]
    # A timing mapping that the move empties is removed; the worker mapping stays.
    assert converted["engine"]["workers"] == {"prefill": {}, "decode": {}}
    assert sorted(moved) == sorted(
        f"engine.workers.{role}.timing.{name}"
        for role, fields in (("prefill", EST), ("decode", OTHER))
        for name in fields
    )
    assert notes == []
    assert _dump(_validated("predict", converted)) == _dump(_validated("predict", raw))
    _assert_same_meaning("predict", "disaggregated", raw, converted)


@pytest.mark.parametrize(
    ("mode", "engine_fields", "timing", "block"),
    [
        pytest.param(
            "aggregated",
            {},
            {"aggregated": EST},
            [{"worker_type": "aggregated", **EST}],
            id="one-worker-type",
        ),
        pytest.param(
            "disaggregated",
            {},
            {"prefill": EST, "decode": EST},
            [{"worker_type": "prefill", **EST}, {"worker_type": "decode", **EST}],
            id="workers-that-agree",
        ),
        pytest.param(
            "disaggregated",
            {},
            {"prefill": {"database_mode": "HYBRID"}},
            [{"worker_type": "prefill", "database_mode": "HYBRID"}, {"worker_type": "decode"}],
            id="a-worker-without-old-fields",
        ),
        pytest.param(
            "disaggregated",
            EST,
            {"decode": {"estimation_mode": "auto", "database_mode": "SILICON"}},
            [
                {"worker_type": "prefill", **EST},
                {"worker_type": "decode", **EST, "estimation_mode": "auto", "database_mode": "SILICON"},
            ],
            id="a-worker-value-replaces-the-engine-value",
        ),
        pytest.param(
            "disaggregated",
            EST,
            {"decode": {"estimator_config": {}}},
            [
                {"worker_type": "prefill", **EST},
                {"worker_type": "decode", **{name: value for name, value in EST.items() if name != "estimator_config"}},
            ],
            id="an-empty-estimator-config-blanks-the-engine-value",
        ),
        pytest.param(
            "disaggregated",
            EST,
            {"decode": {"transfer_policy": []}},
            [{"worker_type": "prefill", **EST}, {"worker_type": "decode", **EST, "transfer_policy": []}],
            id="an-empty-transfer-policy-is-a-value",
        ),
    ],
)
def test_role_level_fields_become_a_list_that_gives_every_worker_type_the_same_settings(
    mode, engine_fields, timing, block
):
    raw = _config("predict", _engine(mode, timing=timing, **engine_fields))

    converted, moved, _ = _migrate(raw)

    assert converted["engine"]["ais_perf_model"] == block
    assert all("timing" not in worker for worker in converted["engine"]["workers"].values())
    assert len(moved) == len(engine_fields) + sum(len(fields) for fields in timing.values())
    _assert_same_meaning("predict", mode, raw, converted)


def test_keys_of_the_new_block_are_written_in_one_order_whatever_order_the_old_fields_have():
    backwards = dict(reversed(EST.items()))
    shared = _config("predict", _engine("aggregated", **backwards))
    per_worker = _config("predict", _engine("disaggregated", timing={"prefill": backwards, "decode": backwards}))
    # The decode worker adds a setting that the key order puts before the one the whole config states.
    added_by_a_worker = _config(
        "predict",
        _engine("disaggregated", timing={"decode": {"estimation_mode": "op_level"}}, fallback_policy="allow"),
    )

    block = _migrate(shared)[0]["engine"]["ais_perf_model"]
    entries = _migrate(per_worker)[0]["engine"]["ais_perf_model"]
    prefill, decode = _migrate(added_by_a_worker)[0]["engine"]["ais_perf_model"]

    in_order = [name for name in ESTIMATOR_KEYS if name in EST]
    assert list(EST) != in_order
    assert list(block) == in_order
    assert [list(entry) for entry in entries] == [["worker_type", *in_order]] * 2
    assert list(prefill) == ["worker_type", "fallback_policy"]
    assert list(decode) == ["worker_type", "estimation_mode", "fallback_policy"]


# --- Legacy spellings -------------------------------------------------------------------------


def test_legacy_systems_path_becomes_a_one_item_systems_paths(tmp_path):
    raw = _config("predict", _engine("aggregated", systems_path=str(tmp_path)))

    converted, moved, notes = _migrate(raw)

    assert converted["engine"]["ais_perf_model"] == {"systems_paths": [str(tmp_path)]}
    assert moved == ["engine.systems_path"]
    assert notes == [f'engine.systems_path written as systems_paths: ["{tmp_path}"]']
    assert "systems_path" not in converted["engine"]
    _assert_same_meaning("predict", "aggregated", raw, converted)


def test_legacy_systems_path_next_to_other_systems_paths_cannot_be_converted(tmp_path):
    engine = _engine("aggregated", systems_path=str(tmp_path / "a"), systems_paths=[str(tmp_path / "b")])

    with pytest.raises(CannotConvert) as failure:
        _migrate(_config("predict", engine))

    assert failure.value.problems == (
        "cannot convert: engine.systems_path: engine.systems_paths names other directories; keep one of the two fields",
    )


def test_legacy_systems_path_that_is_relative_names_the_same_directory_from_the_config_file(tmp_path, monkeypatch):
    cwd, folder = tmp_path / "run", tmp_path / "configs"
    (cwd / "data").mkdir(parents=True)
    folder.mkdir()
    raw = _config("predict", _engine("aggregated", systems_path="data"))

    converted, moved, notes = _migrate(raw, config_dir=folder, cwd=cwd)

    rewritten = os.path.join(os.pardir, "run", "data")
    assert converted["engine"]["ais_perf_model"] == {"systems_paths": [rewritten]}
    assert moved == ["engine.systems_path"]
    assert notes[0] == 'engine.systems_path written as systems_paths: ["data"]'
    # The old field starts its path at the working directory; the converted file names that directory.
    monkeypatch.chdir(cwd)
    before, after = _validated("predict", raw), _validated("predict", converted, folder)
    assert before.engine.systems_paths == after.engine.systems_paths == [str(cwd / "data")]
    assert _perf_model_configs(after) == _perf_model_configs(before)


IN_CWD, IN_PARENT = os.path.join(os.curdir, "default"), os.path.join(os.pardir, "default")


@pytest.mark.parametrize(
    ("config_in", "written"),
    [
        pytest.param(os.curdir, IN_CWD, id="config-in-the-working-directory"),
        pytest.param("configs", IN_PARENT, id="config-in-another-folder"),
    ],
)
def test_legacy_systems_path_named_default_stays_a_directory(config_in, written, tmp_path, monkeypatch):
    """The old field names the directory ``default``; in ``systems_paths`` the bare word is the packaged data."""
    folder = tmp_path / config_in
    folder.mkdir(exist_ok=True)
    (tmp_path / "default").mkdir()
    raw = _config("predict", _engine("aggregated", systems_path="default"))

    converted, _, notes = _migrate(raw, config_dir=folder, cwd=tmp_path)

    assert converted["engine"]["ais_perf_model"] == {"systems_paths": [written]}
    assert notes[0] == f'engine.systems_path written as systems_paths: ["{IN_CWD}"]'
    monkeypatch.chdir(tmp_path)
    before, after = _validated("predict", raw), _validated("predict", converted, folder)
    assert before.engine.systems_paths == after.engine.systems_paths == [str(tmp_path / "default")]
    # Written as the bare word, the same config would read the packaged data.
    bare = _config("predict", _engine("aggregated", ais_perf_model={"systems_paths": ["default"]}))
    assert _validated("predict", bare, folder).engine.systems_paths == ["default"]


@pytest.mark.parametrize(
    "field", [{"systems_path": "x/default"}, {"systems_paths": ["x/default"]}], ids=["legacy", "list"]
)
def test_relative_path_that_ends_at_a_folder_named_default_next_to_the_config_is_not_written_as_the_bare_word(
    field, tmp_path
):
    folder = tmp_path / "x"
    (folder / "default").mkdir(parents=True)
    raw = _config("predict", _engine("aggregated", **field))

    converted, _, _ = _migrate(raw, config_dir=folder, cwd=tmp_path)

    assert converted["engine"]["ais_perf_model"] == {"systems_paths": [IN_CWD]}
    assert _validated("predict", converted, folder).engine.systems_paths == [str(folder / "default")]


@pytest.mark.parametrize("spelling", ["relative", "absolute"])
def test_legacy_systems_path_that_repeats_the_one_directory_of_systems_paths_is_dropped(spelling, tmp_path):
    cwd, folder = tmp_path / "run", tmp_path / "configs"
    (cwd / "data").mkdir(parents=True)
    folder.mkdir()
    directory = "data" if spelling == "relative" else str(cwd / "data")
    raw = _config("predict", _engine("aggregated", systems_path=directory, systems_paths=[directory]))

    converted, moved, notes = _migrate(raw, config_dir=folder, cwd=cwd)

    kept = os.path.join(os.pardir, "run", "data") if spelling == "relative" else directory
    assert converted["engine"]["ais_perf_model"] == {"systems_paths": [kept]}
    assert sorted(moved) == ["engine.systems_path", "engine.systems_paths"]
    assert notes[0] == "engine.systems_path dropped: engine.systems_paths names the same directory"
    assert not {"systems_path", "systems_paths"} & set(converted["engine"])
    assert _validated("predict", converted, folder).engine.systems_paths == [str(cwd / "data")]


@pytest.mark.parametrize(
    ("timing", "entry", "note"),
    [
        pytest.param(
            {"forward_model": "op_level"},
            {"estimation_mode": "op_level", "fallback_policy": "deny"},
            "forward_model: op_level written as estimation_mode: op_level and fallback_policy: deny",
            id="op_level",
        ),
        pytest.param(
            {"forward_model": "fpm"},
            {"estimation_mode": "fpm_interpolation", "fallback_policy": "deny"},
            "forward_model: fpm written as estimation_mode: fpm_interpolation and fallback_policy: deny",
            id="fpm",
        ),
        pytest.param(
            {"forward_model": "fpm", "fallback_policy": "allow"},
            {"estimation_mode": "fpm_interpolation", "fallback_policy": "allow"},
            "forward_model: fpm written as estimation_mode: fpm_interpolation",
            id="a-written-fallback-policy-is-kept",
        ),
        pytest.param(
            {"forward_model": "fpm", "estimation_mode": "op_level"},
            {"estimation_mode": "op_level"},
            "forward_model dropped: estimation_mode is written next to it, and that is the field the config uses today",
            id="a-written-estimation-mode-wins",
        ),
    ],
)
def test_legacy_forward_model_of_a_worker_becomes_the_settings_it_selects_today(timing, entry, note):
    raw = _config("predict", _engine("aggregated", timing={"aggregated": timing}, estimation_mode="auto"))

    converted, moved, notes = _migrate(raw)

    assert converted["engine"]["ais_perf_model"] == [{"worker_type": "aggregated", **entry}]
    assert "engine.workers.aggregated.timing.forward_model" in moved
    assert notes == [f"engine.workers.aggregated.timing.{note}"]
    _assert_same_meaning("predict", "aggregated", raw, converted)


def test_legacy_fpm_parquet_path_of_a_worker_is_written_into_its_estimator_config():
    timing = {"forward_model": "fpm", "fpm_parquet_path": "/data/fpm.parquet"}
    raw = _config(
        "predict", _engine("aggregated", timing={"aggregated": timing}, estimator_config=EST["estimator_config"])
    )

    converted, _, notes = _migrate(raw)

    assert converted["engine"]["ais_perf_model"] == [
        {
            "worker_type": "aggregated",
            "estimation_mode": "fpm_interpolation",
            "fallback_policy": "deny",
            "estimator_config": {
                **EST["estimator_config"],
                "fpm_interpolation": {"fpm_parquet_path": "/data/fpm.parquet"},
            },
        }
    ]
    assert notes[-1] == (
        "engine.workers.aggregated.timing.fpm_parquet_path written as "
        "estimator_config.fpm_interpolation.fpm_parquet_path"
    )
    _assert_same_meaning("predict", "aggregated", raw, converted)


def test_legacy_fpm_parquet_path_next_to_another_file_cannot_be_converted():
    timing = {
        "fpm_parquet_path": "/data/fpm.parquet",
        "estimator_config": {"fpm_interpolation": {"fpm_parquet_path": "/data/other.parquet"}},
    }

    with pytest.raises(CannotConvert) as failure:
        _migrate(_config("predict", _engine("aggregated", timing={"aggregated": timing})))

    assert failure.value.problems == (
        "cannot convert: engine.workers.aggregated.timing.fpm_parquet_path: "
        'estimator_config.fpm_interpolation.fpm_parquet_path already names another file, "/data/other.parquet"',
    )


@pytest.mark.parametrize(
    "estimator_config",
    ["text", {"fpm_interpolation": "text"}],
    ids=["not-a-mapping", "fpm_interpolation-not-a-mapping"],
)
def test_legacy_fpm_parquet_path_next_to_an_estimator_config_that_cannot_hold_it_cannot_be_converted(estimator_config):
    timing = {"fpm_parquet_path": "/data/fpm.parquet", "estimator_config": estimator_config}

    with pytest.raises(CannotConvert) as failure:
        _migrate(_config("predict", _engine("aggregated", timing={"aggregated": timing})))

    assert failure.value.problems == (
        "cannot convert: engine.workers.aggregated.timing.fpm_parquet_path: the estimator_config of this worker "
        "has no mapping to hold estimator_config.fpm_interpolation.fpm_parquet_path",
    )


# --- Relative paths ---------------------------------------------------------------------------


def test_relative_path_is_rewritten_to_name_the_same_folder_from_the_config_file(tmp_path):
    cwd, folder = tmp_path / "run", tmp_path / "configs"
    (cwd / "data" / "systems").mkdir(parents=True)
    folder.mkdir()
    paths = ["data/systems", "default", str(tmp_path / "absolute"), "~/systems"]
    raw = _config("predict", _engine("aggregated", systems_paths=paths))

    converted, _, notes = _migrate(raw, config_dir=folder, cwd=cwd)

    rewritten = os.path.join(os.pardir, "run", "data", "systems")
    # An absolute path, a home path and the word for the packaged data mean the same from any folder.
    assert converted["engine"]["ais_perf_model"] == {"systems_paths": [rewritten, *paths[1:]]}
    # The note names the folder the new path starts at, so the reader need not guess it.
    assert notes == [_rewrite_note("data/systems", rewritten, folder)]
    read_back = _validated("predict", converted, folder).engine.systems_paths
    assert read_back[0] == str(cwd / "data" / "systems")


def test_relative_path_inside_estimator_config_is_rewritten_and_reported_once(tmp_path):
    cwd, folder = tmp_path / "run", tmp_path / "configs"
    cwd.mkdir()
    folder.mkdir()
    estimator_config = {"fpm_interpolation": {"fpm_parquet_path": "data/fpm.parquet"}}
    timing = {"decode": {"database_mode": "HYBRID"}}
    raw = _config("predict", _engine("disaggregated", timing=timing, estimator_config=estimator_config))

    converted, _, notes = _migrate(raw, config_dir=folder, cwd=cwd)

    rewritten = os.path.join(os.pardir, "run", "data", "fpm.parquet")
    entries = converted["engine"]["ais_perf_model"]
    # The engine value is copied into both entries, and the path is one path.
    assert [entry["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] for entry in entries] == [rewritten] * 2
    assert notes == [_rewrite_note("data/fpm.parquet", rewritten, folder)]
    read_back = _validated("predict", converted, folder).engine.estimator_config
    assert read_back["fpm_interpolation"]["fpm_parquet_path"] == str(cwd / "data" / "fpm.parquet")


def test_relative_path_is_kept_as_written_when_the_config_is_run_from_its_own_folder(tmp_path):
    raw = _config("predict", _engine("aggregated", systems_paths=["data/systems"]))

    converted, _, notes = _migrate(raw, config_dir=tmp_path, cwd=tmp_path)

    assert converted["engine"]["ais_perf_model"] == {"systems_paths": ["data/systems"]}
    assert notes == []


def _linked_config_folder(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A working directory with ``data``, a config folder, and that folder as reached through a link elsewhere."""
    cwd, real = tmp_path / "run", tmp_path / "real" / "configs"
    (cwd / "data").mkdir(parents=True)
    real.mkdir(parents=True)
    link = tmp_path / "elsewhere" / "deep" / "link"
    link.parent.mkdir(parents=True)
    link.symlink_to(tmp_path / "real", target_is_directory=True)
    return cwd, real, link / "configs"


# From the folder that really holds the config the data is two levels up; counted along the link it would be four.
FROM_REAL_FOLDER = os.path.join(os.pardir, os.pardir, "run", "data")


def test_config_reached_through_a_linked_folder_has_its_paths_rewritten_from_the_real_folder(tmp_path):
    cwd, _, linked = _linked_config_folder(tmp_path)
    raw = _config("predict", _engine("aggregated", systems_paths=["data"]))

    converted, _, _ = _migrate(raw, config_dir=linked, cwd=cwd)

    assert converted["engine"]["ais_perf_model"] == {"systems_paths": [FROM_REAL_FOLDER]}
    # The loaders start a path of the key at the real folder too, so the file reads the same directory.
    read_from = config_folder(linked / "config.yaml")
    assert _validated("predict", converted, read_from).engine.systems_paths == [str(cwd / "data")]


def test_rewrite_note_names_the_folder_that_really_holds_the_converted_config(tmp_path):
    cwd, real, linked = _linked_config_folder(tmp_path)
    raw = _config("predict", _engine("aggregated", systems_paths=["data"]))

    _, _, notes = _migrate(raw, config_dir=linked, cwd=cwd)

    assert notes == [_rewrite_note("data", FROM_REAL_FOLDER, real)]


# --- A config file with a name in several folders ---------------------------------------------


def _folders_of_a_linked_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A working directory with ``data``, the folder of a config file and the folder of a link to that file."""
    cwd, folder, linked_from = tmp_path / "run", tmp_path / "configs", tmp_path / "links"
    (cwd / "data" / "systems").mkdir(parents=True)
    folder.mkdir()
    linked_from.mkdir()
    return cwd, folder, linked_from


def test_relative_path_is_written_in_full_for_a_config_file_with_a_name_in_another_folder(tmp_path):
    cwd, folder, linked_from = _folders_of_a_linked_config(tmp_path)
    paths = ["data/systems", "default", str(tmp_path / "absolute"), "~/systems"]
    raw = _config("predict", _engine("aggregated", systems_paths=paths))

    converted, _, notes = _migrate(raw, config_dir=linked_from, cwd=cwd, other_config_dirs=[folder])

    absolute = str(cwd / "data" / "systems")
    # An absolute path, a home path and the word for the packaged data mean the same from any folder.
    assert converted["engine"]["ais_perf_model"] == {"systems_paths": [absolute, *paths[1:]]}
    assert notes == [_absolute_note("data/systems", absolute, linked_from, folder)]
    # Read by either name, the config names the folder the old field named.
    for read_from in (linked_from, folder):
        assert _validated("predict", converted, read_from).engine.systems_paths[0] == absolute


def test_relative_path_that_is_right_from_one_folder_of_the_config_is_written_in_full_too(tmp_path):
    cwd, folder, _ = _folders_of_a_linked_config(tmp_path)
    raw = _config("predict", _engine("aggregated", systems_paths=["data/systems"]))

    # One name of the config is in the folder the config is run from, where the path as written is right.
    converted, _, notes = _migrate(raw, config_dir=cwd, cwd=cwd, other_config_dirs=[folder])

    absolute = str(cwd / "data" / "systems")
    assert converted["engine"]["ais_perf_model"] == {"systems_paths": [absolute]}
    assert notes == [_absolute_note("data/systems", absolute, cwd, folder)]


def test_path_that_goes_up_from_a_linked_folder_is_written_in_full_as_the_system_reaches_it(tmp_path, monkeypatch):
    cwd, folder, linked_from = _folders_of_a_linked_config(tmp_path)
    (cwd / "elsewhere" / "deep").mkdir(parents=True)
    (cwd / "elsewhere" / "data").mkdir()
    (cwd / "link").symlink_to(cwd / "elsewhere" / "deep", target_is_directory=True)
    # From the linked folder, ".." is the parent of the folder the link names, so this is elsewhere/data.
    raw = _config("predict", _engine("aggregated", systems_paths=["link/../data"]))

    converted, _, _ = _migrate(raw, config_dir=folder, cwd=cwd, other_config_dirs=[linked_from])

    absolute = str(cwd / "elsewhere" / "data")
    assert converted["engine"]["ais_perf_model"] == {"systems_paths": [absolute]}
    monkeypatch.chdir(cwd)
    assert _validated("predict", raw).engine.systems_paths == [absolute]


def test_note_for_a_path_written_in_full_names_every_folder_the_config_has_a_name_in(tmp_path):
    cwd, folder, linked_from = _folders_of_a_linked_config(tmp_path)
    raw = _config("predict", _engine("aggregated", systems_paths=["data/systems"]))

    _, _, notes = _migrate(raw, config_dir=linked_from, cwd=cwd, other_config_dirs=[folder, cwd, folder])

    assert notes == [_absolute_note("data/systems", str(cwd / "data" / "systems"), linked_from, folder, cwd)]


@pytest.mark.parametrize("spelling", ["the-folder-itself", "a-link-to-the-folder"])
def test_other_name_in_the_folder_of_the_config_keeps_the_relative_path(spelling, tmp_path):
    cwd, folder, linked_from = _folders_of_a_linked_config(tmp_path)
    other = folder
    if spelling == "a-link-to-the-folder":
        other = linked_from / "configs"
        other.symlink_to(folder, target_is_directory=True)
    raw = _config("predict", _engine("aggregated", systems_paths=["data/systems"]))

    converted, _, notes = _migrate(raw, config_dir=folder, cwd=cwd, other_config_dirs=[other])

    rewritten = os.path.join(os.pardir, "run", "data", "systems")
    assert converted["engine"]["ais_perf_model"] == {"systems_paths": [rewritten]}
    assert notes == [_rewrite_note("data/systems", rewritten, folder)]


def test_path_inside_estimator_config_is_written_in_full_and_reported_once(tmp_path):
    cwd, folder, linked_from = _folders_of_a_linked_config(tmp_path)
    estimator_config = {"fpm_interpolation": {"fpm_parquet_path": "data/fpm.parquet"}}
    timing = {"decode": {"database_mode": "HYBRID"}}
    raw = _config("predict", _engine("disaggregated", timing=timing, estimator_config=estimator_config))

    converted, _, notes = _migrate(raw, config_dir=folder, cwd=cwd, other_config_dirs=[linked_from])

    absolute = str(cwd / "data" / "fpm.parquet")
    entries = converted["engine"]["ais_perf_model"]
    assert [entry["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] for entry in entries] == [absolute] * 2
    assert notes == [_absolute_note("data/fpm.parquet", absolute, folder, linked_from)]
    for read_from in (folder, linked_from):
        read_back = _validated("predict", converted, read_from).engine.estimator_config
        assert read_back["fpm_interpolation"]["fpm_parquet_path"] == absolute


def test_legacy_systems_path_named_default_is_written_in_full_as_the_directory(tmp_path, monkeypatch):
    cwd, folder, _ = _folders_of_a_linked_config(tmp_path)
    (cwd / "default").mkdir()
    raw = _config("predict", _engine("aggregated", systems_path="default"))

    converted, _, _ = _migrate(raw, config_dir=cwd, cwd=cwd, other_config_dirs=[folder])

    assert converted["engine"]["ais_perf_model"] == {"systems_paths": [str(cwd / "default")]}
    monkeypatch.chdir(cwd)
    before = _validated("predict", raw).engine.systems_paths
    for read_from in (cwd, folder):
        assert _validated("predict", converted, read_from).engine.systems_paths == before


def test_search_relative_path_is_written_in_full_for_a_config_file_with_a_name_in_another_folder(tmp_path):
    cwd, folder, linked_from = _folders_of_a_linked_config(tmp_path)
    raw = _space("agg", systems_paths=["data/systems"])

    converted, _, notes = _migrate(raw, "search", config_dir=linked_from, cwd=cwd, other_config_dirs=[folder])

    absolute = str(cwd / "data" / "systems")
    assert converted["search_space"]["ais_perf_model"] == {"systems_paths": [absolute]}
    assert notes == [_absolute_note("data/systems", absolute, linked_from, folder)]
    # Read by either name of its file, the search config names the folder the old field named.
    file = _write(folder / "search.yaml", converted)
    link = linked_from / "search.yaml"
    link.symlink_to(file)
    for name in (link, file):
        assert SmartSearchConfig.from_yaml(name).search_space.systems_paths == [absolute]


# --- Configs that cannot be converted ---------------------------------------------------------


def test_nothing_is_converted_when_one_field_cannot_be_converted(tmp_path):
    engine = _engine(
        "aggregated",
        timing={
            "aggregated": {
                "fpm_parquet_path": "/data/fpm.parquet",
                "estimator_config": {"fpm_interpolation": {"fpm_parquet_path": "/data/other.parquet"}},
            }
        },
        estimation_mode="op_level",
        systems_path=str(tmp_path / "a"),
        systems_paths=[str(tmp_path / "b")],
    )
    raw = _config("predict", engine)
    original = copy.deepcopy(raw)

    with pytest.raises(CannotConvert) as failure:
        _migrate(raw)

    # One line for each field that stops the conversion, and none for the field that could be moved.
    assert [problem.split(": ")[1] for problem in failure.value.problems] == [
        "engine.systems_path",
        "engine.workers.aggregated.timing.fpm_parquet_path",
    ]
    assert str(failure.value) == "\n".join(failure.value.problems)
    assert raw == original


def test_old_fields_are_not_merged_into_a_key_that_is_already_set():
    engine = _engine("aggregated", estimation_mode="op_level", database_mode="HYBRID")
    engine["ais_perf_model"] = {"fallback_policy": "allow"}

    with pytest.raises(CannotConvert) as failure:
        _migrate(_config("predict", engine))

    reason = "engine.ais_perf_model is already set, and old fields are not merged into it; remove the field or the key"
    assert failure.value.problems == (
        f"cannot convert: engine.estimation_mode: {reason}",
        f"cannot convert: engine.database_mode: {reason}",
    )


@pytest.mark.parametrize("old_fields", [{}, {"estimation_mode": "op_level"}], ids=["no-old-fields", "old-fields"])
def test_config_the_key_cannot_describe_is_reported_with_the_reason(old_fields):
    engine = _engine("aggregated", timing={"aggregated": {"type": "constant"}}, **old_fields)

    with pytest.raises(CannotConvert) as failure:
        _migrate(_config("predict", engine))

    assert failure.value.problems == (
        'cannot convert: engine.workers.aggregated.timing.type is "constant", '
        "and ais_perf_model covers only default timing",
    )


# A decode worker that runs attention and FFN on separate pools (AFD), as each kind of config writes it.
AFD_DECODE = {
    "predict": {"type": "afd", "afd": {"n_a_nodes": 1, "n_f_nodes": 1, "tp_a": 1, "a_batch_size": 8}},
    "recommend": {"type": "afd", "afd": {"a_batch_size": 8}},
}
OP_LEVEL = {"estimation_mode": "op_level"}


@pytest.mark.parametrize(
    ("fields", "timing"),
    [({}, {}), (OP_LEVEL, {}), ({}, {"decode": OP_LEVEL}), ({}, {"prefill": OP_LEVEL})],
    ids=["no old fields", "engine section", "AFD worker", "worker next to the AFD worker"],
)
@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_config_with_an_afd_worker_is_not_converted_wherever_its_old_fields_are(kind, fields, timing):
    engine = _engine("disaggregated", timing=timing, backend="sglang", **fields)
    engine["workers"]["decode"].update(copy.deepcopy(AFD_DECODE[kind]))
    raw = _config(kind, engine)
    original = copy.deepcopy(raw)
    # The config is a valid one; only the key cannot describe it.
    MODELS[kind].model_validate(copy.deepcopy(raw))

    with pytest.raises(CannotConvert) as failure:
        _migrate(raw, kind)

    assert failure.value.problems == (
        "cannot convert: engine.workers.decode.type is afd, and ais_perf_model does not cover AFD yet",
    )
    assert raw == original


AS_OF = {"op_level": {"as_of": datetime.date(2024, 1, 1)}}
NOT_A_CONFIG_VALUE = "has a value of type date; use a string, a number, true or false, a list or a mapping"


@pytest.mark.parametrize(
    ("kind", "raw", "place"),
    [
        pytest.param(
            "predict",
            _config("predict", _engine("aggregated", estimator_config=AS_OF)),
            "engine.ais_perf_model cannot hold a value of the old fields: estimator_config.op_level.as_of",
            id="engine-field",
        ),
        pytest.param(
            "predict",
            _config("predict", _engine("disaggregated", timing={"decode": {"estimator_config": AS_OF}})),
            "engine.ais_perf_model cannot hold a value of the old fields: entry 2: estimator_config.op_level.as_of",
            id="field-of-a-worker",
        ),
        pytest.param(
            "search",
            _space("agg", estimator_config=AS_OF),
            "search_space.ais_perf_model cannot hold a value of the old fields: estimator_config.op_level.as_of",
            id="search-field",
        ),
    ],
)
def test_old_field_value_the_key_cannot_hold_is_a_field_that_cannot_be_converted(kind, raw, place):
    """The old fields accept a YAML date; the key does not. That is the config's content, not a failed check."""
    original = copy.deepcopy(raw)

    with pytest.raises(CannotConvert) as failure:
        _migrate(raw, kind)

    assert failure.value.problems == (f"cannot convert: {place} {NOT_A_CONFIG_VALUE}",)
    assert raw == original


def test_command_line_reports_a_value_the_key_cannot_hold_as_a_problem_of_the_config(tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", _config("predict", _engine("aggregated", estimator_config=AS_OF)))
    output = tmp_path / "converted.yaml"

    assert _command_line("migrate", config, "--output", output) == 1

    captured = capsys.readouterr()
    assert "cannot convert: engine.ais_perf_model cannot hold a value of the old fields" in captured.out
    assert f"estimator_config.op_level.as_of {NOT_A_CONFIG_VALUE}" in captured.out
    assert "internal error" not in captured.err
    assert "please report" not in captured.err
    assert not output.exists()


def _snapshot_for_the_whole_config(runs: str) -> str:
    return (
        "cannot convert: engine.estimator_config.state_snapshot: the field names a state snapshot for the whole "
        "config, and engine.ais_perf_model takes a state snapshot for one worker type. "
        f"This config runs {runs}; write engine.ais_perf_model as a list with one entry per worker_type"
    )


@pytest.mark.parametrize("snapshot", ["snapshot.json", {"enabled": True}, None], ids=["path", "mapping", "null"])
@pytest.mark.parametrize(
    ("kind", "mode", "runs"),
    [("predict", "disaggregated", "prefill, decode"), ("recommend", "both", "aggregated, prefill, decode")],
    ids=["predict", "recommend"],
)
def test_state_snapshot_for_the_whole_config_is_not_converted_when_several_worker_types_run(kind, mode, runs, snapshot):
    raw = _config(kind, _engine(mode, estimator_config={"state_snapshot": snapshot}))
    original = copy.deepcopy(raw)
    # The config is a valid one; the key takes a state snapshot for one worker type only.
    MODELS[kind].model_validate(copy.deepcopy(raw))

    with pytest.raises(CannotConvert) as failure:
        _migrate(raw, kind)

    assert failure.value.problems == (_snapshot_for_the_whole_config(runs),)
    assert raw == original


def test_command_line_reports_a_state_snapshot_for_the_whole_config_as_a_problem_of_the_config(tmp_path, capsys):
    engine = _engine("disaggregated", estimator_config={"state_snapshot": "snapshot.json"})
    config = _write(tmp_path / "config.yaml", _config("predict", engine))
    before = config.read_bytes()
    output = tmp_path / "converted.yaml"

    assert _command_line("migrate", config, "--output", output) == 1

    captured = capsys.readouterr()
    assert _snapshot_for_the_whole_config("prefill, decode") in captured.out
    assert "Nothing was converted." in captured.out
    assert "internal error" not in captured.err
    assert "please report" not in captured.err
    assert list(tmp_path.iterdir()) == [config]
    assert config.read_bytes() == before


def test_state_snapshot_of_a_config_with_one_worker_type_is_converted():
    raw = _config("predict", _engine("aggregated", estimator_config={"state_snapshot": "snapshot.json"}))

    converted, _, _ = _migrate(raw)

    assert converted["engine"]["ais_perf_model"] == {"estimator_config": {"state_snapshot": "snapshot.json"}}
    assert "estimator_config" not in converted["engine"]


def test_state_snapshot_next_to_a_field_of_one_worker_becomes_an_entry_per_worker_type():
    """The old fields give each worker the snapshot; so does a list, where each entry names it."""
    engine = _engine(
        "disaggregated",
        timing={"prefill": {"database_mode": "SILICON"}},
        estimator_config={"state_snapshot": "snapshot.json"},
    )

    converted, _, _ = _migrate(_config("predict", engine))

    named = {"estimator_config": {"state_snapshot": "snapshot.json"}}
    assert converted["engine"]["ais_perf_model"] == [
        {"worker_type": "prefill", **named, "database_mode": "SILICON"},
        {"worker_type": "decode", **named},
    ]


def _engine_field_that_contains_itself() -> dict[str, Any]:
    raw = _config("predict", _engine("aggregated", estimator_config={"correction": {"mode": "fixed"}}))
    raw["engine"]["estimator_config"]["again"] = raw["engine"]["estimator_config"]
    return raw


def _list_that_contains_itself() -> dict[str, Any]:
    raw = _config("predict", _engine("aggregated", transfer_policy=["xshape"]))
    raw["engine"]["transfer_policy"].append(raw["engine"]["transfer_policy"])
    return raw


def _section_without_old_fields_that_contains_itself() -> dict[str, Any]:
    raw = _config("predict", _engine("aggregated"))
    raw["traffic"] = {**TRAFFIC}
    raw["traffic"]["again"] = raw["traffic"]
    return raw


@pytest.mark.parametrize(
    ("build", "place"),
    [
        (_engine_field_that_contains_itself, "engine.estimator_config.again"),
        (_list_that_contains_itself, "engine.transfer_policy.1"),
        (_section_without_old_fields_that_contains_itself, "traffic.again"),
    ],
    ids=["an-old-field", "a-list", "a-config-without-old-fields"],
)
def test_config_that_contains_itself_is_refused_with_the_place_named(build, place):
    """A YAML alias can name the mapping it is written in. Such a config has no copy to convert."""
    with pytest.raises(ValueError) as failure:
        _migrate(build())

    assert not isinstance(failure.value, CannotConvert)
    assert str(failure.value) == (
        f"{place} contains itself: the value is the mapping or list it is written in, as a YAML alias can say. "
        "A config cannot be circular; write the value out"
    )


def test_command_line_reports_a_config_that_contains_itself_as_a_usage_error(tmp_path, capsys):
    engine = yaml.safe_dump({"engine": _engine("aggregated")}, sort_keys=False)
    config = tmp_path / "config.yaml"
    # The last two lines add an engine field whose value names itself.
    config.write_text(
        yaml.safe_dump({"traffic": TRAFFIC}) + engine + "  estimator_config: &loop\n    again: *loop\n",
        encoding="utf-8",
    )
    output = tmp_path / "converted.yaml"

    assert _command_line("migrate", config, "--output", output) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: engine.estimator_config.again contains itself" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


def test_mapping_that_a_yaml_alias_names_a_second_time_is_converted_as_two_equal_mappings():
    engine = yaml.safe_dump({"engine": _engine("aggregated")}, sort_keys=False)
    # The last three lines add an engine field that names one mapping in two places; no value contains itself.
    raw = yaml.safe_load(
        yaml.safe_dump({"traffic": TRAFFIC})
        + engine
        + "  estimator_config:\n    correction: &same {mode: fixed}\n    fallback: *same\n"
    )
    written = raw["engine"]["estimator_config"]
    assert written["correction"] is written["fallback"]

    converted, _, _ = _migrate(raw)

    stated = converted["engine"]["ais_perf_model"]["estimator_config"]
    assert stated == {"correction": {"mode": "fixed"}, "fallback": {"mode": "fixed"}}
    assert stated["correction"] is not stated["fallback"]


TOO_MANY_VALUES = (
    "the config has more than {} values; a value that a YAML alias names counts every time it is used. "
    "Write the values out and do not nest YAML aliases"
)


def _values(value: Any) -> int:
    """How many values a copy of ``value`` holds: every mapping, list and scalar, each time it is used."""
    if isinstance(value, dict):
        return 1 + sum(_values(item) for item in value.values())
    if isinstance(value, list):
        return 1 + sum(_values(item) for item in value)
    return 1


class _CountsReads(list):
    """A list that counts how many times it is read from its first item on."""

    def __init__(self, items: list[Any]) -> None:
        super().__init__(items)
        self.reads = 0

    def __iter__(self):
        self.reads += 1
        # A copy without a limit would read on for hours; the test stops it here.
        assert self.reads <= 100_000, "the copy read this list far more often than the limit allows"
        return super().__iter__()


def test_limit_counts_every_value_of_the_whole_config_each_time_an_alias_names_it(monkeypatch):
    aliases = "a0: &a0 [x, x]\na1: &a1 [*a0, *a0]\na2: [*a1, *a1]\n"
    raw = yaml.safe_load(aliases + yaml.safe_dump(_config("predict", _engine("aggregated", **EST)), sort_keys=False))
    converted, moved, notes = _migrate(raw)
    # The file writes three lists; a copy holds a list for every place an alias names one.
    assert [_values(raw[name]) for name in ("a0", "a1", "a2")] == [3, 7, 15]
    count = _values(raw)

    # A config that holds exactly as many values as the limit is converted as it is under the limit as shipped.
    monkeypatch.setattr(migrate_module, "_MAX_VALUES", count)
    assert _migrate(raw) == (converted, moved, notes)
    assert converted["engine"]["ais_perf_model"] == EST
    assert converted["a2"] == [[["x", "x"], ["x", "x"]], [["x", "x"], ["x", "x"]]]
    assert converted["a2"][0] is not converted["a2"][1]

    # No section holds that many values alone: the count runs over the whole config.
    assert max(_values(section) for section in raw.values()) < count - 1
    monkeypatch.setattr(migrate_module, "_MAX_VALUES", count - 1)
    with pytest.raises(ValueError) as failure:
        _migrate(raw)
    assert not isinstance(failure.value, CannotConvert)
    assert str(failure.value) == TOO_MANY_VALUES.format(count - 1)


@pytest.mark.parametrize("old_fields", [{}, OP_LEVEL], ids=["no-old-fields", "old-fields"])
def test_copy_stops_at_the_limit_however_many_values_the_aliases_name(old_fields, monkeypatch):
    monkeypatch.setattr(migrate_module, "_MAX_VALUES", 1000)
    raw = _config("predict", _engine("aggregated", **old_fields))
    # A YAML alias is the object its anchor names. Each of forty keys names the key before it twice,
    # so the last key names the first list 2**39 times.
    first = value = _CountsReads(["x", "x"])
    for level in range(40):
        raw[f"a{level}"] = value
        value = [value, value]

    with pytest.raises(ValueError) as failure:
        _migrate(raw)

    assert not isinstance(failure.value, CannotConvert)
    assert str(failure.value) == TOO_MANY_VALUES.format(1000)
    assert 0 < first.reads <= 1000


def test_members_of_a_yaml_set_count_toward_the_limit(monkeypatch):
    monkeypatch.setattr(migrate_module, "_MAX_VALUES", 1000)
    raw = _config("predict", _engine("aggregated", **OP_LEVEL))
    # The copy takes a set whole, so a set of two thousand names is two thousand values and not one.
    raw["names"] = yaml.safe_load("!!set {" + ", ".join(f"n{index}" for index in range(2000)) + "}")
    assert isinstance(raw["names"], set)

    with pytest.raises(ValueError) as failure:
        _migrate(raw)
    assert str(failure.value) == TOO_MANY_VALUES.format(1000)

    raw["names"] = {"a", "b", "c"}
    converted, _, _ = _migrate(raw)
    assert converted["names"] == {"a", "b", "c"}
    assert converted["names"] is not raw["names"]


# YAML reads ``.nan`` as a number, which no path field takes.
NOT_A_PATH = float("nan")


@pytest.mark.parametrize(
    "engine",
    [
        _engine("aggregated", estimation_mode="no_such_mode"),
        _engine("aggregated", timing={"aggregated": {"forward_model": "bogus"}}),
        _engine("aggregated", systems_paths="text"),
        _engine("aggregated", estimator_config="text"),
        _engine("aggregated", timing={"aggregated": {"forward_model": ["op_level"]}}),
        _engine("aggregated", timing={"aggregated": {"forward_model": {"name": "op_level"}}}),
        _engine("aggregated", timing={"aggregated": {"fpm_parquet_path": NOT_A_PATH}}),
        _engine("aggregated", systems_path=""),
        _engine("aggregated", systems_paths=[""]),
        _engine("disaggregated", timing={"decode": {"systems_paths": ["data", ""]}}),
    ],
    ids=[
        "estimation_mode",
        "forward_model-of-a-worker",
        "systems_paths-not-a-list",
        "estimator_config-not-a-mapping",
        "forward_model-a-list",
        "forward_model-a-mapping",
        "fpm_parquet_path-not-a-string",
        "systems_path-empty",
        "systems_paths-empty-item",
        "systems_paths-of-a-worker-empty-item",
    ],
)
def test_config_that_does_not_validate_is_reported_as_unreadable_not_as_converted(engine):
    raw = _config("predict", engine)

    with pytest.raises(ValueError, match="the engine section is not a valid predict config") as failure:
        _migrate(raw)

    assert not isinstance(failure.value, CannotConvert)


@pytest.mark.parametrize(
    "old_fields",
    [{"agg_fpm_parquet_path": NOT_A_PATH}, {"systems_path": ""}, {"systems_paths": [""]}],
    ids=["fpm_parquet_path-not-a-string", "systems_path-empty", "systems_paths-empty-item"],
)
def test_search_space_that_does_not_validate_is_reported_as_unreadable_not_as_converted(old_fields):
    raw = _space("agg", **old_fields)

    with pytest.raises(ValueError, match="^the search_space section is not a valid search config: ") as failure:
        _migrate(raw, "search")

    assert not isinstance(failure.value, CannotConvert)


@pytest.mark.parametrize(
    ("raw", "section"),
    [
        (_config("predict", _engine("aggregated", timing={"aggregated": {"fpm_parquet_path": NOT_A_PATH}})), "engine"),
        (_config("predict", _engine("aggregated", systems_path="")), "engine"),
        (_config("predict", _engine("aggregated", timing={"aggregated": {"forward_model": ["op_level"]}})), "engine"),
        (_space("agg", systems_paths=[""]), "search_space"),
    ],
    ids=["fpm_parquet_path-not-a-string", "systems_path-empty", "forward_model-a-list", "search-empty-item"],
)
def test_command_line_reports_a_path_field_that_does_not_validate_as_a_usage_error(raw, section, tmp_path, capsys):
    config = _write(tmp_path / "config.yaml", raw)
    before = config.read_bytes()
    output = tmp_path / "converted.yaml"

    assert _command_line("migrate", config, "--output", output) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"error: the {section} section is not a valid " in captured.err
    assert "internal error" not in captured.err
    assert "Traceback" not in captured.err
    assert list(tmp_path.iterdir()) == [config]
    assert config.read_bytes() == before


# --- The conversion checks its own result -----------------------------------------------------


def _build_another_block(monkeypatch: pytest.MonkeyPatch, **changed: Any) -> None:
    """Make the conversion write a block that differs from the old fields in the ``changed`` keys."""
    build = migrate_module._new_block
    monkeypatch.setattr(migrate_module, "_new_block", lambda *args: {**build(*args), **changed})


WRITTEN_TWICE = {"estimation_mode": "op_level", "fallback_policy": "allow"}


def test_converted_config_that_gives_a_worker_another_setting_is_refused(monkeypatch):
    _build_another_block(monkeypatch, estimation_mode="auto")
    raw = _config("predict", _engine("aggregated", **WRITTEN_TWICE))
    original = copy.deepcopy(raw)

    with pytest.raises(MigrationCheckError) as failure:
        _migrate(raw)

    assert str(failure.value) == (
        'the converted config gives the aggregated worker estimation_mode="auto", '
        'but the original config gives "op_level"'
    )
    assert raw == original


def test_converted_config_that_does_not_validate_is_refused(monkeypatch):
    _build_another_block(monkeypatch, fallback_policy="sometimes")

    with pytest.raises(MigrationCheckError, match=r"(?s)^the converted config is not valid: .*fallback_policy"):
        _migrate(_config("predict", _engine("aggregated", **WRITTEN_TWICE)))


def test_command_line_writes_nothing_when_the_converted_config_means_something_else(tmp_path, monkeypatch, capsys):
    _build_another_block(monkeypatch, estimation_mode="auto")
    config = _write(tmp_path / "config.yaml", _config("predict", _engine("aggregated", **WRITTEN_TWICE)))
    before = config.read_bytes()
    output = tmp_path / "converted.yaml"

    assert _command_line("migrate", config, "--output", output) == 1

    captured = capsys.readouterr()
    assert "internal error" in captured.err
    assert 'estimation_mode="auto"' in captured.err
    assert "cannot convert" not in captured.out
    assert list(tmp_path.iterdir()) == [config]
    assert config.read_bytes() == before


def test_converted_config_is_read_from_every_folder_it_has_a_name_in(tmp_path, monkeypatch):
    cwd, folder, _ = _folders_of_a_linked_config(tmp_path)
    # A path that stays as written is right from the working directory and wrong from any other folder.
    monkeypatch.setattr(migrate_module, "_rewrite_paths", lambda block, *args: block)
    raw = _config("predict", _engine("aggregated", systems_paths=["data/systems"]))
    assert _migrate(raw, config_dir=cwd, cwd=cwd)[0]["engine"]["ais_perf_model"] == {"systems_paths": ["data/systems"]}

    with pytest.raises(MigrationCheckError) as failure:
        _migrate(raw, config_dir=cwd, cwd=cwd, other_config_dirs=[folder])

    assert str(failure.value) == (
        f'the converted config gives the aggregated worker systems_paths=["{folder}/data/systems"], '
        f'but the original config gives ["{cwd}/data/systems"]'
    )


def test_unknown_kind_and_missing_section_name_what_is_expected():
    with pytest.raises(ValueError, match="unknown config kind 'replay'; use predict, recommend or search"):
        _migrate(_config("predict", _engine("aggregated")), "replay")
    with pytest.raises(ValueError, match="a search config needs a top-level search_space mapping"):
        _migrate(_config("predict", _engine("aggregated")), "search")


# --- Search configs ---------------------------------------------------------------------------


def test_search_fields_at_the_top_level_become_one_config_without_worker_type():
    raw = _space("disagg", **EST)

    converted, moved, notes = _migrate(raw, "search")

    assert converted["search_space"]["ais_perf_model"] == EST
    assert sorted(moved) == sorted(f"search_space.{name}" for name in EST)
    assert notes == []
    dumps = [
        SearchSpace.model_validate(config["search_space"]).model_dump(mode="json", exclude_none=True)
        for config in (converted, raw)
    ]
    assert dumps[0] == dumps[1]
    assert _requests(converted, "disagg") == _requests(raw, "disagg")
    assert _migrate(converted, "search") == (converted, [], [])


def test_search_role_settings_become_one_entry_per_worker_type():
    raw = _space(
        "disagg", **EST, role_estimator_controls={"decode": {"estimation_mode": "auto", "transfer_policy": None}}
    )

    converted, moved, _ = _migrate(raw, "search")

    shared = {name: value for name, value in EST.items() if name != "transfer_policy"}
    assert converted["search_space"]["ais_perf_model"] == [
        {"worker_type": "prefill", **EST},
        # A null of the role blanks the top-level value, so the entry leaves the key out.
        {"worker_type": "decode", **shared, "estimation_mode": "auto"},
    ]
    assert sorted(moved) == sorted(
        [
            *(f"search_space.{name}" for name in EST),
            "search_space.role_estimator_controls.decode.estimation_mode",
            "search_space.role_estimator_controls.decode.transfer_policy",
        ]
    )
    # The move empties the role's mapping and then role_estimator_controls, so both leave the file.
    assert "role_estimator_controls" not in converted["search_space"]
    assert _requests(converted, "disagg") == _requests(raw, "disagg")
    assert _migrate(converted, "search") == (converted, [], [])


@pytest.mark.parametrize(
    ("top_level", "block"),
    [
        pytest.param({}, [{"worker_type": "prefill"}, {"worker_type": "decode"}], id="the-only-old-field"),
        pytest.param(
            {"transfer_policy": "xshape"},
            [{"worker_type": "prefill", "transfer_policy": "xshape"}, {"worker_type": "decode"}],
            id="next-to-a-top-level-policy",
        ),
    ],
)
def test_search_null_transfer_policy_of_a_role_is_moved_like_a_written_value(top_level, block):
    """The search reads a role's null in place of the top-level policy, so the null is a setting."""
    raw = _space("disagg", **top_level, role_estimator_controls={"decode": {"transfer_policy": None}})

    converted, moved, _ = _migrate(raw, "search")

    assert converted["search_space"]["ais_perf_model"] == block
    assert sorted(moved) == sorted(
        [*(f"search_space.{name}" for name in top_level), "search_space.role_estimator_controls.decode.transfer_policy"]
    )
    assert "role_estimator_controls" not in converted["search_space"]
    assert _requests(converted, "disagg") == _requests(raw, "disagg")
    assert _migrate(converted, "search") == (converted, [], [])


def test_search_legacy_forward_model_of_a_role_becomes_the_mode_alone():
    raw = _space("agg", agg_forward_model="fpm", agg_fpm_parquet_path="/data/fpm.parquet")

    converted, moved, notes = _migrate(raw, "search")

    # The search space reads the field as the mode and adds no fallback policy, unlike the engine config.
    assert converted["search_space"]["ais_perf_model"] == [
        {
            "worker_type": "aggregated",
            "estimation_mode": "fpm_interpolation",
            "estimator_config": {"fpm_interpolation": {"fpm_parquet_path": "/data/fpm.parquet"}},
        }
    ]
    assert sorted(moved) == ["search_space.agg_forward_model", "search_space.agg_fpm_parquet_path"]
    assert notes == [
        "search_space.agg_forward_model: fpm written as estimation_mode: fpm_interpolation",
        "search_space.agg_fpm_parquet_path written as estimator_config.fpm_interpolation.fpm_parquet_path",
    ]
    assert not {"agg_forward_model", "agg_fpm_parquet_path"} & set(converted["search_space"])
    assert _requests(converted, "agg") == _requests(raw, "agg")
    assert _migrate(converted, "search") == (converted, [], [])


@pytest.mark.parametrize(
    ("name", "value"),
    [("database_mode", "HYBRID"), ("transfer_policy", None)],
    ids=["a-value", "a-null-transfer-policy"],
)
def test_search_setting_of_a_role_the_space_does_not_run_cannot_be_converted(name, value):
    raw = _space("agg", role_estimator_controls={"prefill": {name: value}})

    with pytest.raises(CannotConvert) as failure:
        _migrate(raw, "search")

    assert failure.value.problems == (
        f"cannot convert: search_space.role_estimator_controls.prefill.{name}: this config runs no prefill "
        "worker, so ais_perf_model has no entry for the field; remove the field from the file",
    )


@pytest.mark.parametrize(
    "old_fields",
    [
        {},
        OP_LEVEL,
        {"role_estimator_controls": {"decode": OP_LEVEL}},
        {"role_estimator_controls": {"prefill": OP_LEVEL}},
    ],
    ids=["no old fields", "top level", "AFD role", "role next to the AFD role"],
)
def test_search_space_with_afd_roles_is_not_converted_wherever_its_old_fields_are(old_fields):
    afd_roles = {"decode": {"type": "afd", "afd": {"a_batch_size": 1}}}
    raw = _space("disagg", backend=["sglang"], afd_roles=afd_roles, **old_fields)
    original = copy.deepcopy(raw)
    # The space is a valid one; only the key cannot describe it.
    SearchSpace.model_validate(copy.deepcopy(raw["search_space"]))

    with pytest.raises(CannotConvert) as failure:
        _migrate(raw, "search")

    assert failure.value.problems == (
        "cannot convert: search_space.afd_roles is set, and ais_perf_model does not cover AFD yet",
    )
    assert raw == original


NOT_A_MAPPING = pytest.mark.parametrize("afd_roles", [None, False, 0, 3], ids=["null", "false", "zero", "number"])


@NOT_A_MAPPING
@pytest.mark.parametrize(
    "old_fields",
    [OP_LEVEL, {"role_estimator_controls": {"prefill": OP_LEVEL}}, {"prefill_forward_model": "op_level"}],
    ids=["top level", "role", "forward model of a role"],
)
def test_search_space_whose_afd_roles_is_not_a_mapping_is_reported_as_unreadable(afd_roles, old_fields):
    raw = _space("disagg", backend=["sglang"], afd_roles=afd_roles, **old_fields)
    original = copy.deepcopy(raw)
    # The search space itself stops on such a value, today with an error that is not a ValueError.
    with pytest.raises((TypeError, ValueError)):
        SearchSpace.model_validate(copy.deepcopy(raw["search_space"]))

    with pytest.raises(ValueError, match="^the search_space section is not a valid search config: ") as failure:
        _migrate(raw, "search")

    assert type(failure.value) is ValueError
    assert raw == original


@NOT_A_MAPPING
def test_command_line_reports_afd_roles_that_is_not_a_mapping_as_a_usage_error(afd_roles, tmp_path, capsys):
    config = _write(tmp_path / "search.yaml", _space("disagg", backend=["sglang"], afd_roles=afd_roles, **OP_LEVEL))
    before = config.read_bytes()
    output = tmp_path / "converted.yaml"

    assert _command_line("migrate", config, "--output", output) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: the search_space section is not a valid search config: " in captured.err
    assert "Traceback" not in captured.err
    assert config.read_bytes() == before
    assert not output.exists()


# --- Known limit ------------------------------------------------------------------------------


def test_converted_onboard_replay_config_is_not_the_config_onboard_reviewed(tmp_path):
    """Onboard compares its replay config with the file as plain data, so it rejects a converted file."""
    from aisim.support.validation import _prediction_config

    source, resources = _onboard_files(tmp_path)
    request_file, plan = tmp_path / "request.yaml", tmp_path / "plan"
    assert cli.main(_onboard_args(request_file, source, resources, model=str(tmp_path))) == 0
    assert cli.main(["onboard", "plan", "--config", str(request_file), "--output-dir", str(plan)]) == 0
    reviewed = _prediction_config(SupportRequest.from_yaml(request_file), plan, tmp_path / "trace.jsonl")

    converted, moved, _ = migrate_mapping(reviewed, "predict", config_dir=plan, cwd=tmp_path)

    assert "engine.workers.aggregated.timing.estimation_mode" in moved
    before, after = _validated("predict", reviewed, plan), _validated("predict", converted, plan)
    assert _effective(after, "aggregated") == _effective(before, "aggregated")
    # Onboard's check is this comparison of the two mappings, not a comparison of what they mean.
    assert converted != reviewed
