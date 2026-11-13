# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``engine.ais_perf_model`` is another spelling of the engine estimator fields.

A config that uses the key must mean exactly what the same config means when
it is written with the existing fields. So most tests here build both
spellings and compare what comes out: the dump, the perf model configs the
compiler builds, the recommend search space, or the error.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import itertools
import json
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError, model_validator

from aisim.compiler import prediction_to_replay_spec
from aisim.config import CorePredictionConfig, CoreRecommendationConfig
from aisim.config import ais_perf_model as rules
from aisim.config.engine import (
    EnginePredictionConfig,
    EngineRecommendationConfig,
    EstimatorPolicyConfig,
    TimingConfig,
    WorkerPredictionConfig,
)
from aisim.recommend import recommendation_to_sweeper
from aisim.sweeper.config import SearchSpace
from aisim_core import ais_perf_model as loader
from aisim_core.ais_perf_model import AisPerfModelDeprecationWarning, AisPerfModelError
from aisim_core.sdk import ForwardPassPerfModelConfig
from aisim_core.sdk.common import TRANSFER_PRESETS, TransferKind, resolve_transfer_policy

from .test_fpm_config_imports import _config as _profiled_config

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
# "both" is a recommend config that leaves the mode open, so it runs all three worker types.
ROLES = {
    "aggregated": ("aggregated",),
    "disaggregated": ("prefill", "decode"),
    "both": ("aggregated", "prefill", "decode"),
}
# Two sets of estimator settings that give a different value for every key they share.
EST = {
    "estimation_mode": "op_level",
    "fallback_policy": "deny",
    "database_mode": "HYBRID",
    "transfer_policy": "xshape",
    "estimator_config": {"correction": {"mode": "fixed", "min_observations": 3}},
}
OTHER = {
    "estimation_mode": "auto",
    "fallback_policy": "allow",
    "database_mode": "SILICON",
    "estimator_config": {"correction": {"mode": "fixed", "min_observations": 5}},
}
DATA_ROOTS = {"systems_paths": ["default"]}
# Lowering must never open the snapshot, so the file does not exist.
SNAPSHOT = {"path": "/no/such/folder/state.json", "sha256": "a" * 64, "on_tune": "frozen"}
# A worker that runs attention and FFN on separate pools (AFD), as each kind of config writes it.
AFD_WORKER = {
    "predict": {"type": "afd", "afd": {"n_a_nodes": 1, "n_f_nodes": 1, "tp_a": 1, "a_batch_size": 8}},
    "recommend": {"type": "afd", "afd": {"a_batch_size": 8}},
}


def _engine(mode: str, **fields: Any) -> dict[str, Any]:
    """A raw engine mapping with an empty worker for every worker type the mode runs."""
    engine: dict[str, Any] = {
        "model": MODEL,
        "hardware": SYSTEM,
        "backend": "vllm",
        "context_length": 256,
        "workers": {role: {} for role in ROLES[mode]},
    }
    if mode != "both":
        engine["mode"] = mode
    engine.update(copy.deepcopy(fields))
    return engine


def _old_spelling(mode: str, engine_fields: dict[str, Any], timing_fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The engine mapping written with the existing fields: engine level, and ``timing`` per worker."""
    engine = _engine(mode, **engine_fields)
    for role, fields in timing_fields.items():
        engine["workers"][role] = {"timing": copy.deepcopy(fields)}
    return engine


def _afd_engine(kind: str, mode: str, afd_role: str, **fields: Any) -> dict[str, Any]:
    """A raw engine mapping whose ``afd_role`` worker is an AFD worker. AFD runs on SGLang only."""
    engine = _engine(mode, backend="sglang", **fields)
    engine["workers"][afd_role] = copy.deepcopy(AFD_WORKER[kind])
    return engine


def _entries(**by_worker_type: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"worker_type": worker_type, **keys} for worker_type, keys in by_worker_type.items()]


def _predict(engine: dict[str, Any]) -> CorePredictionConfig:
    return CorePredictionConfig.model_validate({"engine": engine, "traffic": TRAFFIC})


def _recommend(engine: dict[str, Any]) -> CoreRecommendationConfig:
    return CoreRecommendationConfig.model_validate({"engine": engine, "traffic": TRAFFIC, **SEARCH})


VALIDATE = {"predict": _predict, "recommend": _recommend}


def _dump(config: CorePredictionConfig | CoreRecommendationConfig) -> str:
    """The config as a dump writes it. It is compared as text, so the order of keys counts too."""
    return json.dumps(config.model_dump(mode="json", exclude_none=True), indent=1)


def _perf_model_configs(config: CorePredictionConfig) -> dict[str, dict[str, Any]]:
    """The perf model config the compiler builds for each worker type."""
    deployment = prediction_to_replay_spec(config).backend_deployment
    names = {"aggregated": "agg_engine_args", "prefill": "prefill_engine_args", "decode": "decode_engine_args"}
    return {role: args["timing_model"]["config"] for role, name in names.items() if (args := getattr(deployment, name))}


def _search_space(config: CoreRecommendationConfig) -> dict[str, Any]:
    return recommendation_to_sweeper(config).search_space.model_dump(mode="json")


def _rejected(kind: str, engine: dict[str, Any]) -> str:
    """Validate a config that must fail and return the message."""
    with pytest.raises(ValidationError) as failure:
        VALIDATE[kind](engine)
    return str(failure.value)


# --- Configs without the key ------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_config_without_the_key_validates_and_dumps_as_before(kind):
    engine = _old_spelling(
        "disaggregated",
        {"estimation_mode": "op_level", "database_mode": "hybrid", "systems_path": "/srv/systems"},
        {"decode": {"fallback_policy": "allow", "forward_model": "fpm"}},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = VALIDATE[kind](engine)

    assert config.engine.ais_perf_model is None
    dumped = config.engine.model_dump(mode="json", exclude_none=True)
    assert "ais_perf_model" not in dumped
    assert {
        name: dumped[name] for name in ("estimation_mode", "database_mode", "fallback_policy", "systems_paths")
    } == {
        "estimation_mode": "op_level",
        "database_mode": "HYBRID",
        "fallback_policy": "deny",
        "systems_paths": ["/srv/systems"],
    }
    assert dumped["workers"]["prefill"]["timing"] == {"type": "default"}
    assert dumped["workers"]["decode"]["timing"] == {
        "type": "default",
        "estimation_mode": "fpm_interpolation",
        "fallback_policy": "allow",
    }


@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_a_null_key_is_the_same_as_no_key(kind):
    engine = _engine("aggregated", **EST)

    assert _dump(VALIDATE[kind]({**engine, "ais_perf_model": None})) == _dump(VALIDATE[kind](engine))


@pytest.mark.parametrize(
    "raw_engine",
    [
        None,
        "text",
        [],
        {},
        {"mode": ["aggregated"], "workers": []},
        {"workers": {"aggregated": {"type": "afd"}}, "ais_perf_model": None},
        EnginePredictionConfig.model_construct(),
    ],
    ids=["null", "string", "list", "empty", "unreadable shapes", "null key", "model instance"],
)
def test_lowering_returns_input_without_the_key_untouched(raw_engine):
    assert rules.lower_engine(raw_engine, "predict") is raw_engine
    assert rules.lower_engine(raw_engine, "recommend") is raw_engine


def test_estimator_fields_alone_still_build_without_the_key():
    assert EstimatorPolicyConfig().ais_perf_model is None
    assert EstimatorPolicyConfig(estimation_mode="op_level").model_dump()["estimation_mode"] == "op_level"


# --- The key means the same as the existing fields --------------------------------------------

# mode, the ais_perf_model value, then the same settings in the existing fields:
# engine level, and timing per worker.
SAME_MEANING = [
    pytest.param("aggregated", EST, EST, {}, id="one config, one worker type"),
    pytest.param(
        "aggregated", {"worker_type": "aggregated", **EST}, EST, {}, id="one config that names its worker type"
    ),
    pytest.param("aggregated", _entries(aggregated=EST), EST, {}, id="list of one"),
    pytest.param("aggregated", {**EST, **DATA_ROOTS}, {**EST, **DATA_ROOTS}, {}, id="one config with data roots"),
    pytest.param("disaggregated", EST, EST, {}, id="one config, two worker types"),
    pytest.param("disaggregated", _entries(prefill=EST, decode=EST), EST, {}, id="entries agree"),
    pytest.param("disaggregated", _entries(decode=EST, prefill=EST), EST, {}, id="entries agree, other order"),
    pytest.param(
        "disaggregated",
        _entries(prefill=EST, decode=OTHER),
        {},
        {"prefill": EST, "decode": OTHER},
        id="entries differ",
    ),
    pytest.param(
        "disaggregated",
        _entries(
            prefill={"estimation_mode": "op_level", "database_mode": "hybrid", "fallback_policy": "allow"},
            decode={"estimation_mode": "op_level", "database_mode": "HYBRID", "transfer_policy": "xshape"},
        ),
        {"estimation_mode": "op_level", "database_mode": "HYBRID"},
        {"prefill": {"fallback_policy": "allow"}, "decode": {"transfer_policy": "xshape"}},
        id="some keys agree, some do not",
    ),
    pytest.param("disaggregated", _entries(prefill=EST, decode={}), {}, {"prefill": EST}, id="one worker type only"),
    pytest.param("both", EST, EST, {}, id="one config, every mode"),
    pytest.param(
        "both",
        _entries(aggregated=EST, prefill=EST, decode=OTHER),
        {},
        {"aggregated": EST, "prefill": EST, "decode": OTHER},
        id="every mode, one entry differs",
    ),
]
ONE_MODE = [case for case in SAME_MEANING if case.values[0] != "both"]


@pytest.mark.parametrize(("mode", "value", "engine_fields", "timing_fields"), ONE_MODE)
def test_predict_config_means_the_same_as_its_old_spelling(mode, value, engine_fields, timing_fields):
    new = _predict(_engine(mode, ais_perf_model=value))
    old = _predict(_old_spelling(mode, engine_fields, timing_fields))

    assert _dump(new) == _dump(old)
    assert _perf_model_configs(new) == _perf_model_configs(old)


@pytest.mark.parametrize(("mode", "value", "engine_fields", "timing_fields"), SAME_MEANING)
def test_recommend_config_means_the_same_as_its_old_spelling(mode, value, engine_fields, timing_fields):
    new = _recommend(_engine(mode, ais_perf_model=value))
    old = _recommend(_old_spelling(mode, engine_fields, timing_fields))

    assert _dump(new) == _dump(old)
    assert _search_space(new) == _search_space(old)


@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_data_roots_of_one_worker_type_are_written_to_that_worker(kind):
    value = _entries(prefill={**EST, **DATA_ROOTS}, decode=EST)

    new = VALIDATE[kind](_engine("disaggregated", ais_perf_model=value))

    assert _dump(new) == _dump(VALIDATE[kind](_old_spelling("disaggregated", EST, {"prefill": DATA_ROOTS})))


@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_template_file_is_merged_before_the_keys_are_written(kind, tmp_path):
    template = tmp_path / "estimator.yaml"
    template.write_text("estimation_mode: op_level\ndatabase_mode: HYBRID\nfallback_policy: deny\n")
    value = {"from": str(template), "fallback_policy": "allow"}
    fields = {"estimation_mode": "op_level", "database_mode": "HYBRID", "fallback_policy": "allow"}

    new = VALIDATE[kind](_engine("disaggregated", ais_perf_model=value))

    assert _dump(new) == _dump(VALIDATE[kind](_engine("disaggregated", **fields)))
    assert new.engine.ais_perf_model == [{"worker_type": role, **fields} for role in ("prefill", "decode")]


@pytest.mark.parametrize(
    ("kind", "validate"),
    [
        ("predict", CorePredictionConfig.model_validate),
        ("recommend", lambda raw: CoreRecommendationConfig.model_validate({**raw, **SEARCH})),
    ],
    ids=["predict", "recommend"],
)
def test_fpm_profile_is_written_once_for_the_engine(kind, validate):
    engine = _profiled_config()["engine"]
    profile = engine.pop("fpm_profile")
    lowered = rules.lower_engine({**_engine("disaggregated"), "ais_perf_model": {"fpm_profile": profile}}, kind)

    new = validate({"engine": {**engine, "ais_perf_model": {"fpm_profile": profile, "fallback_policy": "allow"}}})
    old = validate({"engine": {**engine, "fpm_profile": profile, "fallback_policy": "allow"}})

    assert _dump(new) == _dump(old)
    assert lowered["fpm_profile"] == profile
    assert lowered["workers"] == {"prefill": {}, "decode": {}}


def test_old_fpm_profile_next_to_the_key_must_say_the_same():
    config = _profiled_config()
    engine = config["engine"]

    repeated = CorePredictionConfig.model_validate(
        {"engine": {**engine, "ais_perf_model": {"fpm_profile": engine["fpm_profile"]}}}
    )
    with pytest.raises(ValidationError) as failure:
        CorePredictionConfig.model_validate({"engine": {**engine, "ais_perf_model": {"fallback_policy": "allow"}}})

    assert rules.written_old_keys(config, "predict") == ["engine.fpm_profile"]
    assert _dump(repeated) == _dump(CorePredictionConfig.model_validate(config))
    assert "say something else: engine.fpm_profile. Remove the old fields." in str(failure.value)


def test_fpm_profile_must_be_one_value_for_all_worker_types():
    value = _entries(prefill={"fpm_profile": {"schema_version": 1}}, decode={"fpm_profile": {"schema_version": 2}})

    message = _rejected("predict", _engine("disaggregated", ais_perf_model=value))

    assert (
        "engine.ais_perf_model: fpm_profile is one value for the whole config, "
        "but the entries for prefill and decode give different values"
    ) in message


@pytest.mark.parametrize(("gives", "leaves_out"), [("prefill", "decode"), ("decode", "prefill")])
def test_fpm_profile_given_by_one_entry_only_names_the_entry_that_leaves_it_out(gives, leaves_out):
    value = _entries(**{gives: {"fpm_profile": {"schema_version": 1}}, leaves_out: {"estimation_mode": "auto"}})

    message = _rejected("predict", _engine("disaggregated", ais_perf_model=value))

    assert (
        "engine.ais_perf_model: fpm_profile is one value for the whole config, "
        f"but the entry for {gives} gives it and the entry for {leaves_out} does not; "
        "every entry must give the same value"
    ) in message
    assert "give different values" not in message


def test_the_validated_config_keeps_one_entry_per_worker_type_in_role_order():
    value = _entries(decode={**OTHER, "model": MODEL}, prefill=EST)

    config = _predict(_engine("disaggregated", ais_perf_model=value))

    assert config.engine.ais_perf_model == _entries(prefill=EST, decode={**OTHER, "model": MODEL})


# --- State snapshots --------------------------------------------------------------------------

# A snapshot belongs to one worker type, so nothing is written at engine level, agreeing or not.
SNAPSHOT_CASES = [
    pytest.param(
        "aggregated",
        {"from": {"state_snapshot": SNAPSHOT}, "estimation_mode": "fpm_learned", "fallback_policy": "allow"},
        {
            "aggregated": {
                "estimation_mode": "fpm_learned",
                "fallback_policy": "allow",
                "estimator_config": {"state_snapshot": SNAPSHOT},
            }
        },
        id="one config, one worker type",
    ),
    pytest.param(
        "disaggregated",
        _entries(
            prefill={"from": {"state_snapshot": SNAPSHOT}, "fallback_policy": "allow"},
            decode={"fallback_policy": "allow", "estimation_mode": "op_level"},
        ),
        {
            "prefill": {"fallback_policy": "allow", "estimator_config": {"state_snapshot": SNAPSHOT}},
            "decode": {"fallback_policy": "allow", "estimation_mode": "op_level"},
        },
        id="an agreeing key stays per worker",
    ),
    pytest.param(
        "disaggregated",
        _entries(
            prefill={"fallback_policy": "allow"},
            decode={"fallback_policy": "allow", "estimator_config": {"state_snapshot": SNAPSHOT}},
        ),
        {
            "prefill": {"fallback_policy": "allow"},
            "decode": {"fallback_policy": "allow", "estimator_config": {"state_snapshot": SNAPSHOT}},
        },
        id="snapshot written inside estimator_config",
    ),
    pytest.param(
        "aggregated",
        {"from": {"state_snapshot": SNAPSHOT}, "estimator_config": {"correction": {"mode": "fixed"}}},
        {"aggregated": {"estimator_config": {"correction": {"mode": "fixed"}, "state_snapshot": SNAPSHOT}}},
        id="snapshot joins the written estimator_config",
    ),
]


@pytest.mark.parametrize("kind", ["predict", "recommend"])
@pytest.mark.parametrize(("mode", "value", "timing_fields"), SNAPSHOT_CASES)
def test_snapshot_entries_lower_to_the_per_worker_fields(kind, mode, value, timing_fields):
    new = VALIDATE[kind](_engine(mode, ais_perf_model=value))

    assert _dump(new) == _dump(VALIDATE[kind](_old_spelling(mode, {}, timing_fields)))


def test_fpm_profile_of_a_snapshot_entry_still_goes_to_the_engine():
    profile = {"schema_version": 1}
    value = {"from": {"state_snapshot": SNAPSHOT}, "fpm_profile": profile, "fallback_policy": "allow"}

    lowered = rules.lower_engine(_engine("aggregated", ais_perf_model=value), "predict")

    assert lowered["fpm_profile"] == profile
    assert "fallback_policy" not in lowered
    assert lowered["workers"]["aggregated"]["timing"] == {
        "fallback_policy": "allow",
        "estimator_config": {"state_snapshot": SNAPSHOT},
    }


def test_one_snapshot_config_cannot_serve_two_worker_types():
    message = _rejected("predict", _engine("disaggregated", ais_perf_model={"from": {"state_snapshot": SNAPSHOT}}))

    assert "engine.ais_perf_model names a state snapshot, and a state snapshot belongs to one worker type." in message
    assert "This config runs prefill, decode; write a list with one entry per worker_type" in message


def test_snapshot_written_in_both_places_is_an_error():
    value = {"from": {"state_snapshot": SNAPSHOT}, "estimator_config": {"state_snapshot": SNAPSHOT}}

    message = _rejected("predict", _engine("aggregated", ais_perf_model=value))

    assert "engine.ais_perf_model entry for aggregated: the state snapshot is written twice" in message


@pytest.mark.parametrize("kind", ["predict", "recommend"])
@pytest.mark.parametrize(
    ("written", "shown"),
    [("fixed", 'str "fixed"'), (3, "int 3"), ([{"correction": {"mode": "fixed"}}], "a list")],
    ids=["text", "number", "list"],
)
def test_estimator_config_next_to_a_snapshot_must_be_a_mapping(kind, written, shown):
    # The snapshot is stored inside estimator_config, so there must be a mapping to store it in.
    value = {"from": {"state_snapshot": SNAPSHOT}, "estimator_config": written}

    message = _rejected(kind, _engine("aggregated", ais_perf_model=value))

    assert (
        "engine.ais_perf_model entry for aggregated: estimator_config must be a mapping next to "
        f"from.state_snapshot, got {shown}"
    ) in message


# --- The key next to existing fields ----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "engine_fields", "timing_fields", "paths"),
    [
        pytest.param(EST, {"estimation_mode": "fpm_regression"}, {}, "engine.estimation_mode", id="another value"),
        pytest.param(
            EST,
            {"database_mode": "EMPIRICAL", "fallback_policy": "allow"},
            {"aggregated": {"transfer_policy": "other"}},
            "engine.fallback_policy, engine.database_mode, engine.workers.aggregated.timing.transfer_policy",
            id="every conflict is listed",
        ),
        pytest.param(EST, {"systems_path": "default"}, {}, "engine.systems_path", id="systems_path"),
        pytest.param(
            EST,
            {},
            {"aggregated": {"forward_model": "op_level"}},
            "engine.workers.aggregated.timing.forward_model",
            id="forward_model",
        ),
        pytest.param(
            {"estimation_mode": "fpm_interpolation"},
            {},
            {"aggregated": {"fpm_parquet_path": "fpm.parquet"}},
            "engine.workers.aggregated.timing.fpm_parquet_path",
            id="fpm_parquet_path",
        ),
        pytest.param(
            EST,
            {},
            {"aggregated": {"estimation_mode": "op_level"}},
            "engine.workers.aggregated.timing.estimation_mode",
            id="same value at another level",
        ),
        pytest.param(
            {"estimation_mode": "op_level"},
            {"transfer_policy": "xshape"},
            {},
            "engine.transfer_policy",
            id="a field no entry gives",
        ),
    ],
)
def test_old_field_that_says_something_else_is_an_error(value, engine_fields, timing_fields, paths):
    engine = {**_old_spelling("aggregated", engine_fields, timing_fields), "ais_perf_model": value}

    message = _rejected("predict", engine)

    assert (
        "engine.ais_perf_model cannot be combined with old estimator fields that say something else: "
        f"{paths}. Remove the old fields."
    ) in message


@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_default_valued_old_fields_do_not_block_the_key(kind):
    # A dump lists every default, so a default next to the key is not a choice the author made.
    defaults = {
        "estimation_mode": "auto",
        "fallback_policy": "deny",
        "database_mode": "silicon",
        "estimator_config": {},
        "transfer_policy": None,
        "systems_paths": None,
    }
    engine = _old_spelling("aggregated", defaults, {"aggregated": {"type": "default", "estimation_mode": None}})

    new = VALIDATE[kind]({**engine, "ais_perf_model": EST})

    assert _dump(new) == _dump(VALIDATE[kind](_engine("aggregated", **EST)))


def test_old_fields_that_repeat_the_key_are_accepted():
    value = _entries(prefill={"estimation_mode": "op_level", "database_mode": "HYBRID"}, decode=OTHER)
    engine = _old_spelling("disaggregated", {}, {"prefill": {"estimation_mode": "op_level", "database_mode": "hybrid"}})

    new = _predict({**engine, "ais_perf_model": value})

    assert _dump(new) == _dump(_predict(_engine("disaggregated", ais_perf_model=value)))


@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_old_transfer_policy_is_compared_by_the_kinds_it_names(kind):
    # The key names the kind as text. The old fields name it as a list, for the whole engine and for one worker.
    value = _entries(prefill=EST, decode=OTHER)
    whole_engine = _old_spelling("aggregated", {"transfer_policy": ["xshape"]}, {})
    one_worker = _old_spelling("disaggregated", {}, {"prefill": {"transfer_policy": ["xshape"]}})
    other_kind = _old_spelling("disaggregated", {}, {"prefill": {"transfer_policy": ["xquant"]}})

    shared = VALIDATE[kind]({**whole_engine, "ais_perf_model": EST})
    per_worker = VALIDATE[kind]({**one_worker, "ais_perf_model": value})
    message = _rejected(kind, {**other_kind, "ais_perf_model": value})

    assert _dump(shared) == _dump(VALIDATE[kind](_engine("aggregated", ais_perf_model=EST)))
    assert _dump(per_worker) == _dump(VALIDATE[kind](_engine("disaggregated", ais_perf_model=value)))
    assert "say something else: engine.workers.prefill.timing.transfer_policy. Remove the old fields." in message


@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_old_transfer_policy_that_is_not_a_policy_says_something_else_than_the_key(kind):
    message = _rejected(kind, _engine("aggregated", ais_perf_model=EST, transfer_policy=5))

    assert "say something else: engine.transfer_policy. Remove the old fields." in message


def test_old_transfer_policy_tuple_that_is_not_a_policy_repeats_the_same_tuple_in_the_key():
    # Only Python code can pass a tuple. The loader makes a list of the tuple in the key; the two are one value.
    value = {"transfer_policy": ("XSHAPE",)}

    new = _predict(_engine("aggregated", ais_perf_model=value, transfer_policy=("XSHAPE",)))

    assert _dump(new) == _dump(_predict(_engine("aggregated", ais_perf_model=value)))
    assert new.engine.transfer_policy == ["XSHAPE"]


ROUND_TRIPS = [
    pytest.param("predict", "aggregated", EST, id="predict, engine level"),
    pytest.param("predict", "disaggregated", _entries(prefill=EST, decode=OTHER), id="predict, per worker"),
    pytest.param("predict", "disaggregated", SNAPSHOT_CASES[1].values[1], id="predict, snapshot"),
    pytest.param("recommend", "both", EST, id="recommend, engine level"),
    pytest.param(
        "recommend",
        "both",
        _entries(aggregated=EST, prefill=OTHER, decode={**OTHER, "model": MODEL}),
        id="recommend, per worker",
    ),
]


@pytest.mark.parametrize(("kind", "mode", "value"), ROUND_TRIPS)
def test_dump_with_the_key_added_back_is_the_same_config(kind, mode, value):
    first = VALIDATE[kind](_engine(mode, ais_perf_model=value))
    dumped = first.engine.model_dump(mode="json")

    again = VALIDATE[kind]({**dumped, "ais_perf_model": value})

    assert "ais_perf_model" not in dumped
    assert _dump(again) == _dump(first)


@pytest.mark.parametrize(("kind", "mode", "value"), ROUND_TRIPS)
def test_lowering_twice_changes_nothing_and_leaves_the_input_alone(kind, mode, value):
    raw = _engine(mode, ais_perf_model=value)
    raw["workers"][ROLES[mode][0]] = {"timing": {"type": "default"}, "replicas": 1}
    before = copy.deepcopy(raw)

    once = rules.lower_engine(raw, kind)

    assert raw == before
    assert once is not raw
    assert [entry["worker_type"] for entry in once["ais_perf_model"]] == list(ROLES[mode])
    assert rules.lower_engine(once, kind) == once
    assert rules.lower_engine(copy.deepcopy(once), kind) == once


def test_lowered_mapping_no_longer_needs_the_template_file(tmp_path):
    template = tmp_path / "estimator.yaml"
    template.write_text("estimation_mode: op_level\n")
    once = rules.lower_engine(_engine("aggregated", ais_perf_model={"from": str(template)}), "predict")
    template.unlink()

    assert once["ais_perf_model"] == [{"worker_type": "aggregated", "estimation_mode": "op_level"}]
    assert rules.lower_engine(once, "predict") == once


# --- Values and engine shapes lowering cannot read --------------------------------------------


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("estimator.yaml", "engine.ais_perf_model cannot be a string; write the mapping or 'from: <file>'"),
        (3, "engine.ais_perf_model must be a mapping or a list of mappings, got int 3"),
        ({}, "ais_perf_model: the config is empty; write at least one key"),
        ([], "ais_perf_model: the list is empty; write at least one config"),
        ([{"estimation_mode": "auto"}], "worker_type is missing; every config in a list needs one of"),
        (_entries(decode=EST) * 2, "both have worker_type 'decode'"),
        (_entries(prefill=EST), "engine.ais_perf_model has no entry for worker type decode; it has: prefill"),
        (
            _entries(prefill=EST, decode=EST, aggregated=EST),
            "engine.ais_perf_model has an entry for worker type aggregated, which this config does not run; "
            "it runs: prefill, decode",
        ),
        (
            {"from": {"dataset": {"root": "/data", "manifest": "manifest.json"}}},
            "dataset entries are not supported in predict and recommend configs yet; "
            "use 'aisim perf-model check' or a Dynamo component",
        ),
    ],
    ids=[
        "string",
        "number",
        "empty mapping",
        "empty list",
        "list entry without worker_type",
        "worker type twice",
        "worker type missing",
        "worker type the config does not run",
        "dataset",
    ],
)
def test_value_that_cannot_be_lowered_is_reported_by_name(value, message):
    raw = _engine("disaggregated", ais_perf_model=value)

    assert message in _rejected("predict", raw)
    with pytest.raises(AisPerfModelError) as failure:
        rules.lower_engine(raw, "predict")
    assert message in str(failure.value)


def test_lowering_for_a_kind_that_has_no_engine_block_names_the_kinds_that_have_one():
    # A search config keeps the key under search_space, so this is a slip in the calling code, not in a config.
    engine = _engine("aggregated", ais_perf_model=EST)

    with pytest.raises(ValueError, match="^unknown config kind 'search'; use predict or recommend$") as failure:
        rules.lower_engine(engine, "search")

    assert not isinstance(failure.value, AisPerfModelError)


@pytest.mark.parametrize(
    ("kind", "changes", "message"),
    [
        ("predict", {"workers": []}, "needs engine.workers written as plain data (a mapping), got a list"),
        (
            "predict",
            {"workers": {"aggregated": {"timing": "default"}}},
            'needs engine.workers.aggregated.timing written as plain data (a mapping), got str "default"',
        ),
        (
            "predict",
            {"workers": {"aggregated": WorkerPredictionConfig()}},
            "needs engine.workers.aggregated written as plain data (a mapping), got an object of type "
            "WorkerPredictionConfig",
        ),
        (
            "predict",
            {"mode": "serial"},
            'cannot be used with engine.mode "serial"; the modes are: aggregated, disaggregated',
        ),
        ("predict", {"mode": ["aggregated"]}, "needs engine.mode written as one mode name, got a list"),
        ("predict", {"mode": {"choices": ["aggregated"]}}, "needs engine.mode written as one mode name, got a mapping"),
        ("recommend", {"mode": {"choices": []}}, "needs engine.mode written as one mode name or {choices: ["),
        ("recommend", {"mode": {"choices": [1]}}, "cannot be used with engine.mode 1; the modes are:"),
    ],
    ids=[
        "workers is a list",
        "timing is a string",
        "worker is a model",
        "unknown mode",
        "mode is a list",
        "choices in predict",
        "no choices",
        "choice is not a name",
    ],
)
def test_engine_shape_lowering_cannot_read_is_a_named_error(kind, changes, message):
    raw = {**_engine("aggregated", ais_perf_model=EST), **changes}

    with pytest.raises(AisPerfModelError) as failure:
        rules.lower_engine(raw, kind)

    assert f"engine.ais_perf_model {message}" in str(failure.value)
    assert message in _rejected(kind, raw)


FIXED_TIMING = {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
UNCOVERED = [
    pytest.param(
        "predict",
        {"backend": "sglang", "workers": {"aggregated": AFD_WORKER["predict"]}},
        "engine.workers.aggregated.type is afd, and ais_perf_model does not cover AFD yet",
        id="afd worker",
    ),
    pytest.param(
        "recommend",
        {"backend": "sglang", "workers": {"aggregated": {}, "prefill": {}, "decode": AFD_WORKER["recommend"]}},
        "engine.workers.decode.type is afd, and ais_perf_model does not cover AFD yet",
        id="afd worker next to ordinary workers",
    ),
    pytest.param(
        "predict",
        {
            "mode": "disaggregated",
            "backend": "sglang",
            "workers": {"decode": AFD_WORKER["predict"], "prefill": AFD_WORKER["predict"]},
        },
        "engine.workers.prefill.type is afd, and ais_perf_model does not cover AFD yet",
        id="two afd workers",
    ),
    pytest.param(
        "predict",
        {"backend": "sglang", "workers": {"aggregated": {}, "decode": AFD_WORKER["predict"]}},
        "engine.workers.decode.type is afd, and ais_perf_model does not cover AFD yet",
        id="afd worker the mode does not run",
    ),
    pytest.param(
        "predict",
        {"workers": {"aggregated": {}, "encoder": {}}},
        "engine.workers.encoder is set, and ais_perf_model does not cover an encoder pool yet",
        id="encoder pool",
    ),
    pytest.param(
        "predict",
        {"workers": {"aggregated": {"timing": FIXED_TIMING}}},
        'engine.workers.aggregated.timing.type is "fixed", and ais_perf_model covers only default timing',
        id="fixed timing",
    ),
    pytest.param(
        "recommend",
        {"workers": {"aggregated": {}, "prefill": {}, "decode": {"timing": {"type": "polynomial"}}}},
        'engine.workers.decode.timing.type is "polynomial", and ais_perf_model covers only default timing',
        id="polynomial timing",
    ),
]


@pytest.mark.parametrize(("kind", "changes", "reason"), UNCOVERED)
def test_config_the_key_cannot_describe_yet_is_an_error(kind, changes, reason):
    engine = {**_engine("both" if kind == "recommend" else "aggregated"), **changes}

    assert rules.uncovered_reason(engine, kind) == reason
    message = _rejected(kind, {**engine, "ais_perf_model": EST})
    assert f"engine.ais_perf_model cannot be used in this config: {reason}. Keep the existing estimator fields." in (
        message
    )


def test_timing_of_a_worker_the_mode_does_not_run_is_ignored():
    engine = _engine("aggregated")
    engine["workers"]["decode"] = {"timing": FIXED_TIMING}

    assert rules.uncovered_reason(engine, "predict") is None
    assert rules.uncovered_reason(_engine("both"), "recommend") is None


def _afd_engine_with_old_fields(kind: str, place: str) -> dict[str, Any]:
    """A disaggregated engine with an AFD decode worker that states ``EST`` in one place.

    ``place`` is ``engine`` for the engine section, ``decode`` for the timing
    of the AFD worker, and ``prefill`` for the timing of the ordinary worker
    next to it.
    """
    engine = _afd_engine(kind, "disaggregated", "decode", **(EST if place == "engine" else {}))
    if place != "engine":
        engine["workers"][place]["timing"] = copy.deepcopy(EST)
    return engine


@pytest.mark.parametrize("place", ["engine", "decode", "prefill"])
@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_config_with_an_afd_worker_keeps_its_old_fields_and_they_are_not_reported(kind, place):
    engine = _afd_engine_with_old_fields(kind, place)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stored = VALIDATE[kind](engine).engine.model_dump(mode="json", exclude_none=True)
        rules.warn_old_keys({"engine": engine}, kind, source="afd.yaml")

    written = stored if place == "engine" else stored["workers"][place]["timing"]
    assert {name: written[name] for name in ("estimation_mode", "database_mode")} == {
        "estimation_mode": "op_level",
        "database_mode": "HYBRID",
    }
    assert rules.written_old_keys({"engine": engine}, kind) == []


@pytest.mark.parametrize("place", ["engine", "decode", "prefill"])
@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_key_next_to_an_afd_worker_is_rejected_wherever_the_old_fields_are(kind, place):
    engine = _afd_engine_with_old_fields(kind, place)

    message = _rejected(kind, {**engine, "ais_perf_model": _entries(prefill=EST, decode=EST)})

    assert (
        "engine.ais_perf_model cannot be used in this config: engine.workers.decode.type is afd, "
        "and ais_perf_model does not cover AFD yet. Keep the existing estimator fields."
    ) in message


@pytest.mark.parametrize(
    "worker",
    [{"type": "monolithic"}, {"type": None}, {"afd": {}}, {"type": ["afd"]}, {"type": {"choices": ["afd"]}}],
    ids=["monolithic", "null type", "afd block without the type", "list", "choices"],
)
def test_only_a_worker_whose_type_is_afd_is_an_afd_worker(worker):
    # The engine model reads a worker as AFD from the plain value ``type: afd`` and rejects the other spellings.
    engine = _engine("aggregated")
    engine["workers"]["aggregated"] = worker

    assert rules.uncovered_reason(engine, "predict") is None
    assert rules.uncovered_reason(engine, "recommend") is None


# AFD is a worker type. The engine model has no AFD mode and no ``engine.afd`` block, and says so itself.
NO_AFD_MODE = "Input should be 'aggregated' or 'disaggregated'"
NO_AFD_BLOCK = "engine.afd\n  Extra inputs are not permitted"


@pytest.mark.parametrize(
    ("kind", "changes", "without_the_key", "with_the_key"),
    [
        (
            "predict",
            {"mode": "afd"},
            f"engine.mode\n  {NO_AFD_MODE}",
            'engine.ais_perf_model cannot be used with engine.mode "afd"; the modes are: aggregated, disaggregated',
        ),
        (
            "recommend",
            {"mode": {"choices": ["aggregated", "afd"]}},
            f"engine.mode.Choices[Literal['aggregated', 'disaggregated']].choices.1\n  {NO_AFD_MODE}",
            'engine.ais_perf_model cannot be used with engine.mode "afd"; the modes are: aggregated, disaggregated',
        ),
        ("predict", {"afd": {"combined_with_pd": False}}, NO_AFD_BLOCK, NO_AFD_BLOCK),
        ("recommend", {"afd": {"combined_with_pd": False}}, NO_AFD_BLOCK, NO_AFD_BLOCK),
    ],
    ids=["afd mode", "afd mode choice", "afd block", "afd block in recommend"],
)
def test_afd_written_as_a_mode_or_an_engine_block_is_rejected_with_or_without_the_key(
    kind, changes, without_the_key, with_the_key
):
    engine = {**_engine("both" if kind == "recommend" else "aggregated"), **changes}

    assert without_the_key in _rejected(kind, engine)
    assert with_the_key in _rejected(kind, {**engine, "ais_perf_model": EST})


# Each value has settings for a worker of its own, so lowering has something to write there.
@pytest.mark.parametrize(
    ("mode", "workers", "value", "message"),
    [
        ("aggregated", {}, {"from": {"state_snapshot": SNAPSHOT}}, "aggregated mode requires workers.aggregated"),
        ("aggregated", {"aggregated": None}, EST, "aggregated mode requires workers.aggregated"),
        (
            "disaggregated",
            {"prefill": {}},
            _entries(prefill=EST, decode=OTHER),
            "disaggregated mode requires prefill and decode workers",
        ),
    ],
    ids=["no worker", "null worker", "one of two workers"],
)
def test_missing_worker_keeps_the_error_of_the_engine_model(mode, workers, value, message):
    legacy = {**_engine(mode), "workers": workers}
    engine = {**legacy, "ais_perf_model": value}

    assert message in _rejected("predict", legacy)
    assert message in _rejected("predict", engine)
    # Lowering only adds timing to a worker the author wrote.
    assert set(rules.lower_engine(engine, "predict")["workers"]) == set(workers)


# --- Identity keys in a recommend config ------------------------------------------------------


@pytest.mark.parametrize(
    ("keys", "engine_fields"),
    [
        ({"model": MODEL, "system": SYSTEM, "backend": "vllm"}, {}),
        ({"backend": "sglang"}, {"backend": {"choices": ["sglang"]}}),
        ({"nextn": 0, "enable_eplb": False}, {}),
        ({"nextn": 2}, {"nextn": 2, "nextn_accepted": 1.5}),
        ({"backend_version": "0.25.1"}, {"backend_version": "0.25.1"}),
        ({"backend_version": "0.25.1"}, {"backend_version": {"vllm": "0.25.1"}}),
        ({"attention_backend": "flashinfer"}, {"attention_backend": "flashinfer"}),
    ],
    ids=[
        "model, system and backend",
        "one backend choice",
        "unset means the default",
        "nextn",
        "one version",
        "version of the one backend",
        "model control",
    ],
)
def test_recommend_entry_may_restate_what_the_engine_section_states(keys, engine_fields):
    engine = _engine("aggregated", **engine_fields)

    config = _recommend({**engine, "ais_perf_model": {**EST, **keys}})

    assert _dump(config) == _dump(_recommend({**engine, **EST}))


def test_recommend_entry_system_is_matched_against_the_hardware_of_its_worker():
    engine = _engine("disaggregated")
    engine["workers"]["decode"] = {"hardware": "b200_sxm"}
    matching = _entries(prefill={**EST, "system": SYSTEM}, decode={**EST, "system": "b200_sxm"})
    engine_wide = _entries(prefill=EST, decode={**EST, "system": SYSTEM})

    config = _recommend({**engine, "ais_perf_model": matching})

    assert _dump(config) == _dump(_recommend({**engine, **EST}))
    assert (
        'engine.ais_perf_model entry for decode: system="h200_sxm" differs from '
        'engine.workers.decode.hardware="b200_sxm"'
    ) in _rejected("recommend", {**engine, "ais_perf_model": engine_wide})


@pytest.mark.parametrize(
    ("keys", "engine_fields", "message"),
    [
        ({"model": "other/model"}, {}, 'model="other/model" differs from engine.model="Qwen/Qwen3-0.6B"'),
        (
            {"backend": "vllm"},
            {"backend": {"choices": ["vllm", "sglang"]}},
            'backend="vllm" needs one stated value at engine.backend (found {"choices": ["vllm", "sglang"]}); '
            "state it there or drop the key",
        ),
        (
            {"system": SYSTEM},
            {"hardware": "auto"},
            'system="h200_sxm" needs one stated value at engine.hardware (found "auto"); '
            "state it there or drop the key",
        ),
        ({"nextn": 2}, {}, "nextn=2 differs from engine.nextn, which is not set (its default is 0)"),
        ({"moe_backend": "a"}, {"moe_backend": "b"}, 'moe_backend="a" differs from engine.moe_backend="b"'),
        (
            {"backend_version": "0.25.1"},
            {"backend": {"choices": ["vllm", "sglang"]}, "backend_version": {"vllm": "0.25.1", "sglang": "0.5.0"}},
            'backend_version="0.25.1" needs one stated value at engine.backend_version',
        ),
        (
            {"backend_version": "0.25.1"},
            {"backend_version": {"vllm": "0.24.0"}},
            'backend_version="0.25.1" differs from engine.backend_version="0.24.0"',
        ),
        (
            {"from": {"state_snapshot": SNAPSHOT}, "nextn": 0},
            {},
            "nextn=0 cannot be checked because engine.nextn is not set, and a worker timed by a state snapshot "
            "takes an unset value from the snapshot",
        ),
        (
            {"tp": 4, "kv_block_size": 16, "pp": 1},
            {},
            "tp, kv_block_size, pp cannot be checked in a recommend config. The recommend engine section does not "
            "state them (recommend chooses tp and similar values for each candidate). Drop these keys from the "
            "entry, or from the template file when 'from: <file>' brings them in.",
        ),
        ({"nextn": True}, {"nextn": 1, "nextn_accepted": 1.0}, "nextn=true differs from engine.nextn=1"),
        ({"nextn": False}, {}, "nextn=false differs from engine.nextn, which is not set (its default is 0)"),
        ({"enable_eplb": 1}, {"enable_eplb": True}, "enable_eplb=1 differs from engine.enable_eplb=true"),
        (
            {"enable_eplb": 0},
            {},
            "enable_eplb=0 differs from engine.enable_eplb, which is not set (its default is false)",
        ),
    ],
    ids=[
        "model differs",
        "several backends",
        "hardware is auto",
        "unset and not the default",
        "model control differs",
        "version of several backends",
        "version differs",
        "snapshot and unset",
        "keys recommend chooses",
        "true for the number 1",
        "false for the default 0",
        "1 for true",
        "0 for the default false",
    ],
)
def test_recommend_entry_that_cannot_be_matched_is_an_error(keys, engine_fields, message):
    engine = {**_engine("aggregated", ais_perf_model={**EST, **keys}), **engine_fields}

    assert f"engine.ais_perf_model entry for aggregated: {message}" in _rejected("recommend", engine)


# --- Identity keys in a predict config, through the compiler ----------------------------------


def test_predict_entry_may_restate_what_the_engine_section_fixes():
    restated = {"model": MODEL, "system": SYSTEM, "backend": "vllm", "tp": 1, "kv_block_size": 64, "nextn": 0}
    # The compiler leaves enable_eplb out when it is off; the entry may still state that default.
    left_out = {"enable_eplb": False}

    config = _predict(_engine("aggregated", ais_perf_model={**EST, **restated, **left_out}))

    assert _perf_model_configs(config) == _perf_model_configs(_predict(_engine("aggregated", **EST)))


def test_predict_entry_that_contradicts_the_engine_section_fails_when_compiled():
    wrong = {"model": "other/model", "tp": 4, "moe_backend": "custom", "enable_eplb": True}
    config = _predict(_engine("disaggregated", ais_perf_model=_entries(prefill=EST, decode={**EST, **wrong})))

    with pytest.raises(AisPerfModelError) as failure:
        prediction_to_replay_spec(config)

    assert str(failure.value) == (
        "engine.ais_perf_model entry for decode does not match the engine section: "
        'model: entry has "other/model", the engine section gives "Qwen/Qwen3-0.6B"; '
        "tp: entry has 4, the engine section gives 1; "
        'moe_backend: entry has "custom", the engine section does not set it; '
        "enable_eplb: entry has true, the engine section gives false. "
        "Fix the engine section or drop the key from the entry."
    )


def test_predict_entry_that_writes_a_boolean_for_a_number_fails_when_compiled():
    # Python reads True as 1 and False as 0. The perf model takes neither for the other.
    mistyped = {"tp": True, "nextn": False, "enable_eplb": 0}
    config = _predict(_engine("aggregated", ais_perf_model={**EST, **mistyped}))

    with pytest.raises(AisPerfModelError) as failure:
        prediction_to_replay_spec(config)

    assert str(failure.value) == (
        "engine.ais_perf_model entry for aggregated does not match the engine section: "
        "tp: entry has true, the engine section gives 1; "
        "nextn: entry has false, the engine section gives 0; "
        "enable_eplb: entry has 0, the engine section gives false. "
        "Fix the engine section or drop the key from the entry."
    )


@pytest.mark.parametrize(
    ("key", "written", "compiled", "problem"),
    [
        ("pp", True, 1, "pp: entry has true, the engine section gives 1"),
        ("decoder_replay", 1, True, "decoder_replay: entry has 1, the engine section gives true"),
        (
            "speculation",
            {"accepted": [True, 2]},
            {"accepted": [1, 2]},
            'speculation: entry has {"accepted": [true, 2]}, the engine section gives {"accepted": [1, 2]}',
        ),
    ],
    ids=["true for 1", "1 for true", "inside a mapping and a list"],
)
def test_identity_check_keeps_booleans_and_numbers_apart(key, written, compiled, problem):
    with pytest.raises(AisPerfModelError) as failure:
        rules.check_entry_identity(_entries(decode={key: written}), "decode", {key: compiled})

    assert f"entry for decode does not match the engine section: {problem}. " in str(failure.value)
    # The value of the same type restates it.
    rules.check_entry_identity(_entries(decode={key: compiled}), "decode", {key: compiled})


def test_identity_check_needs_an_entry_for_the_worker_type():
    rules.check_entry_identity(None, "decode", {"model": MODEL})
    rules.check_entry_identity([{"worker_type": "prefill", "model": "other/model"}], "decode", {"model": MODEL})
    with pytest.raises(AisPerfModelError, match="entry for prefill does not match the engine section"):
        rules.check_entry_identity([{"worker_type": "prefill", "model": "other/model"}], "prefill", {"model": MODEL})


def test_identity_check_reports_a_key_the_perf_model_does_not_have():
    entries = _entries(decode={"tp": 4, "tensor_parallel": 4})

    with pytest.raises(AisPerfModelError) as failure:
        rules.check_entry_identity(entries, "decode", {"tp": 1})

    assert str(failure.value) == (
        "engine.ais_perf_model entry for decode does not match the engine section: "
        "tp: entry has 4, the engine section gives 1; "
        "tensor_parallel: not a perf model field. "
        "Fix the engine section or drop the key from the entry."
    )


# --- Entry keys that no perf model field has --------------------------------------------------

UNKNOWN_KEY_ADVICE = (
    "Correct the key or drop it from the entry, or from the template file when 'from: <file>' brings it in."
)


@pytest.mark.parametrize(
    ("keys", "problems"),
    [
        ({"tensor_parallel": 4}, "tensor_parallel: not a perf model field"),
        ({"estimaton_mode": "op_level"}, "estimaton_mode: not a perf model field (did you mean estimation_mode?)"),
        (
            {"system_path": "/srv/systems", "tp": 1, "hardware": SYSTEM},
            "system_path: not a perf model field (did you mean systems_paths?); hardware: not a perf model field",
        ),
    ],
    ids=["no field like it", "misspelled estimator key", "several keys next to a real one"],
)
def test_predict_entry_key_that_is_not_a_perf_model_field_is_rejected_when_validated(keys, problems):
    raw = _engine("disaggregated", ais_perf_model=_entries(prefill=EST, decode={**EST, **keys}))
    message = f"engine.ais_perf_model entry for decode: {problems}. {UNKNOWN_KEY_ADVICE}"

    assert message in _rejected("predict", raw)
    with pytest.raises(AisPerfModelError) as failure:
        rules.lower_engine(raw, "predict")
    assert str(failure.value) == message


def test_predict_entry_key_that_a_template_file_brings_in_is_rejected_too(tmp_path):
    template = tmp_path / "estimator.yaml"
    template.write_text("estimation_mode: op_level\ntensor_parallel: 4\n")

    message = _rejected("predict", _engine("aggregated", ais_perf_model={"from": str(template)}))

    assert (
        f"engine.ais_perf_model entry for aggregated: tensor_parallel: not a perf model field. {UNKNOWN_KEY_ADVICE}"
    ) in message


def test_entry_for_a_worker_type_the_config_does_not_run_is_reported_before_its_keys():
    value = _entries(prefill=EST, decode=EST, aggregated={"tensor_parallel": 4})

    message = _rejected("predict", _engine("disaggregated", ais_perf_model=value))

    assert "has an entry for worker type aggregated, which this config does not run" in message
    assert "not a perf model field" not in message


def test_recommend_entry_key_that_is_not_a_perf_model_field_is_reported_as_one_recommend_cannot_check():
    engine = _engine("aggregated", ais_perf_model={**EST, "estimaton_mode": "auto"})

    message = _rejected("recommend", engine)

    assert (
        "engine.ais_perf_model entry for aggregated: estimaton_mode cannot be checked in a recommend config" in message
    )


# A Dynamo planner config can hold a mapping of another shape under the same name: its roles and
# the dataset profile of each. Here it reads as one entry with two keys that are not its own.
ROLES_AND_PROFILES = {"roles": {"decode": {"model": MODEL}}, "dataset_profiles": {"decode": "profile"}}


@pytest.mark.parametrize(
    ("kind", "mode", "problem"),
    [
        (
            "predict",
            "aggregated",
            "entry for aggregated: roles: not a perf model field; dataset_profiles: not a perf model field. "
            f"{UNKNOWN_KEY_ADVICE}",
        ),
        (
            "predict",
            "disaggregated",
            "entry for prefill: roles: not a perf model field; dataset_profiles: not a perf model field. "
            f"{UNKNOWN_KEY_ADVICE}",
        ),
        (
            "recommend",
            "aggregated",
            "entry for aggregated: roles, dataset_profiles cannot be checked in a recommend config. "
            "The recommend engine section does not state them (recommend chooses tp and similar values "
            "for each candidate). Drop these keys from the entry, or from the template file when "
            "'from: <file>' brings them in.",
        ),
        (
            "recommend",
            "both",
            "entry for aggregated: roles, dataset_profiles cannot be checked in a recommend config. "
            "The recommend engine section does not state them (recommend chooses tp and similar values "
            "for each candidate). Drop these keys from the entry, or from the template file when "
            "'from: <file>' brings them in.",
        ),
    ],
)
def test_mapping_of_roles_and_dataset_profiles_is_rejected_as_an_entry_with_two_keys_that_are_not_its_own(
    kind, mode, problem
):
    assert loader.load(ROLES_AND_PROFILES).to_data() == ROLES_AND_PROFILES
    engine = _engine(mode, ais_perf_model=ROLES_AND_PROFILES)
    message = f"engine.ais_perf_model {problem}"

    with pytest.raises(ValidationError) as rejected:
        VALIDATE[kind](engine)
    assert [error["msg"] for error in rejected.value.errors()] == [f"Value error, {message}"]
    with pytest.raises(AisPerfModelError) as failure:
        rules.lower_engine(engine, kind)
    assert str(failure.value) == message


# The two perf model fields that describe an AFD pool.
AFD_FIELDS = {"afd_routing_location": "a", "afd_pool_role": "f"}


@pytest.mark.parametrize("key", list(AFD_FIELDS))
def test_predict_entry_that_sets_an_afd_field_is_rejected_when_validated(key):
    raw = _engine("disaggregated", ais_perf_model=_entries(prefill=EST, decode={**EST, key: AFD_FIELDS[key]}))
    message = (
        f"engine.ais_perf_model entry for decode: {key} is set, and ais_perf_model does not cover AFD yet. "
        "Drop the key from the entry, or from the template file when 'from: <file>' brings it in."
    )

    assert message in _rejected("predict", raw)
    with pytest.raises(AisPerfModelError) as failure:
        rules.lower_engine(raw, "predict")
    assert str(failure.value) == message


def test_predict_entry_with_both_afd_fields_names_the_first_one():
    raw = _engine("aggregated", ais_perf_model={**EST, "afd_pool_role": "f", "afd_routing_location": "a"})

    message = _rejected("predict", raw)

    assert "entry for aggregated: afd_routing_location is set, and ais_perf_model does not cover AFD yet" in message
    assert "afd_pool_role" not in message


def test_predict_entry_with_a_null_afd_field_is_the_entry_without_it():
    with_null = _engine("aggregated", ais_perf_model={**EST, **dict.fromkeys(AFD_FIELDS)})
    without = _engine("aggregated", ais_perf_model=EST)

    assert _dump(_predict(with_null)) == _dump(_predict(without))


@pytest.mark.parametrize("key", list(AFD_FIELDS))
def test_recommend_entry_that_sets_an_afd_field_is_reported_as_one_recommend_cannot_check(key):
    engine = _engine("aggregated", ais_perf_model={**EST, key: AFD_FIELDS[key]})

    message = _rejected("recommend", engine)

    assert f"engine.ais_perf_model entry for aggregated: {key} cannot be checked in a recommend config" in message


def test_perf_model_field_names_are_the_fields_of_the_canonical_config():
    canonical = [field.name for field in dataclasses.fields(ForwardPassPerfModelConfig)]

    assert sorted(rules.PERF_MODEL_FIELD_NAMES) == sorted(canonical)


def test_afd_field_names_are_the_afd_fields_of_the_canonical_config():
    canonical = [field.name for field in dataclasses.fields(ForwardPassPerfModelConfig)]

    assert sorted(rules._AFD_FIELD_NAMES) == sorted(name for name in canonical if name.startswith("afd_"))


def test_transfer_kinds_and_presets_are_those_of_the_perf_model():
    canonical = {name: sorted(kind.value for kind in kinds) for name, kinds in TRANSFER_PRESETS.items()}

    assert list(rules.TRANSFER_KINDS) == [kind.value for kind in TransferKind]
    assert {name: sorted(kinds) for name, kinds in rules.TRANSFER_PRESETS.items()} == canonical


def _policy_texts() -> list[str]:
    """Kinds, presets and words that are neither, alone and joined by commas in twos and threes."""
    names = [*(kind.value for kind in TransferKind), *TRANSFER_PRESETS, "bogus"]
    words = [*names, "", " ", "XSHAPE", " xop "]
    # Close to a kind and not one: a blank or another separator inside, and a long s, a letter outside ASCII.
    words += ["x shape", "xshape;xquant", "xshape xquant", "x\u017fhape"]
    # White space that is not a blank: at the end, at the start and after a comma.
    words += ["xshape\n", "\tbalanced", "xshape,\txquant"]
    texts = set(words)
    for first, second in itertools.product(words, repeat=2):
        texts.update({f"{first},{second}", f"{first}, {second}", f"{first},{second},", f",{first}"})
    texts.update(",".join(three) for three in itertools.product(names, repeat=3))
    return sorted(texts)


def test_transfer_policy_text_is_read_the_way_the_perf_model_reads_it():
    def same(left, right):
        return rules.same_setting(TimingConfig, "transfer_policy", left, right)

    names = [kind.value for kind in TransferKind]
    every_set_of_kinds = [
        list(kinds) for size in range(len(names) + 1) for kinds in itertools.combinations(names, size)
    ]
    read, rejected = 0, 0
    for text in _policy_texts():
        try:
            canonical = sorted(kind.value for kind in resolve_transfer_policy(text))
        except ValueError:
            # A text the perf model rejects is the same setting as itself and as no set of kinds.
            rejected += 1
            assert same(text, text), text
            assert not any(same(text, kinds) for kinds in every_set_of_kinds), text
        else:
            read += 1
            assert [sorted(kinds) for kinds in every_set_of_kinds if same(text, kinds)] == [canonical], text
    assert read and rejected


# --- The field on the engine models -----------------------------------------------------------


def test_pydantic_runs_the_last_declared_before_validator_first():
    # The next test relies on this order.
    seen = []

    class Base(BaseModel):
        @model_validator(mode="before")
        @classmethod
        def _first_declared(cls, value):
            seen.append("first declared")
            return value

        @model_validator(mode="before")
        @classmethod
        def _last_declared(cls, value):
            seen.append("last declared")
            return value

    class Child(Base):
        @model_validator(mode="before")
        @classmethod
        def _child(cls, value):
            seen.append("child")
            return value

    Child.model_validate({})

    assert seen == ["child", "last declared", "first declared"]
    assert list(Child.__pydantic_decorators__.model_validators) == ["_first_declared", "_last_declared", "_child"]


@pytest.mark.parametrize("engine_cls", [EnginePredictionConfig, EngineRecommendationConfig])
def test_lowering_runs_before_every_other_validator_of_the_engine_model(engine_cls):
    # Every other validator reads the existing estimator fields, so it must see them already written.
    early = [
        name
        for name, validator in engine_cls.__pydantic_decorators__.model_validators.items()
        if validator.info.mode != "after"
    ]

    assert early[-1] == "_lower_ais_perf_model"


@pytest.mark.parametrize("config_cls", [CorePredictionConfig, CoreRecommendationConfig])
def test_schema_gains_one_optional_property_and_dumps_never_have_it(config_cls):
    engine_cls = config_cls.model_fields["engine"].annotation
    schema = engine_cls.model_json_schema(mode="validation")

    assert schema["properties"]["ais_perf_model"] == {
        "anyOf": [
            {"additionalProperties": True, "type": "object"},
            {"items": {"additionalProperties": True, "type": "object"}, "type": "array"},
            {"type": "null"},
        ],
        "default": None,
        "title": "Ais Perf Model",
    }
    assert "ais_perf_model" not in schema.get("required", [])
    assert "ais_perf_model" not in engine_cls.model_json_schema(mode="serialization")["properties"]

    kind = "predict" if config_cls is CorePredictionConfig else "recommend"
    config = VALIDATE[kind](_engine("aggregated", ais_perf_model=EST))
    assert config.engine.ais_perf_model == [{"worker_type": "aggregated", **EST}]
    assert "ais_perf_model" not in config.model_dump()["engine"]
    assert "ais_perf_model" not in config.model_dump(mode="json", exclude_none=True)["engine"]
    assert "ais_perf_model" not in config.model_dump_json()
    assert config_cls.model_validate_json(config.model_dump_json()).engine.ais_perf_model is None


@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_repr_of_a_config_reads_as_it_did_before_the_key_existed(kind):
    # Logs and saved comparisons hold the repr of existing configs, so the key never appears in it.
    old = VALIDATE[kind](_engine("aggregated", **EST))
    new = VALIDATE[kind](_engine("aggregated", ais_perf_model=EST))

    assert "ais_perf_model" not in repr(old)
    assert "estimation_mode='op_level'" in repr(old.engine)
    assert repr(new) == repr(old)


# --- A key written twice in the text of a config file -----------------------------------------

REPEATED_KEYS = [
    pytest.param(
        "engine:\n"
        "  model: a\n"
        "  ais_perf_model:\n"
        "    estimation_mode: op_level\n"
        "    fallback_policy: allow\n"
        "    estimation_mode: auto\n",
        "the key 'estimation_mode' is written twice (lines 4 and 6)",
        id="one config",
    ),
    pytest.param(
        "engine:\n"
        "  ais_perf_model:\n"
        "    - worker_type: prefill\n"
        "      estimation_mode: auto\n"
        "    - worker_type: decode\n"
        "      estimation_mode: op_level\n"
        "      estimation_mode: auto\n",
        "the key 'estimation_mode' is written twice (lines 6 and 7)",
        id="an entry of a list",
    ),
    pytest.param(
        "engine:\n"
        "  ais_perf_model:\n"
        "    estimator_config:\n"
        "      correction:\n"
        "        mode: fixed\n"
        "        mode: learned\n",
        "the key 'mode' is written twice (lines 5 and 6)",
        id="a mapping inside estimator_config",
    ),
    pytest.param(
        "engine:\n  ais_perf_model: {estimation_mode: op_level, estimation_mode: auto}\n",
        "the key 'estimation_mode' is written twice (line 2)",
        id="a mapping written on one line",
    ),
    pytest.param(
        'engine:\n  ais_perf_model:\n    estimation_mode: op_level\n    "estimation_mode": auto\n',
        "the key 'estimation_mode' is written twice (lines 3 and 4)",
        id="a plain and a quoted spelling",
    ),
    pytest.param(
        "engine:\n  ais_perf_model:\n    tp: 1\n    tp: 2\n    tp: 4\n",
        "the key 'tp' is written twice (lines 3 and 4)",
        id="three times",
    ),
    pytest.param(
        "shared: &shared {fallback_policy: allow, fallback_policy: deny}\nengine:\n  ais_perf_model: *shared\n",
        "the key 'fallback_policy' is written twice (line 1)",
        id="a mapping named by an alias",
    ),
    pytest.param(
        "shared: &shared\n"
        "  fallback_policy: allow\n"
        "  fallback_policy: deny\n"
        "engine:\n"
        "  ais_perf_model:\n"
        "    <<: *shared\n"
        "    estimation_mode: auto\n",
        "the key 'fallback_policy' is written twice (lines 2 and 3)",
        id="a mapping that is merged in",
    ),
    pytest.param(
        "engine:\n"
        "  ais_perf_model:\n"
        "    estimation_mode: auto\n"
        "    estimator_config:\n"
        "      correction: {mode: fixed, mode: learned}\n"
        "    estimation_mode: op_level\n",
        "the key 'mode' is written twice (line 5)",
        id="two keys: the first repeat in the file is named",
    ),
]


@pytest.mark.parametrize(("text", "problem"), REPEATED_KEYS)
def test_key_written_twice_inside_the_value_is_an_error(text, problem):
    # The YAML reader takes the text and keeps the last value, so the mapping does not show the repeat.
    assert isinstance(yaml.safe_load(text)["engine"]["ais_perf_model"], (dict, list))

    with pytest.raises(AisPerfModelError) as failure:
        rules.reject_repeated_keys(text, "predict", Path("configs") / "predict.yaml")

    assert str(failure.value) == f"engine.ais_perf_model in configs/predict.yaml: {problem}; write it once"


@pytest.mark.parametrize(
    ("kind", "section"), [("predict", "engine"), ("recommend", "engine"), ("search", "search_space")]
)
def test_key_written_twice_is_looked_for_in_the_section_of_the_kind_of_config(kind, section):
    text = f"{section}:\n  ais_perf_model:\n    estimation_mode: op_level\n    estimation_mode: auto\n"
    other = "search" if section == "engine" else "predict"

    with pytest.raises(AisPerfModelError, match=f"^{section}.ais_perf_model in search.yaml: the key 'estimation_mode'"):
        rules.reject_repeated_keys(text, kind, "search.yaml")
    rules.reject_repeated_keys(text, other, "search.yaml")
    with pytest.raises(ValueError, match="unknown config kind 'replay'"):
        rules.reject_repeated_keys(text, "replay", "search.yaml")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# nothing but a comment\n",
        "- engine\n- ais_perf_model\n",
        "engine\n",
        "traffic: {load: 1, load: 2}\n",
        "engine:\n",
        "engine: [ais_perf_model, ais_perf_model]\n",
        "engine:\n  model: a\n  model: b\n",
        "engine:\n  ais_perf_model:\n",
        "engine:\n  ais_perf_model: perf-model.yaml\n",
        "engine:\n  model: a\n  model: b\n  ais_perf_model: {estimation_mode: auto}\ntraffic: {load: 1, load: 2}\n",
        "search_space:\n  ais_perf_model: {estimation_mode: op_level, estimation_mode: auto}\n",
        "engine:\n  ais_perf_model:\n    - {worker_type: prefill, tp: 1}\n    - {worker_type: decode, tp: 1}\n",
        "base: &base {estimation_mode: op_level, tp: 1}\n"
        "engine:\n  ais_perf_model:\n    <<: *base\n    estimation_mode: auto\n",
        "base: &base {estimation_mode: op_level}\n"
        "engine:\n  ais_perf_model:\n"
        "    - {worker_type: prefill, <<: *base}\n    - {worker_type: decode, <<: *base}\n",
        "one: &one {tp: 1}\ntwo: &two {tp: 2}\nengine:\n  ais_perf_model:\n    <<: *one\n    <<: *two\n",
        "engine:\n  ais_perf_model: &value\n    estimator_config: *value\n",
        "engine:\n  ais_perf_model: {estimation_mode: op_level, estimation_mode: auto\n",
        "engine: {}\n---\nengine:\n  ais_perf_model: {estimation_mode: op_level, estimation_mode: auto}\n",
        "engine:\n  ais_perf_model: " + "[" * 100_000 + "\n",
        "a0: &a0 {}\n" + "".join(f"a{i}: &a{i} {{<<: *a{i - 1}}}\n" for i in range(1, 3000)) + "engine: {<<: *a2999}\n",
        # Followed along every path, the first mapping would be looked at 2 ** 40 times.
        "a0: &a0 {}\n"
        + "".join(f"a{i}: &a{i} {{<<: [*a{i - 1}, *a{i - 1}]}}\n" for i in range(1, 41))
        + "engine: {<<: *a40}\n",
        "ais_perf_model: {tp: 1}\nais_perf_model: {tp: 2}\nengine: {}\n",
        "engine:\n  workers:\n    ais_perf_model: {tp: 1}\n    ais_perf_model: {tp: 2}\n",
        "other:\n  engine: {ais_perf_model: {tp: 1}}\n  engine: {ais_perf_model: {tp: 2}}\n",
        "search_space:\n  ais_perf_model: {tp: 1}\n  ais_perf_model: {tp: 2}\nsearch_space: {}\n",
        "engine: [ais_perf_model: {tp: 1}, ais_perf_model: {tp: 2}]\nengine: [ais_perf_model: {tp: 1}]\n",
        "engine:\n  ais_perf_model: {tp: 1}\n  ais_perf_model: {tp: 2\n",
        "engine: {}\n---\nengine:\n  ais_perf_model: {tp: 1}\n  ais_perf_model: {tp: 2}\nengine: {}\n",
        "engine:\n  ais_perf_model: {tp: 1}\n  ais_perf_model: " + "[" * 100_000 + "\n",
        "a0: &a0 {}\n"
        + "".join(f"a{i}: &a{i} {{<<: *a{i - 1}}}\n" for i in range(1, 3000))
        + "engine: {<<: *a2999}\nengine: {<<: *a2999}\n",
        "a0: &a0 {}\n"
        + "".join(f"a{i}: &a{i} {{<<: [*a{i - 1}, *a{i - 1}]}}\n" for i in range(1, 41))
        + "engine: {<<: *a40, ais_perf_model: {tp: 1}}\n",
        "engine: {model: a}\n" * 2000,
        # A key of another type is another key for the reader, also when its text is the same.
        "engine:\n  ais_perf_model: {tp: 1}\n  !!binary ais_perf_model: x\n",
        "engine:\n  ais_perf_model: {tp: 1}\n!!null engine:\n  ais_perf_model: {tp: 2}\n",
    ],
    ids=[
        "an empty file",
        "a comment",
        "a list",
        "a word",
        "no section",
        "a null section",
        "a section that is a list",
        "no key",
        "a null key",
        "a file name",
        "repeats outside the value",
        "the section of another kind of config",
        "one key in two entries",
        "a merged key and the same key written",
        "one mapping merged into two entries",
        "two merges",
        "a value that holds itself",
        "text that is not YAML",
        "two documents",
        "text nested too deep to read",
        "a chain of merges too long to follow",
        "one mapping merged along many paths",
        "the key twice outside a section",
        "the key twice below the section",
        "the section twice inside another mapping",
        "the key and the section twice for another kind of config",
        "the key twice in sections that are lists",
        "the key twice in text that is not YAML",
        "the key and the section twice in a second document",
        "the key twice, the last nested too deep to read",
        "a chain of merges too long to follow in a section written twice",
        "one mapping merged along many paths next to the key",
        "a section written many times without the key",
        "the key next to a key of another type with the same text",
        "the section next to a key of another type with the same text",
    ],
)
def test_text_without_a_repeat_inside_the_value_is_left_to_the_reader(text):
    assert rules.reject_repeated_keys(text, "predict", "predict.yaml") is None


@pytest.mark.parametrize(
    ("text", "repeat"),
    [
        ("{a: 1, b: 1, b: 2, a: 2}", ("b", 1, 1)),
        ("outer:\n  inner: {k: 1}\n  inner: {k: 2}\n", ("inner", 2, 3)),
        ("- {k: 1}\n- {k: 2}\n", None),
        ("tp", None),
        ("", None),
        ("{a: 1, a: 2", None),
        ("[" * 100_000, None),
    ],
    ids=[
        "two repeats on one line",
        "a repeat below the top",
        "one key in two items",
        "a word",
        "nothing",
        "text that is not YAML",
        "text nested too deep to read",
    ],
)
def test_repeated_key_in_names_the_repeat_that_comes_first_in_the_text(text, repeat):
    assert "repeated_key_in" in rules.__all__
    assert rules.repeated_key_in(text) == repeat


def test_mapping_that_an_alias_writes_as_the_section_many_times_is_read_once(monkeypatch):
    looked_for = []
    written = rules._written

    def counted(node, name):
        looked_for.append(name)
        return written(node, name)

    monkeypatch.setattr(rules, "_written", counted)
    text = "base: &base {model: a}\n" + "engine: *base\n" * 500

    assert rules.reject_repeated_keys(text, "predict", "predict.yaml") is None
    # Once for the sections of the file and once for the one mapping that all of them are.
    assert looked_for == ["engine", "ais_perf_model"]


REPEAT, ONCE = "{estimation_mode: op_level, estimation_mode: auto}", "{estimation_mode: op_level}"
# What the reader makes of each value: the one written with a repeat reads as ``auto``.
READ_AS = {REPEAT: {"estimation_mode": "auto"}, ONCE: {"estimation_mode": "op_level"}, None: None}


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        (f"base: &base\n  ais_perf_model: {REPEAT}\nengine:\n  <<: *base\n  model: a\n", REPEAT),
        (f"base: &base\n  ais_perf_model: {REPEAT}\nengine:\n  <<: *base\n  ais_perf_model: {ONCE}\n", ONCE),
        (f"base: &base\n  ais_perf_model: {ONCE}\n  ais_perf_model: {REPEAT}\nengine:\n  <<: *base\n", REPEAT),
        (f"base: &base\n  ais_perf_model: {REPEAT}\n  ais_perf_model: {ONCE}\nengine:\n  <<: *base\n", ONCE),
        (f"base: &base\n  engine:\n    ais_perf_model: {REPEAT}\n<<: *base\n", REPEAT),
        (f"base: &base\n  engine:\n    ais_perf_model: {REPEAT}\n<<: *base\nengine:\n  ais_perf_model: {ONCE}\n", ONCE),
        (f"base: &base\n  engine:\n    ais_perf_model: {REPEAT}\n<<: *base\nengine:\n  model: a\n", None),
        (
            f"base: &base\n  engine:\n    ais_perf_model: {REPEAT}\n  engine:\n    ais_perf_model: {ONCE}\n<<: *base\n",
            ONCE,
        ),
        (f"a: &a\n  ais_perf_model: {REPEAT}\nb: &b\n  <<: *a\nengine:\n  <<: *b\n", REPEAT),
        (f"a: &a\n  ais_perf_model: {ONCE}\nb: &b\n  ais_perf_model: {REPEAT}\nengine:\n  <<: [*a, *b]\n", ONCE),
        (f"a: &a\n  ais_perf_model: {ONCE}\nb: &b\n  ais_perf_model: {REPEAT}\nengine:\n  <<: [*b, *a]\n", REPEAT),
        (
            f"a: &a\n  ais_perf_model: {ONCE}\nb: &b\n  ais_perf_model: {REPEAT}\nengine:\n  <<: *b\n  <<: *a\n",
            ONCE,
        ),
        (
            f"a: &a\n  ais_perf_model: {ONCE}\nb: &b\n  ais_perf_model: {REPEAT}\nengine:\n  <<: *a\n  <<: *b\n",
            REPEAT,
        ),
        ("engine:\n  model: a\nengine:\n  model: b\n", None),
        (f"base: &base\n  ais_perf_model: {REPEAT}\nengine:\n  model: a\nengine:\n  <<: *base\n", REPEAT),
        (f"base: &base\n  ais_perf_model: {REPEAT}\nengine:\n  <<: *base\nengine:\n  model: a\n", None),
        (f"base: &base\n  ais_perf_model: {REPEAT}\nengine:\n  <<: *base\nengine:\n  <<: *base\n", REPEAT),
    ],
    ids=[
        "the value merged into the section",
        "a merged value and the key written",
        "the key twice in the merged mapping, the repeat in the last",
        "the key twice in the merged mapping, the repeat in the first",
        "the section merged into the file",
        "a merged section and the section written",
        "a merged section and the section written without the key",
        "the section twice in the merged mapping",
        "a merge inside a merge",
        "a list of merges, the repeat in the second",
        "a list of merges, the repeat in the first",
        "two merges, the repeat in the first",
        "two merges, the repeat in the last",
        "the section twice, without the key",
        "the section twice, the value merged into the last",
        "the section twice, the value merged into the first",
        "the section twice, the value merged into both",
    ],
)
def test_the_value_that_is_checked_is_the_one_the_yaml_reader_keeps(text, kept):
    # No section in these files writes ``ais_perf_model`` twice, and none that writes it is written twice.
    # A merge brings in what the file does not write, and a repeated section that never writes the key
    # keeps its last value. Which value that leaves is shown here by the reader itself.
    assert yaml.safe_load(text)["engine"].get("ais_perf_model") == READ_AS[kept]

    if kept == REPEAT:
        with pytest.raises(AisPerfModelError, match="the key 'estimation_mode' is written twice"):
            rules.reject_repeated_keys(text, "predict", "predict.yaml")
    else:
        assert rules.reject_repeated_keys(text, "predict", "predict.yaml") is None


# --- The key itself, or its section, written twice --------------------------------------------

AUTO = "{estimation_mode: auto}"
# Written for a predict config. The other kinds of config read the same text with their own section name.
WRITTEN_TWICE = [
    pytest.param(
        f"engine:\n  model: a\n  ais_perf_model: {ONCE}\n  backend: vllm\n  ais_perf_model: {AUTO}\n",
        "the key 'ais_perf_model' is written twice (lines 3 and 5)",
        id="the key twice",
    ),
    pytest.param(
        f"engine: {{model: a, ais_perf_model: {ONCE}, ais_perf_model: {AUTO}}}\n",
        "the key 'ais_perf_model' is written twice (line 1)",
        id="the key twice on one line",
    ),
    pytest.param(
        f"engine: {{\n  ais_perf_model: {ONCE},\n  model: a,\n  ais_perf_model: {AUTO},\n}}\n",
        "the key 'ais_perf_model' is written twice (lines 2 and 4)",
        id="the key twice between braces",
    ),
    pytest.param(
        f'engine:\n  ais_perf_model: {ONCE}\n  "ais_perf_model": {AUTO}\n',
        "the key 'ais_perf_model' is written twice (lines 2 and 3)",
        id="the key in a plain and a quoted spelling",
    ),
    pytest.param(
        f"engine:\n  ais_perf_model: {ONCE}\n  ais_perf_model: {ONCE}\n  model: a\n  ais_perf_model: {AUTO}\n",
        "the key 'ais_perf_model' is written twice (lines 2 and 5)",
        id="the key three times: the first and the last line are named",
    ),
    pytest.param(
        f"engine:\n  ais_perf_model:\n  ais_perf_model: {ONCE}\n",
        "the key 'ais_perf_model' is written twice (lines 2 and 3)",
        id="the key twice, the first without a value",
    ),
    pytest.param(
        f"engine:\n  ais_perf_model: &value {ONCE}\n  ais_perf_model: *value\n",
        "the key 'ais_perf_model' is written twice (lines 2 and 3)",
        id="the key twice with one value",
    ),
    pytest.param(
        f"base: &base\n  ais_perf_model: {ONCE}\n  ais_perf_model: {AUTO}\nengine: *base\n",
        "the key 'ais_perf_model' is written twice (lines 2 and 3)",
        id="the key twice in a section named by an alias",
    ),
    pytest.param(
        f"a: &a\n  ais_perf_model: {AUTO}\nengine:\n  <<: *a\n  ais_perf_model: {ONCE}\n  ais_perf_model: {AUTO}\n",
        "the key 'ais_perf_model' is written twice (lines 5 and 6)",
        id="the key twice next to a merge that brings it in",
    ),
    pytest.param(
        f"engine:\n  ais_perf_model: {ONCE}\ntraffic: {{}}\nengine:\n  model: a\n",
        "the key 'engine' is written twice (lines 1 and 4)",
        id="the section twice, the key in the first",
    ),
    pytest.param(
        f"engine:\n  model: a\ntraffic: {{}}\nengine:\n  ais_perf_model: {ONCE}\n",
        "the key 'engine' is written twice (lines 1 and 4)",
        id="the section twice, the key in the last",
    ),
    pytest.param(
        f"engine:\n  ais_perf_model: {ONCE}\ntraffic: {{}}\nengine:\n  ais_perf_model: {AUTO}\n",
        "the key 'engine' is written twice (lines 1 and 4)",
        id="the section twice, the key in both",
    ),
    pytest.param(
        f"engine:\n  model: a\nengine:\n  ais_perf_model: {ONCE}\nengine:\n  model: b\n",
        "the key 'engine' is written twice (lines 1 and 5)",
        id="the section three times, the key in the second: the first and the last line are named",
    ),
    pytest.param(
        f"{{engine: {{ais_perf_model: {ONCE}}}, engine: {{model: a}}}}\n",
        "the key 'engine' is written twice (line 1)",
        id="the section twice on one line",
    ),
    pytest.param(
        f"engine:\nengine:\n  ais_perf_model: {ONCE}\n",
        "the key 'engine' is written twice (lines 1 and 2)",
        id="the section twice, the first without a value",
    ),
    pytest.param(
        f"engine:\n  ais_perf_model: {ONCE}\nengine: 7\n",
        "the key 'engine' is written twice (lines 1 and 3)",
        id="the section twice, the last not a mapping",
    ),
    pytest.param(
        f"base: &base\n  ais_perf_model: {ONCE}\nengine: *base\nengine: *base\n",
        "the key 'engine' is written twice (lines 3 and 4)",
        id="the section twice, named by one alias",
    ),
    pytest.param(
        f"base: &base\n  engine: {{model: a}}\n<<: *base\nengine:\n  ais_perf_model: {ONCE}\nengine:\n  model: b\n",
        "the key 'engine' is written twice (lines 4 and 6)",
        id="the section twice next to a merge that brings it in",
    ),
    pytest.param(
        f"engine:\n  ais_perf_model: {ONCE}\n  ais_perf_model: {AUTO}\nengine:\n  model: a\n",
        "the key 'engine' is written twice (lines 1 and 4)",
        id="the section twice and the key twice: the section is named",
    ),
    pytest.param(
        f"engine:\n  ais_perf_model: {REPEAT}\n  ais_perf_model: {ONCE}\n",
        "the key 'ais_perf_model' is written twice (lines 2 and 3)",
        id="the key twice and a repeat in its first value: the key is named",
    ),
    pytest.param(
        f"engine:\n  ais_perf_model: {ONCE}\n  ais_perf_model: {REPEAT}\n",
        "the key 'ais_perf_model' is written twice (lines 2 and 3)",
        id="the key twice and a repeat in its last value: the key is named",
    ),
    pytest.param(
        f"engine:\n  ais_perf_model: {REPEAT}\nengine:\n  ais_perf_model: {REPEAT}\n",
        "the key 'engine' is written twice (lines 1 and 3)",
        id="the section twice and a repeat in each value: the section is named",
    ),
]


@pytest.mark.parametrize(
    ("kind", "section"), [("predict", "engine"), ("recommend", "engine"), ("search", "search_space")]
)
@pytest.mark.parametrize(("text", "problem"), WRITTEN_TWICE)
def test_key_or_its_section_written_twice_is_an_error(text, problem, kind, section):
    text, problem = text.replace("engine", section), problem.replace("engine", section)
    # The YAML reader takes the text and keeps one value, so the mapping does not show the other.
    assert isinstance(yaml.safe_load(text), dict)

    with pytest.raises(AisPerfModelError) as failure:
        rules.reject_repeated_keys(text, kind, Path("configs") / "config.yaml")

    assert str(failure.value) == f"{section}.ais_perf_model in configs/config.yaml: {problem}; write it once"


def test_repeat_inside_the_value_is_reported_once_the_key_and_its_section_are_written_once():
    # The repeats are looked for from the outside in, so mending the one that is named shows the next one.
    inside = "the key 'estimation_mode' is written twice (line 2)"
    texts = {
        f"engine:\n  ais_perf_model: {REPEAT}\n  ais_perf_model: {REPEAT}\nengine:\n  model: a\n": (
            "the key 'engine' is written twice (lines 1 and 4)"
        ),
        f"engine:\n  ais_perf_model: {REPEAT}\n  ais_perf_model: {REPEAT}\n": (
            "the key 'ais_perf_model' is written twice (lines 2 and 3)"
        ),
        f"engine:\n  ais_perf_model: {REPEAT}\n": inside,
    }

    for text, problem in texts.items():
        with pytest.raises(AisPerfModelError) as failure:
            rules.reject_repeated_keys(text, "predict", "predict.yaml")
        assert str(failure.value) == f"engine.ais_perf_model in predict.yaml: {problem}; write it once"
    assert rules.reject_repeated_keys(f"engine:\n  ais_perf_model: {ONCE}\n", "predict", "predict.yaml") is None


# --- Paths written in a config file -----------------------------------------------------------


def test_anchor_paths_starts_relative_paths_at_the_folder_of_the_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "configs"
    value = {"from": "estimator.yaml", "systems_paths": ["data", "default"], "estimation_mode": "op_level"}
    raw = {"engine": _engine("aggregated", ais_perf_model=value), "traffic": TRAFFIC}
    before = copy.deepcopy(raw)

    anchored = rules.anchor_paths(raw, folder, "predict")

    assert raw == before
    assert anchored["traffic"] is raw["traffic"]
    assert anchored["engine"] == {
        **raw["engine"],
        "ais_perf_model": {
            "from": str(folder / "estimator.yaml"),
            "systems_paths": [str(folder / "data"), "default"],
            "estimation_mode": "op_level",
        },
    }


def test_anchored_config_means_the_same_as_the_old_field_with_the_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "configs"
    raw = {"engine": _engine("aggregated", ais_perf_model={"systems_paths": ["data"]})}

    anchored = rules.anchor_paths(raw, folder, "recommend")

    old = _engine("aggregated", systems_paths=[str(folder / "data")])
    assert _dump(_recommend(anchored["engine"])) == _dump(_recommend(old))


def test_anchor_paths_reads_the_search_space_of_a_search_config(tmp_path):
    raw = {"search_space": {"ais_perf_model": _entries(decode={"systems_paths": ["data"]}), "backend": ["vllm"]}}

    anchored = rules.anchor_paths(raw, tmp_path, "search")

    assert anchored["search_space"] == {
        "ais_perf_model": _entries(decode={"systems_paths": [str(tmp_path / "data")]}),
        "backend": ["vllm"],
    }


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "text",
        {},
        {"engine": None},
        {"engine": {"model": MODEL}},
        {"engine": {"ais_perf_model": None}},
        {"engine": {"ais_perf_model": "estimator.yaml"}},
        {"engine": {"ais_perf_model": 3}},
        {"search_space": {"ais_perf_model": {"systems_paths": ["data"]}}},
    ],
    ids=["null", "string", "no section", "null section", "no key", "null key", "string key", "number", "other section"],
)
def test_anchor_paths_leaves_everything_else_for_validation(raw, tmp_path):
    assert rules.anchor_paths(raw, tmp_path, "predict") is raw


def test_anchor_paths_reports_a_value_the_loader_rejects(tmp_path):
    with pytest.raises(AisPerfModelError, match="the config is empty"):
        rules.anchor_paths({"engine": {"ais_perf_model": {}}}, tmp_path, "predict")
    with pytest.raises(ValueError, match="unknown config kind 'replay'"):
        rules.anchor_paths({"engine": {}}, tmp_path, "replay")


# --- Old fields: which ones a config states, and the warning ----------------------------------

LEGACY = _old_spelling(
    "disaggregated",
    {"estimation_mode": "op_level", "systems_path": "default", "fallback_policy": "deny", "transfer_policy": None},
    {
        "prefill": {"database_mode": "hybrid", "forward_model": "fpm", "fpm_parquet_path": "fpm.parquet"},
        "decode": {"type": "default", "estimator_config": {}},
    },
)
LEGACY_PATHS = [
    "engine.estimation_mode",
    "engine.systems_path",
    "engine.workers.prefill.timing.database_mode",
    "engine.workers.prefill.timing.forward_model",
    "engine.workers.prefill.timing.fpm_parquet_path",
    "engine.workers.decode.timing.estimator_config",
]


@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_written_old_keys_lists_the_fields_the_author_stated(kind):
    # fallback_policy has its default and transfer_policy is null: neither is a choice.
    assert rules.written_old_keys({"engine": LEGACY, "traffic": TRAFFIC}, kind) == LEGACY_PATHS


@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_a_dump_full_of_defaults_states_no_old_field(kind):
    dumped = VALIDATE[kind](_engine("aggregated")).model_dump(mode="json")

    assert rules.written_old_keys(dumped, kind) == []
    assert rules.written_old_keys({"engine": _engine("aggregated", ais_perf_model=EST)}, kind) == []


def test_written_old_keys_skips_workers_the_mode_does_not_run():
    engine = _engine("aggregated")
    engine["workers"]["decode"] = {"timing": {"estimation_mode": "op_level"}}

    assert rules.written_old_keys({"engine": engine}, "predict") == []


@pytest.mark.parametrize(("kind", "changes", "reason"), UNCOVERED)
def test_old_fields_of_a_config_the_key_cannot_describe_are_not_reported(kind, changes, reason):
    engine = {**_engine("both" if kind == "recommend" else "aggregated", **EST), **changes}

    assert rules.written_old_keys({"engine": engine}, kind) == []


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {},
        {"engine": "text"},
        {"engine": {**LEGACY, "workers": []}},
        {"engine": {**LEGACY, "mode": ["disaggregated"]}},
        {"engine": {**LEGACY, "workers": {"prefill": {"timing": "default"}}}},
    ],
    ids=[
        "null",
        "list",
        "no engine",
        "engine is a string",
        "workers is a list",
        "mode is a list",
        "timing is a string",
    ],
)
def test_written_old_keys_never_fails_on_a_mapping_validation_will_reject(raw):
    assert rules.written_old_keys(raw, "predict") == []
    assert rules.written_old_keys(raw, "recommend") == []


def test_written_old_keys_asks_the_search_rules_for_a_search_config(monkeypatch):
    seen = []

    def written_old_keys_search(space):
        seen.append(space)
        return ["estimation_mode", "prefill_forward_model"]

    stand_in = types.ModuleType("aisim.sweeper.ais_perf_model")
    stand_in.written_old_keys_search = written_old_keys_search
    monkeypatch.setitem(sys.modules, "aisim.sweeper.ais_perf_model", stand_in)
    space = {"estimation_mode": ["op_level"]}

    found = rules.written_old_keys({"search_space": space}, "search")

    assert found == ["search_space.estimation_mode", "search_space.prefill_forward_model"]
    assert seen == [space]
    assert rules.written_old_keys({"search_space": None}, "search") == []
    with pytest.raises(ValueError, match="unknown config kind 'replay'"):
        rules.written_old_keys({"engine": {}}, "replay")


def _warnings_from(call) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return caught


def test_warn_old_keys_gives_one_summary_that_names_the_fields_and_the_fix():
    raw = {"engine": LEGACY}

    caught = _warnings_from(lambda: rules.warn_old_keys(raw, "predict", source="configs/predict.yaml"))

    assert [warning.category for warning in caught] == [AisPerfModelDeprecationWarning]
    assert str(caught[0].message) == (
        f"{loader.DEPRECATION_PREFIX}{', '.join(LEGACY_PATHS)} are deprecated and will be removed in the next release; "
        "use engine.ais_perf_model (run 'aisim perf-model migrate configs/predict.yaml' to convert the file)"
    )


@pytest.mark.parametrize(
    ("source", "named"),
    [
        ("my configs/predict.yaml", "'my configs/predict.yaml'"),
        ("configs/it's.yaml", "'configs/it'\"'\"'s.yaml'"),
        ("configs/$(whoami);.yaml", "'configs/$(whoami);.yaml'"),
        (None, "<config file>"),
    ],
    ids=["space", "quote", "shell syntax", "no file known"],
)
def test_warn_old_keys_names_the_file_so_that_a_shell_reads_the_command_as_written(source, named):
    caught = _warnings_from(lambda: rules.warn_old_keys({"engine": LEGACY}, "predict", source=source))

    assert str(caught[0].message).endswith(f"(run 'aisim perf-model migrate {named}' to convert the file)")


def test_warn_old_keys_points_at_an_override_when_the_file_has_no_old_field():
    file_raw = {"engine": _engine("aggregated")}
    raw = {"engine": _engine("aggregated", estimation_mode="op_level")}

    from_override = _warnings_from(lambda: rules.warn_old_keys(raw, "recommend", source="r.yaml", file_raw=file_raw))
    from_file = _warnings_from(lambda: rules.warn_old_keys(raw, "recommend", source="r.yaml", file_raw=raw))

    assert str(from_override[0].message).endswith(
        "engine.estimation_mode is deprecated and will be removed in the next release; "
        "use engine.ais_perf_model (set engine.ais_perf_model.<key> with --set instead)"
    )
    assert str(from_file[0].message).endswith("(run 'aisim perf-model migrate r.yaml' to convert the file)")


@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_warn_old_keys_says_to_remove_old_fields_that_repeat_the_key(kind):
    # migrate does not merge old fields into a key that is already set, so the hint must not send the author there.
    engine = {**_old_spelling("aggregated", {"estimation_mode": "op_level"}, {}), "ais_perf_model": EST}
    VALIDATE[kind](engine)

    caught = _warnings_from(lambda: rules.warn_old_keys({"engine": engine}, kind, source="c.yaml"))

    assert [str(warning.message) for warning in caught] == [
        f"{loader.DEPRECATION_PREFIX}engine.estimation_mode is deprecated and will be removed in the next release; "
        "use engine.ais_perf_model (it is already set, so remove the old fields)"
    ]


# One old field that repeats ``EST``, alone, next to the key, and next to a key that is null.
OLD_FIELD = {"estimation_mode": "op_level"}
KEY_AND_OLD_FIELD = {**OLD_FIELD, "ais_perf_model": EST}
NULL_KEY_AND_OLD_FIELD = {**OLD_FIELD, "ais_perf_model": None}


@pytest.mark.parametrize(
    ("in_file", "in_run", "hint"),
    [
        (
            NULL_KEY_AND_OLD_FIELD,
            NULL_KEY_AND_OLD_FIELD,
            "run 'aisim perf-model migrate c.yaml' to convert the file",
        ),
        (OLD_FIELD, KEY_AND_OLD_FIELD, "it is already set, so remove the old fields"),
        ({"ais_perf_model": EST}, KEY_AND_OLD_FIELD, "it is already set, so remove the old fields"),
        (KEY_AND_OLD_FIELD, KEY_AND_OLD_FIELD, "it is already set, so remove the old fields"),
        (KEY_AND_OLD_FIELD, NULL_KEY_AND_OLD_FIELD, "the file already sets it, so remove the old fields from the file"),
        ({"ais_perf_model": EST}, NULL_KEY_AND_OLD_FIELD, "set engine.ais_perf_model.<key> with --set instead"),
    ],
    ids=[
        "a null key next to an old field",
        "an old field in the file, the key from an override",
        "the key in the file, an old field from an override",
        "the key and an old field in the file",
        "the key and an old field in the file, the key removed by an override",
        "the key in the file, replaced with an old field by overrides",
    ],
)
def test_warn_old_keys_ends_with_the_fix_for_where_the_key_is_set(in_file, in_run, hint):
    # migrate reads the file: it converts a file without the key and does not convert a file that sets the key.
    file_raw = {"engine": _engine("aggregated", **in_file)}
    raw = {"engine": _engine("aggregated", **in_run)}
    VALIDATE["predict"](raw["engine"])

    caught = _warnings_from(lambda: rules.warn_old_keys(raw, "predict", source="c.yaml", file_raw=file_raw))

    assert [str(warning.message) for warning in caught] == [
        f"{loader.DEPRECATION_PREFIX}engine.estimation_mode is deprecated and will be removed in the next release; "
        f"use engine.ais_perf_model ({hint})"
    ]


ONBOARD_HINT = "'aisim onboard' wrote this file and rejects it once it is changed, so leave it as it is"
# Where onboarding records a replay config it wrote: the report, from the folder of the config, and the keys
# that lead to the record. The first is replay validation, the second the matched workload of serving validation.
ONBOARD_REPORTS = [
    pytest.param("validation.json", ("prediction_config",), id="replay validation"),
    pytest.param("../recipe.json", ("matched_workload", "prediction_config"), id="matched serving workload"),
]


def _replay_config(folder: Path) -> Path:
    """A file that stands for a replay config. The warning reads the mapping it is given, not this text."""
    folder.mkdir(parents=True)
    config = folder / "predict.yaml"
    config.write_text("engine: {estimation_mode: op_level}\n")
    return config


def _record(config: Path, report: str, keys: tuple[str, ...], **changes: Any) -> Path:
    """Save the record onboarding keeps of ``config``: its path, its size and its SHA-256."""
    content = config.read_bytes()
    record: Any = {"path": str(config), "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
    record.update(changes)
    for key in reversed(keys):
        record = {key: record}
    path = config.parent / report
    path.write_text(json.dumps(record))
    return path


def _hint(source: Any) -> str:
    """The text in brackets that ends the warning for old fields read from ``source``."""
    (warning,) = _warnings_from(lambda: rules.warn_old_keys({"engine": LEGACY}, "predict", source=source))
    return str(warning.message).split("use engine.ais_perf_model (", 1)[1].removesuffix(")")


@pytest.mark.parametrize(("report", "keys"), ONBOARD_REPORTS)
def test_warn_old_keys_says_to_leave_a_replay_config_that_onboarding_wrote(tmp_path, monkeypatch, report, keys):
    # Onboarding compares the file with its record later and rejects a converted file.
    config = _replay_config(tmp_path / "run" / "output")
    _record(config, report, keys)
    (tmp_path / "link").symlink_to(config.parent, target_is_directory=True)
    monkeypatch.chdir(config.parent)

    assert _hint(config) == ONBOARD_HINT
    assert _hint(str(config)) == ONBOARD_HINT
    assert _hint("predict.yaml") == ONBOARD_HINT
    assert _hint(tmp_path / "link" / "predict.yaml") == ONBOARD_HINT


@pytest.mark.parametrize(("report", "keys"), ONBOARD_REPORTS)
def test_warn_old_keys_names_migrate_for_a_file_that_onboarding_does_not_check(tmp_path, report, keys):
    config = _replay_config(tmp_path / "run" / "output")
    migrate = f"run 'aisim perf-model migrate {config}' to convert the file"
    assert _hint(config) == migrate, "no report"

    record = _record(config, report, keys)
    assert _hint(config) == ONBOARD_HINT
    config.write_text(config.read_text().replace("op_level", "OP_LEVEL"))
    assert _hint(config) == migrate, "the file was changed and kept its size"
    config.write_text(config.read_text() + "# edited\n")
    assert _hint(config) == migrate, "the file was changed after it was recorded"

    _record(config, report, keys, path=str(tmp_path / "elsewhere" / "predict.yaml"))
    assert _hint(config) == migrate, "the record is about a file in another folder"
    _record(config, report, keys, size_bytes=0)
    assert _hint(config) == migrate, "the record gives another size"

    for text in ("", "not JSON", "[]", "{}", json.dumps(dict.fromkeys(keys, "text")), "[" * 100_000):
        record.write_text(text)
        assert _hint(config) == migrate, f"a report that holds {text[:20]!r}"
    _record(config, report, keys, path=3)
    assert _hint(config) == migrate, "a record whose path is a number"
    record.unlink()
    record.mkdir()
    assert _hint(config) == migrate, "a folder in place of the report"
    assert _hint(tmp_path / "missing" / "predict.yaml").startswith("run 'aisim perf-model migrate ")


def test_warn_old_keys_names_migrate_for_a_copy_of_a_replay_config_in_another_folder(tmp_path):
    config = _replay_config(tmp_path / "run")
    _record(config, "validation.json", ("prediction_config",))
    copy_of_it = _replay_config(tmp_path / "copy")
    (tmp_path / "copy" / "validation.json").write_text((tmp_path / "run" / "validation.json").read_text())

    assert _hint(config) == ONBOARD_HINT
    assert _hint(copy_of_it) == f"run 'aisim perf-model migrate {copy_of_it}' to convert the file"


def test_warn_old_keys_keeps_the_other_fixes_for_a_replay_config_that_onboarding_wrote(tmp_path):
    # The record only replaces the advice to convert the file.
    config = _replay_config(tmp_path / "run")
    _record(config, "validation.json", ("prediction_config",))
    old, key, both = (
        _engine("aggregated", **OLD_FIELD),
        _engine("aggregated"),
        _engine("aggregated", **KEY_AND_OLD_FIELD),
    )

    def ending(raw: dict[str, Any], file_raw: dict[str, Any]) -> str:
        def warn() -> None:
            rules.warn_old_keys({"engine": raw}, "predict", source=config, file_raw={"engine": file_raw})

        return str(_warnings_from(warn)[0].message).rsplit("(", 1)[1]

    assert ending(old, old) == f"{ONBOARD_HINT})"
    assert ending(old, key) == "set engine.ais_perf_model.<key> with --set instead)"
    assert ending(both, old) == "it is already set, so remove the old fields)"


@pytest.mark.parametrize(
    "engine",
    [
        _engine("aggregated"),
        _engine("aggregated", ais_perf_model=EST),
        _afd_engine("predict", "aggregated", "aggregated", **EST),
    ],
    ids=["no estimator field", "the key", "a config the key cannot describe"],
)
def test_warn_old_keys_is_silent_without_old_fields(engine):
    assert _warnings_from(lambda: rules.warn_old_keys({"engine": engine}, "predict", source="p.yaml")) == []


def test_warning_points_at_the_caller_of_the_function_that_reads_the_config():
    # A reader function calls warn_old_keys; the line to show is the one that called the reader.
    reader = {"warn_old_keys": rules.warn_old_keys}
    source = "def read_config(raw, **options):\n    warn_old_keys(raw, 'predict', **options)\n"
    exec(compile(source, "config_reader.py", "exec"), reader)

    by_default = _warnings_from(lambda: reader["read_config"]({"engine": LEGACY}))
    in_reader = _warnings_from(lambda: reader["read_config"]({"engine": LEGACY}, stacklevel=1))

    assert by_default[0].filename == __file__
    assert in_reader[0].filename == "config_reader.py"


def test_validating_a_config_never_warns():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _predict(copy.deepcopy(LEGACY))
        _recommend(_engine("both", ais_perf_model=EST))


# --- Rules shared with search configs ---------------------------------------------------------


def test_repr_of_a_search_space_reads_as_it_did_before_the_key_existed():
    space = {"model_name": MODEL, "hardware_sku": SYSTEM, "backend": ["vllm"], "deployment_mode": ["agg"]}
    old = SearchSpace.model_validate({**space, "estimation_mode": "op_level"})
    new = SearchSpace.model_validate({**space, "ais_perf_model": {"estimation_mode": "op_level"}})

    assert "ais_perf_model" not in repr(old)
    assert "estimation_mode='op_level'" in repr(old)
    assert repr(new) == repr(old)


def test_estimator_keys_are_the_ones_the_loader_names():
    # One tuple, not a copy: a key the loader gains is an estimator key here at once.
    assert rules.ESTIMATOR_KEYS is loader.ESTIMATOR_KEYS
    assert tuple(key for key in loader.ESTIMATOR_KEYS if key != "fpm_profile") == rules.ROLE_ESTIMATOR_KEYS
    assert set(rules.WORKER_TYPE_ORDER) == set(loader.WORKER_TYPES)


# The fields both engine configs share that are no estimator field: the key, and the engine-wide precision and
# model controls.
OTHER_SHARED_ENGINE_FIELDS = {
    "ais_perf_model",
    "attention_backend",
    "comm_quant_mode",
    "enable_chunked_prefill",
    "enable_eplb",
    "fmha_quant_mode",
    "gemm_quant_mode",
    "kvcache_quant_mode",
    "moe_backend",
    "moe_kernel_source",
    "moe_quant_mode",
    "nextn",
    "nextn_accepted",
    "wideep_num_slots",
}
# The fields of a worker's timing that are no estimator field.
OTHER_TIMING_FIELDS = {
    "type",
    "prefill_ms",
    "decode_ms",
    "attention_backend",
    "comm_quant_mode",
    "fmha_quant_mode",
    "gemm_quant_mode",
    "kvcache_quant_mode",
    "moe_quant_mode",
}


def test_old_engine_keys_are_the_estimator_fields_of_the_engine_configs():
    """A field added to the settings both engine configs share must be put into one of the two groups."""
    old = set(rules._OLD_ENGINE_KEYS)

    assert old.isdisjoint(OTHER_SHARED_ENGINE_FIELDS)
    # fpm_profile is a field of each engine config, not of the settings they share.
    assert set(EstimatorPolicyConfig.model_fields) == (old - {"fpm_profile"}) | OTHER_SHARED_ENGINE_FIELDS
    for model in (EnginePredictionConfig, EngineRecommendationConfig):
        assert old <= set(model.model_fields)


# The fields each engine config adds to the settings both share that are no estimator field.
OTHER_OWN_ENGINE_FIELDS = {
    EnginePredictionConfig: {
        "backend",
        "backend_version",
        "context_length",
        "decoder_replay",
        "detokenization",
        "enable_shared_layer",
        "hardware",
        "kv_transfer",
        "mode",
        "model",
        "speculation",
        "strict_provenance",
        "tokenization",
        "workers",
    },
    EngineRecommendationConfig: {
        "backend",
        "backend_version",
        "context_length",
        "hardware",
        "kv_transfer",
        "mode",
        "model",
        "speculation",
        "workers",
    },
}


@pytest.mark.parametrize("model", [EnginePredictionConfig, EngineRecommendationConfig])
def test_old_engine_keys_hold_the_one_estimator_field_an_engine_config_adds(model):
    """A field added to one engine config alone must be put into one of the two groups too."""
    own = set(model.model_fields) - set(EstimatorPolicyConfig.model_fields)

    assert own == OTHER_OWN_ENGINE_FIELDS[model] | {"fpm_profile"}
    assert own & set(rules._OLD_ENGINE_KEYS) == {"fpm_profile"}


def test_old_timing_keys_are_the_estimator_fields_of_a_worker_timing():
    """A field added to a worker's timing must be put into one of the two groups."""
    old = set(rules._OLD_TIMING_KEYS)

    assert old.isdisjoint(OTHER_TIMING_FIELDS)
    assert set(TimingConfig.model_fields) == old | OTHER_TIMING_FIELDS


def test_active_worker_types_follow_the_mode():
    assert rules.active_worker_types({}, "predict") == ("aggregated",)
    assert rules.active_worker_types({"mode": "disaggregated"}, "predict") == ("prefill", "decode")
    assert rules.active_worker_types({}, "recommend") == ("aggregated", "prefill", "decode")
    assert rules.active_worker_types({"mode": {"choices": ["disaggregated"]}}, "recommend") == ("prefill", "decode")
    assert rules.active_worker_types({"mode": {"choices": ["disaggregated", "aggregated"]}}, "recommend") == (
        "aggregated",
        "prefill",
        "decode",
    )


def test_normalize_entries_and_the_plan_use_the_name_and_order_they_are_given():
    value = _entries(decode={**OTHER, "tp": 2}, prefill=EST)

    entries = rules.normalize_entries(value, ("decode", "prefill"), name="search_space.ais_perf_model")
    shared, per_worker = rules.plan_estimator_keys(entries, name="search_space.ais_perf_model", config_kinds="search")

    assert entries == _entries(decode={**OTHER, "tp": 2}, prefill=EST)
    assert [rules.identity_keys(entry) for entry in entries] == [{"tp": 2}, {}]
    assert [rules.is_snapshot_entry(entry) for entry in entries] == [False, False]
    assert shared == {}
    assert per_worker == {"decode": OTHER, "prefill": EST}
    with pytest.raises(AisPerfModelError, match=r"^search_space\.ais_perf_model cannot be a string"):
        rules.normalize_entries("estimator.yaml", ("decode",), name="search_space.ais_perf_model")
    with pytest.raises(AisPerfModelError, match="dataset entries are not supported in search configs yet"):
        rules.plan_estimator_keys(
            [{"worker_type": "decode", "from": {"dataset": {"root": "/data", "manifest": "m.json"}}}],
            name="search_space.ais_perf_model",
            config_kinds="search configs",
        )


def test_plan_shares_a_key_only_when_every_entry_gives_the_same_value():
    entries = _entries(
        prefill={"estimation_mode": "op_level", "database_mode": "hybrid", "fpm_profile": {"schema_version": 1}},
        decode={"estimation_mode": "op_level", "fallback_policy": "allow", "fpm_profile": {"schema_version": 1}},
    )

    shared, per_worker = rules.plan_estimator_keys(entries, name="engine.ais_perf_model", config_kinds="predict")

    assert shared == {"fpm_profile": {"schema_version": 1}, "estimation_mode": "op_level"}
    assert per_worker == {"prefill": {"database_mode": "hybrid"}, "decode": {"fallback_policy": "allow"}}
    assert shared["fpm_profile"] is not entries[0]["fpm_profile"]


@pytest.mark.parametrize(
    ("model_cls", "name", "mapping", "written"),
    [
        (EnginePredictionConfig, "estimation_mode", {}, False),
        (EnginePredictionConfig, "estimation_mode", {"estimation_mode": None}, False),
        (EnginePredictionConfig, "estimation_mode", {"estimation_mode": "auto"}, False),
        (EnginePredictionConfig, "estimation_mode", {"estimation_mode": "op_level"}, True),
        (EnginePredictionConfig, "database_mode", {"database_mode": "silicon"}, False),
        (EnginePredictionConfig, "estimator_config", {"estimator_config": {}}, False),
        (EnginePredictionConfig, "systems_path", {"systems_path": "default"}, True),
        (TimingConfig, "estimator_config", {"estimator_config": {}}, True),
        (TimingConfig, "estimation_mode", {"estimation_mode": "auto"}, True),
        (TimingConfig, "transfer_policy", {"transfer_policy": []}, True),
    ],
)
def test_a_field_is_written_when_it_is_present_not_null_and_not_the_declared_default(model_cls, name, mapping, written):
    assert rules.is_written(model_cls, name, mapping) is written


def test_same_setting_compares_values_the_way_the_field_stores_them():
    assert rules.same_setting(EnginePredictionConfig, "database_mode", "hybrid", "HYBRID")
    assert rules.same_setting(TimingConfig, "estimator_config", {"a": {"b": 1}}, {"a": {"b": 1}})
    assert not rules.same_setting(TimingConfig, "estimator_config", {"a": {"b": 1}}, {"a": {"b": 2}})
    assert not rules.same_setting(EnginePredictionConfig, "estimation_mode", "auto", "op_level")
    assert not rules.same_setting(EnginePredictionConfig, "forward_model", "op_level", "op_level")


def test_same_setting_compares_a_transfer_policy_by_the_kinds_it_names():
    assert rules.same_setting(EnginePredictionConfig, "transfer_policy", ["xshape"], "xshape")
    assert rules.same_setting(TimingConfig, "transfer_policy", "balanced", "xquant,xshape")
    assert not rules.same_setting(EnginePredictionConfig, "transfer_policy", "xshape", "xquant")
    # A policy the field cannot read is compared as written.
    assert rules.same_setting(TimingConfig, "transfer_policy", "other", "other")
    assert not rules.same_setting(TimingConfig, "transfer_policy", "other", "xshape")


@pytest.mark.parametrize(
    ("old", "key", "same"),
    [
        pytest.param(["xshape", "xshape"], "xshape", True, id="a kind named twice"),
        pytest.param(("xshape",), "xshape", True, id="a tuple of kinds"),
        pytest.param([], "off", True, id="an empty list"),
        pytest.param(["XSHAPE"], "xshape", False, id="a kind in capitals in a list"),
        pytest.param(["balanced"], "balanced", False, id="a preset in a list"),
        pytest.param(["balanced", "xshape"], ["xshape", "balanced"], False, id="two lists that hold a preset"),
        pytest.param(["balanced", "balanced"], ["balanced"], False, id="a preset in a list, once and twice"),
        pytest.param([["xshape"]], "xshape", False, id="a list that holds a list"),
        pytest.param({"xshape": True}, "xshape", False, id="a mapping"),
        pytest.param(5, "xshape", False, id="a number"),
        pytest.param(True, "xshape", False, id="a bool"),
        pytest.param("other", "another", False, id="two texts that are not policies"),
        pytest.param("other", "other", True, id="one text that is not a policy, twice"),
    ],
)
def test_same_setting_reads_a_transfer_policy_the_way_a_run_reads_it(old, key, same):
    # A list names kinds exactly: no preset and no change of letter case. Text goes through presets.
    assert rules.same_setting(TimingConfig, "transfer_policy", old, key) is same
    assert rules.same_setting(EnginePredictionConfig, "transfer_policy", key, old) is same


def test_same_setting_does_not_hide_an_error_that_the_value_did_not_cause(monkeypatch):
    import pydantic

    class Broken:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AttributeError("a fault in the comparison")

    monkeypatch.setattr(pydantic, "TypeAdapter", Broken)

    with pytest.raises(AttributeError, match="a fault in the comparison"):
        rules.same_setting(EnginePredictionConfig, "estimation_mode", "auto", "auto")


@pytest.mark.parametrize("kind", ["predict", "recommend"])
def test_old_path_under_a_home_that_does_not_exist_still_says_something_else_than_the_key(kind):
    # The field cannot read this path and says so with a RuntimeError, not a ValueError.
    old = {"systems_paths": ["~no-such-user-for-this-test/data"]}
    engine = _engine("aggregated", **old, ais_perf_model=DATA_ROOTS)

    with pytest.raises(AisPerfModelError, match="say something else: engine.systems_paths. Remove the old fields."):
        rules.lower_engine(engine, kind)


def test_shared_errors_name_the_key_they_are_about():
    conflict = rules.old_fields_conflict("search_space.ais_perf_model", ["search_space.estimation_mode"])
    uncovered = rules.uncovered_config("search_space.ais_perf_model", "the space searches AFD")

    assert isinstance(conflict, AisPerfModelError)
    assert str(conflict) == (
        "search_space.ais_perf_model cannot be combined with old estimator fields that say something else: "
        "search_space.estimation_mode. Remove the old fields."
    )
    assert str(uncovered) == (
        "search_space.ais_perf_model cannot be used in this config: the space searches AFD. "
        "Keep the existing estimator fields."
    )
