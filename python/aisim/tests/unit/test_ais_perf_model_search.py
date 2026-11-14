# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``search_space.ais_perf_model`` is another spelling of the search space estimator fields.

A search space that uses the key must mean exactly what the same space means
when it is written with the existing fields. So most tests here build both
spellings and compare what comes out: the dump, the estimator request the
search builds for a candidate, or the error.
"""

from __future__ import annotations

import copy
import os
import shlex
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from aisim.config import ais_perf_model as rules
from aisim.config.engine import EstimatorPolicyConfig, TimingConfig
from aisim.sweeper.ais_perf_model import (
    _OLD_TOP_LEVEL_KEYS,
    active_worker_types_search,
    lower_search_space,
    uncovered_reason_search,
    written_old_keys_search,
)
from aisim.sweeper.config import SearchSpace, SmartSearchConfig
from aisim.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver
from aisim_core.ais_perf_model import DEPRECATION_PREFIX, AisPerfModelDeprecationWarning, AisPerfModelError

from .test_fpm_config_imports import _config as _profiled_config
from .test_recommend_state_snapshot import _calibrate, _native

pytestmark = pytest.mark.unit

MODEL, SYSTEM = "Qwen/Qwen3-0.6B", "h200_sxm"
# The roles each mode runs. "both" is a space that leaves deployment_mode unstated, so it searches both modes.
ROLES = {"agg": ("agg",), "disagg": ("prefill", "decode"), "both": ("agg", "prefill", "decode")}
# The search space calls the aggregated worker "agg" in its field names.
WORKER_TYPES = {"agg": "aggregated", "prefill": "prefill", "decode": "decode"}
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
WORKLOAD = {"isl": 128, "osl": 16, "request_rate": 2, "num_request_ratio": 3}
CONFLICT = "search_space.ais_perf_model cannot be combined with old estimator fields that say something else: "
# A null transfer policy of a role: the search reads it in place of the top-level policy.
NULL_DECODE_POLICY = {"role_estimator_controls": {"decode": {"transfer_policy": None}}}
# A decode worker that runs attention and FFN on separate pools (AFD). Only sglang has AFD workers.
AFD_DECODE = {"backend": ["sglang"], "afd_roles": {"decode": {"type": "afd", "afd": {"a_batch_size": 1}}}}
AFD_REASON = "search_space.afd_roles is set, and ais_perf_model does not cover AFD yet"


def _space(mode: str, **fields: Any) -> dict[str, Any]:
    """A raw search space for one model on one system."""
    space: dict[str, Any] = {"model_name": MODEL, "hardware_sku": SYSTEM, "backend": ["vllm"]}
    if mode != "both":
        space["deployment_mode"] = [mode]
    space.update(copy.deepcopy(fields))
    return space


def _old_spelling(mode: str, top_fields: dict[str, Any], role_fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The space written with the existing fields: top level, and ``role_estimator_controls`` per role."""
    space = _space(mode, **top_fields)
    if role_fields:
        space["role_estimator_controls"] = copy.deepcopy(role_fields)
    return space


def _entries(**by_worker_type: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"worker_type": worker_type, **keys} for worker_type, keys in by_worker_type.items()]


def _validated(raw: dict[str, Any]) -> SearchSpace:
    return SearchSpace.model_validate(raw)


def _dump(space: SearchSpace) -> dict[str, Any]:
    """The space as a dump writes it."""
    return space.model_dump(mode="json", exclude_none=True)


def _rejected(raw: dict[str, Any]) -> str:
    """Validate a space that must fail and return the message."""
    with pytest.raises(ValidationError) as failure:
        SearchSpace.model_validate(raw)
    return str(failure.value)


def _candidate(role: str) -> dict[str, Any]:
    """The values of one search candidate that the estimator request of a role reads."""
    prefix = "" if role == "agg" else f"{role}_"
    parallelism = {"tp": 2, "pp": 1, "attention_dp": 1, "moe_tp": 1, "moe_ep": 1}
    return {
        "backend": "vllm",
        "hardware_sku": SYSTEM,
        f"{role}_block_size": 64,
        **{f"{prefix}{name}": value for name, value in parallelism.items()},
    }


def _requests(space: SearchSpace, mode: str) -> dict[str, dict[str, Any]]:
    """The estimator request the search builds for a candidate, for each role the mode runs."""
    resolver = ForwardPassEstimatorResolver(space)
    return {role: resolver._request(_candidate(role), role).to_dict() for role in ROLES[mode]}


def _snapshot_entry(path: str) -> dict[str, Any]:
    return {"from": {"state_snapshot": {"path": path}}}


def _snapshot_setting(path: str) -> dict[str, Any]:
    """The existing role-level spelling of a state snapshot reference."""
    return {"estimator_config": {"state_snapshot": {"path": path}}}


@pytest.fixture(scope="module")
def snapshots(tmp_path_factory) -> dict[str, str]:
    """The folder of a calibrated state snapshot for each worker type."""
    root = tmp_path_factory.mktemp("snapshots")
    folders = {}
    for worker_type in WORKER_TYPES.values():
        _calibrate(_native(worker_type), worker_type).save_state_snapshot(root / worker_type, include_perf_db=False)
        folders[worker_type] = str(root / worker_type)
    return folders


# --- Spaces without the key -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "stored"),
    [
        pytest.param(_space("both", **EST), {**EST, "role_estimator_controls": {}}, id="top-level fields"),
        pytest.param(
            _space("disagg", role_estimator_controls={"prefill": EST, "decode": OTHER}),
            {
                "estimation_mode": "auto",
                "role_estimator_controls": {"prefill": {**EST, "transfer_policy": ["xshape"]}, "decode": OTHER},
            },
            id="role-level fields",
        ),
        pytest.param(
            _space("agg", agg_forward_model="op_level"),
            {"estimation_mode": "auto", "role_estimator_controls": {"agg": {"estimation_mode": "op_level"}}},
            id="forward model selector",
        ),
        pytest.param(
            _space("agg", encoder={}),
            {"estimation_mode": "auto", "agg_forward_model": "op_level", "role_estimator_controls": {}},
            id="encoder pool",
        ),
        pytest.param(
            _space("both"),
            {"estimation_mode": "auto", "fallback_policy": "deny", "database_mode": "SILICON", "estimator_config": {}},
            id="no estimator field",
        ),
    ],
)
def test_space_without_the_key_validates_and_dumps_as_before(raw, stored):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        space = _validated(copy.deepcopy(raw))
        dumped = space.model_dump(mode="json")
        again = _validated(dumped)

    assert space.ais_perf_model is None
    assert "ais_perf_model" not in dumped
    assert {name: dumped[name] for name in stored} == stored
    assert again.model_dump(mode="json") == dumped
    assert lower_search_space(raw) is raw


def test_a_null_key_is_the_same_as_no_key():
    raw = _space("both", **EST)

    assert _dump(_validated({**raw, "ais_perf_model": None})) == _dump(_validated(raw))
    assert lower_search_space({**raw, "ais_perf_model": None}) == {**raw, "ais_perf_model": None}


@pytest.mark.parametrize("raw", [None, "text", 3, ["agg"]], ids=["null", "string", "number", "list"])
def test_lowering_leaves_input_that_is_not_a_mapping_for_validation(raw):
    assert lower_search_space(raw) is raw


def test_a_validated_space_passes_through_lowering():
    space = _validated(_space("agg", ais_perf_model=EST))

    assert lower_search_space(space) is space
    assert _dump(_validated(space)) == _dump(space)


# --- The key and the existing fields mean the same --------------------------------------------

SAME_MEANING = [
    pytest.param("agg", EST, EST, {}, id="one worker type"),
    pytest.param("disagg", EST, EST, {}, id="bare mapping, two worker types"),
    pytest.param("both", EST, EST, {}, id="bare mapping, mode left open"),
    pytest.param("disagg", _entries(prefill=EST, decode=EST), EST, {}, id="entries agree"),
    pytest.param(
        "disagg", _entries(prefill=EST, decode=OTHER), {}, {"prefill": EST, "decode": OTHER}, id="entries differ"
    ),
    pytest.param(
        "both",
        _entries(aggregated=EST, prefill=OTHER, decode=EST),
        {},
        {"agg": EST, "prefill": OTHER, "decode": EST},
        id="one of three entries differs",
    ),
    pytest.param(
        "disagg",
        _entries(
            prefill={"estimation_mode": "op_level", "database_mode": "HYBRID"},
            decode={"estimation_mode": "op_level", "database_mode": "SILICON"},
        ),
        {"estimation_mode": "op_level"},
        {"prefill": {"database_mode": "HYBRID"}, "decode": {"database_mode": "SILICON"}},
        id="entries agree on one key",
    ),
    pytest.param("disagg", _entries(prefill=EST, decode={}), {}, {"prefill": EST}, id="one entry gives no key"),
    pytest.param(
        "disagg",
        _entries(prefill={**EST, **DATA_ROOTS}, decode=EST),
        EST,
        {"prefill": DATA_ROOTS},
        id="data roots of one worker type",
    ),
]


@pytest.mark.parametrize(("mode", "value", "top_fields", "role_fields"), SAME_MEANING)
def test_space_means_the_same_as_its_old_spelling(mode, value, top_fields, role_fields):
    new = _validated(_space(mode, ais_perf_model=value))
    old = _validated(_old_spelling(mode, top_fields, role_fields))

    assert _dump(new) == _dump(old)
    assert _requests(new, mode) == _requests(old, mode)


def test_template_file_is_merged_before_the_keys_are_written(tmp_path):
    template = tmp_path / "estimator.yaml"
    template.write_text("estimation_mode: op_level\ndatabase_mode: HYBRID\nfallback_policy: deny\n")
    value = {"from": str(template), "fallback_policy": "allow"}
    fields = {"estimation_mode": "op_level", "database_mode": "HYBRID", "fallback_policy": "allow"}

    new = _validated(_space("disagg", ais_perf_model=value))

    assert _dump(new) == _dump(_validated(_space("disagg", **fields)))
    assert new.ais_perf_model == _entries(prefill=fields, decode=fields)


def test_the_validated_space_keeps_one_entry_per_worker_type_in_role_order():
    value = _entries(decode={**OTHER, "model": MODEL}, prefill=EST, aggregated=EST)

    listed = _validated(_space("both", ais_perf_model=value))
    bare = _validated(_space("disagg", ais_perf_model=EST))

    assert listed.ais_perf_model == _entries(aggregated=EST, prefill=EST, decode={**OTHER, "model": MODEL})
    assert bare.ais_perf_model == _entries(prefill=EST, decode=EST)


def test_fpm_profile_is_written_once_for_the_space():
    profile = _profiled_config()["engine"]["fpm_profile"]
    space = _space("agg", model_name=profile["model"])

    new = _validated({**space, "ais_perf_model": {"fpm_profile": profile, "fallback_policy": "allow"}})
    old = _validated({**space, "fpm_profile": profile, "fallback_policy": "allow"})

    assert _dump(new) == _dump(old)
    assert _requests(new, "agg") == _requests(old, "agg")


def test_fpm_profile_must_be_one_value_for_all_worker_types():
    value = _entries(prefill={"fpm_profile": {"schema_version": 1}}, decode={"estimation_mode": "auto"})

    message = _rejected(_space("disagg", ais_perf_model=value))

    assert "search_space.ais_perf_model: fpm_profile is one value for the whole config" in message
    assert "the entry for prefill gives it and the entry for decode does not" in message


def test_old_fpm_profile_next_to_the_key_must_say_the_same():
    profile = _profiled_config()["engine"]["fpm_profile"]
    space = _space("agg", model_name=profile["model"], fpm_profile=profile)
    first = _validated(space)
    dumped = first.model_dump(mode="json")

    repeated = _validated({**dumped, "ais_perf_model": {"fpm_profile": profile}})
    without = _rejected({**space, "ais_perf_model": {"fallback_policy": "allow"}})
    changed = _rejected({**space, "ais_perf_model": {"fpm_profile": {**profile, "context_length": 8192}}})

    # The space stores the profile with its defaults filled in, so a dump holds more than the author wrote.
    assert dumped["fpm_profile"] != profile
    assert _dump(repeated) == _dump(first)
    assert f"{CONFLICT}search_space.fpm_profile. Remove the old fields." in without
    assert f"{CONFLICT}search_space.fpm_profile. Remove the old fields." in changed
    assert rules.written_old_keys({"search_space": space}, "search") == ["search_space.fpm_profile"]


def test_old_fpm_profile_comparison_does_not_hide_an_error_that_the_value_did_not_cause(monkeypatch):
    import aisim.fpm_profile as profiles

    def broken(value: Any) -> Any:
        raise AttributeError("a fault in the comparison")

    monkeypatch.setattr(profiles, "load_fpm_profile", broken)
    profile = {"schema_version": 1}

    with pytest.raises(AttributeError, match="a fault in the comparison"):
        lower_search_space(_space("agg", fpm_profile=profile, ais_perf_model={"fpm_profile": profile}))


@pytest.mark.parametrize(
    ("name", "old", "new"),
    [
        ("systems_paths", ["~no-such-user-for-this-test/data"], ["default"]),
        ("fpm_profile", "not JSON", {"schema_version": 1}),
        ("fpm_profile", "[" * 100_000, {"schema_version": 1}),
    ],
    ids=["a path under a home that does not exist", "a profile that is not JSON", "a profile nested too deep"],
)
def test_old_value_the_space_cannot_read_still_says_something_else_than_the_key(name, old, new):
    # The space rejects each old value on its own, with an error that is not always a ValueError.
    space = _space("agg", **{name: old}, ais_perf_model={name: new})

    with pytest.raises(AisPerfModelError) as failure:
        lower_search_space(space)

    assert str(failure.value) == f"{CONFLICT}search_space.{name}. Remove the old fields."


# --- State snapshots --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "plain"),
    [
        ("agg", {}),
        ("disagg", {}),
        ("disagg", {"decode": {"estimation_mode": "op_level", "fallback_policy": "allow"}}),
    ],
    ids=["one worker type", "a snapshot for each worker type", "a snapshot and plain settings"],
)
def test_snapshot_entry_means_the_same_as_the_role_level_reference(mode, plain, snapshots):
    paths = {role: snapshots[WORKER_TYPES[role]] for role in ROLES[mode] if role not in plain}
    value = [
        {"worker_type": WORKER_TYPES[role], **(_snapshot_entry(paths[role]) if role in paths else plain[role])}
        for role in ROLES[mode]
    ]
    role_fields = {role: _snapshot_setting(paths[role]) if role in paths else plain[role] for role in ROLES[mode]}

    new = _validated(_space(mode, ais_perf_model=value))
    old = _validated(_old_spelling(mode, {}, role_fields))

    assert _dump(new) == _dump(old)
    assert _requests(new, mode) == _requests(old, mode)
    # A snapshot belongs to one worker type, so nothing is written for the whole space.
    assert {name: _dump(new)[name] for name in ("estimation_mode", "fallback_policy")} == {
        "estimation_mode": "auto",
        "fallback_policy": "deny",
    }


def test_one_snapshot_cannot_serve_two_worker_types():
    message = _rejected(_space("both", ais_perf_model=_snapshot_entry("/no/such/snapshot")))

    assert (
        "search_space.ais_perf_model names a state snapshot, and a state snapshot belongs to one worker type. "
        "This config runs aggregated, prefill, decode; write a list with one entry per worker_type"
    ) in message


def test_snapshot_written_in_both_places_is_an_error():
    value = {**_snapshot_entry("/no/such/snapshot"), **_snapshot_setting("/no/such/snapshot")}

    message = _rejected(_space("agg", ais_perf_model=value))

    assert (
        "search_space.ais_perf_model entry for aggregated: the state snapshot is written twice, "
        "as from.state_snapshot and as estimator_config.state_snapshot; keep one"
    ) in message


def test_snapshot_problems_keep_the_errors_of_the_search_space(snapshots):
    # The existing validators read the snapshot right after the key is rewritten.
    pinned = {**_snapshot_entry(snapshots["aggregated"]), "database_mode": "HYBRID"}

    other_worker = _rejected(_space("agg", ais_perf_model=_snapshot_entry(snapshots["decode"])))
    missing = _rejected(_space("agg", ais_perf_model=_snapshot_entry("/no/such/snapshot")))
    extra_setting = _rejected(_space("agg", ais_perf_model=pinned))

    assert "was calibrated for a decode worker, not aggregated" in other_worker
    assert "cannot read the snapshot at /no/such/snapshot" in missing
    assert "database_mode conflicts with the agg role's state snapshot" in extra_setting


def test_dumped_snapshot_space_with_the_key_added_back_is_the_same_config(snapshots):
    value = _snapshot_entry(snapshots["aggregated"])
    first = _validated(_space("agg", ais_perf_model=value))
    dumped = first.model_dump(mode="json")

    again = _validated({**dumped, "ais_perf_model": value})

    # The dump lists the mode and the fallback policy the snapshot pins, next to the reference.
    assert set(dumped["role_estimator_controls"]["agg"]) == {"estimation_mode", "fallback_policy", "estimator_config"}
    assert _dump(again) == _dump(first)


def test_old_setting_that_contradicts_the_snapshot_of_the_key_is_an_error(snapshots):
    value = _snapshot_entry(snapshots["aggregated"])
    dumped = _validated(_space("agg", ais_perf_model=value)).model_dump(mode="json")

    def with_old(**settings: Any) -> dict[str, Any]:
        stored = dumped["role_estimator_controls"]["agg"]
        return {**dumped, "role_estimator_controls": {"agg": {**stored, **settings}}, "ais_perf_model": value}

    other_mode = _rejected(with_old(estimation_mode="fpm_regression"))
    other_policy = _rejected(with_old(fallback_policy="allow"))
    other_snapshot = _rejected(with_old(**_snapshot_setting(snapshots["decode"])))
    extra_setting = _rejected(with_old(database_mode="HYBRID"))

    assert "estimation_mode='fpm_regression' conflicts with the agg role's state snapshot" in other_mode
    assert "fallback_policy='allow' conflicts with the agg role's state snapshot" in other_policy
    assert f"{CONFLICT}search_space.role_estimator_controls.agg.estimator_config. Remove" in other_snapshot
    assert f"{CONFLICT}search_space.role_estimator_controls.agg.database_mode. Remove" in extra_setting


# --- Keys that name the engine ----------------------------------------------------------------


def test_entry_may_restate_what_the_space_states():
    value = _entries(
        aggregated={**EST, "model": MODEL, "system": SYSTEM, "backend": "vllm"},
        prefill={**EST, "system": "h100_sxm"},
        decode={**EST, "system": SYSTEM},
    )

    new = _validated(_space("both", prefill_hardware_sku="h100_sxm", ais_perf_model=value))

    assert _dump(new) == _dump(_validated(_space("both", prefill_hardware_sku="h100_sxm", **EST)))


@pytest.mark.parametrize(
    ("keys", "space_fields", "message"),
    [
        pytest.param(
            {"model": "another/model"},
            {},
            f'entry for aggregated: model="another/model" differs from search_space.model_name="{MODEL}"',
            id="model",
        ),
        pytest.param(
            {"system": "h100_sxm"},
            {},
            f'entry for aggregated: system="h100_sxm" differs from search_space.hardware_sku="{SYSTEM}"',
            id="system",
        ),
        pytest.param(
            {"system": SYSTEM},
            {"prefill_hardware_sku": "h100_sxm"},
            f'entry for prefill: system="{SYSTEM}" differs from search_space.prefill_hardware_sku="h100_sxm"',
            id="system of a role with its own hardware",
        ),
        pytest.param(
            {"backend": "sglang"},
            {},
            'entry for aggregated: backend="sglang" differs from search_space.backend="vllm"',
            id="backend",
        ),
        pytest.param(
            {"backend": "vllm"},
            {"backend": ["vllm", "sglang"]},
            'entry for aggregated: backend="vllm" needs one stated value at search_space.backend '
            '(found ["vllm", "sglang"]); state one backend there or drop the key',
            id="several backends",
        ),
        pytest.param(
            {"backend": "vllm"},
            {"backend": None},
            'entry for aggregated: backend="vllm" needs one stated value at search_space.backend '
            "(found null); state one backend there or drop the key",
            id="backend left to the default",
        ),
        pytest.param(
            {"system": True},
            {},
            f'entry for aggregated: system=true differs from search_space.hardware_sku="{SYSTEM}"',
            id="a boolean",
        ),
        pytest.param(
            {"tp": 2, "kv_block_size": 16},
            {},
            "entry for aggregated: tp, kv_block_size cannot be checked in a search config. "
            "The search space does not state them (the search chooses tp and similar values for each candidate). "
            "Drop these keys from the entry, or from the template file when 'from: <file>' brings them in.",
            id="a key the search chooses",
        ),
        pytest.param(
            {"afd_routing_location": "a", "afd_pool_role": "f"},
            {},
            "entry for aggregated: afd_routing_location, afd_pool_role cannot be checked in a search config. ",
            id="keys of an AFD pool",
        ),
    ],
)
def test_entry_that_cannot_be_matched_is_an_error(keys, space_fields, message):
    raw = _space("both", ais_perf_model={**EST, **keys}, **space_fields)

    assert f"search_space.ais_perf_model {message}" in _rejected(raw)


def test_dataset_entry_is_not_supported_yet():
    value = {"from": {"dataset": {"root": "datasets", "manifest": "manifest.json"}}}

    message = _rejected(_space("agg", ais_perf_model=value))

    assert "dataset entries are not supported in search configs yet" in message


# --- The key next to the existing fields ------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "value", "old_fields", "paths"),
    [
        pytest.param("both", EST, {"estimation_mode": "fpm_regression"}, ["estimation_mode"], id="another value"),
        pytest.param("both", EST, DATA_ROOTS, ["systems_paths"], id="a field the key does not give"),
        pytest.param(
            "agg",
            EST,
            {"role_estimator_controls": {"agg": {"database_mode": "HYBRID"}}},
            ["role_estimator_controls.agg.database_mode"],
            id="role level, key written for the whole space",
        ),
        pytest.param(
            "disagg",
            _entries(prefill=EST, decode=OTHER),
            {"database_mode": "HYBRID"},
            ["database_mode"],
            id="whole space, key written per role",
        ),
        pytest.param(
            "agg",
            EST,
            {"role_estimator_controls": {"prefill": {"database_mode": "HYBRID"}}},
            ["role_estimator_controls.prefill.database_mode"],
            id="a role the space does not run",
        ),
        pytest.param("agg", EST, {"agg_forward_model": "op_level"}, ["agg_forward_model"], id="forward model selector"),
        pytest.param("both", DATA_ROOTS, {"systems_path": "default"}, ["systems_path"], id="single data root"),
        pytest.param(
            "both", EST, {"prefill_fpm_parquet_path": "fpm.parquet"}, ["prefill_fpm_parquet_path"], id="parquet path"
        ),
        pytest.param(
            "both",
            EST,
            {
                "decode_forward_model": "fpm",
                "role_estimator_controls": {"decode": {"database_mode": "HYBRID"}},
                "systems_path": "default",
                "estimation_mode": "fpm_regression",
            },
            [
                "estimation_mode",
                "systems_path",
                "role_estimator_controls.decode.database_mode",
                "decode_forward_model",
            ],
            id="several fields",
        ),
        pytest.param(
            "disagg",
            {"transfer_policy": "xshape"},
            NULL_DECODE_POLICY,
            ["role_estimator_controls.decode.transfer_policy"],
            id="null transfer policy of a role, key written for the whole space",
        ),
        pytest.param(
            "disagg",
            _entries(prefill=EST, decode={"transfer_policy": "xshape"}),
            NULL_DECODE_POLICY,
            ["role_estimator_controls.decode.transfer_policy"],
            id="null transfer policy of a role, key gives the role a policy",
        ),
        pytest.param(
            "disagg",
            OTHER,
            NULL_DECODE_POLICY,
            ["role_estimator_controls.decode.transfer_policy"],
            id="null transfer policy of a role, key gives no policy",
        ),
    ],
)
def test_old_field_that_says_something_else_is_an_error(mode, value, old_fields, paths):
    message = _rejected(_space(mode, ais_perf_model=value, **old_fields))

    assert f"{CONFLICT}{', '.join(f'search_space.{path}' for path in paths)}. Remove the old fields." in message


def test_default_valued_old_fields_do_not_block_the_key():
    # A dump lists every default, so a default next to the key is not a choice the author made.
    defaults = {
        "estimation_mode": "auto",
        "fallback_policy": "deny",
        "database_mode": "silicon",
        "estimator_config": {},
        "transfer_policy": None,
        "systems_paths": None,
        "systems_path": None,
        "agg_fpm_parquet_path": None,
        "role_estimator_controls": {"agg": {}},
    }

    new = _validated(_space("both", ais_perf_model=EST, **defaults))

    assert _dump(new) == _dump(_validated(_space("both", **EST, role_estimator_controls={"agg": {}})))


def test_old_fields_that_repeat_the_key_are_accepted():
    # Role settings are compared the way the space stores them: the mode in upper case, the policy as its kinds.
    value = _entries(prefill=EST, decode=OTHER)
    repeated = {"prefill": {"estimation_mode": "op_level", "database_mode": "hybrid", "transfer_policy": ["xshape"]}}

    per_role = _validated(_space("disagg", ais_perf_model=value, role_estimator_controls=repeated))
    whole_space = _validated(_space("both", ais_perf_model=EST, estimation_mode="op_level", database_mode="hybrid"))

    assert _dump(per_role) == _dump(_validated(_space("disagg", ais_perf_model=value)))
    assert _dump(whole_space) == _dump(_validated(_space("both", ais_perf_model=EST)))


def test_top_level_transfer_policy_accepts_another_spelling_of_the_same_kinds():
    # The top-level field keeps the policy as written, and one set of kinds has more than one spelling.
    as_text = _validated(_space("agg", transfer_policy="xshape"))
    as_list = _validated(_space("agg", transfer_policy=["xshape"]))
    key = {"transfer_policy": "xshape"}
    both_kinds = {"transfer_policy": ["xshape", "xquant"]}

    repeated = _validated(_space("agg", ais_perf_model=key, transfer_policy="xshape"))
    other_spelling = _validated(_space("agg", ais_perf_model=key, transfer_policy=["xshape"]))
    comma_separated = _validated(_space("agg", ais_perf_model=both_kinds, transfer_policy="xshape,xquant"))
    other_kinds = _rejected(_space("agg", ais_perf_model=key, transfer_policy=["xquant"]))

    assert (_dump(as_text)["transfer_policy"], _dump(as_list)["transfer_policy"]) == ("xshape", ["xshape"])
    assert _requests(as_text, "agg") == _requests(as_list, "agg")
    assert _dump(repeated) == _dump(as_text)
    # The space keeps the spelling of the key.
    assert _dump(other_spelling) == _dump(as_text)
    assert _dump(comma_separated) == _dump(_validated(_space("agg", ais_perf_model=both_kinds)))
    assert f"{CONFLICT}search_space.transfer_policy. Remove the old fields." in other_kinds


def test_old_transfer_policy_that_is_not_a_policy_says_something_else_than_the_key():
    # The names of a mapping are not read as kinds: a search does not accept a mapping as a policy.
    key = {"transfer_policy": "xshape"}

    message = _rejected(_space("agg", ais_perf_model=key, transfer_policy={"xshape": True}))

    assert f"{CONFLICT}search_space.transfer_policy. Remove the old fields." in message


def test_null_transfer_policy_of_a_role_repeats_a_key_that_names_every_kind():
    # A null policy allows every transfer kind, and that is what the preset "aggressive" names.
    value = _entries(prefill=EST, decode={"transfer_policy": "aggressive"})

    new = _validated(_space("disagg", ais_perf_model=value, **NULL_DECODE_POLICY))

    assert _dump(new) == _dump(_validated(_space("disagg", ais_perf_model=value)))


@pytest.mark.parametrize(
    ("mode", "value"),
    [("agg", EST), ("disagg", _entries(prefill=EST, decode=OTHER))],
    ids=["whole space", "per role"],
)
def test_empty_role_settings_of_a_generated_space_are_filled_in(mode, value):
    empty = {role: {} for role in ROLES[mode]}
    plain = _validated(_space(mode, ais_perf_model=value))

    new = _validated(_space(mode, ais_perf_model=value, role_estimator_controls=empty))

    assert _dump(new)["role_estimator_controls"] == {**empty, **_dump(plain)["role_estimator_controls"]}
    assert _requests(new, mode) == _requests(plain, mode)


ROUND_TRIPS = [
    pytest.param("agg", EST, id="whole space"),
    pytest.param("both", EST, id="whole space, mode left open"),
    pytest.param("disagg", _entries(prefill=EST, decode=OTHER), id="per role"),
    pytest.param(
        "both",
        _entries(aggregated=EST, prefill=OTHER, decode={**OTHER, "model": MODEL}),
        id="per role, mode left open",
    ),
    pytest.param(
        "disagg",
        _entries(prefill={"transfer_policy": "balanced"}, decode={"transfer_policy": ["xshape", "xquant"]}),
        id="transfer policy per role",
    ),
    pytest.param("disagg", _entries(prefill={**EST, **DATA_ROOTS}, decode=EST), id="data roots of one role"),
]


@pytest.mark.parametrize(("mode", "value"), ROUND_TRIPS)
def test_dump_with_the_key_added_back_is_the_same_config(mode, value):
    first = _validated(_space(mode, ais_perf_model=value))
    dumped = first.model_dump(mode="json")

    again = _validated({**dumped, "ais_perf_model": value})

    assert "ais_perf_model" not in dumped
    assert _dump(again) == _dump(first)
    assert _dump(_validated(dumped)) == _dump(first)


@pytest.mark.parametrize(("mode", "value"), ROUND_TRIPS)
def test_lowering_twice_changes_nothing_and_leaves_the_input_alone(mode, value):
    raw = _space(mode, ais_perf_model=value, role_estimator_controls={ROLES[mode][0]: {}})
    before = copy.deepcopy(raw)

    once = lower_search_space(raw)

    assert raw == before
    assert once is not raw
    assert [entry["worker_type"] for entry in once["ais_perf_model"]] == [WORKER_TYPES[role] for role in ROLES[mode]]
    assert lower_search_space(once) == once
    assert lower_search_space(copy.deepcopy(once)) == once


# --- Spaces and shapes the key cannot describe ------------------------------------------------

UNCOVERED = [
    pytest.param(AFD_DECODE, AFD_REASON, id="AFD role"),
    pytest.param({**AFD_DECODE, "deployment_mode": ["disagg"]}, AFD_REASON, id="AFD role next to an ordinary role"),
    pytest.param(
        {"encoder": {}},
        "search_space.encoder is set, and ais_perf_model does not cover an encoder pool yet",
        id="encoder pool",
    ),
    pytest.param(
        {"decode_timing_model": {"type": "fixed"}},
        "search_space.decode_timing_model is set, and ais_perf_model covers only default timing",
        id="timing model of one role",
    ),
]


@pytest.mark.parametrize(("changes", "reason"), UNCOVERED)
def test_space_the_key_cannot_describe_yet_is_an_error(changes, reason):
    space = _space("both", **changes)

    message = _rejected({**space, "ais_perf_model": EST})

    assert f"search_space.ais_perf_model cannot be used in this config: {reason}." in message
    assert uncovered_reason_search(space) == reason


def test_space_the_key_cannot_describe_is_reported_before_the_other_estimator_rules():
    # On its own this space fails the rule about a snapshot for the whole space.
    space = _space("disagg", **AFD_DECODE, estimator_config={"state_snapshot": {"path": "/no/such/snapshot"}})

    assert "estimator_config.state_snapshot applies to one worker role" in _rejected(space)
    assert "search_space.ais_perf_model cannot be used in this config" in _rejected({**space, "ais_perf_model": EST})


def test_timing_model_of_a_role_the_space_does_not_run_is_ignored():
    timing = {"prefill_timing_model": {"type": "fixed"}}

    new = _validated(_space("agg", ais_perf_model=EST, **timing))

    assert _dump(new) == _dump(_validated(_space("agg", **EST, **timing)))
    assert uncovered_reason_search(_space("agg", **timing)) is None


# The estimator settings of one role, in the old spelling.
ROLE_SETTINGS = {"estimation_mode": "op_level", "systems_paths": ["default"]}
# Where a space with an AFD decode role can state old estimator fields.
AFD_OLD_FIELDS = {
    "top level": {"estimation_mode": "op_level", "fallback_policy": "deny"},
    "AFD role": {"role_estimator_controls": {"decode": ROLE_SETTINGS}},
    "ordinary role": {"role_estimator_controls": {"prefill": ROLE_SETTINGS}},
    "forward model of the AFD role": {"decode_forward_model": "op_level"},
}


@pytest.mark.parametrize("place", list(AFD_OLD_FIELDS))
def test_space_with_afd_roles_and_no_key_is_lowered_untouched(place):
    raw = _space("disagg", **AFD_DECODE, **AFD_OLD_FIELDS[place])
    before = copy.deepcopy(raw)

    assert lower_search_space(raw) is raw
    assert raw == before


@pytest.mark.parametrize("place", list(AFD_OLD_FIELDS))
def test_old_fields_of_a_space_with_afd_roles_keep_working_and_are_not_reported(place, tmp_path):
    raw = _space("disagg", **AFD_DECODE, **AFD_OLD_FIELDS[place])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        space = _validated(copy.deepcopy(raw))
        config = SmartSearchConfig.from_yaml(_write_config(tmp_path, raw))

    assert _dump(config.search_space) == _dump(space)
    assert written_old_keys_search(raw) == []
    assert rules.written_old_keys({"search_space": raw}, "search") == []


def test_old_settings_of_an_afd_role_still_move_into_the_timing_of_that_role():
    both = {"role_estimator_controls": {"prefill": ROLE_SETTINGS, "decode": ROLE_SETTINGS}}

    dumped = _dump(_validated(_space("disagg", **AFD_DECODE, **both)))

    assert dumped["afd_roles"]["decode"]["timing"] == ROLE_SETTINGS
    assert dumped["role_estimator_controls"] == {"prefill": ROLE_SETTINGS}


@pytest.mark.parametrize("place", list(AFD_OLD_FIELDS))
def test_key_next_to_afd_roles_is_rejected_wherever_the_old_fields_are(place):
    raw = _space("disagg", **AFD_DECODE, **AFD_OLD_FIELDS[place], ais_perf_model=EST)

    message = _rejected(raw)

    assert f"search_space.ais_perf_model cannot be used in this config: {AFD_REASON}." in message
    assert "cannot be combined with old estimator fields" not in message


def test_key_next_to_afd_roles_is_rejected_before_the_settings_of_the_afd_role_are_read():
    # The role settings contradict the timing of the AFD role, which is an error of its own.
    space = _space("disagg", **AFD_DECODE, role_estimator_controls={"decode": {"estimation_mode": "auto"}})
    space["afd_roles"]["decode"]["timing"] = {"estimation_mode": "op_level"}
    own_error = "role_estimator_controls.decode.estimation_mode conflicts with workers.decode.timing"

    assert own_error in _rejected(space)
    with_key = _rejected({**space, "ais_perf_model": EST})
    assert AFD_REASON in with_key
    assert own_error not in with_key


def test_empty_afd_roles_is_a_space_without_afd():
    empty = {"afd_roles": {}}

    new = _validated(_space("disagg", ais_perf_model=EST, **empty))

    assert _dump(new) == _dump(_validated(_space("disagg", **EST, **empty)))
    assert uncovered_reason_search(_space("disagg", **empty)) is None
    assert written_old_keys_search(_space("disagg", **EST, **empty)) != []


@pytest.mark.parametrize("afd_roles", ["decode", ["decode"]], ids=["string", "list"])
def test_afd_roles_that_are_not_a_mapping_get_the_error_of_the_search_space(afd_roles):
    space = _space("disagg", afd_roles=afd_roles)
    own_error = "afd_roles must be a mapping by language role"

    assert own_error in _rejected(space)
    assert own_error in _rejected({**space, "ais_perf_model": EST})


@pytest.mark.parametrize("modes", [["afd"], ["agg", "afd+pd"]], ids=["afd", "afd+pd next to agg"])
def test_afd_written_as_a_deployment_mode_gets_the_error_of_the_search_space_without_the_key(modes):
    space = _space("both", deployment_mode=modes, **EST)

    assert f"deployment_mode has invalid choice '{modes[-1]}'; allowed: ['disagg', 'agg']" in _rejected(space)
    assert written_old_keys_search(space) == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param(
            {"ais_perf_model": "estimator.yaml"},
            "search_space.ais_perf_model cannot be a string; write the mapping or 'from: <file>'",
            id="string",
        ),
        pytest.param(
            {"deployment_mode": ["disagg"], "ais_perf_model": _entries(prefill=EST)},
            "search_space.ais_perf_model has no entry for worker type decode; it has: prefill",
            id="worker type without an entry",
        ),
        pytest.param(
            {"deployment_mode": ["agg"], "ais_perf_model": _entries(aggregated=EST, decode=EST)},
            "search_space.ais_perf_model has an entry for worker type decode, which this config does not run; "
            "it runs: aggregated",
            id="entry for a worker type the space does not run",
        ),
        pytest.param(
            {"deployment_mode": "agg"},
            "search_space.ais_perf_model needs search_space.deployment_mode written as a list "
            'with at least one mode, got "agg"',
            id="deployment mode that is not a list",
        ),
        pytest.param(
            {"deployment_mode": ["agg", "edge"]},
            'search_space.ais_perf_model cannot be used with search_space.deployment_mode "edge"; '
            "the modes are: disagg, agg",
            id="unknown deployment mode",
        ),
        pytest.param(
            {"deployment_mode": ["afd"]},
            'search_space.ais_perf_model cannot be used with search_space.deployment_mode "afd"; '
            "the modes are: disagg, agg",
            id="AFD written as a deployment mode",
        ),
        pytest.param(
            {"role_estimator_controls": {"agg": "op_level"}},
            "search_space.ais_perf_model needs search_space.role_estimator_controls.agg written as a mapping, "
            'got "op_level"',
            id="role settings that are not a mapping",
        ),
    ],
)
def test_value_or_space_that_cannot_be_lowered_is_reported_by_name(changes, message):
    assert message in _rejected(_space("both", **{"ais_perf_model": EST, **changes}))


def test_active_worker_types_follow_the_deployment_modes():
    assert active_worker_types_search({"deployment_mode": ["agg"]}) == ("aggregated",)
    assert active_worker_types_search({"deployment_mode": ["disagg"]}) == ("prefill", "decode")
    assert active_worker_types_search({"deployment_mode": ["disagg", "agg"]}) == ("aggregated", "prefill", "decode")
    assert active_worker_types_search({}) == ("aggregated", "prefill", "decode")
    with pytest.raises(AisPerfModelError, match="written as a list with at least one mode"):
        active_worker_types_search({"deployment_mode": []})


def test_deployment_mode_written_as_a_tuple_reads_like_the_list():
    # Callers that build a space in Python may pass a tuple; a YAML file always gives a list.
    value = _entries(prefill=EST, decode=OTHER)
    old = _old_spelling("disagg", {}, {"prefill": EST, "decode": OTHER})
    new = _space("disagg", ais_perf_model=value)
    as_tuple = {"deployment_mode": ("disagg",)}

    assert _dump(_validated({**old, **as_tuple})) == _dump(_validated(old))
    assert _dump(_validated({**new, **as_tuple})) == _dump(_validated(old))
    assert lower_search_space({**new, **as_tuple}) == {**lower_search_space(new), **as_tuple}
    assert active_worker_types_search(as_tuple) == ("prefill", "decode")
    assert written_old_keys_search({**old, **as_tuple}) == written_old_keys_search(old) != []


# --- The field on the search space model ------------------------------------------------------


def test_schema_gains_one_optional_property_and_dumps_never_have_it():
    schema = SearchSpace.model_json_schema(mode="validation")

    assert schema["properties"]["ais_perf_model"] == {
        "anyOf": [
            {"additionalProperties": True, "type": "object"},
            {"items": {"additionalProperties": True, "type": "object"}, "type": "array"},
            {"type": "null"},
        ],
        "default": None,
        "title": "Ais Perf Model",
    }
    assert "ais_perf_model" not in schema["required"]
    assert "ais_perf_model" not in SearchSpace.model_json_schema(mode="serialization")["properties"]

    space = _validated(_space("agg", ais_perf_model=EST))
    assert space.ais_perf_model == _entries(aggregated=EST)
    assert "ais_perf_model" not in space.model_dump()
    assert "ais_perf_model" not in space.model_dump(mode="json", exclude_none=True)
    assert "ais_perf_model" not in space.model_dump_json()
    assert SearchSpace.model_validate_json(space.model_dump_json()).ais_perf_model is None


def test_validating_a_space_never_warns():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validated(_old_spelling("both", EST, {"agg": OTHER}))
        _validated(_space("both", ais_perf_model=_entries(aggregated=EST, prefill=OTHER, decode=EST)))


# --- Old fields a config states ---------------------------------------------------------------


def test_written_old_keys_lists_the_fields_the_author_stated():
    space = _space(
        "both",
        prefill_fpm_parquet_path="fpm.parquet",
        agg_forward_model="op_level",
        role_estimator_controls={"agg": {"database_mode": "HYBRID"}, "decode": {}},
        systems_path="default",
        estimation_mode="op_level",
        database_mode="silicon",
    )
    paths = [
        "estimation_mode",
        "systems_path",
        "role_estimator_controls.agg.database_mode",
        "agg_forward_model",
        "prefill_fpm_parquet_path",
    ]

    assert written_old_keys_search(space) == paths
    assert rules.written_old_keys({"search_space": space}, "search") == [f"search_space.{path}" for path in paths]


def test_null_transfer_policy_of_a_role_is_a_stated_old_field():
    space = _space("disagg", transfer_policy="xshape", **NULL_DECODE_POLICY)

    requests = _requests(_validated(space), "disagg")

    # The null is a choice: the decode role runs without the policy the space states.
    assert (requests["prefill"]["transfer_policy"], requests["decode"]["transfer_policy"]) == (["xshape"], None)
    assert written_old_keys_search(space) == ["transfer_policy", "role_estimator_controls.decode.transfer_policy"]


def test_null_of_any_other_role_setting_is_not_a_stated_old_field():
    nulls = {name: None for name in rules.ROLE_ESTIMATOR_KEYS if name != "transfer_policy"}

    assert written_old_keys_search(_space("disagg", role_estimator_controls={"decode": nulls})) == []


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"transfer_policy": "xshape"},
        {"ais_perf_model": {"transfer_policy": "xshape"}},
        {"ais_perf_model": _entries(prefill={"estimation_mode": "op_level"}, decode={})},
    ],
    ids=["defaults", "top-level policy", "key for the whole space", "key per role"],
)
def test_dump_of_a_space_that_wrote_no_role_transfer_policy_states_none(fields):
    dumped = _validated(_space("disagg", **fields)).model_dump(mode="json")

    stated = written_old_keys_search(dumped)

    assert not [path for path in stated if path.endswith(".transfer_policy")]


@pytest.mark.parametrize("mode", ["agg", "disagg", "both"])
def test_a_dump_full_of_defaults_states_no_old_field(mode):
    dumped = _validated(_space(mode, role_estimator_controls={role: {} for role in ROLES[mode]})).model_dump(
        mode="json"
    )

    assert rules.written_old_keys({"search_space": dumped}, "search") == []


@pytest.mark.parametrize(("changes", "reason"), UNCOVERED)
def test_old_fields_of_a_space_the_key_cannot_describe_are_not_reported(changes, reason):
    space = _space("both", **EST, **changes)

    assert written_old_keys_search(space) == []
    assert rules.written_old_keys({"search_space": space}, "search") == []


@pytest.mark.parametrize(
    "space",
    [
        None,
        "text",
        ["agg"],
        {"estimation_mode": "op_level", "deployment_mode": "agg"},
        {"estimation_mode": "op_level", "deployment_mode": ["edge"]},
        {"estimation_mode": "op_level", "role_estimator_controls": "op_level"},
        {"estimation_mode": "op_level", "role_estimator_controls": {"agg": "op_level"}},
    ],
    ids=["null", "string", "list", "mode not a list", "unknown mode", "role settings", "settings of one role"],
)
def test_written_old_keys_never_fails_on_a_space_validation_will_reject(space):
    assert written_old_keys_search(space) == []
    assert rules.written_old_keys({"search_space": space}, "search") == []


def test_old_top_level_keys_are_the_estimator_fields_of_the_search_space():
    # The search space has the settings a role may override, the legacy one-path field and the profile.
    assert set(_OLD_TOP_LEVEL_KEYS) == {*rules.ROLE_ESTIMATOR_KEYS, "systems_path", "fpm_profile"}
    assert set(_OLD_TOP_LEVEL_KEYS) <= set(SearchSpace.model_fields)


# A value the search space takes for each estimator setting of a role.
ROLE_SETTING_VALUES = {
    "estimation_mode": "op_level",
    "fallback_policy": "deny",
    "estimator_config": {},
    "systems_paths": ["default"],
    "database_mode": "SILICON",
    "transfer_policy": "off",
}
# Every field a role-level setting could be named after: the engine's shared settings and a worker's timing.
ROLE_SETTING_CANDIDATES = sorted({*EstimatorPolicyConfig.model_fields, *TimingConfig.model_fields, "fpm_profile"})


@pytest.mark.parametrize("name", ROLE_SETTING_CANDIDATES)
def test_role_estimator_controls_take_the_role_estimator_keys_and_no_other_field(name):
    """The search space keeps its own list of the settings a role may override; the old-field rules use ours."""
    controls = {"agg": {name: ROLE_SETTING_VALUES.get(name, "anything")}}
    raw = _space("agg", role_estimator_controls=controls)

    if name in rules.ROLE_ESTIMATOR_KEYS:
        assert SearchSpace.model_validate(raw).role_estimator_controls["agg"][name] is not None
    else:
        # The text is the search space's own. It shows the name is refused as a setting of a role, which
        # is what keeps ROLE_ESTIMATOR_KEYS equal to the list the search space checks against.
        with pytest.raises(ValidationError, match="unknown estimator override for role 'agg'"):
            SearchSpace.model_validate(raw)


# --- Reading a search config file -------------------------------------------------------------


def _write_config(folder: Path, space: dict[str, Any]) -> Path:
    path = folder / "search.yaml"
    path.write_text(yaml.safe_dump({"search_space": space, "workload": WORKLOAD}))
    return path


def test_from_yaml_starts_relative_paths_at_the_folder_of_the_file(tmp_path, monkeypatch):
    folder = tmp_path / "configs"
    (folder / "data").mkdir(parents=True)
    (folder / "estimator.yaml").write_text("estimation_mode: op_level\nsystems_paths: [data]\n")
    path = _write_config(folder, _space("agg", ais_perf_model={"from": "estimator.yaml", "database_mode": "HYBRID"}))
    monkeypatch.chdir(tmp_path)

    with warnings.catch_warnings():
        # A file that states no old field reads without a warning.
        warnings.simplefilter("error")
        config = SmartSearchConfig.from_yaml(Path("configs") / "search.yaml")
        absolute = SmartSearchConfig.from_yaml(path)

    fields = {
        "estimation_mode": "op_level",
        "database_mode": "HYBRID",
        "systems_paths": [os.path.realpath(folder / "data")],
    }
    assert _dump(config.search_space) == _dump(_validated(_space("agg", **fields)))
    assert _dump(absolute.search_space) == _dump(config.search_space)


def test_from_yaml_warns_once_about_the_old_fields_of_the_file(tmp_path):
    old_fields = {"estimation_mode": "op_level", "role_estimator_controls": {"agg": {"database_mode": "HYBRID"}}}
    path = _write_config(tmp_path, _space("agg", **old_fields))

    with pytest.warns(AisPerfModelDeprecationWarning) as caught:
        config = SmartSearchConfig.from_yaml(path)

    assert [str(warning.message) for warning in caught] == [
        f"{DEPRECATION_PREFIX}search_space.estimation_mode, search_space.role_estimator_controls.agg.database_mode "
        "are deprecated and will be removed in the next release; use search_space.ais_perf_model "
        f"(run 'aisim perf-model migrate {shlex.quote(str(path))}' to convert the file)"
    ]
    # The warning points at the line that asked for the file.
    assert caught[0].filename == __file__
    assert _dump(config.search_space) == _dump(_validated(_space("agg", **old_fields)))


def test_from_yaml_warns_about_a_null_transfer_policy_of_a_role(tmp_path):
    path = _write_config(tmp_path, _space("disagg", **NULL_DECODE_POLICY))

    with pytest.warns(AisPerfModelDeprecationWarning) as caught:
        config = SmartSearchConfig.from_yaml(path)

    assert [str(warning.message) for warning in caught] == [
        f"{DEPRECATION_PREFIX}search_space.role_estimator_controls.decode.transfer_policy is deprecated "
        "and will be removed in the next release; use search_space.ais_perf_model "
        f"(run 'aisim perf-model migrate {shlex.quote(str(path))}' to convert the file)"
    ]
    assert _dump(config.search_space) == _dump(_validated(_space("disagg", **NULL_DECODE_POLICY)))


def test_from_yaml_says_to_remove_old_fields_that_repeat_the_key(tmp_path):
    # migrate does not merge old fields into a key that is already set, so the hint must not send the author there.
    key = {"transfer_policy": "xshape"}
    path = _write_config(tmp_path, _space("agg", ais_perf_model=key, transfer_policy=["xshape"]))

    with pytest.warns(AisPerfModelDeprecationWarning) as caught:
        config = SmartSearchConfig.from_yaml(path)

    assert [str(warning.message) for warning in caught] == [
        f"{DEPRECATION_PREFIX}search_space.transfer_policy is deprecated and will be removed in the next release; "
        "use search_space.ais_perf_model (it is already set, so remove the old fields)"
    ]
    assert _dump(config.search_space) == _dump(_validated(_space("agg", ais_perf_model=key)))


@pytest.mark.parametrize("mode", ["agg", "disagg", "both"])
def test_from_yaml_of_a_saved_default_space_does_not_warn(tmp_path, mode):
    path = _write_config(tmp_path, _validated(_space(mode)).model_dump(mode="json"))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = SmartSearchConfig.from_yaml(path)

    assert _dump(config.search_space) == _dump(_validated(_space(mode)))


def test_from_yaml_of_a_saved_space_reports_the_fields_the_key_was_written_into(tmp_path):
    # A dump never holds the key, so a saved space reads like one written with the old fields.
    saved = _dump(_validated(_space("agg", ais_perf_model={"estimation_mode": "op_level"})))
    path = _write_config(tmp_path, saved)

    with pytest.warns(AisPerfModelDeprecationWarning, match="search_space.estimation_mode is deprecated"):
        config = SmartSearchConfig.from_yaml(path)

    assert _dump(config.search_space) == saved


def test_from_yaml_of_a_file_that_does_not_validate_fails_without_a_warning(tmp_path):
    path = _write_config(tmp_path, _space("agg", estimation_mode="op_level", no_such_field=1))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValidationError, match="no_such_field"):
            SmartSearchConfig.from_yaml(path)


def test_from_yaml_reports_a_key_it_cannot_read(tmp_path):
    empty = _write_config(tmp_path, _space("agg", ais_perf_model={}))
    with pytest.raises(AisPerfModelError, match="the config is empty"):
        SmartSearchConfig.from_yaml(empty)

    text = _write_config(tmp_path, _space("agg", ais_perf_model="estimator.yaml"))
    with pytest.raises(ValidationError, match="search_space.ais_perf_model cannot be a string"):
        SmartSearchConfig.from_yaml(text)


WORKLOAD_TEXT = "workload: {isl: 128, osl: 16, request_rate: 2, num_request_ratio: 3}\n"


def _space_text(*space_lines: str) -> str:
    """The text of a search space that is written line by line."""
    return "search_space:\n" + "".join(f"  {line}\n" for line in space_lines)


def _config_text(*space_lines: str) -> str:
    """The text of a search config file whose search space is written line by line."""
    return _space_text(*space_lines) + WORKLOAD_TEXT


def test_from_yaml_rejects_a_key_written_twice_inside_the_key(tmp_path):
    path = tmp_path / "search.yaml"
    path.write_text(
        _config_text(
            f"model_name: {MODEL}",
            f"hardware_sku: {SYSTEM}",
            "backend: [vllm]",
            "deployment_mode: [agg]",
            "ais_perf_model:",
            "  estimation_mode: op_level",
            "  fallback_policy: allow",
            "  estimation_mode: fpm_regression",
        )
    )
    # The YAML reader keeps the last value, and the search would follow it without a word.
    assert yaml.safe_load(path.read_text())["search_space"]["ais_perf_model"]["estimation_mode"] == "fpm_regression"

    with pytest.raises(AisPerfModelError) as failure:
        SmartSearchConfig.from_yaml(path)

    assert str(failure.value) == (
        f"search_space.ais_perf_model in {path}: the key 'estimation_mode' is written twice (lines 7 and 9); "
        "write it once"
    )


def test_from_yaml_reads_a_repeat_outside_the_key_and_a_merge_inside_it_as_before(tmp_path):
    path = tmp_path / "search.yaml"
    path.write_text(
        _config_text(
            "model_name: another/model",
            f"model_name: {MODEL}",
            f"hardware_sku: {SYSTEM}",
            "backend: [vllm]",
            "deployment_mode: [disagg]",
            "ais_perf_model:",
            "  - &prefill",
            "    worker_type: prefill",
            "    estimation_mode: op_level",
            "    fallback_policy: allow",
            # The merge brings worker_type in, and the entry writes it once.
            "  - <<: *prefill",
            "    worker_type: decode",
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = SmartSearchConfig.from_yaml(path)

    old_spelling = _space("disagg", estimation_mode="op_level", fallback_policy="allow")
    assert _dump(config.search_space) == _dump(_validated(old_spelling))


# The four lines every search space of the tests below starts with, so its first extra line is line 6.
SPACE_LINES = (f"model_name: {MODEL}", f"hardware_sku: {SYSTEM}", "backend: [vllm]", "deployment_mode: [agg]")
OP_LEVEL = "ais_perf_model: {estimation_mode: op_level}"
REGRESSION = "ais_perf_model: {estimation_mode: fpm_regression, fallback_policy: allow}"
OLD_LINES = ("estimation_mode: fpm_regression", "fallback_policy: allow")


@pytest.mark.parametrize(
    ("text", "kept", "problem"),
    [
        pytest.param(
            _config_text(*SPACE_LINES, OP_LEVEL, REGRESSION),
            "fpm_regression",
            "the key 'ais_perf_model' is written twice (lines 6 and 7)",
            id="the key twice",
        ),
        pytest.param(
            f"search_space: {{model_name: {MODEL}, hardware_sku: {SYSTEM}, {OP_LEVEL}, {REGRESSION}}}\n{WORKLOAD_TEXT}",
            "fpm_regression",
            "the key 'ais_perf_model' is written twice (line 1)",
            id="the key twice on one line",
        ),
        pytest.param(
            _space_text(*SPACE_LINES, OP_LEVEL) + _space_text(*SPACE_LINES, *OLD_LINES) + WORKLOAD_TEXT,
            "fpm_regression",
            "the key 'search_space' is written twice (lines 1 and 7)",
            id="the search space twice, the key in the first",
        ),
        pytest.param(
            _space_text(*SPACE_LINES, *OLD_LINES) + _space_text(*SPACE_LINES, OP_LEVEL) + WORKLOAD_TEXT,
            "op_level",
            "the key 'search_space' is written twice (lines 1 and 8)",
            id="the search space twice, the key in the last",
        ),
        pytest.param(
            _space_text(*SPACE_LINES, OP_LEVEL) + _space_text(*SPACE_LINES, REGRESSION) + WORKLOAD_TEXT,
            "fpm_regression",
            "the key 'search_space' is written twice (lines 1 and 7)",
            id="the search space twice, the key in both",
        ),
    ],
)
def test_from_yaml_rejects_the_key_or_the_search_space_written_twice(tmp_path, text, kept, problem):
    path = tmp_path / "search.yaml"
    path.write_text(text)
    # The YAML reader keeps the last one, which alone is a config, and the search would follow it without a word.
    assert SmartSearchConfig.model_validate(yaml.safe_load(text)).search_space.estimation_mode == kept

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AisPerfModelError) as failure:
            SmartSearchConfig.from_yaml(path)

    assert str(failure.value) == f"search_space.ais_perf_model in {path}: {problem}; write it once"


def test_from_yaml_reads_a_search_space_written_twice_without_the_key_as_before(tmp_path):
    # Neither search space writes ais_perf_model, so the last one is the config and the first says nothing.
    first = _space_text("model_name: another/model", "hardware_sku: h100_sxm", "estimation_mode: op_level")
    path = tmp_path / "search.yaml"
    path.write_text(first + _config_text(*SPACE_LINES))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = SmartSearchConfig.from_yaml(path)

    assert _dump(config.search_space) == _dump(_validated(_space("agg")))


# A Dynamo planner config can hold a mapping of another shape under the same name: its roles and
# the dataset profile of each. Here it reads as one entry with two keys the search cannot check.
ROLES_AND_PROFILES = {"roles": {"decode": {"model": MODEL}}, "dataset_profiles": {"decode": "profile"}}


@pytest.mark.parametrize(
    ("mode", "worker_type"), [("agg", "aggregated"), ("disagg", "prefill"), ("both", "aggregated")]
)
def test_mapping_of_roles_and_dataset_profiles_is_rejected_as_an_entry_with_two_keys_the_search_cannot_check(
    mode, worker_type
):
    raw = _space(mode, ais_perf_model=ROLES_AND_PROFILES)
    message = (
        f"search_space.ais_perf_model entry for {worker_type}: roles, dataset_profiles cannot be checked "
        "in a search config. The search space does not state them (the search chooses tp and similar values "
        "for each candidate). Drop these keys from the entry, or from the template file when "
        "'from: <file>' brings them in."
    )

    with pytest.raises(ValidationError) as rejected:
        SearchSpace.model_validate(raw)
    assert [error["msg"] for error in rejected.value.errors()] == [f"Value error, {message}"]
    with pytest.raises(AisPerfModelError) as failure:
        lower_search_space(raw)
    assert str(failure.value) == message


# --- Module boundary --------------------------------------------------------------------------


def test_shared_rules_load_without_the_search_modules():
    # Predict and recommend configs use the shared rules and must not load the search side with them.
    probe = (
        "import sys\n"
        "import aisim.config.ais_perf_model\n"
        "loaded = [name for name in sys.modules if name.startswith('aisim.sweeper')]\n"
        "assert not loaded, loaded\n"
    )

    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30)

    assert result.returncode == 0, result.stdout + result.stderr
