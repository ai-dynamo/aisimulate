# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract of the ``ais_perf_model`` loader.

The loader owns value forms, shape, paths, ``from``, template merging and
fill. It does not know the canonical fields: several configs here use made-up
field values on purpose.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import types
import warnings
from pathlib import Path
from typing import Any

import pytest

from aisimulate_core import ais_perf_model as apm
from aisimulate_core.ais_perf_model import AisPerfModelError, expand_template, load, resolve

from .sdk.test_dataset_contracts import dataset  # noqa: F401

pytestmark = pytest.mark.unit


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


# --- Value forms ------------------------------------------------------------------------------


def test_public_constants():
    assert apm.WORKER_TYPES == ("prefill", "decode", "aggregated")
    assert apm.ESTIMATOR_KEYS == (
        "estimation_mode",
        "fallback_policy",
        "estimator_config",
        "systems_paths",
        "database_mode",
        "transfer_policy",
        "fpm_profile",
    )
    assert issubclass(AisPerfModelError, ValueError)
    assert issubclass(apm.AisPerfModelDeprecationWarning, FutureWarning)


def test_inline_json_mapping_is_a_bare_document(tmp_path):
    document = load('  {"estimation_mode": "op_level", "systems_paths": ["data"]}', base_dir=tmp_path)
    expected = {"estimation_mode": "op_level", "systems_paths": [str(tmp_path / "data")]}
    assert document.bare
    assert document.source is None
    assert document.entries == (expected,)
    assert document.worker_types == ()
    assert document.to_data() == expected


def test_inline_json_list_keeps_document_order():
    document = load('\n [{"worker_type": "prefill", "tp": 1}, {"worker_type": "decode", "tp": 2}]')
    assert not document.bare
    assert document.source is None
    assert document.worker_types == ("prefill", "decode")
    assert document.to_data() == [{"worker_type": "prefill", "tp": 1}, {"worker_type": "decode", "tp": 2}]


@pytest.mark.parametrize("as_path", [True, False])
def test_yaml_file_as_str_or_path(tmp_path, as_path):
    file = _write(
        tmp_path / "conf" / "model.yaml",
        """
        - worker_type: prefill
          systems_paths: [data, default]
        - worker_type: decode
          estimation_mode: fpm_regression
        """,
    )
    document = load(file if as_path else str(file))
    assert document.source == str(file)
    assert not document.bare
    assert document.to_data() == [
        {"worker_type": "prefill", "systems_paths": [str(tmp_path / "conf" / "data"), "default"]},
        {"worker_type": "decode", "estimation_mode": "fpm_regression"},
    ]


def test_json_suffix_selects_the_json_parser(tmp_path):
    flow = "{worker_type: decode, tp: 2}"
    assert load(_write(tmp_path / "flow.yaml", flow)).to_data() == {"worker_type": "decode", "tp": 2}
    assert load(_write(tmp_path / "no-suffix", flow)).to_data() == {"worker_type": "decode", "tp": 2}
    assert load(_write(tmp_path / "ok.JSON", '{"worker_type": "decode", "tp": 2}')).bare
    with pytest.raises(AisPerfModelError, match="invalid JSON") as error:
        load(_write(tmp_path / "flow.json", flow))
    assert str(tmp_path / "flow.json") in str(error.value)


def test_relative_file_path_starts_at_base_dir(tmp_path, monkeypatch):
    file = _write(tmp_path / "conf" / "model.yaml", "systems_paths: [data]\n")
    expected = {"systems_paths": [str(tmp_path / "conf" / "data")]}
    document = load("model.yaml", base_dir=tmp_path / "conf")
    assert document.source == str(file)
    assert document.to_data() == expected
    monkeypatch.chdir(tmp_path)
    assert load("conf/model.yaml").to_data() == expected


def test_parsed_data_is_copied_and_not_mutated(tmp_path):
    estimator = {"fpm_learned": {"artifact_path": Path("model.bin")}}
    data = {"worker_type": "decode", "tp": None, "systems_paths": ("rel",), "estimator_config": estimator}
    before = copy.deepcopy(data)
    # Any mapping works, not only dict.
    read_only = types.MappingProxyType({**data, "estimator_config": types.MappingProxyType(estimator)})
    document = load(read_only, base_dir=tmp_path)
    assert document.bare
    assert document.entries == (
        {
            "worker_type": "decode",
            "systems_paths": [str(tmp_path / "rel")],
            "estimator_config": {"fpm_learned": {"artifact_path": str(tmp_path / "model.bin")}},
        },
    )
    document.entries[0]["estimator_config"]["fpm_learned"]["artifact_path"] = "changed"
    assert data == before


def test_parsed_list_and_tuple():
    entries = [{"worker_type": "prefill"}, {"worker_type": "aggregated", "tp": 4}]
    assert load(entries).to_data() == entries
    assert load(tuple(entries)).to_data() == entries
    assert load(entries).worker_types == ("prefill", "aggregated")


def test_to_data_round_trips(tmp_path):
    document = load([{"worker_type": "decode", "systems_paths": ["data"], "from": "base.yaml"}], base_dir=tmp_path)
    data = document.to_data()
    data[0]["systems_paths"].append("changed")
    assert document.entries[0]["systems_paths"] == [str(tmp_path / "data")]
    assert load(document.to_data(), base_dir="/somewhere/else").entries == document.entries


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "the value is empty"),
        ("   ", "the value is empty"),
        (None, "expected one config (a mapping) or a list of configs, got null"),
        (5, "got int 5"),
        (b"{}", "got bytes"),
    ],
)
def test_rejected_value_types(value, message):
    with pytest.raises(AisPerfModelError, match=re.escape(message)):
        load(value)


@pytest.mark.parametrize(("text", "kind"), [("hello\n", 'str "hello"'), ("", "null"), ("# only a comment\n", "null")])
def test_file_that_holds_no_config(tmp_path, text, kind):
    file = _write(tmp_path / "model.yaml", text)
    with pytest.raises(AisPerfModelError) as error:
        load(file)
    assert str(error.value) == f"{file}: expected one config (a mapping) or a list of configs, got {kind}"


def test_byte_order_mark_is_ignored(tmp_path):
    # Some editors and shells put a byte order mark at the start of a file or of a value.
    assert load('﻿{"tp": 1}').to_data() == {"tp": 1}
    assert load('﻿ [{"worker_type": "decode", "tp": 1}]').to_data() == [{"worker_type": "decode", "tp": 1}]
    as_json = tmp_path / "model.json"
    as_json.write_bytes(b'\xef\xbb\xbf{"tp": 1}')
    as_yaml = tmp_path / "model.yaml"
    as_yaml.write_bytes(b"\xef\xbb\xbftp: 1\n")
    assert load(as_json).to_data() == {"tp": 1}
    assert load(as_yaml).to_data() == {"tp": 1}
    assert resolve({"from": str(as_json), "dp": 2}) == {"tp": 1, "dp": 2}
    assert resolve({"from": str(as_yaml), "dp": 2}) == {"tp": 1, "dp": 2}


@pytest.mark.parametrize("name", ["[prod].yaml", "{a}.yaml"])
def test_file_name_that_starts_like_inline_json(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    # There is no such file: the value is broken JSON and nothing more.
    with pytest.raises(AisPerfModelError) as error:
        load(name)
    assert str(error.value).startswith("ais_perf_model (inline JSON): invalid JSON: ")
    assert "is read as inline JSON" not in str(error.value)

    # The file exists: the error says why it was not read and how to read it.
    file = _write(tmp_path / name, "tp: 1\n")
    for options in ({}, {"base_dir": tmp_path}):
        with pytest.raises(AisPerfModelError) as error:
            load(name, **options)
        message = str(error.value)
        assert message.startswith("ais_perf_model (inline JSON): invalid JSON: ")
        assert message.endswith(
            f"; a value that starts with {{ or [ is read as inline JSON. To read the file {file}, write ./{name}"
        )
        assert isinstance(error.value.__cause__, json.JSONDecodeError)
    assert load(f"./{name}").to_data() == {"tp": 1}
    assert load(Path(name)).to_data() == {"tp": 1}
    assert load(str(file)).to_data() == {"tp": 1}
    # from is always a path.
    assert resolve({"from": name, "dp": 2}) == {"tp": 1, "dp": 2}


# --- Duplicate keys ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("tp: 1\ndp: 2\ntp: 3\n", "tp"),
        ("estimator_config:\n  correction:\n    mode: fixed\n    mode: auto\n", "mode"),
        ("- worker_type: prefill\n- worker_type: decode\n  tp: 1\n  tp: 1\n", "tp"),
        ("estimator_config: {correction: {mode: fixed, mode: fixed}}\n", "mode"),
    ],
)
def test_yaml_duplicate_key_is_an_error(tmp_path, text, key):
    file = _write(tmp_path / "model.yaml", text)
    with pytest.raises(AisPerfModelError, match=f"duplicate key '{key}'") as error:
        load(file)
    assert str(error.value).startswith(f"{file}: invalid YAML: ")


def test_yaml_merge_keys_and_anchors_are_not_duplicates(tmp_path):
    file = _write(
        tmp_path / "model.yaml",
        """
        - &shared
          worker_type: prefill
          estimation_mode: op_level
          tp: 1
        - <<: *shared
          worker_type: decode
          tp: 2
        """,
    )
    assert load(file).to_data() == [
        {"worker_type": "prefill", "estimation_mode": "op_level", "tp": 1},
        {"worker_type": "decode", "estimation_mode": "op_level", "tp": 2},
    ]


def test_json_duplicate_key_is_an_error(tmp_path):
    file = _write(tmp_path / "model.json", '{"estimator_config": {"a": 1, "b": 2, "a": 1}}')
    with pytest.raises(AisPerfModelError, match="duplicate key 'a'") as error:
        load(file)
    assert str(error.value).startswith(f"{file}: invalid JSON: ")
    with pytest.raises(AisPerfModelError, match=r"ais_perf_model \(inline JSON\): invalid JSON: duplicate key 'tp'"):
        load('[{"worker_type": "decode", "tp": 1, "tp": 2}]')


def test_template_duplicate_key_is_an_error(tmp_path):
    template = _write(tmp_path / "base.yaml", "tp: 1\ntp: 2\n")
    with pytest.raises(AisPerfModelError, match="duplicate key 'tp'") as error:
        expand_template({"from": str(template)})
    assert str(template) in str(error.value)


# --- Shape ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({}, "ais_perf_model: the config is empty; write at least one key"),
        ([], "ais_perf_model: the list is empty; write at least one config"),
        ({"tp": None, "estimator_config": {}}, "the config is empty"),
        ([{"worker_type": "prefill"}, 5], "ais_perf_model: entry 2: expected a mapping, got int 5"),
        ([{"worker_type": "prefill"}, [{"worker_type": "decode"}]], "entry 2: expected a mapping, got a list"),
        ([{"tp": 1}], "entry 1: worker_type is missing; every config in a list needs one of: prefill, decode"),
        ([{"worker_type": None, "tp": 1}], "entry 1: worker_type is missing"),
        ({"worker_type": "encoder"}, 'worker_type is "encoder"; use one of: prefill, decode, aggregated'),
        ({"worker_type": 3}, "worker_type is 3; use one of"),
        ([{"worker_type": "Decode"}], 'entry 1: worker_type is "Decode"'),
        (
            [{"worker_type": "prefill"}, {"worker_type": "decode"}, {"worker_type": "prefill", "tp": 2}],
            "entries 1 and 3 both have worker_type 'prefill'; a worker type may appear once",
        ),
        ({1: "x"}, "key 1 is not a string; quote it"),
        ({"estimator_config": {"a": {2: 1}}}, "estimator_config.a: key 2 is not a string"),
        ({"tp": float("nan")}, "tp is nan; only finite numbers are allowed"),
        ({"estimator_config": {"scale": [1.0, float("inf")]}}, "estimator_config.scale[1] is inf"),
        ('{"tp": NaN}', "NaN is not a finite number"),
        ('{"tp": -Infinity}', "-Infinity is not a finite number"),
        (
            {"systems_paths": "data"},
            'systems_paths must be a list of directories (or the word default), got str "data"',
        ),
        ({"tp": object()}, "tp has a value of type object"),
        ({"tags": {"a", "b"}}, "tags has a value of type set"),
    ],
)
def test_shape_errors(value, message):
    with pytest.raises(AisPerfModelError, match=re.escape(message)):
        load(value)


def test_yaml_values_that_json_cannot_carry_are_errors(tmp_path):
    file = _write(tmp_path / "key.yaml", "1: x\n")
    with pytest.raises(AisPerfModelError, match=re.escape(f"{file}: key 1 is not a string; quote it")):
        load(file)
    with pytest.raises(AisPerfModelError, match="key True is not a string; quote it"):
        load(_write(tmp_path / "bool.yaml", "on: x\n"))
    with pytest.raises(AisPerfModelError, match="released has a value of type date"):
        load(_write(tmp_path / "date.yaml", "released: 2026-01-01\n"))
    with pytest.raises(AisPerfModelError, match="tp is nan"):
        load(_write(tmp_path / "nan.yaml", "tp: .nan\n"))


def test_field_names_and_values_are_not_validated():
    # Rust owns the canonical fields. The loader passes unknown ones through.
    config = {"worker_type": "decode", "estimation_mode": "no_such_mode", "made_up_key": {"x": [1, True, None]}}
    assert load(config).to_data() == config
    assert resolve(config) == config


def _alias_bomb(levels: int, width: int = 9) -> str:
    """YAML in which every key repeats the key before it ``width`` times."""
    lines = ["a0: &a0 [" + ", ".join(["x"] * width) + "]"]
    lines += [f"a{level}: &a{level} [" + ", ".join([f"*a{level - 1}"] * width) + "]" for level in range(1, levels)]
    return "\n".join(lines) + "\n"


def test_yaml_aliases_cannot_expand_without_limit(tmp_path, monkeypatch):
    # Twelve levels are a file of a few hundred bytes that would expand to hundreds of billions of
    # values. The loader stops at the limit. It does not copy the values first.
    bomb = _write(tmp_path / "bomb.yaml", _alias_bomb(12))
    assert bomb.stat().st_size < 1000
    with pytest.raises(AisPerfModelError) as error:
        load(bomb)
    assert str(error.value) == (
        f"{bomb}: the config has more than 1000000 values; write the values out and do not nest YAML aliases"
    )

    # Three levels hold 921 values.
    small = _write(tmp_path / "small.yaml", _alias_bomb(3))
    monkeypatch.setattr(apm, "_MAX_VALUES", 921)
    assert len(load(small).to_data()["a2"]) == 9
    assert len(resolve({"from": str(small), "tp": 1})["a2"]) == 9
    monkeypatch.setattr(apm, "_MAX_VALUES", 920)
    with pytest.raises(AisPerfModelError, match=re.escape(f"{small}: the config has more than 920 values")):
        load(small)
    with pytest.raises(AisPerfModelError) as error:
        resolve({"from": str(small), "tp": 1})
    assert str(error.value).startswith(
        f"ais_perf_model entry: cannot use the template named by from: {small}: the config has more than 920 values"
    )


def test_value_limit_counts_every_value_of_one_config(monkeypatch):
    monkeypatch.setattr(apm, "_MAX_VALUES", 5)
    # Every scalar, list and mapping below the top level counts once. Each config has its own count.
    five = {"worker_type": "prefill", "a": [1, 2, 3]}
    assert load([five, {**five, "worker_type": "decode"}]).to_data() == [five, {**five, "worker_type": "decode"}]
    assert load({"a": 1, "dropped": None, "c": {"d": {"e": 1}}, "f": 1}).to_data() == {
        "a": 1,
        "c": {"d": {"e": 1}},
        "f": 1,
    }
    for six in ({"a": [1, 2, 3, 4], "b": 1}, {"a": 1, "c": {"d": {"e": 1}}, "f": 1, "g": 1}):
        with pytest.raises(AisPerfModelError) as error:
            load(six)
        assert str(error.value) == (
            "ais_perf_model: the config has more than 5 values; write the values out and do not nest YAML aliases"
        )
    # fill and defaults are read the same way.
    with pytest.raises(AisPerfModelError, match="ais_perf_model entry: fill: the config has more than 5 values"):
        resolve({"a": 1}, fill={"b": [1, 2, 3, 4, 5]})


def _nested(levels: int, wrap) -> Any:
    """The number 1 inside ``levels`` containers, each made by ``wrap`` from the one inside it."""
    value: Any = 1
    for _ in range(levels):
        value = wrap(value)
    return value


@pytest.mark.parametrize(
    ("wrap", "step"),
    [(lambda inner: [inner], "[0]"), (lambda inner: {"a": inner}, ".a")],
    ids=["lists", "mappings"],
)
def test_nesting_limit_names_the_first_value_that_is_too_deep(tmp_path, wrap, step):
    # A key may hold 65 containers, one inside the other: the innermost is 64 levels below the outermost.
    deepest = {"estimator_config": _nested(65, wrap)}
    assert load(deepest).to_data() == deepest
    assert resolve({"tp": 1}, fill=deepest)["estimator_config"] == deepest["estimator_config"]

    too_deep = {"estimator_config": _nested(66, wrap)}
    problem = f"estimator_config{step * 65} is nested more than 64 levels deep"
    with pytest.raises(AisPerfModelError) as error:
        load(too_deep)
    assert str(error.value) == f"ais_perf_model: {problem}"

    # A file, a template named by `from` and fill are read with the same limit.
    file = _write(tmp_path / "deep.json", json.dumps(too_deep))
    with pytest.raises(AisPerfModelError) as error:
        load(file)
    assert str(error.value) == f"{file}: {problem}"
    with pytest.raises(AisPerfModelError) as error:
        resolve({"from": str(file), "tp": 1})
    assert str(error.value) == f"ais_perf_model entry: cannot use the template named by from: {file}: {problem}"
    with pytest.raises(AisPerfModelError) as error:
        resolve({"tp": 1}, fill=too_deep)
    assert str(error.value) == f"ais_perf_model entry: fill: {problem}"


# --- `from` shapes ----------------------------------------------------------------------------


def test_from_shapes(tmp_path):
    def source(value):
        return load({"from": value}, base_dir=tmp_path).entries[0]["from"]

    assert source("templates/base.yaml") == str(tmp_path / "templates" / "base.yaml")
    assert source({"dataset": {"root": "ds", "manifest": "metadata/manifest.json"}}) == {
        "dataset": {"root": str(tmp_path / "ds"), "manifest": "metadata/manifest.json"}
    }
    assert source({"state_snapshot": {"path": "snap"}}) == {"state_snapshot": {"path": str(tmp_path / "snap")}}
    assert source({"state_snapshot": {"path": "/abs/snap", "sha256": "AB12"}}) == {
        "state_snapshot": {"path": "/abs/snap", "sha256": "AB12"}
    }
    # on_tune is kept as written, for resolve and for a caller that lowers the entry itself.
    assert source({"state_snapshot": {"path": "snap", "on_tune": "continue"}}) == {
        "state_snapshot": {"path": str(tmp_path / "snap"), "on_tune": "continue"}
    }
    # Nulls inside from are dropped like null keys of a config.
    assert source({"dataset": None, "state_snapshot": {"path": "snap", "sha256": None}}) == {
        "state_snapshot": {"path": str(tmp_path / "snap")}
    }
    assert load({"from": None, "tp": 1}).to_data() == {"tp": 1}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (
            5,
            "from must be a template file path, {dataset: {root, manifest}} "
            "or {state_snapshot: {path, sha256, on_tune}}; got int 5",
        ),
        ("  ", "from is empty; write a template file path"),
        (["base.yaml"], "from must be a template file path"),
        ({}, "got a mapping with keys: (none)"),
        (
            {"dataset": {"root": "ds", "manifest": "m.json"}, "state_snapshot": {"path": "snap"}},
            "got a mapping with keys: dataset, state_snapshot",
        ),
        ({"file": "base.yaml", "pointer": "/a"}, "got a mapping with keys: file, pointer"),
        ({"dataset": "ds"}, 'from.dataset must be a mapping, got str "ds"'),
        ({"dataset": {"root": "ds"}}, "from.dataset.manifest is missing"),
        ({"dataset": {"manifest": "m.json"}}, "from.dataset.root is missing"),
        (
            {"dataset": {"root": "ds", "manifest": "m.json", "sha256": "ab"}},
            "from.dataset accepts only root, manifest; remove: sha256",
        ),
        (
            {"dataset": {"root": "ds", "manifest": "/abs/m.json"}},
            "from.dataset.manifest must be a path relative to root",
        ),
        ({"dataset": {"root": 5, "manifest": "m.json"}}, "from.dataset.root must be a non-empty string, got int 5"),
        ({"dataset": {"root": "ds", "manifest": " "}}, "from.dataset.manifest must be a non-empty string"),
        ({"state_snapshot": {}}, "from.state_snapshot.path is missing"),
        (
            {"state_snapshot": {"path": "snap", "on_tune": "continue", "mode": "fixed"}},
            "from.state_snapshot accepts only path, sha256, on_tune; remove: mode",
        ),
        (
            {"state_snapshot": {"path": "snap", "on_tune": True}},
            "from.state_snapshot.on_tune must be a non-empty string, got bool true",
        ),
        (
            {"state_snapshot": {"path": "snap", "on_tune": " "}},
            "from.state_snapshot.on_tune must be a non-empty string",
        ),
        ({"state_snapshot": {"path": ["snap"]}}, "from.state_snapshot.path must be a non-empty string, got a list"),
        ({"state_snapshot": {"path": "snap", "sha256": 5}}, "from.state_snapshot.sha256 must be a non-empty string"),
    ],
)
def test_from_shape_errors(value, message):
    with pytest.raises(AisPerfModelError, match=re.escape(message)):
        load({"worker_type": "decode", "from": value})


# --- select -----------------------------------------------------------------------------------


def test_select_from_a_list_returns_a_deep_copy():
    document = load([{"worker_type": "prefill", "tp": 1}, {"worker_type": "decode", "estimator_config": {"a": [1]}}])
    entry = document.select("decode")
    assert entry == {"worker_type": "decode", "estimator_config": {"a": [1]}}
    entry["estimator_config"]["a"].append(2)
    assert document.select("decode") == {"worker_type": "decode", "estimator_config": {"a": [1]}}


def test_select_missing_worker_type_lists_what_is_there(tmp_path):
    document = load([{"worker_type": "prefill"}, {"worker_type": "decode"}])
    with pytest.raises(AisPerfModelError) as error:
        document.select("aggregated")
    assert str(error.value) == "ais_perf_model: no entry for worker type 'aggregated'; it has: prefill, decode"
    file = _write(tmp_path / "model.yaml", "- worker_type: prefill\n")
    with pytest.raises(AisPerfModelError) as error:
        load(file).select("decode")
    assert str(error.value) == f"{file}: no entry for worker type 'decode'; it has: prefill"


def test_select_from_a_bare_config():
    typed = load({"worker_type": "prefill", "tp": 1})
    assert typed.select("prefill") == {"worker_type": "prefill", "tp": 1}
    with pytest.raises(AisPerfModelError, match="no entry for worker type 'decode'; it has: prefill"):
        typed.select("decode")
    untyped = load({"tp": 1})
    assert untyped.select("decode") == {"worker_type": "decode", "tp": 1}
    assert untyped.select("aggregated") == {"worker_type": "aggregated", "tp": 1}
    assert untyped.entries == ({"tp": 1},)


def test_select_rejects_an_unknown_worker_type_name():
    with pytest.raises(AisPerfModelError, match='the requested worker type is "encoder"; use one of: prefill'):
        load({"tp": 1}).select("encoder")


# --- Paths ------------------------------------------------------------------------------------

_PATHS_YAML = """
systems_paths: [data, default, Default, /abs/systems, ~/systems, ../up]
estimator_config:
  state_snapshot: {path: snap, sha256: abc}
  fpm_learned: {artifact_path: learned/model.bin, checkpoint_path: ./learned/ckpt}
  temporary_mixed_composition: {original_a: {manifest_path: mixed/manifest.json}}
  op_level: {execution_profile: {path: profile.json, sha256: profile.json}}
  fpm_interpolation: {fpm_parquet_path: fpm.parquet, other_path: stays/relative}
  correction: {path: stays/relative}
"""


def _anchored(base: Path, home: Path) -> dict:
    return {
        "systems_paths": [
            str(base / "data"),
            "default",
            "Default",
            "/abs/systems",
            str(home / "systems"),
            str(base.parent / "up"),
        ],
        "estimator_config": {
            "state_snapshot": {"path": str(base / "snap"), "sha256": "abc"},
            "fpm_learned": {
                "artifact_path": str(base / "learned" / "model.bin"),
                "checkpoint_path": str(base / "learned" / "ckpt"),
            },
            "temporary_mixed_composition": {"original_a": {"manifest_path": str(base / "mixed" / "manifest.json")}},
            "op_level": {"execution_profile": {"path": str(base / "profile.json"), "sha256": "profile.json"}},
            "fpm_interpolation": {"fpm_parquet_path": str(base / "fpm.parquet"), "other_path": "stays/relative"},
            "correction": {"path": "stays/relative"},
        },
    }


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_paths_in_a_file_start_at_the_file_folder(tmp_path, home, monkeypatch):
    file = _write(tmp_path / "conf" / "model.yaml", _PATHS_YAML)
    monkeypatch.chdir(tmp_path)
    assert load(file).to_data() == _anchored(tmp_path / "conf", home)
    # base_dir does not move the paths inside a file.
    assert load(file, base_dir=tmp_path / "other").to_data() == _anchored(tmp_path / "conf", home)


def test_paths_in_inline_json_and_data_start_at_base_dir(tmp_path, home):
    import yaml

    data = yaml.safe_load(_PATHS_YAML)
    base = tmp_path / "base"
    assert load(json.dumps(data), base_dir=base).to_data() == _anchored(base, home)
    assert load(data, base_dir=str(base)).to_data() == _anchored(base, home)
    assert load([{"worker_type": "decode", **data}], base_dir=base).to_data() == [
        {"worker_type": "decode", **_anchored(base, home)}
    ]


def test_base_dir_defaults_to_the_current_directory(tmp_path, home, monkeypatch):
    import yaml

    monkeypatch.chdir(tmp_path)
    assert load(json.dumps(yaml.safe_load(_PATHS_YAML))).to_data() == _anchored(Path.cwd(), home)
    assert load({"from": "base.yaml"}).to_data() == {"from": str(Path.cwd() / "base.yaml")}
    assert load({"systems_paths": ["data"]}, base_dir="~/conf").to_data() == {
        "systems_paths": [str(home / "conf" / "data")]
    }


def test_from_paths_in_a_file(tmp_path, home):
    file = _write(
        tmp_path / "conf" / "model.yaml",
        """
        - worker_type: prefill
          from: ../templates/base.yaml
        - worker_type: decode
          from: {dataset: {root: ~/datasets/a, manifest: metadata/manifest.json}}
        - worker_type: aggregated
          from: {state_snapshot: {path: snapshots/agg}}
        """,
    )
    assert [entry["from"] for entry in load(file).entries] == [
        str(tmp_path / "templates" / "base.yaml"),
        {"dataset": {"root": str(home / "datasets" / "a"), "manifest": "metadata/manifest.json"}},
        {"state_snapshot": {"path": str(tmp_path / "conf" / "snapshots" / "agg")}},
    ]


def test_values_the_loader_cannot_anchor_are_left_for_rust(tmp_path):
    config = {
        "systems_paths": ["", 5],
        "estimator_config": {"state_snapshot": {"path": 5}, "fpm_learned": "not a mapping", "op_level": None},
    }
    assert load(config, base_dir=tmp_path).to_data() == config


# --- Nulls ------------------------------------------------------------------------------------


def test_null_keys_and_empty_containers_are_dropped():
    document = load(
        [
            {"worker_type": "decode", "tp": None, "systems_paths": [], "estimator_config": {}, "dp": 0},
            {"worker_type": "prefill", "systems_paths": None, "estimator_config": None, "note": ""},
        ]
    )
    assert document.to_data() == [{"worker_type": "decode", "dp": 0}, {"worker_type": "prefill", "note": ""}]


def test_null_inside_estimator_config_is_kept():
    # Below the top level a null is a value. Rust gives it its meaning.
    config = {"estimator_config": {"correction": {"factor_bound": None, "steps": [None, 1]}}}
    assert load(config).to_data() == config
    assert resolve(config) == config


def test_null_inside_estimator_config_stops_defaults_and_is_compared():
    entry = {"tp": None, "estimator_config": {"a": {"x": None}, "b": None}}
    defaults = {"tp": 1, "estimator_config": {"a": {"x": 1, "y": 2}, "b": {"z": 3}, "c": 4}}
    # At the top level a null key is missing, so the default fills it. Below the top level the
    # null is the value the user wrote, so the default does not replace it.
    assert resolve(entry, defaults=defaults) == {
        "tp": 1,
        "estimator_config": {"a": {"x": None, "y": 2}, "b": None, "c": 4},
    }
    same = {"estimator_config": {"a": {"x": None}, "b": None}}
    assert resolve(entry, fill=same) == same
    with pytest.raises(AisPerfModelError) as error:
        resolve(entry, fill={"estimator_config": {"a": {}, "b": None}})
    assert str(error.value) == (
        'ais_perf_model entry: estimator_config is {"a": {"x": null}, "b": null} but the caller requires '
        '{"a": {}, "b": null}; write the same value or remove estimator_config'
    )


# --- Templates --------------------------------------------------------------------------------

_TEMPLATE = """
estimation_mode: op_level
fallback_policy: deny
systems_paths: [data, more]
tp: 1
dp: 1
estimator_config:
  correction: {enabled: true, factors: [1, 2], mode: fixed}
  op_level: {execution_profile: {path: profile.json, sha256: abc}}
"""


def test_template_merge_entry_wins_and_estimator_config_merges(tmp_path):
    templates = tmp_path / "templates"
    _write(templates / "base.yaml", _TEMPLATE)
    document = load(
        {
            "from": "../templates/base.yaml",
            "worker_type": "decode",
            "tp": 2,
            "systems_paths": ["mine"],
            "estimator_config": {
                "correction": {"factors": [3], "mode": None},
                "fpm_learned": {"artifact_path": "model.bin"},
            },
        },
        base_dir=tmp_path / "conf",
    )
    entry = document.select("decode")
    before = copy.deepcopy(entry)
    assert expand_template(entry) == {
        "estimation_mode": "op_level",
        "fallback_policy": "deny",
        "systems_paths": [str(tmp_path / "conf" / "mine")],
        "tp": 2,
        "dp": 1,
        "worker_type": "decode",
        "estimator_config": {
            "correction": {"enabled": True, "factors": [3], "mode": None},
            "op_level": {"execution_profile": {"path": str(templates / "profile.json"), "sha256": "abc"}},
            "fpm_learned": {"artifact_path": str(tmp_path / "conf" / "model.bin")},
        },
    }
    assert entry == before


def test_template_scalar_and_mapping_replace_each_other(tmp_path):
    template = _write(tmp_path / "base.yaml", "estimator_config: {a: {x: 1}, b: 2, c: {y: 1}}\n")
    entry = {"from": str(template), "estimator_config": {"a": 5, "b": {"z: 1": 1}, "c": {}}}
    assert expand_template(entry) == {"estimator_config": {"a": 5, "b": {"z: 1": 1}, "c": {"y": 1}}}
    # A top-level key other than estimator_config is replaced as a whole.
    template = _write(tmp_path / "other.yaml", "fpm_profile: {model: m, deployments: [1]}\n")
    assert expand_template({"from": str(template), "fpm_profile": {"model": "n"}}) == {"fpm_profile": {"model": "n"}}


def test_template_list_gives_the_config_of_the_same_worker_type(tmp_path):
    template = _write(
        tmp_path / "base.json",
        '[{"worker_type": "prefill", "tp": 1, "systems_paths": ["p"]}, {"worker_type": "decode", "tp": 2}]',
    )
    assert expand_template({"from": str(template), "worker_type": "decode", "dp": 4}) == {
        "worker_type": "decode",
        "tp": 2,
        "dp": 4,
    }
    assert expand_template({"from": str(template), "worker_type": "prefill"}) == {
        "worker_type": "prefill",
        "tp": 1,
        "systems_paths": [str(tmp_path / "p")],
    }
    with pytest.raises(AisPerfModelError) as error:
        expand_template({"from": str(template), "worker_type": "aggregated"})
    assert str(error.value) == (
        f"ais_perf_model entry for worker type 'aggregated': the template {template} "
        "has no entry for worker type 'aggregated'; it has: prefill, decode"
    )
    with pytest.raises(AisPerfModelError) as error:
        expand_template({"from": str(template)})
    assert str(error.value) == (
        f"ais_perf_model entry: the template {template} is a list; "
        "set worker_type on the entry to pick one of: prefill, decode"
    )


def test_bare_template_without_worker_type_serves_every_worker_type(tmp_path):
    template = _write(tmp_path / "base.yaml", "tp: 1\n")
    assert expand_template({"from": str(template), "worker_type": "decode"}) == {"worker_type": "decode", "tp": 1}
    assert expand_template({"from": str(template)}) == {"tp": 1}
    assert resolve({"from": str(template)}, worker_type="prefill") == {"worker_type": "prefill", "tp": 1}


def test_bare_template_with_worker_type_serves_only_that_worker_type(tmp_path):
    template = _write(tmp_path / "base.yaml", "worker_type: decode\ntp: 1\n")
    expected = {"worker_type": "decode", "tp": 1}
    assert expand_template({"from": str(template), "worker_type": "decode"}) == expected
    assert resolve({"from": str(template)}, worker_type="decode") == expected
    # An entry without worker_type takes the template's.
    assert expand_template({"from": str(template)}) == expected
    assert resolve({"from": str(template)}) == expected
    # The same rule as select() on a bare document: a decode template is never used for prefill.
    for use in (
        lambda: expand_template({"from": str(template), "worker_type": "prefill"}),
        lambda: resolve({"from": str(template), "worker_type": "prefill"}),
        lambda: resolve({"from": str(template)}, worker_type="prefill"),
        lambda: resolve(load({"from": str(template)}).select("prefill")),
    ):
        with pytest.raises(AisPerfModelError) as error:
            use()
        assert str(error.value) == (
            f"ais_perf_model entry for worker type 'prefill': the template {template} "
            "has no entry for worker type 'prefill'; it has: decode"
        )


def test_template_may_not_contain_from(tmp_path):
    other = _write(tmp_path / "other.yaml", "tp: 1\n")
    template = _write(tmp_path / "base.yaml", f"from: {other}\ndp: 1\n")
    with pytest.raises(AisPerfModelError) as error:
        expand_template({"from": str(template), "worker_type": "decode"})
    assert str(error.value) == (
        f"ais_perf_model entry for worker type 'decode': the template {template} contains from; "
        "a template may not name another source"
    )
    sealed = _write(tmp_path / "sealed.yaml", "from: {state_snapshot: {path: snap}}\n")
    with pytest.raises(AisPerfModelError, match="contains from"):
        resolve({"from": str(sealed)})


def test_template_that_cannot_be_read_names_the_file(tmp_path):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(AisPerfModelError) as error:
        expand_template({"from": str(missing), "worker_type": "decode"})
    message = str(error.value)
    assert message.startswith("ais_perf_model entry for worker type 'decode': cannot use the template named by from: ")
    assert f"cannot read {missing}" in message
    empty = _write(tmp_path / "empty.yaml", "{}\n")
    with pytest.raises(AisPerfModelError, match=re.escape(f"{empty}: the config is empty")):
        expand_template({"from": str(empty)})


def test_expand_template_leaves_other_entries_unchanged(tmp_path):
    for entry in (
        {"worker_type": "decode", "tp": 1, "estimator_config": {"a": None}},
        {"from": {"state_snapshot": {"path": str(tmp_path / "snap"), "sha256": "ab"}}, "tp": 1},
        {"from": {"dataset": {"root": str(tmp_path / "ds"), "manifest": "m.json"}}},
    ):
        before = copy.deepcopy(entry)
        result = expand_template(entry)
        assert result == before
        assert result is not entry
        assert entry == before


# --- resolve: fill, defaults, priority ----------------------------------------------------------


def test_resolve_sets_or_checks_the_requested_worker_type():
    assert resolve({"tp": 1}, worker_type="decode") == {"tp": 1, "worker_type": "decode"}
    assert resolve({"tp": 1, "worker_type": "decode"}, worker_type="decode") == {"tp": 1, "worker_type": "decode"}
    assert resolve({"tp": 1}) == {"tp": 1}
    with pytest.raises(AisPerfModelError) as error:
        resolve({"worker_type": "prefill", "tp": 1}, worker_type="decode")
    assert str(error.value) == "ais_perf_model entry: worker_type is 'prefill' but worker type 'decode' was requested"
    with pytest.raises(AisPerfModelError, match='the requested worker type is "agg"'):
        resolve({"tp": 1}, worker_type="agg")
    with pytest.raises(AisPerfModelError, match="fill has worker_type \"prefill\" but worker type 'decode'"):
        resolve({"tp": 1}, worker_type="decode", fill={"worker_type": "prefill"})


def test_fill_adds_missing_keys_and_checks_written_keys():
    entry = {"worker_type": "decode", "tp": 2, "estimator_config": {"a": {"x": 1}}}
    fill = {"tp": 2, "model": "m", "backend_version": None, "estimator_config": {"a": {"x": 1}}}
    before = copy.deepcopy((entry, fill))
    assert resolve(entry, fill=fill) == {**entry, "model": "m"}
    assert (entry, fill) == before
    with pytest.raises(AisPerfModelError) as error:
        resolve(entry, fill={"tp": 4})
    assert str(error.value) == (
        "ais_perf_model entry for worker type 'decode': tp is 2 but the caller requires 4; "
        "write the same value or remove tp"
    )
    with pytest.raises(AisPerfModelError, match='system is "h100" but the caller requires "h200"'):
        resolve({"system": "h100"}, fill={"system": "h200"})
    with pytest.raises(AisPerfModelError, match=r'estimator_config is \{"a": \{"x": 1\}\} but the caller requires'):
        resolve(entry, fill={"estimator_config": {"a": {"x": 1, "y": 2}}})


def test_fill_keeps_true_and_one_apart():
    assert resolve({"enable_shared_layer": True, "tp": 1}, fill={"enable_shared_layer": True, "tp": 1.0})
    with pytest.raises(AisPerfModelError, match="enable_shared_layer is true but the caller requires 1"):
        resolve({"enable_shared_layer": True}, fill={"enable_shared_layer": 1})


def test_defaults_only_add_what_is_missing():
    entry = {"tp": 2, "estimator_config": {"correction": {"enabled": False, "factors": [1]}}, "fpm_profile": {"a": 1}}
    defaults = {
        "tp": 8,
        "dp": 8,
        "gpu": None,
        "estimator_config": {"correction": {"enabled": True, "mode": "auto", "factors": [9, 9]}, "op_level": {"x": 1}},
        "fpm_profile": {"a": 2, "b": 2},
    }
    before = copy.deepcopy((entry, defaults))
    result = resolve(entry, defaults=defaults)
    assert result == {
        "tp": 2,
        "dp": 8,
        "estimator_config": {"correction": {"enabled": False, "mode": "auto", "factors": [1]}, "op_level": {"x": 1}},
        "fpm_profile": {"a": 1},
    }
    result["estimator_config"]["op_level"]["x"] = 2
    assert (entry, defaults) == before
    assert resolve({"tp": 1}, defaults={"estimator_config": {"a": 1}}) == {"tp": 1, "estimator_config": {"a": 1}}


def test_priority_is_written_then_template_then_fill_then_defaults(tmp_path):
    template = _write(tmp_path / "base.yaml", "tp: 1\ndp: 1\nmodel: m\nestimator_config: {a: {from_template: 1}}\n")
    entry = {"from": str(template), "tp": 2, "estimator_config": {"a": {"written": 1}}}
    fill = {"model": "m", "system": "h200"}
    defaults = {
        "tp": 9,
        "dp": 9,
        "system": "other",
        "backend": "vllm",
        "estimator_config": {"a": {"written": 9, "from_template": 9, "from_defaults": 9}},
    }
    assert resolve(entry, worker_type="decode", fill=fill, defaults=defaults) == {
        "worker_type": "decode",
        "tp": 2,
        "dp": 1,
        "model": "m",
        "system": "h200",
        "backend": "vllm",
        "estimator_config": {"a": {"written": 1, "from_template": 1, "from_defaults": 9}},
    }
    # fill never overrides: a template value that differs is an error too.
    with pytest.raises(AisPerfModelError, match="dp is 1 but the caller requires 5"):
        resolve(entry, fill={"dp": 5})


def test_fill_and_defaults_are_read_like_a_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file = _write(tmp_path / "model.yaml", "systems_paths: [data]\n")
    entry = load(file).select("decode")
    assert resolve(entry, fill={"systems_paths": ["data"]}, defaults={"estimator_config": {}}) == {
        "worker_type": "decode",
        "systems_paths": [str(Path.cwd() / "data")],
    }
    assert resolve({"tp": 1}, defaults={"estimator_config": {"state_snapshot": {"path": "snap"}}}) == {
        "tp": 1,
        "estimator_config": {"state_snapshot": {"path": str(Path.cwd() / "snap")}},
    }
    with pytest.raises(AisPerfModelError, match="fill may not contain from; only an entry can name a source"):
        resolve({"tp": 1}, fill={"from": "base.yaml"})
    with pytest.raises(AisPerfModelError, match="defaults may not contain from"):
        resolve({"tp": 1}, defaults={"from": "base.yaml"})
    with pytest.raises(AisPerfModelError, match="fill: expected a mapping, got a list"):
        resolve({"tp": 1}, fill=[("tp", 1)])


def test_fill_and_defaults_match_keys_by_exact_spelling():
    # The loader does not know that Rust also reads tp_size as tp. Both keys stay in the result.
    # test_real_alias_spelling_is_left_to_rust shows what Rust does with the pair.
    assert resolve({"tp_size": 4}, fill={"tp": 4}, defaults={"pp": 1, "pp_size": 2}) == {
        "tp_size": 4,
        "tp": 4,
        "pp": 1,
        "pp_size": 2,
    }


# --- Sealed source: state snapshot --------------------------------------------------------------

_SNAPSHOT_SHA = "0123abcd" * 8


def _saved_snapshot_config() -> dict:
    return {
        "model": "test/model",
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "unknown",
        "worker_type": "decode",
        "tp": 2,
        "kv_block_size": None,
        "enable_shared_layer": False,
        "estimation_mode": "fpm_regression",
        "systems_paths": [],
        "estimator_config": {
            "correction": {"mode": "fixed", "factor_bound": None},
            "state_snapshot": {"path": "/where/it/was/saved", "sha256": _SNAPSHOT_SHA, "on_tune": "reject"},
        },
    }


@pytest.fixture
def snapshot(tmp_path):
    directory = tmp_path / "snapshot"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(_saved_snapshot_config()))
    return directory


def _from_snapshot(directory: Path, **written) -> dict:
    return {"from": {"state_snapshot": {"path": str(directory)}}, **written}


def test_snapshot_config_points_at_the_snapshot_folder(snapshot, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = _saved_snapshot_config()
    del expected["kv_block_size"], expected["systems_paths"]
    expected["estimator_config"]["state_snapshot"]["path"] = str(snapshot)
    assert resolve({"from": {"state_snapshot": {"path": "snapshot"}}}) == expected
    assert resolve(_from_snapshot(snapshot), worker_type="decode") == expected
    assert json.loads((snapshot / "config.json").read_text()) == _saved_snapshot_config()


def test_snapshot_sha256_must_equal_the_saved_one(snapshot):
    for sha256 in (_SNAPSHOT_SHA, _SNAPSHOT_SHA.upper()):
        entry = {"from": {"state_snapshot": {"path": str(snapshot), "sha256": sha256}}}
        assert resolve(entry)["estimator_config"]["state_snapshot"]["sha256"] == _SNAPSHOT_SHA
    with pytest.raises(AisPerfModelError) as error:
        resolve({"from": {"state_snapshot": {"path": str(snapshot), "sha256": "ff" * 32}}}, worker_type="decode")
    assert str(error.value) == (
        f"ais_perf_model entry for worker type 'decode': from.state_snapshot.sha256 is \"{'ff' * 32}\" "
        f'but {snapshot / "config.json"} records "{_SNAPSHOT_SHA}"'
    )


def test_snapshot_written_keys_must_equal_the_saved_values(snapshot):
    written = {
        "worker_type": "decode",
        "tp": 2,
        "backend_version": "unknown",
        "enable_shared_layer": False,
        "estimator_config": {"correction": {"mode": "fixed", "factor_bound": None, "absent": None}},
    }
    assert resolve(_from_snapshot(snapshot, **written)) == resolve(_from_snapshot(snapshot))
    origin = f"the state snapshot {snapshot / 'config.json'}"
    with pytest.raises(AisPerfModelError) as error:
        resolve(_from_snapshot(snapshot, tp=4, worker_type="decode"))
    assert str(error.value) == (
        f"ais_perf_model entry for worker type 'decode': tp is 4 but {origin} has 2; "
        "a config taken from a dataset or a state snapshot cannot be changed. Remove tp or write the same value"
    )
    with pytest.raises(AisPerfModelError, match=re.escape(f'backend_version is "1.0" but {origin} has "unknown"')):
        resolve(_from_snapshot(snapshot, backend_version="1.0"))
    with pytest.raises(AisPerfModelError, match=re.escape(f"enable_shared_layer is 0 but {origin} has false")):
        resolve(_from_snapshot(snapshot, enable_shared_layer=0))
    with pytest.raises(
        AisPerfModelError, match=re.escape(f'estimator_config.correction.mode is "auto" but {origin} has "fixed"')
    ):
        resolve(_from_snapshot(snapshot, estimator_config={"correction": {"mode": "auto"}}))
    with pytest.raises(
        AisPerfModelError, match=re.escape(f'estimator_config.correction is "auto" but {origin} has {{"factor_bound"')
    ):
        resolve(_from_snapshot(snapshot, estimator_config={"correction": "auto"}))


@pytest.mark.parametrize(
    ("written", "field"),
    [
        ({"dcp": 2}, "dcp"),
        ({"kv_block_size": 16}, "kv_block_size"),
        ({"estimator_config": {"correction": {"scale": 2}}}, "estimator_config.correction.scale"),
        ({"estimator_config": {"fpm_regression": {"fit": {"kind": "spline"}}}}, "estimator_config.fpm_regression"),
    ],
)
def test_snapshot_cannot_be_extended(snapshot, written, field):
    with pytest.raises(AisPerfModelError) as error:
        resolve(_from_snapshot(snapshot, **written))
    assert str(error.value) == (
        f"ais_perf_model entry: {field} is written but the state snapshot {snapshot / 'config.json'} "
        f"does not state it; a config taken from a dataset or a state snapshot cannot be extended. Remove {field}"
    )


def test_snapshot_systems_paths_is_a_location(snapshot, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = resolve(_from_snapshot(snapshot, systems_paths=["data", "default"]), fill={"systems_paths": ["other"]})
    assert result["systems_paths"] == [str(Path.cwd() / "data"), "default"]
    assert "systems_paths" not in resolve(_from_snapshot(snapshot), fill={"systems_paths": ["other"]})
    saved = {**_saved_snapshot_config(), "systems_paths": ["/saved/systems"]}
    (snapshot / "config.json").write_text(json.dumps(saved))
    assert resolve(_from_snapshot(snapshot))["systems_paths"] == ["/saved/systems"]
    assert resolve(_from_snapshot(snapshot, systems_paths=["/new"]))["systems_paths"] == ["/new"]


def test_snapshot_fill_is_compared_only_with_known_values(snapshot):
    plain = resolve(_from_snapshot(snapshot))
    fill = {
        "model": "test/model",  # equal
        "backend_version": "0.25.1",  # the snapshot says "unknown": skipped
        "kv_block_size": 16,  # the snapshot says null: skipped
        "dcp": 2,  # the snapshot does not have the key: skipped
        "estimator_config": {"correction": {"mode": "fixed", "scale": 2}, "op_level": {"x": 1}},
    }
    assert resolve(_from_snapshot(snapshot), fill=fill) == plain
    with pytest.raises(AisPerfModelError) as error:
        resolve(_from_snapshot(snapshot), fill={"model": "other/model"})
    assert str(error.value) == (
        f'ais_perf_model entry: the state snapshot {snapshot / "config.json"} has model "test/model" '
        'but the caller requires "other/model"'
    )
    with pytest.raises(AisPerfModelError, match='has estimator_config.correction.mode "fixed" but the caller requires'):
        resolve(_from_snapshot(snapshot), fill={"estimator_config": {"correction": {"mode": "auto"}}})


def test_snapshot_ignores_defaults(snapshot):
    defaults = {"tp": 8, "dcp": 2, "kv_block_size": 16, "estimator_config": {"correction": {"scale": 2}}}
    assert resolve(_from_snapshot(snapshot), defaults=defaults) == resolve(_from_snapshot(snapshot))


def test_snapshot_of_another_worker_type_is_an_error(snapshot):
    # The worker type can come from the caller, from select() or from the entry. The error is the
    # same and never tells the user to remove a key that select() added.
    for use in (
        lambda: resolve(_from_snapshot(snapshot), worker_type="prefill"),
        lambda: resolve(load(_from_snapshot(snapshot)).select("prefill")),
        lambda: resolve(_from_snapshot(snapshot, worker_type="prefill")),
        lambda: resolve(_from_snapshot(snapshot, worker_type="prefill"), worker_type="prefill"),
    ):
        with pytest.raises(AisPerfModelError) as error:
            use()
        assert str(error.value) == (
            f"ais_perf_model entry for worker type 'prefill': the state snapshot {snapshot / 'config.json'} "
            'is for worker type "decode", not "prefill"'
        )
    with pytest.raises(AisPerfModelError) as error:
        resolve(_from_snapshot(snapshot), fill={"worker_type": "prefill"})
    assert str(error.value) == (
        f"ais_perf_model entry: the state snapshot {snapshot / 'config.json'} "
        'is for worker type "decode", not "prefill"'
    )


def test_snapshot_without_a_worker_type_cannot_serve_a_requested_one(snapshot):
    saved = _saved_snapshot_config()
    del saved["worker_type"]
    (snapshot / "config.json").write_text(json.dumps(saved))
    assert "worker_type" not in resolve(_from_snapshot(snapshot))
    # Nothing is added to a sealed config, so the requested worker type cannot be set on it.
    for use in (
        lambda: resolve(_from_snapshot(snapshot), worker_type="decode"),
        lambda: resolve(load(_from_snapshot(snapshot)).select("decode")),
    ):
        with pytest.raises(AisPerfModelError) as error:
            use()
        assert str(error.value) == (
            f"ais_perf_model entry for worker type 'decode': the state snapshot {snapshot / 'config.json'} "
            'does not state a worker type; it cannot be used for worker type "decode"'
        )


def test_snapshot_folder_problems_name_the_file(snapshot, tmp_path):
    missing = tmp_path / "nowhere"
    with pytest.raises(AisPerfModelError) as error:
        resolve(_from_snapshot(missing))
    assert str(error.value).startswith(
        f"ais_perf_model entry: from.state_snapshot: cannot read {missing / 'config.json'}: "
    )
    config = snapshot / "config.json"
    for text in ('{"model": "m", "estimator_config": {"correction": {}}}', '{"estimator_config": null}', "[1]"):
        config.write_text(text)
        with pytest.raises(AisPerfModelError) as error:
            resolve(_from_snapshot(snapshot))
        assert str(error.value) == (
            f"ais_perf_model entry: {config} is not the config of a state snapshot "
            "(it has no estimator_config.state_snapshot); "
            "from.state_snapshot.path must be a folder written by save_state_snapshot"
        )
    config.write_text('{"model": ')
    with pytest.raises(AisPerfModelError, match=re.escape(f"from.state_snapshot: {config}: invalid JSON")):
        resolve(_from_snapshot(snapshot))
    saved = _saved_snapshot_config()
    del saved["estimator_config"]["state_snapshot"]["sha256"]
    config.write_text(json.dumps(saved))
    with pytest.raises(AisPerfModelError, match=re.escape(f'sha256 is "ab" but {config} records null')):
        resolve({"from": {"state_snapshot": {"path": str(snapshot), "sha256": "ab"}}})


def test_snapshot_on_tune_comes_from_the_entry(snapshot):
    def state(config):
        return config["estimator_config"]["state_snapshot"]

    # Without on_tune the saved value stays.
    unchanged = resolve(_from_snapshot(snapshot), worker_type="decode")
    assert state(unchanged) == {"path": str(snapshot), "sha256": _SNAPSHOT_SHA, "on_tune": "reject"}

    entry = {"from": {"state_snapshot": {"path": str(snapshot), "on_tune": "continue"}}}
    # load keeps on_tune, so a caller that lowers the entry itself can pass it on.
    assert load(entry).select("decode")["from"] == {"state_snapshot": {"path": str(snapshot), "on_tune": "continue"}}
    config = resolve(entry, worker_type="decode")
    assert state(config) == {"path": str(snapshot), "sha256": _SNAPSHOT_SHA, "on_tune": "continue"}
    # Nothing else changes.
    state(unchanged)["on_tune"] = "continue"
    assert config == unchanged

    # Rust owns the allowed values. The loader passes the text on.
    entry["from"]["state_snapshot"]["on_tune"] = "sometimes"
    assert state(resolve(entry))["on_tune"] == "sometimes"

    # A saved config without on_tune: nothing is added, unless the entry asks for it.
    saved = _saved_snapshot_config()
    del saved["estimator_config"]["state_snapshot"]["on_tune"]
    (snapshot / "config.json").write_text(json.dumps(saved))
    assert state(resolve(_from_snapshot(snapshot))) == {"path": str(snapshot), "sha256": _SNAPSHOT_SHA}
    assert state(resolve(entry))["on_tune"] == "sometimes"


def test_snapshot_on_tune_is_set_only_in_from(snapshot):
    origin = f"the state snapshot {snapshot / 'config.json'}"
    wanted = {"estimator_config": {"state_snapshot": {"on_tune": "continue"}}}
    with pytest.raises(AisPerfModelError) as error:
        resolve(_from_snapshot(snapshot, **wanted))
    assert str(error.value) == (
        f'ais_perf_model entry: estimator_config.state_snapshot.on_tune is "continue" but {origin} '
        'is loaded with on_tune "reject"; set from.state_snapshot.on_tune to change it'
    )
    with pytest.raises(AisPerfModelError) as error:
        resolve(_from_snapshot(snapshot), worker_type="decode", fill=wanted)
    assert str(error.value) == (
        "ais_perf_model entry for worker type 'decode': the caller requires "
        f'estimator_config.state_snapshot.on_tune "continue" but {origin} '
        'is loaded with on_tune "reject"; set from.state_snapshot.on_tune to change it'
    )
    # defaults never change a sealed config.
    assert resolve(_from_snapshot(snapshot), defaults=wanted) == resolve(_from_snapshot(snapshot))

    # Written and filled values are compared with the value the entry asks for.
    entry = {"from": {"state_snapshot": {"path": str(snapshot), "on_tune": "continue"}}}
    assert resolve({**entry, **wanted}, fill=wanted) == resolve(entry)
    with pytest.raises(AisPerfModelError) as error:
        resolve(entry, fill={"estimator_config": {"state_snapshot": {"on_tune": "reject"}}})
    assert str(error.value) == (
        f'ais_perf_model entry: the caller requires estimator_config.state_snapshot.on_tune "reject" but {origin} '
        'is loaded with on_tune "continue"; set from.state_snapshot.on_tune to change it'
    )


def test_snapshot_saved_relative_paths_start_at_the_snapshot_folder(snapshot, tmp_path, monkeypatch):
    saved = _saved_snapshot_config()
    saved["systems_paths"] = ["systems", "default", "/abs/systems"]
    saved["estimator_config"]["fpm_interpolation"] = {"fpm_parquet_path": "data/fpm.parquet"}
    saved["estimator_config"]["state_snapshot"]["path"] = "."
    (snapshot / "config.json").write_text(json.dumps(saved))
    # The current directory is another folder: the result does not depend on where the process runs.
    monkeypatch.chdir(tmp_path)
    config = resolve(_from_snapshot(snapshot))
    assert config["systems_paths"] == [str(snapshot / "systems"), "default", "/abs/systems"]
    assert config["estimator_config"]["fpm_interpolation"] == {
        "fpm_parquet_path": str(snapshot / "data" / "fpm.parquet")
    }
    assert config["estimator_config"]["state_snapshot"]["path"] == str(snapshot)
    # Written and filled paths start at the current directory and are compared as absolute paths.
    same = {"estimator_config": {"fpm_interpolation": {"fpm_parquet_path": "snapshot/data/fpm.parquet"}}}
    assert resolve(_from_snapshot(snapshot, **same), fill=same) == config


@pytest.mark.parametrize("source", ["base.yaml", {"state_snapshot": {"path": "other"}}], ids=["template", "snapshot"])
def test_snapshot_saved_config_may_not_contain_from(snapshot, source):
    (snapshot / "config.json").write_text(json.dumps({**_saved_snapshot_config(), "from": source}))
    with pytest.raises(AisPerfModelError) as error:
        resolve(_from_snapshot(snapshot), worker_type="decode")
    assert str(error.value) == (
        f"ais_perf_model entry for worker type 'decode': {snapshot / 'config.json'} contains from; "
        "a saved config may not name another source"
    )


# --- Sealed source: dataset ---------------------------------------------------------------------


@pytest.fixture
def raw_dataset(tmp_path):
    root = tmp_path / "raw-dataset"
    (root / "metadata").mkdir(parents=True)
    (root / "metadata" / "manifest.json").write_text('{"purpose": "execution-profile"}')
    return root


@pytest.fixture
def prepare_calls(monkeypatch):
    """Stand in for ``prepare_execution_profile``: same outputs, no native work."""
    from aisimulate_core import datasets

    calls = []

    def prepare(root, manifest_path, output_dir):
        calls.append((root, manifest_path, output_dir))
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "systems").mkdir()
        (output_dir / "profile.json").write_text("{}")
        config = {
            "model": "test/model",
            "system": "h200_sxm",
            "backend": "vllm",
            "backend_version": "1.0",
            "worker_type": "prefill",
            "tp": None,
            "estimation_mode": "op_level",
            "manifest": (root / manifest_path).read_text(),
            "systems_paths": [str(output_dir / "systems")],
            "estimator_config": {
                "op_level": {"execution_profile": {"path": str(output_dir / "profile.json"), "sha256": "abc"}}
            },
        }
        (output_dir / "config.json").write_text(json.dumps(config))
        return config

    monkeypatch.setattr(datasets, "prepare_execution_profile", prepare)
    return calls


def _from_dataset(root: Path, **written) -> dict:
    return {"from": {"dataset": {"root": str(root), "manifest": "metadata/manifest.json"}}, **written}


def test_dataset_needs_a_workspace(raw_dataset, prepare_calls):
    with pytest.raises(AisPerfModelError) as error:
        resolve(_from_dataset(raw_dataset), worker_type="prefill")
    assert str(error.value) == (
        "ais_perf_model entry for worker type 'prefill': from.dataset needs a workspace folder to prepare "
        f"the dataset {raw_dataset / 'metadata' / 'manifest.json'} in; "
        "the caller did not give one (resolve(..., workspace=DIR))"
    )
    assert prepare_calls == []


def test_dataset_is_prepared_once_and_reused(raw_dataset, prepare_calls, tmp_path):
    workspace = tmp_path / "deep" / "workspace"
    first = resolve(_from_dataset(raw_dataset), workspace=workspace)
    [prepared] = workspace.iterdir()
    assert re.fullmatch("[0-9a-f]{16}", prepared.name)
    assert first == {
        "model": "test/model",
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "1.0",
        "worker_type": "prefill",
        "estimation_mode": "op_level",
        "manifest": '{"purpose": "execution-profile"}',
        "systems_paths": [str(prepared / "systems")],
        "estimator_config": {
            "op_level": {"execution_profile": {"path": str(prepared / "profile.json"), "sha256": "abc"}}
        },
    }
    [(root, manifest_path, output_dir)] = prepare_calls
    assert (root, manifest_path) == (raw_dataset, "metadata/manifest.json")
    # Prepared in a sibling folder, then moved: config.json still names the sibling.
    assert output_dir.parent == workspace
    assert output_dir.name.startswith(f".tmp-{prepared.name}-")
    assert not output_dir.exists()
    assert json.loads((prepared / "config.json").read_text())["systems_paths"] == [str(output_dir / "systems")]

    assert resolve(_from_dataset(raw_dataset), workspace=str(workspace)) == first
    assert resolve(_from_dataset(raw_dataset, model="test/model"), fill={"tp": 2}, workspace=workspace) == first
    assert len(prepare_calls) == 1
    assert [path.name for path in workspace.iterdir()] == [prepared.name]


def test_dataset_folder_depends_on_root_manifest_path_and_manifest_bytes(raw_dataset, prepare_calls, tmp_path):
    workspace = tmp_path / "workspace"
    resolve(_from_dataset(raw_dataset), workspace=workspace)
    copied = tmp_path / "copied-dataset"
    shutil.copytree(raw_dataset, copied)
    resolve(_from_dataset(copied), workspace=workspace)
    shutil.copyfile(raw_dataset / "metadata" / "manifest.json", raw_dataset / "metadata" / "second.json")
    resolve({"from": {"dataset": {"root": str(raw_dataset), "manifest": "metadata/second.json"}}}, workspace=workspace)
    assert len(prepare_calls) == 3
    (raw_dataset / "metadata" / "manifest.json").write_text('{"purpose": "execution-profile", "revision": 2}')
    changed = resolve(_from_dataset(raw_dataset), workspace=workspace)
    assert changed["manifest"] == '{"purpose": "execution-profile", "revision": 2}'
    assert len(prepare_calls) == 4
    assert len(list(workspace.iterdir())) == 4


def test_dataset_written_keys_and_fill_are_compared(raw_dataset, prepare_calls, tmp_path):
    workspace = tmp_path / "workspace"
    origin = f"the dataset {raw_dataset / 'metadata' / 'manifest.json'}"
    assert resolve(_from_dataset(raw_dataset), worker_type="prefill", workspace=workspace)["worker_type"] == "prefill"
    with pytest.raises(AisPerfModelError, match=re.escape(f'backend is "sglang" but {origin} has "vllm"')):
        resolve(_from_dataset(raw_dataset, backend="sglang"), workspace=workspace)
    with pytest.raises(AisPerfModelError, match=re.escape(f"tp is written but {origin} does not state it")):
        resolve(_from_dataset(raw_dataset, tp=2), workspace=workspace)
    with pytest.raises(AisPerfModelError, match=re.escape(f'{origin} has system "h200_sxm" but the caller requires')):
        resolve(_from_dataset(raw_dataset), fill={"system": "b200"}, workspace=workspace)
    for entry in (_from_dataset(raw_dataset), load(_from_dataset(raw_dataset)).select("decode")):
        with pytest.raises(AisPerfModelError) as error:
            resolve(entry, worker_type="decode", workspace=workspace)
        assert str(error.value) == (
            f'ais_perf_model entry for worker type \'decode\': {origin} is for worker type "prefill", not "decode"'
        )
    # A dataset brings its own performance data: systems_paths is compared like any other key.
    with pytest.raises(AisPerfModelError, match=re.escape(f'systems_paths is ["/mine"] but {origin} has')):
        resolve(_from_dataset(raw_dataset, systems_paths=["/mine"]), workspace=workspace)
    assert resolve(_from_dataset(raw_dataset), defaults={"tp": 8}, workspace=workspace).get("tp") is None
    assert len(prepare_calls) == 1


def test_dataset_prepared_by_another_process_first_is_reused(raw_dataset, prepare_calls, tmp_path, monkeypatch):
    from aisimulate_core import datasets

    # The folder name does not depend on the workspace. Learn it in a first workspace.
    resolve(_from_dataset(raw_dataset), workspace=tmp_path / "first")
    [name] = os.listdir(tmp_path / "first")
    workspace = tmp_path / "second"
    prepare = datasets.prepare_execution_profile

    def prepare_and_lose_the_race(root, manifest_path, output_dir):
        prepare(root, manifest_path, workspace / name)
        winner = json.loads((workspace / name / "config.json").read_text())
        (workspace / name / "config.json").write_text(json.dumps({**winner, "model": "the/winner"}))
        return prepare(root, manifest_path, output_dir)

    monkeypatch.setattr(datasets, "prepare_execution_profile", prepare_and_lose_the_race)
    result = resolve(_from_dataset(raw_dataset), workspace=workspace)
    assert result["model"] == "the/winner"
    assert result["systems_paths"] == [str(workspace / name / "systems")]
    assert os.listdir(workspace) == [name]


def test_dataset_folder_that_cannot_be_replaced_is_an_error(raw_dataset, prepare_calls, tmp_path):
    resolve(_from_dataset(raw_dataset), workspace=tmp_path / "first")
    [name] = os.listdir(tmp_path / "first")
    workspace = tmp_path / "second"
    (workspace / name / "leftover").mkdir(parents=True)
    with pytest.raises(AisPerfModelError) as error:
        resolve(_from_dataset(raw_dataset), workspace=workspace)
    message = str(error.value)
    assert message.startswith(
        f"ais_perf_model entry: from.dataset: cannot move the prepared dataset to {workspace / name}: "
    )
    assert message.endswith(f"remove {workspace / name} and try again")
    assert os.listdir(workspace) == [name]


def test_dataset_prepare_failure_is_wrapped_and_leaves_nothing_behind(raw_dataset, tmp_path, monkeypatch):
    from aisimulate_core import datasets

    def fail(root, manifest_path, output_dir):
        output_dir.mkdir(parents=True)
        (output_dir / "partial").write_text("x")
        raise FileNotFoundError("artifact data/table.parquet is missing")

    monkeypatch.setattr(datasets, "prepare_execution_profile", fail)
    workspace = tmp_path / "workspace"
    with pytest.raises(AisPerfModelError) as error:
        resolve(_from_dataset(raw_dataset), worker_type="prefill", workspace=workspace)
    assert str(error.value) == (
        "ais_perf_model entry for worker type 'prefill': from.dataset: cannot prepare the dataset "
        f"{raw_dataset / 'metadata' / 'manifest.json'} in {workspace}: artifact data/table.parquet is missing"
    )
    assert isinstance(error.value.__cause__, FileNotFoundError)
    assert os.listdir(workspace) == []


def test_dataset_problems_name_the_file(raw_dataset, prepare_calls, tmp_path):
    workspace = tmp_path / "workspace"
    entry = {"from": {"dataset": {"root": str(raw_dataset), "manifest": "metadata/missing.json"}}}
    with pytest.raises(AisPerfModelError) as error:
        resolve(entry, workspace=workspace)
    assert str(error.value).startswith(
        f"ais_perf_model entry: from.dataset: cannot read the manifest {raw_dataset / 'metadata' / 'missing.json'}: "
    )
    assert prepare_calls == []
    resolve(_from_dataset(raw_dataset), workspace=workspace)
    [prepared] = workspace.iterdir()
    (prepared / "config.json").write_text('{"model": "m", "estimator_config": {"op_level": {}}}')
    with pytest.raises(AisPerfModelError) as error:
        resolve(_from_dataset(raw_dataset), workspace=workspace)
    assert str(error.value) == (
        f"ais_perf_model entry: {prepared / 'config.json'} is not the config of a prepared dataset "
        f"(it has no estimator_config.op_level.execution_profile); remove {prepared} and try again"
    )


def test_dataset_saved_config_is_read_like_a_config(raw_dataset, prepare_calls, tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    resolve(_from_dataset(raw_dataset), workspace=workspace)
    [prepared] = workspace.iterdir()
    saved = json.loads((prepared / "config.json").read_text())
    saved["estimator_config"]["fpm_interpolation"] = {"fpm_parquet_path": "data/fpm.parquet"}
    (prepared / "config.json").write_text(json.dumps(saved))
    # Relative paths in the saved config start at its folder, not at the current directory.
    monkeypatch.chdir(tmp_path)
    config = resolve(_from_dataset(raw_dataset), workspace=workspace)
    assert config["estimator_config"]["fpm_interpolation"] == {
        "fpm_parquet_path": str(prepared / "data" / "fpm.parquet")
    }

    (prepared / "config.json").write_text(json.dumps({**saved, "from": "base.yaml"}))
    with pytest.raises(AisPerfModelError) as error:
        resolve(_from_dataset(raw_dataset), workspace=workspace)
    assert str(error.value) == (
        f"ais_perf_model entry: {prepared / 'config.json'} contains from; a saved config may not name another source"
    )
    assert len(prepare_calls) == 1


@pytest.mark.parametrize("workspace", ["", "  "])
def test_dataset_blank_workspace_is_a_missing_workspace(raw_dataset, prepare_calls, tmp_path, monkeypatch, workspace):
    # A caller that passes an unset option on as "" must not get the dataset prepared in the current directory.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    with pytest.raises(AisPerfModelError) as error:
        resolve(_from_dataset(raw_dataset), workspace=workspace)
    assert str(error.value) == (
        "ais_perf_model entry: from.dataset needs a workspace folder to prepare "
        f"the dataset {raw_dataset / 'metadata' / 'manifest.json'} in; "
        "the caller did not give one (resolve(..., workspace=DIR))"
    )
    assert prepare_calls == []
    assert os.listdir(cwd) == []


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_dataset_prepare_interrupt_is_passed_on_and_leaves_nothing_behind(
    raw_dataset, tmp_path, monkeypatch, interrupt
):
    from aisimulate_core import datasets

    def interrupted(root, manifest_path, output_dir):
        output_dir.mkdir(parents=True)
        (output_dir / "partial").write_text("x")
        raise interrupt()

    monkeypatch.setattr(datasets, "prepare_execution_profile", interrupted)
    workspace = tmp_path / "workspace"
    with pytest.raises(interrupt):
        resolve(_from_dataset(raw_dataset), workspace=workspace)
    assert os.listdir(workspace) == []


_RACE_WORKER = """
import json
import os
import sys
import time
import types
from pathlib import Path

import aisimulate_core
from aisimulate_core import ais_perf_model as apm

root, workspace, barrier, racers = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])


def prepare(root, manifest_path, output_dir):
    # Same outputs as prepare_execution_profile, no native work. Wait inside until every process
    # is preparing the same dataset, so that all of them try to move their folder into place.
    output_dir.mkdir(parents=True, exist_ok=False)
    (barrier / f"inside-{os.getpid()}").touch()
    deadline = time.monotonic() + 30
    while len(list(barrier.glob("inside-*"))) < racers:
        if time.monotonic() > deadline:
            raise TimeoutError("the other processes did not start to prepare")
        time.sleep(0.005)
    (output_dir / "systems").mkdir()
    (output_dir / "profile.json").write_text("{}")
    config = {
        "model": "test/model",
        "worker_type": "prefill",
        "prepared_by": os.getpid(),
        "systems_paths": [str(output_dir / "systems")],
        "estimator_config": {"op_level": {"execution_profile": {"path": str(output_dir / "profile.json")}}},
    }
    (output_dir / "config.json").write_text(json.dumps(config))


stub = types.ModuleType("aisimulate_core.datasets")
stub.prepare_execution_profile = prepare
sys.modules["aisimulate_core.datasets"] = aisimulate_core.datasets = stub
entry = {"from": {"dataset": {"root": root, "manifest": "metadata/manifest.json"}}}
print(json.dumps(apm.resolve(entry, worker_type="prefill", workspace=workspace)))
"""


def test_dataset_prepared_by_several_processes_at_once_ends_in_one_folder(raw_dataset, tmp_path):
    workspace, barrier, racers = tmp_path / "workspace", tmp_path / "barrier", 4
    barrier.mkdir()
    command = [sys.executable, "-W", "error", "-c", _RACE_WORKER, str(raw_dataset), str(workspace), str(barrier)]
    processes = [
        subprocess.Popen([*command, str(racers)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(racers)
    ]
    try:
        outputs = [process.communicate(timeout=60) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert [process.returncode for process in processes] == [0] * racers, [stderr for _, stderr in outputs]
    # Every process prepared the dataset. One folder won and nothing else is left in the workspace.
    assert len(list(barrier.glob("inside-*"))) == racers
    [prepared] = workspace.iterdir()
    assert re.fullmatch("[0-9a-f]{16}", prepared.name)
    # Every process returns the config of the folder that won.
    configs = [json.loads(stdout) for stdout, _ in outputs]
    assert configs == [configs[0]] * racers
    assert configs[0]["prepared_by"] in [process.pid for process in processes]
    assert configs[0]["systems_paths"] == [str(prepared / "systems")]
    assert configs[0]["estimator_config"]["op_level"]["execution_profile"]["path"] == str(prepared / "profile.json")


# --- Error conversion ---------------------------------------------------------------------------


def test_file_errors_become_loader_errors_that_name_the_file(tmp_path):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(AisPerfModelError) as error:
        load(missing)
    assert str(error.value).startswith(f"cannot read {missing}: ")
    assert isinstance(error.value, ValueError)
    assert isinstance(error.value.__cause__, FileNotFoundError)

    with pytest.raises(AisPerfModelError, match=re.escape(f"cannot read {tmp_path}: ")) as error:
        load(tmp_path)
    assert isinstance(error.value.__cause__, OSError)

    binary = tmp_path / "binary.yaml"
    binary.write_bytes(b"tp: \xff\xfe\n")
    with pytest.raises(AisPerfModelError, match=re.escape(f"cannot read {binary}: ")):
        load(binary)


def test_parse_errors_become_loader_errors_that_name_the_file(tmp_path):
    import yaml

    broken_yaml = _write(tmp_path / "broken.yaml", "systems_paths: [a, b\ntp: 1\n")
    with pytest.raises(AisPerfModelError) as error:
        load(broken_yaml)
    assert str(error.value).startswith(f"{broken_yaml}: invalid YAML: ")
    assert f'in "{broken_yaml}", line ' in str(error.value)
    assert isinstance(error.value.__cause__, yaml.YAMLError)

    two_documents = _write(tmp_path / "two.yaml", "tp: 1\n---\ntp: 2\n")
    with pytest.raises(AisPerfModelError, match=re.escape(f"{two_documents}: invalid YAML: ")):
        load(two_documents)

    unsafe = _write(tmp_path / "unsafe.yaml", "tp: !!python/object/apply:os.getcwd []\n")
    with pytest.raises(AisPerfModelError, match=re.escape(f"{unsafe}: invalid YAML: ")):
        load(unsafe)

    broken_json = _write(tmp_path / "broken.json", '{"tp": }')
    with pytest.raises(AisPerfModelError) as error:
        load(broken_json)
    assert str(error.value).startswith(f"{broken_json}: invalid JSON: ")
    assert isinstance(error.value.__cause__, json.JSONDecodeError)

    with pytest.raises(AisPerfModelError, match=r"^ais_perf_model \(inline JSON\): invalid JSON: "):
        load('{"tp": ')
    with pytest.raises(AisPerfModelError, match=r"^ais_perf_model \(inline JSON\): invalid JSON: "):
        load("[" * 100_000)


@pytest.mark.parametrize("bad", ["a\0b", "a\ud800b"], ids=["nul", "lone-surrogate"])
def test_paths_the_system_cannot_take_become_loader_errors(bad, raw_dataset, prepare_calls, tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    workspace = tmp_path / "workspace"
    # Inline JSON can carry both: "a\u0000b" and "a\ud800b" are valid JSON strings.
    in_json = json.dumps({"from": {"dataset": {"root": str(raw_dataset), "manifest": f"{bad}.json"}}})
    bad_root = {"from": {"dataset": {"root": bad, "manifest": "metadata/manifest.json"}}}
    uses = [
        ("cannot read ", lambda: load(f"{bad}.yaml")),
        ("cannot read ", lambda: load(Path(f"{bad}.json"))),
        ("cannot read ", lambda: load("model.yaml", base_dir=bad)),
        ("cannot use the template named by from: cannot read ", lambda: expand_template({"from": f"{bad}.yaml"})),
        ("cannot use the template named by from: cannot read ", lambda: resolve({"from": f"{bad}.yaml"})),
        ("from.state_snapshot: cannot read ", lambda: resolve({"from": {"state_snapshot": {"path": bad}}})),
        ("from.dataset: cannot read the manifest ", lambda: resolve(bad_root, workspace=workspace)),
        (
            "from.dataset: cannot read the manifest ",
            lambda: resolve(load(in_json).select("prefill"), workspace=workspace),
        ),
        (
            "from.dataset: cannot prepare the dataset ",
            lambda: resolve(_from_dataset(raw_dataset), workspace=tmp_path / bad),
        ),
    ]
    for fragment, use in uses:
        with pytest.raises(AisPerfModelError) as error:
            use()
        assert fragment in str(error.value)
        # The message can be printed and logged, whatever the path holds.
        str(error.value).encode("utf-8")
    assert prepare_calls == []
    assert os.listdir(cwd) == []
    assert not workspace.exists()


def test_error_text_that_cannot_be_encoded_is_escaped():
    error = AisPerfModelError("cannot read /data/\ud800.yaml: no such file")
    assert str(error) == "cannot read /data/\\ud800.yaml: no such file"
    assert error.args == ("cannot read /data/\\ud800.yaml: no such file",)
    assert AisPerfModelError("plain text", 5).args == ("plain text", 5)


def test_missing_pyyaml_is_a_loader_error(tmp_path, monkeypatch):
    as_yaml = _write(tmp_path / "model.yaml", "tp: 1\n")
    no_suffix = _write(tmp_path / "model", "tp: 1\n")
    as_json = _write(tmp_path / "model.json", '{"tp": 1}')
    # None in sys.modules makes "import yaml" fail, as it does when the package is not installed.
    monkeypatch.setitem(sys.modules, "yaml", None)
    apm._yaml.cache_clear()
    try:
        for file, use in (
            (as_yaml, lambda: load(as_yaml)),
            (no_suffix, lambda: load(no_suffix)),
            (as_yaml, lambda: expand_template({"from": str(as_yaml)})),
            (as_yaml, lambda: resolve({"from": str(as_yaml)})),
        ):
            with pytest.raises(AisPerfModelError) as error:
                use()
            assert f"{file}: reading a YAML file needs the pyyaml package (" in str(error.value)
            assert str(error.value).endswith("); install pyyaml or use a .json file")
        with pytest.raises(AisPerfModelError) as error:
            load(as_yaml)
        assert isinstance(error.value.__cause__, ImportError)
        # JSON needs no extra package.
        assert load(as_json).to_data() == {"tp": 1}
        assert load('{"tp": 1}').to_data() == {"tp": 1}
        assert resolve({"from": str(as_json), "dp": 2}) == {"tp": 1, "dp": 2}
    finally:
        # A failed import is not remembered. Make sure of it for the tests that follow.
        apm._yaml.cache_clear()


# --- Deprecation warning ------------------------------------------------------------------------


def test_warn_deprecated_message_category_and_location():
    lines = {}

    def old_api(stacklevel=None):
        lines["inner"] = inspect.currentframe().f_lineno + 2
        if stacklevel is None:
            apm.warn_deprecated("engine.systems_paths", "engine.ais_perf_model")
        else:
            apm.warn_deprecated("engine.systems_paths", "engine.ais_perf_model", stacklevel=stacklevel)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outer = inspect.currentframe().f_lineno + 1
        old_api()
        old_api(stacklevel=1)
    assert [type(item.message) for item in caught] == [apm.AisPerfModelDeprecationWarning] * 2
    assert str(caught[0].message) == (
        "ais-perf-model: engine.systems_paths is deprecated and will be removed in the next release; "
        "use engine.ais_perf_model"
    )
    assert str(caught[0].message).startswith(apm.DEPRECATION_PREFIX)
    # The default points at the caller of the function that warns.
    assert (caught[0].filename, caught[0].lineno) == (__file__, outer)
    assert (caught[1].filename, caught[1].lineno) == (__file__, lines["inner"] + 2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", apm.AisPerfModelDeprecationWarning)
        with pytest.raises(FutureWarning, match="^ais-perf-model: old is deprecated"):
            apm.warn_deprecated("old", "new")


@pytest.mark.parametrize(
    ("old", "subject"),
    [
        (["engine.systems_paths"], "engine.systems_paths is deprecated"),
        (
            ("engine.estimation_mode", "engine.workers.decode.timing.systems_paths"),
            "engine.estimation_mode, engine.workers.decode.timing.systems_paths are deprecated",
        ),
    ],
    ids=["a list of one name", "several names"],
)
def test_warn_deprecated_names_every_old_field_in_one_sentence(old, subject):
    with pytest.warns(apm.AisPerfModelDeprecationWarning) as caught:
        apm.warn_deprecated(old, "engine.ais_perf_model")

    assert [str(item.message) for item in caught] == [
        f"ais-perf-model: {subject} and will be removed in the next release; use engine.ais_perf_model"
    ]


def test_warn_deprecated_needs_an_old_field_to_name():
    with pytest.raises(ValueError, match="^warn_deprecated needs at least one old field name$"):
        apm.warn_deprecated([], "engine.ais_perf_model")


# --- Import weight ------------------------------------------------------------------------------

_IMPORT_PROBE = """
import sys

import aisimulate_core.ais_perf_model as apm

heavy = ["numpy", "pandas", "pyarrow", "yaml", "aisimulate.sweeper", "aisimulate_core._native",
         "aisimulate_core.sdk", "aisimulate_core.datasets"]

def loaded():
    return sorted(name for name in sys.modules if any(name == item or name.startswith(item + ".") for item in heavy))

assert not loaded(), loaded()
document = apm.load('[{"worker_type": "decode", "systems_paths": ["data"], "from": {"state_snapshot": {"path": "s"}}}]')
entry = document.select("decode")
del entry["from"]
assert apm.resolve(entry, fill={"model": "m"}, defaults={"tp": 1})["tp"] == 1
assert not loaded(), loaded()

# YAML is imported on first use. Nothing else comes with it.
assert apm.resolve(apm.load(sys.argv[1]).select("prefill")) == {"worker_type": "prefill", "tp": 1}
assert "yaml" in sys.modules
heavy.remove("yaml")
assert not loaded(), loaded()
"""


def test_import_and_plain_use_load_no_native_numpy_or_pyarrow(tmp_path):
    template = _write(tmp_path / "base.yaml", "tp: 1\n")
    entry = _write(tmp_path / "model.yaml", f"from: {template.name}\n")
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", _IMPORT_PROBE, str(entry)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


# --- Real data ----------------------------------------------------------------------------------


def test_real_dataset_resolves_to_a_working_model(dataset, tmp_path, monkeypatch):  # noqa: F811
    from aisimulate_core import datasets
    from aisimulate_core.sdk import RustForwardPassPerfModel

    root, manifest_path, _ = dataset
    workspace = tmp_path / "workspace"
    entry = {"from": {"dataset": {"root": str(root), "manifest": manifest_path}}}
    fill = {"model": "meta-llama/Meta-Llama-3.1-8B", "system": "h200_sxm", "backend": "vllm", "tp": 1}
    config = resolve(entry, worker_type="prefill", fill=fill, defaults={"dp": 4}, workspace=workspace)

    [prepared] = workspace.iterdir()
    assert config["systems_paths"] == [str(prepared / "systems")]
    profile = config["estimator_config"]["op_level"]["execution_profile"]
    assert profile["path"] == str(prepared / "profile.json")
    # The real config.json names the folder the dataset was prepared in, which is gone.
    saved = json.loads((prepared / "config.json").read_text())
    assert saved["systems_paths"] != config["systems_paths"]
    assert not Path(saved["systems_paths"][0]).exists()
    assert ".tmp-" in json.dumps(saved)
    assert ".tmp-" not in json.dumps(config)

    def prepared_twice(*args, **kwargs):
        pytest.fail("a prepared dataset must be reused")

    monkeypatch.setattr(datasets, "prepare_execution_profile", prepared_twice)
    assert resolve(entry, workspace=workspace) == config
    with pytest.raises(AisPerfModelError, match='has backend "vllm" but the caller requires "sglang"'):
        resolve(entry, fill={"backend": "sglang"}, workspace=workspace)

    # The prepared folder is enough: the dataset and the producer's data are not read again.
    shutil.rmtree(root)
    shutil.rmtree(tmp_path / "producer-systems")
    model = RustForwardPassPerfModel.best_available(config)
    # Same oracle as test_dataset_contracts: three repetitions of 2 ms or 5 ms.
    assert model.predict_prefill_latency(1, 4, 0) == 6.0
    assert model.predict_prefill_latency(1, 8, 4) == 15.0


def _rank(prefill: int, decode: int, kv: int, wall_ms: float) -> dict:
    return {
        "version": 1,
        "wall_time": wall_ms / 1000.0,
        "scheduled_requests": {
            "num_prefill_requests": int(prefill > 0),
            "sum_prefill_tokens": prefill,
            "sum_prefill_kv_tokens": prefill // 4,
            "num_decode_requests": decode,
            "sum_decode_kv_tokens": kv if decode else 0,
        },
    }


def _stream(count: int) -> list[list[dict]]:
    iterations = []
    for index in range(count):
        prefill, decode = 64 + 131 * index, 1 + (7 * index) % 600
        kv = decode * (64 + 37 * index)
        # Hand-specified positive surface; Rust tests own fitting accuracy.
        iterations.append([_rank(prefill, decode, kv, 2.0 + 4e-4 * prefill + 2e-5 * kv + 1e-2 * decode)])
    return iterations


def _save_tuned_snapshot(folder: Path):
    """Tune a regression model, save its state in ``folder`` and return the model."""
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model="test/model",
            system="test",
            backend="vllm",
            worker_type="aggregated",
            estimation_mode="fpm_regression",
            estimator_config={"fpm_regression": {"fit": {"kind": "spline"}}},
        )
    )
    model.tune_with_fpms(_stream(120))
    model.save_state_snapshot(str(folder))
    return model


def test_real_state_snapshot_restores_after_a_move(tmp_path):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    original = _save_tuned_snapshot(tmp_path / "saved")
    moved = tmp_path / "moved"
    shutil.move(tmp_path / "saved", moved)
    saved = json.loads((moved / "config.json").read_text())
    sha256 = saved["estimator_config"]["state_snapshot"]["sha256"]

    entry = {"from": {"state_snapshot": {"path": str(moved), "sha256": sha256.upper()}}, "backend": "vllm"}
    fill = {"model": "test/model", "system": "test", "kv_block_size": 16}
    config = resolve(entry, worker_type="aggregated", fill=fill, defaults={"tp": 4})
    assert config["estimator_config"]["state_snapshot"]["path"] == str(moved)
    assert config["estimator_config"]["fpm_regression"] == saved["estimator_config"]["fpm_regression"]
    assert None not in config.values()
    assert "from" not in config
    assert str(tmp_path / "saved") not in json.dumps(config)
    # The snapshot saved null here, so the caller's value is neither compared nor added.
    assert saved["kv_block_size"] is None
    assert "kv_block_size" not in config

    restored = RustForwardPassPerfModel.best_available(config)
    query = [_rank(512, 8, 4096, 0.0)]
    assert restored.estimate_forward_pass_time_ms(query) == original.estimate_forward_pass_time_ms(query)

    with pytest.raises(AisPerfModelError, match='has system "test" but the caller requires "h200_sxm"'):
        resolve(entry, fill={"system": "h200_sxm"})
    with pytest.raises(AisPerfModelError, match="from.state_snapshot.sha256 is"):
        resolve({"from": {"state_snapshot": {"path": str(moved), "sha256": "0" * 64}}})


def test_real_state_snapshot_is_tuned_only_with_on_tune_continue(tmp_path):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    saved = tmp_path / "saved"
    _save_tuned_snapshot(saved)
    source = {"path": str(saved)}
    # Without on_tune the saved value (reject) stays, and the restored model takes no new samples.
    frozen = RustForwardPassPerfModel.best_available(resolve({"from": {"state_snapshot": source}}))
    with pytest.raises(ValueError, match="on_tune='reject'"):
        frozen.tune_with_fpms(_stream(8))

    config = resolve({"from": {"state_snapshot": {**source, "on_tune": "continue"}}}, worker_type="aggregated")
    assert config["estimator_config"]["state_snapshot"]["on_tune"] == "continue"
    RustForwardPassPerfModel.best_available(config).tune_with_fpms(_stream(8))

    # Rust owns the allowed values: the loader passes the text on and Rust rejects it.
    config = resolve({"from": {"state_snapshot": {**source, "on_tune": "sometimes"}}})
    with pytest.raises(ValueError, match="unknown variant `sometimes`") as error:
        RustForwardPassPerfModel.best_available(config)
    assert not isinstance(error.value, AisPerfModelError)


_REAL_IDENTITY = {"model": "test/model", "system": "test", "backend": "vllm", "estimation_mode": "fpm_regression"}


def test_real_nested_null_is_not_a_missing_key():
    from aisimulate_core.sdk import RustForwardPassPerfModel

    def lower_bound(config):
        normalized = RustForwardPassPerfModel.normalize_config(config)
        return normalized["estimator_config"]["correction"]["factor_bounds"]["min"]

    defaults = {"estimator_config": {"correction": {"factor_bounds": {"min": 0.25}}}}
    unbounded = {**_REAL_IDENTITY, "estimator_config": {"correction": {"factor_bounds": {"min": None}}}}
    config = resolve(unbounded, worker_type="aggregated", defaults=defaults)
    # Rust reads this null as "no lower bound". Dropping it would give the caller's default or Rust's.
    assert config["estimator_config"]["correction"]["factor_bounds"] == {"min": None}
    assert lower_bound(config) is None
    assert lower_bound(resolve(_REAL_IDENTITY, worker_type="aggregated", defaults=defaults)) == 0.25
    assert lower_bound(resolve(_REAL_IDENTITY, worker_type="aggregated")) is not None
    RustForwardPassPerfModel.best_available(config)


def test_real_alias_spelling_is_left_to_rust():
    from aisimulate_core.sdk import RustForwardPassPerfModel

    entry = {**_REAL_IDENTITY, "tp_size": 4}
    RustForwardPassPerfModel.best_available(resolve(entry, worker_type="aggregated"))
    # A caller that fills the canonical name gets both spellings. Rust rejects the pair, also when
    # the values are equal, so the mistake is never silent.
    both = resolve(entry, worker_type="aggregated", fill={"tp": 4})
    assert (both["tp"], both["tp_size"]) == (4, 4)
    with pytest.raises(ValueError, match="duplicate field `tp`"):
        RustForwardPassPerfModel.best_available(both)
