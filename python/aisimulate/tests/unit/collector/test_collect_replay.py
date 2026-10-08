# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure replay (`collect.py --cases-from`): exact-string selection of an earlier run's
failing cases plus a seeded control sample, and the marker that keeps such a run out of
shipped data. Selection semantics only — the executor path is unchanged."""

import gzip
import json
import random
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_COLLECTOR_DIR = str(Path(__file__).resolve().parents[3] / "collector")
if _COLLECTOR_DIR not in sys.path:
    sys.path.insert(0, _COLLECTOR_DIR)

import collect

CASES = [
    [32, 255, 128, 1, 128, False, "flashinfer"],
    [32, 511, 128, 1, 128, False, "flashinfer"],
    [1, 8191, 96, 1, 128, True, "flashinfer"],
    [2, 1023, 32, 8, 128, False, "fa3"],
    {"id": "sglang.moe:run:[7, 8]", "params": [7, 8]},  # dict-form task
]


def _spec(wanted, control=0):
    return collect.ReplaySpec("errors.json", set(wanted), control=control)


def test_select_matches_exact_case_string_not_substring():
    spec = _spec({"[32, 511, 128, 1, 128, False, 'flashinfer']"})
    out = spec.select("attention_generation", CASES, random.Random(42))
    assert out == [CASES[1]]
    # ", 128, 1, 128, " would substring-match two cases; exact identity keeps one
    assert spec.matched["attention_generation"] == {"[32, 511, 128, 1, 128, False, 'flashinfer']"}
    assert spec.unmatched() == []


def test_select_uses_params_for_dict_form_tasks():
    spec = _spec({"[7, 8]"})
    out = spec.select("moe", CASES, random.Random(42))
    assert out == [CASES[4]]


def test_control_sample_is_seeded_and_excludes_requested():
    wanted = {"[32, 255, 128, 1, 128, False, 'flashinfer']"}
    a = _spec(wanted, control=2).select("op", CASES, random.Random(42))
    b = _spec(wanted, control=2).select("op", CASES, random.Random(42))
    assert a == b
    assert a[0] == CASES[0]
    assert len(a) == 3 and CASES[0] not in a[1:]
    big = _spec(wanted, control=100)
    assert len(big.select("op", CASES, random.Random(1))) == len(CASES)  # capped at the rest


def test_report_lists_unmatched_requests_across_ops():
    spec = _spec({"[32, 255, 128, 1, 128, False, 'flashinfer']", "[9, 9, 9]"})
    spec.select("attention_generation", CASES, random.Random(42))
    spec.select("gemm", [], random.Random(42))
    rep = spec.report()
    assert rep["requested"] == 2 and rep["matched"] == 1
    assert rep["unmatched"] == ["[9, 9, 9]"]
    assert rep["matched_by_op"] == {"attention_generation": 1, "gemm": 0}


def test_load_replay_cases_from_errors_json_summary_gz_and_plain_list(tmp_path: Path):
    errors = [
        {"task_params": "[32, 255, 128, 1, 128, False, 'flashinfer']", "error_type": "RuntimeError"},
        {"task_params": None, "error_type": "UnresolvedFailures"},  # checkpoint summary entry, no case
    ]
    ej = tmp_path / "errors_sglang.attention_generation.json"
    ej.write_text(json.dumps(errors))
    assert collect.load_replay_cases(ej) == {"[32, 255, 128, 1, 128, False, 'flashinfer']"}

    summary = tmp_path / "collection_summary_sglang.json.gz"
    with gzip.open(summary, "wt") as f:
        json.dump({"summary": {"total_errors": 1}, "errors": errors}, f)
    assert collect.load_replay_cases(summary) == {"[32, 255, 128, 1, 128, False, 'flashinfer']"}

    plain = tmp_path / "cases.json"
    plain.write_text(json.dumps(["[1, 2]", "[3, 4]"]))
    assert collect.load_replay_cases(plain) == {"[1, 2]", "[3, 4]"}

    lines = tmp_path / "cases.txt"
    lines.write_text("[1, 2]\n\n[3, 4]\n")
    assert collect.load_replay_cases(lines) == {"[1, 2]", "[3, 4]"}

    (tmp_path / "empty.json").write_text("[]")
    with pytest.raises(ValueError):
        collect.load_replay_cases(tmp_path / "empty.json")


def test_finish_replay_writes_marker_and_warns_on_unmatched(tmp_path: Path, caplog):
    spec = _spec({"[32, 255, 128, 1, 128, False, 'flashinfer']", "[9, 9, 9]"})
    spec.select("attention_generation", CASES, random.Random(42))
    import logging

    log = logging.getLogger("replay-test")
    with caplog.at_level(logging.INFO, logger="replay-test"):
        collect._finish_replay(spec, tmp_path, log)
    marker = json.loads((tmp_path / collect.ReplaySpec.MARKER).read_text())
    assert marker["requested"] == 2 and marker["matched"] == 1 and marker["unmatched"] == ["[9, 9, 9]"]
    assert any("not in this run's case plan" in r.message for r in caplog.records)
    assert collect._finish_replay(None, tmp_path, log) is None
