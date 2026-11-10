# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controlled-worker checks for the one-off order isolation experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import zipfile
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pytest
from tools.forward_perf_gate import worker

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location(
    "pr585_diagnostic", REPO / "scripts/performance/diagnose_pr585_cold_queries.py"
)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def response(case, revision, iterations, cold=100.0):
    return {
        "protocol_version": 2,
        "revision": revision,
        "case_id": case["case_id"],
        "case_hash": worker.canonical_case_hash(case),
        "status": "OK",
        "group_case_count": 9,
        "steady_state_setup_queries": 2,
        "cold_definition": "steady_state_unseen_query",
        "cold_us": cold,
        "warm_samples_us": [25.0] * iterations,
        "warm": {"call_median_us": 25.0},
        "predicted_value_ms": 2.0,
    }


def lanes():
    return {
        name: {
            "python": Path("/venvs") / name / "bin/python",
            "worker": Path("/sources/head/worker.py" if name == "head" else "/sources/base/worker.py"),
            "revision": diagnostic.HEAD_SHA if name == "head" else diagnostic.BASE_SHA,
        }
        for name in ("base", "head", "control")
    }


def test_balanced_study_preserves_physical_identity_and_exposes_order_effect(monkeypatch, tmp_path):
    calls = []

    def controlled_worker(**kwargs):
        index = len(calls)
        calls.append(kwargs)
        second = index >= 3 and (index - 3) % 2 == 1
        cold = 100.0 + 20.0 * (kwargs["revision"] == diagnostic.HEAD_SHA) + 15.0 * second
        return [response(case, kwargs["revision"], kwargs["iterations"], cold) for case in kwargs["cases"]], None

    monkeypatch.setattr(diagnostic, "run_worker_batch", controlled_worker)
    analysis = diagnostic.collect(lanes(), tmp_path, {"controlled_worker": True}, cpu=0)
    raw = json.loads((tmp_path / "raw.json").read_text())
    assert raw["complete"] and not raw["errors"] and not raw["invalid_batches"]
    assert len(calls) == 51  # Three prewarm batches, then two workers for each of 24 pairs.
    observed = Counter((r["arm"], r["traversal"], tuple(r["order"])) for r in raw["observations"])
    assert len(observed) == 8 and set(observed.values()) == {3}
    assert [r["sequence"] for r in raw["observations"]] == list(range(1, 25))
    for arm in diagnostic.ARMS:
        assert sum(r["arm_order"][0] == arm for r in raw["observations"] if r["arm"] == arm) == 6
    forward = [case["case_id"] for case in diagnostic.selected_cases()]
    for call in calls:
        assert [case["case_id"] for case in call["cases"]] in (forward, forward[::-1])
        assert Counter(case["database_mode"] for case in call["cases"]) == {"SILICON": 9, "EMPIRICAL": 9}
    for call in calls[3:]:
        assert call["warmup"] == 10 and call["iterations"] == 100
        if call["python"] == lanes()["control"]["python"]:
            assert call["revision"] == diagnostic.BASE_SHA and call["worker"] == lanes()["base"]["worker"]
    for observation in raw["observations"]:
        if observation["arm"] == "base_vs_base":
            assert all(
                result["revision"] == diagnostic.BASE_SHA
                for label in ("left", "right")
                for result in observation["lanes"][label]
            )
    for point in analysis["points"]:
        assert "head_median_us" not in point and "base_median_us" not in point
        if point["arm"] == "base_vs_base":
            assert point["physical_revisions"] == {"left": diagnostic.BASE_SHA, "right": diagnostic.BASE_SHA}
        if point["metric"] != "cold":
            continue
        if point["arm"] == "base_vs_head":
            expected = 1.35 if point["lane_order"][0] == "left" else 120 / 115
        else:
            expected = 1.15 if point["lane_order"][0] == "left" else 100 / 115
        assert point["median_ratio"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "mutation",
    [
        {"revision": "wrong"},
        {"case_hash": "wrong"},
        {"protocol_version": 1},
        {"group_case_count": 8},
        {"steady_state_setup_queries": 1},
        {"cold_us": float("nan")},
        {"warm_samples_us": [25.0] * 99},
        {"warm_samples_us": None},
        {"warm": {"call_median_us": 26.0}},
        {"predicted_value_ms": False},
    ],
)
def test_invalid_worker_evidence_fails_closed(mutation):
    case = diagnostic.selected_cases()[0]
    result = response(case, diagnostic.BASE_SHA, 100)
    result.update(mutation)
    with pytest.raises(ValueError):
        diagnostic.validate_result(case, result, diagnostic.BASE_SHA, 100)


@pytest.mark.parametrize("failure", ["worker", "identity", "prediction"])
def test_failed_or_inconsistent_measurement_is_retained_and_never_complete(monkeypatch, tmp_path, failure):
    calls = 0

    def controlled_worker(**kwargs):
        nonlocal calls
        calls += 1
        if failure == "worker" and calls == 5:
            return [], "TIMEOUT: controlled failure"
        results = [response(case, kwargs["revision"], kwargs["iterations"]) for case in kwargs["cases"]]
        if calls == 5:
            if failure == "identity":
                results[0]["revision"] = "wrong"
            elif failure == "prediction":
                results[0]["predicted_value_ms"] = 9.0
        return results, None

    monkeypatch.setattr(diagnostic, "run_worker_batch", controlled_worker)
    with pytest.raises(ValueError):
        diagnostic.collect(lanes(), tmp_path, {"controlled_worker": True}, cpu=0)
    raw = json.loads((tmp_path / "raw.json").read_text())
    assert not raw["complete"] and raw["errors"]
    assert not (tmp_path / "comparison.json").exists()
    if failure in ("worker", "identity"):
        assert len(raw["invalid_batches"]) == 1
    else:
        assert raw["observations"][0]["lanes"]["right"][0]["predicted_value_ms"] == 9.0


@pytest.mark.parametrize("mismatch", [None, "source", "control_native", "head_native", "packages", "shared_install"])
def test_provenance_requires_pinned_sources_and_matching_wheel_installations(monkeypatch, tmp_path, mismatch):
    args = Namespace()
    for side in ("base", "head"):
        source = tmp_path / side
        source.mkdir()
        for path in (REPO / diagnostic.GATE).glob("*.py"):
            target = source / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        grid = Path("python/aisimulate/tools/prediction_regression_gate/grid.py")
        (source / grid).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / grid, source / grid)
        setattr(args, side + "_source", source)
        wheel = tmp_path / (side + ".whl")
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("aisimulate/_runtime.abi3.so", side.encode())
        setattr(args, side + "_wheel", wheel)
    installations = {}
    for side in ("base", "head", "control"):
        prefix = tmp_path / (side + "-venv")
        setattr(args, side + "_python", prefix / "bin/python")
        physical = "head" if side == "head" else "base"
        installations[side] = {
            "prefix": str(prefix),
            "native_path": str(prefix / "aisimulate/_runtime.abi3.so"),
            "native_sha256": hashlib.sha256(physical.encode()).hexdigest(),
            "python": "3.12.16",
            "packages": [["aisim", "0.14.0"]],
        }
    if mismatch == "control_native":
        installations["control"]["native_sha256"] = "wrong"
    elif mismatch == "head_native":
        installations["head"]["native_sha256"] = "wrong"
    elif mismatch == "packages":
        installations["control"]["packages"] = [["aisim", "0.0.0"]]
    elif mismatch == "shared_install":
        installations["control"]["prefix"] = installations["base"]["prefix"]

    def source_git(source, *command):
        if command[0] == "status":
            return ""
        if source == args.base_source:
            return "wrong" if mismatch == "source" else diagnostic.BASE_SHA
        return diagnostic.HEAD_SHA

    monkeypatch.setattr(diagnostic, "git", source_git)
    monkeypatch.setattr(
        diagnostic,
        "describe_installation",
        lambda python: installations[next(s for s in installations if python == getattr(args, s + "_python"))],
    )
    monkeypatch.setattr(diagnostic, "_command_version", lambda _: "controlled")
    if mismatch:
        with pytest.raises(ValueError):
            diagnostic.provenance(args)
    else:
        evidence = diagnostic.provenance(args)
        assert evidence["source_revisions"] == {"base": diagnostic.BASE_SHA, "head": diagnostic.HEAD_SHA}
        assert (
            evidence["installations"]["base"]["native_sha256"] == evidence["installations"]["control"]["native_sha256"]
        )
