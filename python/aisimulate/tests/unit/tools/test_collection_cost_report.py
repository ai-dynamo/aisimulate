# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unknown coverage and failed validation must never become finite forecasts."""

import csv
import hashlib

import pytest

from tools.perf_database import collection_cost_report as report

pytestmark = pytest.mark.unit


def sample(length=1, seconds=2, **extra):
    return {
        "system": "b300_sxm",
        "backend": "vllm",
        "operation": "attention_context",
        "case": {"input_len": str(length), "head_dim": "128", "dispatch": "native", "attn_dtype": "fp8"},
        "group_id": "group",
        "measurement_scope": "point",
        "collection_seconds": seconds,
        "setup_seconds": None,
        "forward_latency_ms": 999,
        "source_commit": "abc",
        "participating_gpus": 1,
        "pilot_splits_same_cache_proven": True,
        "cache_scope_declared": "both fresh workers share predecessor already-warmed cache",
        "status": "passed",
        **extra,
    }


def test_setup_once_and_forward_not_added_twice():
    points = [sample(1, 2, setup_seconds=5), sample(2, 3, setup_seconds=5)]
    assert report.measured_cost(points) == 10
    assert report.measured_cost([*points, sample(seconds=11, measurement_scope="group")]) == 11


def test_failures_and_timeouts_cannot_fit():
    rows = [sample(1, 1), sample(3, 3, status="timeout"), sample(3, 2, status="failed")]
    assert report.predict(sample(2)["case"], rows) is None


def test_no_geometry_precision_dispatch_extrapolation():
    rows = [sample(1, 1), sample(3, 3)]
    assert report.predict(sample(2)["case"], rows) == (2, 2)
    assert report.predict(sample(4)["case"], rows) is None
    for key, value in (("head_dim", "256"), ("dispatch", "other"), ("attn_dtype", "bf16"), ("model_path", "new")):
        assert report.predict({**sample(2)["case"], key: value}, rows) is None


def test_heldout_twenty_percent_gate():
    rows = [sample(1, 1), sample(3, 3), sample(2, 2, role="holdout")]
    assert report.validation(rows)["passed"]
    rows[-1]["collection_seconds"] = 4
    assert not report.validation(rows)["passed"]
    assert not report.validation(rows[:2])["passed"]


def test_group_fifteen_percent_gate_and_missing_setup():
    rows = [
        sample(1, 2, setup_seconds=5),
        sample(2, 3, setup_seconds=5),
        sample(seconds=10, measurement_scope="group", role="holdout"),
    ]
    assert report.phase_accounting_check(rows)["passed"]
    rows[-1]["collection_seconds"] = 20
    assert not report.phase_accounting_check(rows)["passed"]
    assert not report.phase_accounting_check([sample(), rows[-1]])["passed"]


def test_exact_paired_groups_include_setup_growth_once():
    before = sample(
        seconds=10,
        measurement_scope="group",
        scenario="baseline",
        paired_group_id="p",
        group_member_cases=[sample(1)["case"]],
    )
    after = sample(
        seconds=16,
        measurement_scope="group",
        scenario="model",
        paired_group_id="p",
        group_member_cases=[sample(1)["case"], sample(2)["case"]],
    )
    added = {report.case_key(report.clean_case(sample(2)["case"]))}
    result = report.paired_groups([before, after], "model", added, set())
    assert result["inventory_complete"] and not result["complete"] and result["gpu_seconds_range"] == [6, 6]
    assert not report.paired_groups([after], "model", added, set())["complete"]
    after["status"] = "timeout"
    assert not report.paired_groups([before, after], "model", added, set())["complete"]


def test_missing_stratum_keeps_total_unknown(tmp_path):
    case = sample()["case"]
    for name, rows in (("baseline", [case]), ("model", [case, sample(2)["case"]])):
        directory = tmp_path / name
        directory.mkdir()
        with (directory / "vllm_sm103_attention_context.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=case)
            writer.writeheader()
            writer.writerows(rows)
    result = report.report_lane(tmp_path / "baseline", tmp_path / "model", [], "b300_sxm", "vllm")
    operation = result["attention_context"]
    assert operation["added_points"] == operation["uncovered_added_points"] == 1
    assert operation["extra_gpu_hours_range"] is None


def test_unrelated_source_revisions_do_not_form_interpolation_corners():
    assert report.predict(sample(2)["case"], [sample(1, 1), sample(3, 3, source_commit="different")]) is None


def test_full_causal_three_anchor_quadratic_holdout():
    rows = [sample(262144, 9.8785), sample(1048576, 169.3648), sample(524288, 39.7233, role="holdout")]
    for row in rows:
        row["case"]["window_size"] = "0"
    validation = report.validation(rows)
    assert validation["passed"]
    assert validation["max_relative_error"] < 0.06
    assert not report.validation(rows[:2])["passed"]
    for row in rows:
        row["case"]["window_size"] = "4096"
    assert not report.validation(rows)["passed"]


def test_preparation_and_execution_interpolate_separately_without_double_counting():
    rows = [
        sample(1, 3, preparation_seconds=2, execution_seconds=1),
        sample(3, 15, preparation_seconds=6, execution_seconds=9),
    ]
    for row in rows:
        row["case"]["window_size"] = "0"
    case = {**sample(2)["case"], "window_size": "0"}
    assert report.predict(case, rows) == (8, 8)


def test_source_equivalence_verifies_both_immutable_file_hashes(tmp_path, monkeypatch):
    proof = {
        "source_commit": "old",
        "target_commit": "new",
        "backend": "vllm",
        "operation": "attention_context",
        "runtime_dependency_scope_complete": True,
        "files": [
            {"path": path, "sha256": hashlib.sha256(b"identical").hexdigest()}
            for path in (
                "python/aisimulate/collector/vllm/collect_attn.py",
                "python/aisimulate/collector/helper.py",
                "python/aisimulate/collector/case_generator.py",
                "python/aisimulate/collector/registry_types.py",
                "python/aisimulate/collector/vllm/registry.py",
                "python/aisimulate/collector/vllm/utils.py",
            )
        ],
    }
    monkeypatch.setattr(report.subprocess, "check_output", lambda argv: b"identical")
    assert report.verified_sources([proof], tmp_path, "new") == {("old", "vllm", "attention_context")}
    monkeypatch.setattr(
        report.subprocess, "check_output", lambda argv: b"changed" if argv[-1].startswith("new:") else b"identical"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        report.verified_sources([proof], tmp_path, "new")


def test_module_equivalence_requires_model_config_and_exact_model_scope(tmp_path, monkeypatch):
    paths = [f"python/aisimulate/collector/{name}" for name in ("helper.py", "case_generator.py", "registry_types.py")]
    paths += [
        f"python/aisimulate/collector/sglang/{name}"
        for name in (
            "registry.py",
            "collect_mla_module.py",
            "runtime_compat.py",
            "runtime_limits.py",
            "dsa_prefill_graph.py",
        )
    ]
    paths += [
        f"python/aisimulate/src/aisimulate_core/model_configs/nvidia--GLM-5.2-NVFP4_{name}.json"
        for name in ("config", "hf_quant_config")
    ]
    proof = {
        "source_commit": "old",
        "target_commit": "new",
        "backend": "sglang",
        "operation": "dsa_generation_module",
        "model_path": "nvidia/GLM-5.2-NVFP4",
        "files": [{"path": p, "sha256": hashlib.sha256(b"same").hexdigest()} for p in paths],
    }
    monkeypatch.setattr(report.subprocess, "check_output", lambda argv: b"same")
    assert report.verified_sources([proof], tmp_path, "new") == {
        ("old", "sglang", "dsa_generation_module", "nvidia/GLM-5.2-NVFP4")
    }
    proof["files"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        report.verified_sources([proof], tmp_path, "new")


def test_shared_point_anchor_can_cover_identical_model_delta(tmp_path):
    case = sample()["case"]
    for name, rows in (("baseline", [case]), ("model", [case, sample(2)["case"]])):
        directory = tmp_path / name
        directory.mkdir()
        with (directory / "vllm_sm103_attention_context.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=case)
            writer.writeheader()
            writer.writerows(rows)
    anchor = sample(2, 3, scenario="shared")
    lane = report.report_lane(
        tmp_path / "baseline", tmp_path / "model", [anchor], "b300_sxm", "vllm", source_commit="abc"
    )
    assert lane["attention_context"]["covered_added_points"] == 1
    assert lane["attention_context"]["extra_gpu_hours_range"] is None


def repeat_pairs():
    records = []
    for i, (old, new) in enumerate(((10, 16), (11, 18), (10.5, 17)), 1):
        for scenario, seconds, members in (
            ("baseline", old, [sample(1)["case"]]),
            ("model", new, [sample(1)["case"], sample(2)["case"]]),
        ):
            records.append(
                sample(
                    seconds=seconds,
                    measurement_scope="group",
                    scenario=scenario,
                    sample_id=f"repeat-{i}-{scenario}",
                    group_id=f"repeat-{i}-{scenario}",
                    paired_group_id=f"validation-r{i}",
                    group_member_cases=members,
                )
            )
    return records


def test_three_repeat_pair_ids_fold_once_with_independent_gate():
    added = {report.case_key(report.clean_case(sample(2)["case"]))}
    result = report.paired_groups(repeat_pairs(), "model", added, set())
    assert result["complete"] and result["covered_added_points"] == 1
    assert result["gpu_seconds_range"] == [6, 7]
    assert result["repeat_pools"][0]["repeat_count"] == 3


def test_pair_ids_do_not_prove_source_rank_or_cache_compatibility():
    added = {report.case_key(report.clean_case(sample(2)["case"]))}
    for changes in (
        {"source_commit": "unknown"},
        {"participating_gpus": 2},
        {"cache_scope_declared": "cold first-run cache"},
        {"pilot_splits_same_cache_proven": False},
    ):
        records = repeat_pairs()[:2]
        records[1].update(changes)
        assert report.paired_groups(records, "model", added, set())["covered_added_points"] == 0
    records = repeat_pairs()[:2]
    records.append({**records[1], "sample_id": "collision", "participating_gpus": 8})
    result = report.paired_groups(records, "model", added, set())
    assert not result["complete"] and result["excluded_pairs"][0]["reason"] == "ambiguous_pair_id_collision"


def test_conservation_is_not_independent_group_forecast():
    rows = []
    for i, seconds in enumerate((10, 10, 100), 1):
        rows.extend(
            [
                sample(seconds=4, setup_seconds=seconds - 4, group_id=f"g{i}", sample_id=f"p{i}"),
                sample(
                    seconds=seconds,
                    measurement_scope="group",
                    role="holdout",
                    group_id=f"g{i}",
                    sample_id=f"g{i}",
                    group_member_cases=[sample()["case"]],
                ),
            ]
        )
    assert report.phase_accounting_check(rows)["passed"]
    assert not report.group_validation(rows)["passed"]
    assert report.group_validation(rows)["max_relative_error"] == 0.9


def test_report_rejects_unsupported_source_pairs(tmp_path):
    case = sample()["case"]
    for name, rows in (("baseline", [case]), ("model", [case, sample(2)["case"]])):
        directory = tmp_path / name
        directory.mkdir()
        with (directory / "vllm_sm103_attention_context.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=case)
            writer.writeheader()
            writer.writerows(rows)
    records = repeat_pairs()
    lane = report.report_lane(
        tmp_path / "baseline", tmp_path / "model", records, "b300_sxm", "vllm", source_commit="supported"
    )
    operation = lane["attention_context"]
    assert operation["paired_group_measurement"]["covered_added_points"] == 0
    assert operation["extra_gpu_hours_range"] is None


def test_nonhistory_pairs_support_identical_alternatives_without_double_charge(tmp_path):
    records = repeat_pairs()
    for record in records:
        record["operation"] = "moe"
        record["group_member_cases"] = [
            {"num_tokens": c["input_len"], "model_name": "exact-checkpoint", "dispatch": "native", "tp": "1"}
            for c in record["group_member_cases"]
        ]
    before, after = records[0]["group_member_cases"], records[1]["group_member_cases"]
    for name, rows in (("baseline", before), ("shared", after), ("model", after)):
        directory = tmp_path / name
        directory.mkdir()
        with (directory / "vllm_sm103_moe.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    shared = report.report_lane(
        tmp_path / "baseline",
        tmp_path / "shared",
        records,
        "b300_sxm",
        "vllm",
        "shared",
        source_commit="abc",
        alternate_candidate=tmp_path / "model",
    )["moe"]
    model = report.report_lane(
        tmp_path / "baseline", tmp_path / "model", records, "b300_sxm", "vllm", source_commit="abc"
    )["moe"]
    assert shared["paired_group_measurement"]["complete"] and model["paired_group_measurement"]["complete"]
    assert shared["extra_gpu_hours_range"] == model["extra_gpu_hours_range"] == [6 / 3600, 7 / 3600]
    assert shared["paired_group_measurement"]["covered_added_points"] == 1
    assert len(shared["scenario_equivalence_proofs"]) == 3
    assert report.measured_cost(records) == 82.5  # alternative forecasts do not duplicate observations
    assert all(s["scenario"] in {"baseline", "model"} for s in records)
    # Missing or mismatched model/dispatch/task geometry cannot establish equivalence.
    target = report.frozen_tables(str(tmp_path / "shared"))["vllm_sm103_moe"]
    assert report.non_history_scenario_proof(records[1], "moe", target, {}) is None
    assert report.non_history_scenario_proof(records[1], "attention_context", target, target) is None
    changed = {key: {**row, "_task_case": {**row, "tp": "8"}} for key, row in target.items()}
    assert report.non_history_scenario_proof(records[1], "moe", target, changed) is None
