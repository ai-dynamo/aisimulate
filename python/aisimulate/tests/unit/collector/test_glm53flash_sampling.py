# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Candidate planning checks; none of these tests establishes GPU admission."""

import argparse
import hashlib
import json
from dataclasses import replace

import pytest

from collector.fpm_forward.config import FPMCollectionOptions
from collector.fpm_forward.glm53flash_sampling import (
    SamplingOptions,
    add_calibration_points,
    canonical,
    generate,
    main,
    write_bundle,
)
from collector.fpm_forward.glm53flash_validation import _geometry, _group

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def campaign():
    return generate()


def geometry(candidate):
    return _geometry(
        dict(
            candidate["point"],
            point_type=candidate["phase"],
            total_prefill_tokens=candidate["point"].get("total_prefill_tokens", 0),
        )
    )


def test_bounded_defaults_cover_all_cells_phases_and_formal_batches(campaign):
    assert len(campaign["cells"]) == 8
    assert 300 < sum(len(points) for points in campaign["points"].values()) < 800
    assert campaign["status"] == "CANDIDATES_NOT_QUALIFIED"
    assert campaign["accuracy_acceptance"] == "NOT_EVALUATED"
    for role, candidates in campaign["points"].items():
        for phase in ("prefill", "decode"):
            selected = [candidate for candidate in candidates if candidate["phase"] == phase]
            assert {1, 2, 4, 8, 16, 32} <= {item["point"]["batch_size"] for item in selected}
            assert {"1K-32K", "64K", "128K"} <= {
                _group(
                    dict(
                        item["point"],
                        point_type=phase,
                        total_prefill_tokens=item["point"].get("total_prefill_tokens", 0),
                    )
                )
                for item in selected
            }
        assert all(candidate["qualification_status"] == "NOT_EVALUATED" for candidate in candidates)
        assert all(geometry(candidate) for candidate in candidates)
        assert len({candidate["candidate_id"] for candidate in candidates}) == len(candidates)


def test_calibration_holdout_identity_and_geometry_are_globally_disjoint(campaign):
    calibration = campaign["points"]["calibration"]
    holdout = campaign["points"]["holdout"]
    assert not ({geometry(point) for point in calibration} & {geometry(point) for point in holdout})
    assert not ({point["candidate_id"] for point in calibration} & {point["candidate_id"] for point in holdout})
    assert campaign["corpora"]["calibration"]["sha256"] != campaign["corpora"]["holdout"]["sha256"]
    assert campaign["request_namespaces"]["calibration"] != campaign["request_namespaces"]["holdout"]


def test_every_holdout_has_actual_same_axis_calibration_brackets(campaign):
    calibration = {candidate["candidate_id"]: geometry(candidate) for candidate in campaign["points"]["calibration"]}
    for candidate in campaign["points"]["holdout"]:
        point = geometry(candidate)
        assert candidate["calibration_brackets"]
        for bracket in candidate["calibration_brackets"]:
            lower, upper = calibration[bracket["lower"]], calibration[bracket["upper"]]
            axis = 2 if bracket["axis"] == "query" else 3
            assert lower[axis] < point[axis] < upper[axis]
            assert all(lower[index] == point[index] == upper[index] for index in range(4) if index != axis)


def test_mod4_chunk_and_graph_edges_are_requested_without_admission(campaign):
    for role, candidates in campaign["points"].items():
        residues = {
            candidate["point"]["total_kv_read_tokens"] // candidate["point"]["batch_size"] % 4
            for candidate in candidates
        }
        assert residues == {0, 1, 2, 3}
    calibration = campaign["points"]["calibration"]
    labels = {label for candidate in calibration for label in candidate["families"]}
    assert {"prefix_block_4352_indexpool_mod4", "new_tokens_chunk_8192", "decode_graph_batch_32"} <= labels
    assert any(candidate["phase"] == "decode" and candidate["point"]["batch_size"] == 31 for candidate in calibration)
    assert any(candidate["inclusive_context"] == 131072 for candidate in calibration)
    assert all(candidate["point"].get("total_prefill_tokens", 0) <= 8192 for candidate in calibration)
    assert all(item["reason"].startswith("declared_") for item in campaign["excluded_by_declared_bounds"])
    assert any(item["batch_size"] == 33 for item in campaign["excluded_by_declared_bounds"])


def test_existing_planner_accepts_exact_schema3_payloads_and_hashes(campaign, tmp_path):
    output = tmp_path / "bundle"
    write_bundle(output, campaign)
    for role in ("calibration", "holdout"):
        points = output / f"{role}-points.json"
        options = FPMCollectionOptions.from_args(
            argparse.Namespace(fpm_max_gpus=4, fpm_benchmark_points_file=str(points))
        )
        assert options.benchmark_points_json == canonical(campaign["payloads"][role])
        assert options.benchmark_points_sha256 == campaign["payload_sha256"][role]
        assert hashlib.sha256((output / f"{role}.txt").read_bytes()).hexdigest() == campaign["corpora"][role]["sha256"]
    qualification = json.loads((output / "qualification-template.json").read_text())
    total = sum(len(points) for points in campaign["points"].values())
    assert len(qualification["cells"]) == 8
    assert all(len(cell["candidates"]) == total for cell in qualification["cells"])
    assert all(
        point["status"] == "NOT_EVALUATED" and not point["native_schedule_receipts"] and not point["failure_receipts"]
        for cell in qualification["cells"]
        for point in cell["candidates"]
    )
    with pytest.raises(FileExistsError):
        write_bundle(output, campaign)


def test_reproducible_cli_and_pre_freeze_density_changes(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert main(["--output", str(first)]) == 0
    assert main(["--output", str(second)]) == 0
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    changed = generate(
        replace(SamplingOptions(), contexts=(1024, 2048, 4096, 8192, 12288, 16384, 32768, 65536, 131072))
    )
    default = json.loads((first / "candidate-inventory.json").read_text())
    assert changed["campaign_id"] != default["campaign_id"]
    assert changed["payload_sha256"] != default["payload_sha256"]


@pytest.mark.parametrize(
    "options",
    [
        replace(SamplingOptions(), batches=(1, 2, 4)),
        replace(SamplingOptions(), contexts=(1024, 65536)),
        replace(SamplingOptions(), cached_queries=(32, 512)),
        replace(SamplingOptions(), graph_batch_anchors=(4, 64)),
    ],
)
def test_invalid_declared_matrix_fails_before_generating_points(options):
    with pytest.raises(ValueError):
        generate(options)


def test_additive_anchors_preserve_original_holdout_and_candidate_identity(campaign, tmp_path):
    supplement = {
        "schema_version": 3,
        "prefill": [{"batch_size": 2, "total_prefill_tokens": 6, "total_kv_read_tokens": 14}],
        "decode": [campaign["payloads"]["calibration"]["decode"][0]],
    }
    before = canonical(campaign)
    extended = add_calibration_points(campaign, supplement)
    assert canonical(campaign) == before
    assert extended["campaign_id"] != campaign["campaign_id"]
    assert extended["points"]["holdout"] == campaign["points"]["holdout"]
    assert canonical(extended["payloads"]["holdout"]) == canonical(campaign["payloads"]["holdout"])
    assert extended["payload_sha256"]["holdout"] == campaign["payload_sha256"]["holdout"]
    original = {item["candidate_id"]: item for item in campaign["points"]["calibration"]}
    assert all(
        item == original[item["candidate_id"]]
        for item in extended["points"]["calibration"]
        if item["candidate_id"] in original
    )
    assert extended["calibration_extension"]["added_points"] == 1
    assert extended["calibration_extension"]["already_requested_points"] == 1
    assert all(item["qualification_status"] == "NOT_EVALUATED" for item in extended["points"]["calibration"])
    test_every_holdout_has_actual_same_axis_calibration_brackets(extended)
    test_calibration_holdout_identity_and_geometry_are_globally_disjoint(extended)
    path = tmp_path / "supplement.json"
    path.write_text(canonical(supplement))
    output = tmp_path / "extended"
    assert main(["--output", str(output), "--supplemental-calibration-points", str(path)]) == 0
    actual = json.loads((output / "candidate-inventory.json").read_text())
    assert actual["campaign_id"] == extended["campaign_id"]
    assert json.loads((output / "supplemental-calibration-points.json").read_text()) == supplement
    test_existing_planner_accepts_exact_schema3_payloads_and_hashes(extended, tmp_path)


@pytest.mark.parametrize(
    "phase,point",
    [
        ("prefill", {"batch_size": True, "total_prefill_tokens": 6, "total_kv_read_tokens": 14}),
        ("prefill", {"batch_size": 2, "total_prefill_tokens": 5, "total_kv_read_tokens": 14}),
        ("prefill", {"batch_size": 2, "total_prefill_tokens": 6, "total_kv_read_tokens": 15}),
        ("prefill", {"batch_size": 1, "total_prefill_tokens": 8193, "total_kv_read_tokens": 0}),
        ("decode", {"batch_size": 1, "total_kv_read_tokens": 131072}),
        ("decode", {"batch_size": 33, "total_kv_read_tokens": 33}),
        ("decode", {"batch_size": 1, "total_kv_read_tokens": 0}),
        ("decode", {"batch_size": 1, "total_kv_read_tokens": 1, "latency": 0.1}),
    ],
)
def test_additive_anchors_reject_nonphysical_or_noncanonical_points(campaign, phase, point):
    payload = {"schema_version": 3, "prefill": [], "decode": []}
    payload[phase].append(point)
    with pytest.raises(ValueError):
        add_calibration_points(campaign, payload)


def test_additive_anchors_never_admit_holdout_contamination_or_duplicate_requests(campaign):
    row = campaign["payloads"]["holdout"]["prefill"][0]
    payload = {"schema_version": 3, "prefill": [row], "decode": []}
    with pytest.raises(ValueError, match="contaminate"):
        add_calibration_points(campaign, payload)
    row = campaign["payloads"]["calibration"]["prefill"][0]
    payload["prefill"] = [row, row]
    with pytest.raises(ValueError, match="duplicate"):
        add_calibration_points(campaign, payload)
    payload["prefill"] = []
    with pytest.raises(ValueError, match="at least one"):
        add_calibration_points(campaign, payload)
