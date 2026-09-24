# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from collector.fpm_forward.config import FPMCollectionOptions
from collector.fpm_forward.planner import build_collection_plan
from collector.fpm_forward.shards import (
    canonical,
    digest,
    make_shards,
    run_sharded_collection,
    shard_manifest,
    validate_point_union,
)

pytestmark = pytest.mark.unit


def plan():
    payload = {
        "schema_version": 3,
        "prefill": [
            {"batch_size": 1, "total_prefill_tokens": 32, "total_kv_read_tokens": 1024},
            {"batch_size": 1, "total_prefill_tokens": 3, "total_kv_read_tokens": 4097},
            {"batch_size": 1, "total_prefill_tokens": 4, "total_kv_read_tokens": 131068},
        ],
        "decode": [{"batch_size": 1, "total_kv_read_tokens": 1024}],
    }
    options = FPMCollectionOptions(
        max_gpus=2,
        gpu_counts=(2,),
        parallel_presets=("pure_tp",),
        parallel_axes=(),
        moe_backend="auto",
        attention_backend="auto",
        enable_wideep="false",
        enable_eplb="false",
        weight_quantizations=(),
        kv_cache_dtypes=("auto",),
        vllm_max_model_len=131072,
        warmup_iterations=0,
        benchmark_points_json=canonical(payload),
        benchmark_points_sha256=digest(payload),
        shard_token_budget=20000,
    )
    return build_collection_plan(
        backend="sglang",
        model_path="zai-org/GLM-5.3-Flash",
        system="gb300",
        selected_ops=set(),
        options=options,
    )


def test_shards_preserve_every_original_point_and_oversized_repetition_group():
    parent = plan()
    before = parent.options.benchmark_points_json
    shards = make_shards(parent)
    assert len(shards) == 4
    assert len({shard.plan.cells[0].cell_id for shard in shards}) == 4
    assert len({shard.plan.sha256 for shard in shards}) == 4
    assert any(shard.identity["oversized_single_point"] for shard in shards)
    assert parent.options.benchmark_points_json == before
    assert shard_manifest(parent) == shard_manifest(parent)
    validate_point_union(
        parent.to_dict(),
        shard_manifest(parent),
        {shard.plan.cells[0].cell_id: shard.plan.to_dict() for shard in shards},
    )
    prefill = [shard for shard in shards if shard.identity["phase"] == "prefill"]
    assert [shard.identity["point_map"][0]["original_point_id"] for shard in prefill] == [1, 2, 3]
    assert all(shard.identity["point_map"][0]["native_benchmark_id"] == 1 for shard in prefill)
    assert all(shard.plan.to_dict()["options"]["warmup_repeats"] == 5 for shard in shards)
    assert all(shard.plan.to_dict()["options"]["measurement_repeats"] == 10 for shard in shards)


@pytest.mark.parametrize(
    "corruption", ["missing", "overlap", "coordinate", "corpus", "role", "capability", "parent_payload"]
)
def test_shard_union_rejects_lost_or_altered_original_work(corruption):
    parent = plan()
    shards = make_shards(parent)
    manifest = shard_manifest(parent, shards)
    children = {shard.plan.cells[0].cell_id: shard.plan.to_dict() for shard in shards}
    first = manifest["shards"][0]
    if corruption == "missing":
        manifest["shards"].pop()
    elif corruption == "overlap":
        manifest["shards"].append(copy.deepcopy(first))
    elif corruption == "coordinate":
        first["point_map"][0]["point"] = {"batch_size": 2, "total_kv_read_tokens": 4, "total_prefill_tokens": 2}
    elif corruption == "corpus":
        children[first["child_cell_id"]]["options"]["input_text_sha256"] = "f" * 64
    elif corruption == "role":
        children[first["child_cell_id"]]["options"]["dataset_role"] = "holdout"
    elif corruption == "capability":
        children[first["child_cell_id"]]["capability"]["model_config"] = {"substituted_config": True}
    parent_payload = parent.to_dict()
    if corruption == "parent_payload":
        parent_payload["options"]["benchmark_points"]["payload"]["prefill"][0]["total_prefill_tokens"] = 33
    with pytest.raises(ValueError):
        validate_point_union(parent_payload, manifest, children)


def fake_campaign(monkeypatch, tmp_path, *, fail=False):
    import collector.fpm_forward.database as database
    import collector.fpm_forward.runner as runner

    calls, writes = [], []

    def execute(child, **kwargs):
        assert kwargs["collect_only"] is True
        calls.append(child)
        cid = child.cells[0].cell_id
        checkpoint = Path(kwargs["checkpoint_dir"])
        checkpoint.mkdir(parents=True, exist_ok=True)
        status = "failed" if fail and len(calls) == 1 else "passed"
        (checkpoint / "fpm_forward.json").write_text(
            json.dumps({"cells": {cid: {"status": status, "attempt_id": f"attempt-{len(calls)}"}}})
        )
        raw = Path(kwargs["artifact_root"]) / child.sha256[:16] / "cells" / cid / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / "cpu-only-fixture.json").write_text(json.dumps({"status": status, "gpu_evidence": False}))
        return [{"error_type": "fixture_failure"}] if status == "failed" else []

    def aggregate(child, cell, directory, *, expected_attempt_id):
        return [
            {
                **point,
                "total_prefill_tokens": point.get("total_prefill_tokens", 0),
                "cell_id": cell.cell_id,
                "source_plan_sha256": child.sha256,
                "collector_attempt_id": expected_attempt_id,
                "runtime_run_id": f"native-{cell.cell_id}",
            }
            for point in json.loads(child.options.benchmark_points_json)[cell.workload_kind]
        ]

    def publish(parent, rows, **kwargs):
        assert kwargs["reject_replaced_cells"] is True
        assert len(calls) >= len(make_shards(parent))
        assert {row["source_plan_sha256"] for row in rows} == {child.sha256 for child in calls}
        assert len({row["cell_id"] for row in rows}) == len(make_shards(parent))
        writes.append(rows)
        return tmp_path / "db.parquet", tmp_path / "db.json", ()

    monkeypatch.setattr(runner, "run_collection", execute)
    monkeypatch.setattr(database, "aggregate_cell", aggregate)
    monkeypatch.setattr(database, "write_formal_database", publish)
    monkeypatch.setattr(database, "validate_formal_database_commit", lambda *args, **kwargs: {"cell_rows": {}})
    return calls, writes


@pytest.mark.parametrize("fail", [False, True])
def test_existing_child_runner_continues_failures_and_publishes_complete_union_once(monkeypatch, tmp_path, fail):
    calls, writes = fake_campaign(monkeypatch, tmp_path, fail=fail)
    parent = plan()
    errors = run_sharded_collection(
        parent,
        generator_overrides={},
        checkpoint_dir=str(tmp_path / "checkpoint"),
        artifact_root=str(tmp_path / "artifacts"),
        resume=False,
        retry_failed=False,
        database_root=str(tmp_path / "db"),
    )
    assert len(calls) == 4
    assert bool(errors) is fail
    assert len(writes) == (0 if fail else 1)
    checkpoint = json.loads((tmp_path / "checkpoint/fpm_forward_sharded.json").read_text())
    assert checkpoint["status"] == ("incomplete" if fail else "passed")
    assert len(checkpoint["shards"]) == 4
    if fail:
        assert checkpoint["missing_children"]


def test_holdout_complete_union_never_publishes_consumer_data(monkeypatch, tmp_path):
    calls, writes = fake_campaign(monkeypatch, tmp_path)
    parent = plan()
    parent = replace(parent, options=replace(parent.options, dataset_role="holdout"))
    assert not run_sharded_collection(
        parent,
        generator_overrides={},
        checkpoint_dir=str(tmp_path / "checkpoint"),
        artifact_root=str(tmp_path / "artifacts"),
        resume=False,
        retry_failed=False,
    )
    assert len(calls) == 4 and writes == []


@pytest.mark.parametrize("mixed_runtime", [False, True])
def test_acceptance_maps_child_values_back_to_original_ids(monkeypatch, mixed_runtime):
    from collector.fpm_forward import glm53flash_validation as validation

    children = [
        {
            "key": ("sglang", "fp8", 2, "prefill"),
            "role": "holdout",
            "cell": {"cell_id": f"child-{index}"},
            "plan": {"sha256": str(index)},
            "original_point_ids": {1: original},
        }
        for index, original in enumerate((3, 1, 2))
    ]
    parent = {
        "children": children,
        "key": children[0]["key"],
        "role": "holdout",
        "points": [{"benchmark_id": index} for index in (1, 2, 3)],
    }
    monkeypatch.setattr(
        validation,
        "_native_run",
        lambda run, base: {
            "values": {1: run["original_point_ids"][1] * 1.1},
            "request_ids": {run["cell"]["cell_id"]},
            "backend_version": "0.5.20+other" if mixed_runtime and run is children[1] else "0.5.20",
            "receipts": [],
        },
    )
    if mixed_runtime:
        with pytest.raises(ValueError, match="shard runtime versions differ"):
            validation._load_native(parent, Path("."), "fpm")
        return
    native = validation._load_native(parent, Path("."), "fpm")
    assert native["values"] == {1: 1.1, 2: 2.2, 3: 3.3000000000000003}
    assert len(native["shards"]) == 3


def test_retry_preserves_complete_attempt_and_refuses_archive_overwrite(tmp_path):
    from collector.fpm_forward.runner import _archive_native_attempt, _file_metadata

    cell = tmp_path / "cell"
    (cell / "raw").mkdir(parents=True)
    (cell / "logs").mkdir()
    original = {
        "raw/rank.jsonl": b"failed native observation\n",
        "logs/stderr": b"native traceback",
        "run.sh": b"native command",
    }
    for name, content in original.items():
        (cell / name).write_bytes(content)
    previous = {"attempt_id": "failed-123", "status": "failed"}
    _archive_native_attempt(cell, previous)
    archive = cell / "attempts/failed-123"
    receipt = json.loads((archive / "file-receipts.json").read_text())
    for name, content in original.items():
        assert (archive / name).read_bytes() == content
        assert receipt[name] == _file_metadata(archive / name)
    sealed = (archive / "file-receipts.json").read_bytes()
    _archive_native_attempt(cell, previous)  # no new working attempt, no mutation
    assert (archive / "file-receipts.json").read_bytes() == sealed
    (cell / "run.sh").write_bytes(b"new command")
    with pytest.raises(ValueError, match="overwrite archived"):
        _archive_native_attempt(cell, previous)
    assert (cell / "run.sh").read_bytes() == b"new command"
    assert (archive / "file-receipts.json").read_bytes() == sealed


def test_parent_resume_rejects_changed_budget_and_retains_failed_map(monkeypatch, tmp_path):
    calls, writes = fake_campaign(monkeypatch, tmp_path, fail=True)
    parent = plan()
    kwargs = dict(
        generator_overrides={},
        checkpoint_dir=str(tmp_path / "checkpoint"),
        artifact_root=str(tmp_path / "artifacts"),
        retry_failed=True,
    )
    assert run_sharded_collection(parent, resume=False, **kwargs)
    checkpoint = tmp_path / "checkpoint/fpm_forward_sharded.json"
    before = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="requires --resume"):
        run_sharded_collection(parent, resume=False, **kwargs)
    assert checkpoint.read_bytes() == before
    changed = replace(parent, options=replace(parent.options, shard_token_budget=100000))
    with pytest.raises(ValueError, match="frozen shard identity changed"):
        run_sharded_collection(changed, resume=True, **kwargs)
    assert checkpoint.read_bytes() == before
    assert len(calls) == 4 and writes == []
    assert not run_sharded_collection(parent, resume=True, **kwargs)
    assert len(calls) == 8 and len(writes) == 1


def test_acceptance_freezes_child_mapping_from_hashed_parent_inventory(tmp_path):
    import hashlib

    from collector.fpm_forward import glm53flash_validation as validation

    parent = plan()
    parent = replace(
        parent,
        options=replace(parent.options, input_text_sha256="a" * 64),
        cells=tuple(replace(cell, input_text_sha256="a" * 64) for cell in parent.cells),
    )
    shards = make_shards(parent)
    manifest = shard_manifest(parent, shards)

    def receipt(name, value):
        path = tmp_path / name
        path.write_text(canonical(value))
        return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    cid = parent.cells[0].cell_id
    spec = {
        "plan": receipt("parent.json", parent.to_dict()),
        "cell_id": cid,
        "shard_manifest": receipt("shards.json", manifest),
        "shards": [
            {
                "plan": receipt(f"{shard.identity['child_cell_id']}.json", shard.plan.to_dict()),
                "cell_id": shard.identity["child_cell_id"],
                "attempt_id": "native-attempt",
                "raw_root": "raw",
            }
            for shard in shards
            if shard.identity["parent_cell_id"] == cid
        ],
    }
    run = validation._plan_run(spec, tmp_path, "calibration")
    assert run["parent_cell_id"] == cid
    assert [child["original_point_ids"] for child in run["children"]] == [{1: 1}, {1: 2}, {1: 3}]
    spec["shards"].pop()
    with pytest.raises(ValueError, match="missing|union"):
        validation._plan_run(spec, tmp_path, "calibration")


def test_cli_budget_and_outer_timeout_are_explicit_frozen_controls():
    import argparse

    from collector.fpm_forward.config import add_fpm_arguments, reject_fpm_arguments_without_fpm

    parser = argparse.ArgumentParser()
    add_fpm_arguments(parser)
    args = parser.parse_args(
        ["--fpm-max-gpus", "2", "--fpm-shard-token-budget", "2000000", "--fpm-execution-timeout-seconds", "18000"]
    )
    options = FPMCollectionOptions.from_args(args)
    assert options.to_dict()["shard_token_budget"] == 2000000
    assert options.to_dict()["execution_timeout_seconds"] == 18000
    args.ops = []
    with pytest.raises(ValueError):
        reject_fpm_arguments_without_fpm(args)
    with pytest.raises(SystemExit):
        parser.parse_args(["--fpm-shard-token-budget", "0"])
