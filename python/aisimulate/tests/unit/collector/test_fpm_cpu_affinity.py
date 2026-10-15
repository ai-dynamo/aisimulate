# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.fpm_forward.cpu_affinity import inspect_cpu_affinity
from collector.fpm_forward.runtime import fpm_memory_observer as observer

from .test_fpm_profile_collection import no_models_or_timing_data  # noqa: F401
from .test_fpm_repeatability import _write_campaign, campaign  # noqa: F401

pytestmark = pytest.mark.unit


def _snapshot(host, pid, cpus):
    return {
        "hostname": host,
        "pid": pid,
        "observer_thread_id": pid,
        "main_thread_allowed_cpus": cpus,
        "threads": [{"tid": pid, "allowed_cpus": cpus}],
        "thread_ids_before": [pid],
        "thread_ids_after": [pid],
        "thread_errors": [],
        "errors": [],
        "status": "observed",
    }


def _campaign(tmp_path, *, tp=1, dp=8, cpus=16, bind="cores"):
    plan = SimpleNamespace(
        sha256="a" * 64,
        options=SimpleNamespace(executor="slurm", slurm_cpus_per_task=cpus, slurm_cpu_bind=bind),
    )
    cell = SimpleNamespace(cell_id="prefill", topology=SimpleNamespace(tp=tp, dp=dp, pp=1))
    collection = SimpleNamespace(collector_attempt_id="attempt")
    provenance = {"plan_sha256": plan.sha256, "cell_id": cell.cell_id, "attempt_id": "attempt"}

    def write(kind, node, pid, dp_rank=None, tp_rank=None, mask=None):
        rank = {"dp_rank": dp_rank, "tp_rank": tp_rank, "pp_rank": 0 if kind == "worker" else None}
        suffix = f"-dp{dp_rank}-tp{tp_rank}-pp0" if kind == "worker" else f"-dp{dp_rank}" if kind == "scheduler" else ""
        path = tmp_path / f"node{node}" / f"fpm-cpu-{kind}{suffix}.json"
        path.parent.mkdir(exist_ok=True)
        payload = {
            "schema_name": "aisimulate_fpm_cpu_affinity",
            "schema_version": 1,
            "kind": kind,
            "measurement": {
                "launcher": "before_engine_start",
                "worker": "after_warmup",
                "scheduler": "scheduler_initialized",
            }[kind],
            "collector_provenance": provenance,
            **rank,
            **_snapshot(f"node{node}", pid, list(range(16)) if mask is None else mask),
        }
        if kind == "launcher":
            payload.update(requested_cpus_per_task=cpus, cpu_bind=bind, local_gpu_count=4)
        path.write_text(json.dumps(payload))
        return path

    for node in range(tp * dp // 4):
        write("launcher", node, 100)
    for d in range(dp):
        for t in range(tp):
            node, local = divmod(d * tp + t, 4)
            write("worker", node, 200 + local, d, t)
        node, local = divmod(d * tp, 4)
        write("scheduler", node, 300 + local, d)
    return plan, cell, collection


def _inspect(tmp_path, context):
    plan, cell, collection = context
    return inspect_cpu_affinity(cell, tmp_path, collection, plan=plan, expected_nodes=2)


def _change(path, transform):
    payload = json.loads(path.read_text())
    transform(payload)
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize("tp,dp", [(1, 8), (8, 1)])
def test_shared_pools_and_identical_cpu_ids_on_different_hosts_are_valid(tmp_path, tp, dp):
    report = _inspect(tmp_path, _campaign(tmp_path, tp=tp, dp=dp))
    assert report["status"] == "qualified"
    assert report["policy_required"] is True
    assert len(report["nodes"]) == 2
    assert not report["failures"]
    for item in report["observations"]:
        raw = Path(item["source"]["path"]).read_bytes()
        assert item["source"]["sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("masks", [[[0]] * 4, [[0], [0], [1, 2], [3]]])
def test_dep_scheduler_masks_must_allow_distinct_logical_cpus(tmp_path, masks):
    context = _campaign(tmp_path)
    for dp, cpus in enumerate(masks):
        path = tmp_path / "node0" / f"fpm-cpu-scheduler-dp{dp}.json"
        _change(path, lambda value, cpus=cpus: value.update(_snapshot("node0", value["pid"], cpus)))
    report = _inspect(tmp_path, context)
    assert report["status"] == "failed"
    assert "cannot run on distinct logical CPUs" in str(report["failures"])


def test_disjoint_scheduler_cpu_sets_are_allowed_but_not_required(tmp_path):
    context = _campaign(tmp_path)
    for dp in range(8):
        node, local = divmod(dp, 4)
        path = tmp_path / f"node{node}" / f"fpm-cpu-scheduler-dp{dp}.json"
        _change(path, lambda value, local=local: value.update(_snapshot(value["hostname"], value["pid"], [local])))
    assert _inspect(tmp_path, context)["status"] == "qualified"


@pytest.mark.parametrize(
    "change,expected",
    [
        (lambda p: p["collector_provenance"].update(attempt_id="other"), "attempt"),
        (lambda p: p.update(dp_rank=8), "rank"),
        (lambda p: p.update(main_thread_allowed_cpus=[True]), "mask"),
        (lambda p: p.update(main_thread_allowed_cpus=[1, 1]), "mask"),
        (lambda p: p.update(hostname="unlaunched-host"), "launcher"),
        (lambda p: p["threads"].append({"tid": 999, "allowed_cpus": [99]}), "thread"),
    ],
)
def test_invalid_or_mismatched_runtime_evidence_is_not_qualified(tmp_path, change, expected):
    context = _campaign(tmp_path)
    _change(tmp_path / "node0/fpm-cpu-worker-dp0-tp0-pp0.json", change)
    report = _inspect(tmp_path, context)
    assert report["status"] == "failed"
    assert expected in str(report["failures"])


def test_observed_thread_outside_launcher_pool_is_detected(tmp_path):
    context = _campaign(tmp_path)
    path = tmp_path / "node0/fpm-cpu-worker-dp0-tp0-pp0.json"

    def change(payload):
        payload["thread_ids_before"].append(999)
        payload["thread_ids_after"].append(999)
        payload["threads"].append({"tid": 999, "allowed_cpus": [99]})

    _change(path, change)
    report = _inspect(tmp_path, context)
    assert report["status"] == "failed"
    assert "outside the launcher CPU pool" in str(report["failures"])


@pytest.mark.parametrize(
    "change,expected",
    [
        (lambda p: p.update(requested_cpus_per_task=8), "policy"),
        (lambda p: p.update(cpu_bind="none"), "policy"),
        (lambda p: p.update(local_gpu_count=8), "worker"),
        (lambda p: p.update(_snapshot("node0", 100, [0])), "requested"),
    ],
)
def test_launch_pool_and_requested_policy_are_independently_checked(tmp_path, change, expected):
    context = _campaign(tmp_path)
    _change(tmp_path / "node0/fpm-cpu-launcher.json", change)
    report = _inspect(tmp_path, context)
    assert report["status"] == "failed"
    assert expected in str(report["failures"])


@pytest.mark.parametrize("requested,observed", [(16, 16.0), (1, True)])
def test_launcher_policy_requires_an_integer_even_when_numeric_value_matches(tmp_path, requested, observed):
    context = _campaign(tmp_path, tp=8, dp=1, cpus=requested)
    assert _inspect(tmp_path, context)["status"] == "qualified"
    _change(tmp_path / "node0/fpm-cpu-launcher.json", lambda value: value.update(requested_cpus_per_task=observed))
    report = _inspect(tmp_path, context)
    assert report["status"] == "failed"
    assert "policy" in str(report["failures"])


@pytest.mark.parametrize("missing", ["worker-dp0-tp0-pp0", "scheduler-dp0", "launcher"])
def test_missing_role_evidence_leaves_cpu_qualification_incomplete(tmp_path, missing):
    context = _campaign(tmp_path)
    (tmp_path / f"node0/fpm-cpu-{missing}.json").unlink()
    report = _inspect(tmp_path, context)
    assert report["status"] == "incomplete"
    assert report["missing_evidence"]


def test_partial_thread_snapshot_cannot_claim_complete_affinity(tmp_path):
    context = _campaign(tmp_path)
    path = tmp_path / "node0/fpm-cpu-worker-dp0-tp0-pp0.json"

    def change(payload):
        payload["status"] = "partial"
        payload["thread_ids_after"].append(999)
        payload["errors"].append("thread list changed during observation")

    _change(path, change)
    report = _inspect(tmp_path, context)
    assert report["status"] == "incomplete"
    assert "partial" in str(report["missing_evidence"])


def test_legacy_plan_is_not_retroactively_given_verified_affinity(tmp_path):
    context = _campaign(tmp_path, cpus=None, bind=None)
    report = _inspect(tmp_path, context)
    assert report["status"] == "incomplete"
    assert report["policy_required"] is False
    assert "frozen CPU policy" in str(report["missing_evidence"])


def test_duplicate_rank_observation_is_rejected(tmp_path):
    context = _campaign(tmp_path)
    path = tmp_path / "node0/fpm-cpu-worker-dp0-tp0-pp0.json"
    copied = tmp_path / "copy"
    copied.mkdir()
    (copied / path.name).write_bytes(path.read_bytes())
    report = _inspect(tmp_path, context)
    assert report["status"] == "failed"
    assert "duplicate" in str(report["failures"])


def test_scheduler_host_cannot_hide_local_cpu_contention(tmp_path):
    context = _campaign(tmp_path)
    path = tmp_path / "node0/fpm-cpu-scheduler-dp0.json"
    _change(path, lambda payload: payload.update(_snapshot("node1", 999, list(range(16)))))
    report = _inspect(tmp_path, context)
    assert report["status"] == "failed"
    assert "scheduler host differs" in str(report["failures"])


def test_different_rank_cannot_reuse_another_process_observation(tmp_path):
    context = _campaign(tmp_path)
    path = tmp_path / "node0/fpm-cpu-scheduler-dp1.json"
    _change(path, lambda payload: payload.update(_snapshot("node0", 300, list(range(16)))))
    report = _inspect(tmp_path, context)
    assert report["status"] == "failed"
    assert "duplicate process" in str(report["failures"])


def test_complete_snapshot_requires_observation_thread_membership(tmp_path):
    context = _campaign(tmp_path)
    _change(tmp_path / "node0/fpm-cpu-launcher.json", lambda payload: payload.update(observer_thread_id=999))
    assert _inspect(tmp_path, context)["status"] == "failed"


def test_json_duplicate_fields_do_not_overwrite_evidence_identity(tmp_path):
    context = _campaign(tmp_path)
    path = tmp_path / "node0/fpm-cpu-launcher.json"
    path.write_text(path.read_text()[:-1] + ', "pid": 999}')
    report = _inspect(tmp_path, context)
    assert report["status"] == "failed"
    assert "duplicate" in str(report["failures"])


def test_generated_node_count_cannot_be_replaced_by_observed_host_claims(tmp_path):
    plan, cell, collection = _campaign(tmp_path)
    report = inspect_cpu_affinity(cell, tmp_path, collection, plan=plan, expected_nodes=1)
    assert report["status"] == "failed"
    assert "generated" in str(report["failures"])


def test_launcher_support_threads_must_stay_inside_its_main_thread_pool(tmp_path):
    context = _campaign(tmp_path)
    path = tmp_path / "node0/fpm-cpu-launcher.json"

    def change(payload):
        payload["thread_ids_before"].append(101)
        payload["thread_ids_after"].append(101)
        payload["threads"].append({"tid": 101, "allowed_cpus": [99]})

    _change(path, change)
    assert "outside the launcher CPU pool" in str(_inspect(tmp_path, context)["failures"])


@pytest.mark.parametrize("state", ["missing", "qualified", "failed"])
def test_fresh_cpu_policy_gates_readiness_and_quality_without_discarding_native_rows(
    campaign,  # noqa: F811
    tmp_path,
    monkeypatch,
    state,
):
    from collector.fpm_forward import repeatability, runner

    from aisimulate.support.collection_readiness import _cell_report
    from aisimulate.support.validation_workflow import _execution_status

    source_plan, _, _ = campaign
    plan = replace(
        source_plan,
        options=replace(
            source_plan.options,
            executor="slurm",
            slurm_container_image="synthetic",
            slurm_cpus_per_task=16,
            slurm_cpu_bind="cores",
        ),
    )
    plan = repeatability._subset_plan(
        plan, {"cell_id": plan.cells[0].cell_id, "benchmark_points": json.loads(plan.options.benchmark_points_json)}
    )
    source, checkpoint = tmp_path / "fresh-cpu-policy", tmp_path / "fresh-checkpoint.json"
    _write_campaign(source, checkpoint, plan)
    cell = plan.cells[0]
    cell_dir = source / "cells" / cell.cell_id
    raw = cell_dir / "raw"
    nodes = runner._expected_nodes(cell_dir / runner.FPM_MANIFEST_FILENAME)
    local_gpus = cell.topology.total_gpus // nodes
    node_directories = [raw / "pod" if node == 0 else raw / f"cpu-node{node}" for node in range(nodes)]
    if state != "missing":
        for node in range(nodes):
            directory = node_directories[node]
            directory.mkdir(exist_ok=True)
            (directory / "collector-provenance.json").write_bytes((raw / "pod/collector-provenance.json").read_bytes())
            (directory / runner.POINTS_RECEIPT_FILENAME).write_bytes(
                (raw / "pod" / runner.POINTS_RECEIPT_FILENAME).read_bytes()
            )
            monkeypatch.setattr(
                observer, "cpu_affinity_snapshot", lambda node=node: _snapshot(f"node{node}", 100, list(range(16)))
            )
            observer.observe_cpu(
                "launcher",
                requested_cpus_per_task=16,
                cpu_bind="cores",
                local_gpu_count=local_gpus,
                directory=directory,
            )
        for dp in range(cell.topology.dp):
            for tp in range(cell.topology.tp):
                node, local = divmod(dp * cell.topology.tp + tp, local_gpus)
                directory = node_directories[node]
                mask = [99] if state == "failed" and (dp, tp) == (0, 0) else list(range(16))
                monkeypatch.setattr(
                    observer,
                    "cpu_affinity_snapshot",
                    lambda node=node, local=local, mask=mask: _snapshot(f"node{node}", 200 + local, mask),
                )
                observer.observe_cpu("worker", dp_rank=dp, tp_rank=tp, pp_rank=0, directory=directory)
            node, local = divmod(dp * cell.topology.tp, local_gpus)
            monkeypatch.setattr(
                observer,
                "cpu_affinity_snapshot",
                lambda node=node, local=local: _snapshot(f"node{node}", 300 + local, list(range(16))),
            )
            observer.observe_cpu("scheduler", dp_rank=dp, directory=node_directories[node])
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    entry = json.loads(checkpoint.read_text())["cells"][cell.cell_id]
    readiness = _cell_report(plan, cell, source, entry)
    assert readiness["direct_eligible_points"] > 0
    assert readiness["execution"]["cpu_affinity"]["status"] == ("incomplete" if state == "missing" else state)
    assert readiness["status"] == ("ready" if state == "qualified" else "blocked")
    frozen = repeatability.freeze_repeatability_plan(plan, source, checkpoint, max_points_per_cell=6)
    assert _execution_status(frozen) == {"missing": "incomplete", "qualified": "passed", "failed": "failed"}[state]
    assert {path: path.read_bytes() for path in source.rglob("*") if path.is_file()} == before


def test_linux_snapshot_distinguishes_main_thread_from_observation_thread(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for tid in (100, 101):
        (tasks / str(tid)).mkdir()
    monkeypatch.setattr(observer.os, "getpid", lambda: 100)
    monkeypatch.setattr(observer.threading, "get_native_id", lambda: 101)
    monkeypatch.setattr(observer.socket, "gethostname", lambda: "node0")
    calls = []

    def affinity(tid):
        calls.append(tid)
        return {0, 1, 2, 3} if tid == 100 else {2}

    monkeypatch.setattr(observer.os, "sched_getaffinity", affinity, raising=False)
    payload = observer.cpu_affinity_snapshot(task_directory=tasks)
    assert payload["status"] == "observed"
    assert payload["main_thread_allowed_cpus"] == [0, 1, 2, 3]
    assert payload["observer_thread_id"] == 101
    assert payload["threads"] == [{"tid": 100, "allowed_cpus": [0, 1, 2, 3]}, {"tid": 101, "allowed_cpus": [2]}]
    assert 0 not in calls


def test_thread_disappearance_is_recorded_without_changing_affinity(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for tid in (100, 101):
        (tasks / str(tid)).mkdir()
    monkeypatch.setattr(observer.os, "getpid", lambda: 100)

    def affinity(tid):
        if tid == 101:
            raise ProcessLookupError("thread exited")
        return {0, 1}

    monkeypatch.setattr(observer.os, "sched_getaffinity", affinity, raising=False)
    payload = observer.cpu_affinity_snapshot(task_directory=tasks)
    assert payload["status"] == "partial"
    assert payload["main_thread_allowed_cpus"] == [0, 1]
    assert payload["thread_errors"][0]["tid"] == 101


def test_unavailable_affinity_stays_unknown(monkeypatch):
    monkeypatch.delattr(observer.os, "sched_getaffinity", raising=False)
    payload = observer.cpu_affinity_snapshot()
    assert payload["status"] == "unavailable"
    assert payload["main_thread_allowed_cpus"] == []
    assert payload["errors"]


def test_sidecar_preserves_failed_snapshot_and_attempt_binding(tmp_path, monkeypatch):
    provenance = {"plan_sha256": "a" * 64, "cell_id": "prefill", "attempt_id": "current"}
    (tmp_path / "collector-provenance.json").write_text(json.dumps(provenance))
    snapshot = _snapshot("node0", 100, [0])
    monkeypatch.setattr(observer, "cpu_affinity_snapshot", lambda: snapshot)
    result = observer.observe_cpu(
        "launcher", requested_cpus_per_task=16, cpu_bind="cores", local_gpu_count=4, directory=tmp_path
    )
    path = tmp_path / "fpm-cpu-launcher.json"
    original = path.read_bytes()
    assert result == json.loads(original)
    assert result["collector_provenance"] == provenance
    assert result["main_thread_allowed_cpus"] == [0]
    assert result["requested_cpus_per_task"] == 16
    with pytest.raises(RuntimeError, match="duplicate"):
        observer.observe_cpu(
            "launcher", requested_cpus_per_task=16, cpu_bind="cores", local_gpu_count=4, directory=tmp_path
        )
    assert path.read_bytes() == original
