# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os

import pytest
from collector.campaigns.run_shard import finalized_all_ids, reap_finished_workers

pytestmark = pytest.mark.unit


def test_cleanup_requires_all_ids_and_clean_committed_sidecar(tmp_path):
    cp = tmp_path / "checkpoint/vllm"
    cp.mkdir(parents=True)
    (cp / "a.json").write_text(json.dumps({"done": ["a"], "failed": ["b"]}))
    assert not finalized_all_ids(tmp_path, 2)
    (tmp_path / "collection_meta.yaml").write_text("committed")
    assert finalized_all_ids(tmp_path, 2)
    assert not finalized_all_ids(tmp_path, 3)
    (tmp_path / ".collection_meta.transaction.json").write_text("pending")
    assert not finalized_all_ids(tmp_path, 2)


def test_reap_only_own_completed_worker_children(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append(pid))
    for pid, parent, uid, command in [
        (11, 10, os.getuid(), b"multiprocessing.spawn"),
        (12, 20, os.getuid(), b"multiprocessing.spawn"),
        (13, 10, os.getuid() + 1, b"multiprocessing.spawn"),
        (14, 10, os.getuid(), b"unrelated_program"),
    ]:
        p = tmp_path / str(pid)
        p.mkdir()
        (p / "status").write_text(f"PPid:\t{parent}\nUid:\t{uid} {uid} {uid} {uid}\n")
        (p / "cmdline").write_bytes(command)
    assert reap_finished_workers(10, proc_root=tmp_path) == [11]
    assert killed == [11]


def test_restore_is_exact_and_preserves_orphan_evidence(tmp_path):
    from collector.campaigns.run_shard import replace_snapshot

    data = tmp_path / "local/data"
    data.mkdir(parents=True)
    for name in ("stale.parquet", ".collection_meta.transaction.json", "old-checkpoint.json"):
        (data / name).write_text("interrupted attempt")
    snapshot = tmp_path / "shared/snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "canonical.json").write_text("canonical state")
    quarantine = replace_snapshot(data, snapshot)
    assert [p.name for p in data.iterdir()] == ["canonical.json"]
    assert (data / "canonical.json").read_text() == "canonical state"
    from pathlib import Path

    assert (Path(quarantine) / "stale.parquet").read_text() == "interrupted attempt"


def test_fresh_attempt_does_not_reuse_unpublished_local_state(tmp_path):
    from collector.campaigns.run_shard import replace_snapshot

    data = tmp_path / "data"
    data.mkdir()
    (data / "checkpoint.json").write_text("not canonical")
    assert replace_snapshot(data) is not None
    assert list(data.iterdir()) == []


def test_restore_copy_failure_keeps_previous_data(tmp_path, monkeypatch):
    from collector.campaigns import run_shard

    data, snapshot = tmp_path / "data", tmp_path / "snapshot"
    data.mkdir()
    snapshot.mkdir()
    (data / "evidence").write_text("keep")

    def fail_copy(*args, **kwargs):
        raise OSError("copy interrupted")

    monkeypatch.setattr(run_shard.shutil, "copytree", fail_copy)
    with pytest.raises(OSError, match="copy interrupted"):
        run_shard.replace_snapshot(data, snapshot)
    assert (data / "evidence").read_text() == "keep"
    assert not list(tmp_path.glob(".data-restore-*"))


def source_fixture(tmp_path):
    import hashlib
    import subprocess

    source = tmp_path / "source"
    source.mkdir()
    declaration = source / "runtime.yaml"
    declaration.write_text("schema_version: 2\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    return {
        "source_commit": commit,
        "runtime_manifest": {"path": "runtime.yaml", "sha256": hashlib.sha256(declaration.read_bytes()).hexdigest()},
    }


def test_attest_clean_committed_runtime(tmp_path):
    from collector.campaigns.run_shard import attest_source

    manifest = source_fixture(tmp_path)
    result = attest_source(tmp_path, manifest)
    assert result["source_commit"] == manifest["source_commit"]
    assert result["sha256"] == manifest["runtime_manifest"]["sha256"]


@pytest.mark.parametrize("failure", ["head", "dirty", "untracked", "hash", "missing_declaration", "escape"])
def test_attestation_rejects_unreproducible_inputs(tmp_path, failure):
    from collector.campaigns.run_shard import attest_source

    manifest = source_fixture(tmp_path)
    if failure == "head":
        manifest["source_commit"] = "0" * 40
    elif failure == "dirty":
        (tmp_path / "source/runtime.yaml").write_text("modified")
    elif failure == "untracked":
        (tmp_path / "source/hidden-patch.py").write_text("modified")
    elif failure == "hash":
        manifest["runtime_manifest"]["sha256"] = "0" * 64
    elif failure == "missing_declaration":
        del manifest["runtime_manifest"]
    elif failure == "escape":
        manifest["runtime_manifest"]["path"] = "../outside.yaml"
    with pytest.raises(ValueError):
        attest_source(tmp_path, manifest)


def test_collector_child_receives_only_attested_source(tmp_path, monkeypatch):
    from collector.campaigns.run_shard import collector_environment

    monkeypatch.setenv("PYTHONPATH", "/unrelated/source")
    monkeypatch.setenv("AISIM_COLLECTOR_RUNTIME_MANIFEST", "/stale/runtime")
    declaration = {"path": "/attested/runtime.yaml", "sha256": "a" * 64}
    env = collector_environment(tmp_path, declaration)
    assert env["AISIM_COLLECTOR_RUNTIME_MANIFEST"] == declaration["path"]
    assert env["AISIM_COLLECTOR_RUNTIME_MANIFEST_SHA256"] == declaration["sha256"]
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str((tmp_path / "source/python/aisimulate").resolve()),
    ]

    assert env["PYTHONNOUSERSITE"] == "1"
