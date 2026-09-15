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
