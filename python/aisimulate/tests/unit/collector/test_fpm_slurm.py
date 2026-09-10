# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from collector.fpm_forward.slurm import SlurmCellRunner

pytestmark = pytest.mark.unit


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "test-node")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        json.dumps(
            {
                "kind": "LeaderWorkerSet",
                "metadata": {"name": "cell"},
                "spec": {"replicas": 1, "leaderWorkerTemplate": {"size": 1}},
            }
        )
    )
    return SlurmCellRunner(manifest, tmp_path, image="image@sha256:abc", mounts=("/cache:/cache",), total_gpus=4)


def test_slurm_requires_scheduler_allocation(runner, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(ValueError, match="existing sbatch/salloc"):
        SlurmCellRunner(runner.cell_dir / "manifest.yaml", runner.cell_dir, image=runner.image, mounts=(), total_gpus=4)


def test_slurm_stage_and_argv_keep_shared_result_unit_identity(runner, monkeypatch):
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        return SimpleNamespace(stdout="test-node\n", stderr="")

    monkeypatch.setattr(runner, "_command", command)
    units = runner.wait_ready(1)
    source = runner.cell_dir / "fpm_exec.sh"
    source.write_text("exit 0\n")
    runner.stage(units, [source])
    runner._exec(units[0], ["bash", "/tmp/fpm-bench/fpm_exec.sh"], timeout=10)
    assert units == ["node0000"]
    assert (runner.cell_dir / "slurm-runtime" / source.name).read_text() == source.read_text()
    argv = commands[-1]
    assert "--jobid=1234" in argv and "--gpus-per-node=4" in argv
    assert "FPM_NODE_RANK=0" in argv and "FPM_MASTER_ADDR=test-node" in argv
    assert f"{runner.cell_dir}/raw/node0000:/results" in next(a for a in argv if a.startswith("--container-mounts="))


def test_slurm_cleanup_cancels_only_receipted_steps_and_verifies_exit(runner, monkeypatch):
    runner.owner_path.parent.mkdir(parents=True, exist_ok=True)
    runner.owner_path.write_text(json.dumps({"job_id": "1233", "step_name": runner.step_name}))
    commands = []
    snapshots = iter(
        [f"1234.2|{runner.step_name}\n1233.1|{runner.step_name}\n1234.4|other\n1234.batch|{runner.step_name}\n", ""]
    )

    def command(args, **kwargs):
        commands.append(args)
        return SimpleNamespace(stdout=next(snapshots) if args[0] == "squeue" else "", stderr="")

    monkeypatch.setattr(runner, "_command", command)
    runner.cleanup()
    assert [args for args in commands if args[0] == "scancel"] == [["scancel", "1234.2"], ["scancel", "1233.1"]]
    assert len([args for args in commands if args[0] == "squeue"]) == 2


def test_slurm_cleanup_reports_leaked_steps(runner, monkeypatch):
    monkeypatch.setattr(
        runner, "_command", lambda *a, **k: SimpleNamespace(stdout=f"1234.2|{runner.step_name}", stderr="")
    )
    clock = iter([0, 61])
    monkeypatch.setattr("collector.fpm_forward.slurm.time.monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="remain after cleanup"):
        runner.cleanup()


def test_slurm_refuses_allocation_geometry_mismatch(runner, monkeypatch):
    monkeypatch.setattr(runner, "_command", lambda *a, **k: SimpleNamespace(stdout="node-a node-b"))
    with pytest.raises(ValueError, match="exactly 1 allocated nodes"):
        runner.wait_ready(1)
