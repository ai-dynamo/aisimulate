# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU campaign coverage; the fake cluster emits explicitly synthetic timings."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from collector.fpm_forward import cli, database, runner, runtime_memory
from collector.fpm_forward.config import FPMCollectionOptions
from collector.fpm_forward.runtime import fpm_memory_observer as observer

from .test_fpm_profile_collection import _argv, _plan, _profile, no_models_or_timing_data  # noqa: F401
from .test_fpm_runner import _native_payload
from .test_fpm_runtime_memory import _initialized

pytestmark = pytest.mark.unit

_GENERATOR_INPUTS = {
    "K8sConfig": {
        "extra_env": [
            {"name": "FPM_READINESS_TIMEOUT_SECONDS", "value": "480"},
            {"name": "PYTHONPATH", "value": "/site-plugins"},
        ]
    }
}


def _slurm_plan(tmp_path, *, points_file=None):
    profile = _profile()
    profile.update(model="example/slurm-fpm-model", architecture="UnregisteredMoeForCausalLM", num_experts=8)
    profile["deployments"] = [profile["deployments"][1]]
    deployment = profile["deployments"][0]
    deployment.update(
        system="gb200",
        backend_version="0.27.0",
        kv_cache_dtype="bfloat16",
        fmha_quant_mode="bfloat16",
        gemm_quant_mode="bfloat16",
        moe_quant_mode="bfloat16",
    )
    for key in ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes"):
        deployment["resources"].pop(key)
    config = {
        "architectures": [profile["architecture"]],
        "hidden_size": 128,
        "intermediate_size": 256,
        "moe_intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "num_hidden_layers": 5,
        "max_position_embeddings": 8192,
        "vocab_size": 1024,
        "num_experts": 8,
        "torch_dtype": "bfloat16",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    args = cli._parser().parse_args(
        [
            *_argv(profile),
            "--fpm-executor",
            "slurm",
            "--fpm-slurm-container-image",
            "image@sha256:synthetic",
            "--fpm-slurm-container-mount",
            "/cache:/cache",
            "--fpm-prefill-cudagraph-policy",
            "runtime",
            "--fpm-gpu-memory-utilization",
            "0.9",
            "--fpm-max-model-len",
            "4096",
            "--fpm-max-num-batched-tokens",
            "1024",
            "--fpm-max-num-seqs",
            "64",
            *(["--fpm-benchmark-points-file", str(points_file)] if points_file is not None else []),
        ]
    )
    options = replace(FPMCollectionOptions.from_args(args), parallel_presets=("tep",), kv_cache_dtypes=("bfloat16",))
    return _plan(
        profile,
        options=options,
        system="gb200",
        model_config_path=str(config_path),
        has_model_cases=False,
        selected_ops={"attention_context", "attention_generation"},
        generator_overrides=_GENERATOR_INPUTS,
    )


class _SyntheticSlurm:
    """Emulate the external Slurm/Pyxis boundary, retaining all real campaign code."""

    def __init__(self, plan, monkeypatch):
        self.plan = plan
        self.commands = []
        self.executed = []
        self.steps = {"1234.99": "unrelated-job"}
        self.sequence = 0
        self.version = "0.27.0"
        self.missing_worker = False
        self.interrupt_phase = None
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(2)
        self.patch = monkeypatch
        monkeypatch.setenv("SLURM_JOB_ID", "1234")
        monkeypatch.setattr("collector.fpm_forward.slurm.shutil.which", lambda name: f"/fake/{name}")
        monkeypatch.setattr(runner, "_run_command", self.command)
        # Only the artifact-settle delay is elided; no runtime is simulated by sleeping.
        monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
        real_version = observer.importlib.metadata.version
        monkeypatch.setattr(
            observer.importlib.metadata, "version", lambda name: self.version if name == "vllm" else real_version(name)
        )
        monkeypatch.setattr(observer, "_offloader_type", lambda: "vllm.model_executor.offloader.base.NoopOffloader")

    def command(self, args, **_kwargs):
        with self.lock:
            self.commands.append(args)
            if args[0] == "scontrol":
                text = "JobId=1234 JobState=RUNNING NodeList=node-a,node-b" if "job" in args else "node-a node-b"
                return subprocess.CompletedProcess(args, 0, stdout=text, stderr="")
            if args[0] == "squeue":
                return subprocess.CompletedProcess(
                    args, 0, stdout="\n".join(f"{step}|{name}" for step, name in self.steps.items()), stderr=""
                )
            if args[0] == "scancel":
                del self.steps[args[1]]
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            assert args[0] == "srun", args
            assert "--jobid=1234" in args and "--gpus-per-node=4" in args
            assert "--container-image=image@sha256:synthetic" in args
            mounts = next(value.split("=", 1)[1] for value in args if value.startswith("--container-mounts="))
            bindings = dict(value.rsplit(":", 1)[::-1] for value in mounts.split(","))
            assert bindings["/cache"] == "/cache"
            raw = Path(bindings["/results"])
            stage = Path(bindings["/tmp/fpm-bench"])
            rank = int(next(value.split("=", 1)[1] for value in args if value.startswith("FPM_NODE_RANK=")))
            assert raw.name == f"node{rank:04d}"
            assert "FPM_MASTER_ADDR=node-a" in args
            command = args[args.index("env") + 3 :]
            if command[:2] == ["python3", "-c"]:
                # Interpret the actual preparation program with the Pyxis /results mount mapped locally.
                with self.patch.context() as patch:
                    patch.setattr(sys, "argv", ["-c", *command[3:]])
                    try:
                        exec(command[2].replace("/results", str(raw)), {})
                    except RuntimeError as error:
                        raise subprocess.CalledProcessError(1, args, output="preparation", stderr=str(error)) from error
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            assert command == ["bash", "/tmp/fpm-bench/fpm_exec.sh"]
            assert {"fpm_memory_observer.py", "fpm_memory_worker.py", "fpm_memory_scheduler.py"} <= {
                item.name for item in stage.iterdir()
            }
            startup = stage / "collector-runtime-env.sh"
            assert startup.is_file()
            startup_probe = subprocess.run(
                [
                    "bash",
                    "-c",
                    'set -Eeuo pipefail; source "$1"; source "$2"; '
                    'printf "%s|%s|%s|%s|%s" "$FPM_NODE_RANK" "$FPM_MASTER_ADDR" '
                    '"$FPM_NODE_COUNT" "$FPM_READINESS_TIMEOUT_SECONDS" "$PYTHONPATH"',
                    "probe",
                    str(startup),
                    str(stage / "fpm_env.sh"),
                ],
                env={**os.environ, "FPM_NODE_RANK": str(rank), "FPM_MASTER_ADDR": "node-a"},
                capture_output=True,
                text=True,
                check=True,
            )
            assert startup_probe.stdout == f"{rank}|node-a|2|480|/site-plugins"
            launch = (stage / "run.sh").read_text()
            assert "--worker-cls fpm_memory_worker.FpmResourceWorker" in launch
            assert "--scheduler-cls fpm_memory_scheduler.FpmResourceInstrumentedScheduler" in launch
            assert launch.count("--no-async-scheduling") == 1
            assert "--enforce-eager" not in launch and "DYN_FPM_DSV41_REAL_KV" not in startup.read_text()
            assert "--compilation-config" not in launch and "fpm_points.json" not in {p.name for p in stage.iterdir()}
            runtime = (stage / "fpm_exec.sh").read_text()
            assert "collector-runtime-env.sh" in runtime and 'export PYTHONPATH="${workdir}' in runtime
            cell = next(cell for cell in self.plan.cells if cell.cell_id == raw.parent.parent.name)
            self.executed.append((cell.workload_kind, rank))
            self.sequence += 1
            step = f"1234.{self.sequence}"
            self.steps[step] = next(value.split("=", 1)[1] for value in args if value.startswith("--job-name="))
            with self.patch.context() as patch:
                patch.setattr(observer, "RESULTS_DIR", raw)
                for tp in range(rank * 4, (rank + 1) * 4):
                    if self.missing_worker and tp == 7:
                        continue
                    worker, scheduler, cache = _initialized()
                    worker.vllm_config.parallel_config.tensor_parallel_size = 8
                    worker.vllm_config.parallel_config.data_parallel_size = 1
                    worker.vllm_config.quant_config = None
                    worker.vllm_config.model_config.quantization = None
                    worker.vllm_config.model_config.model = self.plan.model_path
                    worker.vllm_config.model_config.revision = self.plan.fpm_profile.model_revision
                    observer.observe("worker", worker, dp_rank=0, tp_rank=tp, pp_rank=0)
                    if tp == 0:
                        observer.observe("scheduler", scheduler, dp_rank=0, cache_config=cache)
                if rank == 0:
                    (raw / "benchmark.json").write_text(
                        json.dumps(_native_payload(phase=cell.workload_kind, rank=0, dp=1))
                    )
            interrupt = cell.workload_kind == self.interrupt_phase
            if not interrupt:
                del self.steps[step]
        if interrupt:
            # Both nodes have emitted their complete attempt before the transport disconnects.
            self.barrier.wait(timeout=5)
            if rank == 1:
                raise KeyboardInterrupt
        return subprocess.CompletedProcess(args, 0, stdout="synthetic container completed", stderr="")


def _run(plan, tmp_path, *, resume=False):
    return runner.run_collection(
        plan,
        generator_overrides=_GENERATOR_INPUTS,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        artifact_root=str(tmp_path / "artifacts"),
        database_root=str(tmp_path / "data"),
        resume=resume,
        retry_failed=False,
    )


def _checkpoint(tmp_path):
    return json.loads((tmp_path / "checkpoints" / "fpm_forward.json").read_text())


def _resolve(plan, cell, tmp_path):
    entry = _checkpoint(tmp_path)["cells"][cell.cell_id]
    return runtime_memory.resolve_runtime_resources(
        cell,
        Path(entry["artifact_dir"]) / "raw",
        expected_plan_sha256=plan.sha256,
        expected_attempt_id=entry["attempt_id"],
        expected_backend_version="0.27.0",
        expected_context_length=4096,
        expected_max_num_tokens=1024,
        expected_max_batch_size=64,
        expected_gpu_memory_utilization=0.9,
        expected_model_revision=plan.fpm_profile.model_revision,
    )


@pytest.mark.usefixtures("no_models_or_timing_data")
@pytest.mark.parametrize("missing_worker", [False, True])
def test_slurm_profile_campaign_preserves_timing_and_resolves_only_complete_memory(
    tmp_path, monkeypatch, missing_worker
):
    plan = _slurm_plan(tmp_path)
    cluster = _SyntheticSlurm(plan, monkeypatch)
    cluster.missing_worker = missing_worker
    assert _run(plan, tmp_path) == []
    checkpoint = _checkpoint(tmp_path)
    assert {entry["status"] for entry in checkpoint["cells"].values()} == {"passed"}
    publication = checkpoint["database"]
    assert publication["status"] == "passed"
    saved = runtime_memory.saved_plan_identity(plan.to_dict())
    database.validate_formal_database_commit(Path(publication["parquet"]), Path(publication["metadata"]), saved)
    assert sorted(cluster.executed) == [("decode", 0), ("decode", 1), ("prefill", 0), ("prefill", 1)]
    assert cluster.steps == {"1234.99": "unrelated-job"}
    for cell in plan.cells:
        if missing_worker:
            with pytest.raises(ValueError, match="worker rank evidence is incomplete"):
                _resolve(plan, cell, tmp_path)
        else:
            resolved = _resolve(plan, cell, tmp_path)
            assert resolved["runtime_memory"]["kv_cache_bytes"] == 98 * 128
            assert len(resolved["cache_groups"]) == 3
            evidence = json.loads(resolved["runtime_memory"]["provenance"])["artifacts"]
            assert {Path(item["path"]).parts[0] for item in evidence} == {"node0000", "node0001"}
    before = runner._file_manifest(tmp_path / "artifacts")
    commands = len(cluster.commands)
    assert _run(plan, tmp_path, resume=True) == []
    assert len(cluster.commands) == commands
    # Completed resume may refresh invocation metadata; it must retain raw runtime evidence.
    after = runner._file_manifest(tmp_path / "artifacts")
    assert {key: value for key, value in before.items() if "/raw/" in key} == {
        key: value for key, value in after.items() if "/raw/" in key
    }


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_slurm_profile_runtime_mismatch_preserves_observed_pin_without_benchmarking(tmp_path, monkeypatch):
    plan = _slurm_plan(tmp_path)
    cluster = _SyntheticSlurm(plan, monkeypatch)
    cluster.version = "0.28.0"
    errors = _run(plan, tmp_path)
    assert len(errors) == 2 and all(error["classification"] == "campaign_cell_failed" for error in errors)
    assert cluster.executed == []
    checkpoint = _checkpoint(tmp_path)
    assert "database" not in checkpoint
    for entry in checkpoint["cells"].values():
        cell_dir = Path(entry["artifact_dir"])
        provenance = json.loads((cell_dir / "raw/node0000/collector-provenance.json").read_text())
        assert provenance["runtime"]["backend_version"] == "0.28.0"
        failures = list((cell_dir / "logs/transport-failures").glob("*/stderr.log"))
        assert len(failures) == 1 and "FPM profile runtime mismatch" in failures[0].read_text()
    assert cluster.steps == {"1234.99": "unrelated-job"}


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_slurm_profile_interrupt_salvages_memory_and_resumes_exact_completed_attempt(tmp_path, monkeypatch):
    plan = _slurm_plan(tmp_path)
    cluster = _SyntheticSlurm(plan, monkeypatch)
    first = plan.cells[0]
    cluster.interrupt_phase = first.workload_kind
    with pytest.raises(KeyboardInterrupt):
        _run(plan, tmp_path)
    original = _checkpoint(tmp_path)["cells"][first.cell_id]
    assert original["status"] == "interrupted"
    before = runner._file_manifest(Path(original["artifact_dir"]) / "raw")
    assert _resolve(plan, first, tmp_path)["runtime_memory"]["kv_cache_bytes"] == 98 * 128
    assert cluster.steps == {"1234.99": "unrelated-job"}
    cancelled = [args[1] for args in cluster.commands if args[0] == "scancel"]
    assert len(cancelled) == 2 and "1234.99" not in cancelled and "1234" not in cancelled
    cluster.interrupt_phase = None
    assert _run(plan, tmp_path, resume=True) == []
    checkpoint = _checkpoint(tmp_path)
    recovered = checkpoint["cells"][first.cell_id]
    assert recovered["status"] == "passed" and recovered["attempt_id"] == original["attempt_id"]
    assert recovered["artifact_recovery"]["original_status"] == "interrupted"
    assert runner._file_manifest(Path(recovered["artifact_dir"]) / "raw") == before
    assert sorted(cluster.executed) == [("decode", 0), ("decode", 1), ("prefill", 0), ("prefill", 1)]
    assert checkpoint["database"]["status"] == "passed"


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_saved_slurm_plan_preserves_explicit_point_coverage_at_formal_consumer(tmp_path, monkeypatch):
    plan = _slurm_plan(tmp_path)
    _SyntheticSlurm(plan, monkeypatch)
    assert _run(plan, tmp_path) == []
    publication = _checkpoint(tmp_path)["database"]
    points = {
        "schema_version": 1,
        "prefill": [{"batch_size": 4, "total_prefill_tokens": 257, "total_kv_read_tokens": 512}],
        "decode": [{"batch_size": 4, "total_kv_read_tokens": 512}],
    }
    path = tmp_path / "points.json"
    path.write_text(json.dumps(points))
    explicit = _slurm_plan(tmp_path, points_file=path)
    saved = runtime_memory.saved_plan_identity(explicit.to_dict())
    assert json.loads(saved.options.benchmark_points_json) == points
    # Existing first-publisher data may have another plan identity; its actual
    # coordinates must still satisfy the new saved manifest before reuse.
    with pytest.raises(ValueError, match="does not cover requested coordinates"):
        database.validate_formal_database_commit(
            Path(publication["parquet"]),
            Path(publication["metadata"]),
            saved,
            reused_cell_ids=[cell.cell_id for cell in saved.cells],
        )
