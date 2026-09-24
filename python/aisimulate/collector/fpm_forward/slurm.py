# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the ordinary FPM runtime in an existing Slurm/Pyxis allocation.

Only execution transport differs from Kubernetes. Generator scripts, native
result validation, attempt identity and publication remain owned by the common
campaign. The caller owns the allocation; this runner owns its named steps.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aisimulate.fpm_contract import FPM_BENCHMARK_RESULT_GLOB


class SlurmCellRunner:
    def __init__(
        self,
        manifest: Path,
        cell_dir: Path,
        *,
        image: str,
        mounts: tuple[str, ...],
        total_gpus: int,
        backend: str = "vllm",
    ):
        from .runner import _expected_nodes

        self.cell_dir = cell_dir.resolve()
        if backend not in ("vllm", "sglang"):
            raise ValueError("unsupported native FPM backend")
        self.backend = backend
        self.node_count = _expected_nodes(manifest)
        if total_gpus % self.node_count:
            raise ValueError("FPM GPUs must divide evenly across Slurm nodes")
        self.gpus_per_node = total_gpus // self.node_count
        self.image = image
        if not image or any(char in image for char in ("\n", "\r")):
            raise ValueError("Slurm FPM requires an explicit container image")
        self.mounts = mounts
        self.job_id = os.environ.get("SLURM_JOB_ID", "")
        if not re.fullmatch(r"[0-9]+", self.job_id):
            raise ValueError("Slurm FPM must run inside an existing sbatch/salloc allocation")
        self.step_name = f"fpm-{hashlib.sha256(str(self.cell_dir).encode()).hexdigest()[:20]}"
        # Keep ownership outside the replaceable cell payload so a fresh
        # invocation can tear down an abandoned allocation's named steps.
        self.owner_path = self.cell_dir.parent / ".slurm-owners" / f"{self.step_name}.json"
        self.hosts: list[str] = []

    def _command(self, args: list[str], *, timeout: float = 60, check: bool = True):
        from .runner import _run_command

        try:
            return _run_command(args, timeout=timeout, check=check)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            # Preparation and cleanup also invoke Slurm. Preserve their failure
            # streams even when a campaign formats only str(error), and keep
            # concurrent failures or retries from overwriting earlier evidence.
            logs = self.cell_dir / "logs" / "transport-failures" / uuid.uuid4().hex
            try:
                logs.mkdir(parents=True)
                for stream in ("stdout", "stderr"):
                    output = getattr(error, stream, None) or ""
                    if isinstance(output, bytes):
                        output = output.decode(errors="replace")
                    (logs / f"{stream}.log").write_text(output)
                (logs / "failure.json").write_text(
                    json.dumps(
                        {
                            "executable": Path(args[0]).name,
                            "exception": type(error).__name__,
                            "returncode": getattr(error, "returncode", None),
                            "timeout_seconds": timeout,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            except OSError as log_error:
                error.add_note(f"Could not preserve Slurm failure streams: {log_error}")
            raise

    def apply(self) -> None:
        for executable in ("srun", "scontrol", "squeue", "scancel"):
            if not shutil.which(executable):
                raise RuntimeError(f"Slurm FPM requires {executable}")
        self.owner_path.parent.mkdir(parents=True, exist_ok=True)
        self.owner_path.write_text(json.dumps({"job_id": self.job_id, "step_name": self.step_name}) + "\n")

    def wait_ready(self, expected_nodes: int, timeout_seconds: float = 900) -> list[str]:
        if expected_nodes != self.node_count:
            raise ValueError("Slurm expected node count disagrees with the generated manifest")
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Slurm readiness timeout must be finite and positive")
        deadline = time.monotonic() + timeout_seconds
        last_state = "unobserved"

        def remaining() -> float:
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise TimeoutError(f"Slurm allocation {self.job_id} not ready before deadline: {last_state}")
            return budget

        # This method qualifies the existing allocation, not a container or
        # model. Pyxis starts later in _exec; its startup skew has a separate
        # bounded rendezvous budget in the staged Collector runtime settings.
        while True:
            snapshot = self._command(["scontrol", "show", "job", self.job_id, "--oneliner"], timeout=remaining()).stdout
            fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", snapshot))
            if fields.get("JobId") != self.job_id:
                raise ValueError("Slurm readiness response does not identify the owned allocation")
            last_state = fields.get("JobState", "missing")
            if last_state == "RUNNING":
                nodelist = fields.get("NodeList")
                if not nodelist or nodelist in {"(null)", "None"}:
                    raise ValueError("running Slurm allocation has no node list")
                hosts = self._command(["scontrol", "show", "hostnames", nodelist], timeout=remaining()).stdout.split()
                remaining()
                if len(hosts) != self.node_count or len(set(hosts)) != self.node_count:
                    raise ValueError(f"FPM Slurm cell requires exactly {self.node_count} allocated nodes, got {hosts}")
                self.hosts = hosts
                return self.pods()
            if last_state not in {"PENDING", "CONFIGURING", "SUSPENDED"}:
                raise RuntimeError(f"Slurm allocation {self.job_id} cannot become ready from {last_state}")
            time.sleep(min(1.0, remaining()))

    def pods(self, *, include_terminating: bool = True) -> list[str]:
        del include_terminating
        return [f"node{rank:04d}" for rank in range(len(self.hosts))]

    def stage(self, pods: list[str], files: list[Path]) -> None:
        stage = self.cell_dir / "slurm-runtime"
        stage.mkdir(exist_ok=True)
        for path in files:
            shutil.copy2(path, stage / path.name)
        for unit in pods:
            (self.cell_dir / "raw" / unit).mkdir(parents=True, exist_ok=True)

    def _exec(self, unit: str, command: list[str], *, timeout: int):
        rank = self.pods().index(unit)
        mounts = [
            *self.mounts,
            f"{self.cell_dir / 'slurm-runtime'}:/tmp/fpm-bench",
            f"{self.cell_dir / 'raw' / unit}:/results",
        ]
        if any("\n" in mount or "," in mount for mount in mounts):
            raise ValueError("Slurm container mounts cannot contain newlines or commas")
        # Each srun is a one-node step in the caller's allocation. The engine
        # itself starts the node's TP/DP workers, exactly as in the Pod runtime.
        return self._command(
            [
                "srun",
                f"--jobid={self.job_id}",
                f"--job-name={self.step_name}",
                "--nodes=1",
                "--ntasks=1",
                "--ntasks-per-node=1",
                "--exclusive",
                "--exact",
                f"--nodelist={self.hosts[rank]}",
                f"--gpus-per-node={self.gpus_per_node}",
                f"--container-image={self.image}",
                f"--container-mounts={','.join(mounts)}",
                "--container-writable",
                "--container-workdir=/tmp/fpm-bench",
                "env",
                f"FPM_NODE_RANK={rank}",
                f"FPM_MASTER_ADDR={self.hosts[0]}",
                # Pyxis can start its command as a process-group leader. Keep
                # a parent alive so native launchers may create their own
                # session after their exec chain (os.setsid rejects leaders).
                # Positional arguments preserve the original argv literally.
                "bash",
                "-c",
                '"$@"; status=$?; exit "$status"',
                "fpm-slurm-command",
                *command,
            ],
            timeout=timeout,
        )

    def prepare_attempt(self, pods: list[str], *, cell_id: str, plan_sha256: str, attempt_id: str) -> None:
        from .native_artifact import COLLECTOR_PROVENANCE_FILENAME
        from .runner import REMOTE_WORKDIR, RUNTIME_ENV_FILENAME

        payload = json.dumps(
            {
                "schema_name": "aic_fpm_collector_provenance",
                "schema_version": 1,
                "cell_id": cell_id,
                "plan_sha256": plan_sha256,
                "attempt_id": attempt_id,
            }
        )
        script = (
            "import importlib.metadata,json,pathlib,sys; p=json.loads(sys.argv[1]); "
            "p['runtime']={'backend':sys.argv[3],'backend_version':importlib.metadata.version(sys.argv[3])}; "
            "pathlib.Path('/results',sys.argv[2]).write_text(json.dumps(p,sort_keys=True)+'\\n')"
        )
        for unit in pods:
            # Resolve the actual installed distribution through the same frozen
            # environment used by fpm_exec.sh. Slurm does not inherit the Pod's
            # extra_env, and inspecting the image before sourcing PYTHONPATH
            # would misidentify a task-private runtime as the image's baseline.
            self._exec(
                unit,
                [
                    "bash",
                    "-euo",
                    "pipefail",
                    "-c",
                    'source "$1"; shift; exec "$@"',
                    "fpm-slurm-prepare",
                    f"{REMOTE_WORKDIR}/{RUNTIME_ENV_FILENAME}",
                    "python3",
                    "-c",
                    script,
                    payload,
                    COLLECTOR_PROVENANCE_FILENAME,
                    self.backend,
                ],
                timeout=300,
            )

    def execute(self, pods: list[str], timeout_seconds: int = 14400) -> None:
        from .runner import CommandScope, _cancel_preserving_interrupt

        logs = self.cell_dir / "logs"
        logs.mkdir(exist_ok=True)

        def run(unit: str) -> None:
            try:
                result = self._exec(unit, ["bash", "/tmp/fpm-bench/fpm_exec.sh"], timeout=timeout_seconds)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                for stream in ("stdout", "stderr"):
                    output = getattr(error, stream, None) or ""
                    if isinstance(output, bytes):
                        output = output.decode(errors="replace")
                    (logs / f"{unit}.{stream}.log").write_text(output)
                raise
            (logs / f"{unit}.stdout.log").write_text(result.stdout)
            (logs / f"{unit}.stderr.log").write_text(result.stderr)

        scope = CommandScope()
        pool = ThreadPoolExecutor(max_workers=len(pods))
        try:
            futures = [pool.submit(scope.run, run, pod) for pod in pods]
            for future in futures:
                future.result()
        except BaseException as error:
            # Worker threads do not receive the main thread's interrupt.
            # Stop their srun children before joining so the campaign can
            # promptly salvage artifacts and clean up its owned steps.
            _cancel_preserving_interrupt(scope, error)
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def _remote_result_manifest(self, unit: str) -> dict:
        from .runner import _file_manifest

        return _file_manifest(self.cell_dir / "raw" / unit)

    def collect(self, pods: list[str], *, require_benchmark: bool = True) -> None:
        if require_benchmark and not any(
            list((self.cell_dir / "raw" / unit).glob(FPM_BENCHMARK_RESULT_GLOB)) for unit in pods
        ):
            raise RuntimeError("Slurm FPM result set is missing native benchmark artifacts")

    def cleanup(self) -> None:
        # Never cancel the allocation or unrelated steps. This also handles
        # resume after a collector process died while its named srun survived.
        jobs = {self.job_id}
        if self.owner_path.exists():
            owner = json.loads(self.owner_path.read_text())
            if owner.get("step_name") != self.step_name or not re.fullmatch(r"[0-9]+", owner.get("job_id", "")):
                raise ValueError("Slurm FPM ownership receipt does not match this campaign cell")
            jobs.add(owner["job_id"])

        def owned_steps() -> list[str]:
            # Listing the user's steps works even when an old allocation has
            # expired, unlike squeue --jobs=<expired-id> on some Slurm versions.
            result = self._command(["squeue", "--steps", "--me", "--noheader", "--format=%i|%j"])
            found = []
            for line in result.stdout.splitlines():
                step_id, _, name = line.strip().partition("|")
                if name == self.step_name and any(re.fullmatch(re.escape(job) + r"\.[0-9]+", step_id) for job in jobs):
                    found.append(step_id)
            return found

        for step_id in owned_steps():
            self._command(["scancel", step_id])
        deadline = time.monotonic() + 60
        while remaining := owned_steps():
            if time.monotonic() >= deadline:
                raise RuntimeError(f"owned FPM Slurm steps remain after cleanup: {remaining}")
            time.sleep(1)
