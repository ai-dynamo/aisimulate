# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``gpu_time``: GPU cost of a measurement run, its check behavior and Slurm capture.

``gpu_time`` counts allocated GPU time and excludes queue wait. Missing is not
zero: an absent record or ``authority="unknown"`` means "not recorded" and
carries only nulls, while a measured or declared record with ``0`` seconds is a
real (if unusual) value. ``check_gpu_time`` warns on missing data for archival
evidence and errors for qualification.

Slurm capture parses text the caller already holds (``sacct --parsable2``).
Kubernetes capture reads Job and Pod documents through a caller-supplied
``KubeReader``; this module never contacts a scheduler or cluster itself.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

GPU_TIME_SCHEMA = "aisim-gpu-time/v1"
Authority = Literal["measured", "declared", "unknown"]
CheckLevel = Literal["archival", "qualification"]

# Relative tolerance when gpu_seconds is given alongside wall_seconds * gpu_count.
_CONSISTENCY_TOL = 1e-6


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Attempts(_Record):
    """GPU time spent on attempts that did not produce the result."""

    count: int = Field(ge=0)
    wall_seconds: float = Field(ge=0)
    gpu_seconds: float = Field(ge=0)


class Phases(_Record):
    """Wall-second split; every phase is independently nullable (null = not recorded)."""

    setup_load: float | None = Field(default=None, ge=0)
    warmup: float | None = Field(default=None, ge=0)
    measure: float | None = Field(default=None, ge=0)
    teardown: float | None = Field(default=None, ge=0)


class GpuTime(_Record):
    schema_id: Literal["aisim-gpu-time/v1"] = Field(default=GPU_TIME_SCHEMA, alias="schema")
    authority: Authority
    gpu_type: str | None = Field(default=None, min_length=1)
    gpu_count: int | None = Field(default=None, ge=0)
    wall_seconds: float | None = Field(default=None, ge=0)
    gpu_seconds: float | None = Field(default=None, ge=0)
    node_hours: float | None = Field(default=None, ge=0)
    phases: Phases | None = None
    failed: Attempts | None = None
    retried: Attempts | None = None
    # Provenance of a measured value, e.g. {"source": "slurm", "job_ids": "123,124"}.
    source: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)

    @model_validator(mode="after")
    def _authority_contract(self) -> Self:
        values = (self.gpu_count, self.wall_seconds, self.gpu_seconds, self.node_hours)
        if self.authority == "unknown":
            if any(v is not None for v in values) or self.phases or self.failed or self.retried:
                raise ValueError("authority 'unknown' must not carry values; unknown is null, never zero")
            return self
        if self.gpu_count is None or self.wall_seconds is None:
            raise ValueError(f"authority {self.authority!r} requires gpu_count and wall_seconds")
        expected = self.wall_seconds * self.gpu_count
        if self.gpu_seconds is None:
            object.__setattr__(self, "gpu_seconds", expected)
        elif abs(self.gpu_seconds - expected) > _CONSISTENCY_TOL * max(1.0, expected):
            raise ValueError(f"gpu_seconds {self.gpu_seconds} != wall_seconds * gpu_count {expected}")
        return self

    @classmethod
    def unknown(cls) -> Self:
        return cls(authority="unknown")

    @property
    def is_recorded(self) -> bool:
        return self.authority != "unknown"


@dataclass(frozen=True)
class Finding:
    severity: Literal["error", "warn"]
    code: str
    message: str


def check_gpu_time(gpu_time: GpuTime | None, level: CheckLevel) -> list[Finding]:
    """Findings for one run or point. Missing (None or unknown) is distinct from zero."""
    if gpu_time is None or not gpu_time.is_recorded:
        severity = "error" if level == "qualification" else "warn"
        reason = "absent" if gpu_time is None else "authority is unknown"
        return [Finding(severity, "gpu_time.missing", f"gpu_time {reason}; required for qualification")]
    findings: list[Finding] = []
    if gpu_time.authority == "declared" and level == "qualification":
        findings.append(Finding("warn", "gpu_time.declared", "gpu_time is declared, not measured from a job record"))
    if gpu_time.gpu_type is None:
        findings.append(Finding("warn", "gpu_time.gpu_type_missing", "gpu_type not recorded"))
    return findings


def gpu_time_ok(findings: Iterable[Finding]) -> bool:
    return not any(f.severity == "error" for f in findings)


# --------------------------------------------------------------------------- Slurm

# sacct states whose allocated time was not a successful run.
_REQUEUED = {"REQUEUED"}
_FAILED = {
    "FAILED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "CANCELLED",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
    "SUSPENDED",
}
_TRES_GPU = re.compile(r"^gres/gpu(?::([^=,]+))?=(\d+)$")
_ELAPSED = re.compile(r"^(?:(\d+)-)?(\d+):(\d{2}):(\d{2})(?:\.\d+)?$")


class SlurmParseError(ValueError):
    pass


@dataclass(frozen=True)
class SlurmJob:
    job_id: str
    state: str  # first token, e.g. "CANCELLED by 123" -> "CANCELLED"
    elapsed_seconds: float
    nnodes: int
    gpu_count: int
    gpu_type: str | None


def _elapsed_seconds(raw_seconds: str | None, elapsed: str | None) -> float:
    if raw_seconds not in (None, ""):
        try:
            return float(raw_seconds)
        except ValueError as exc:
            raise SlurmParseError(f"bad ElapsedRaw {raw_seconds!r}") from exc
    match = _ELAPSED.match(elapsed or "")
    if not match:
        raise SlurmParseError(f"missing or unparseable Elapsed {elapsed!r}")
    days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return float(((days * 24 + hours) * 60 + minutes) * 60 + seconds)


def _gpus_from_tres(tres: str) -> tuple[int, str | None]:
    """Total GPUs and type from AllocTRES such as ``billing=8,gres/gpu:h100=8,node=1``."""
    typed: dict[str, int] = {}
    untyped = 0
    for item in tres.split(","):
        match = _TRES_GPU.match(item.strip())
        if not match:
            continue
        gpu_type, count = match.groups()
        if gpu_type:
            typed[gpu_type] = int(count)
        else:
            untyped = int(count)
    # sacct often lists both gres/gpu=N and gres/gpu:type=N for the same GPUs.
    total = sum(typed.values()) if typed else untyped
    types = set(typed)
    if len(types) > 1:
        return total, "+".join(sorted(types))
    return total, next(iter(types), None)


def parse_sacct(text: str, fields: Sequence[str] | None = None) -> list[SlurmJob]:
    """Parse ``sacct --parsable2`` output into allocation-level jobs.

    The header line is used when present; otherwise pass ``fields``. Required
    columns: JobID, State, AllocTRES, NNodes and ElapsedRaw (or Elapsed).
    Step rows (``123.batch``, ``123.0``) are skipped, because the allocation row
    already covers them. Jobs that have not ended are rejected: their cost is
    not final.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("|")
    if "JobID" in header:
        columns, rows = header, lines[1:]
    elif fields:
        columns, rows = list(fields), lines
    else:
        raise SlurmParseError("no header line; pass fields=")
    for need in ("JobID", "State", "AllocTRES", "NNodes"):
        if need not in columns:
            raise SlurmParseError(f"sacct output lacks required column {need}")
    if "ElapsedRaw" not in columns and "Elapsed" not in columns:
        raise SlurmParseError("sacct output needs ElapsedRaw or Elapsed")

    jobs: list[SlurmJob] = []
    for row in rows:
        cells = row.split("|")
        if len(cells) != len(columns):
            raise SlurmParseError(f"expected {len(columns)} columns, got {len(cells)}: {row!r}")
        rec = dict(zip(columns, cells, strict=True))
        job_id = rec["JobID"].strip()
        if "." in job_id:
            continue
        state = rec["State"].strip().split(" ")[0].upper()
        if state in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "RESIZING"}:
            raise SlurmParseError(f"job {job_id} is {state}; gpu_time is final only after the job ends")
        if not rec["AllocTRES"].strip():
            raise SlurmParseError(f"job {job_id} has empty AllocTRES; cannot tell GPU count from zero")
        gpus, gpu_type = _gpus_from_tres(rec["AllocTRES"])
        try:
            nnodes = int(rec["NNodes"])
        except ValueError as exc:
            raise SlurmParseError(f"job {job_id}: bad NNodes {rec['NNodes']!r}") from exc
        jobs.append(
            SlurmJob(
                job_id=job_id,
                state=state,
                elapsed_seconds=_elapsed_seconds(rec.get("ElapsedRaw"), rec.get("Elapsed")),
                nnodes=nnodes,
                gpu_count=gpus,
                gpu_type=gpu_type,
            )
        )
    return jobs


def gpu_time_from_slurm(jobs: Sequence[SlurmJob], *, gpu_type: str | None = None) -> GpuTime:
    """Aggregate job records into a ``measured`` ``gpu_time``.

    COMPLETED jobs are the result; REQUEUED jobs count as ``retried`` and other
    non-successful jobs as ``failed``, each kept out of the primary totals.
    Elapsed time starts at job start, so queue wait is excluded by construction.
    ``gpu_type`` overrides the type read from AllocTRES (Slurm often omits it).
    """
    if not jobs:
        raise SlurmParseError("no Slurm jobs to build gpu_time from")
    ok = [j for j in jobs if j.state == "COMPLETED"]
    retried = [j for j in jobs if j.state in _REQUEUED]
    failed = [j for j in jobs if j.state not in _REQUEUED and j.state != "COMPLETED"]
    unknown_states = {j.state for j in failed} - _FAILED
    if unknown_states:
        raise SlurmParseError(f"unrecognized Slurm states: {sorted(unknown_states)}")
    if not ok:
        raise SlurmParseError("no COMPLETED job; failed attempts alone do not make a result's gpu_time")
    counts = {j.gpu_count for j in ok}
    if len(counts) != 1:
        raise SlurmParseError(f"COMPLETED jobs disagree on GPU count: {sorted(counts)}")

    def bucket(group: list[SlurmJob]) -> Attempts | None:
        if not group:
            return None
        return Attempts(
            count=len(group),
            wall_seconds=sum(j.elapsed_seconds for j in group),
            gpu_seconds=sum(j.elapsed_seconds * j.gpu_count for j in group),
        )

    wall = sum(j.elapsed_seconds for j in ok)
    types = {j.gpu_type for j in ok if j.gpu_type}
    return GpuTime(
        authority="measured",
        gpu_type=gpu_type or (next(iter(types)) if len(types) == 1 else None),
        gpu_count=counts.pop(),
        wall_seconds=wall,
        gpu_seconds=sum(j.elapsed_seconds * j.gpu_count for j in ok),
        node_hours=sum(j.elapsed_seconds * j.nnodes for j in ok) / 3600.0,
        failed=bucket(failed),
        retried=bucket(retried),
        source={"source": "slurm", "job_ids": ",".join(j.job_id for j in jobs)},
    )


def gpu_time_from_sacct(text: str, *, gpu_type: str | None = None, fields: Sequence[str] | None = None) -> GpuTime:
    """``sacct --parsable2`` text to a measured ``GpuTime`` in one call."""
    return gpu_time_from_slurm(parse_sacct(text, fields), gpu_type=gpu_type)


# ----------------------------------------------------------------------- Kubernetes

GPU_RESOURCE = "nvidia.com/gpu"


class KubeParseError(ValueError):
    pass


class KubeReader(Protocol):
    """Read-only view of a cluster; implement with kubectl/client or a test fake.

    Both methods return Kubernetes API JSON documents (dicts).
    """

    def get_job(self, namespace: str, name: str) -> Mapping[str, Any]: ...

    def list_job_pods(self, namespace: str, job_name: str) -> Sequence[Mapping[str, Any]]: ...


def _ts(value: Any, what: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise KubeParseError(f"missing {what}")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise KubeParseError(f"bad {what} {value!r}") from exc


def _gpus_requested(spec: Mapping[str, Any], resource: str) -> int:
    """GPUs a pod spec requests (requests, else limits, per container)."""
    total = 0
    for container in spec.get("containers") or []:
        res = container.get("resources") or {}
        raw = (res.get("requests") or {}).get(resource, (res.get("limits") or {}).get(resource))
        if raw is not None:
            try:
                total += int(raw)
            except (TypeError, ValueError) as exc:
                raise KubeParseError(f"bad {resource} quantity {raw!r}") from exc
    return total


def _pod_window(pod: Mapping[str, Any]) -> tuple[datetime, datetime]:
    """Start and end of container execution; excludes scheduling/queue wait."""
    status = pod.get("status") or {}
    starts, ends = [], []
    for cs in status.get("containerStatuses") or []:
        term = (cs.get("state") or {}).get("terminated")
        if term and term.get("startedAt") and term.get("finishedAt"):
            starts.append(_ts(term["startedAt"], "startedAt"))
            ends.append(_ts(term["finishedAt"], "finishedAt"))
    if not starts:
        raise KubeParseError(f"pod {(pod.get('metadata') or {}).get('name', '?')} has no terminated container times")
    return min(starts), max(ends)


def gpu_time_from_kubernetes(
    job: Mapping[str, Any],
    pods: Sequence[Mapping[str, Any]],
    *,
    gpu_type: str | None = None,
    gpu_resource: str = GPU_RESOURCE,
) -> GpuTime:
    """Build a ``measured`` ``gpu_time`` from a finished Job and its Pods.

    Exactly one Succeeded pod is the result; Failed pods are ``failed`` attempts.
    Time runs from container start to finish (``startedAt``/``finishedAt``), so
    queue and scheduling wait are excluded; when a pod lacks container
    timestamps the Job's ``startTime``/``completionTime`` are used for the
    succeeded pod. GPU count comes from the pod spec's ``gpu_resource``
    request, else the Job's pod template. Unfinished work is rejected.
    """
    jmeta = job.get("metadata") or {}
    jname = jmeta.get("name", "?")
    jstatus = job.get("status") or {}
    if not jstatus.get("succeeded"):
        raise KubeParseError(f"job {jname} has no succeeded pod; gpu_time is final only after success")
    template_spec = ((job.get("spec") or {}).get("template") or {}).get("spec") or {}
    ok: list[tuple[Mapping[str, Any], float, int]] = []
    failed: list[tuple[float, int]] = []
    for pod in pods:
        phase = (pod.get("status") or {}).get("phase")
        if phase in ("Pending", "Running", "Unknown", None):
            raise KubeParseError(f"pod {(pod.get('metadata') or {}).get('name', '?')} is {phase}; not final")
        gpus = _gpus_requested(pod.get("spec") or {}, gpu_resource) or _gpus_requested(template_spec, gpu_resource)
        try:
            start, end = _pod_window(pod)
        except KubeParseError:
            if phase != "Succeeded":
                # A failed pod that never started consumed no allocated GPU time.
                continue
            start, end = (
                _ts(jstatus.get("startTime"), "job startTime"),
                _ts(jstatus.get("completionTime"), "job completionTime"),
            )
        seconds = (end - start).total_seconds()
        if seconds < 0:
            raise KubeParseError("pod finished before it started")
        if phase == "Succeeded":
            ok.append((pod, seconds, gpus))
        else:
            failed.append((seconds, gpus))
    if len(ok) != 1:
        raise KubeParseError(f"expected exactly one Succeeded pod for job {jname}, found {len(ok)}")
    pod, wall, gpus = ok[0]
    if gpus <= 0:
        raise KubeParseError(f"job {jname} requests no {gpu_resource}; cannot tell GPU count from zero")
    return GpuTime(
        authority="measured",
        gpu_type=gpu_type,
        gpu_count=gpus,
        wall_seconds=wall,
        gpu_seconds=wall * gpus,
        node_hours=wall / 3600.0,
        failed=Attempts(
            count=len(failed), wall_seconds=sum(s for s, _ in failed), gpu_seconds=sum(s * g for s, g in failed)
        )
        if failed
        else None,
        source={
            "source": "kubernetes",
            "job": f"{jmeta.get('namespace', '')}/{jname}",
            "pod": (pod.get("metadata") or {}).get("name", ""),
        },
    )


def capture_gpu_time_kubernetes(
    reader: KubeReader, namespace: str, job_name: str, *, gpu_type: str | None = None, gpu_resource: str = GPU_RESOURCE
) -> GpuTime:
    """Read a Job and its Pods through ``reader`` and build its ``gpu_time``."""
    return gpu_time_from_kubernetes(
        reader.get_job(namespace, job_name),
        reader.list_job_pods(namespace, job_name),
        gpu_type=gpu_type,
        gpu_resource=gpu_resource,
    )
