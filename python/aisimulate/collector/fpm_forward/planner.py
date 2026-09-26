# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an immutable cell matrix for Dynamo-native FPM self-benchmarks."""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import importlib.metadata
import io
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any

from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, LEGACY_EXECUTION_IDENTITY, execution_identity
from aisimulate_core.sdk.fpm_profile import FpmDeploymentProfile, FpmModelProfile, load_fpm_profile

from .capabilities import ModelCapabilityProfile, ResolvedDTypeProfile, resolve_model_capability
from .config import (
    FPM_MAX_PREFILL_ISL,
    PARALLEL_AXES,
    VLLM_AUTO_FIT_MAX_MODEL_LEN,
    FPMCollectionOptions,
    with_kv_warmup_defaults,
)
from .memory_admission import TopologyMemoryDecision, filter_memory_infeasible_topologies
from .runtime.fpm_memory_observer import SUPPORTED_VERSION as MEMORY_OBSERVER_VERSION
from .topology import enumerate_fpm_topologies, topology_strategy
from .types import ParallelTopology

logger = logging.getLogger(__name__)


def _runtime_memory_policy(
    profile: FpmModelProfile | None, backend_version: str, runtime_observation: dict[str, Any] | None = None
) -> dict[str, object] | None:
    if runtime_observation is not None:
        return {
            "source": "runtime_instrumentation",
            "observation": "enabled",
            "selected_vllm_version": backend_version,
            "bundle_sha256": runtime_observation["bundle_sha256"],
            "async_scheduling": False,
        }
    if profile is None or not any(
        deployment.resources.memory_source == "pending" for deployment in profile.deployments
    ):
        return None
    available = backend_version == MEMORY_OBSERVER_VERSION
    return {
        "source": "vllm_initialization",
        "supported_vllm_version": MEMORY_OBSERVER_VERSION,
        "selected_vllm_version": backend_version,
        "observation": "enabled" if available else "unavailable_for_runtime",
        "evidence_schema_version": 1,
        "async_scheduling": False if available else None,
    }


_INSTALLED_DISTRIBUTION = "aisimulate"
_INSTALLED_PAYLOAD_ROOTS = frozenset(("aisimulate_core", "aisimulate", "collector"))
_INSTALLED_PLANNER_PATH = PurePosixPath("collector/fpm_forward/planner.py")
_REQUIRED_INSTALLED_FPM_PAYLOAD = frozenset(
    (
        _INSTALLED_PLANNER_PATH,
        PurePosixPath("collector/fpm_forward/runner.py"),
        PurePosixPath("collector/fpm_forward/runtime/fpm_exec.sh"),
        PurePosixPath("collector/fpm_forward/runtime/preflight.py"),
        PurePosixPath("collector/fpm_forward/runtime/fpm_memory_observer.py"),
        PurePosixPath("collector/fpm_forward/runtime/fpm_memory_worker.py"),
        PurePosixPath("collector/fpm_forward/runtime/fpm_memory_scheduler.py"),
    )
)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _distribution_root(distribution: importlib.metadata.Distribution) -> Path:
    return Path(os.fspath(distribution.locate_file(""))).resolve()


def _distribution_owns_loaded_planner(distribution: importlib.metadata.Distribution) -> bool:
    """Distinguish an unpacked wheel from editable/source metadata."""

    root = _distribution_root(distribution)
    located = Path(os.fspath(distribution.locate_file(_INSTALLED_PLANNER_PATH))).resolve()
    expected = (root / Path(*_INSTALLED_PLANNER_PATH.parts)).resolve()
    return located == expected == Path(__file__).resolve()


def _is_installer_console_script(path: PurePosixPath) -> bool:
    parts = path.parts
    parent_count = 0
    while parent_count < len(parts) and parts[parent_count] == "..":
        parent_count += 1
    target = parts[parent_count:]
    return parent_count > 0 and target in {
        ("bin", "aiconfigurator"),
        ("bin", "aisimulate"),
        ("Scripts", "aiconfigurator.exe"),
        ("Scripts", "aiconfigurator-script.py"),
        ("Scripts", "aisimulate.exe"),
        ("Scripts", "aisimulate-script.py"),
    }


def _is_installer_bytecode(path: PurePosixPath) -> bool:
    return path.suffix == ".pyc" and "__pycache__" in path.parts


def _decode_record_sha256(value: str, *, path: PurePosixPath) -> bytes:
    prefix = "sha256="
    encoded = value.removeprefix(prefix)
    if not value.startswith(prefix) or len(encoded) != 43:
        raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} RECORD has malformed SHA-256 for {path}")
    try:
        digest = base64.b64decode(encoded + "=", altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} RECORD has malformed SHA-256 for {path}") from error
    if len(digest) != hashlib.sha256().digest_size:
        raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} RECORD has malformed SHA-256 for {path}")
    return digest


def _resolve_distribution_payload(
    distribution: importlib.metadata.Distribution, root: Path, path: PurePosixPath
) -> Path:
    located = Path(os.fspath(distribution.locate_file(path))).resolve()
    expected = (root / Path(*path.parts)).resolve()
    if located != expected or not located.is_relative_to(root):
        raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} RECORD path escapes its distribution: {path}")
    return located


def _installed_distribution_revision(distribution: importlib.metadata.Distribution) -> str:
    """Return a content-addressed identity for an installed app wheel.

    Validate ``RECORD`` against the bytes owned by the installed distribution,
    then hash the canonical actual payload. Installer-generated bytecode,
    direct-url metadata, and environment-specific console scripts are not
    wheel payload and do not affect the identity.
    """

    try:
        version = distribution.version
        record = distribution.read_text("RECORD")
        root = _distribution_root(distribution)
    except (AttributeError, OSError) as error:
        raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} metadata is unavailable: {error}") from error
    if version is None or not isinstance(version, str) or not version.strip() or version != version.strip():
        raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} metadata has no version")
    if not record:
        raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} metadata has no RECORD")

    actual_payload: list[tuple[str, str, int]] = []
    payload_paths: set[PurePosixPath] = set()
    try:
        rows = csv.reader(io.StringIO(record), strict=True)
        for row in rows:
            if len(row) != 3:
                raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} RECORD has a malformed row")
            raw_path, file_hash, raw_size = row
            path = PurePosixPath(raw_path)
            if _is_installer_console_script(path):
                continue
            if (
                not raw_path
                or "\\" in raw_path
                or path.is_absolute()
                or ".." in path.parts
                or ":" in path.parts[0]
                or path.as_posix() != raw_path
            ):
                raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} RECORD has unsafe path {raw_path!r}")
            if len(path.parts) == 2 and path.parts[0].endswith(".dist-info") and path.parts[1] == "direct_url.json":
                continue
            if path.parts[0] not in _INSTALLED_PAYLOAD_ROOTS or _is_installer_bytecode(path):
                continue
            if path in payload_paths:
                raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} RECORD duplicates payload row {path}")
            payload_paths.add(path)
            if not file_hash:
                raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} RECORD has unhashed payload row {path}")
            expected_digest = _decode_record_sha256(file_hash, path=path)
            if not raw_size.isascii() or not raw_size.isdecimal() or str(int(raw_size)) != raw_size:
                raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} RECORD has malformed size for {path}")
            expected_size = int(raw_size)
            payload_path = _resolve_distribution_payload(distribution, root, path)
            try:
                payload = payload_path.read_bytes()
            except OSError as error:
                raise ValueError(
                    f"installed {_INSTALLED_DISTRIBUTION!r} payload is missing or unreadable: {path}"
                ) from error
            actual_digest = hashlib.sha256(payload).digest()
            if len(payload) != expected_size or actual_digest != expected_digest:
                raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} payload does not match RECORD: {path}")
            actual_payload.append((path.as_posix(), actual_digest.hex(), len(payload)))
    except csv.Error as error:
        raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} RECORD is malformed: {error}") from error

    if not actual_payload:
        raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} RECORD has no content-addressed payload entries")
    missing = sorted(str(path) for path in _REQUIRED_INSTALLED_FPM_PAYLOAD - payload_paths)
    if missing:
        raise ValueError(f"installed {_INSTALLED_DISTRIBUTION!r} RECORD is missing required payload rows: {missing}")
    digest = _canonical_hash({"payload": sorted(actual_payload), "version": version})
    return f"installed:{_INSTALLED_DISTRIBUTION}=={version}:record-sha256:{digest}"


def _git_revision() -> str:
    # Hermetic environments (CI containers, wheel installs) run the collector
    # outside a git checkout; an explicit revision keeps plan identity honest
    # there while the default below stays fail-closed.
    override = os.environ.get("FPM_COLLECTOR_SOURCE_REVISION", "").strip()
    if override:
        return override

    try:
        installed_distribution = importlib.metadata.distribution(_INSTALLED_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        installed_distribution = None
    if installed_distribution is not None and _distribution_owns_loaded_planner(installed_distribution):
        return _installed_distribution_revision(installed_distribution)

    root = Path(__file__).resolve().parents[2]

    def _git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return completed.stdout

    try:
        head = _git("rev-parse", "HEAD").strip()
        # Uncommitted tracked changes alter collector behavior without moving
        # HEAD; fold their digest into the revision so plan identity — and
        # with it checkpoint resume — distinguishes measurements taken under
        # different working trees. An unchanged dirty tree keeps resuming.
        # Untracked files are excluded: campaign artifact and checkpoint dirs
        # live inside the checkout and would spuriously invalidate every
        # resume.
        status = _git("status", "--porcelain", "--untracked-files=no")
        if not status.strip():
            return head
        # Plumbing diff-index, not porcelain diff: external diff drivers
        # (diff.external / .gitattributes diff=lfs|parquet in this very repo)
        # embed per-invocation temp paths that would change the digest on
        # every call, and host diff config (mnemonicPrefix, abbrev) would make
        # it host-dependent. --full-index keeps binary edits distinguishable
        # via full blob hashes without dumping content.
        diff = _git("diff-index", "--no-ext-diff", "--full-index", "-p", "HEAD")
        dirty_digest = hashlib.sha256((status + diff).encode()).hexdigest()[:12]
        return f"{head}-dirty-{dirty_digest}"
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as git_error:
        detail = getattr(git_error, "stderr", "") or str(git_error)
        raise ValueError(
            "FPM plan identity requires the collector source revision, but Git failed and this module is not owned "
            f"by the installed {_INSTALLED_DISTRIBUTION!r} distribution. Git failed under {root}: {detail.strip()}"
        ) from git_error


def _hash_stable_admission(decision: TopologyMemoryDecision) -> dict[str, object]:
    """Admission facts for the plan hash: dispositions and envelopes only.

    The free-text ``reason`` fields embed exception text (transient network
    errors, host-dependent paths) that varies across runs and hosts without
    changing behavior; hashing them would spuriously invalidate resume — and
    move the artifact root — for a plan whose actual cell population is
    identical. The full reasons stay in the persisted collection-plan.json.
    """

    payload = decision.to_dict()
    payload.pop("reason", None)
    for estimate in payload.get("estimates", []):
        if isinstance(estimate, dict):
            estimate.pop("reason", None)
    return payload


def _canonical_mapping(payload: dict[str, Any], *, field_name: str) -> str:
    if not isinstance(payload, dict):
        raise TypeError(f"BackendPolicy.{field_name} must be a mapping")
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise TypeError(f"BackendPolicy.{field_name} must be JSON serializable") from error


@dataclass(frozen=True, slots=True, init=False)
class BackendPolicy:
    policy_id: str
    _generator_overrides_json: str
    _expected_markers_json: str
    _aic_fields_json: str
    admission_reason: str

    def __init__(
        self,
        policy_id: str,
        generator_overrides: dict[str, Any],
        expected_markers: dict[str, str],
        aic_fields: dict[str, object] | None = None,
        admission_reason: str = "",
    ) -> None:
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(
            self,
            "_generator_overrides_json",
            _canonical_mapping(generator_overrides, field_name="generator_overrides"),
        )
        object.__setattr__(
            self,
            "_expected_markers_json",
            _canonical_mapping(expected_markers, field_name="expected_markers"),
        )
        object.__setattr__(
            self,
            "_aic_fields_json",
            _canonical_mapping(aic_fields or {}, field_name="aic_fields"),
        )
        object.__setattr__(self, "admission_reason", admission_reason)

    @property
    def generator_overrides(self) -> dict[str, Any]:
        """Return a detached copy so callers cannot mutate the frozen policy."""

        return json.loads(self._generator_overrides_json)

    @property
    def expected_markers(self) -> dict[str, str]:
        """Return a detached copy so validation cannot alter plan identity."""

        return json.loads(self._expected_markers_json)

    @property
    def aic_fields(self) -> dict[str, object]:
        """Return a detached copy of the structured AIC capability fields."""

        return json.loads(self._aic_fields_json)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "generator_overrides": self.generator_overrides,
            "expected_markers": self.expected_markers,
            "aic_fields": self.aic_fields,
            "admission_reason": self.admission_reason,
        }


def backend_identity_columns(policy: BackendPolicy) -> dict[str, str | bool]:
    """The four explicit backend identity columns (schema v6).

    Unspecified string knobs record "auto" (the engine decided); specified
    knobs record the pinned value. The boolean knobs stay real booleans
    (default False): the parquet column stays boolean and the modeling-side
    str() normalization yields "False"/"True".
    """
    fields = policy.aic_fields

    def _norm_backend(value: object) -> str:
        return "auto" if value is None else str(value)

    return {
        "moe_backend": _norm_backend(fields.get("moe_backend")),
        "attention_backend": _norm_backend(fields.get("attention_backend")),
        # Real booleans: the parquet column stays boolean and the
        # modeling-side str() normalization yields "False"/"True",
        # matching ModelConfig's Python bool defaults.
        "enable_wideep": bool(fields.get("enable_wideep")),
        "enable_eplb": bool(fields.get("enable_eplb")),
    }


def _backend_policies(
    options: FPMCollectionOptions,
    collector_config: dict[str, Any],
    *,
    backend: str,
) -> tuple[BackendPolicy, ...]:
    declarations = collector_config.get("backend_variants", {})
    if not isinstance(declarations, dict):
        raise TypeError("FpmCollector.backend_variants must be a mapping")
    if declarations:
        raise ValueError(
            "FpmCollector.backend_variants is no longer an admission mechanism; backend policies must come from "
            "AIC structured capabilities"
        )
    moe = options.moe_backend
    attention = options.attention_backend
    wideep = options.enable_wideep
    eplb = options.enable_eplb
    specified = {
        name: value
        for name, value in (
            ("moe_backend", moe),
            ("attention_backend", attention),
            ("enable_wideep", wideep),
            ("enable_eplb", eplb),
        )
        if value not in ("auto", "false")
    }

    # Fail closed on anything the collector cannot deliver to the engine and
    # verify: a row claiming a backend the engine never ran is worse than no
    # row. vLLM plumbing exists for moe_backend (--kernel-config) and eplb
    # (--enable-eplb); wide-EP is an SGLang concept; pinning the attention
    # backend has no verified vLLM plumbing yet.
    if specified and backend != "vllm":
        raise ValueError(
            f"explicit FPM backend identity is only plumbed for the vllm backend; got {backend} with "
            f"{sorted(specified)} specified"
        )
    if wideep == "true":
        raise ValueError("enable_wideep=true is not collectable on vllm (wide-EP is SGLang-only)")
    if attention != "auto":
        raise ValueError(
            "attention_backend pinning has no verified vllm plumbing yet; collect with auto or add the "
            "engine flag and its resolved-config marker first"
        )

    extra_cli_args: list[str] = []
    expected_markers: dict[str, str] = {}
    if options.enforce_eager:
        expected_markers["config.engine_args.enforce_eager"] = "True"
    if moe != "auto":
        extra_cli_args += ["--kernel-config", json.dumps({"moe_backend": moe})]
        expected_markers["config.engine_args.kernel_config.moe_backend"] = moe
    if eplb == "true":
        extra_cli_args += ["--enable-eplb"]
        expected_markers["config.engine_args.enable_eplb"] = "True"

    generator_overrides: dict[str, Any] = (
        {"params": {"agg": {"extra_cli_args": extra_cli_args}}} if extra_cli_args else {}
    )
    policy_id = (
        "baseline_auto"
        if not specified
        else "explicit-" + "-".join(f"{name}={value}" for name, value in sorted(specified.items()))
    )
    return (
        BackendPolicy(
            policy_id,
            generator_overrides,
            expected_markers,
            {
                "moe_backend": None if moe == "auto" else moe,
                "attention_backend": None if attention == "auto" else attention,
                "enable_wideep": wideep == "true",
                "enable_eplb": eplb == "true",
            },
            "AIC automatic baseline for the selected model/backend"
            if not specified
            else "explicitly pinned backend identity",
        ),
    )


@dataclass(frozen=True, slots=True)
class FPMCell:
    cell_id: str
    workload_kind: str
    topology: ParallelTopology
    weight_quantization: str
    kv_cache_dtype: str
    backend_policy: BackendPolicy
    gemm_quant_mode: str
    parallel_strategy: str = "unspecified"
    moe_quant_mode: str | None = None
    fmha_quant_mode: str | None = None
    comm_quant_mode: str | None = None
    fmha_resolution: str | None = None
    execution_identity: tuple[str, ...] = LEGACY_EXECUTION_IDENTITY
    input_text_sha256: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "execution_identity": dict(zip(EXECUTION_COLUMNS, self.execution_identity, strict=True)),
            "input_text_sha256": self.input_text_sha256,
            "workload_kind": self.workload_kind,
            "point_source": "dynamo_native_self_benchmark",
            "topology": self.topology.to_dict(),
            "parallel_strategy": self.parallel_strategy,
            "weight_quantization": self.weight_quantization,
            "kv_cache_dtype": self.kv_cache_dtype,
            "resolved_dtypes": {
                "gemm_quant_mode": self.gemm_quant_mode,
                "moe_quant_mode": self.moe_quant_mode,
                "fmha_quant_mode": self.fmha_quant_mode,
                "comm_quant_mode": self.comm_quant_mode,
                "kvcache_quant_mode": self.kv_cache_dtype,
                "fmha_resolution": self.fmha_resolution,
            },
            "backend_policy": self.backend_policy.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FPMCollectionPlan:
    backend: str
    model_path: str
    system: str
    aic_revision: str
    generator_config_sha256: str
    options: FPMCollectionOptions
    capability: ModelCapabilityProfile
    dtype_profile: ResolvedDTypeProfile
    topologies: tuple[ParallelTopology, ...]
    topology_memory_admission: tuple[TopologyMemoryDecision, ...]
    backend_policies: tuple[BackendPolicy, ...]
    cells: tuple[FPMCell, ...]
    sha256: str
    _fpm_profile_json: str | None = field(default=None, repr=False)
    _runtime_observation_json: str | None = field(default=None, repr=False)

    @property
    def runtime_instrumentation(self):
        if self._runtime_observation_json is None:
            return None
        from .runtime_instrumentation import load_instrumentation

        reference = json.loads(self._runtime_observation_json)
        bundle = load_instrumentation(reference["manifest"], expected_version=self.capability.aic_database_version)
        if bundle.sha256 != reference["bundle_sha256"]:
            raise ValueError("formal collection instrumentation changed after planning")
        return bundle

    @property
    def runtime_launch(self) -> dict[str, Any] | None:
        return json.loads(self._runtime_observation_json)["launch"] if self._runtime_observation_json else None

    @property
    def runtime_configuration(self) -> str | None:
        return json.loads(self._runtime_observation_json)["configuration"] if self._runtime_observation_json else None

    @property
    def fpm_profile(self) -> FpmModelProfile | None:
        """Return detached profile metadata from the frozen collection input."""
        return load_fpm_profile(self._fpm_profile_json) if self._fpm_profile_json is not None else None

    def deployment_profile(self, cell: FPMCell) -> FpmDeploymentProfile | None:
        profile = self.fpm_profile
        if profile is None:
            return None
        return profile.select(
            model=self.model_path,
            system=self.system,
            backend=self.backend,
            backend_version=self.capability.aic_database_version,
            tp_size=cell.topology.tp,
            pp_size=cell.topology.pp,
            attention_dp_size=cell.topology.dp,
            moe_tp_size=cell.topology.moe_tp,
            moe_ep_size=cell.topology.moe_ep,
            cp_size=cell.topology.cp,
        )

    def to_dict(self) -> dict[str, object]:
        prefill_sampling = self.options.prefill_sampling.to_dict()
        explicit_points = (
            json.loads(self.options.benchmark_points_json) if self.options.benchmark_points_json is not None else None
        )
        payload = {
            "schema_name": "aic_fpm_collection_plan",
            "schema_version": 11,
            "backend": self.backend,
            "model_path": self.model_path,
            "system": self.system,
            "aic_revision": self.aic_revision,
            "generator_config_sha256": self.generator_config_sha256,
            "options": self.options.to_dict(),
            "capability": self.capability.to_dict(),
            "dtype_profile": self.dtype_profile.to_dict(),
            "point_generation": {
                "owner": "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler",
                "method": "native_self_benchmark",
                "source": "frozen_explicit_manifest" if explicit_points is not None else "native_auto_grid",
                "manifest_sha256": self.options.benchmark_points_sha256,
                "coordinates": [
                    "batch_size",
                    "total_prefill_tokens",
                    "total_kv_read_tokens",
                ],
                "partition_policy": "balanced_v1",
                "point_admission": "dynamo_live_scheduler",
                "precondition": "vllm_engine_initialized",
                "prefill_sampling": prefill_sampling,
                "planned_point_count": (
                    sum(len(explicit_points.get(phase, [])) for phase in ("prefill", "decode"))
                    if explicit_points is not None
                    else None
                ),
            },
            "topologies": [
                {
                    **topology.to_dict(),
                    "strategy": topology_strategy(topology, is_moe=self.capability.is_moe),
                }
                for topology in self.topologies
            ],
            "topology_memory_admission": [decision.to_dict() for decision in self.topology_memory_admission],
            "backend_policies": [policy.to_dict() for policy in self.backend_policies],
            "cells": [cell.to_dict() for cell in self.cells],
            "counts": {
                "candidate_topologies": len(self.topology_memory_admission),
                "topologies": len(self.topologies),
                "memory_rejected_topologies": sum(
                    decision.disposition == "rejected" for decision in self.topology_memory_admission
                ),
                "memory_unknown_topologies": sum(
                    decision.disposition == "unknown" for decision in self.topology_memory_admission
                ),
                "backend_policies": len(self.backend_policies),
                "cells": len(self.cells),
                "prefill_cudagraph_capture_sizes": prefill_sampling["cudagraph_capture_size_count"],
                "prefill_new_token_axis_points": prefill_sampling["new_token_axis_point_count"],
                "points": "runtime-determined",
            },
            "sha256": self.sha256,
        }
        if self._fpm_profile_json is not None:
            payload["fpm_profile"] = json.loads(self._fpm_profile_json)
        runtime_observation = json.loads(self._runtime_observation_json) if self._runtime_observation_json else None
        memory_policy = _runtime_memory_policy(
            self.fpm_profile, self.capability.aic_database_version, runtime_observation
        )
        if memory_policy is not None:
            payload["runtime_memory_policy"] = memory_policy
        if runtime_observation is not None:
            payload["runtime_observation"] = runtime_observation
        return payload


def _cell_id(
    *,
    backend: str,
    model_path: str,
    system: str,
    phase: str,
    topology: ParallelTopology,
    weight_quantization: str,
    kv_cache_dtype: str,
    policy: BackendPolicy,
    execution: tuple[str, ...] = LEGACY_EXECUTION_IDENTITY,
    input_text_sha256: str = "",
) -> str:
    payload = {
        "backend": backend,
        "model_path": model_path,
        "system": system,
        "phase": phase,
        "topology": topology.to_dict(),
        "weight_quantization": weight_quantization,
        "kv_cache_dtype": kv_cache_dtype,
        **backend_identity_columns(policy),
        **dict(zip(EXECUTION_COLUMNS, execution, strict=True)),
        "input_text_sha256": input_text_sha256,
        "point_source": "dynamo_native_self_benchmark",
    }
    return f"fpm-{_canonical_hash(payload)[:16]}"


def build_collection_plan(
    *,
    backend: str,
    model_path: str,
    system: str,
    selected_ops: set[str],
    options: FPMCollectionOptions,
    model_architecture: str | None = None,
    has_model_cases: bool = True,
    model_config_path: str | None = None,
    fpm_profile: FpmModelProfile | dict[str, Any] | None = None,
    collector_config: dict[str, Any] | None = None,
    generator_overrides: dict[str, Any] | None = None,
    runtime_instrumentation=None,
    runtime_launch: dict[str, Any] | None = None,
    runtime_configuration: str = "collection",
) -> FPMCollectionPlan:
    if backend != "vllm":
        raise ValueError("FPM Generator V1 currently supports only backend=vllm")
    collector_config = dict(collector_config or {})
    profile = load_fpm_profile(fpm_profile) if fpm_profile is not None else None
    if profile is not None:
        if profile.model != model_path:
            raise ValueError(
                f"FPM profile model identity mismatch: requested {model_path!r}, profile={profile.model!r}"
            )
        versions = {
            deployment.backend_version
            for deployment in profile.deployments
            if deployment.system == system and deployment.backend == backend
        }
        if "aic_database_version" not in collector_config:
            if len(versions) != 1:
                raise ValueError(
                    "FPM collection profile must identify one runtime version for the target "
                    f"{system}/{backend}; found {sorted(versions)}"
                )
            collector_config["aic_database_version"] = next(iter(versions))
    # Freeze resolved warm-up defaults as well as explicit deployment inputs;
    # changing the default must not resume a campaign under its old plan hash.
    generator_config_sha256 = _canonical_hash(with_kv_warmup_defaults(generator_overrides or {}))
    capability = resolve_model_capability(
        backend=backend,
        model_path=model_path,
        model_architecture=model_architecture,
        selected_ops=selected_ops,
        has_model_cases=has_model_cases,
        system=system,
        requested_weight_quantizations=options.weight_quantizations,
        requested_kv_cache_dtypes=options.kv_cache_dtypes,
        model_config_path=model_config_path,
        database_version=(
            str(collector_config["aic_database_version"]) if "aic_database_version" in collector_config else None
        ),
        checkpoint_native_dtypes=profile is not None,
    )
    execution = execution_identity(
        capability.model_config.payload,
        decoder_replay=options.decoder_replay,
        backend=backend,
        # The rendered V4.1 collection contract requests text-only HBM Engram.
        # The producer must independently attest these actual runtime facts.
        engram_cpu_offload=False,
        input_modality="text",
    )
    if execution[0] and not options.enforce_eager:
        raise ValueError("V4.1 FPM collection currently requires --fpm-enforce-eager; graph timing is not qualified")
    if options.enforce_eager and not execution[0]:
        raise ValueError("explicit eager FPM collection is currently qualified only for DeepSeek V4.1")
    input_text_sha256 = (
        hashlib.sha256((Path(__file__).parent / "runtime" / "fpm_text.txt").read_bytes()).hexdigest()
        if execution[0]
        else ""
    )
    candidate_topologies = enumerate_fpm_topologies(
        backend=backend,
        is_moe=capability.is_moe,
        options=options,
        allow_pure_tp=capability.allow_pure_tp,
    )
    policies = _backend_policies(options, collector_config, backend=backend)
    if profile is not None:
        deployments = _validate_profile_identities(
            profile, capability, candidate_topologies, policies, model_path, system, backend
        )
        if options.vllm_max_model_len > profile.context_length:
            raise ValueError(
                f"--fpm-max-model-len={options.vllm_max_model_len} exceeds "
                f"FPM profile context_length={profile.context_length}"
            )
        max_prefill_isl = options.max_prefill_isl
        if max_prefill_isl is None:
            max_prefill_isl = options.max_num_batched_tokens
            if max_prefill_isl is None:
                max_prefill_isl = min(
                    [FPM_MAX_PREFILL_ISL, *(deployment.resources.max_num_tokens for deployment in deployments)]
                )
        options = replace(
            options,
            vllm_max_model_len=(
                profile.context_length
                if options.vllm_max_model_len == VLLM_AUTO_FIT_MAX_MODEL_LEN
                else options.vllm_max_model_len
            ),
            max_prefill_isl=max_prefill_isl,
        )
        # Validate the shared decode envelope as well as narrower prefill
        # controls before memory admission can queue or drop any deployment.
        for deployment in deployments:
            resources = deployment.resources
            options.validate_scheduler_limits(
                profile_max_num_tokens=resources.max_num_tokens, profile_max_batch_size=resources.max_batch_size
            )
            resources.validate_envelope(
                max_num_tokens=options.max_num_batched_tokens or max_prefill_isl,
                max_batch_size=max(
                    options.max_decode_batch_size or options.max_num_seqs or resources.max_batch_size,
                    options.max_prefill_batch_size or options.max_num_seqs or resources.max_batch_size,
                ),
            )
    topologies, topology_memory_admission = filter_memory_infeasible_topologies(
        backend=backend,
        model_path=model_path,
        system=system,
        capability=capability,
        topologies=candidate_topologies,
        max_new_tokens=options.prefill_sampling.max_total_prefill_tokens,
        fpm_profile=profile,
        max_batch_size=options.prefill_sampling.max_batch_size,
        gpu_memory_utilization=options.gpu_memory_utilization,
    )
    weight_quantization = capability.dtype.gemm_quant_mode
    runnable_dtype_pairs = {
        (decision.topology, estimate.kv_cache_dtype)
        for decision in topology_memory_admission
        for estimate in decision.estimates
        if estimate.disposition != "rejected"
    }
    # An admitted topology can still lose individual KV dtypes to the memory
    # budget; those cells silently vanish from the plan unless counted here
    # (the memory filter itself only logs fully rejected topologies).
    kept_topologies = set(topologies)
    dropped_dtype_pairs = [
        (decision.topology, estimate.kv_cache_dtype)
        for decision in topology_memory_admission
        if decision.topology in kept_topologies
        for estimate in decision.estimates
        if estimate.disposition == "rejected"
    ]
    if dropped_dtype_pairs:
        details = "; ".join(f"{topology.to_dict()}/kv={dtype}" for topology, dtype in dropped_dtype_pairs)
        logger.warning(
            "fpm_forward: dropped %d/%d (topology, kv_dtype) cell groups (memory budget, system=%s): %s",
            len(dropped_dtype_pairs),
            len(kept_topologies) * len(capability.dtype.kv_cache_dtypes),
            system,
            details,
        )
    cells = tuple(
        FPMCell(
            cell_id=_cell_id(
                backend=backend,
                model_path=model_path,
                system=system,
                phase=phase,
                topology=topology,
                weight_quantization=weight_quantization,
                kv_cache_dtype=kv_cache_dtype,
                policy=policy,
                execution=execution,
                input_text_sha256=input_text_sha256,
            ),
            execution_identity=execution,
            input_text_sha256=input_text_sha256,
            workload_kind=phase,
            topology=topology,
            weight_quantization=weight_quantization,
            kv_cache_dtype=kv_cache_dtype,
            backend_policy=policy,
            parallel_strategy=topology_strategy(topology, is_moe=capability.is_moe),
            gemm_quant_mode=capability.dtype.gemm_quant_mode,
            moe_quant_mode=capability.dtype.moe_quant_mode,
            fmha_quant_mode=capability.dtype.fmha_by_kv_dtype[kv_cache_dtype],
            comm_quant_mode=capability.dtype.comm_quant_mode,
            fmha_resolution=capability.dtype.fmha_resolution_by_kv_dtype[kv_cache_dtype],
        )
        for phase in ("prefill", "decode")
        for topology in topologies
        for kv_cache_dtype in capability.dtype.kv_cache_dtypes
        if (topology, kv_cache_dtype) in runnable_dtype_pairs
        for policy in policies
    )
    revision = _git_revision()
    canonical = {
        "backend": backend,
        "model_path": model_path,
        "system": system,
        "aic_revision": revision,
        "generator_config_sha256": generator_config_sha256,
        "options": options.to_dict(),
        "capability": capability.to_dict(),
        "dtype_profile": capability.dtype.to_dict(),
        "point_generation": "dynamo_native_self_benchmark",
        "topology_memory_admission": [_hash_stable_admission(decision) for decision in topology_memory_admission],
        "topologies": [topology.to_dict() for topology in topologies],
        "policies": [policy.to_dict() for policy in policies],
        "cells": [cell.to_dict() for cell in cells],
    }
    profile_json = None
    if profile is not None:
        canonical["fpm_profile"] = profile.model_dump(mode="json")
        profile_json = json.dumps(canonical["fpm_profile"], sort_keys=True, separators=(",", ":"))
    runtime_observation_json = None
    if runtime_instrumentation is not None:
        from .runtime_instrumentation import load_instrumentation
        from .runtime_probe import normalize_probe_launch, validate_collection_probe_launch

        if runtime_launch is None:
            raise ValueError("formal runtime instrumentation requires the accepted probe launch facts")
        bundle = (
            load_instrumentation(runtime_instrumentation, expected_version=capability.aic_database_version)
            if isinstance(runtime_instrumentation, (str, Path))
            else runtime_instrumentation
        )
        launch = normalize_probe_launch(runtime_launch)
        validate_collection_probe_launch(
            launch,
            bundle,
            model_path=model_path,
            system=system,
            backend=backend,
            backend_version=capability.aic_database_version,
            cells=cells,
            options=options,
            profile=profile,
            generator_overrides=generator_overrides or {},
        )
        canonical["runtime_observation"] = {
            "manifest": str(bundle.manifest_path),
            "bundle_sha256": bundle.sha256,
            "launch": launch,
            "configuration": runtime_configuration,
        }
        runtime_observation_json = json.dumps(canonical["runtime_observation"], sort_keys=True)
    elif runtime_launch is not None:
        raise ValueError("accepted probe launch facts require runtime instrumentation for formal collection")
    memory_policy = _runtime_memory_policy(
        profile, capability.aic_database_version, canonical.get("runtime_observation")
    )
    if memory_policy is not None:
        canonical["runtime_memory_policy"] = memory_policy
    return FPMCollectionPlan(
        backend=backend,
        model_path=model_path,
        system=system,
        aic_revision=revision,
        generator_config_sha256=generator_config_sha256,
        options=options,
        capability=capability,
        dtype_profile=capability.dtype,
        topologies=topologies,
        topology_memory_admission=topology_memory_admission,
        backend_policies=policies,
        cells=cells,
        sha256=_canonical_hash(canonical),
        _fpm_profile_json=profile_json,
        _runtime_observation_json=runtime_observation_json,
    )


def _validate_profile_identities(
    profile: FpmModelProfile,
    capability: ModelCapabilityProfile,
    topologies: tuple[ParallelTopology, ...],
    policies: tuple[BackendPolicy, ...],
    model_path: str,
    system: str,
    backend: str,
) -> tuple[FpmDeploymentProfile, ...]:
    """Require a resource declaration for every requested cell, before admission."""
    if profile.architecture != capability.architecture:
        raise ValueError(
            "FPM collection profile architecture mismatch: "
            f"checkpoint={capability.architecture!r}, profile={profile.architecture!r}"
        )
    identity_fields = (
        "gemm_quant_mode",
        "moe_quant_mode",
        "fmha_quant_mode",
        "comm_quant_mode",
        "kv_cache_dtype",
        *PARALLEL_AXES,
        "moe_backend",
        "attention_backend",
        "enable_wideep",
        "enable_eplb",
    )
    deployments = []
    for topology in topologies:
        deployment = profile.select(
            model=model_path,
            system=system,
            backend=backend,
            backend_version=capability.aic_database_version,
            tp_size=topology.tp,
            pp_size=topology.pp,
            attention_dp_size=topology.dp,
            moe_tp_size=topology.moe_tp,
            moe_ep_size=topology.moe_ep,
            cp_size=topology.cp,
        )
        deployments.append(deployment)
        for kv_dtype in capability.dtype.kv_cache_dtypes:
            for policy in policies:
                resolved = [
                    capability.dtype.gemm_quant_mode,
                    capability.dtype.moe_quant_mode,
                    capability.dtype.fmha_by_kv_dtype[kv_dtype],
                    capability.dtype.comm_quant_mode,
                    kv_dtype,
                    *(str(getattr(topology, axis)) for axis in PARALLEL_AXES),
                    *(str(value) for value in backend_identity_columns(policy).values()),
                ]
                if deployment.match_identity() != resolved:
                    conflicts = "; ".join(
                        f"{name}: resolved={actual!r}, profile={expected!r}"
                        for name, actual, expected in zip(
                            identity_fields, resolved, deployment.match_identity(), strict=True
                        )
                        if actual != expected
                    )
                    raise ValueError(
                        f"FPM collection profile identity mismatch: {conflicts}; supply a profile matching "
                        "the collection configuration. The profile does not override serving dispatch."
                    )
    return tuple(deployments)
