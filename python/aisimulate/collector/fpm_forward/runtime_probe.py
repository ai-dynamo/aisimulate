# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch runtime observations before simulation cache geometry is known."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import re
import shutil
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import runner
from .config import FPM_WARMUP_ITERATIONS, FPMCollectionOptions
from .entry import _load_generator_overrides
from .model_capability import ResolvedModelConfig, load_model_config
from .planner import FPMCell, _backend_policies, _canonical_hash
from .runtime_instrumentation import contained_file, read_json, validate_sha256
from .topology import enumerate_fpm_topologies, topology_strategy
from .types import ParallelTopology

INDEX_SCHEMA = "aisimulate-runtime-observations/v1"
LAUNCH_SCHEMA = "aisimulate-runtime-probe-launch/v1"
CONTEXT_FILENAME = "runtime-probe-context.json"
BUNDLE_FILENAME = "runtime-instrumentation.zip"
PHASES = ("prefill", "decode")
# Dynamo b83b1d9304ebfc624709ac46db32b1b6f1ff1615, backend_args.py:
# _validate_benchmark_sampling requires two endpoints on each uniform axis.
DEFAULT_SAMPLING = {"max_new_token_samples": 2, "max_kv_read_token_samples": 2, "max_batch_size_samples": 2}


@dataclass(frozen=True, slots=True)
class _RuntimeModel:
    architecture: str
    aic_database_version: str
    model_config: ResolvedModelConfig


@dataclass(frozen=True, slots=True)
class RuntimeProbePlan:
    """The renderer's launch inputs, with no performance or resource profile."""

    configuration: str
    _launch_json: str
    options: FPMCollectionOptions
    capability: _RuntimeModel
    cells: tuple[FPMCell, ...]
    runtime_instrumentation: Any
    sha256: str

    @property
    def launch(self) -> dict[str, Any]:
        return json.loads(self._launch_json)

    @property
    def model_path(self) -> str:
        return self.launch["identity"]["model"]

    @property
    def system(self) -> str:
        return self.launch["identity"]["gpu"]

    @property
    def backend(self) -> str:
        return self.launch["identity"]["framework"]

    @property
    def fpm_profile(self) -> None:
        return None

    @property
    def runtime_launch(self) -> dict[str, Any]:
        return self.launch

    @property
    def probe_sampling(self) -> dict[str, int]:
        return dict(DEFAULT_SAMPLING)


def _required_mapping(payload: dict[str, Any], field: str) -> dict[str, Any]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"runtime probe requires {field} launch facts")
    return value


def normalize_probe_launch(value: dict[str, Any]) -> dict[str, Any]:
    """Freeze explicit stage-2 launch facts and documented sampling defaults."""

    launch = json.loads(json.dumps(value, allow_nan=False))
    identity = _required_mapping(launch, "identity")
    for name in ("model", "model_revision", "model_kind", "framework", "framework_version", "gpu", "interconnect"):
        item = identity.get(name)
        if not isinstance(item, str) or not item.strip() or item != item.strip() or any(ord(c) < 32 for c in item):
            raise ValueError(f"runtime probe requires identity.{name}")
    if identity["framework"] != "vllm" or identity["model_kind"] not in {"dense", "moe"}:
        raise ValueError("runtime probe requires framework=vllm and explicit model_kind=dense|moe")
    if any(
        identity[name].lower() in {"main", "master", "latest", "head"}
        for name in ("model_revision", "framework_version")
    ):
        raise ValueError("runtime probe requires pinned model and framework revisions")
    topology = _required_mapping(launch, "topology")
    for name in ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp"):
        if type(topology.get(name)) is not int or topology[name] < 1:
            raise ValueError(f"runtime probe requires positive topology.{name}")
    if topology["pp"] != 1 or topology["cp"] != 1:
        raise ValueError("runtime observation currently supports PP1 and CP1")
    precision = _required_mapping(launch, "precision")
    for name in ("gemm_quant_mode", "moe_quant_mode", "fmha_quant_mode", "kvcache_quant_mode", "comm_quant_mode"):
        if not isinstance(precision.get(name), str) or not precision[name] or precision[name] == "auto":
            raise ValueError(f"runtime probe requires explicit precision.{name}")
    if precision["comm_quant_mode"] != "half":
        raise ValueError("runtime probe requires the collector comm_quant_mode=half identity")
    for name in ("moe_backend", "attention_backend"):
        precision.setdefault(name, "auto")
        if not isinstance(precision[name], str) or not precision[name]:
            raise ValueError(f"precision.{name} must be a nonempty string")
    for name in ("enable_wideep", "enable_eplb"):
        precision.setdefault(name, False)
        if type(precision[name]) is not bool:
            raise ValueError(f"precision.{name} must be boolean")
    collection = _required_mapping(launch, "collection")
    for name in ("max_model_len", "max_num_batched_tokens", "max_num_seqs"):
        if type(collection.get(name)) is not int or collection[name] < 1:
            raise ValueError(f"runtime probe requires positive collection.{name}")
    fraction = collection.get("gpu_memory_utilization")
    if type(fraction) not in {int, float} or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("runtime probe requires collection.gpu_memory_utilization in (0, 1]")
    policy = collection.get("prefill_cudagraph_policy")
    if policy not in {"runtime", "explicit"}:
        raise ValueError("runtime probe requires collection.prefill_cudagraph_policy=runtime|explicit")
    if collection.get("max_prefill_cudagraph_size") is None:
        collection["max_prefill_cudagraph_size"] = None if policy == "runtime" else 2048
    collection.setdefault("async_scheduling", False)
    if collection["async_scheduling"] is not False:
        raise ValueError("the native prefill observation protocol requires async_scheduling=false in both phases")
    collection.setdefault("enforce_eager", False)
    if type(collection["enforce_eager"]) is not bool:
        raise ValueError("collection.enforce_eager must be boolean")
    collection.setdefault("warmup_iterations", FPM_WARMUP_ITERATIONS)
    if type(collection["warmup_iterations"]) is not int or collection["warmup_iterations"] < 0:
        raise ValueError("collection.warmup_iterations must be a nonnegative integer")
    config = _required_mapping(launch, "model_config")
    if not isinstance(config.get("path"), str) or not config["path"]:
        raise ValueError("runtime probe requires model_config.path and model_config.sha256")
    validate_sha256(config.get("sha256"), "model configuration")
    sources = config.get("source_files", {})
    if not isinstance(sources, dict) or sources.keys() - {"hf_quant_config.json"}:
        raise ValueError("model_config.source_files supports the adjacent hf_quant_config.json only")
    for name, digest in sources.items():
        validate_sha256(digest, f"model configuration source {name}")
    deployment = _required_mapping(launch, "deployment")
    deployment.setdefault("executor", "kubernetes")
    deployment.setdefault("container_mount", [])
    if deployment["executor"] not in {"kubernetes", "slurm"}:
        raise ValueError("runtime probe executor must be kubernetes or slurm")
    image = deployment.get("image")
    if not isinstance(image, str) or not image or any(c.isspace() or ord(c) < 32 for c in image):
        raise ValueError("runtime probe requires deployment.image for the pinned runtime")
    mounts = deployment["container_mount"]
    if not isinstance(mounts, list) or any(
        not isinstance(m, str) or not m or any(ord(c) < 32 or c == "," for c in m) for m in mounts
    ):
        raise ValueError("deployment.container_mount must contain valid Slurm mount strings")
    if deployment["executor"] == "slurm" and any(
        deployment.get(key) is not None for key in ("namespace", "model_cache", "image_pull_secret")
    ):
        raise ValueError("Slurm runtime probes reject Kubernetes namespace, model_cache and image_pull_secret")
    if deployment["executor"] != "slurm" and mounts:
        raise ValueError("container_mount requires the Slurm executor")
    cpus, binding = deployment.get("cpus_per_task"), deployment.get("cpu_bind")
    if cpus is not None or binding is not None:
        if deployment["executor"] != "slurm":
            raise ValueError("CPU deployment options require the Slurm executor")
        if type(cpus) is not int or cpus < 1 or binding not in {"cores", "none"}:
            raise ValueError("Slurm CPU policy requires positive cpus_per_task and cpu_bind cores or none")
    return launch


def _model_config_bytes(config: dict[str, Any]) -> dict[str, bytes]:
    """Bind the raw config and every adjacent file consumed by the local loader."""

    path = Path(config["path"]).expanduser().absolute()
    sources = config.get("source_files", {})
    adjacent = path.parent / "hf_quant_config.json"
    if (adjacent.exists() or adjacent.is_symlink()) and adjacent.name not in sources:
        raise ValueError("adjacent hf_quant_config.json must be recorded in model_config.source_files")
    contents = {"model-config.json": path.read_bytes()}
    if hashlib.sha256(contents["model-config.json"]).hexdigest() != config["sha256"]:
        raise ValueError("runtime probe model config does not match its recorded SHA-256")
    for name, digest in sources.items():
        # HF snapshots use adjacent symlinks into a shared blob directory. The
        # logical name is constrained above; its loaded bytes remain hash-bound.
        source = path.parent / name
        if not source.is_file():
            raise ValueError(f"model configuration source is not a regular file: {name}")
        contents[name] = source.read_bytes()
        if hashlib.sha256(contents[name]).hexdigest() != digest:
            raise ValueError(f"model configuration source {name} does not match its recorded SHA-256")
    return contents


def probe_generator_overrides(launch: dict[str, Any]) -> dict[str, Any]:
    """Translate the existing deployment options through their shared parser."""

    deployment = launch["deployment"]
    image = deployment["image"]
    args = argparse.Namespace(
        generator_config=None,
        generator_set=[f"K8sConfig.k8s_image={json.dumps(image)}"],
        generator_dynamo_version=deployment.get("dynamo_version"),
        generated_config_version=None,
        namespace=deployment.get("namespace"),
        transport=deployment.get("transport"),
        image_pull_secret=deployment.get("image_pull_secret"),
        model_cache=deployment.get("model_cache"),
    )
    return _load_generator_overrides(args)


def build_runtime_probe_plan(configuration: str, facts: dict[str, Any], bundle: Any) -> RuntimeProbePlan:
    launch = normalize_probe_launch(facts)
    identity, precision, collection, deployment = (
        launch[key] for key in ("identity", "precision", "collection", "deployment")
    )
    if bundle.manifest["runtime"]["version"] != identity["framework_version"]:
        raise ValueError("instrumentation runtime version does not match the requested pin")
    _model_config_bytes(launch["model_config"])
    model = load_model_config(identity["model"], explicit_config_path=launch["model_config"]["path"])
    # The shared loader reads the adjacent sidecar; reject changes during loading
    # without replacing the raw primary-config hash with a merged payload hash.
    _model_config_bytes(launch["model_config"])
    config = model.effective_payload
    architecture = (config.get("architectures") or [None])[0]
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("runtime probe model config requires an architecture")
    from aisimulate.sdk.models.helpers import _infer_quant_modes_from_raw_config
    from aisimulate.sdk.utils import _attach_inferred_quant_fields

    inferred = _infer_quant_modes_from_raw_config(_attach_inferred_quant_fields(config), architecture)
    for name in ("gemm_quant_mode", "moe_quant_mode"):
        actual = getattr(inferred.get(name), "name", inferred.get(name))
        if actual is None:
            actual = {"bfloat16": "bfloat16", "float16": "half", "half": "half"}.get(
                config.get("dtype") or config.get("torch_dtype")
            )
        if actual != precision[name]:
            raise ValueError(f"precision.{name}={precision[name]!r} does not match checkpoint-native {actual!r}")
    topology = ParallelTopology(**launch["topology"])
    is_moe = identity["model_kind"] == "moe"
    strategy = topology_strategy(topology, is_moe=is_moe)
    preset = strategy if strategy != "single" else ("tep" if is_moe else "tp")
    options = FPMCollectionOptions(
        max_gpus=topology.total_gpus,
        gpu_counts=(topology.total_gpus,),
        parallel_presets=(preset,),
        parallel_axes=(),
        moe_backend=precision["moe_backend"],
        attention_backend=precision["attention_backend"],
        enable_wideep=str(precision["enable_wideep"]).lower(),
        enable_eplb=str(precision["enable_eplb"]).lower(),
        weight_quantizations=(precision["gemm_quant_mode"],),
        kv_cache_dtypes=(precision["kvcache_quant_mode"],),
        warmup_iterations=collection["warmup_iterations"],
        vllm_max_model_len=collection["max_model_len"],
        max_num_batched_tokens=collection["max_num_batched_tokens"],
        max_num_seqs=collection["max_num_seqs"],
        prefill_cudagraph_policy=collection["prefill_cudagraph_policy"],
        max_prefill_cudagraph_size=collection["max_prefill_cudagraph_size"],
        gpu_memory_utilization=collection["gpu_memory_utilization"],
        enforce_eager=collection["enforce_eager"],
        executor=deployment["executor"],
        slurm_container_image=deployment["image"] if deployment["executor"] == "slurm" else "",
        slurm_container_mounts=tuple(deployment["container_mount"]),
        slurm_cpus_per_task=deployment.get("cpus_per_task"),
        slurm_cpu_bind=deployment.get("cpu_bind"),
    )
    admitted = enumerate_fpm_topologies(backend="vllm", is_moe=is_moe, options=options, allow_pure_tp=True)
    if topology not in admitted:
        raise ValueError(f"collector does not admit exact topology {topology.to_dict()}")
    policy = _backend_policies(options, {}, backend="vllm")[0]
    sha256 = _canonical_hash({"launch": launch, "bundle_sha256": bundle.sha256})
    cells = tuple(
        FPMCell(
            cell_id="probe-" + _canonical_hash({"configuration": configuration, "phase": phase, "plan": sha256})[:20],
            workload_kind=phase,
            topology=topology,
            weight_quantization=precision["gemm_quant_mode"],
            kv_cache_dtype=precision["kvcache_quant_mode"],
            backend_policy=policy,
            gemm_quant_mode=precision["gemm_quant_mode"],
            moe_quant_mode=precision["moe_quant_mode"],
            fmha_quant_mode=precision["fmha_quant_mode"],
            comm_quant_mode=precision["comm_quant_mode"],
            parallel_strategy=strategy,
            fmha_resolution="runtime_probe_requested",
        )
        for phase in PHASES
    )
    return RuntimeProbePlan(
        configuration,
        json.dumps(launch, sort_keys=True),
        options,
        _RuntimeModel(architecture, identity["framework_version"], model),
        cells,
        bundle,
        sha256,
    )


def validate_collection_probe_launch(
    launch, bundle, *, model_path, system, backend, backend_version, cells, options, profile, generator_overrides
):
    """Bind optional formal instrumentation to the actual collection inputs."""

    expected = build_runtime_probe_plan("collection", launch, bundle)
    identity = launch["identity"]
    if (model_path, system, backend, backend_version) != (
        identity["model"],
        identity["gpu"],
        identity["framework"],
        identity["framework_version"],
    ):
        raise ValueError("formal collection identity differs from the accepted runtime probe")
    if profile is None or profile.model_revision != identity["model_revision"]:
        raise ValueError("formal collection model revision differs from the accepted runtime probe")
    expected_cell = expected.cells[0]
    for cell in cells:
        fields = (
            "topology",
            "gemm_quant_mode",
            "moe_quant_mode",
            "fmha_quant_mode",
            "comm_quant_mode",
            "kv_cache_dtype",
        )
        if (
            any(getattr(cell, key) != getattr(expected_cell, key) for key in fields)
            or cell.backend_policy.to_dict() != expected_cell.backend_policy.to_dict()
        ):
            raise ValueError("formal collection topology or precision differs from the accepted runtime probe")
    expected_options = expected.options
    for name in (
        "vllm_max_model_len",
        "prefill_cudagraph_policy",
        "max_prefill_cudagraph_size",
        "gpu_memory_utilization",
        "enforce_eager",
        "warmup_iterations",
        "executor",
        "slurm_container_image",
        "slurm_container_mounts",
        "slurm_cpus_per_task",
        "slurm_cpu_bind",
    ):
        if getattr(options, name) != getattr(expected_options, name):
            raise ValueError(f"formal collection {name} differs from the accepted runtime probe")
    deployment = profile.select(
        model=model_path,
        system=system,
        backend=backend,
        backend_version=backend_version,
        tp_size=expected_cell.topology.tp,
        pp_size=expected_cell.topology.pp,
        attention_dp_size=expected_cell.topology.dp,
        moe_tp_size=expected_cell.topology.moe_tp,
        moe_ep_size=expected_cell.topology.moe_ep,
        cp_size=expected_cell.topology.cp,
    )
    limits = {
        "decode": (
            options.max_num_batched_tokens or deployment.resources.max_num_tokens,
            options.max_decode_batch_size or options.max_num_seqs or deployment.resources.max_batch_size,
        ),
        "prefill": (
            options.prefill_sampling.max_total_prefill_tokens,
            options.prefill_sampling.max_batch_size or options.max_num_seqs or deployment.resources.max_batch_size,
        ),
    }
    required_limits = (launch["collection"]["max_num_batched_tokens"], launch["collection"]["max_num_seqs"])
    if any(actual != required_limits for actual in limits.values()):
        raise ValueError("formal phase scheduler limits differ from the accepted runtime probe")
    wanted = probe_generator_overrides(launch)
    actual = copy.deepcopy(generator_overrides)
    if options.executor == "slurm":
        wanted.get("K8sConfig", {}).pop("k8s_image", None)
        actual.get("K8sConfig", {}).pop("k8s_image", None)
    if runner.with_kv_warmup_defaults(wanted) != runner.with_kv_warmup_defaults(actual):
        raise ValueError("formal collection deployment settings differ from the accepted runtime probe")


def prepare_collection_observations(plan, root: Path, parent_attempt_id: str):
    """Freeze one parent attempt shared by phase-specific collector attempts."""

    from .runtime_instrumentation import freeze_instrumentation, load_instrumentation

    destination = root / "runtime-observations" / parent_attempt_id / "instrumentation"
    bundle = plan.runtime_instrumentation
    if destination.exists():
        frozen = load_instrumentation(destination / "manifest.json")
        if frozen.sha256 != bundle.sha256:
            raise ValueError("formal collection frozen instrumentation hash mismatch")
        return frozen
    destination.parent.mkdir(parents=True, exist_ok=True)
    return freeze_instrumentation(bundle, destination)


def record_collection_observations(
    plan, root: Path, parent_attempt_id: str, cell, cell_attempt_id: str, cell_dir: Path, status: str
) -> Path:
    """Archive a cell's evidence before the ordinary collector can retry it."""

    index_path = root / "runtime-observations.json"
    index = (
        json.loads(index_path.read_text())
        if index_path.exists()
        else {"schema_version": INDEX_SCHEMA, "configurations": {}}
    )
    configuration = plan.runtime_configuration
    entry = index["configurations"].setdefault(
        configuration, {"launch": plan.runtime_launch, "active_attempt_id": parent_attempt_id, "attempts": []}
    )
    if entry["launch"] != plan.runtime_launch or entry["active_attempt_id"] != parent_attempt_id:
        raise ValueError("formal collection observation attempt identity mismatch")
    if not entry["attempts"]:
        entry["attempts"].append(
            {
                "attempt_id": parent_attempt_id,
                "status": "incomplete",
                "bundle": {
                    "manifest": f"runtime-observations/{parent_attempt_id}/instrumentation/manifest.json",
                    "sha256": plan.runtime_instrumentation.sha256,
                },
                "phases": {},
                "phase_attempts": {},
            }
        )
    attempt = entry["attempts"][-1]
    if attempt["attempt_id"] != parent_attempt_id or attempt["bundle"]["sha256"] != plan.runtime_instrumentation.sha256:
        raise ValueError("formal collection frozen observation identity mismatch")

    def verify_context(path: Path) -> None:
        expected = launch_context(plan, cell, configuration=configuration, attempt_id=parent_attempt_id)
        expected.update(collector_attempt_id=cell_attempt_id, cell_id=cell.cell_id)
        if read_json(path) != expected:
            raise ValueError("formal collection saved runtime context does not match its attempt")

    history = attempt["phase_attempts"].setdefault(cell.workload_kind, [])
    for previous in reversed(history):
        if previous["collector_attempt_id"] != cell_attempt_id or previous["status"] != status or status != "passed":
            continue
        if previous.get("launch_manifest") is not None:
            verify_context(_verified_artifact(root, previous["launch_manifest"]))
        elif status == "passed":
            raise ValueError("formal collection passed attempt requires its saved runtime context")
        for artifact in previous["artifacts"]:
            _verified_artifact(root, artifact)
        phase = previous
        break
    else:
        context = cell_dir / CONTEXT_FILENAME
        if context.is_file():
            verify_context(context)
        elif status == "passed":
            raise ValueError("formal collection passed attempt requires its saved runtime context")
        snapshot = root / "runtime-observations" / parent_attempt_id / cell.workload_kind / cell_attempt_id
        # A host can die after copying files and before committing the index.
        # Preserve that unindexed directory and publish a new complete snapshot.
        if snapshot.exists():
            snapshot = snapshot.with_name(snapshot.name + "-" + uuid.uuid4().hex)
        staging = snapshot.with_name(".staging-" + uuid.uuid4().hex)
        staging.mkdir(parents=True)
        for name in ("raw", "logs"):
            if (cell_dir / name).exists():
                shutil.copytree(cell_dir / name, staging / name, symlinks=True)
        for path in cell_dir.iterdir():
            if path.is_file():
                if path.is_symlink():
                    raise ValueError("formal collection launch files cannot be symlinks")
                shutil.copy2(path, staging / path.name)
        # Validate containment before exposing the snapshot to its index.
        observation_artifacts(root, staging)
        if (staging / CONTEXT_FILENAME).exists():
            verify_context(staging / CONTEXT_FILENAME)
        staging.rename(snapshot)
        phase = {
            "status": status,
            "collector_attempt_id": cell_attempt_id,
            "cell_id": cell.cell_id,
            "artifacts": observation_artifacts(root, snapshot),
        }
        if context.is_file():
            phase["launch_manifest"] = {
                "path": str((snapshot / CONTEXT_FILENAME).relative_to(root)),
                **runner._file_metadata(snapshot / CONTEXT_FILENAME),
            }
        history.append(phase)
    attempt["phases"][cell.workload_kind] = phase
    attempt["status"] = (
        "captured"
        if set(attempt["phases"]) == set(PHASES)
        and all(item["status"] == "passed" for item in attempt["phases"].values())
        else "incomplete"
    )
    runner._atomic_json(index_path, index)
    return index_path


def launch_context(plan, cell: FPMCell, *, configuration: str, attempt_id: str) -> dict[str, Any]:
    topology = cell.topology
    return {
        "schema_version": LAUNCH_SCHEMA,
        "configuration": configuration,
        "attempt_id": attempt_id,
        "phase": cell.workload_kind,
        "bundle_sha256": plan.runtime_instrumentation.sha256,
        "launch": plan.runtime_launch,
        "sampling": getattr(plan, "probe_sampling", None),
        "expected_ranks": {
            "workers": [
                {"dp_rank": dp, "tp_rank": tp, "pp_rank": pp}
                for dp in range(topology.dp)
                for tp in range(topology.tp)
                for pp in range(topology.pp)
            ],
            "schedulers": [{"dp_rank": dp} for dp in range(topology.dp)],
        },
    }


def stage_runtime_instrumentation(bundle, cell_dir: Path, context: dict[str, Any]) -> list[Path]:
    """Package data without importing it; activation occurs only in the worker."""

    for name, payload in _model_config_bytes(context["launch"]["model_config"]).items():
        (cell_dir / name).write_bytes(payload)
    archive = cell_dir / BUNDLE_FILENAME
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as target:
        target.writestr("manifest.json", json.dumps(bundle.manifest, sort_keys=True))
        for relative, expected in bundle.files.items():
            payload = (bundle.root / relative).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"instrumentation changed before staging: {relative}")
            target.writestr(relative, payload)
    context_path = cell_dir / CONTEXT_FILENAME
    runner._atomic_json(context_path, context)
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    # Archive bytes and entry names are verified before extraction. The host
    # never imports campaign Python modules, including during preview.
    setup = f"""
python3 - <<'AISIMULATE_INSTRUMENTATION'
import hashlib, pathlib, zipfile
archive = pathlib.Path('/tmp/fpm-bench/{BUNDLE_FILENAME}')
if hashlib.sha256(archive.read_bytes()).hexdigest() != {archive_sha256!r}:
    raise RuntimeError('runtime instrumentation archive hash mismatch')
root = pathlib.Path('/tmp/fpm-bench/runtime-instrumentation')
with zipfile.ZipFile(archive) as bundle:
    for name in bundle.namelist():
        path = pathlib.PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
            raise RuntimeError('unsafe runtime instrumentation path')
    bundle.extractall(root)
AISIMULATE_INSTRUMENTATION
export PYTHONPATH="/tmp/fpm-bench/runtime-instrumentation${{PYTHONPATH:+:${{PYTHONPATH}}}}"
"""
    with (cell_dir / runner.RUNTIME_ENV_FILENAME).open("a") as handle:
        handle.write(setup)
    return [archive, context_path]


def observation_artifacts(root: Path, phase_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for directory in (phase_dir, phase_dir / "raw", phase_dir / "logs"):
        paths = directory.iterdir() if directory == phase_dir else directory.rglob("*")
        for path in sorted(paths) if directory.exists() else ():
            if not path.is_file():
                continue
            if path.name == CONTEXT_FILENAME and path.parent == phase_dir:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("runtime artifacts must be regular files within the probe output")
            kind = "runtime-artifact"
            if path.suffix == ".json":
                try:
                    payload = json.loads(path.read_text())
                except (ValueError, UnicodeError):
                    payload = None
                if isinstance(payload, dict) and payload.get("schema_version") == "aisimulate-runtime-observation/v1":
                    kind = "observation"
            artifacts.append({"path": str(path.relative_to(root)), "kind": kind, **runner._file_metadata(path)})
    return artifacts


def _snapshot_probe_artifacts(root: Path, phase_dir: Path) -> list[dict[str, Any]]:
    """Keep imported evidence outside the executor's potentially live result mount."""

    references = observation_artifacts(root, phase_dir)
    snapshot = phase_dir / ("evidence-" + uuid.uuid4().hex)
    snapshot.mkdir()
    artifacts = []
    for reference in references:
        source = root / reference["path"]
        target = snapshot / source.relative_to(phase_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        artifacts.append(
            {"path": str(target.relative_to(root)), "kind": reference["kind"], **runner._file_metadata(target)}
        )
    return artifacts


def _recorded_attempt_valid(root: Path, attempt: dict[str, Any], bundle) -> bool:
    from .runtime_instrumentation import load_instrumentation

    if attempt.get("status") != "completed" or attempt.get("bundle", {}).get("sha256") != bundle.sha256:
        return False
    frozen = load_instrumentation(root / attempt["bundle"]["manifest"])
    if frozen.sha256 != bundle.sha256 or set(attempt.get("phases", {})) != set(PHASES):
        raise ValueError("saved probe attempt does not match its frozen instrumentation")
    for phase in attempt["phases"].values():
        for artifact in [phase["launch_manifest"], *phase["artifacts"]]:
            path = root / artifact["path"]
            if (
                not path.resolve().is_relative_to(root)
                or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]
            ):
                raise ValueError(f"saved probe artifact changed: {artifact['path']}")
    return True


def _verified_artifact(root: Path, reference: dict[str, Any]) -> Path:
    path = contained_file(root, reference["path"])
    digest = validate_sha256(reference.get("sha256"), "saved artifact")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"saved artifact SHA-256 mismatch: {reference['path']}")
    return path


def resolve_probe_cpu_policy(
    configuration: str, deployment: dict[str, Any], output_dir: str | Path, *, resume: bool
) -> dict[str, Any]:
    """Resolve omitted CPU fields before launch validation, retaining old absence."""
    from .config import resolve_slurm_cpu_policy

    deployment = copy.deepcopy(deployment)
    if deployment.get("executor") != "slurm":
        return deployment
    root = Path(output_dir).expanduser().resolve()
    index_path = root / "observations.json"
    if resume and index_path.exists():
        index = read_json(index_path)
        if index.get("schema_version") != INDEX_SCHEMA or not isinstance(index.get("configurations"), dict):
            raise ValueError("runtime probe observation index has an unsupported schema")
        existing = index["configurations"].get(configuration)
        if existing is not None:
            if not isinstance(existing, dict):
                raise ValueError("saved runtime probe configuration must be an object")
            saved = _required_mapping(_required_mapping(existing, "launch"), "deployment")
            inherited = {
                field: saved[field]
                for field in ("cpus_per_task", "cpu_bind")
                if deployment.get(field) is None and field in saved
            }
            if inherited:
                attempts = existing.get("attempts", [])
                if not isinstance(attempts, list) or any(not isinstance(attempt, dict) for attempt in attempts):
                    raise ValueError("saved runtime probe attempts must be a list of objects")
                phases = _required_mapping(attempts[-1], "phases") if attempts else {}
                if any(not isinstance(phase, dict) for phase in phases.values()):
                    raise ValueError("saved runtime probe phases must be objects")
                references = [phase["launch_manifest"] for phase in phases.values() if "launch_manifest" in phase]
                if not references or any(not isinstance(reference, dict) for reference in references):
                    raise ValueError("saved CPU policy has no SHA-bound runtime launch context")
                for reference in references:
                    context = read_json(_verified_artifact(root, reference))
                    if context.get("launch") != existing["launch"]:
                        raise ValueError("saved CPU policy differs from its archived runtime launch context")
            deployment.update(inherited)
            return deployment
    cpus, binding = resolve_slurm_cpu_policy(deployment.get("cpus_per_task"), deployment.get("cpu_bind"))
    deployment.update(cpus_per_task=cpus, cpu_bind=binding)
    return deployment


def _probe_configuration_slug(configuration: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", configuration)[:40] + "-" + _canonical_hash(configuration)[:8]


def _resource_manifest_reference(root: Path, plan, cell, directory: Path) -> dict[str, Any]:
    manifest = contained_file(root, str((directory / runner.FPM_MANIFEST_FILENAME).relative_to(root)))
    documents = runner._manifest_documents(manifest)
    runner._workload_document(documents)
    ownership = {
        "aiconfigurator.nvidia.com/owned-by": "fpm-forward-collector",
        "aiconfigurator.nvidia.com/plan": plan.sha256[:16],
        runner.FPM_CELL_LABEL: cell.cell_id,
    }
    for document in documents:
        labels = (document.get("metadata") or {}).get("labels") or {}
        if any(labels.get(key) != value for key, value in ownership.items()):
            raise ValueError("saved runtime resource does not match collector ownership")
    return {
        "path": str(manifest.relative_to(root)),
        **runner._file_metadata(manifest),
        "ownership": ownership,
        "resources": [list(runner._resource_identity(document)) for document in documents],
    }


def _cleanup_previous_attempt(root: Path, plan, attempt: dict[str, Any]) -> None:
    """Verify saved launch ownership before any recovery command is allowed."""

    from .runtime_instrumentation import load_instrumentation

    attempt_id = attempt["attempt_id"]
    if not isinstance(attempt_id, str) or not re.fullmatch(r"[0-9a-f]{32}", attempt_id):
        raise ValueError("invalid saved runtime probe attempt identity")
    directory = root / "attempts" / _probe_configuration_slug(plan.configuration) / attempt_id
    manifest_path = contained_file(root, attempt["bundle"]["manifest"])
    if manifest_path != directory / "instrumentation" / "manifest.json":
        raise ValueError("saved runtime bundle path does not belong to this attempt")
    bundle = load_instrumentation(manifest_path)
    if bundle.sha256 != attempt["bundle"]["sha256"]:
        raise ValueError("saved runtime instrumentation SHA-256 mismatch")
    previous_plan = build_runtime_probe_plan(plan.configuration, plan.launch, bundle)
    verified = []
    for cell in previous_plan.cells:
        phase = attempt.get("phases", {}).get(cell.workload_kind, {})
        if phase.get("status") not in {"running", "interrupted", "cleanup_failed"}:
            continue
        reference = phase.get("launch_manifest")
        if reference is None:
            raise ValueError("interrupted runtime probe has no saved launch manifest")
        context_path = _verified_artifact(root, reference)
        phase_dir = directory / cell.workload_kind
        if context_path != phase_dir / CONTEXT_FILENAME:
            raise ValueError("saved runtime launch path does not belong to this attempt/phase")
        expected = launch_context(previous_plan, cell, configuration=plan.configuration, attempt_id=attempt_id)
        if read_json(context_path) != expected:
            raise ValueError("saved runtime launch context does not match its attempt identity")
        resource_reference = phase.get("resource_manifest")
        if not isinstance(resource_reference, dict):
            raise ValueError("interrupted runtime probe has no saved resource manifest")
        manifest = _verified_artifact(root, resource_reference)
        if resource_reference != _resource_manifest_reference(root, previous_plan, cell, phase_dir):
            raise ValueError("saved runtime resource manifest identity mismatch")
        verified.append((cell, phase, phase_dir, manifest))
    for cell, phase, phase_dir, manifest in verified:
        resource = runner._cell_runner(previous_plan, cell, manifest, phase_dir)
        runner._salvage_artifacts(resource, cell.cell_id)
        resource.cleanup()
        phase["recovery_artifacts"] = _snapshot_probe_artifacts(root, phase_dir)


@contextmanager
def _probe_lock(root: Path):
    with (root / ".runtime-probe.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another runtime probe owns this output directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def probe_runtime(
    configurations: dict[str, dict[str, Any]],
    *,
    instrumentation=None,
    output_dir: str | Path,
    execute: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Preview or capture all selected exact tuples; preserve every attempt."""

    from .runtime_instrumentation import freeze_instrumentation, load_instrumentation

    if not isinstance(configurations, dict) or not configurations:
        raise ValueError("runtime probe requires at least one selected configuration")
    if instrumentation is None:
        from .bundled_instrumentation import bundled_instrumentation

        versions = {item.get("identity", {}).get("framework_version") for item in configurations.values()}
        if len(versions) != 1:
            raise ValueError("bundled instrumentation selection requires one pinned runtime version")
        version = next(iter(versions))
        instrumentation = bundled_instrumentation(version)
        if instrumentation is None:
            raise ValueError(
                f"no bundled runtime observer for vLLM {version}; provide campaign-local --instrumentation"
            )
    bundle = load_instrumentation(instrumentation) if isinstance(instrumentation, (str, Path)) else instrumentation
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not (root / "probe-plan.json").exists():
        raise ValueError("runtime probe output must be fresh or contain this probe's saved plan")
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    with _probe_lock(root), runner._sigterm_as_interrupt():
        index_path = root / "observations.json"
        if index_path.exists() and execute and not resume:
            raise ValueError("runtime probe observations already exist; use --resume or a fresh output directory")
        index = (
            json.loads(index_path.read_text())
            if index_path.exists()
            else {"schema_version": INDEX_SCHEMA, "configurations": {}}
        )
        if index.get("schema_version") != INDEX_SCHEMA or not isinstance(index.get("configurations"), dict):
            raise ValueError("runtime probe observation index has an unsupported schema")
        plans = {}
        for configuration, facts in configurations.items():
            try:
                if not isinstance(configuration, str) or not configuration:
                    raise ValueError("configuration labels must be nonempty strings")
                if facts.get("deployment", {}).get("executor") == "slurm":
                    facts = copy.deepcopy(facts)
                    facts["deployment"] = resolve_probe_cpu_policy(
                        configuration, facts["deployment"], root, resume=resume
                    )
                plan = build_runtime_probe_plan(configuration, facts, bundle)
                existing = index["configurations"].get(configuration)
                if existing is not None and existing["launch"] != plan.launch:
                    raise ValueError("launch settings changed; use a fresh probe output directory")
                plans[configuration] = plan
            except (OSError, TypeError, ValueError) as error:
                results[configuration] = {"status": "failed", "diagnostics": [str(error)]}
        runner._atomic_json(
            root / "probe-plan.json",
            {
                "schema_version": "aisimulate-runtime-probe-plan/v1",
                "configurations": {name: plan.launch for name, plan in plans.items()},
                "bundle_sha256": bundle.sha256,
            },
        )
        for configuration, plan in plans.items():
            slug = _probe_configuration_slug(configuration)
            result = results[configuration] = {
                "status": "preview" if not execute else "completed",
                "minimum_gpus": plan.cells[0].topology.total_gpus,
                "launch": plan.launch,
                "phases": {},
                "diagnostics": [],
            }
            entry = index["configurations"].setdefault(configuration, {"launch": plan.launch, "attempts": []})
            previous = entry["attempts"][-1] if entry["attempts"] else None
            if execute and resume and previous is not None:
                try:
                    if _recorded_attempt_valid(root, previous, bundle):
                        result.update(
                            status="completed",
                            resumed=True,
                            attempt_id=previous["attempt_id"],
                            phases=copy.deepcopy(previous["phases"]),
                        )
                        continue
                except (OSError, TypeError, KeyError, ValueError) as error:
                    result["diagnostics"].append(f"previous attempt cannot be reused: {error}")
            if execute and plan.options.executor == "slurm" and plan.options.slurm_cpus_per_task is None:
                result["status"] = "failed"
                result["diagnostics"].append(
                    "saved Slurm campaign has no frozen CPU policy; compatible recorded observations remain "
                    "readable, but new workers require a fresh campaign and smoke with explicit CPU settings"
                )
                continue
            if execute and resume and previous is not None:
                try:
                    _cleanup_previous_attempt(root, plan, previous)
                except Exception as error:
                    result["status"] = "failed"
                    result["diagnostics"].append(f"previous attempt cleanup failed: {error}")
                    continue
            attempt_id = uuid.uuid4().hex
            attempt_dir = root / ("attempts" if execute else "previews") / slug / attempt_id
            attempt_dir.mkdir(parents=True)
            frozen = freeze_instrumentation(bundle, attempt_dir / "instrumentation")
            attempt = {
                "attempt_id": attempt_id,
                "status": "running",
                "started_at": runner._utc_now(),
                "bundle": {"manifest": str(frozen.manifest_path.relative_to(root)), "sha256": frozen.sha256},
                "phases": {},
            }
            if execute:
                entry["attempts"].append(attempt)
                entry["active_attempt_id"] = attempt_id
                runner._atomic_json(index_path, index)
            for cell in plan.cells:
                phase_dir = attempt_dir / cell.workload_kind
                phase_dir.mkdir()
                phase = attempt["phases"][cell.workload_kind] = {"status": "running", "artifacts": []}
                resource = None
                try:
                    runner._render_cell(plan, cell, phase_dir, probe_generator_overrides(plan.launch))
                    context = launch_context(plan, cell, configuration=configuration, attempt_id=attempt_id)
                    extras = stage_runtime_instrumentation(frozen, phase_dir, context)
                    phase["launch_manifest"] = {
                        "path": str((phase_dir / CONTEXT_FILENAME).relative_to(root)),
                        **runner._file_metadata(phase_dir / CONTEXT_FILENAME),
                    }
                    phase["resource_manifest"] = _resource_manifest_reference(root, plan, cell, phase_dir)
                    phase["generator_request"] = str(phase_dir / "generator-request.json")
                    if execute:
                        runner._atomic_json(index_path, index)
                        manifest = phase_dir / runner.FPM_MANIFEST_FILENAME
                        resource = runner._cell_runner(plan, cell, manifest, phase_dir)
                        resource.cleanup()
                        resource.apply()
                        pods = resource.wait_ready(runner._expected_nodes(manifest))
                        runtime = Path(runner.__file__).parent / "runtime"
                        resource.stage(
                            pods,
                            [
                                phase_dir / runner.FPM_RUN_SCRIPT_FILENAME,
                                phase_dir / runner.FPM_ENV_FILENAME,
                                phase_dir / runner.RUNTIME_ENV_FILENAME,
                                runtime / "fpm_exec.sh",
                                runtime / "preflight.py",
                                *([runtime / "fpm_memory_observer.py"] if plan.options.executor == "slurm" else []),
                                *extras,
                            ],
                        )
                        resource.prepare_attempt(
                            pods,
                            cell_id=cell.cell_id,
                            plan_sha256=plan.sha256,
                            attempt_id=attempt_id,
                            expected_backend_version=plan.capability.aic_database_version,
                        )
                        resource.execute(pods)
                        resource.collect(pods, require_benchmark=False)
                    phase["status"] = "completed" if execute else "preview"
                except KeyboardInterrupt:
                    phase["status"] = "interrupted"
                    result["status"] = attempt["status"] = "interrupted"
                    if resource is not None:
                        runner._salvage_artifacts(resource, cell.cell_id)
                    raise
                except Exception as error:
                    phase.update(status="failed", error=str(error), error_type=type(error).__name__)
                    result["status"] = "failed"
                    result["diagnostics"].append(f"{cell.workload_kind}: {error}")
                    if resource is not None:
                        runner._salvage_artifacts(resource, cell.cell_id)
                finally:
                    if resource is not None:
                        try:
                            resource.cleanup()
                        except Exception as error:
                            phase.update(status="cleanup_failed", cleanup_error=str(error))
                            result["status"] = "failed"
                            result["diagnostics"].append(f"{cell.workload_kind} cleanup: {error}")
                    try:
                        phase["artifacts"] = _snapshot_probe_artifacts(root, phase_dir)
                    except (OSError, ValueError) as error:
                        phase.update(status="failed", capture_error=str(error))
                        result["status"] = "failed"
                        result["diagnostics"].append(f"{cell.workload_kind} evidence capture: {error}")
                    result["phases"][cell.workload_kind] = copy.deepcopy(phase)
                    if execute:
                        attempt["status"] = result["status"] if result["status"] != "completed" else "running"
                        runner._atomic_json(index_path, index)
            attempt.update(status=result["status"], completed_at=runner._utc_now())
            result["attempt_id"] = attempt_id
            if execute:
                runner._atomic_json(index_path, index)
    failed = sum(result["status"] == "failed" for result in results.values())
    status = "failed" if failed == len(results) else "partial" if failed else "completed" if execute else "preview"
    return {
        "status": status,
        "observations_path": str(root / "observations.json") if execute else None,
        "configurations": results,
    }
