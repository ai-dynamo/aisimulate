# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect observed execution separately from requested benchmark settings."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .config import with_kv_warmup_defaults
from .cpu_affinity import inspect_cpu_affinity
from .native_artifact import NativeCollection, _rank_artifacts
from .planner import FPMCell, FPMCollectionPlan, _canonical_hash
from .runtime.fpm_memory_observer import EXECUTION_SUPPORTED_VERSIONS
from .runtime_memory import _dtype, _validate_graph, _validate_precision, _validate_revision

_GRAPH_FIELDS = {"mode", "cudagraph_mode", "cudagraph_capture_sizes", "max_cudagraph_capture_size"}
_CONFIG_TYPES = {
    "model_config": {
        "model": (str,),
        "revision": (str, type(None)),
        "dtype": (str,),
        "quantization": (str, type(None)),
        "max_model_len": (int,),
        "enforce_eager": (bool,),
    },
    "cache_config": {
        "cache_dtype": (str,),
        "gpu_memory_utilization": (int, float),
        "enable_prefix_caching": (bool,),
        "kv_cache_memory_bytes": (int, type(None)),
        "num_gpu_blocks_override": (int, type(None)),
    },
    "scheduler_config": {"max_num_batched_tokens": (int,), "max_num_seqs": (int,), "async_scheduling": (bool,)},
    "parallel_config": {
        "tensor_parallel_size": (int,),
        "pipeline_parallel_size": (int,),
        "data_parallel_size": (int,),
        "data_parallel_size_local": (int,),
        "data_parallel_external_lb": (bool,),
        "enable_expert_parallel": (bool,),
        "enable_eplb": (bool,),
        "decode_context_parallel_size": (int,),
        "prefill_context_parallel_size": (int,),
    },
    "kernel_config": {"moe_backend": (str,)},
}


class _MissingExecutionEvidence(ValueError):
    pass


def _effective_launch(plan: FPMCollectionPlan, cell_dir: Path) -> dict[str, Any]:
    """Read the generator's literal argv array; never evaluate archived shell."""
    try:
        for name in ("fpm_env.sh", "collector-runtime-env.sh"):
            (cell_dir / name).read_bytes()
        deployment = json.loads((cell_dir.parent.parent / "generator-overrides.json").read_text())
        request = json.loads((cell_dir / "generator-request.json").read_text())
        script = (cell_dir / "run.sh").read_text()
    except FileNotFoundError as error:
        raise _MissingExecutionEvidence(f"frozen generated launch is missing: {error.filename}") from error
    if not isinstance(deployment, dict):
        raise ValueError("archived deployment inputs must be an object")
    mount = deployment.get("K8sConfig", {})
    if not isinstance(mount, dict):
        raise ValueError("archived K8sConfig must be an object")
    if _canonical_hash(with_kv_warmup_defaults(deployment)) != plan.generator_config_sha256:
        raise ValueError("archived deployment inputs differ from the source plan")
    expected_model = plan.model_path
    if mount.get("k8s_model_path_in_pvc"):
        if not mount.get("k8s_pvc_mount_path"):
            raise ValueError("checkpoint mount has no explicit container mount path")
        expected_model = str(PurePosixPath(mount["k8s_pvc_mount_path"]) / mount["k8s_model_path_in_pvc"])
    if not isinstance(request, dict):
        raise ValueError("generated request must be an object")
    service = request.get("ServiceConfig")
    if not isinstance(service, dict):
        raise ValueError("generated ServiceConfig must be an object")
    if service.get("model_path") != expected_model:
        raise ValueError("generated model path differs from the plan and its explicit checkpoint mount")
    commands = re.findall(r"^engine_command=\((.*)\)$", script, re.MULTILINE)
    if len(commands) != 1:
        raise _MissingExecutionEvidence("frozen launch does not contain one literal generated engine_command array")
    argv = shlex.split(commands[0])
    if argv[:3] != ["python3", "-m", "dynamo.vllm"]:
        raise _MissingExecutionEvidence("frozen launch uses an unsupported engine command")
    # The current generator emits one shell-quoted token per argument. Read
    # only these literal tokens, including argparse's --option=value form.
    flags = {}
    for index, token in enumerate(argv):
        if not token.startswith("--"):
            continue
        name, separator, value = token.partition("=")
        flags.pop(name, None)
        flags[name] = (
            value
            if separator
            else argv[index + 1]
            if index + 1 < len(argv) and not argv[index + 1].startswith("--")
            else True
        )
    actual_model = flags.get("--model")
    if isinstance(actual_model, str) and ("$" in actual_model or "`" in actual_model):
        raise _MissingExecutionEvidence("generated model path is not a literal runtime setting")
    if actual_model != expected_model:
        raise ValueError("generated engine model differs from the frozen plan and checkpoint mount")
    return flags


def _inspect_config(payload, cell, plan, flags, missing, failures):
    config = payload.get("resolved_config")
    if not isinstance(config, dict):
        missing.append("resolved runtime configuration is unreported")
        config = {}
    valid = {}
    for section, fields in _CONFIG_TYPES.items():
        values = config.get(section)
        if not isinstance(values, dict):
            missing.append(f"runtime {section} is unreported")
            continue
        valid[section] = {}
        for name, types in fields.items():
            label = f"{section}.{name}"
            if name not in values:
                missing.append(f"runtime {label} is unreported")
            elif type(values[name]) not in types or (isinstance(values[name], str) and not values[name]):
                failures.append(f"runtime {label} has an invalid type or empty value")
            elif type(values[name]) in (int, float) and (not math.isfinite(values[name]) or values[name] <= 0):
                failures.append(f"runtime {label} must be positive and finite")
            else:
                valid[section][name] = values[name]
    model, cache, parallel = (valid.get(key, {}) for key in ("model_config", "cache_config", "parallel_config"))
    if cache.get("gpu_memory_utilization", 1) > 1:
        failures.append("runtime GPU memory utilization must not exceed one")
    if parallel.get("data_parallel_size_local", 1) > parallel.get("data_parallel_size", 1):
        failures.append("runtime local DP size exceeds global DP size")
    expected = {
        "model_config": {"enforce_eager": plan.options.enforce_eager},
        "parallel_config": {
            "tensor_parallel_size": cell.topology.tp,
            "pipeline_parallel_size": cell.topology.pp,
            "data_parallel_size": cell.topology.dp,
            "decode_context_parallel_size": cell.topology.cp,
            "prefill_context_parallel_size": 1,
            "enable_expert_parallel": cell.topology.moe_ep > 1,
            "enable_eplb": bool(cell.backend_policy.aic_fields.get("enable_eplb", False)),
        },
    }
    if plan.options.vllm_max_model_len > 0:
        expected["model_config"]["max_model_len"] = plan.options.vllm_max_model_len
    if plan.options.gpu_memory_utilization is not None:
        expected["cache_config"] = {"gpu_memory_utilization": plan.options.gpu_memory_utilization}
    for section, fields in expected.items():
        for name, value in fields.items():
            if name in valid.get(section, {}) and valid[section][name] != value:
                failures.append(f"runtime {section}.{name} differs from the frozen plan")
    launch_fields = {
        "--model": ("model_config", "model", str),
        "--max-model-len": ("model_config", "max_model_len", int),
        "--gpu-memory-utilization": ("cache_config", "gpu_memory_utilization", float),
        "--kv-cache-memory-bytes": ("cache_config", "kv_cache_memory_bytes", int),
        "--num-gpu-blocks-override": ("cache_config", "num_gpu_blocks_override", int),
        "--max-num-batched-tokens": ("scheduler_config", "max_num_batched_tokens", int),
        "--max-num-seqs": ("scheduler_config", "max_num_seqs", int),
        "--tensor-parallel-size": ("parallel_config", "tensor_parallel_size", int),
        "--pipeline-parallel-size": ("parallel_config", "pipeline_parallel_size", int),
        "--data-parallel-size": ("parallel_config", "data_parallel_size", int),
        "--decode-context-parallel-size": ("parallel_config", "decode_context_parallel_size", int),
        "--prefill-context-parallel-size": ("parallel_config", "prefill_context_parallel_size", int),
        "--moe-backend": ("kernel_config", "moe_backend", str),
    }
    if flags is not None:
        for name in ("kv_cache_memory_bytes", "num_gpu_blocks_override"):
            if f"--{name.replace('_', '-')}" not in flags and cache.get(name) is not None:
                failures.append(f"runtime cache_config.{name} differs from automatic generated cache sizing")
        for flag, (section, name, convert) in launch_fields.items():
            if flag not in flags or name not in valid.get(section, {}):
                continue
            raw = flags[flag]
            if type(raw) is not str or "$" in raw or "`" in raw:
                missing.append(f"generated {flag} is not a literal runtime setting")
                continue
            try:
                value = convert(raw)
            except ValueError:
                failures.append(f"generated {flag} is not a valid runtime setting")
                continue
            if (flag == "--max-model-len" and value == -1) or (flag == "--moe-backend" and value == "auto"):
                continue
            if valid[section][name] != value:
                failures.append(f"runtime {section}.{name} differs from generated {flag}")
        for flag, section, name in (
            ("async-scheduling", "scheduler_config", "async_scheduling"),
            ("enable-prefix-caching", "cache_config", "enable_prefix_caching"),
            ("enforce-eager", "model_config", "enforce_eager"),
        ):
            choices = [key == f"--{flag}" for key in flags if key in {f"--{flag}", f"--no-{flag}"}]
            if choices and name in valid.get(section, {}) and valid[section][name] != choices[-1]:
                failures.append(f"runtime {section}.{name} differs from generated flags")
        for flag, actual in (("--dtype", model.get("dtype")), ("--kv-cache-dtype", cache.get("cache_dtype"))):
            if flag in flags and flags[flag] != "auto" and actual is not None and _dtype(actual) != _dtype(flags[flag]):
                failures.append(f"runtime precision differs from generated {flag}")
        if "--quantization" in flags and "quantization" in model and model["quantization"] != flags["--quantization"]:
            failures.append("runtime quantization differs from generated --quantization")
    expected_revision = plan.fpm_profile.model_revision if plan.fpm_profile is not None else None
    revisions = {value for value in (expected_revision, (flags or {}).get("--revision")) if value is not None}
    if not revisions:
        missing.append("source has no pinned model revision")
    raw_model = config.get("model_config")
    loaded = raw_model.get("loaded_config_commit_hash") if isinstance(raw_model, dict) else None
    if loaded is not None and (not isinstance(loaded, str) or re.fullmatch(r"[0-9a-fA-F]{40}", loaded) is None):
        failures.append("runtime loaded model config commit hash is malformed")
    for revision in revisions:
        observed = model.get("revision")
        immutable = isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision) is not None
        if observed == revision or (immutable and isinstance(loaded, str) and loaded.lower() == revision.lower()):
            pass
        elif observed is None:
            missing.append(f"runtime model revision has no evidence for {revision!r}")
        else:
            failures.append("runtime model revision differs from the pinned source revision")
        if immutable and isinstance(config.get("model_config"), dict):
            try:
                _validate_revision(config, revision)
            except ValueError as error:
                failures.append(str(error))
    if "dtype" in model and "cache_dtype" in cache:
        actual = model["dtype"] if cache["cache_dtype"] == "auto" else cache["cache_dtype"]
        if _dtype(actual) != cell.kv_cache_dtype:
            failures.append("runtime KV precision differs from the planned identity")
    method = model.get("quantization")
    quant = config.get("quantization_config")
    if "quantization_config" not in config:
        missing.append("runtime quantization_config is unreported")
    elif "quantization" in model and "dtype" in model:
        required = {"type"}
        if method in {"modelopt", "modelopt_fp4"}:
            required.add("quant_method")
        elif method == "fp8":
            required.update({"activation_scheme", "weight_block_size"})
        if method is not None and quant is None:
            failures.append("runtime quantization method contradicts its absent quantization configuration")
        elif method is not None and (not isinstance(quant, dict) or not required.issubset(quant)):
            missing.append("runtime quantization family configuration is incomplete")
        elif method not in {None, "modelopt", "modelopt_fp4", "fp8"}:
            missing.append(f"runtime quantization mapping is unaudited: {method}")
        else:
            try:
                _validate_precision(config, cell)
            except ValueError as error:
                failures.append(str(error))
    graphs = []
    for label, graph in (
        ("graph_config", payload.get("graph_config")),
        ("compilation_config", config.get("compilation_config")),
    ):
        if not isinstance(graph, dict) or not _GRAPH_FIELDS.issubset(graph):
            missing.append(f"runtime {label} is incomplete")
            continue
        if "enforce_eager" not in model or "max_num_batched_tokens" not in valid.get("scheduler_config", {}):
            continue
        try:
            graphs.append(
                _validate_graph(
                    graph,
                    eager=model["enforce_eager"],
                    tokens=valid["scheduler_config"]["max_num_batched_tokens"],
                    allow_unused_captures=True,
                )
            )
        except ValueError as error:
            failures.append(str(error))
    if len(graphs) == 2 and graphs[0] != graphs[1]:
        failures.append("worker graph_config contradicts its resolved_config.compilation_config")


def file_evidence(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def inspect_execution_evidence(
    cell: FPMCell, raw_root: Path, collection: NativeCollection, *, plan: FPMCollectionPlan
) -> dict[str, Any]:
    """Require actual worker observations; native point expectations stay labelled.

    The caller first validates native point/rank/timing integrity. Missing
    execution observations leave qualification incomplete without invalidating
    historical timing artifacts or manufacturing a requested backend identity.
    """
    missing = []
    failures = []
    flags = None
    try:
        flags = _effective_launch(plan, raw_root.parent)
    except _MissingExecutionEvidence as error:
        missing.append(str(error))
    except (ValueError, TypeError) as error:
        failures.append(str(error))
    if collection.backend_version != plan.capability.aic_database_version:
        failures.append("observed backend version differs from the frozen plan")
    if collection.backend_version not in EXECUTION_SUPPORTED_VERSIONS:
        missing.append(f"execution configuration inspection is unaudited for {collection.backend_version}")
    workers = []
    ranks = set()
    for path in sorted(raw_root.glob("**/fpm-execution-worker-*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"runtime execution evidence must be an object: {path}")
        provenance = payload.get("collector_provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError(f"runtime execution collector_provenance must be an object: {path}")
        if (
            payload.get("schema_name") != "aisimulate_fpm_runtime_execution"
            or payload.get("schema_version") != 1
            or provenance.get("plan_sha256") != plan.sha256
            or provenance.get("cell_id") != cell.cell_id
            or provenance.get("attempt_id") != collection.collector_attempt_id
            or payload.get("backend_version") != collection.backend_version
        ):
            raise ValueError(f"runtime execution evidence has a different plan, cell, attempt or version: {path}")
        rank = tuple(payload.get(key) for key in ("dp_rank", "tp_rank", "pp_rank"))
        if any(type(value) is not int or value < 0 for value in rank) or rank in ranks:
            raise ValueError(f"runtime execution worker ranks are invalid or duplicated: {path}")
        ranks.add(rank)
        item = {**payload, "source": file_evidence(path)}
        workers.append(item)
        if payload.get("status") != "observed":
            missing.append(f"worker {rank}: {payload.get('error', 'execution observation unresolved')}")
            continue
        groups = payload.get("attention_groups")
        if (
            not isinstance(groups, list)
            or not groups
            or any(
                not isinstance(group, dict)
                or not isinstance(group.get("backend_class"), str)
                or not group["backend_class"]
                or not isinstance(group.get("layer_names"), list)
                or not group["layer_names"]
                or any(not isinstance(name, str) or not name for name in group["layer_names"])
                for group in groups
            )
        ):
            failures.append(f"runtime attention group observation is malformed: {path}")
        _inspect_config(payload, cell, plan, flags, missing, failures)
    expected = {
        (dp, tp, pp)
        for dp in range(cell.topology.dp)
        for tp in range(cell.topology.tp)
        for pp in range(cell.topology.pp)
    }
    if ranks - expected:
        raise ValueError("runtime execution observations contain unexpected worker ranks")
    if ranks != expected:
        missing.append(f"worker execution observations missing for ranks {sorted(expected - ranks)}")
    native_graphs = []
    for path, payload in _rank_artifacts(raw_root):
        native_graphs.append(
            {"dp_rank": payload["dp"]["rank"], "config": payload.get("cudagraph"), "source": file_evidence(path)}
        )
        if not isinstance(payload.get("cudagraph"), dict):
            missing.append(f"native graph configuration unreported for {path.name}")
    regimes = Counter(measurement.kv_seed_regime or "unreported" for measurement in collection.points)
    if "unreported" in regimes:
        missing.append("per-point KV initialization regime is unreported")
    cpu_nodes, cpu_manifest, cpu_geometry_missing, cpu_geometry_failure = None, None, None, None
    if getattr(plan.options, "slurm_cpus_per_task", None) is not None:
        from .runner import FPM_MANIFEST_FILENAME, _expected_nodes

        manifest = raw_root.parent / FPM_MANIFEST_FILENAME
        try:
            cpu_manifest = file_evidence(manifest)
            cpu_nodes = _expected_nodes(manifest)
        except OSError as error:
            cpu_geometry_missing = str(error)
        except (TypeError, ValueError, KeyError, yaml.YAMLError) as error:
            cpu_geometry_failure = str(error)
    cpu = inspect_cpu_affinity(cell, raw_root, collection, plan=plan, expected_nodes=cpu_nodes)
    cpu["launch_geometry_source"] = cpu_manifest
    if cpu_geometry_missing is not None:
        cpu["missing_evidence"].append(f"generated launch geometry: {cpu_geometry_missing}")
    if cpu_geometry_failure is not None:
        cpu["failures"].append(f"generated launch geometry: {cpu_geometry_failure}")
        cpu["status"] = "failed"
    if cpu["policy_required"]:
        missing.extend(f"CPU affinity: {message}" for message in cpu["missing_evidence"])
        failures.extend(f"CPU affinity: {message}" for message in cpu["failures"])
    return {
        "status": "failed" if failures else "incomplete" if missing else "qualified",
        "missing_evidence": sorted(set(missing)),
        "failures": sorted(set(failures)),
        "observed_workers": workers,
        "native_graph_config": native_graphs,
        "kv_seed_regime_counts": dict(regimes),
        "cpu_affinity": cpu,
        "per_point_dispatch": "unreported",
        "scope": "initialized attention backends and resolved graph configuration; no per-point dispatch trace",
    }
