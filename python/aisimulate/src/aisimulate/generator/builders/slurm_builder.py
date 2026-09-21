# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lower resolved generator parameters to self-contained Pyxis/Slurm jobs.

The backend CLI/engine templates remain the source of engine configuration.
Only process orchestration and allocation belong here. V1 supports colocated
agg or P/D workers on one NVIDIA GPU node; unsupported topologies fail closed.
"""

from __future__ import annotations

import copy
import json
import re
import shlex
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..dynamo_features import frontend_cli_args_from_dyn_config, kvbm_env_from_dyn_config, vllm_worker_role_args
from ..rendering.schemas import apply_defaults

_TEMPLATES = Path(__file__).resolve().parents[1] / "config/backend_templates/slurm"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OWNED_ENV = {
    "CUDA_VISIBLE_DEVICES",
    "ETCD_ENDPOINTS",
    "NATS_SERVER",
    "DYN_NAMESPACE",
    "DYN_SYSTEM_PORT",
    "DYN_HTTP_PORT",
    "DYN_DISCOVERY_BACKEND",
    "DYN_STORE_KV",
    "DYN_REQUEST_PLANE",
    "DYN_TRTLLM_OVERRIDE_ENGINE_ARGS",
}
_TOPOLOGY_ENV = {
    "vllm": {
        "VLLM_DP_SIZE": "1",
        "VLLM_DP_RANK": "0",
        "VLLM_DP_RANK_LOCAL": "0",
        "VLLM_DP_MASTER_IP": "127.0.0.1",
        "VLLM_DP_MASTER_PORT": "0",
    },
    "sglang": {"DYN_SGL_DISAGG_CONFIG": "", "DYN_SGL_DISAGG_CONFIG_KEY": ""},
}
_TOPOLOGY_OPTIONS = {
    "vllm": {
        "-tp",
        "-pp",
        "-dp",
        "-dpl",
        "-dpn",
        "-dpr",
        "-dph",
        "-dpe",
        "-dpb",
        "-dpm",
        "-dcp",
        "-pcp",
        "-ep",
        "-n",
        "-r",
        "--data-parallel-size",
        "--data-parallel-size-local",
        "--data-parallel-rank",
        "--data-parallel-start-rank",
        "--data-parallel-hybrid-lb",
        "--data-parallel-external-lb",
        "--data-parallel-backend",
        "--data-parallel-multi-port-external-lb",
        "--device-ids",
        "--decode-context-parallel-size",
        "--prefill-context-parallel-size",
        "--enable-expert-parallel",
        "--no-enable-expert-parallel",
        "--enable-elastic-ep",
        "--distributed-executor-backend",
    },
    "sglang": {
        "--tp-size",
        "--pp-size",
        "--data-parallel-size",
        "--dp-size",
        "--expert-parallel-size",
        "--ep-size",
        "--ep",
        "--attention-context-parallel-size",
        "--attn-cp-size",
        "--moe-data-parallel-size",
        "--moe-dp-size",
        "--moe-dense-tp-size",
        "--enable-dp-attention",
        "--base-gpu-id",
        "--gpu-id-step",
        "--decode-context-parallel-size",
        "--dcp-size",
        "--elastic-ep-backend",
        "--elastic-ep-join-mode",
        "--elastic-ep-join-rank-offset",
        "--elastic-ep-initial-size",
        "--max-ep-size",
        "--elastic-ep-rejoin",
        "--disagg-config",
        "--disagg-config-key",
    },
    "trtllm": {
        "--expert-parallel-size",
        "--enable-attention-dp",
        "--no-enable-attention-dp",
        "--gpus-per-node",
        "--override-engine-args",
    },
}


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _directive(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:+/-]+", value):
        raise ValueError(f"SlurmConfig.{name} must be a nonempty Slurm option value without whitespace")
    return value


def _worker_command(context: dict, params: dict, backend: str, role: str) -> tuple[list[str], dict]:
    service = context["ServiceConfig"]
    dyn = context.get("DynConfig") or {}
    model_flag = "--model" if backend == "vllm" else "--model-path"
    command = [
        "python3",
        "-m",
        f"dynamo.{backend}",
        model_flag,
        str(service["model_path"]),
        "--served-model-name",
        str(service.get("served_model_name") or service["model_path"]),
    ]
    command.extend(context.get(f"{role}_cli_args_list") or [])
    worker_env = {
        item["name"]: item["value"]
        for item in kvbm_env_from_dyn_config(dyn if role != "decode" else {}, backend=backend)
    }
    kvbm = bool(worker_env)
    # Backend environment fallbacks must not replace the structured topology.
    # vLLM consults these DP defaults only when engine args do not request DP > 1.
    worker_env.update(_TOPOLOGY_ENV.get(backend, {}))
    if backend == "trtllm":
        command.extend(["--extra-engine-args", f"{role}_config.yaml", "--gpus-per-node", str(context[f"{role}_gpu"])])
        # An inherited Dynamo override must not replace the generated topology.
        worker_env["DYN_TRTLLM_OVERRIDE_ENGINE_ARGS"] = ""
        if kvbm:
            command.extend(["--connector", "kvbm"])
    if role != "agg":
        command.extend(
            vllm_worker_role_args(role, params.get("generator_dynamo_version"))
            if backend == "vllm"
            else ["--disaggregation-mode", role]
        )
    if backend == "vllm" and (role != "agg" or kvbm):
        nixl = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        kvbm_connector = {
            "kv_connector": "DynamoConnector",
            "kv_connector_module_path": "kvbm.vllm_integration.connector",
            "kv_role": "kv_both",
        }
        connector = nixl if role != "agg" else kvbm_connector
        if kvbm and role == "prefill":
            connector = {
                "kv_connector": "PdConnector",
                "kv_role": "kv_both",
                "kv_connector_module_path": "kvbm.vllm_integration.connector",
                "kv_connector_extra_config": {"connectors": [kvbm_connector, nixl]},
            }
        command.extend(["--kv-transfer-config", json.dumps(connector)])
    if dyn.get("enable_router") or dyn.get("router_mode") == "kv" or dyn.get("router_config"):
        if backend == "trtllm":
            command.append("--publish-events-and-metrics")
        else:
            # The runtime substitutes a separately reserved port for each worker.
            event = {"publisher": "zmq", "topic": "kv-events", "endpoint": "tcp://*:@EVENT_PORT@"}
            if backend == "vllm":
                event["enable_kv_cache_events"] = True
            command.extend(["--kv-events-config", json.dumps(event)])
    if backend == "sglang" and role != "agg":
        # Bootstrap servers must not contend when several prefill workers share a host.
        command.extend(["--disaggregation-bootstrap-port", "@BOOTSTRAP_PORT@"])
    extra = (params.get("params", {}).get(role) or {}).get("extra_cli_args", [])
    if not isinstance(extra, list) or not all(isinstance(token, str) for token in extra):
        raise ValueError(f"Workers.{role}.extra_cli_args must be a list of strings")
    owned = {
        "--model",
        "--model-path",
        "--served-model-name",
        "--disaggregation-mode",
        "--extra-engine-args",
        "--nnodes",
        "--node-rank",
        "--store-kv",
        "--request-plane",
        "--disaggregation-bootstrap-port",
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        "--config",
    }
    owned.update(_TOPOLOGY_OPTIONS[backend])
    for token in extra:
        option = token.split("=", 1)[0].replace("_", "-")
        if not option.startswith("-") or option in {"-", "--"}:
            continue
        # argparse accepts unambiguous prefixes and attached one-letter aliases
        # (e.g. -n2); vLLM also normalizes underscores in option names.
        if (
            any(flag.startswith(option) for flag in owned)
            or any(len(flag) == 2 and option.startswith(flag) for flag in owned)
            or (backend == "trtllm" and option.startswith("--trtllm."))
        ):
            raise ValueError(
                f"Workers.{role}.extra_cli_args overrides a Slurm-owned launch option: {option}; "
                "set topology and engine configuration through the structured generator configuration"
            )
    command.extend(extra)
    return command, worker_env


def build_slurm_artifacts(
    context: dict, params: dict, backend: str, rendered: dict, benchmark_env: Environment
) -> dict[str, str]:
    if backend not in {"vllm", "sglang", "trtllm"}:
        raise ValueError(f"Unsupported Slurm backend: {backend}")
    cfg = apply_defaults("SlurmConfig", params.get("SlurmConfig") or {}, backend=backend)
    dyn = context.get("DynConfig") or {}
    mode = dyn.get("mode")
    if mode not in {"agg", "disagg"}:
        raise ValueError("Slurm requires DynConfig.mode=agg or disagg")
    if (params.get("params") or {}).get("encode") or dyn.get("planner_config"):
        raise ValueError("Slurm does not yet support encode workers or Dynamo Planner")
    if (params.get("NodeConfig") or {}).get("system_name") == "b60":
        raise ValueError("Slurm currently supports NVIDIA GPUs only")
    for key in ("account", "partition", "job_name", "time", "memory"):
        cfg[key] = _directive(cfg.get(key), key)
    for key in ("cpus_per_task", "startup_timeout", "benchmark_timeout", "benchmark_rounds"):
        cfg[key] = _positive(cfg.get(key), f"SlurmConfig.{key}")
    image = cfg.get("container_image")
    if not isinstance(image, str) or not image.strip() or any(c in image for c in "\r\n\0"):
        raise ValueError("SlurmConfig.container_image is required (Pyxis image URI or squashfs path)")
    mounts = cfg.get("container_mounts") or []
    if not isinstance(mounts, list) or not all(
        isinstance(m, str) and m and not any(c in m for c in ",\r\n\0") for m in mounts
    ):
        raise ValueError("SlurmConfig.container_mounts must be a list of Pyxis source:destination[:ro] mounts")
    extra_env = cfg.get("env") or {}
    if not isinstance(extra_env, dict) or any(
        not isinstance(k, str) or not _ENV_NAME.fullmatch(k) or not isinstance(v, str) or "\0" in v
        for k, v in extra_env.items()
    ):
        raise ValueError("SlurmConfig.env must map environment variable names to strings")
    if _OWNED_ENV.union(_TOPOLOGY_ENV.get(backend, {})).intersection(extra_env):
        raise ValueError("SlurmConfig.env must not override allocation, discovery or service-port variables")
    concurrencies = cfg.get("benchmark_concurrency")
    if not isinstance(concurrencies, list) or not concurrencies:
        raise ValueError("SlurmConfig.benchmark_concurrency must be a nonempty list")
    for value in concurrencies:
        _positive(value, "SlurmConfig.benchmark_concurrency entry")
    roles = ["agg"] if mode == "agg" else ["prefill", "decode"]
    workers = []
    total_gpus = 0
    for role in roles:
        if not (params.get("params") or {}).get(role):
            raise ValueError(f"Slurm {mode} topology requires Workers.{role}")
        count = _positive(context.get(f"{role}_workers"), f"WorkerConfig.{role}_workers")
        gpus = _positive(context.get(f"{role}_gpu"), f"Workers.{role}.gpus_per_worker")
        if backend == "trtllm" and not rendered.get(f"extra_engine_args_{role}.yaml"):
            raise ValueError(f"Missing TRT-LLM engine config for {role}")
        for index in range(count):
            command, worker_env = _worker_command(context, params, backend, role)
            workers.append(
                {
                    "name": f"{role}-{index}",
                    "argv": command,
                    "env": worker_env,
                    "gpu_offset": total_gpus,
                    "gpu_count": gpus,
                }
            )
            total_gpus += gpus
    capacity = _positive((params.get("NodeConfig") or {}).get("num_gpus_per_node", 8), "NodeConfig.num_gpus_per_node")
    if total_gpus > capacity:
        raise ValueError(f"Slurm V1 supports one node: topology needs {total_gpus} GPUs, node capacity is {capacity}")
    service = context["ServiceConfig"]
    if not service.get("model_path"):
        raise ValueError("Slurm requires ServiceConfig.model_path")
    if service.get("include_frontend") is False:
        raise ValueError("Slurm jobs require their own frontend (ServiceConfig.include_frontend=true)")
    port = _positive(service.get("port"), "ServiceConfig.port")
    if port > 65535:
        raise ValueError("ServiceConfig.port must be <= 65535")
    frontend = ["python3", "-m", "dynamo.frontend", *frontend_cli_args_from_dyn_config(dyn, service)]
    spec = {
        "backend": backend,
        "mode": mode,
        "gpus": total_gpus,
        "port": port,
        "model": service.get("served_model_name") or service["model_path"],
        "frontend": frontend,
        "workers": workers,
        "env": extra_env,
        "startup_timeout": cfg["startup_timeout"],
        "benchmark_timeout": cfg["benchmark_timeout"],
        "benchmark_concurrency": concurrencies,
        "benchmark_rounds": cfg["benchmark_rounds"],
    }
    env = Environment(
        loader=FileSystemLoader(_TEMPLATES), undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True
    )
    env.filters["shellquote"] = lambda value: shlex.quote(str(value))
    template_ctx = {"cfg": cfg, "gpus": total_gpus, "mounts": ",".join(mounts)}
    artifacts = {
        "deployment.json": json.dumps(spec, indent=2) + "\n",
        "slurm_runtime.py": Path(__file__).with_name("slurm_runtime.py").read_text(),
    }
    for filename in ("environment.sh", "submit.sh"):
        artifacts[filename] = env.get_template(f"{filename}.j2").render(**template_ctx)
    for operation, filename in (("serve", "deploy.sbatch"), ("benchmark", "benchmark.sbatch")):
        artifacts[filename] = env.get_template("job.sbatch.j2").render(**template_ctx, operation=operation)
    bench_context = copy.deepcopy(context)
    bench_context.setdefault("BenchConfig", {})["model"] = spec["model"]
    artifacts["bench_run.sh"] = benchmark_env.get_template("bench_run.sh.j2").render(**bench_context)
    for role in roles:
        if f"extra_engine_args_{role}.yaml" in rendered:
            artifacts[f"{role}_config.yaml"] = rendered[f"extra_engine_args_{role}.yaml"]
    return artifacts
