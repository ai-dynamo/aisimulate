# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze a GLM attention collection attempt and render its Slurm script.

CPU-only. ``prepare`` writes ``manifest.json`` (pinned runtime, checkpoint,
geometry, the frozen plan and the attempt's set selection) plus ``run.sbatch``
into a fresh attempt directory; the caller copies the
directory to shared storage and submits it. One attempt = one (backend,
checkpoint, TP) deployment and one context class (one server context limit)
on one exclusive node of the target system (``--system``: its YAML gives the
GPUs per node, the per-GPU memory and whether the node sits in a multi-node
NVLink domain).
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from collector.glm53flash_attention_contract import (
    ARCHES,
    CHECKPOINTS,
    KV_CACHE_DTYPES,
    OP_NAME,
    RUNTIME_VERSIONS,
    SERVING_KV_CACHE_DTYPE,
    build_plan,
    context_class_sets,
    geometry,
    representative_layer_is_uniform,
    runtime_image,
    sha256_json,
    unaligned_targets,
)
from collector.glm53flash_attention_tokens import spec as input_token_spec

SGLANG_ARGS = (
    "--moe-runner-backend auto --disable-radix-cache "
    "--chunked-prefill-size 8192 --mem-fraction-static 0.82 "
    "--max-running-requests 32 --cuda-graph-bs-decode " + " ".join(str(b) for b in range(1, 33))
)
# Serving CUDA-graph policy (the serving server flags): vLLM
# keeps its default FULL_AND_PIECEWISE mode with breakable graphs and these 62
# capture sizes (its default 11 sizes plus SGLang's 58 prefill buckets);
# SGLang captures breakable prefill graphs up to 8192 tokens (its own buckets)
# and full decode graphs for bs 1..32.
SGLANG_PREFILL_GRAPH_ARGS = "--cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 8192"
VLLM_PREFILL_GRAPH_CAPTURE_SIZES = (
    [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]
    + list(range(80, 257, 16))
    + list(range(288, 513, 32))
    + list(range(576, 1025, 64))
    + list(range(1280, 4097, 256))
    + list(range(4608, 8193, 512))
)
# vLLM serving environment on every system: NVLink SHARP and vLLM's NCCL
# symmetric memory stay on for HGX (H100/H200/B300) as on GB300.
VLLM_ENV = {
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_NVLS_ENABLE": "1",
    "NCCL_P2P_LEVEL": "NVL",
    "VLLM_USE_NCCL_SYMM_MEM": "1",
}
# Multi-node NVLink (MNNVL) only where the NVLink domain spans nodes
# (GB200/GB300 NVL72); HGX nodes are one NVLink domain each.
VLLM_NVL72_ENV = {"NCCL_MNNVL_ENABLE": "1"}
# Checkpoint safetensors payload (zai-org/GLM-5.3-Flash@eb9eb208: 305.8 GiB;
# nvidia/GLM-5.3-Flash-NVFP4@09b04e5e: 190.4 GiB). A deployment whose
# per-rank weight share exceeds one GPU's memory cannot load (capacity fact);
# every other TP is admitted and its fit is observed at runtime.
CHECKPOINT_GIB = {"fp8": 305.8, "nvfp4": 190.4}
GIB = 1 << 30


def system_spec(system: str, systems_dir: str | None = None) -> dict:
    """GPUs per node, per-GPU memory and NVLink domain of a system YAML."""
    from collector.model_cases import default_system_spec_path, load_yaml_file

    path = Path(systems_dir) / f"{system}.yaml" if systems_dir else default_system_spec_path(system)
    if not path.is_file():
        raise SystemExit(f"no system YAML {path}")
    data = load_yaml_file(path)
    node = data.get("node") or {}
    per_node = int(node["num_gpus_per_node"])
    return {
        "gpus_per_node": per_node,
        "mem_capacity": int(data["gpu"]["mem_capacity"]),
        "sm_version": data["gpu"].get("sm_version"),
        # An NVLink domain wider than one node (GB200/GB300 NVL72 racks).
        "multi_node_nvlink": int(node.get("num_gpus_per_rack") or per_node) > per_node,
    }


def check_deployment(args, spec: dict) -> None:
    """Capacity and topology facts only; nothing about expected kernels."""
    if args.tp > args.gpus_per_node:
        raise SystemExit(f"TP{args.tp} exceeds the {args.gpus_per_node} GPUs of one {args.system} node")
    if args.kv_cache_dtype is None:
        # No silent default where it cannot be served: the stock SM90 runtimes
        # serve this model's sparse-MLA KV only in BF16, so Hopper attempts
        # must name the KV dtype; elsewhere the default stays FP8.
        if spec["sm_version"] is not None and 90 <= int(spec["sm_version"]) < 100:
            raise SystemExit(
                f"{args.system} (SM{spec['sm_version']}) needs an explicit --kv-cache-dtype: the stock "
                "SM90 runtimes serve GLM-5.3-Flash with bfloat16 KV only (pass --kv-cache-dtype bfloat16)"
            )
        args.kv_cache_dtype = "fp8"
    if args.kv_cache_dtype not in KV_CACHE_DTYPES[args.backend]:
        raise SystemExit(f"{args.backend} GLM attention serves KV {KV_CACHE_DTYPES[args.backend]}")
    per_rank = CHECKPOINT_GIB[args.checkpoint] * GIB / args.tp
    if per_rank > spec["mem_capacity"]:
        raise SystemExit(
            f"TP{args.tp} {args.checkpoint} weights ({per_rank / GIB:.1f} GiB per rank) exceed one "
            f"{args.system} GPU ({spec['mem_capacity'] / GIB:.1f} GiB)"
        )


def serving_env(backend: str, spec: dict) -> dict:
    if backend != "vllm":
        return {}
    return {**VLLM_ENV, **(VLLM_NVL72_ENV if spec["multi_node_nvlink"] else {})}


def prepare(args) -> Path:
    import yaml

    spec = system_spec(args.system, args.systems_dir)
    if args.gpus_per_node is None:
        args.gpus_per_node = spec["gpus_per_node"]
    check_deployment(args, spec)
    attempt = Path(args.attempt)
    if attempt.exists() and any(attempt.iterdir()):
        raise SystemExit(f"{attempt} is not a fresh attempt directory")
    config = json.loads(Path(args.config).read_text())
    sweep = yaml.safe_load(Path(args.sweep).read_text())["common_case_values"][OP_NAME]
    representative_layer_is_uniform(config, sweep["layer_id"], args.checkpoint)
    plan = build_plan(sweep)
    selection = context_class_sets(plan, args.context_class)
    if not selection:
        raise SystemExit(f"the plan has no {args.context_class} sets")
    if args.only_sets or args.skip_sets:
        unknown = (set(args.only_sets or ()) | set(args.skip_sets or ())) - set(selection)
        chosen = set(args.only_sets or selection) - set(args.skip_sets or ())
        if unknown or not chosen:
            raise SystemExit(f"invalid set selection: unknown {sorted(unknown)}, selected {len(chosen)}")
        selection = [s for s in selection if s in chosen]
    selected = [s for s in plan["sets"] if s["set_id"] in set(selection)]
    if args.backend == "vllm":
        bad = unaligned_targets({"sets": selected})
        if bad:
            # Stock vLLM 0.31.0 must never run an unaligned chunk start.
            raise SystemExit(f"kpool_align4: vLLM plan has unaligned prefill targets {bad[:6]}")
    max_model_len = selected[0]["max_model_len"]
    env_serving = serving_env(args.backend, spec)
    body = {
        "schema_version": 2,
        "op": OP_NAME,
        "geometry": geometry(config, args.backend, args.checkpoint, args.tp, args.kv_cache_dtype),
        "checkpoint_revision": CHECKPOINTS[args.checkpoint][1],
        "checkpoint_model_id": CHECKPOINTS[args.checkpoint][0],
        "framework_version": RUNTIME_VERSIONS[args.backend],
        "runtime_digest": runtime_image(args.backend, args.arch),
        "runtime_arch": args.arch,
        "system": args.system,
        "gpus_per_node": args.gpus_per_node,
        # Exported framework environment (MNNVL excluded on single-node
        # NVLink domains).
        "serving_env": env_serving,
        "layer_id": sweep["layer_id"],
        "sweep": sweep,
        "plan": plan,
        # Request token ids come from the in-repo seeded generator.
        "input_tokens": input_token_spec(plan),
        "source_commit": args.source_commit,
        "context_class": args.context_class,
        # Server context limit (vLLM --max-model-len, SGLang --context-length)
        # of the selected context class; capacity only.
        "max_model_len": max_model_len,
        # Allocator policy only (no kernel change), e.g. a 16384 MiB split
        # against fragmentation of the ragged IndexPool MQA-logits buffer at
        # long batched context.
        "allocator_max_split_size_mb": args.allocator_max_split_mb,
        "serving_graph": (
            {"vllm_cudagraph_capture_sizes": VLLM_PREFILL_GRAPH_CAPTURE_SIZES}
            if args.backend == "vllm"
            else {"sglang_args": SGLANG_PREFILL_GRAPH_ARGS}
        ),
    }
    if len(selection) != len(plan["sets"]):
        # Split attempt: one deployment's plan measured by several attempts
        # whose set selections finalize unions exactly once.
        body["only_sets"] = sorted(selection)
    if args.sglang_mem_fraction is not None:
        body["sglang_mem_fraction_static"] = args.sglang_mem_fraction
    if args.vllm_gpu_memory_utilization is not None:
        body["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    attempt.mkdir(parents=True, exist_ok=True)
    manifest = {**body, "manifest_sha256": sha256_json(body)}
    (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    suffix = f"-{args.tag}" if args.tag else ""
    job = f"glm53-attn-{args.backend}-{args.checkpoint}-tp{args.tp}{suffix}"
    container = f"{args.remote_attempt}"
    runner = f"collector.{args.backend}.glm53flash_attention_runner"
    common = "--manifest /results/manifest.json --output /results/raw"
    model = f"/models/{args.checkpoint}"
    mounts = [
        f"{args.remote_source}:/workspace:ro",
        f"{args.remote_model}:{model}:ro",
        f"{container}:/results",
    ]
    # PYTHONNOUSERSITE: a host user site-packages must never shadow the
    # pinned image's framework packages.
    env = {"PYTHONPATH": "/workspace", "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"}
    if args.allocator_max_split_mb is not None:
        env["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{args.allocator_max_split_mb}"
    env.update(env_serving)
    if args.backend == "vllm":
        command = f"python3 -m {runner} {common} --model-path {model}"
        if args.vllm_gpu_memory_utilization is not None:
            # Capacity only: leaves headroom for the per-target module graphs.
            command += f" --gpu-memory-utilization {args.vllm_gpu_memory_utilization}"
    else:
        kv = SERVING_KV_CACHE_DTYPE[args.kv_cache_dtype]
        sglang_args = f"--kv-cache-dtype {kv} {SGLANG_ARGS} --context-length {max_model_len}"
        if args.sglang_mem_fraction is not None:
            # Capacity only: a smaller static KV pool leaves headroom for the
            # per-target module graphs beside the framework's prefill graph
            # pool; the KV still holds every planned request.
            sglang_args = sglang_args.replace(
                "--mem-fraction-static 0.82", f"--mem-fraction-static {args.sglang_mem_fraction}"
            )
        command = (
            f"python3 -m {runner} {common} --model-path {model} --tp-size {args.tp} "
            f"{sglang_args} {SGLANG_PREFILL_GRAPH_ARGS}"
        )
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    script = f"""#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --nodes=1
#SBATCH --gres=gpu:{args.gpus_per_node}
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time={args.time}
#SBATCH --output={container}/slurm-%j.out
set -euo pipefail
cd {container}
sha256sum manifest.json > inputs.sha256
nvidia-smi -q > nvidia-smi-before.txt || true
srun --container-image={args.image} --container-mounts={",".join(mounts)} --no-container-mount-home \\
  --container-workdir=/workspace bash -c "export {exports}; \\
  python3 -c 'import torch,sys; print(torch.__version__, torch.cuda.get_device_name(0), file=sys.stderr)'; \\
  exec {command}"
"""
    (attempt / "run.sbatch").write_text(script)
    return attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(RUNTIME_VERSIONS), required=True)
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--system", required=True, help="system YAML name, e.g. h200_sxm, b300_sxm, gb200")
    parser.add_argument("--systems-dir", default=None, help="directory holding <system>.yaml (default: packaged)")
    parser.add_argument("--arch", choices=ARCHES, required=True, help="CPU architecture of the pinned image")
    parser.add_argument(
        "--gpus-per-node", type=int, default=None, help="Slurm GPUs per node (default: the system YAML)"
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("fp8", "bfloat16"),
        default=None,
        help="sparse-MLA latent KV precision of the table (default fp8; required on SM90, "
        "whose runtimes serve only bfloat16)",
    )
    parser.add_argument("--config", required=True, help="checkpoint config.json (pinned revision)")
    parser.add_argument("--sweep", default="collector/cases/base_ops/glm53flash_attention.yaml")
    parser.add_argument("--context-class", choices=("regular", "long"), default="regular")
    parser.add_argument("--attempt", required=True, help="local fresh attempt directory")
    parser.add_argument("--remote-attempt", required=True)
    parser.add_argument("--remote-source", required=True, help="shared-storage copy of python/aisimulate")
    parser.add_argument("--remote-model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--tag", default="", help="job-name suffix of a split attempt")
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--allocator-max-split-mb", type=int, default=None)
    parser.add_argument("--sglang-mem-fraction", type=float, default=None)
    parser.add_argument("--only-sets", nargs="+", help="split attempt: measure only these plan sets")
    parser.add_argument("--skip-sets", nargs="+", help="split attempt: measure every other set of the class")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    args = parser.parse_args()
    print(prepare(args))


if __name__ == "__main__":
    main()
