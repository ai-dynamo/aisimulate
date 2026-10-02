# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze a GLM attention collection attempt and render its Slurm script.

CPU-only. ``prepare`` writes ``manifest.json`` (pinned runtime, checkpoint,
geometry and the frozen plan) plus ``run.sbatch`` into a fresh attempt
directory; the caller copies the directory to shared storage and submits it.
One attempt = one (backend, checkpoint, TP) deployment on one exclusive node.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from collector.glm53flash_attention_contract import (
    CHECKPOINTS,
    OP_NAME,
    RUNTIME_IMAGES,
    RUNTIME_VERSIONS,
    build_plan,
    geometry,
    representative_layer_is_uniform,
    sha256_json,
)

SMOKE_SWEEP = {
    "layer_id": 3,
    "warmup": 3,
    "iterations": 10,
    "max_step_tokens": 8192,
    "max_context": 131072,
    "prefill": {
        "batch_sizes": [1, 4],
        "query_lengths": {"1": [256, 2048], "4": [64, 2048]},
        "prefix_lengths": [0, 1024, 8192, 32768],
    },
    "decode": {
        "batch_sizes": [1, 4, 32],
        "sequence_lengths": {"1": [1024, 2049, 16384], "4": [2048, 2052, 8192], "32": [4096]},
    },
}
SGLANG_ARGS = (
    "--kv-cache-dtype fp8_e4m3 --moe-runner-backend auto --disable-radix-cache "
    "--context-length 131079 --chunked-prefill-size 8192 --mem-fraction-static 0.82 "
    "--max-running-requests 32 --cuda-graph-bs-decode " + " ".join(str(b) for b in range(1, 33))
)
# Serving prefill CUDA-graph policy adopted 2026-10-01 (pgraph smoke v4):
# vLLM keeps its default FULL_AND_PIECEWISE mode with breakable graphs and these
# 62 capture sizes (its default 11 sizes plus SGLang's 58 prefill buckets);
# SGLang captures breakable prefill graphs up to 8192 tokens (its own buckets).
SGLANG_PREFILL_GRAPH_ARGS = "--cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 8192"
VLLM_PREFILL_GRAPH_CAPTURE_SIZES = (
    [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]
    + list(range(80, 257, 16))
    + list(range(288, 513, 32))
    + list(range(576, 1025, 64))
    + list(range(1280, 4097, 256))
    + list(range(4608, 8193, 512))
)
VLLM_ENV = {
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_MNNVL_ENABLE": "1",
    "NCCL_NVLS_ENABLE": "1",
    "NCCL_P2P_LEVEL": "NVL",
    "VLLM_USE_NCCL_SYMM_MEM": "1",
}


def prepare(args) -> Path:
    import yaml

    attempt = Path(args.attempt)
    if attempt.exists() and any(attempt.iterdir()):
        raise SystemExit(f"{attempt} is not a fresh attempt directory")
    attempt.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    if args.smoke:
        sweep = dict(SMOKE_SWEEP)
        if args.layer_id is not None:
            # Smoke-only cross-check that another sparse-MLA layer times like
            # the representative one; full attempts always use the YAML layer.
            sweep["layer_id"] = args.layer_id
    else:
        sweep = yaml.safe_load(Path(args.sweep).read_text())["common_case_values"][OP_NAME]
    representative_layer_is_uniform(config, sweep["layer_id"], args.checkpoint)
    body = {
        "schema_version": 1,
        "op": OP_NAME,
        "role": "smoke" if args.smoke else "full",
        "geometry": geometry(config, args.backend, args.checkpoint, args.tp),
        "checkpoint_revision": CHECKPOINTS[args.checkpoint][1],
        "checkpoint_model_id": CHECKPOINTS[args.checkpoint][0],
        "framework_version": RUNTIME_VERSIONS[args.backend],
        "runtime_digest": RUNTIME_IMAGES[args.backend],
        "layer_id": sweep["layer_id"],
        "sweep": sweep,
        "plan": build_plan(sweep),
        "source_commit": args.source_commit,
        # Allocator policy only (no kernel change): the same 16384 MiB split the
        # qualified SGLang FP8 TP2 FPM campaign uses against fragmentation of
        # the ragged IndexPool MQA-logits buffer at long batched context.
        "allocator_max_split_size_mb": args.allocator_max_split_mb,
    }
    if args.sglang_mem_fraction is not None:
        body["sglang_mem_fraction_static"] = args.sglang_mem_fraction
    if args.prefill_graph:
        # Revision 2: re-collect prefill only, under the serving prefill graphs.
        body["phases"] = ["context"]
        body["prefill_execution"] = "framework_breakable_cuda_graph"
        body["serving_graph"] = (
            {"vllm_cudagraph_capture_sizes": VLLM_PREFILL_GRAPH_CAPTURE_SIZES}
            if args.backend == "vllm"
            else {"sglang_args": SGLANG_PREFILL_GRAPH_ARGS}
        )
    manifest = {**body, "manifest_sha256": sha256_json(body)}
    (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    suffix = ("-smoke" if args.smoke else "") + (f"-l{args.layer_id}" if args.layer_id is not None else "")
    prefix = "glm53-sa-w4g" if args.prefill_graph else "glm53-sa-w4"
    job = f"{prefix}-{args.backend}-{args.checkpoint}-tp{args.tp}{suffix}"
    container = f"{args.remote_attempt}"
    runner = f"collector.{args.backend}.glm53flash_attention_runner"
    common = "--manifest /results/manifest.json --output /results/raw --corpus /results/corpus.txt"
    model = f"/models/{args.checkpoint}"
    mounts = [
        f"{args.remote_source}:/workspace:ro",
        f"{args.remote_model}:{model}:ro",
        f"{container}:/results",
    ]
    env = {"PYTHONPATH": "/workspace", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"}
    if args.allocator_max_split_mb is not None:
        env["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{args.allocator_max_split_mb}"
    if args.backend == "vllm":
        mounts.append(f"{args.remote_tail}:/opt/glm53flash-candidate:ro")
        env["PYTHONPATH"] = "/opt/glm53flash-candidate:/workspace"
        env.update(VLLM_ENV)
        command = f"python3 -m {runner} {common} --model-path {model}"
    else:
        graph_args = f" {SGLANG_PREFILL_GRAPH_ARGS}" if args.prefill_graph else ""
        sglang_args = SGLANG_ARGS
        if args.sglang_mem_fraction is not None:
            # Capacity only: a smaller static KV pool leaves headroom for the
            # per-target module graphs beside the framework's prefill graph
            # pool; the KV still holds every planned request.
            sglang_args = sglang_args.replace(
                "--mem-fraction-static 0.82", f"--mem-fraction-static {args.sglang_mem_fraction}"
            )
        command = f"python3 -m {runner} {common} --model-path {model} --tp-size {args.tp} {sglang_args}{graph_args}"
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    script = f"""#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time={args.time}
#SBATCH --output={container}/slurm-%j.out
set -euo pipefail
cd {container}
sha256sum manifest.json corpus.txt > inputs.sha256
nvidia-smi -q > nvidia-smi-before.txt || true
srun --container-image={args.image} --container-mounts={",".join(mounts)} --no-container-mount-home \\
  --container-workdir=/workspace bash -c "export {exports}; \\
  python3 -c 'import torch,sys; print(torch.__version__, torch.cuda.get_device_name(0), file=sys.stderr)'; \\
  exec {command}"
"""
    (attempt / "run.sbatch").write_text(script)
    # CPU dry run in the same container and mounts: resolves the framework,
    # manifest, checkpoint geometry and native arguments, then exits before
    # any CUDA work, so a GPU allocation is never spent on a driver error.
    dryrun = f"""#!/bin/bash
#SBATCH --job-name={job}-dryrun
#SBATCH --account={args.account}
#SBATCH --partition=cpu
#SBATCH --qos=cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output={container}/dryrun-%j.out
set -euo pipefail
srun --container-image={args.image} --container-mounts={",".join(mounts)} --no-container-mount-home \\
  --container-workdir=/workspace bash -c "export {exports}; exec {command} --dry-run"
"""
    (attempt / "dryrun.sbatch").write_text(dryrun)
    return attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(RUNTIME_VERSIONS), required=True)
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--tp", type=int, choices=(2, 4), required=True)
    parser.add_argument("--config", required=True, help="checkpoint config.json (pinned revision)")
    parser.add_argument("--sweep", default="collector/cases/base_ops/glm53flash_attention.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--layer-id", type=int, help="smoke-only representative-layer cross-check")
    parser.add_argument("--attempt", required=True, help="local fresh attempt directory")
    parser.add_argument("--remote-attempt", required=True)
    parser.add_argument("--remote-source", required=True, help="shared-storage copy of python/aisimulate")
    parser.add_argument("--remote-model", required=True)
    parser.add_argument("--remote-tail", default="")
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--account", default="coreai_comparch_inferencex")
    parser.add_argument("--partition", default="batch")
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--allocator-max-split-mb", type=int, default=None)
    parser.add_argument("--prefill-graph", action="store_true", help="revision 2: prefill under serving graphs")
    parser.add_argument("--sglang-mem-fraction", type=float, default=None)
    args = parser.parse_args()
    if args.layer_id is not None and not args.smoke:
        parser.error("--layer-id is a smoke-only cross-check")
    if args.backend == "vllm" and not args.remote_tail:
        parser.error("vLLM requires the reviewed glm53tail overlay (--remote-tail)")
    print(prepare(args))


if __name__ == "__main__":
    main()
