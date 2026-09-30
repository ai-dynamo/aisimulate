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
        sweep = SMOKE_SWEEP
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
    }
    manifest = {**body, "manifest_sha256": sha256_json(body)}
    (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    job = f"glm53-sa-w4-{args.backend}-{args.checkpoint}-tp{args.tp}{'-smoke' if args.smoke else ''}"
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
    if args.backend == "vllm":
        mounts.append(f"{args.remote_tail}:/opt/glm53flash-candidate:ro")
        env["PYTHONPATH"] = "/opt/glm53flash-candidate:/workspace"
        env.update(VLLM_ENV)
        command = f"python3 -m {runner} {common} --model-path {model}"
    else:
        command = f"python3 -m {runner} {common} --model-path {model} --tp-size {args.tp} {SGLANG_ARGS}"
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
    return attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(RUNTIME_VERSIONS), required=True)
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--tp", type=int, choices=(2, 4), required=True)
    parser.add_argument("--config", required=True, help="checkpoint config.json (pinned revision)")
    parser.add_argument("--sweep", default="collector/cases/base_ops/glm53flash_attention.yaml")
    parser.add_argument("--smoke", action="store_true")
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
    args = parser.parse_args()
    if args.backend == "vllm" and not args.remote_tail:
        parser.error("vLLM requires the reviewed glm53tail overlay (--remote-tail)")
    print(prepare(args))


if __name__ == "__main__":
    main()
