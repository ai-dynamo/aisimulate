# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze a GLM attention collection attempt and render its Slurm script.

CPU-only. ``prepare`` writes ``manifest.json`` (pinned runtime, checkpoint,
geometry, the frozen plan and the attempt's set selection) plus ``run.sbatch``
and ``dryrun.sbatch`` into a fresh attempt directory; the caller copies the
directory to shared storage and submits it. One attempt = one (backend,
checkpoint, TP) deployment and one context class (one server context limit)
on one exclusive node.
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
    context_class_sets,
    geometry,
    representative_layer_is_uniform,
    sha256_json,
    unaligned_targets,
)
from collector.glm53flash_attention_tokens import spec as input_token_spec

_SMOKE_COMMON = {
    "layer_id": 3,
    "warmup": 3,
    "iterations": 10,
    "max_step_tokens": 8192,
    "max_context": 131072,
    "max_model_len": 131079,
}
_SMOKE_DECODE = {"batch_sizes": [1, 32], "sequence_lengths": {"1": [2048, 2049, 16384], "32": [4096]}}
# Small smoke plans. The validation geometries are the in-serving nsys points:
# vLLM (stock serve, prefix ON, per request prefix/new tokens, all multiples of
# 4) and SGLang (the clean-truth holdout points profiled in
# sglang-prefill-gap-profile: B1 T32 KV128, B1 T2042, B1 T8189, B4 T1024
# KV48128, B32 T320, B32 T1024 KV139264, B32 T8000, B32 T8192 KV3137536).
# "long" holds one ~1M prefill and the 1M decode, plus two regular-length
# decodes that expose any dependence on the server context limit.
SMOKE_SWEEPS = {
    "validation-vllm": {
        **_SMOKE_COMMON,
        "prefill": {
            "batch_sizes": [1, 4, 32],
            "query_lengths": {"1": [32, 2048, 8192], "4": [256], "32": [8]},
            "prefix_lengths": {"1": {"32": [128], "2048": [0], "8192": [0]}, "4": [12032], "32": [0]},
        },
        "decode": _SMOKE_DECODE,
    },
    "validation-sglang": {
        **_SMOKE_COMMON,
        "prefill": {
            "batch_sizes": [1, 4, 32],
            "query_lengths": {"1": [32, 2042, 8189], "4": [256], "32": [10, 32, 250, 256]},
            "prefix_lengths": {
                "1": {"32": [128], "2042": [0], "8189": [0]},
                "4": [12032],
                "32": {"10": [0], "32": [4352], "250": [0], "256": [98048]},
            },
        },
        "decode": _SMOKE_DECODE,
    },
    "long": {
        **_SMOKE_COMMON,
        "prefill": {"batch_sizes": [], "query_lengths": {}, "prefix_lengths": []},
        "decode": {"batch_sizes": [], "sequence_lengths": {}},
        "long_context": {
            "max_context": 1048575,
            "max_model_len": 1048576,
            "prefill": {"batch_sizes": [1], "query_lengths": {"1": [1024]}, "prefix_lengths": [1032192]},
            "decode": {"batch_sizes": [1], "sequence_lengths": {"1": [2048, 16384, 1048575]}},
        },
    },
}
SGLANG_ARGS = (
    "--kv-cache-dtype fp8_e4m3 --moe-runner-backend auto --disable-radix-cache "
    "--chunked-prefill-size 8192 --mem-fraction-static 0.82 "
    "--max-running-requests 32 --cuda-graph-bs-decode " + " ".join(str(b) for b in range(1, 33))
)
# Serving CUDA-graph policy (pgraph smoke v4, clean-truth server flags): vLLM
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
VLLM_ENV = {
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_MNNVL_ENABLE": "1",
    "NCCL_NVLS_ENABLE": "1",
    "NCCL_P2P_LEVEL": "NVL",
    "VLLM_USE_NCCL_SYMM_MEM": "1",
}
# One GB300 holds ~279 GiB; the FP8 checkpoint's safetensors are 305.8 GiB
# (zai-org/GLM-5.3-Flash@eb9eb208), the NVFP4 checkpoint's 190.4 GiB. TP1 is
# therefore an NVFP4-only deployment (campaign decision; capacity fact).
TP1_CHECKPOINTS = ("nvfp4",)


def prepare(args) -> Path:
    import yaml

    attempt = Path(args.attempt)
    if attempt.exists() and any(attempt.iterdir()):
        raise SystemExit(f"{attempt} is not a fresh attempt directory")
    config = json.loads(Path(args.config).read_text())
    if args.smoke:
        sweep = json.loads(json.dumps(SMOKE_SWEEPS[args.smoke]))
        if args.layer_id is not None:
            # Smoke-only cross-check that another sparse-MLA layer times like
            # the representative one; full attempts always use the YAML layer.
            sweep["layer_id"] = args.layer_id
        if args.warmup is not None:
            # Smoke-only study of the warmup length (GPU clock steady state).
            sweep["warmup"] = args.warmup
    else:
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
    body = {
        "schema_version": 2,
        "op": OP_NAME,
        "role": "smoke" if args.smoke else "full",
        "geometry": geometry(config, args.backend, args.checkpoint, args.tp),
        "checkpoint_revision": CHECKPOINTS[args.checkpoint][1],
        "checkpoint_model_id": CHECKPOINTS[args.checkpoint][0],
        "framework_version": RUNTIME_VERSIONS[args.backend],
        "runtime_digest": RUNTIME_IMAGES[args.backend],
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
        # Allocator policy only (no kernel change): the 16384 MiB split the
        # qualified SGLang FP8 TP2 FPM campaign uses against fragmentation of
        # the ragged IndexPool MQA-logits buffer at long batched context.
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
    suffix = (f"-smoke-{args.smoke}" if args.smoke else "") + (f"-{args.tag}" if args.tag else "")
    suffix += f"-l{args.layer_id}" if args.layer_id is not None else ""
    suffix += f"-w{args.warmup}" if args.warmup is not None else ""
    job = f"glm53-v031c-attn-{args.backend}-{args.checkpoint}-tp{args.tp}{suffix}"
    container = f"{args.remote_attempt}"
    runner = f"collector.{args.backend}.glm53flash_attention_runner"
    common = "--manifest /results/manifest.json --output /results/raw"
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
        env.update(VLLM_ENV)
        command = f"python3 -m {runner} {common} --model-path {model}"
        if args.vllm_gpu_memory_utilization is not None:
            # Capacity only: leaves headroom for the per-target module graphs.
            command += f" --gpu-memory-utilization {args.vllm_gpu_memory_utilization}"
    else:
        sglang_args = f"{SGLANG_ARGS} --context-length {max_model_len}"
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
#SBATCH --gres=gpu:4
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
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--config", required=True, help="checkpoint config.json (pinned revision)")
    parser.add_argument("--sweep", default="collector/cases/base_ops/glm53flash_attention.yaml")
    parser.add_argument("--smoke", choices=sorted(SMOKE_SWEEPS))
    parser.add_argument("--context-class", choices=("regular", "long"), default="regular")
    parser.add_argument("--layer-id", type=int, help="smoke-only representative-layer cross-check")
    parser.add_argument("--warmup", type=int, help="smoke-only warmup repetitions per target")
    parser.add_argument("--attempt", required=True, help="local fresh attempt directory")
    parser.add_argument("--remote-attempt", required=True)
    parser.add_argument("--remote-source", required=True, help="shared-storage copy of python/aisimulate")
    parser.add_argument("--remote-model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--tag", default="", help="job-name suffix of a split attempt")
    parser.add_argument("--account", default="coreai_comparch_inferencex")
    parser.add_argument("--partition", default="batch")
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--allocator-max-split-mb", type=int, default=None)
    parser.add_argument("--sglang-mem-fraction", type=float, default=None)
    parser.add_argument("--only-sets", nargs="+", help="split attempt: measure only these plan sets")
    parser.add_argument("--skip-sets", nargs="+", help="split attempt: measure every other set of the class")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    args = parser.parse_args()
    if (args.layer_id is not None or args.warmup is not None) and not args.smoke:
        parser.error("--layer-id and --warmup are smoke-only")
    if args.tp == 1 and args.checkpoint not in TP1_CHECKPOINTS:
        parser.error("TP1 is NVFP4-only: the FP8 checkpoint's weights exceed one GB300")
    print(prepare(args))


if __name__ == "__main__":
    main()
