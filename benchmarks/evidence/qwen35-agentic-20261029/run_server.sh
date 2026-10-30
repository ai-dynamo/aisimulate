#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Run inside the pinned vLLM image in one eight-GPU Slurm allocation.
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the verified read-only checkpoint}"
: "${RUN_DIR:?Set RUN_DIR to a new absolute results directory}"
test ! -e "$RUN_DIR/server.pid"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RUN_DIR/cache" "$RUN_DIR/tmp"
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$RUN_DIR/cache/huggingface"
export XDG_CACHE_HOME="$RUN_DIR/cache"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/cache/torchinductor"
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
export CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
export VLLM_CACHE_ROOT="$RUN_DIR/cache/vllm"
export TMPDIR="$RUN_DIR/tmp"
export VLLM_SERVER_DEV_MODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

python3 - <<'PY' > "$RUN_DIR/runtime.json"
import importlib.metadata
import json
import os
import platform
import subprocess
import torch

identity = {
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "hostname": platform.node(),
    "machine": platform.machine(),
    "vllm": importlib.metadata.version("vllm"),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "model_path": os.environ["MODEL_PATH"],
    "gpus": subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version,clocks.sm,power.limit",
        "--format=csv,noheader"
    ], text=True).splitlines(),
}
if not identity["vllm"].startswith("0.31.0"):
    raise RuntimeError(f"Unexpected vLLM version: {identity['vllm']}")
if len(identity["gpus"]) != 8 or any("B300" not in gpu for gpu in identity["gpus"]):
    raise RuntimeError("This campaign requires eight B300 GPUs")
print(json.dumps(identity, indent=2))
PY

server=(python3 -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH" --served-model-name Qwen/Qwen3.5-397B-A17B
    --host 127.0.0.1 --port 8800 --language-model-only
    --tensor-parallel-size 8 --pipeline-parallel-size 1 --data-parallel-size 1
    --dtype bfloat16 --kv-cache-dtype auto
    --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype float32
    --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill
    --no-async-scheduling --max-model-len 65536 --max-num-seqs 64
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 --seed 0
    --enable-prompt-tokens-details --enable-per-request-metrics
    --worker-extension-cls cache_probe.AlignmentCacheProbe)
printf '%q ' "${server[@]}" > "$RUN_DIR/server-command.txt"
printf '\n' >> "$RUN_DIR/server-command.txt"
"${server[@]}" > "$RUN_DIR/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" > "$RUN_DIR/server.pid"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,power.limit \
    --format=csv --loop=5 > "$RUN_DIR/gpu-monitor.csv" 2>&1 &
monitor_pid=$!
cleanup() {
    kill -TERM "$monitor_pid" 2>/dev/null || true
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 - <<'PY'
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

pid = int((Path(os.environ["RUN_DIR"]) / "server.pid").read_text())
deadline = time.monotonic() + 1800
while time.monotonic() < deadline:
    os.kill(pid, 0)
    try:
        with urllib.request.urlopen("http://127.0.0.1:8800/health", timeout=5) as response:
            if response.status == 200:
                break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(5)
else:
    raise TimeoutError("vLLM readiness exceeded 30 minutes")
PY

# The public entry points below are kept in the adjacent probe and collector.
python3 "$script_dir/cache_probe.py" --base-url http://127.0.0.1:8800 \
    --output "$RUN_DIR/cache-receipt.json"
python3 "$script_dir/collect.py" --base-url http://127.0.0.1:8800 \
    --model Qwen/Qwen3.5-397B-A17B --output "$RUN_DIR/smoke" \
    --cache-receipt "$RUN_DIR/cache-receipt.json" --smoke
touch "$RUN_DIR/SMOKE_READY"

deadline=$((SECONDS + 1800))
while [[ ! -f "$RUN_DIR/CACHE_GATE_PASSED" ]]; do
    kill -0 "$server_pid"
    if [[ -f "$RUN_DIR/CACHE_GATE_FAILED" || "$SECONDS" -ge "$deadline" ]]; then
        echo "Cache gate failed or was not approved within 30 minutes" >&2
        exit 2
    fi
    sleep 5
done

python3 "$script_dir/collect.py" --base-url http://127.0.0.1:8800 \
    --model Qwen/Qwen3.5-397B-A17B --output "$RUN_DIR/campaign" \
    --cache-receipt "$RUN_DIR/cache-receipt.json" --matrix --repeats 3
touch "$RUN_DIR/COLLECTION_COMPLETE"
