#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run inside the built image with one allocated GPU and writable /run/agentx.
set -euo pipefail
OUT=/run/agentx
export HF_HOME="$OUT/hf"
export XDG_CACHE_HOME="$OUT/cache"
export SGLANG_CACHE_DIR="$OUT/sglang-cache"
export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV="$OUT/discovery"
export DYN_NAMESPACE=agentx-arm-smoke
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
mkdir -p "$OUT"
python3 - <<'PY'
import importlib.metadata as m
import platform
import torch
assert platform.machine() == "aarch64"
assert torch.cuda.device_count() == 1, "Expose only the Slurm-allocated GPU"
a = torch.ones((32, 32), device="cuda")
assert torch.all(a @ a == 32).item()
torch.cuda.synchronize()
print("CUDA_SMOKE_OK", torch.cuda.get_device_name(0), flush=True)
print({p: m.version(p) for p in ["ai-dynamo", "sglang", "flashinfer-python"]})
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca",
                  local_dir="/run/agentx/model")
PY
python3 -m dynamo.sglang --help >"$OUT/sglang-help.txt" 2>&1
grep -q hicache-mem-layout "$OUT/sglang-help.txt"
/opt/agentx-aiperf/bin/aiperf profile --help >"$OUT/aiperf-help.txt"
grep -q warmup-requests-per-lane "$OUT/aiperf-help.txt"
frontend_pid= worker_pid=
cleanup() {
  for child in "$worker_pid" "$frontend_pid"; do
    if [ -n "$child" ]; then kill "$child" 2>/dev/null || true; fi
  done
  for child in "$worker_pid" "$frontend_pid"; do
    if [ -n "$child" ]; then wait "$child" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT
python3 -m dynamo.frontend --router-mode round-robin --http-port 8000 \
  >"$OUT/frontend.log" 2>&1 &
frontend_pid=$!
DYN_SYSTEM_PORT=9090 python3 -m dynamo.sglang \
  --model-path "$OUT/model" --served-model-name Qwen/Qwen3-0.6B \
  --tp 1 --dtype bfloat16 --attention-backend triton \
  --mem-fraction-static 0.4 --max-total-tokens 8192 \
  --context-length 4096 --chunked-prefill-size 1024 --max-running-requests 4 \
  --disable-cuda-graph --enable-metrics \
  --enable-hierarchical-cache --hicache-size 2 --hicache-write-policy write_back \
  --hicache-io-backend direct --hicache-mem-layout page_first_direct \
  >"$OUT/worker.log" 2>&1 &
worker_pid=$!
ready=false
for attempt in $(seq 1 120); do
  kill -0 "$worker_pid" || { tail -80 "$OUT/worker.log"; exit 1; }
  kill -0 "$frontend_pid" || { tail -80 "$OUT/frontend.log"; exit 1; }
  if curl -fsS http://localhost:8000/v1/models | grep -q Qwen/Qwen3-0.6B; then
    ready=true
    break
  fi
  sleep 5
done
test "$ready" = true
curl -fsS --max-time 120 http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}' \
  >"$OUT/chat.json"
python3 - <<'PY'
import json
r = json.load(open("/run/agentx/chat.json"))
assert r.get("choices") and r.get("usage", {}).get("completion_tokens", 0) > 0, r
print("CHAT_SMOKE_OK", r["usage"])
PY
/opt/agentx-aiperf/bin/aiperf profile \
  --url http://localhost:8000 --endpoint /v1/chat/completions --endpoint-type chat \
  --model Qwen/Qwen3-0.6B --tokenizer "$OUT/model" \
  --synthetic-input-tokens-mean 128 --output-tokens-mean 32 \
  --request-count 8 --concurrency 2 --streaming --extra-inputs ignore_eos:true \
  --use-server-token-count --no-gpu-telemetry --artifact-dir "$OUT/aiperf"
curl -fsS http://localhost:9090/metrics >"$OUT/worker-metrics.txt"
echo ARM_SMOKE_COMMANDS_COMPLETE
