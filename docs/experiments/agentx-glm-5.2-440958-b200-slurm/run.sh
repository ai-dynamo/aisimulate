#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Locally authored launcher for the experiment documented in README.md.
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/models
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_FILE_KV=/tmp/agentx-discovery-${SLURM_JOB_ID:?}
export DYN_NAMESPACE=agentx-glm52
export SGLANG_CACHE_DIR=/tmp/sglang-cache
export SGLANG_TIMEOUT_KEEP_ALIVE=900
export TORCH_CUDA_ARCH_LIST=10.0
CKPT=/scratch/models/hub/models--nvidia--GLM-5.2-NVFP4/snapshots/53e0691e21895a3863a606dfd12910c69eba94ab
OUT=/scratch/agentx-results/job-${SLURM_JOB_ID}
if [ -e "$OUT" ]; then echo "Refusing to overwrite $OUT" >&2; exit 1; fi
test -f "$CKPT/model.safetensors.index.json"
mkdir -p "$OUT"
export PIP_CACHE_DIR="$OUT/pip-cache"
export XDG_CACHE_HOME=/tmp/agentx-cache
CLIENT="$OUT/client"
python3 -m venv "$CLIENT"
"$CLIENT/bin/pip" install --quiet --disable-pip-version-check 'aiperf @ git+https://github.com/SemiAnalysisAI/aiperf.git@754356e9a39acc6cc6afb242d123bb57c3fb6f75'
"$CLIENT/bin/pip" freeze >"$OUT/client-freeze.txt"
"$CLIENT/bin/aiperf" profile --help >"$OUT/aiperf-help.txt"
grep -q 'warmup-requests-per-lane' "$OUT/aiperf-help.txt"
exec > >(tee -a "$OUT/run.log") 2>&1
printf 'START %s JOB=%s\n' "$(date -Is)" "$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,power.limit --format=csv
python3 -c 'import importlib.metadata as m; print({p:m.version(p) for p in ["ai-dynamo","sglang","flashinfer-python"]})'
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
python3 -m dynamo.frontend --router-mode round-robin --http-port 8000 --trust-remote-code >"$OUT/frontend.log" 2>&1 &
frontend_pid=$!
start_worker() {
  DYN_SYSTEM_PORT=9090 python3 -m dynamo.sglang --model-path "$CKPT" --served-model-name nvidia/GLM-5.2-NVFP4 \
    --trust-remote-code --tp 8 --ep-size 1 --quantization modelopt_fp4 \
    --kv-cache-dtype fp8_e4m3 --bf16-gemm-backend cutedsl \
    --chunked-prefill-size 8192 --max-prefill-tokens 8192 --mem-fraction-static 0.83 \
    --max-running-requests 8 --cuda-graph-max-bs 8 \
    --speculative-algorithm EAGLE --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --watchdog-timeout 1800 --enable-metrics >"$OUT/worker.log" 2>&1 &
  worker_pid=$!
}
# Synthetic acceptance matches the AgentX performance point, not text quality.
export SGLANG_SIMULATE_ACC_LEN=2.99
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
start_worker
for attempt in $(seq 1 360); do
  kill -0 "$worker_pid" || { tail -80 "$OUT/worker.log"; exit 1; }
  kill -0 "$frontend_pid" || { tail -80 "$OUT/frontend.log"; exit 1; }
  if curl -fsS http://localhost:8000/v1/models | grep -q 'nvidia/GLM-5.2-NVFP4'; then break; fi
  sleep 10
done
curl -fS --max-time 300 http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/GLM-5.2-NVFP4","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}' >"$OUT/smoke.json"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
"$CLIENT/bin/aiperf" profile \
  --scenario inferencex-agentx-mvp --url http://localhost:8000 \
  --endpoint /v1/chat/completions --endpoint-type chat \
  --model nvidia/GLM-5.2-NVFP4 --tokenizer "$CKPT" --tokenizer-trust-remote-code \
  --public-dataset semianalysis_cc_traces_weka_062126 --num-dataset-entries 393 \
  --concurrency 4 --benchmark-duration 3600 --random-seed 42 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --warmup-grace-period 1800 \
  --trace-idle-gap-cap-seconds 300 --system-idle-gap-cap-seconds 10 \
  --cache-bust first_turn_prefix --streaming --extra-inputs ignore_eos:true \
  --use-server-token-count --no-gpu-telemetry --slice-duration 1 --stats-interval 30 \
  --server-metrics http://localhost:9090/metrics \
  --artifact-dir "$OUT/aiperf"
printf 'COMPLETE %s\n' "$(date -Is)"
