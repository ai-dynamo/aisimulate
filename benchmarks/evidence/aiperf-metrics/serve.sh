#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage: serve.sh MODEL REVISION. Serves one model on port 18741 inside the container.
set -euo pipefail
MODEL=$1 REV=$2
OUT=/work/results/$(basename "$MODEL"); mkdir -p "$OUT"
pkill -f "vllm serve" || true; sleep 3
nohup vllm serve "$MODEL" --revision "$REV" --tokenizer-revision "$REV" --dtype bfloat16 \
  --max-model-len 4096 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching --gpu-memory-utilization 0.85 --seed 0 --port 18741 > "$OUT/vllm.log" 2>&1 &
for i in $(seq 120); do curl -sf localhost:18741/health >/dev/null && { echo READY; exit 0; }; sleep 5; done
echo NOT_READY; tail -20 "$OUT/vllm.log"; exit 1
