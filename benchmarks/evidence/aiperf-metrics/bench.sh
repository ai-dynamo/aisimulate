#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage: bench.sh MODEL REVISION CONCURRENCY REQUESTS TAG [extra aiperf flags]
set -euo pipefail
MODEL=$1 REV=$2 CONC=$3 NREQ=$4 TAG=$5; shift 5
OUT=/work/results/$(basename "$MODEL"); mkdir -p "$OUT"
A=/work/aiperf-venv/bin/aiperf
COMMON=(--model "$MODEL" --tokenizer "$MODEL" --tokenizer-revision "$REV" --url http://localhost:18741
  --endpoint-type chat --streaming --use-server-token-count --extra-inputs ignore_eos:true
  --random-seed 0 --ui-type none)
# Discarded warmup on synthetic prompts; the measured run then consumes exactly NREQ trace rows.
$A profile "${COMMON[@]}" --synthetic-input-tokens-mean 512 --output-tokens-mean 64 \
  --concurrency "$CONC" --request-count $((CONC * 2 > 8 ? CONC * 2 : 8)) --artifact-dir "$OUT/warmup-$TAG" > "$OUT/warmup-$TAG.log" 2>&1
$A profile "${COMMON[@]}" --input-file /work/trace.jsonl --custom-dataset-type mooncake_trace --isl-block-size 64 \
  --dataset-sampling-strategy sequential --concurrency "$CONC" --request-count "$NREQ" \
  --export-level raw --artifact-dir "$OUT/$TAG" "$@" > "$OUT/$TAG.log" 2>&1
echo "DONE $TAG"; ls "$OUT/$TAG"
