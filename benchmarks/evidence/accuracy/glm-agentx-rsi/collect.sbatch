#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=glm-rsi-timing
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=02:00:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
: "${REMAINING_GPU_SECONDS:?Set from reconciled campaign ledger}"
test "$REMAINING_GPU_SECONDS" -ge 57600
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
cd "$storage_root"
model="$storage_root/models/GLM-5.3-NVFP4-e3b8f9f2"
test -s "$CAMPAIGN_ROOT/preparation/weights.sha256"
test -s "$CAMPAIGN_ROOT/preparation/model-revision.txt"
test -s "$CAMPAIGN_ROOT/preparation/model-verification.json"
test -s "$CAMPAIGN_ROOT/preparation/runtime-preflight.json"
test -s "$CAMPAIGN_ROOT/preparation/instrumentation-preflight.json"
test -s "$model/config.json"
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
export ENROOT_DATA_PATH="$CAMPAIGN_ROOT/enroot-$SLURM_JOB_ID"
mkdir -p "$ENROOT_DATA_PATH" "$CAMPAIGN_ROOT/gpu-$SLURM_JOB_ID"
cp -R "$CAMPAIGN_ROOT/code" "$CAMPAIGN_ROOT/code-$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$CAMPAIGN_ROOT/gpu-$SLURM_JOB_ID/gpus.csv"
root="${CAMPAIGN_ROOT/#$storage_root/\/workspace}"
srun --ntasks=1 --container-image="$storage_root/images/sglang-0521-2dbe4c7f.sqsh" \
    --container-writable --container-mounts="$storage_root:/workspace,$model:/model:ro" \
    python3 "$root/code-$SLURM_JOB_ID/collect.py" --model /model \
    --output "$root/gpu-$SLURM_JOB_ID/controls" --seconds 6900
