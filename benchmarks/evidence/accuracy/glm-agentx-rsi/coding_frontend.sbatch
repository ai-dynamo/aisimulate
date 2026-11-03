#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=glm-rsi-coding-cpu
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
: "${RSI_CODE:?Immutable collector directory}"
test "${SLURM_GPUS_ON_NODE:-0}" -eq 3
: "${REMAINING_GPU_SECONDS:?Reconciled campaign budget}"
# Ten-minute wall limit plus one minute of cleanup, at three allocated GPUs.
test "$REMAINING_GPU_SECONDS" -ge 1980
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
image="$storage_root/images/sglang-0520-b27fce60.sqsh"
sha256sum --check "$CAMPAIGN_ROOT/historical-preparation/image.sha256"
# Cluster policy requires a GPU reservation on B300 hosts. Charge it, but expose
# no CUDA devices to this CPU-only component measurement.
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
export ENROOT_DATA_PATH="$CAMPAIGN_ROOT/enroot-coding-cpu-$SLURM_JOB_ID"
mkdir -p "$ENROOT_DATA_PATH"
root="${CAMPAIGN_ROOT/#$storage_root/\/workspace}"
code="${RSI_CODE/#$storage_root/\/workspace}"
srun --ntasks=1 --container-image="$image" --container-writable \
    --container-mounts="$storage_root:/workspace" \
    env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 RSI_ALLOCATED_GPUS=3 \
    python3 "$code/measure_coding_frontend.py" \
    --inputs "$root/E029-prepared-v1" --assets "$root/E029-independent-audit.json" \
    --model /workspace/models/GLM-5.3-NVFP4-e3b8f9f2 \
    --source /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py \
    --output "$root/coding-cpu-$SLURM_JOB_ID"
