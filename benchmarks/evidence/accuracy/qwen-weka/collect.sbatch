#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=qwen-weka-agentx
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch_long
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=08:00:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
: "${REMAINING_GPU_SECONDS:?Set from aggregate Slurm elapsed time before each attempt}"
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
cd "$storage_root"
model="$storage_root/models/Qwen3.5-397B-A17B-84726181"
image="$storage_root/images/vllm-0310-cu130-a4a4c043.sqsh"
run="$CAMPAIGN_ROOT/gpu-$SLURM_JOB_ID"
test -s "$CAMPAIGN_ROOT/preparation/corpus-receipt.json"
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_DATA_PATH="$CAMPAIGN_ROOT/enroot-$SLURM_JOB_ID"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
mkdir -p "$ENROOT_DATA_PATH"
root_in_container="${run/#$storage_root/\/workspace}"
cp -R "$CAMPAIGN_ROOT/code" "$CAMPAIGN_ROOT/code-$SLURM_JOB_ID"
code_in_container="${CAMPAIGN_ROOT/#$storage_root/\/workspace}/code-$SLURM_JOB_ID"
srun --ntasks=1 --container-image="$image" \
    --container-mounts="$storage_root:/workspace,$model:/model:ro" \
    python3 "$code_in_container/collect.py" --root "$root_in_container" --model /model --budget-seconds "$REMAINING_GPU_SECONDS"
