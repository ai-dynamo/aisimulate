#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=q35agentic-20261029
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec/logs/q35agentic-20261029/gpu-%j.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec/logs/q35agentic-20261029/gpu-%j.err

set -euo pipefail
export lustre=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
export storage_root="$lustre"
cd "$storage_root"
case "$PWD/" in "$HOME"/*) exit 1 ;; esac
image="$storage_root/images/vllm-0310-cu130-a4a4c043.sqsh"
checkpoint="$storage_root/models/Qwen3.5-397B-A17B-84726181"
campaign="$storage_root/results/qwen35-agentic-alignment-20261029"
test -s "$image"
test -s "$checkpoint/model.safetensors.index.json"
test -s "$campaign/preparation/READY.json"
run="$campaign/gpu-$SLURM_JOB_ID"
mkdir "$run"
cp "$campaign/preparation/READY.json" "$run/preparation.json"
cp -R "$campaign/code" "$run/code"
(cd "$run/code" && sha256sum run_server.sh collect.py cache_probe.py job.sbatch campaign.json) > "$run/code-sha256.txt"
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_DATA_PATH="$run/enroot"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"
unset TMPDIR

srun --ntasks=1 --container-image="$image" \
    --container-mounts="$storage_root:/workspace,$checkpoint:/model:ro" \
    --container-workdir="/workspace/results/qwen35-agentic-alignment-20261029/gpu-$SLURM_JOB_ID/code" \
    env MODEL_PATH=/model \
    RUN_DIR="/workspace/results/qwen35-agentic-alignment-20261029/gpu-$SLURM_JOB_ID" \
    bash run_server.sh
