#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --account=aifm
#SBATCH --qos=batch-short
#SBATCH --partition=b300@ts4/b300-nvl8@ts3/8gpu-224cpu-2048gb,b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb,b300@ts8/b300-nvl8@cr+mp/8gpu-224cpu-2048gb,b300@qs1/b300-nvl8@cr+mp/8gpu-224cpu-2048gb,b300@ts7/b300-nvl8@ts7/8gpu-224cpu-2048gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gpus=8
#SBATCH --time=02:30:00
#SBATCH --signal=TERM@180
#SBATCH --job-name=hk-agentx-dsv4-rerun
#SBATCH --output=/home/scratch.hongkuanz_gpu/agentx-dsv4-rerun-%j.log
set -euo pipefail
BASE=/home/scratch.hongkuanz_gpu
BUNDLE=$BASE/agentx-dsv4-rerun-20260910
hostname
scontrol show job "$SLURM_JOB_ID"
sha256sum -c "$BUNDLE/image-squashfs.sha256"
srun --ntasks=1 \
  --container-image="$BASE/images/sglang-agentx-fpm-f856a455-amd64.sqsh" \
  --container-mounts="$BASE:/scratch" --container-remap-root \
  python3 /scratch/agentx-dsv4-rerun-20260910/campaign.py
python3 "$BUNDLE/compare.py" "$BASE/agentx-dsv4-rerun-results/job-$SLURM_JOB_ID"
