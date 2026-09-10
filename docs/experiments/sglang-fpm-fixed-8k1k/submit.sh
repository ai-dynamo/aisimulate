#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --account=aifm
#SBATCH --qos=batch-short
#SBATCH --partition=b300@ts6/dgx-b300@ts1/8gpu-256cpu-2048gb,b300@ts7/dgx-b300@ts1/8gpu-256cpu-2048gb,b300@ts4/b300-nvl8@ts3/8gpu-224cpu-2048gb,b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb,b300@ts8/b300-nvl8@cr+mp/8gpu-224cpu-2048gb,b300@qs1/b300-nvl8@cr+mp/8gpu-224cpu-2048gb,b300@ts7/b300-nvl8@ts7/8gpu-224cpu-2048gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gpus=8
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@180
#SBATCH --job-name=hk-fpm-fixed-8k1k
#SBATCH --output=/home/scratch.hongkuanz_gpu/fpm-fixed-8k1k-%j.log
set -euo pipefail
BASE=/home/scratch.hongkuanz_gpu
BUNDLE=$BASE/sglang-fpm-fixed-8k1k-20260910
hostname
scontrol show job "$SLURM_JOB_ID"
sha256sum -c "$BUNDLE/image-squashfs.sha256"
srun --ntasks=1 \
  --container-image="$BASE/images/sglang-agentx-fpm-f856a455-amd64.sqsh" \
  --container-mounts="$BASE:/scratch" --container-remap-root \
  python3 /scratch/sglang-fpm-fixed-8k1k-20260910/campaign.py

