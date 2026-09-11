#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --account=aifm
#SBATCH --qos=batch-short
#SBATCH --partition=b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,b200@501-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=1T
#SBATCH --gpus=8
#SBATCH --time=02:30:00
#SBATCH --job-name=hk-agentx-fpm-on
#SBATCH --output=/home/scratch.hongkuanz_gpu/agentx-fpm-ab-%j.log
set -euo pipefail
hostname
scontrol show job "$SLURM_JOB_ID"
srun --ntasks=1 \
  --container-image=/home/scratch.hongkuanz_gpu/images/sglang-agentx-fpm-f856a455-amd64.sqsh \
  --container-mounts=/home/scratch.hongkuanz_gpu:/scratch \
  --container-remap-root \
  bash /scratch/agentx-fpm-ab-20260909/campaign.sh
