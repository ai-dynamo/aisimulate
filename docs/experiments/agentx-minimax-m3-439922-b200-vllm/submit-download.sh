#!/bin/bash
#SBATCH --account=aifm
#SBATCH --qos=batch-short
#SBATCH --partition=b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,b200@501-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=0
#SBATCH --time=02:00:00
#SBATCH --job-name=hk-minimax-m3-checkpoint
#SBATCH --output=/home/scratch.hongkuanz_gpu/minimax-m3-439922-vllm-20260911/download-%j.log
set -euo pipefail
export HF_HOME=/scratch/models
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONUNBUFFERED=1
srun --ntasks=1 \
  --container-image=/home/scratch.hongkuanz_gpu/images/vllm-agentx-fpm-996fed4671-amd64.sqsh \
  --container-mounts=/home/scratch.hongkuanz_gpu:/scratch --container-remap-root \
  /opt/fpm/.venv/bin/python /scratch/minimax-m3-439922-vllm-20260911/download-checkpoints.py
