#!/bin/bash
#SBATCH --account=aifm
#SBATCH --qos=batch-short
#SBATCH --partition=b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,b200@501-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=768G
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@180
#SBATCH --job-name=hk-minimax-m3-fpm-pair
#SBATCH --output=/home/scratch.hongkuanz_gpu/minimax-m3-439922-vllm-20260911/benchmark-%j.log
set -euo pipefail
scontrol show job "$SLURM_JOB_ID"
srun --ntasks=1 \
  --container-image=/home/scratch.hongkuanz_gpu/images/vllm-agentx-fpm-b3563fc65a-amd64.sqsh \
  --container-mounts=/home/scratch.hongkuanz_gpu:/scratch --container-remap-root \
  /opt/fpm/.venv/bin/python /scratch/minimax-m3-439922-vllm-20260911/benchmark.py
