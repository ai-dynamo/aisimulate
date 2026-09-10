#!/bin/bash
#SBATCH --account=aifm
#SBATCH --qos=batch-short
#SBATCH --partition=b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,b200@501-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gpus=8
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@180
#SBATCH --job-name=hk-dsv4-vllm-fpm-pair
#SBATCH --output=/home/scratch.hongkuanz_gpu/vllm-fpm-sd-20260910/benchmark-%j.log
set -euo pipefail
ROOT=/home/scratch.hongkuanz_gpu/vllm-fpm-sd-20260910
test -f "$ROOT/image-published.json"
scontrol show job "$SLURM_JOB_ID"
srun --ntasks=1 \
  --container-image=/home/scratch.hongkuanz_gpu/images/vllm-agentx-fpm-996fed4671-amd64.sqsh \
  --container-mounts=/home/scratch.hongkuanz_gpu:/scratch --container-remap-root \
  /opt/fpm/.venv/bin/python /scratch/vllm-fpm-sd-20260910/benchmark.py
