#!/bin/bash
#SBATCH --account=aifm
#SBATCH --qos=batch-short
#SBATCH --partition=b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,b200@501-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=hk-minimax-fpm-image
#SBATCH --output=/home/scratch.hongkuanz_gpu/minimax-m3-439922-vllm-20260911/image-%j.log
set -euo pipefail
bash /home/scratch.hongkuanz_gpu/minimax-m3-439922-vllm-20260911/prepare-image.sh
