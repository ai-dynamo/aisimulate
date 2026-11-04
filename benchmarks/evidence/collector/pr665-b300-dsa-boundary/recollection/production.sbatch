#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=aisim665-b52-recollect
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:55:00
#SBATCH --gpu-freq=high
set -euo pipefail
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
cd "$storage_root"
job_dir="$storage_root/jobs/pr665-b52-recollect-20261103"
result_dir="$storage_root/results/pr665-b52-recollect-20261103/$SLURM_JOB_ID"
mkdir -p "$result_dir/source" "$result_dir/cases"
cp "$job_dir/production_worker.py" "$job_dir/production.sbatch" "$result_dir/"
tar -xf "$job_dir/source-production.tar.gz" -C "$result_dir/source"
sha256sum "$job_dir/source-production.tar.gz" "$result_dir/production_worker.py" > "$result_dir/input-sha256.txt"
scontrol show job "$SLURM_JOB_ID" > "$result_dir/slurm-job.txt"
nvidia-smi -q > "$result_dir/nvidia-smi-before.txt"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_DATA_PATH="$result_dir/enroot"
mkdir -p "$ENROOT_DATA_PATH"
export SGLANG_TEST_NUM_LAYERS=2 SGLANG_LOAD_FORMAT=dummy OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
container_dir="/workspace/results/pr665-b52-recollect-20261103/$SLURM_JOB_ID"
export XDG_CACHE_HOME="$container_dir/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export PYTHONPATH="$container_dir/source/python/aisimulate"
image="$storage_root/images/sglang-0521-2dbe4c7f.sqsh"
for mode in indexer skip_indexer; do
 srun --ntasks=1 --container-image="$image" --container-mounts="$storage_root:/workspace" --container-workdir="$container_dir" python3 production_worker.py --mode "$mode" --output "$container_dir/cases/$mode" > "$result_dir/$mode.log" 2>&1
 printf '0\n' > "$result_dir/$mode.exit-code.txt"
done
nvidia-smi -q > "$result_dir/nvidia-smi-after.txt"
