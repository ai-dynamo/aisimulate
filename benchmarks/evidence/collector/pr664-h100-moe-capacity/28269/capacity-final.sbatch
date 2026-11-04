#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=aisim664-moe-capacity
#SBATCH --account=coreai_comparch_lights-out-inf
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:20:00
set -euo pipefail
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_lights-out-inf/users/simonec
cd "$storage_root"
job_dir="$storage_root/jobs/pr664-capacity-20261103"
result_dir="$storage_root/results/pr664-capacity-20261103/$SLURM_JOB_ID"
mkdir -p "$result_dir/source" "$result_dir/cases"
cp "$job_dir/capacity_worker.py" "$job_dir/capacity-final.sbatch" "$result_dir/"
tar -xf "$job_dir/source-production.tar.gz" -C "$result_dir/source"
sha256sum "$job_dir/source-production.tar.gz" "$result_dir/capacity_worker.py" > "$result_dir/input-sha256.txt"
scontrol show job "$SLURM_JOB_ID" > "$result_dir/slurm-job.txt"
nvidia-smi -q > "$result_dir/nvidia-smi-before.txt"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot" ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_DATA_PATH="$result_dir/enroot"
mkdir -p "$ENROOT_DATA_PATH"
container_dir="/workspace/results/pr664-capacity-20261103/$SLURM_JOB_ID"
export PYTHONPATH="$container_dir/source/python/aisimulate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
for kind in bf16-control; do
 set +e
 srun --ntasks=1 --container-image="$storage_root/images/trtllm-130rc29-pr664.sqsh" --container-mounts="$storage_root:/workspace" --container-workdir="$container_dir" python3 capacity_worker.py --kind "$kind" --output "$container_dir/cases/$kind" > "$result_dir/$kind.log" 2>&1
 code=$?
 set -e
 printf '%s\n' "$code" > "$result_dir/$kind.exit-code.txt"
done
nvidia-smi -q > "$result_dir/nvidia-smi-after.txt"
