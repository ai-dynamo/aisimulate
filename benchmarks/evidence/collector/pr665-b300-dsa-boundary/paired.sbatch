#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=aisim664-sg-pairs
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --gpu-freq=high
set -euo pipefail
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
cd "$storage_root"
case "$PWD/" in "$HOME"/*) exit 1 ;; esac
task=pr664-gpu-review-20261102/sglang
job_dir="$storage_root/jobs/$task"
result_dir="$storage_root/results/$task/pairs-$SLURM_JOB_ID"
mkdir -p "$result_dir/sources" "$result_dir/cases"
cp "$job_dir/"*.py "$job_dir/"*.json "$job_dir/paired.sbatch" "$result_dir/"
for variant in baseline collected current; do
 mkdir -p "$result_dir/sources/$variant"
 tar -xf "$job_dir/$variant.tar.gz" -C "$result_dir/sources/$variant"
done
image="$storage_root/images/sglang-0521-2dbe4c7f.sqsh"
sha256sum "$image" > "$result_dir/image-squashfs.sha256"
sha256sum "$job_dir/"*.tar.gz "$result_dir/"*.py > "$result_dir/harness-source.sha256"
scontrol show job "$SLURM_JOB_ID" > "$result_dir/slurm-job.txt"
nvidia-smi -q > "$result_dir/nvidia-smi-before.txt"
nvidia-smi --query-gpu=timestamp,index,uuid,name,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu --format=csv -l 1 > "$result_dir/gpu-telemetry.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_DATA_PATH="$result_dir/enroot"
mkdir -p "$ENROOT_DATA_PATH"
container_dir="/workspace/results/$task/pairs-$SLURM_JOB_ID"
export SGLANG_TEST_NUM_LAYERS=2 SGLANG_LOAD_FORMAT=dummy OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$container_dir/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
set +e
srun --ntasks=1 --container-image="$image" --container-mounts="$storage_root:/workspace" --container-workdir="$container_dir" python3 run_pairs.py
status=$?
set -e
nvidia-smi -q > "$result_dir/nvidia-smi-after.txt"
printf '%s\n' "$status" > "$result_dir/exit-code.txt"
exit "$status"
