#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=aisim664-fp8-native
#SBATCH --account=coreai_comparch_lights-out-inf
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --nodelist=pool0-0002
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:15:00
set -euo pipefail
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_lights-out-inf/users/simonec
cd "$storage_root"
job_dir="$storage_root/jobs/pr664-fp8-native-20261103"
result_dir="$storage_root/results/pr664-fp8-native-20261103/$SLURM_JOB_ID"
mkdir -p "$result_dir/source"
cp "$job_dir/native_audit.py" "$job_dir/audit.sbatch" "$result_dir/"
tar -xf "$job_dir/source.tar.gz" -C "$result_dir/source"
sha256sum "$job_dir/source.tar.gz" "$job_dir/native_audit.py" images/vllm-0310-pr664.sqsh > "$result_dir/input-sha256.txt"
scontrol show job "$SLURM_JOB_ID" > "$result_dir/slurm-job.txt"
nvidia-smi -q > "$result_dir/nvidia-smi-before.txt"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot" ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_DATA_PATH="$result_dir/enroot" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
mkdir -p "$ENROOT_DATA_PATH"
container_dir="/workspace/results/pr664-fp8-native-20261103/$SLURM_JOB_ID"
for kind in collector native; do
 for context in 1 0; do
  name="$kind-$context"
  srun --ntasks=1 --container-image="$storage_root/images/vllm-0310-pr664.sqsh" --container-mounts="$storage_root:/workspace" --container-workdir="$container_dir" bash -c 'start=$(date +%s.%N); python3 native_audit.py --kind "$1" --context "$2" --output "$4/$1-$2"; code=$?; end=$(date +%s.%N); printf "%s %s %s\n" "$start" "$end" "$code" > "$4/$1-$2.process-timestamps.txt"; exit "$code"' _ "$kind" "$context" unused "$container_dir" > "$result_dir/$name.log" 2>&1
 done
done
nvidia-smi -q > "$result_dir/nvidia-smi-after.txt"
