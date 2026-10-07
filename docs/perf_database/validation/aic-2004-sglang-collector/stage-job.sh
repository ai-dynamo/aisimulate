#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=aic2004-sg-final-bounded
#SBATCH --account=tensorrt
#SBATCH --qos=interactive-isolated-short
#SBATCH --partition=b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb
#SBATCH --gres=gpu:b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:35:00
#SBATCH --nodelist=umbriel-b200-079
set -euo pipefail
base=/home/scratch.harrli_sw/aic2004-four-pr-20261006/sglang-evidence
output="$base/job-$SLURM_JOB_ID-target-008"
localroot="/tmp/aic2004-sgformal-$SLURM_JOB_ID"
mkdir -p "$output" "$localroot/cache"
exec > "$output/host.log" 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$output/phase-exit.txt"; nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv > "$output/gpu-postflight.csv"; date -u > "$output/HOST_FINISHED"; scontrol show node "$SLURMD_NODENAME" > "$output/node-postflight.txt"' EXIT
date -u
hostname
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,power.limit --format=csv > "$output/gpu-preflight.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv > "$output/process-preflight.csv"
df -h /tmp /dev/shm "$base"
export ENROOT_RUNTIME_PATH="$localroot/runtime" ENROOT_CACHE_PATH="$localroot/cache-enroot" ENROOT_DATA_PATH="$localroot/data" ENROOT_TEMP_PATH="$localroot/temp"
export ENROOT_MAX_PROCESSORS=8
mkdir -p "$ENROOT_RUNTIME_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH"
image="$localroot/sglang-v0514.sqsh"
scontrol show node "$SLURMD_NODENAME" > "$output/node-preflight.txt"
enroot import -o "$image" 'docker://docker.io#lmsysorg/sglang@sha256:9611bd4c5624b0e9e17829506188a12f17205f2083de0dd44d6c521733553a50' > "$output/import-sglang.log" 2>&1
sha256sum "$image" > "$output/image-sha256.txt"
cp "$base/target-008/stage-job.sh" "$output/launched-stage.sh"
cp "$base/target-008/launch.sh" "$output/launched-script.sh"
srun --container-image="$image" --container-name="aic2004-sgformal-$SLURM_JOB_ID" --container-mounts="$base:/task,$output:/output,$localroot/cache:/cache" --container-workdir=/task --container-writable bash /task/target-008/launch.sh > "$output/collection.log" 2>&1
