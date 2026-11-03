#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=glm-rsi-paged-indexer
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
: "${RSI_INDEXER_CODE:?Immutable qualified probe directory}"
: "${RSI_OP_REMAINING_GPU_SECONDS:?Reconciled op budget}"
test "$RSI_OP_REMAINING_GPU_SECONDS" -ge 2760
test "${SLURM_GPUS_ON_NODE:-}" = 1
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
cd "$storage_root"
output="$CAMPAIGN_ROOT/indexer-$SLURM_JOB_ID"
mkdir "$output"
scontrol show job "$SLURM_JOB_ID" -o > "$output/allocation.txt"
grep -q 'gres/gpu=1\([, ]\|$\)' "$output/allocation.txt"
sha256sum --check "$CAMPAIGN_ROOT/historical-preparation/image.sha256"
(cd "$RSI_INDEXER_CODE" && sha256sum --check source.sha256)
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
export ENROOT_DATA_PATH="$CAMPAIGN_ROOT/enroot-indexer-$SLURM_JOB_ID"
mkdir -p "$ENROOT_DATA_PATH" "$output/cache"
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem,power.limit --format=csv > "$output/gpus-before.csv"
container_output="${output/#$storage_root/\/workspace}"
container_code="${RSI_INDEXER_CODE/#$storage_root/\/workspace}"
srun --ntasks=1 --container-image="$storage_root/images/sglang-0520-b27fce60.sqsh" \
    --container-writable --container-mounts="$storage_root:/workspace" \
    env "XDG_CACHE_HOME=$container_output/cache/xdg" "DG_JIT_CACHE_DIR=$container_output/cache/deepgemm" \
    "TRITON_CACHE_DIR=$container_output/cache/triton" "TORCHINDUCTOR_CACHE_DIR=$container_output/cache/inductor" \
    timeout --signal=TERM --kill-after=30s 2520s python3 "$container_code/paged_indexer_probe.py" \
    --upstream-test "$container_code/upstream_paged_test.py" --output "$container_output/probe"
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem,power.limit --format=csv > "$output/gpus-after.csv"
