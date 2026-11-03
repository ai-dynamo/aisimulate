#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=glm-rsi-submit-preflight
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
: "${RSI_CODE:?Immutable collector directory}"
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
image="$storage_root/images/sglang-0520-b27fce60.sqsh"
sha256sum --check "$CAMPAIGN_ROOT/historical-preparation/image.sha256"
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
export ENROOT_DATA_PATH="$CAMPAIGN_ROOT/enroot-matrix-preflight-$SLURM_JOB_ID"
mkdir -p "$ENROOT_DATA_PATH"
root="${CAMPAIGN_ROOT/#$storage_root/\/workspace}"
code="${RSI_CODE/#$storage_root/\/workspace}"
srun --ntasks=1 --container-image="$image" --container-writable \
    --container-mounts="$storage_root:/workspace" \
    python3 "$code/dispatch_preflight.py" --output "$root/matrix-dispatch-preflight.json"
