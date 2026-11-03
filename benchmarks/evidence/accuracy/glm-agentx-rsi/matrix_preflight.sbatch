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
if test "${RSI_PREFLIGHT_KIND:-dispatch}" = profile; then
    : "${RSI_PROFILE_PREFLIGHT:?Versioned profiler output receipt}"
else
    test "${RSI_PREFLIGHT_KIND:-dispatch}" = dispatch
    : "${RSI_DISPATCH_PREFLIGHT:?Versioned output receipt}"
fi
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
image="$storage_root/images/sglang-0520-b27fce60.sqsh"
sha256sum --check "$CAMPAIGN_ROOT/historical-preparation/image.sha256"
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
export ENROOT_DATA_PATH="$CAMPAIGN_ROOT/enroot-matrix-preflight-$SLURM_JOB_ID"
mkdir -p "$ENROOT_DATA_PATH"
root="${CAMPAIGN_ROOT/#$storage_root/\/workspace}"
code="${RSI_CODE/#$storage_root/\/workspace}"
if test "${RSI_PREFLIGHT_KIND:-dispatch}" = profile; then
    receipt="${RSI_PROFILE_PREFLIGHT/#$storage_root/\/workspace}"
    source_arguments=()
    if test -n "${RSI_PROFILE_SOURCE_DIR:-}"; then
        source_dir="${RSI_PROFILE_SOURCE_DIR/#$storage_root/\/workspace}"
        source_arguments=(--sources-dir "$source_dir")
    fi
    srun --ntasks=1 --container-image="$image" --container-writable \
        --container-mounts="$storage_root:/workspace" \
        python3 "$code/profile_preflight.py" --output "$receipt" "${source_arguments[@]}"
    exit 0
fi
receipt="${RSI_DISPATCH_PREFLIGHT/#$storage_root/\/workspace}"
coding_arguments=()
if test -n "${RSI_CODING_INPUTS:-}"; then
    coding_inputs="${RSI_CODING_INPUTS/#$storage_root/\/workspace}"
    coding_arguments=(--coding-inputs "$coding_inputs" --model /workspace/models/GLM-5.3-NVFP4-e3b8f9f2)
fi
srun --ntasks=1 --container-image="$image" --container-writable \
    --container-mounts="$storage_root:/workspace" \
    python3 "$code/dispatch_preflight.py" --output "$receipt" "${coding_arguments[@]}"
