#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=glm-rsi-forward-export
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
: "${RSI_TRACE_ROOT:?Completed private controls directory}"
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
case "$RSI_TRACE_ROOT" in "$CAMPAIGN_ROOT"/gpu-*/controls) ;; *) exit 1 ;; esac
test -s "$RSI_TRACE_ROOT/completion.json"
test -s "$RSI_TRACE_ROOT/instrumented-mtp/completion.json"
image="$storage_root/images/sglang-0520-b27fce60.sqsh"
sha256sum --check "$CAMPAIGN_ROOT/historical-preparation/image.sha256"
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
export ENROOT_DATA_PATH="$CAMPAIGN_ROOT/enroot-forward-export-$SLURM_JOB_ID"
mkdir -p "$ENROOT_DATA_PATH"
trace_root="${RSI_TRACE_ROOT/#$storage_root/\/workspace}"
srun --ntasks=1 --container-image="$image" --container-writable \
    --container-mounts="$storage_root:/workspace" \
    bash -euo pipefail -c '
        trace_root=$1
        traces=("$trace_root"/instrumented-mtp/nsys/*.nsys-rep)
        test "${#traces[@]}" -eq 2
        export_root="$trace_root/numerical-exports"
        mkdir "$export_root"
        sha256sum "${traces[@]}" > "$export_root/source-traces.sha256"
        nsys --version > "$export_root/nsys-version.txt"
        for trace in "${traces[@]}"; do
            prefix="$export_root/$(basename "$trace" .nsys-rep)"
            nsys stats --report cuda_gpu_trace,nvtx_gpu_proj_trace \
                --format csv --output "$prefix" "$trace" > "$prefix.log" 2>&1
        done
        sha256sum --check "$export_root/source-traces.sha256"
        tables=("$export_root"/*.csv)
        test "${#tables[@]}" -eq 4
        sha256sum "${tables[@]}" > "$export_root/tables.sha256"
        printf "%s\n" complete > "$export_root/completed.txt"
    ' forward-export "$trace_root"
