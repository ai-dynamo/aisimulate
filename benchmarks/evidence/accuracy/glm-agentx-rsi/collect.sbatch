#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=glm-rsi-timing
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=01:15:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
: "${REMAINING_GPU_SECONDS:?Set from reconciled campaign ledger}"
# The matrix wrapper has a 45-minute wall limit; the original runner has 75.
control_seconds=4200
reservation_seconds=36480
if test "${RSI_SUITE:-timing}" = matrix; then
    matrix_wall_limit=$(scontrol show job "$SLURM_JOB_ID" -o | awk '{for(i=1;i<=NF;i++) if($i ~ /^TimeLimit=/) print substr($i,11)}')
    test "$matrix_wall_limit" = 00:45:00
    control_seconds=2640
    reservation_seconds=22080
fi
if test "${RSI_SUITE:-timing}" = coding-blocking; then
    coding_wall_limit=$(scontrol show job "$SLURM_JOB_ID" -o | awk '{for(i=1;i<=NF;i++) if($i ~ /^TimeLimit=/) print substr($i,11)}')
    test "$coding_wall_limit" = 00:33:00
    control_seconds=1920
    reservation_seconds=16320
fi
if test "${RSI_SUITE:-timing}" = forward-profile; then
    profile_wall_limit=$(scontrol show job "$SLURM_JOB_ID" -o | awk '{for(i=1;i<=NF;i++) if($i ~ /^TimeLimit=/) print substr($i,11)}')
    test "$profile_wall_limit" = 00:45:00
    control_seconds=2640
    reservation_seconds=22080
fi
test "$REMAINING_GPU_SECONDS" -ge "$reservation_seconds"
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
cd "$storage_root"
model="$storage_root/models/GLM-5.3-NVFP4-e3b8f9f2"
test -s "$CAMPAIGN_ROOT/preparation/weights.sha256"
test -s "$CAMPAIGN_ROOT/preparation/model-revision.txt"
test -s "$CAMPAIGN_ROOT/preparation/model-verification.json"
case "${RSI_RUNTIME:-prediction}" in
    prediction)
        image="$storage_root/images/sglang-0521-2dbe4c7f.sqsh"
        cache_key=sglang-0521-2dbe4c7f
        test -s "$CAMPAIGN_ROOT/preparation/runtime-preflight.json"
        test -s "$CAMPAIGN_ROOT/preparation/instrumentation-preflight.json"
        ;;
    historical)
        image="$storage_root/images/sglang-0520-b27fce60.sqsh"
        cache_key=sglang-0520-b27fce60
        test -s "$CAMPAIGN_ROOT/historical-preparation/runtime.json"
        test -s "$CAMPAIGN_ROOT/historical-preparation/instrumentation.json"
        sha256sum --check "$CAMPAIGN_ROOT/historical-preparation/image.sha256"
        ;;
    *) echo 'Unknown RSI runtime' >&2; exit 1 ;;
esac
test -s "$model/config.json"
if test "${RSI_SUITE:-timing}" = frontend || test "${RSI_SUITE:-timing}" = blocking || test "${RSI_SUITE:-timing}" = matrix || test "${RSI_SUITE:-timing}" = coding-blocking; then
    test -s "$CAMPAIGN_ROOT/frontend-template-preflight.json"
fi
if test "${RSI_SUITE:-timing}" = blocking; then
    test -s "$CAMPAIGN_ROOT/blocking-dispatch-preflight.json"
fi
if test "${RSI_SUITE:-timing}" = matrix || test "${RSI_SUITE:-timing}" = coding-blocking; then
    test -s "${RSI_DISPATCH_PREFLIGHT:?Versioned CPU qualification receipt}"
fi
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
export ENROOT_DATA_PATH="$CAMPAIGN_ROOT/enroot-$SLURM_JOB_ID"
mkdir -p "$ENROOT_DATA_PATH" "$CAMPAIGN_ROOT/gpu-$SLURM_JOB_ID"
cp -R "${RSI_CODE:-$CAMPAIGN_ROOT/code}" "$CAMPAIGN_ROOT/code-$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$CAMPAIGN_ROOT/gpu-$SLURM_JOB_ID/gpus.csv"
# Refuse a different GPU or partial allocation before loading model weights.
awk -F, 'NR > 1 {if ($1 !~ /B300/) bad=1; count++} END {exit (bad || count != 8)}' "$CAMPAIGN_ROOT/gpu-$SLURM_JOB_ID/gpus.csv"
root="${CAMPAIGN_ROOT/#$storage_root/\/workspace}"
dispatch_arguments=()
if test "${RSI_SUITE:-timing}" = matrix || test "${RSI_SUITE:-timing}" = coding-blocking; then
    dispatch_receipt="${RSI_DISPATCH_PREFLIGHT/#$storage_root/\/workspace}"
    dispatch_arguments=(--dispatch-preflight "$dispatch_receipt")
fi
coding_arguments=()
if test "${RSI_SUITE:-timing}" = coding-blocking; then
    coding_inputs="${RSI_CODING_INPUTS:?Qualified private coding inputs}"
    coding_inputs="${coding_inputs/#$storage_root/\/workspace}"
    coding_arguments=(--coding-inputs "$coding_inputs")
fi
profile_arguments=()
if test "${RSI_SUITE:-timing}" = forward-profile; then
    profile_receipt="${RSI_PROFILE_PREFLIGHT:?Versioned profiler qualification receipt}"
    profile_receipt="${profile_receipt/#$storage_root/\/workspace}"
    profile_arguments=(--profile-preflight "$profile_receipt")
fi
srun --ntasks=1 --container-image="$image" \
    --container-writable --container-mounts="$storage_root:/workspace,$model:/model:ro" \
    env "RSI_CACHE_ROOT=$root/runtime-cache/$cache_key" \
    python3 "$root/code-$SLURM_JOB_ID/collect.py" --model /model \
    --output "$root/gpu-$SLURM_JOB_ID/controls" --seconds "$control_seconds" --acceptance-policy agentx-golden \
    --suite "${RSI_SUITE:-timing}" "${dispatch_arguments[@]}" "${coding_arguments[@]}" "${profile_arguments[@]}"
