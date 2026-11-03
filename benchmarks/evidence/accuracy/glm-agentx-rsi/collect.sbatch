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
# Reserve the wall limit plus a minute for scheduler/GPU cleanup.
test "$REMAINING_GPU_SECONDS" -ge 36480
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
if test "${RSI_SUITE:-timing}" = frontend; then
    test -s "$CAMPAIGN_ROOT/frontend-template-preflight.json"
fi
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
export ENROOT_DATA_PATH="$CAMPAIGN_ROOT/enroot-$SLURM_JOB_ID"
mkdir -p "$ENROOT_DATA_PATH" "$CAMPAIGN_ROOT/gpu-$SLURM_JOB_ID"
cp -R "$CAMPAIGN_ROOT/code" "$CAMPAIGN_ROOT/code-$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$CAMPAIGN_ROOT/gpu-$SLURM_JOB_ID/gpus.csv"
root="${CAMPAIGN_ROOT/#$storage_root/\/workspace}"
srun --ntasks=1 --container-image="$image" \
    --container-writable --container-mounts="$storage_root:/workspace,$model:/model:ro" \
    env "RSI_CACHE_ROOT=$root/runtime-cache/$cache_key" \
    python3 "$root/code-$SLURM_JOB_ID/collect.py" --model /model \
    --output "$root/gpu-$SLURM_JOB_ID/controls" --seconds 4200 --acceptance-policy agentx-golden \
    --suite "${RSI_SUITE:-timing}"
