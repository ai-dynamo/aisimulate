#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=glm-rsi-historical-image
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
cd "$storage_root"
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
export ENROOT_DATA_PATH="$CAMPAIGN_ROOT/historical-enroot-$SLURM_JOB_ID"
export TMPDIR="$CAMPAIGN_ROOT/historical-tmp-$SLURM_JOB_ID"
mkdir -p "$ENROOT_DATA_PATH" "$TMPDIR" "$CAMPAIGN_ROOT/historical-preparation"
image="$storage_root/images/sglang-0520-b27fce60.sqsh"
if ! test -s "$image"; then
    enroot import --arch x86_64 -o "$image.partial-$SLURM_JOB_ID" \
        'docker://registry-1.docker.io#lmsysorg/sglang@sha256:b27fce60bc5494c118c4910702812bcfa8cee67abcdd1ff8b0902f21647552f4'
    mv "$image.partial-$SLURM_JOB_ID" "$image"
fi
sha256sum "$image" > "$CAMPAIGN_ROOT/historical-preparation/image.sha256"
root="${CAMPAIGN_ROOT/#$storage_root/\/workspace}"
srun --ntasks=1 --container-image="$image" --container-writable \
    --container-mounts="$storage_root:/workspace" bash -c 'set -euo pipefail
      root="$1"
      uname -m > "$root/historical-preparation/architecture.txt"
      python3 -c "import importlib.metadata,json; print(json.dumps({k:importlib.metadata.version(k) for k in [\"sglang\",\"torch\",\"transformers\"]}))" > "$root/historical-preparation/runtime.json"
      python3 "$root/code/instrument.py" > "$root/historical-preparation/instrumentation.json"
      python3 -m sglang.launch_server --help > "$root/historical-preparation/server-help.txt"
    ' bash "$root"
