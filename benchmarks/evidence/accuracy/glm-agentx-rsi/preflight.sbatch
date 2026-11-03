#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=glm-rsi-preflight
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
cd "$storage_root"
test -s "$CAMPAIGN_ROOT/preparation/model-revision.txt"
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
export ENROOT_DATA_PATH="$CAMPAIGN_ROOT/preflight-enroot-$SLURM_JOB_ID"
mkdir -p "$ENROOT_DATA_PATH"
root="${CAMPAIGN_ROOT/#$storage_root/\/workspace}"
srun --ntasks=1 --container-image="$storage_root/images/sglang-0521-2dbe4c7f.sqsh" \
    --container-writable --container-mounts="$storage_root:/workspace" \
    bash -c 'set -euo pipefail
      root="$1"
      python3 "$root/code/instrument.py" > "$root/preparation/instrumentation-preflight.json"
      python3 -m sglang.launch_server --help > "$root/preparation/server-help.txt"
      python3 "$root/code/probe.py" --help > "$root/preparation/probe-help.txt"
      python3 -c "import importlib.metadata,json; print(json.dumps({k:importlib.metadata.version(k) for k in [\"sglang\",\"torch\",\"transformers\"]}))" > "$root/preparation/runtime-preflight.json"
    ' bash "$root"
