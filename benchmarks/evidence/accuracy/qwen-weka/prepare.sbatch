#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=qwen-weka-prepare
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
image="$storage_root/images/vllm-0310-cu130-a4a4c043.sqsh"
cd "$storage_root"
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_DATA_PATH="$CAMPAIGN_ROOT/preparation/enroot"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
mkdir -p "$ENROOT_DATA_PATH" "$CAMPAIGN_ROOT/preparation"
srun --ntasks=1 --container-image="$image" --container-mounts="$storage_root:/workspace" \
    bash -c 'set -euo pipefail
      root="${1/#\/lustre\/fsw\/portfolios\/coreai\/projects\/coreai_comparch_inferencex\/users\/simonec/\/workspace}"
      export HF_HOME="$root/preparation/hf" PIP_CACHE_DIR="$root/preparation/pip"
      python3 -m venv "$root/client"
      "$root/client/bin/pip" install "agentx-harness==1.0.12"
      "$root/client/bin/python" "$root/code/client.py" --prepare "$root/preparation"
      "$root/client/bin/pip" freeze > "$root/preparation/client-packages.txt"
      sha256sum /workspace/images/vllm-0310-cu130-a4a4c043.sqsh > "$root/preparation/image.sha256"
      test "$(cut -d " " -f 1 "$root/preparation/image.sha256")" = 35b6881618f1ab48ef054126d659e18aebf2a009dd918b726f4220b32a274eae
      "$root/client/bin/agentx" profile --help > "$root/preparation/profile-help.txt"
    ' bash "$CAMPAIGN_ROOT"
