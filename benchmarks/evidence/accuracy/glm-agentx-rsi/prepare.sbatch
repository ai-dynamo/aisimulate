#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=glm-rsi-prepare
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
cd "$storage_root"
set +x
if [[ -z "${HF_TOKEN:-}" ]]; then
    set +u
    source ~/.bashrc
    set -u
fi
export HF_HOME="$storage_root/.cache/huggingface"
export HF_XET_CACHE="$storage_root/.cache/huggingface/xet"
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$CAMPAIGN_ROOT/preparation" "$storage_root/models"
model="$storage_root/models/GLM-5.3-NVFP4-e3b8f9f2"
hf download nvidia/GLM-5.3-NVFP4 --revision e3b8f9f2fad29727dfa95e766c5cac18ea1414bf --local-dir "$model"
test -s "$model/config.json"
test -s "$model/model.safetensors.index.json"
find "$model" -maxdepth 1 -type f -name '*.safetensors' -print0 | sort -z | xargs -0 sha256sum > "$CAMPAIGN_ROOT/preparation/weights.sha256"
sha256sum "$model/config.json" "$model/tokenizer_config.json" > "$CAMPAIGN_ROOT/preparation/config.sha256"
printf '%s\n' 'nvidia/GLM-5.3-NVFP4@e3b8f9f2fad29727dfa95e766c5cac18ea1414bf' > "$CAMPAIGN_ROOT/preparation/model-revision.txt"
