#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=qwen-weka-preflight
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
cd "$storage_root"
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_DATA_PATH="$CAMPAIGN_ROOT/preflight-enroot-$SLURM_JOB_ID"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
mkdir -p "$ENROOT_DATA_PATH"
srun --ntasks=1 --container-image="$storage_root/images/vllm-0310-cu130-a4a4c043.sqsh" \
  --container-mounts="$storage_root:/workspace,$storage_root/models/Qwen3.5-397B-A17B-84726181:/model:ro" \
  bash -c 'set -euo pipefail
    root="${1/#\/lustre\/fsw\/portfolios\/coreai\/projects\/coreai_comparch_inferencex\/users\/simonec/\/workspace}"
    export HF_HOME="$root/preparation/hf" HF_HUB_OFFLINE= TRANSFORMERS_OFFLINE= HF_DATASETS_OFFLINE=1
    export PYTHONPATH="$root/code/client_bootstrap"
    export AGENTX_COLLECTION_RECORD="$root/preflight-$SLURM_JOB_ID"
    "$root/client/bin/python" "$root/code/client.py" --preflight --record "$AGENTX_COLLECTION_RECORD" -- \
      profile --url http://127.0.0.1:8800 --model Qwen/Qwen3.5-397B-A17B --tokenizer /model \
      --concurrency 14 --benchmark-duration 600 --benchmark-grace-period 30 --random-seed 42 \
      --max-context-length 262144 --cache-bust none --unsafe-override --tokenizer-trust-remote-code \
      --server-metrics http://127.0.0.1:8800/metrics --output-artifact-dir "$AGENTX_COLLECTION_RECORD/benchmark" \
      --public-dataset semianalysis_cc_traces_weka_062126 --export-level raw --extra-inputs return_token_ids:true
  ' bash "$CAMPAIGN_ROOT"
