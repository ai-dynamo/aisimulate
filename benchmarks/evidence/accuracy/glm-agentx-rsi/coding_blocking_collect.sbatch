#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=glm-rsi-coding-block
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=00:33:00
set -euo pipefail
: "${RSI_CODE:?Immutable collector source}"
: "${RSI_CODING_INPUTS:?Qualified E029 private inputs}"
: "${RSI_DISPATCH_PREFLIGHT:?Versioned exact-image CPU receipt}"
export RSI_SUITE=coding-blocking RSI_RUNTIME=historical
exec bash "$RSI_CODE/collect.sbatch"
