#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=glm-rsi-submit-matrix
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=00:45:00
set -euo pipefail
: "${RSI_CODE:?Immutable qualified collector directory}"
export RSI_SUITE=matrix
export RSI_RUNTIME=historical
exec bash "$RSI_CODE/collect.sbatch"
