#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=glm-rsi-forward-geometry
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=00:20:00
set -euo pipefail
: "${RSI_CODE:?Immutable qualified collector directory}"
: "${RSI_PROFILE_PREFLIGHT:?Actual-runtime profiler receipt}"
: "${RSI_OP_FORWARD_REMAINING_GPU_SECONDS:?Reconciled op/forward pool}"
test "$RSI_OP_FORWARD_REMAINING_GPU_SECONDS" -ge 10080
export RSI_SUITE=forward-profile
export RSI_RUNTIME=historical
export RSI_PROFILE_TOOL=torch
export RSI_FORWARD_GEOMETRY=1
exec bash "$RSI_CODE/collect.sbatch"
