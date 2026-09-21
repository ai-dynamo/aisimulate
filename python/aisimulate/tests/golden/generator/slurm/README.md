<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Slurm bundle fixtures

These are static, complete artifact expectations for `test_all_backends_render_standalone_slurm_bundle`. They use the test's explicit TP2/PP1/DP1 worker configuration with Dynamo 1.2.0 and the backend version in each directory. Aggregated jobs allocate two GPUs; disaggregated jobs allocate four, with one prefill and one decode worker.

YAML literal blocks preserve every emitted byte, including final newlines. `common.yaml` contains the identical supervisor, submission, environment, and benchmark scripts. `agg.yaml` and `disagg.yaml` contain the allocation-specific batch scripts. Backend/version fixtures contain deployment commands and any engine files. Their union is the complete bundle; tests never refresh these files or read production templates as expected values.

Review fixture changes against the intended command, allocation, process lifecycle, and backend behavior before accepting them. Separate semantic tests exercise MoE GPU accounting, prefix caching and router events, MTP configuration, and unsupported prompt lookup. These fixtures are generated-code regression evidence, not Slurm or GPU runtime qualification.
