<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance-data collector

The collector measures operations or complete forward passes in a pinned runtime
and publishes data with its execution identity. Op-level timing consumes the
operation tables; whole-forward FPM consumes a separate validated Parquet/metadata
pair. Collection success, data coverage, and serving accuracy are distinct gates.

- [Upgrade a runtime](upgrade.md): scope, runtime proof, smoke, full collection, and delivery.
- [Data format and identity](data-format.md): case keys, physical rows, head axes, provenance, and reuse.
- [Review data](reviewing-data.md): Parquet text diffs, content hashes, and row changes.
- [DeepSeek-V4 attention modules](deepseek-v4.md): CSA/HCA measurement boundaries and calibration.
- [MoE sampling](moe-sampling.md): expert quotas and token assignment.
- [FPM self-service](../fpm-self-service/README.md): collect/import whole-forward data.

Read the repository [collector rules](../../../AGENTS.md#required-collector-first-step)
before changing collector code, cases, or the data contract. The executable
[collector README](../../../python/aisimulate/collector/README.md) documents current
commands; `framework_manifest.yaml` pins runtimes and `op_backend_catalog.yaml`
maps registry operations to storage families.

Run collector commands from `python/aisimulate/` unless a command explicitly uses
a repository-root path. Keep campaign checkpoints and raw logs in a run-specific
output directory, separate from published data. The packaged data root is
`python/aisimulate/src/aisimulate_core/systems/data/`.
