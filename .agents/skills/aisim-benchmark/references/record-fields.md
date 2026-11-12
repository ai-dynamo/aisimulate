<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Where each guideline row is recorded

The rows come from the [benchmark guidelines](../../../../docs/perf-model/collector/README.md#benchmark-guidelines).
This table only says where the library keeps the answer.

| Guideline row | Stored in |
| --- | --- |
| Effective settings | `Identity.effective_settings`, `Identity.precision`, `Identity.parallelism` (observed); `Manifest.requested_settings` (requested) |
| Workload, warmup and cache policy | Manifest `provenance`; the E2E observation table; `Assumption` entries of kind `truth` |
| Measurement boundaries, rank handling | Manifest `provenance`; `Assumption` entries; the truth table's `inputs_json` |
| GPU time | `GpuTime` on the run and on `EvalRecord.gpu_time` |
| Provenance | `Artifact` (path, SHA-256, bytes, role), `Manifest.provenance`, `EvalRecord.producer`, `EvalRecord.findings` |
| Failures and exclusions | `Results.exclusions`, `EvalRecord.status` and `status_reason`, `Manifest.evidence_gaps` |

An answer with no field above goes into the PR provenance section or a linked
artifact. A value you cannot observe is left null and listed as an evidence gap.
