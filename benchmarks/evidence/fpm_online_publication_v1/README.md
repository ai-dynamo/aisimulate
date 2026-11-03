<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Online binned GBDT publication source

This implementation extracts AISimulate's native online learner from the local
`aisimulate-grouped-history-v1` research snapshot. That directory has no Git
metadata and the local `codex/fpm-online-gbdt` branch identifies only its original
base, `127f0eed39754adad20cbf890f2a9cc1f19c7d61`. It is not a published online
implementation commit. [source-import.json](source-import.json) records exact
source hashes and their agreement with the existing sealed workload experiment.
The seed exporter and independent reference tests were checked directly from the
same snapshot; those test files were not included in the earlier source seal.

The integration preserves the current feature attribution, `fpm5` feature preset,
rank-local counter protection, duplicate-feature checks, and PR493 hybrid
calibration. It imports only online fitting and seed export, removes the earlier
experimental hybrid gate and dynamic recent-row sampler, and keeps FIFO recent
observations. Grouping historical rows by their complete fitted-bin vectors
preserves multiplicity and target sums; it does not replace the fitting population
with equally weighted bin representatives. Immutable grouped anchors are shared
across clones. Prediction performs no fitting; synchronous fitting occurs only
when an accepted per-store batch is full or explicitly flushed. Runtime
checkpoint version 5 rejects the earlier research version 4 binary layout.

The mathematical inspiration is Lin, Chung, Lao, and Zhao,
[Online Gradient Boosting Decision Tree: In-Place Updates for Efficient
Adding/Deleting Data](https://arxiv.org/abs/2502.01634v1). The reference repository
is [huawei-lin/InplaceOnlineGBDT at
1d07be28cc9178ed559e149291aad96e30ffe4ee](https://github.com/huawei-lin/InplaceOnlineGBDT/tree/1d07be28cc9178ed559e149291aad96e30ffe4ee)
(Apache-2.0). The AISimulate Rust updater and reference tests were independently
written; no source, fixtures, or datasets from that repository are imported.
Sklearn initialization, fixed bins/baseline/capacity, scalar squared error,
weighted historical/recent populations, transactional publication, and periodic
eager refresh are local adaptations. The paper's theoretical guarantees and
accuracy results are not claimed for these adaptations.

Existing feature-definition attribution remains in `learned.rs`,
`fpm_learned.py`, and the canonical third-party notices. Generated research HTML,
PDFs, MathJax output, and measured reports are not imported.

Publication also removes one avoidable fitting scan: when an eager structural
update rebuilds a region, it reuses that region root's already-current gradient
sums and histogram. The original eager pass accumulated the same members in the
same order. Descendant statistics and split-candidate resampling remain unchanged.
The focused Rust suite includes independent regularized-loss, grouped-history,
rollback, and continuation oracles; timing and native Python qualification are
recorded separately from these correctness checks.

Canonical loading hashes and parses the same small manifest/artifact byte buffer.
Large state and Parquet payloads are hashed and consumed through the same opened
file handle, preserving their identity across atomic path replacement. Input
checkpoint files must remain immutable while loading; concurrent in-place file
mutation is unsupported. Checkpoint publication reserves its destination with an
atomic directory creation before installing the completed staging directory, so
concurrent writers cannot replace an existing empty destination. Failed installs
remove only the caller's still-empty reservation.
