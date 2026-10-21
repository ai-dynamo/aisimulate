<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR533 native-reader compatibility

`compare_readers.py` loads the actual PR533 and candidate extension modules in
separate Python processes. It passes candidate-normalized JSON back through both
native readers; no older schema is reconstructed in Python. The baseline binary
must match the SHA-256 in the supplied PR533 validation receipt, which records
implementation revision `1de653c9ef28788a877a660cb81cb3379edb8c6a`.

The probe checks:

- Twelve ordinary configurations: auto, op-level, interpolation, and regression,
  each with omitted, empty, or explicit-default selector controls. Regression
  also constructs through `best_available`; other ordinary modes test schema
  compatibility without requiring operation/profile data.
- Nine standalone learned configurations: frozen artifacts, frozen training
  checkpoints, and online training checkpoints, each with the same three forms
  of default selector controls. Both binaries construct, predict, accept eight
  labels, and export. Online models flush their pending updates. Normalized JSON,
  prediction trajectories, and exported artifacts must match exactly.
- Six causal-selector configurations: default controls, each individual
  nondefault control, and combined custom controls. Candidate construction and
  serialization roundtrips must succeed; PR533 must reject the new feature
  explicitly. Nondefault controls must survive serialization unchanged.
- Runtime checkpoint continuation in both directions. Each binary creates a
  separate snapshot after five accepted labels, with four fitted rows and one
  pending label. Both binaries then load each exact snapshot and process the
  remaining 251 labels. Their full 256-query predictions, exported artifacts,
  and semantic diagnostics must match the snapshot origin's uninterrupted run.
  Only `history_grouping_ms` and capacity-based `training_state_bytes` are
  excluded from diagnostics. The run uses grouped history, one mutable tail
  tree, batches of four, a seven-row recent window, and an age bound of nine.
  It verifies pending-label preservation, reconstruction of the frozen-prefix
  cache, unchanged frozen trees, bounded retention, and unchanged snapshot files.

The fixture is the checkpoint and 256 queries produced by the maintained
`benchmarks/evidence/fpm_online_publication_v1/benchmark.py prepare` command.
The 10,000-row fixture exercises actual history compression; the 256-row fixture
is a small smoke case with grouping enabled but unique groups.
The output receipt records all input hashes, both native binary hashes, the
candidate revision, and compact per-case outcomes. This is compatibility and
behavior evidence, not a serving-accuracy or performance measurement.

```sh
python benchmarks/evidence/fpm_online_publication_v1/benchmark.py \
  prepare /tmp/pr534-compatibility-fixture --history-rows 10000
python benchmarks/evidence/fpm_pr534_review_v1/compatibility/compare_readers.py \
  --candidate /absolute/path/to/candidate/_runtime.abi3.so \
  --baseline /absolute/path/to/pr533/_runtime.abi3.so \
  --fixture /tmp/pr534-compatibility-fixture \
  --baseline-validation benchmarks/evidence/fpm_pr533_review_v1/validation.json \
  --candidate-revision IMPLEMENTATION_COMMIT \
  --runtime-dir /tmp/pr534-runtime-checkpoints \
  --out /tmp/pr534-native-reader-compatibility.json
```

Use the recorded PR533 binary for its existing validation receipt. When rebuilding
PR533 on another host, create a receipt that pins that build's actual binary hash
and source revision before running the comparison.
