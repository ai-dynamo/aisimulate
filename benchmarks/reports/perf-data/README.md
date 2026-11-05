<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance-data scan reports

These reports are generated data snapshots, not authoritative descriptions of
the current resolver or support matrix. They have been moved without editing
their tables or re-running measurement campaigns.

- [Kernel-source audit](op-kernel-sources.md): retained earlier scan.
- [Reuse analysis](perf-data-reuse-analysis.md): generated reuse report.

To generate a fresh report without replacing package data, run from the repo
root in the development environment:

```bash
python python/aisimulate/tools/perf_database/generate_perf_data_reuse_manifest.py \
  --data-root python/aisimulate/src/aisimulate_core/systems/data \
  --out-json /tmp/perf-data-reuse-analysis.json \
  --out-md benchmarks/reports/perf-data/perf-data-reuse-analysis.md
```

Updating the runtime manifest is a separate operation using `--out-manifest`;
review its effect on data reuse rather than treating a report refresh as
permission to change estimator inputs.
