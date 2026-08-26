<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate E2E Accuracy Overview

This directory contains a public-safe, static overview of matched end-to-end
AISimulate accuracy evidence. It intentionally does not include the exploratory
E2E Gym interface, raw measurements, internal run identifiers, or internal
service links.

## Published snapshot

The checked-in `summary.json` is derived from the public
[SemiAnalysis InferenceX `db-dump/2026-07-20` release](https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-07-20)
and a completed AISimulate 0.12.0 engine-replay evaluation. The summary records
the source digests and artifact versions needed to identify that exact evidence.

The first public view excludes multi-node configurations. It reports:

- client-observed TTFT and TPOT MAPE separately;
- per-topology TTFT and TPOT curve-shape error separately;
- successful prediction coverage and explicit failure counts;
- model, workload, GPU, framework, and precision dimensions; and
- AIC compatibility predictions as a separately selectable baseline.

These values are evidence for exact measured operating points, not a universal
support claim or release gate.

## Regeneration

The publisher supplies three validated inputs from the accuracy-evidence
pipeline:

1. the merged prediction rows;
2. the matching AISimulate evidence metadata; and
3. the matching coverage report.

Generate the sanitized aggregate:

```bash
python scripts/build_e2e_accuracy_overview.py \
  --predictions /path/to/predictions.json \
  --metadata /path/to/aisimulate_points.meta.json \
  --coverage /path/to/coverage.json \
  --source-url https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-07-20 \
  --output python/aisimulate/docs/e2e-accuracy/summary.json
```

The builder fails closed when release tags, row counts, AIC revisions, or the
completed AISimulate run identity disagree. Its output contains aggregates and
public provenance only; raw latency values and internal run identifiers are not
copied.

Run the focused contract tests with:

```bash
python -m pytest -q tests/test_e2e_accuracy_overview.py
```

Serve the documentation tree locally to inspect the page:

```bash
python -m http.server 8000 --directory python/aisimulate/docs
```

Then open `http://127.0.0.1:8000/e2e-accuracy/`.
