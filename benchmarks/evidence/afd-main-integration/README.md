<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AFD parity after main's request-metric change

The six native AFD golden cases now use the request populations introduced by main commit `394504f22dfd1abdac208925d659e1b2ba876ee1` (PR #741). The merge preserved that reporting implementation; this update brings the AFD expectations into agreement with it. These are simulation parity results, not measurements of serving accuracy.

The baseline expectations are from `1dc3be014ace2f93f256864711270d3f2f7bd5d6`. The build manifest and each case's generation record identify the runtime source, native binary, data tree, configuration, and command used for the updated expectations. Prior generation records remain in `historical_generations`.

## Why the values change

- ITL now has one sample per completed request: `(last_token_ms - first_token_ms) / (output_length - 1)`. Its distribution matches the existing per-request TPOT distribution instead of the distribution of every token gap.
- User TPS has one `1000 / ITL` sample per request with positive ITL. It is averaged after inversion; it is neither the reciprocal of the average ITL nor an average of every token gap's reciprocal.
- Percentiles now select exact nearest-rank samples instead of DDSketch approximations.

Across the six cases, 95 fields change beyond the existing `1e-6` relative / `1e-9` absolute tolerance. All are ITL or user-TPS statistics. Request lifecycle, token totals, GPU accounting, simulated duration, TTFT/TPOT, memory evidence, and power evidence remain unchanged within that tolerance. The refresh does not modify timing models, configurations, or performance data.

## Evidence

| File | Content |
| --- | --- |
| `modeling-deltas.json` | Every changed field's before/after value, with within-tolerance rounding differences listed separately. |
| `request-metric-checks.json` | A second CLI run with per-request capture for each case; recorded token endpoints and output lengths reproduce all 18 ITL/user-TPS statistics independently of native summary aggregation. |
| `build-manifest.json` | Complete runtime-source inventory, source/data tree identities, native build identity, binary digest, and build environment. |
| `build-manifest.source.tar.gz` | Deterministic immutable source snapshot verified by the generation harness. |

The reporting check computes means and population standard deviations with Python's `statistics.fmean` and `statistics.pstdev`. For percentile `p`, it selects sorted sample index `floor((n - 1) * p / 100 + 0.5)`. This checks the reporting calculation against replayed request endpoints; it is not a second performance-model oracle.

## Reproduce

Use a checkout containing these updated goldens and the runtime sources and data tree recorded in `build-manifest.json`, then build the native extension with the recorded command. Run the native golden gate from the repository root:

```sh
python/aisimulate/.venv/bin/python -m pytest -p no:timeout \
  -c python/aisimulate/pytest.ini -o addopts= -q \
  crates/core/parity_tests/perfmodel/test_afd_parity.py::TestAfdE2eGolden
```

For a new intentional refresh, create a fresh manifest with `scripts/afd_parity.py snapshot-build`, then run `refresh-e2e` with the six case IDs and an explicit reason. A newly built binary has its own digest; do not reuse this manifest for a different binary. Add `--capture-per-request` to each recorded prediction command to inspect the per-request endpoints independently.
