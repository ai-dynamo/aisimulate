<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCR quickstart validation

Validated on 2026-11-04, Linux x86-64, Python 3.12.11, CPU-only.

## Environment and inputs

- Dynamo Python package and native runtime: clean merged source at
  `147b78a8aa6ffc8d11baa06c1f589bbfa46b7523` (PR #17856).
- AISimulate: `0.13.0.dev202610290000000087`, as pinned by that Dynamo revision.
- Native runtime: `maturin develop --uv --features ais-forward-pass`, Rust 1.96.1,
  debug build. `uv pip check` passed for all 127 installed packages.
- Configuration: [dynamo-kvcr-predict.yaml](../../examples/cli/dynamo-kvcr-predict.yaml).
  Validation used the same parsed configuration with an absolute trace path.
- Trace: unmodified [Dynamo 1,000-request fixture](https://github.com/ai-dynamo/dynamo/blob/280df2a1e4166917a293f3c50d28e54b3ab348ab/lib/bench/testdata/mooncake_trace_1000.jsonl),
  Git blob `ebf55fdabc577f3f486e499b12c6050ce1dbf9f0`, 187,043 bytes.
  It has 59 arrival timestamps spanning approximately 177 seconds.

## Commands

From the repository root with the downloaded trace and the environment above:

```bash
aisimulate predict --stack dynamo -c examples/cli/dynamo-kvcr-predict.yaml \
  --output-dir kvcr-on --format json
aisimulate predict --stack dynamo -c examples/cli/dynamo-kvcr-predict.yaml \
  --output-dir kvcr-off --format json \
  --set engine.workers.aggregated.kv_cache.kvcr=null
```

## Results

| Case | Completed requests | Successful / partial pulls | Loaded blocks | Pulled bytes | Mean simulated TTFT (ms) |
|---|---:|---:|---:|---:|---:|
| KVCR on, run 1 | 1,000 | 1 / 3 | 579 | 2,428,502,016 | 89,290.98722840915 |
| KVCR on, run 2 | 1,000 | 1 / 3 | 579 | 2,428,502,016 | 89,290.98722840915 |
| KVCR off | 1,000 | Not emitted | Not emitted | Not emitted | 91,523.67001583803 |

The existing AISimulate `tests/sweeper/data/mooncake_tiny.jsonl` completed all
16 requests but produced zero pulls in the same clean-source environment.
The tested public `ai-dynamo==1.7.0.dev20261103` environment also completed
the larger trace with zero pulls, motivating the explicit router-hint prerequisite.

These results establish that this fixture exercises peer reuse through the
Dynamo adapter. They do not establish GPU fidelity or general speedup. The
capacity, bytes per token, and transfer settings are user-specified assumptions.
Do not use this debug build's host execution time as a release performance result.
