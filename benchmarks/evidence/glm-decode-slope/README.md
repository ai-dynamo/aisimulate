<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GLM decode context-length investigation

`diagnose_decode.py` queries the public native performance-model API for the
GLM-5.3-NVFP4 / B300 / SGLang 0.5.20 / TP8 identity used by AgentX point 446063.
It uses `op_level`, SILICON data, denied estimator fallback, and no online
correction. It performs no serving replay or GPU measurement.

Run with the environment containing the native build being investigated:

```bash
python benchmarks/evidence/glm-decode-slope/diagnose_decode.py \
  --label baseline --output-dir /tmp/glm-decode-baseline
```

The output directory must be new. `--expected-native-sha256` can enforce the
exact loaded extension. The manifest records its digest, imported source
checkout, source status, model files, B300 data files, effective constructor
provenance, and elapsed wall time. A label is not proof of native build revision.

Defaults query contexts 32k, 64k, 128k, 256k, 384k, 512k, and 600,000 tokens,
batches 1, 2, 4, 8, 13, and 14, and `nextn` 0 and 3. The binary `k` values are
multiples of 1,024. Override with `--contexts`, `--batches`, or `--nextn`.
Each coordinate is queried for the actual checkpoint and two synthetic model
copies: `all_full` and `all_skip`. Use `--variants checkpoint` for the actual
checkpoint alone. The complete default matrix has 252 queries.

- `native-queries.jsonl`: full native operation latency/source/SOL evidence.
- `summary.csv`: round latency, attention latency, and remaining operation time.
- `measured-dsa-rows.jsonl`: raw baseline DSA measurements for GLM, eight local
  heads, FP8 KV, and BF16 projections, preserving full/skip-indexer variants.
- `manifest.json`: identity, measured coordinate ranges, exact synthetic model
  edits, semantics, and status.
- `synthetic-models/`: full copies of the model and quantization configuration
  used by each ablation. Only `index_topk_pattern` changes: all `F` or all `S`.

`context_length` denotes the past context; the native static decode API queries
`context_length + 1`. One output-token pair (`output_tokens=2`) represents one
generation timing round. With MTP this includes the configured native speculative
composition; it is **not TPOT** and is not divided by assumed acceptance.
Every scalar is checked against the sum of its native per-operation values.

The checkpoint's public operation diagnostics blend full-indexer and reused-index
layers into `generation_attention`. The synthetic ablations ask Rust to price all
layers as full-indexer or reused-index layers through the same public constructor.
These are diagnostic model variants, not serving configurations or measurements.
Non-attention operation values must equal the checkpoint's values exactly at
every shared coordinate. All returned attention times include the native layer
count and MTP scaling; they are not per-layer table timings.

This script does not invent separate extrapolated costs.
Raw measured variants and table domains are preserved for comparison; they do
not prove which native lookup was executed. The baseline candidate table is
SGLang 0.5.14, while the requested runtime identity is 0.5.20. Inspect executed
lookup evidence before attributing an observed discrepancy to version reuse.

All timing comes from Rust through `RustForwardPassPerfModel.best_available` and
`static_phase_diagnostics` / `static_phase_latency`. Python only records and
checks returned values. No estimator math or substitute timing model is added.

## Replay comparison

`compare_replays.py` reuses the AgentX report's source-identity matching,
recorded-output-length eligibility, TPOT definition, and throughput accounting.
Run complete benchmark replays with the original configs and a new output root;
the expected layout is `point-ID/observed-replay/` and its sibling
`observed-replay-execution.json`.

```bash
python benchmarks/evidence/glm-decode-slope/compare_replays.py \
  --baseline-root /path/to/original/agentx-glm-concurrency \
  --candidate-root /path/to/corrected-replays \
  --output-dir /path/to/comparison
```

Use `--point-id 446063` for one point; omit it for all six. `--self-check` verifies
the common-cohort, lost/gained coverage, input-bin, and empty-subset semantics.
The baseline strict source identities define the cohort. Both sets of latency
errors are recomputed on its retained intersection; lost and newly eligible
requests are reported separately. Full-profile throughput uses each realized
run and must not be interpreted as a fixed-cohort statistic. Input-length bins
use decimal boundaries, matching the request-level investigation.

Keep the original configs, source corpus hashes, native build and source-patch
identities, execution logs, and wall times beside each replay. A dirty checkout
must retain its exact patch. Native timing curves are diagnostic predictions;
neither they nor a CPU replay establish new silicon accuracy measurements.

## Controlled correction results (2026-10-29)

The SGLang decode reuse-layer extrapolation correction improves TPOT and E2E MAPE
at all six GLM points. All 8,260 original strict source identities are retained.
Measured performance tables, full-indexer timings, and prefill formulas are unchanged.

| Concurrency | Matched requests | TPOT MAPE before → after | E2E MAPE before → after | Output throughput APE before → after |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 289 | 84.9% → 66.0% | 36.8% → 28.6% | 14.77% → 12.00% |
| 4 | 601 | 118.1% → 92.3% | 37.6% → 30.7% | 12.69% → 11.49% |
| 8 | 1352 | 135.1% → 87.1% | 49.8% → 32.2% | 5.17% → 2.37% |
| 10 | 1545 | 192.3% → 116.6% | 59.5% → 33.9% | 4.42% → 2.68% |
| 13 | 2137 | 189.0% → 172.9% | 85.4% → 55.5% | 3.48% → 7.11% |
| 14 | 2336 | 209.9% → 124.3% | 83.1% → 57.1% | 4.76% → 0.59% |

Latency errors use the fixed retained cohort; full-profile throughput uses each
realized cohort. C13 throughput error worsens. TTFT MAPE worsens at C1, C4, C8, C10,
and C14; it improves at C13. These regressions are preserved in `results/summary.json`.

At C1 with input ≥384k, mean TPOT changes from 6.95 to 6.11 ms against 2.93 ms
measured. At one request and 524,288 context, the predicted MTP timing round
changes from 23.7581 to 21.2078 ms (10.73%). The correction is partial: full-indexer
extrapolation and speculative verification remain unqualified by new silicon data.

Validation: 32 DSA Rust tests and 389 frozen-golden parity checks pass without
changing goldens. The 252-query matrices retain all 252 non-attention operation
maps, all 84 all-full variants, and all 21 batch-one/non-speculative variants.
The original measured DSA rows are unchanged. See `results/validation.json`.

The six replays used the original configs and recorded output lengths, including
preparation, 3,600-second profiling and grace. They ran two at a time, with individual
wall times of 112.9–119.3 seconds; this is CPU simulation, not a GPU benchmark.

## Figures

```bash
python benchmarks/evidence/glm-decode-slope/plot_diagnostics.py --root /path/to/evidence-root
```

The root contains `baseline-static/`, `corrected-static/` and optionally
`comparison/comparison.json`. Rendering produces PNG/SVG files from saved native
values and the fixed-cohort comparison. Attention ablations include layer/MTP
scaling; round latency is not TPOT. The original measured per-layer module curve
is shown separately. Retained figures and compact raw evidence are in `results/`.
