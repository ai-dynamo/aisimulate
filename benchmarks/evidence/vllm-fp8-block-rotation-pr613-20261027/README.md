<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-FileCopyrightText: Copyright (c) 2026 Jan Fostier
SPDX-License-Identifier: Apache-2.0
-->

# PR #613: reproducible block-FP8 rotation evidence

This capsule addresses CodeRabbit's request for actual before/after collector timings. It contains two distinct experiments. The fresh comparison measures the exact latency passed to `log_perf` with the collector's normal settings. The historical experiment compares profiled kernels in fixed-length graphs with the same kernels inside a real model graph. These measurements answer different questions.

The controlled kernel experiment supports the cache mechanism. The fresh comparison does **not** establish that the production collector's published latencies are more accurate. Most published row latencies decrease with the adaptive policy, so the earlier kernel-duration benefit must not be generalized to the production timing path without qualification.

## Fresh direct comparison

- Base: `cd6efc5e3cac59d7a6d63c32e32210077d6c7d58` (single-copy block-FP8).
- PR: `64489fe6d79008fdffa474571404c74dc2597520` (adaptive count).
- Hardware/runtime: NVIDIA H100 NVL, 60 MiB L2, vLLM 0.24.0, PyTorch 2.11.0+cu130. Full installed-package inventory, runtime source hashes, device details, clocks and power-limit observations are in `direct/*-environment.json` and `direct/*-gpu-after.json`.
- Unmodified production timing arguments: three warmups, six timed replays, one repetition per graph, CUDA graphs required, power measurement disabled. Graph operation count follows the actual allocation count; it is not equalized between arms.
- Three fresh processes per arm, seven collector calls per shape/start: **504 published latencies** across 12 shapes. Arm order alternates across starts; paired arms share seeded shape order and initial inputs. The protocol was saved before launch. All six processes exited zero.
- Only the publication sink is replaced with JSONL. The real timing helper is wrapped to observe settings, allocations and results; no allocation count, kernel, warmup count or graph length is overridden. Each worker imports the entire collector package extracted from its exact Git revision. The timing helper and vLLM utility sources match across arms.
- Statistics: median within a start, then median across three starts. All observations are retained. There are zero missing/invalid/duplicate observations and zero samples flagged by the predeclared rule (>15% from the within-start median). These three starts do not establish a population confidence interval.

| M | N x K | Copies | Base (us) | PR (us) | Change |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 6144 x 2560 | 6 | 11.061 | 10.482 | -5.24% |
| 1 | 2560 x 4096 | 6 | 11.771 | 11.024 | -6.34% |
| 1 | 19456 x 2560 | 3 | 25.867 | 24.466 | -5.42% |
| 1 | 2560 x 9728 | 6 | 19.381 | 18.664 | -3.70% |
| 16 | 6144 x 2560 | 6 | 10.677 | 10.621 | -0.52% |
| 16 | 2560 x 4096 | 6 | 11.643 | 11.078 | -4.85% |
| 16 | 19456 x 2560 | 3 | 26.235 | 25.013 | -4.66% |
| 16 | 2560 x 9728 | 6 | 19.504 | 18.935 | -2.92% |
| 64 | 6144 x 2560 | 6 | 9.771 | 10.346 | +5.89% |
| 64 | 19456 x 2560 | 3 | 25.552 | 24.578 | -3.81% |
| 1 | 8448 x 8192 | 2 | 30.416 | 29.675 | -2.44% |
| 16 | 16384 x 8192 | 1 | 50.128 | 50.197 | +0.14% |

`analysis/direct.csv` and `analysis/summary.json` contain the full-precision results and per-start medians. `direct/*-measurements.jsonl` contains every observed published and loop latency, allocation count, distinct-storage count, graph status, seed and selected kernel label. Dispatch labels match between arms at every shape.

These are complete collector row latencies, including activation quantization, GEMM and graph/event/submission effects. Changing the number of allocations also changes graph length. A decrease here can coexist with an increase in the isolated GEMM kernel duration. This experiment cannot attribute the difference uniquely to caching or graph overhead. The one-copy control above twice L2 changes by only about +0.14%; the M64 QKV point increases by about +5.89%. Both are retained alongside the decreases.

## Historical model-kernel comparison

`historical/` preserves the prior fixed-72-operation sweep: 201 isolated cases and six model cases across three independent process starts, with **207 raw compressed CUDA traces** in `cuda-traces.tar.gz`. The complete shape/copy grid includes the eight Qwen projection shapes, counts 1/2/3/4/6/8/12/24, and a 96 MiB negative-control matrix at counts 1/2/4.

Reference: Qwen/Qwen3-4B-FP8 at immutable revision `96b30dc13593a244a5e59e84687309f53c375cfa`, TP=1, real checkpoint weights and prefilled KV. Model cases are batch/context 1/128 and 16/512. Captured logits exactly match the runtime reference at all six observations.

The CPU analysis reads the raw traces, checks their hashes, verifies complete shape/copy coverage, requires 180 model and 360 isolated kernel observations per shape/start, and matches exact CUDA kernel names, grids and block dimensions. It selects this PR's capped copy counts from the sweep and recomputes the following values:

| Measurement | Single-copy MAPE | PR-count MAPE | Shapes improved |
| --- | ---: | ---: | ---: |
| GEMM kernel duration | 9.067856% | 1.518637% | 8/8 |
| Quantization plus GEMM active interval union | 8.023906% | 2.275138% | 8/8 |

MAPE is `mean(abs(isolated / model - 1)) * 100`, with equal weight for each of the eight shape points. The median is taken within each independent start, then across starts. No kernel or timing observation is excluded. Full per-shape values and per-start medians are regenerated into `analysis/historical.csv` and `analysis/summary.json`.

This reproduces the earlier **controlled kernel** claim; it is not new model validation of the shorter production graphs. No L2 hit-rate counters were collected. The two-L2 target remains a heuristic, and tiny matrices can be limited by the six-copy cap.

The earlier uncapped-policy full-forward experiment remains negative: decode MAPE goes from 5.823069% to 10.686789%. `historical/full-forward-comparison.json` contains the recorded comparison points, and the analysis checks that arithmetic. These are historical derived points, not a new forward run or a result for the capped PR policy. No improved overall prediction accuracy is claimed.

## Reproduce

Run from the repository root. CPU reanalysis needs only the Python standard library and makes no GitHub or GPU calls:

```bash
capsule=benchmarks/evidence/vllm-fp8-block-rotation-pr613-20261027
python3 "$capsule/analyze.py" --output /tmp/pr613-reanalysis
cd "$capsule"
sha256sum -c SHA256SUMS
```

The analyzer fails on missing, duplicated or nonfinite/nonpositive direct measurements, unexpected allocation counts, broken normalization, runtime/source drift, invalid model geometry/parity, incomplete raw trace coverage or kernel-identity mismatches. Outlier flags are diagnostic; observations are never dropped. The integrity checks were also exercised with deliberate missing, duplicate, negative, wrong-kernel and outlier mutations; see `analysis/validation.json`.

For a fresh GPU comparison, use Python 3.12 with the pinned runtime above and a CUDA development toolkit available to DeepGEMM. Both immutable source revisions must be available in the local Git object database (the original PR head is intentionally pinned even if the PR is amended later). If needed, fetch it with `git fetch https://github.com/jfostier/aisimulate.git 64489fe6d79008fdffa474571404c74dc2597520`. The runner refuses a non-idle GPU or an existing output directory.

The exact campaign command on this host was:

```bash
module load CUDA
export PATH=/tmp/aisimulate-forward-ab/venv/bin:$PATH
/tmp/aisimulate-forward-ab/venv/bin/python   benchmarks/evidence/vllm-fp8-block-rotation-pr613-20261027/run.py   --output benchmarks/evidence/vllm-fp8-block-rotation-pr613-20261027/direct
```

To reproduce without overwriting the archived observations, use a new `--output`, such as `/tmp/pr613-fresh`, then analyze it with `analyze.py --direct /tmp/pr613-fresh --output /tmp/pr613-fresh-analysis`. Successful collection checks include graph replay, distinct weight storage, six actual timed runs and the unmodified published-latency normalization.

`direct/runner-source.tar.gz` preserves the exact executed runner bytes, matching the SHA-256 recorded by all workers. The top-level executable adds the repository-required NVIDIA SPDX copyright line; the analyzer verifies that its Python AST is identical to the executed snapshot. `historical/collection-sources.tar.gz` preserves the prior collection scripts and pre-launch protocol byte-for-byte; their frozen hashes are checked during analysis. The historical scripts retain their original `/tmp/aisimulate-rotation-case` paths and require the pinned model checkpoint there; they are provenance inputs, not part of the fresh direct comparison. No framework source is vendored or modified.

## Other validation and limits

`memory-checks/` retains the earlier real-collector smoke logs and original harness: six-copy failure versus one-copy success under a 26 GiB allocator quota for a 1 GiB matrix, a 66 MiB/two-copy case, and small six-copy FlashInfer/DeepGEMM cases. The six-copy failure is a guard against an unconditional-rotation variant, not a failure of the single-copy base collector. These checks are separate from the fresh paired campaign.

The evidence covers H100 NVL/vLLM 0.24.0. Blackwell, other runtime versions and NVFP4 rotation remain unverified. Unit tests use synthetic timing to verify control flow, whereas this capsule contains GPU measurements. No packaged profiles, model coefficients or production collector code are changed by this evidence update.

## Local checks

The existing 22 focused collector tests pass on the unchanged production source in `64489fe6`. The application-wide conftest is excluded because this collector environment does not have the AISimulate application package installed. This is not a full-suite or hosted-CI result.

```bash
/tmp/aisimulate-forward-ab/venv/bin/python -m pytest -c python/aisimulate/pytest.ini --confcutdir=python/aisimulate/tests/unit/collector python/aisimulate/tests/unit/collector/test_vllm_gemm_graph_timing.py -m unit
/tmp/aisimulate-forward-ab/venv/bin/python -m ruff check --config python/aisimulate/pyproject.toml benchmarks/evidence/vllm-fp8-block-rotation-pr613-20261027/{run,analyze,validate}.py python/aisimulate/collector/vllm/collect_gemm.py python/aisimulate/tests/unit/collector/test_vllm_gemm_graph_timing.py
/tmp/aisimulate-forward-ab/venv/bin/python -m ruff format --check --config python/aisimulate/pyproject.toml benchmarks/evidence/vllm-fp8-block-rotation-pr613-20261027/{run,analyze,validate}.py python/aisimulate/collector/vllm/collect_gemm.py python/aisimulate/tests/unit/collector/test_vllm_gemm_graph_timing.py
python3 benchmarks/evidence/vllm-fp8-block-rotation-pr613-20261027/validate.py
```

Results: 22 tests passed; Ruff lint/format passed on five files; six evidence-reader validation scenarios passed. The runner's and analyzer's numerical behavior were not changed to obtain a preferred benchmark direction. SHA256SUMS covers all packaged source, raw measurements, logs, archived traces, provenance and computed reports; it excludes itself and Python bytecode caches.
