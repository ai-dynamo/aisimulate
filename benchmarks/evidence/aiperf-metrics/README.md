<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf request metric evidence

This record supports the [AIPerf metric selectors](../../../docs/sweeper/optimization-goals.md#aiperf-metric-selectors).
It separates two claims:

1. **Calculation agreement.** AISimulate's `summarize_requests` gives the same
   statistics as pinned AIPerf when both read the same observed request records.
2. **Prediction error.** AISimulate replay predicts these statistics for a matched
   real workload. Matching formulas does not make the prediction accurate.

## Setup

| Item | Value |
|---|---|
| AIPerf | [ai-dynamo/aiperf `b0276e19398169eb450f5a79db28509dd4bcb48f`](https://github.com/ai-dynamo/aiperf/tree/b0276e19398169eb450f5a79db28509dd4bcb48f), installed from the source archive (reports `0.13.0`) |
| Server | `vllm/vllm-openai:v0.30.0` (arm64 digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`), torch 2.13.0, FlashInfer attention |
| GPU | One NVIDIA GB200, driver 595.84.01 |
| Models | `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`, `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (model and tokenizer) |
| Engine arguments | `--dtype bfloat16 --max-model-len 4096 --max-num-seqs 64 --max-num-batched-tokens 8192 --no-enable-prefix-caching --gpu-memory-utilization 0.85 --seed 0`, TP1 ([serve.sh](serve.sh)) |
| Workload | Mooncake trace from [gen_trace.py](gen_trace.py) `256 64 7`: unique prompts, input 128 to 1536 tokens, output 8, 24, 64, 160 or 400 tokens |
| Client | Chat endpoint, streaming, `--use-server-token-count`, `ignore_eos`, sequential trace sampling ([bench.sh](bench.sh)) |
| Load | Concurrency 1 with 48 requests and concurrency 16 with 256 requests, each with and without `--per-chunk-usage` |
| Warmup | One discarded synthetic AIPerf run before each measured run. Replay starts with an idle engine. |
| AISimulate timing | Op-level `SILICON` data for `gb200`, vLLM `0.30.0`, from the checked-in systems data |

[compare.py](compare.py) rebuilds each request from `profile_export_raw.jsonl`:
arrival, first and last content chunk, server output tokens and, with per-chunk
usage, the first content chunk's token count. It compares the result with
`profile_export_aiperf.json`. It then replays the observed input and output
lengths, in dispatch order, with `aisimulate predict` and `evaluation.aiperf`.

## Calculation agreement

All 8 runs finished with every request valid. The maximum relative difference
covers the 4 sample counts and all 20 statistics. The last two columns compare
the legacy per-gap user TPS mean, computed over content-chunk gaps, with the
AIPerf request mean. At
concurrency 16, the legacy definition overstates user TPS by 41 to 48 percent.

| Model | Run | Records | Errors | Calc max rel diff | First-chunk tokens (count) | Per-gap TPS mean | AIPerf user TPS avg |
|---|---|---:|---:|---:|---|---:|---:|
| Qwen3-0.6B | c1-pcu | 48 | 0 | 3.0e-16 | 1 token: 48 | 665.8 | 677.3 |
| Qwen3-0.6B | c1 | 48 | 0 | 3.4e-16 | not requested | 689.8 | 688.7 |
| Qwen3-0.6B | c16-pcu | 256 | 0 | 4.0e-16 | 1 token: 250, 2 tokens: 6 | 518.6 | 350.8 |
| Qwen3-0.6B | c16 | 256 | 0 | 5.0e-16 | not requested | 518.2 | 355.8 |
| Qwen3-1.7B | c1-pcu | 48 | 0 | 3.2e-16 | 1 token: 47, 2 tokens: 1 | 656.2 | 660.2 |
| Qwen3-1.7B | c1 | 48 | 0 | 3.2e-16 | not requested | 656.9 | 660.7 |
| Qwen3-1.7B | c16-pcu | 256 | 0 | 4.1e-16 | 1 token: 252, 2 tokens: 4 | 490.9 | 340.0 |
| Qwen3-1.7B | c16 | 256 | 0 | 4.9e-16 | not requested | 497.3 | 351.9 |

## Prediction error

Relative error is `(AISimulate - AIPerf) / AIPerf`. These configurations do not
enable `engine.tokenization` or a frontend model, and replay has no HTTP model. Mean TTFT is therefore 16 to 28 ms low in every run.
Replay also models every first chunk as one token, so per-chunk usage does not
change its ITL.

| Model | Run | Metric | AIPerf avg / p50 / p99 | AISimulate avg / p50 / p99 | Rel. error avg / p50 / p99 |
|---|---|---|---|---|---|
| Qwen3-0.6B | c1-pcu | TTFT ms | 19.17 / 19.36 / 23.97 | 3.11 / 3.22 / 3.65 | -84% / -83% / -85% |
| Qwen3-0.6B | c1-pcu | Latency ms | 239.06 / 114.39 / 631.06 | 281.23 / 123.11 / 780.28 | +18% / +8% / +24% |
| Qwen3-0.6B | c1-pcu | ITL ms | 1.48 / 1.52 / 1.57 | 1.93 / 1.93 / 1.95 | +30% / +27% / +24% |
| Qwen3-0.6B | c1-pcu | User TPS | 677.34 / 656.85 / 757.92 | 517.78 / 516.88 / 523.10 | -24% / -21% / -31% |
| Qwen3-0.6B | c1 | TTFT ms | 21.23 / 20.82 / 29.25 | 3.11 / 3.22 / 3.65 | -85% / -85% / -88% |
| Qwen3-0.6B | c1 | Latency ms | 236.97 / 115.79 / 622.44 | 281.23 / 123.11 / 780.28 | +19% / +6% / +25% |
| Qwen3-0.6B | c1 | ITL ms | 1.46 / 1.49 / 1.51 | 1.93 / 1.93 / 1.95 | +33% / +30% / +29% |
| Qwen3-0.6B | c1 | User TPS | 688.70 / 669.39 / 770.83 | 517.78 / 516.88 / 523.10 | -25% / -23% / -32% |
| Qwen3-0.6B | c16-pcu | TTFT ms | 36.38 / 38.18 / 70.97 | 9.07 / 8.62 / 15.82 | -75% / -77% / -78% |
| Qwen3-0.6B | c16-pcu | Latency ms | 422.55 / 174.65 / 1512.10 | 278.06 / 140.44 / 924.22 | -34% / -20% / -39% |
| Qwen3-0.6B | c16-pcu | ITL ms | 3.14 / 3.20 / 6.11 | 2.25 / 2.26 / 2.65 | -28% / -30% / -57% |
| Qwen3-0.6B | c16-pcu | User TPS | 350.82 / 312.06 / 684.39 | 447.04 / 443.34 / 510.33 | +27% / +42% / -25% |
| Qwen3-0.6B | c16 | TTFT ms | 36.94 / 38.80 / 70.83 | 9.01 / 8.57 / 15.75 | -76% / -78% / -78% |
| Qwen3-0.6B | c16 | Latency ms | 427.53 / 182.48 / 1517.64 | 278.28 / 141.21 / 918.49 | -35% / -23% / -39% |
| Qwen3-0.6B | c16 | ITL ms | 3.07 / 3.22 / 4.52 | 2.25 / 2.26 / 2.60 | -27% / -30% / -42% |
| Qwen3-0.6B | c16 | User TPS | 355.84 / 310.96 / 685.55 | 446.32 / 443.30 / 510.33 | +25% / +43% / -26% |
| Qwen3-1.7B | c1-pcu | TTFT ms | 19.58 / 19.56 / 24.33 | 4.19 / 4.34 / 5.33 | -79% / -78% / -78% |
| Qwen3-1.7B | c1-pcu | Latency ms | 244.56 / 116.17 / 646.07 | 317.11 / 138.98 / 877.93 | +30% / +20% / +36% |
| Qwen3-1.7B | c1-pcu | ITL ms | 1.52 / 1.55 / 1.58 | 2.17 / 2.18 / 2.19 | +43% / +41% / +39% |
| Qwen3-1.7B | c1-pcu | User TPS | 660.18 / 645.94 / 739.23 | 460.25 / 459.55 / 464.45 | -30% / -29% / -37% |
| Qwen3-1.7B | c1 | TTFT ms | 23.69 / 21.31 / 71.23 | 4.19 / 4.34 / 5.33 | -82% / -80% / -93% |
| Qwen3-1.7B | c1 | Latency ms | 248.55 / 118.19 / 651.52 | 317.11 / 138.98 / 877.93 | +28% / +18% / +35% |
| Qwen3-1.7B | c1 | ITL ms | 1.52 / 1.55 / 1.59 | 2.17 / 2.18 / 2.19 | +43% / +41% / +38% |
| Qwen3-1.7B | c1 | User TPS | 660.72 / 645.93 / 734.43 | 460.25 / 459.55 / 464.45 | -30% / -29% / -37% |
| Qwen3-1.7B | c16-pcu | TTFT ms | 37.07 / 38.69 / 70.32 | 11.79 / 10.62 / 28.89 | -68% / -73% / -59% |
| Qwen3-1.7B | c16-pcu | Latency ms | 437.43 / 185.65 / 1541.60 | 336.07 / 168.94 / 1124.51 | -23% / -9% / -27% |
| Qwen3-1.7B | c16-pcu | ITL ms | 3.19 / 3.29 / 4.84 | 2.73 / 2.72 / 3.41 | -15% / -17% / -30% |
| Qwen3-1.7B | c16-pcu | User TPS | 340.03 / 303.81 / 650.86 | 369.19 / 367.62 / 431.05 | +9% / +21% / -34% |
| Qwen3-1.7B | c16 | TTFT ms | 38.14 / 39.14 / 76.51 | 11.73 / 10.61 / 28.89 | -69% / -73% / -62% |
| Qwen3-1.7B | c16 | Latency ms | 436.13 / 183.64 / 1547.43 | 335.48 / 168.59 / 1119.45 | -23% / -8% / -28% |
| Qwen3-1.7B | c16 | ITL ms | 3.11 / 3.20 / 4.81 | 2.71 / 2.71 / 3.41 | -13% / -15% / -29% |
| Qwen3-1.7B | c16 | User TPS | 351.89 / 312.22 / 696.69 | 371.53 / 368.87 / 431.05 | +6% / +18% / -38% |

## Recommendation workflow

[recommend.yaml](recommend.yaml) searches concurrency 1 and 16 for Qwen3-0.6B with
`target: throughput_per_gpu` and an `output_token_throughput_per_user.p50`
constraint. [rec_roundtrip.py](rec_roundtrip.py) runs `predict` on each exported
`recommendations/0001.yaml`.

| Threshold | `strict_sla` | Feasible | Selected concurrency | User TPS p50 | Output tok/s/GPU | Predict versus search |
|---:|---|---:|---:|---:|---:|---:|
| 50 | true | 2 of 2 | 16 | 443.6 | 6357.4 | 0.0 |
| 50 | false | 2 of 2 | 16 | 443.6 | 6357.4 | 0.0 |
| 450 | true | 1 of 2 | 1 | 517.5 | 514.9 | 0.0 |
| 450 | false | 2 of 2 | 16 | 443.6 | 6357.4 | 0.0 |

With `strict_sla: true` and threshold 450, the concurrency-16 candidate is
rejected with `output_token_throughput_per_user.p50=443.65 violates gte 450`.
With `strict_sla: false`, the same objective keeps it. Every exported
configuration retains `evaluation.aiperf`, and `predict` reproduces all 24
`aiperf.*` values exactly.

## Coverage gaps

- FPM timing: no FPM profile exists for these models. A synthetic FPM fixture
  test covers the FPM replay path. There is no FPM serving-accuracy result.
- Hardware: one GB200 GPU, TP1, aggregated serving only.
- AIPerf goodput, `adj_*` failure-adjusted metrics and full-response metrics
  are out of scope. Every request in these runs was valid, so failure handling
  is covered only by unit tests.
- The fixture [aiperf_b0276e19_request_metrics.json](../../../tests/fixtures/aiperf_b0276e19_request_metrics.json)
  holds values that [aiperf_expected.py](aiperf_expected.py) computes with the
  pinned AIPerf metric classes.
