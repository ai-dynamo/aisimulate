<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang FPM overhead: fixed 8K input / 1K output

Status: Computelab job `4224680` submitted on September 10, 2026. Results are
pending. This follow-up measures FPM-off/on differences across low, mid and high
request concurrency on one eight-GPU B300 node, using DeepSeek-V4-Pro.

The previous [AgentX DSv4 comparison](../agentx-deepseek-v4-pro-440845-b300-slurm/README.md)
observed a small throughput change from one ordered pair. This experiment adds
fixed token lengths, zero prompt reuse and repeated pairs to assess variance.

## Protocol

| Setting | Value |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4-Pro`, revision `b5968e9190ef611bbf34a7229255be88a0e937c1` |
| Hardware | One exclusive node, eight B300 GPUs |
| Runtime | Same FPM-fixed amd64 image as the AgentX experiment |
| Parallelism | TP8 / EP8 / attention DP8 / PP1 |
| Workload | Exactly 8192 input token IDs and 1024 output tokens per request |
| Low / mid / high concurrency | 1 / 32 / 128 |
| Measured requests per repetition | 24 / 256 / 1024 respectively |
| Separate warmup per cell | `max(8, 2 * concurrency)` requests of the same lengths |
| Repetitions | Three per mode at each concurrency (18 measured cells) |
| Serving | Native SGLang `/generate`, explicit cyclic DP routing |
| Cache | HiCache off; unique first pages, cache flush before every cell; assert zero cached tokens |
| Scheduler | Max running requests 256 across the sweep; static fraction 0.93; global prefill chunk 65536; decode graph max batch 544 |
| Speculation | EAGLE/native MTP, steps3/topk1/draft4, simulated acceptance length2.49 |
| FPM on | Native per-rank emission plus buffered external JSONL recorder |

Image:

```text
nvcr.io/nvidian/dynamo-dev/sglang-agentx@sha256:f856a45537f82e1900ea7607edcbaa7f77fbb2e70220eae522d1d50d0046727e
```

The image backports SGLang PR #38711 at
`ed18d64951b93ac252d921ee037ce0d3327eda1f` onto
`71de97b264b04dcd514cf904003028aefe9775c8`.
SGLang's image-pinned `sglang.benchmark.serving` supplies the request driver,
stream parser, server token accounting and latency/throughput calculations.
[fixed_bench.py](fixed_bench.py) supplies deterministic token-ID inputs and checks
all response metadata for actual 8192/1024 lengths, completed counts, zero errors
and zero cached prompt tokens. It does not replace engine source. Synthetic
acceptance and synthetic prompts make this a performance experiment.

## Ordering and interpretation

[campaign.py](campaign.py) runs six fresh-engine blocks: **off, on, on, off,
off, on**. Pair blocks 0/1, 3/2 and 4/5. Each pair shares the exact input hashes
and seed; the concurrency order rotates by pair, so both modes see each order.
Each cell has a separate warmup. All cells flush cache before requests, while
compilation caches may be reused across engine restarts.

The low setting exercises sparse active DP work; the high setting requests128
concurrent HTTP requests. These labels do not claim the high setting is the
absolute saturation limit. Final exports retain achieved concurrency and request
counts. Low-concurrency tail percentiles have relatively few samples.

[compare.py](compare.py) reports means and sample standard deviations over three
runs per mode, and the mean/standard deviation of paired percentage differences.
The primary comparison uses output throughput and TPOT/TTFT. It also retains all
individual runs. FPM-on includes recording, so it does not isolate emission from
consumer overhead. Large or unstable effects may need a targeted follow-up.

## Run and artifacts

Stage the Python files and [submit.sh](submit.sh) in
`/home/scratch.hongkuanz_gpu/sglang-fpm-fixed-8k1k-20260910/` alongside
`image-squashfs.sha256`. Reuse the verified checkpoint and
`images/sglang-agentx-fpm-f856a455-amd64.sqsh`; do not download another copy.
Submit through Computelab Slurm with an eligible account and B300 partition.
The script requests one exclusive eight-GPU node for at most four hours.

Raw results:
`/home/scratch.hongkuanz_gpu/sglang-fpm-fixed-8k1k-results/job-4224680/`.

Each `block-I-MODE/` contains commands, environment, GPU snapshots, separate
warmup/measured client exports, input hashes, actual server usage and validation
markers. FPM-on blocks also retain the complete FPM JSONL, checksum and rank
validation. `state.json` records progress or failure. Final `comparison.json`
and `comparison.md` are generated only after every measured cell validates.

CPU-only x86 preflight validated the exact engine arguments and native client
against a local streaming endpoint, including actual token counts, cache fields
and cyclic DP routing. This preflight does not establish GPU correctness or
performance; the Slurm run performs those checks.

## Configuration provenance

The engine settings carry forward the earlier DSv4 AgentX launcher, whose
configuration derives from SemiAnalysisAI/InferenceX commit
`fb85931b1edec09f9498509835a8c814bebe3c65`,
`benchmarks/single_node/agentic/dsv4_fp4_b300_sglang_mtp.sh` (Apache-2.0).
Changes here select a fixed native workload, omit the router/chat template,
disable HiCache, increase the fixed request pool to256 and add repeated FPM pairs.
See the root `THIRD_PARTY_NOTICES.md`. The native SGLang benchmark is used as an
installed dependency at the pinned engine revision; its implementation is not
vendored here.
