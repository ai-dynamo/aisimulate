<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.5 AgentX step capture

Step captures (per-scheduler-step FPM records) for training and scoring a
learned forward-pass model of the self-collected Qwen3.5 AgentX E2E point:
`Qwen/Qwen3.5-397B-A17B` BF16, vLLM 0.31.0, one B300 node, TP8, no speculation.

## Source

This harness is the E2E collection harness of job 1002947, from
`benchmarks/evidence/accuracy/qwen-weka/` at commit `024d22d68`. These files
are byte-identical copies:

| File | SHA256 |
| --- | --- |
| `client.py` | `352d15e4abdf800a533b0950d4e535f39d11f03c4d972c03bc0f8487de8908fe` |
| `cache_probe.py` | `d126f3e6930e1fd217102e41b5d41272fcc3ddad6cb7c692901549c7f605f202` |
| `receipt_probe.py` | `6363e63a9605755d781574a733dd6035f75bc45e26afb04831bae48fc6ddea2a` |
| `preflight.sbatch` | `486800cdb1e877caf642e925b97340a99695c8cfddbf3194284c8254de3fa8fb` |
| `client_bootstrap/sitecustomize.py` | `8c1886d38faccf6539eae5f74d3e60caa80532b03b66badd9b33aed18df2ef9b` |

`prepare.sbatch`, `collect.sbatch` and `collect.py` are modified copies.
`test_collect.py` is new. The E2E normalization and qualification files were not
copied: this harness records step data, not an E2E result.

## What differs from the E2E run

The server command, image, model, engine settings, client and corpus are
unchanged. The differences:

| Change | Why |
| --- | --- |
| `--scheduler-cls dynamo.vllm.instrumented_scheduler.InstrumentedScheduler` | Publishes one FPM record per step. It subclasses vLLM's scheduler and also runs in sync mode (`--no-async-scheduling`). |
| `ai-dynamo==1.6.1` and `ai-dynamo-runtime==1.6.1`, installed with `--no-deps` into `preparation/dynamo-site` | Dynamo 1.6.1 pins vLLM 0.31.0. The image's own packages are untouched; `prepare.sbatch` fails if `pip freeze` changes or `vllm` resolves elsewhere. |
| `aisim_core.fpm_hooks` on `PYTHONPATH` | Adds `extend_lengths` / `past_kv_lengths` per request. |
| `python -m aisim_core.fpm_record` beside the server | Writes `steps.jsonl.gz` and its summary per tier. |
| Tiers 1, 4, 8, 10, 13, 14, 16, 24, 32, 48, 64 at 1200 s | Same lanes as the E2E points plus wider batch and context coverage. `--max-num-seqs 64` caps the batch, so higher tiers would only queue. |
| Client `--random-seed` 42 or 7 | Seed 42 trains and seed 7 tests: two **capture runs**. |
| A smoke run instead of the pilot qualification | It gates the capture runs on the recorded data. |

Each tier starts a fresh server, like the E2E profiles, so prefix-cache state
starts cold. Each tier's client records are kept, so its latencies can be
checked against the E2E points at the same lanes.

## Run

1. Put a clean aisim checkout of the commit to use at `$CAMPAIGN_ROOT/aisim`,
   and copy this folder to `$CAMPAIGN_ROOT/code`.
2. `sbatch prepare.sbatch` (CPU). Records the aisim commit, the client, the
   corpus, and the Dynamo install check.
3. `CAPTURE_MODE=smoke sbatch collect.sbatch`. One 20-minute tier at 4 lanes.
   It writes `smoke-check.json`, which passes only when all of these hold:
   - steps were recorded;
   - there are no sequence gaps or undecodable messages;
   - at least 99% of steps carry per-request lists that match the request counts;
   - both decode-only and mixed steps appear.
4. For each seed, run
   `CAPTURE_MODE=run SEED=42 SMOKE_CHECK=<gpu-dir>/smoke-check.json sbatch collect.sbatch`,
   then the same with `SEED=7`. Each job is one capture run of about 6 hours.
   `CAPTURE_TIERS="..."` resumes the remaining tiers in a new job.

Per tier, `seed<S>-l<T>/` holds:
- `steps.jsonl.gz` and `steps.jsonl.gz.summary.json`;
- the server, recorder and client logs;
- the cache receipt and one-second server metrics;
- the client records.

`provenance.json` pins:
- the image;
- the model revision;
- the corpus;
- the aisim commit and hook file hashes;
- the Dynamo install.

Before training, check that each capture run has all three step kinds:
prefill only, decode only, and mixed.

Raw requests, streams and logs stay in private storage. Publish only the step
captures, through the aisim-data candidate review.
