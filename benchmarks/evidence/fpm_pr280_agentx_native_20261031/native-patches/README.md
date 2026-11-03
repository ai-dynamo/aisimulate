<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang native prefill metrics correction

This patch corrects scheduled-prefill accounting in one pinned SGLang native
producer. It derives new-token and prior-KV counts from the batch's saved
pre-forward lengths instead of page-rounded admission totals or request prefix
state that may already include the completed forward pass. Mixed batches retain
the original request indexes and exclude decode requests from prefill totals.
Native decode counters, CUDA timing, event emission and serving execution remain
unchanged. This is a native producer correction, not a per-request data hook.

The source is SGLang's
[`metrics_reporter.py`](https://github.com/sgl-project/sglang/blob/e087e662ba1ac4ef7747537e2a9141085efd4561/python/sglang/srt/managers/scheduler_components/metrics_reporter.py)
at revision `e087e662ba1ac4ef7747537e2a9141085efd4561`, copyright 2023–2024 SGLang
Team, Apache-2.0. The patch and bounded original-method fixture are modified
SGLang derivatives; their attribution is retained in each file. The generated
module preserves the source attribution and marks the modification.
[provenance.json](provenance.json) identifies the source, revision and all hashes;
[the full upstream license](licenses/sglang-e087e662-LICENSE) is included. The
repository's [canonical notices](../../../../THIRD_PARTY_NOTICES.md) record these
derivatives. The materializer and test orchestration are original code.

## Materialize and verify

From the repository root, use a fresh run-owned output path:

```bash
FPM_NATIVE_PATCH=/absolute/owned/path/native-sglang-producer
python3 -B benchmarks/evidence/fpm_pr280_agentx_native_20261031/native-patches/materialize_sglang_patch.py \
  --output "$FPM_NATIVE_PATCH"
python3 -B benchmarks/evidence/fpm_pr280_agentx_native_20261031/native-patches/materialize_sglang_patch.py \
  --verify-dir "$FPM_NATIVE_PATCH"
```

The first command downloads the immutable upstream source, verifies its hash,
applies the patch without offsets or fuzzy matching, and verifies the corrected
hash. To avoid downloading, add `--upstream-file /path/to/original/metrics_reporter.py`
to the materialization command; it must contain those same exact source bytes.
The output includes the corrected module, patch, provenance, license and
`materialization.json`. The verification command only reads existing files.

| File | SHA256 |
| --- | --- |
| Original module | `6ef4618862bd1e0f7c7527353fd2bc77ec365646aae9273ed023f5cce4f85a98` |
| Corrected module | `80ca324b60beb470b4ea6057fde7ee9553dd32e851e9741831459614e40bee63` |

Verify the original runtime module against the original hash. Mount the
corrected file read-only in a disposable runtime at
`/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/metrics_reporter.py`.
Then verify the actual imported module's path and hash inside each worker.
Retain the complete materialized directory with the run. Never relabel old stock
captures as corrected data; a different engine version needs its own source
review and compatible fix.

## Checks and qualification boundary

Run the CPU regression and materialization tests from the repository root:

```bash
python3 -B benchmarks/evidence/fpm_pr280_agentx_native_20261031/native-patches/test_sglang_native_five_fields.py -v
python3 -B benchmarks/evidence/fpm_pr280_agentx_native_20261031/native-patches/test_materialize_sglang_patch.py -v
```

To also check an extracted full source file and verify that unrelated methods
retain the same AST, set `SGLANG_FPM_BASE_SOURCE=/path/to/original/metrics_reporter.py`
for the first test command. CPU tests verify logical counter construction and
pinned bytes; they do not establish GPU timer/batch attribution, publication
losslessness or prediction accuracy.

The [self-service worker example](../../../../docs/perf-model/fpm-self-service/examples.md#d1-pin-the-aggregated-sglang-runtime)
uses this correction with serial scheduling and disabled mixed scheduling,
speculation, attention DP and prefill CP. Its
[aggregate collection summary](../../fpm_sglang_aggregated_native_20261103/self-service-summary.md)
records the completed run and remaining limits. Qualify cold, cached and chunked
requests in each new deployment; preserve its own source, timer and worker
identity with the resulting capture.
