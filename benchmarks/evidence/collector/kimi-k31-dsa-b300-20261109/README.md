<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kimi K3.1 Pro native DSA measurements on B300

These tables measure the released Kimi Gated DSA module on NVIDIA B300 SXM6 AC
(SM103, 275040 MiB, 1100 W, driver 615.71.10). The collector uses the native
`KimiK3MLAAttention` and `Indexer`, preserves all 96 model heads, and creates
real TP1/2/4 groups with 96/48/24 heads per rank. Job 5767295 allocated four
GPUs on one NVL8 node; this is operator data, not a full-model run.

The measured boundary includes attention/indexer projections, sparse attention,
output gate and output projection. Decoder normalization and TP reduction are
outside the boundary. Native prefill establishes the complete KV and index-K
history before decode or cached-prefix measurements. The retained runtime
receipts validate token count, per-request history, prefix, local heads and gate.

Execution is explicitly **eager**, with three warmups and the median of ten CUDA
event observations, followed by a maximum across TP ranks outside the timed
boundary. Kernel-source labels retain `_eager` and the observed physical cache
layout. Short prefill uses the runtime-selected dense path while still writing
packed FlashMLA KV; those rows belong to the FlashMLA cache bucket. The collector
uses native dummy initialization for one decoder layer. Full-model accuracy,
trained routing, default graph/overlap behavior, TP8, other GPUs and prediction
error are not evaluated. Attention CP and decode CP are both 1; speculative
decoding is disabled. These samples do not reproduce SemiAnalysis deployment
settings involving speculation or context parallelism. See the deviation note in
`collector/sglang/collect_kimi_k31_dsa_v1.py`.

The published tables contain 39 context and 33 generation cases, with
13 context and 11 generation cases at each of TP1/2/4. They combine two
independently finalized campaigns: the original 60 successful rows and 12
follow-up rows from native chunked history preparation. Their keys are disjoint
and every column is preserved in the union. The selected grid was:

- Each phase covers TP1, TP2 and TP4 with BF16 attention projections and FP8 KV.
- Context: B1 with 128/512/2048/4096/8192/16384 new tokens, B4 with
  128/512/2048/4096 new tokens, and B1 cached-prefix pairs
  (new,prefix)=(512,2048),(512,32768),(2048,8192).
- Generation: B1/4/16 at total history 2048/8192/32768, plus B1/4 at 131072.

In the original campaign, twelve generation cases failed while constructing the
full history, before the timed decode boundary: 10 native BF16 value-projection cuBLAS errors and two
FlashMLA temporary-allocation OOMs. The native one-batch helper performs a
monolithic prefill; this preparation limitation does not establish a serving
decode limit. No smaller batch/history, synthetic state, forced fallback or
replacement latency was used. Exact failed shapes and exceptions are retained in
`benchmarks/evidence/collector/kimi-k31-dsa-b300-20261109/receipt.json`.

All twelve cases subsequently passed using native Req continuations with at most
8192 new history tokens per batch. Every latent KV and index-K entry is built by
native prefill; the final decode history, batch, module and ten-sample timing
protocol are unchanged. A same-model 4096-token control produced identical KV
slots, packed KV bytes, index-K bytes, decode inputs and decode logits between
monolithic and chunked preparation. The follow-up's raw table, sidecar and
receipts are retained under that evidence directory's `chunked-history/`;
`derivation.json` and `merge-campaigns.py` record the exact disjoint union.

This is an explicitly filtered subset of the model's declared plan. A
`complete` collection event means its attempted subset finished, with individual
failures recorded in the standard collection summary; it does not mean every
selected case succeeded or the full structural plan/model was qualified.
`collection_meta.yaml` preserves the standard collector finalizer's original source events from both
campaigns. The existing provenance renderer combines their histories without
rewriting source refs, content hashes, plan hashes or row counts. All 72 rows
passed the strict-provenance reader and an exact native Rust DSA lookup returning `silicon`.
This is a data-integrity check, not a prediction-error result. Four context shapes
also repeated independently between preflight and the formal run; absolute
latency differences were 0.28%–2.00%, without an imposed accuracy threshold.

Runtime and model identity:

- SGLang package: `0.0.0.dev1+g41208373c.d20261104`.
- Native source: `afcd7d6b487a9cbe73e25941971b4cc107e6f9ae`.
- Official image index: `lmsysorg/sglang:dev-kimi-k31-pro@sha256:67fc77d1cbe67de27ca54d874c82757a4844743fea899ea7d0df547e30c2c391`.
- Executed amd64 manifest: `sha256:4a098ad7414702a2271296ac148854448d10710ec3554ac93b57ef194999305a`.
- Model metadata: `moonshotai/Kimi-K3.1-Pro` revision `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e`.
- Original collector commit: `9718c10c8cc74f7901b10fd4dd2922a66409915b`.
- Native history-continuation collector: `b0bcf4e28f7c0d2937469a8fb08a331de7b9e1f2`.

The tables contain measurements and identifiers; they do not redistribute the
checkpoint configuration, tokenizer code or model weights. Canonical source
attribution for the collector is in the repository's `THIRD_PARTY_NOTICES.md`.
