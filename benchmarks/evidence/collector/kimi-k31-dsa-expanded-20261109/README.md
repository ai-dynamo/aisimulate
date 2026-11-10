<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kimi K3.1 Pro DSA coverage across Hopper and Blackwell

This campaign extends the isolated native gated-DSA measurements in PR #745.
The fixed matrix requests H100, H200, B200, GB200 and GB300 at TP1/2/4/8, plus
B300 TP8. Each configuration has 13 context and 11 generation coordinates.
The 21 new configurations therefore request 504 new measurements; the earlier
72 B300 TP1/2/4 measurements remain separate source events when combined.
`matrix.json` and the selected case files define the exact requested subset.
The public matrix replaces only the original local selection-file locator with
`plans/original-b300-tp124.json`; it records the original frozen matrix SHA-256.
The original selection bytes, every configuration and all case hashes are preserved.
Completing this subset does not mean the full structural grid is covered.

## Measured boundary

The source-pinned SGLang `KimiK3MLAAttention` and native `Indexer` execute with
96 total model heads: 96/48/24/12 per rank at TP1/2/4/8. The boundary includes
attention/indexer projections, sparse attention, output gate and output
projection. Decoder normalization and TP reduction are outside that boundary.
One native dummy-initialized decoder layer is used. This is component data;
whole-model weight capacity, trained routing and serving accuracy are outside
this campaign.

Execution is eager with BF16 projections, FP8 packed FlashMLA KV, CP=DCP=1,
and speculation disabled. Each observation uses three warmups and ten CUDA-event
samples per rank; the retained latency is the maximum of rank medians. The
native builder provides synthetic token inputs. These component measurements
do not verify a common input-token hash across TP ranks or establish numerical
correctness. No prediction error is inferred from exact table queries.

Hopper short prefill follows the native FA3 dense path; Blackwell short prefill
follows native TRTLLM ragged attention. Both retain their actual FlashMLA KV
storage identity. Sparse prefill and decode record their executed native
FlashMLA path. Kernel-source labels retain the eager mode and physical cache
layout, so execution backend and cache storage are not conflated. The collector's
serving-deviation note is in `collector/sglang/collect_kimi_k31_dsa_v1.py`.

Long decode histories use the existing native request-continuation preparation
from #745, retaining the full latent KV and index-K history before timing. No
shorter history, synthetic cache, substitute kernel or replacement latency is
introduced for larger shapes. See the original B300 evidence directory's
`history-setup.md` and bit-exact continuation control.

## Runtime and topology

All measurements use official SGLang image index
`sha256:67fc77d1cbe67de27ca54d874c82757a4844743fea899ea7d0df547e30c2c391`
and audited native source `afcd7d6b487a9cbe73e25941971b4cc107e6f9ae`.
The index's platform builds have different exact package versions:

| Platform | SGLang package | Executed image manifest |
| --- | --- | --- |
| x86_64 | `0.0.0.dev1+g41208373c.d20261104` | `sha256:4a098ad7414702a2271296ac148854448d10710ec3554ac93b57ef194999305a` |
| aarch64 | `0.0.0.dev1+ge79de62d2.d20261104` | `sha256:3c2b577c398ea3557009b2001fac542473443e86997deb8360bf85a980cda240` |

The runtime resolver chooses an exact pin by CPU platform; it does not accept a
version range. Native source hashes are checked before execution. Checkpoint
metadata is pinned to `moonshotai/Kimi-K3.1-Pro` revision
`d9e99b0792a5e93b78ea9bea62a68167e36ddd9e`; model metadata and tokenizer assets
are read from a verified offline cache and are not redistributed here.

The SXM campaigns use native local TP groups. GB200/GB300 TP8 spans two physical
four-GPU hosts in one NVLink fabric; TP1/2/4 stays within one host. Native
`one_batch` rank mapping determines local GPU ordinal and global TP rank. The
parent initializes native launch ports once, and every node receives the same
rendezvous address. An available port is observed in the actual master's network
namespace immediately before each two-node invocation; this is not a persistent
port reservation. Source, allocation, fabric, eight-rank NCCL and native smoke
receipts stay attached to their respective campaigns.

## Data admission and provenance

The x86 campaigns use the standard Collector finalizer and retain its source
events. ARM campaigns invoke the unchanged standalone native collector through
an external Slurm driver. Their canonical Parquet/metadata are finalized only
after verifying the exact planned/attempted case set, original native logs,
ten-sample observations, raw row identities and source bindings. The external
driver receipts are not represented as `collect.py` checkpoints.

Every admitted row is queried through the native Rust SILICON database with
strict provenance and must return its exact stored latency. This establishes
consumer compatibility and data integrity. It is not interpolation validation,
whole-model accuracy or a matched SemiAnalysis AgentX comparison.

The exact development package versions above are measurement coordinates. The
normal `current` query slot consumes them through the existing shared data layer,
which fills missing architecture/coordinate rows from older source versions.
H100/H200/B200/B300 context and decode controls passed this public query path with
strict provenance and identical SILICON latencies. Explicitly disabling that
shared layer restricts queries to the selected version's own rows. These checks
do not qualify the newer query-slot runtime as a measured K3.1 serving backend.

Source tables combine only as disjoint row unions with every column and every
original collection event preserved. Actual CUDA OOMs, if any, remain explicit
missing coverage with their original case and log hashes. Environment, launch
and transport failures remain separate failed attempts; they are not memory
skips. Original failure records and source identities are retained even when a
later attempt succeeds.

Native default graph/overlap execution, CP/DCP greater than one and DFLASH/DSpark
speculation remain outside this campaign. The separately frozen GB300 TP8
trained full-model Op accuracy result in #745 was scored before adding these
coverage tables and is not recalibrated here.
