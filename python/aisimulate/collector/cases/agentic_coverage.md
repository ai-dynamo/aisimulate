<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Agentic collector coverage

The optional model-owned history extension is stacked on collector planner
#676 at `69232f77ce94de85e8ca7c40da8c2e484a6bdff4`. Normal history coverage
preserves that revision's case population. That baseline already includes some
1M module points: off does not mean removing all long histories.

## Evidence used to choose coverage

Three independent inputs define a case. Each future extension must identify all
three and say what remains unmeasured:

| Input | Evidence | What it establishes |
|---|---|---|
| Model architecture/config | Packaged checkpoint config and existing `model_case_values`, with source revision | Operator family, correlated geometry, legal TP shards and context validity limit |
| Supported serving recipe | AgentX effective runtime logs and launch configuration, cross-checked against pinned framework source | Local/global token mapping, TP/DP/EP, precision, selected backend and speculative width |
| Target workload | AgentX profiling **and warmup** request traces; per-rank forwards when available | History demand and observed joint batch/query/prefix shapes |

The frozen 2 November 2026 audit covers 808 active NVIDIA AgentX
configurations. It derives request history from each request's input plus
actual output, not independent maxima. These request traces do not establish
physical GPU batches. Resolved launch allowances establish bounds, not filled
forwards. Full per-rank batch/query/prefix records remain missing; current
additional joint tuples are explicit interpolation/coverage choices, not
claimed observed forwards. AgentX seeds checked-in declarations; the collector
has no live dependency on AgentX.

The detailed workload evidence and eventual per-system cost report belong in
`benchmarks/evidence/collector/`, outside implementation documentation. The
campaign records exact manifests, signed collector/runtime revisions, source
URLs and artifact hashes there. Do not substitute another model or checkpoint
when an identity cannot be resolved.

## Model-owned history policy

| Model route | Added coverage | Qualification |
|---|---|---|
| Qwen3.5-397B dense layers | Full-lookup anchors 24K/32K/40K at measured B1/2/4/8/16, 64K/128K/262K at selected B1/2/4; decode through 262K at B1/2/4/8 | Uses the declared 32 query / 2 KV heads, d256 and valid TP shards. History stays at the audited model range; no 1M Qwen declaration. |
| Kimi-K3 native MLA | B1 context at 512K/1M; selected B1/2/4 decode at those histories | Uses 96 native heads at TP1/2/4/8. Native-module support is separate from the legacy dense fallback. |
| GLM-5.2/5.3 DSA | Intermediate 786432/917504 history, selected sparse queries including 32K, and B8/16/32 tail decode | Uses 64 native heads and valid TP shards. Full and reused-indexer operations stay separate. The NVFP4 TP8/head8 row also declares #665's 22 measured B52 history points through 720896; B52 is a calibration key, not an observed GPU-forward batch. The 32K query is a qualified collection shape, not proof every serving recipe uses it. |
| MiniMax-M3 MSA | Intermediate history, 32K new queries, selected tail decode batches | Native MSA geometry and precision; global history does not enlarge recurrent/GEMM token work. |
| DSV4.1, MuseSpark, QwenNext unsupported routes | No invented extension | Exact model/backend/feature support and forward tuples must be qualified separately. |

All tuples live on the model rows; the shared grid is not crossed with these
additional axes. Context tuples are `[batch, query, prefix]` for modules and
`[batch, full_lookup_sequence, 0]` for dense attention. Decode tuples are
`[batch, past, 0]`, leaving room for one new token. Model context limits validate
points, not automatically generate them. Native indexing, dispatched kernels
and measured device capacity remain independent requirements.

## Disposition of the previous shared expansion

| Previous #664 change | Stacked implementation |
|---|---|
| Larger shared dense context/decode grids and relaxed scalar sampling caps | Replaced by optional, model-owned points; #676 ordinary density retained |
| Shared extra MLA/DSA/MSA history axes | Replaced by exact optional model points, unioned with #676 baseline |
| Getter-time memory filtering and memory-filtered task manifests | Not ported; #676 plans remain device-independent and workers refuse capacity |
| vLLM 0.31 attention family pin/native adaptations | Retained with shared-plan getters; applies to both dense phases |
| TRT standalone MLA synthetic Scenario/RoPE capacity mismatch | Retained native capacity repair; does not change checkpoint context validity |
| MoE 32K/64K, KDA verifier widths 3/5/6, NCCL through 1 GiB | Retained independently of the long-history option |
| TRT MoE silently shrinking dry-run/tuning sweeps | Removed; a task must account for its requested sweep |
| Batch-52 serving padding, measurement boundary and source attribution | Separate correctness/consumer review work; history selection does not settle them |
| Packaged parquet and historical collector provenance | Preserved; no rewriting of existing measurements |

## Cost and validation contract

Compare pinned #676 normal coverage, the old broad #664 proposal expressed on
that planner, and the model-owned extension. Count unique physical cases and
actual grouped workers, including older anchors newly admitted by larger caps.
Benchmark each system directly. Record forward latency separately alongside
point preparation, history initialization, warmup/JIT/capture, timed iterations
and total collector time. Forward execution is already inside total time: do
not add it twice. Count changes in shared initialization once per group, and
exclude scheduler wait, allocation startup and reserved idle time.

Existing B300 qualification demonstrates specific shapes, not fleet cost. For
example, the SGLang B32 near-1M DSA point took 75.43 seconds within a loaded
runner, including 70.91 seconds history initialization, despite 0.7897 ms
forward latency. Its full native call took 102.35 seconds. Neither value is a
universal rate. Cold single-point worker startup must not be charged to every
inner point; new geometries or larger runtime pools can still add setup cost.

CPU validation checks pinned baseline fingerprints, additive plans on three
frameworks/SM generations, aliases, limits, runtime-safe deduplication,
explicit row manifests, CLI and resumed grouped collection. GPU qualification
and numerical/serving parity remain separate checks. No full-sweep cost or
coverage-completeness claim is made from structural counts alone.


Runtime-sensitive deduplication keeps the pinned planner's representative when
native module semantics agree. For audited GLM checkpoints, tokenizer and MLP
metadata are outside the extracted attention module; native explicit qk/v
fields override redundant `head_dim`. An expert-only quantization recipe is
ignored for this equivalence check only when it excludes every self-attention
module, or explicitly quantizes only MLP modules. Unknown quantization and
changed attention/indexer/RoPE dimensions remain significant. A new incompatible
collision at the same persisted key raises rather than writing duplicate or
misidentified consumer rows. Targeted NVFP4 runs retain their own checkpoint.
