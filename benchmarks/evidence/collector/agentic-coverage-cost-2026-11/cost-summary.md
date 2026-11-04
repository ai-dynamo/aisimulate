# Collection-cost audit: corrected 597-sample interim report

**Interim: sampling continues. All 24 system/framework total forecasts and the fleet total remain unknown.**
This snapshot contains 552 passed, 20 failed and 25 partial observations. Passed samples
qualify only where native forward evidence and runtime checks exist. Failed/partial records and
returned clocks without forward evidence never fit successful timings; diagnostic spend is retained.
No GPU allocation, queue or reserved-idle time is charged.

B is pinned #676 coverage. S is the previous shared expansion. M is the optional model-owned
extension plus the independent MoE/KDA/NCCL changes. Counts below are exact planned physical
cells after grouped-axis expansion; they include unqualified planned routes and are not promises
of runtime admission. NCCL is outside these op-plan CSVs. Its [measured native components](nccl-cost-components.json)
cover 1 GiB calls with 512 MiB controls on recorded rank groups; they are charged once per system,
not once per framework. Missing/censored collective/dtype/rank cases and unproven parent-setup
increments keep complete NCCL totals unknown. L40S lacks a supported multi-GPU local topology.

Added columns show **history / non-history** cells relative to B. Point columns show
**qualified added cells / inclusive seconds** for that measured or held-out-qualified subset.
These point seconds exclude separately recorded shared setup and are not total collection costs.
Pair columns show **covered added cells / observed native-group difference in GPU-hours**.
Unknown strata count operations with any uncovered additions/removals; this count does not
quantify finer unmeasured geometry, precision or checkpoint gaps. Held-out columns show passed/tested point
strata with a numeric error; a dash means no usable held-out test, not a pass.

| System | Framework | B cells | S added H/N | M added H/N | Removed S/M | S points / s | M points / s | S paired GPUh | M paired GPUh | Unknown ops S/M | Held-out S/M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| a100_sxm | sglang | 60,303 | 610/4,314 | 95/4,314 | 0/0 | 0 / — | 0 / — | — | — | 4/4 | — / — |
| a100_sxm | trtllm | 82,848 | 7,780/3,864 | 232/3,864 | 0/0 | 0 / — | 0 / — | — | — | 5/4 | — / — |
| a100_sxm | vllm | 61,388 | 9,654/1,734 | 262/1,734 | 0/0 | 0 / — | 0 / — | — | — | 7/6 | — / — |
| b200_sxm | sglang | 350,952 | 30,832/16,764 | 1,794/16,764 | 0/0 | 12 / 249.657 | 6 / 240.090 | 2 / -0.002171…+0.001528 | 2 / -0.002171…+0.001528 | 12/10 | — / — |
| b200_sxm | trtllm | 395,923 | 32,363/18,396 | 921/18,396 | 0/0 | 6 / 113.824 | 3 / 38.160 | — | — | 10/7 | — / — |
| b200_sxm | vllm | 245,209 | 29,110/6,906 | 872/6,906 | 0/0 | 6 / 59.208 | 3 / 45.710 | 2 / -0.000254…+0.000349 | 2 / -0.000254…+0.000349 | 9/8 | — / — |
| b300_sxm | sglang | 350,952 | 30,832/16,764 | 1,794/16,764 | 0/0 | 19 / 55.517 | 7 / 45.141 | 2 / -0.004641…+0.004272 | 2 / -0.004641…+0.004272 | 12/10 | 1/2 / 1/2 |
| b300_sxm | trtllm | 395,923 | 32,363/18,396 | 921/18,396 | 0/0 | 23 / 599.869 | 13 / 46.612 | — | — | 10/7 | 2/2 / 2/2 |
| b300_sxm | vllm | 245,209 | 29,110/6,906 | 872/6,906 | 0/0 | 19 / 302.065 | 9 / 58.487 | 2 / -0.004602…+0.000069 † | 2 / -0.004602…+0.000069 † | 9/8 | 1/2 / 1/2 |
| gb200 | sglang | 350,952 | 30,832/16,764 | 1,794/16,764 | 0/0 | 9 / 58.712 | 3 / 49.900 | 2 / +0.000477 † | 2 / +0.000477 † | 12/10 | — / — |
| gb200 | trtllm | 395,923 | 32,363/18,396 | 921/18,396 | 0/0 | 6 / 71.667 | 3 / 13.980 | — | — | 10/7 | — / — |
| gb200 | vllm | 245,209 | 29,110/6,906 | 872/6,906 | 0/0 | 6 / 97.912 | 3 / 82.510 | 2 / +0.000192 † | 2 / +0.000192 † | 9/8 | — / — |
| gb300 | sglang | 350,952 | 30,832/16,764 | 1,794/16,764 | 0/0 | 6 / 12.363 | 0 / — | 2 / -0.001996 † | 2 / -0.001996 † | 12/10 | — / — |
| gb300 | trtllm | 395,923 | 32,363/18,396 | 921/18,396 | 0/0 | 6 / 80.081 | 3 / 17.180 | — | — | 10/7 | — / — |
| gb300 | vllm | 245,209 | 29,110/6,906 | 872/6,906 | 0/0 | 6 / 100.419 | 3 / 83.030 | 2 / -0.000311…+0.000012 † | 2 / -0.000311…+0.000012 † | 9/8 | — / — |
| h100_sxm | sglang | 245,123 | 27,326/9,726 | 1,372/9,726 | 0/0 | 6 / 81.012 | 3 / 71.500 | 2 / +0.000445…+0.001002 | 2 / +0.000445…+0.001002 | 12/10 | — / — |
| h100_sxm | trtllm | 270,336 | 27,299/11,790 | 755/11,790 | 0/0 | 6 / 14.662 | 3 / 9.530 | 2 / -0.000654…+0.000125 | 2 / -0.000654…+0.000125 | 10/6 | — / — |
| h100_sxm | vllm | 186,409 | 25,660/4,746 | 948/4,746 | 0/0 | 5 / 63.479 | 2 / 44.730 | 2 / -0.000183…+0.000070 | 2 / -0.000183…+0.000070 | 9/8 | — / — |
| h200_sxm | sglang | 245,123 | 27,326/9,726 | 1,372/9,726 | 0/0 | 9 / 293.353 | 6 / 285.360 | 2 / +0.000724…+0.001486 | 2 / +0.000724…+0.001486 | 12/10 | — / — |
| h200_sxm | trtllm | 270,336 | 27,299/11,790 | 755/11,790 | 0/0 | 6 / 13.175 | 3 / 8.360 | — | — | 10/6 | — / — |
| h200_sxm | vllm | 186,409 | 25,660/4,746 | 948/4,746 | 0/0 | 6 / 73.930 | 3 / 58.270 | 2 / +0.000036…+0.000696 | 2 / +0.000036…+0.000696 | 9/8 | — / — |
| l40s | sglang | 211,847 | 23,061/9,726 | 429/9,726 | 0/0 | 0 / — | 0 / — | — | — | 8/6 | — / — |
| l40s | trtllm | 168,468 | 21,165/7,038 | 543/7,038 | 0/0 | 0 / — | 0 / — | — | — | 5/4 | — / — |
| l40s | vllm | 149,305 | 20,528/4,746 | 524/4,746 | 0/0 | 0 / — | 0 / — | — | — | 7/6 | — / — |

No S or M comparison removes baseline cells. Identical MoE/KDA groups can support both
alternatives only through explicit per-operation physical-member and complete-task manifest
proofs. These alternatives do not multiply observations or GPU spend; history setup never transfers.

Pair ranges fold identical physical 27→29-cell MoE repeats once; they do not multiply cost
or coverage by the repeat count. Only explicitly proven predecessor-warm cache/protocol/rank
pairs qualify. A dagger (†) marks fewer than three controlled pairs or failure of the independent
15% group forecast gate; these remain diagnostic components. The first two repeats train
the held-out third repeat. Same-group point clocks plus residual setup are an accounting
identity and cannot pass this forecast gate. Negative or zero-crossing differences show
cache/JIT/order noise, **not savings**. A single observation provides no measured repeat
variation. All ranges are empirical, not
statistical confidence intervals.

For B300 vLLM full causal prefill at B1, Q64/KV1, d128, the 262K and 1M anchors predict
the withheld 512K inclusive call within **5.17%** using squared sequence length. Another
Q4/KV1, d256 prefill stratum fails at **34.86%** and cannot support interpolated forecasts.
The 20% point and 15% independent group gates are reported separately; an exact measured
point is evidence for that point even when a broader interpolation fails.

## Setup, capacity and checkpoint qualifications

Collection time includes point preparation/history, initialization, warmup/JIT/capture and
forward execution. Forward latency and nested phase clocks are retained as diagnostics and
are never added to an already-inclusive clock. Separate setup is charged once per group.
Existing workers may need different pools, buffers or autotuning for added shapes; their
initialization is not assumed free or charged anew to every inner point. Full extra totals
require matched group evidence covering the exact inventory delta; those totals remain null.

GLM attention equivalence establishes the persisted attention timing contract, not equivalent
checkpoint startup or memory capacity. SGLang constructs two full layers before extracting
attention; BF16 versus NVFP4 expert weights can change initialization cost and available KV
capacity. The #676 canonical representative remains unchanged. Targeted NVFP4 collection
retains its own checkpoint, and canonical BF16 refusals remain gaps. The report keeps C44
NVFP4 costs separate from canonical BF16 model keys; it does not invent cost equivalence.

Native Kimi NoPE long-history extension is deferred and unmeasured. Its zero declared optional
MLA additions do not establish zero cost or complete 1M coverage. The independent KDA width
extension remains included.

## Conditional components and remaining evidence

An independent [B52 warm-history preparation fit](warm-history-preparation-fit.md) isolates
first-use cost, trains on 17 warm points per indexer mode and tests four held-out points.
The full/skip maximum errors are 9.54%/8.45%. This supports a conditional preparation
component only for that checkpoint, native geometry, mode and warm state. It supplies no
independent whole-group setup estimate and no full-operation GPU-hour total.

Fresh dense anchors often vary only batch at one sequence length. Two points cannot identify
allocated-KV-byte preparation, attention work and fixed overhead while also providing a
held-out test. A broader fit needs at least two lengths × two batches, actual allocation-floor
and repetition evidence, a held-out geometry and three independent matched group repeats.
Failed/censored shapes need direct qualification; scaling successful shapes cannot price them.
Later MLA and sparse-prefill runs and further KDA receipts remain outside this frozen snapshot.

## Corrected attribution and frozen provenance

This snapshot supersedes the original 597-record attribution. Native clocks, rows and identities
were preserved; 44 fresh dense-context case labels were bound to the actual executed native
parameters, and 36 holdout roles moved with their predeclared physical identities. Independent
checks covered all 140 dense point identities and the unchanged grouped/module observations.
The old inputs remain preserved separately; they must not reproduce this corrected report.

The corrected [input archive](fit-inputs-597-corrected.zip) and
[manifest](fit-inputs-597-corrected.manifest.json) contain the exact sample/proof files.
Archive SHA256: `346b8c15e8d31e7d92dad144cd10c6dd236ee4e77e99a8d738d19a3f0ac3260e`.
Runtime evidence retains observed image hashes, GPU/framework versions and power policy.
Unobserved Torch/CUDA versions, actual threadpool size and concurrent host-worker counts
remain null; requested OMP8/eight CPUs do not prove observed host isolation.

- Native candidate: `c569abb1cd034c934f47974209515e3507f60b8a`.
- Helper revision: `bd1efe42fb0d17e75b4d04cc33100180a184a936` (public byte-identical equivalent `46b34d0590a4f6c14c271cc5043307af316fe3bd`); SHA256 of `collection_cost_report.py`: `be43e4b18001e1c64d671dd6fdf23a44e6d13b7ca55c995ae8a8d6a6b963bc1f`.
- B/M inventories: pinned #676 / `4eaef24ff72a4dce5d4acf2976b746a65de3ca19`. S uses corrected final5 exports from shared-policy fix `07caae244`; it unions every baseline generation point. Exact scalar CSV rows are the comparison contract.
- Report SHA256: `471e06b99676dc49daee1ea310fefbd3dd4b5e7eb1c1e9f56efc9dd377771506`.
- Frozen sample JSONL SHA256: `fa257fc68a7f25a6333ba4d58244d259f9eb48d7354c8348fc2b8d3d39396b46`.
- Frozen source-equivalence proof SHA256: `412af791ab15503b0e34514a5fd8fba614b7a3026de55ccb37dc463237659ac2`.
- Actual native source revisions retained in the report: `95a52e8f520ee2d25ead27cfb04cfcb78ad89cd9`, `c4b44d81f526b3c5299f7c15ccafe02d90b39229`, `c569abb1cd034c934f47974209515e3507f60b8a`.
- Older op evidence is admitted only by the recorded per-operation immutable file-hash proofs; actual sample SHAs, repetitions or null values, raw artifact references and phase clocks are preserved.

The inventory manifest digest below hashes each sorted relative CSV path, a NUL byte and the
binary SHA256 of its contents. It identifies exactly the CSV population used by this snapshot.

| Inventory | CSV files | Manifest SHA256 |
|---|---:|---|
| B | 264 | `84f96489845f8b5a7ef1c0c5b9a340134489158c1619b4e2287e4e29e3c0b21b` |
| S | 264 | `005623c3e65908aa2d9c325efa8c4f5eed4d8d99a852655cc0cd49fe13b25005` |
| M | 264 | `425b69cbf1158b0c31bd62b838cfe4feb3254164afaff1d3ed365e18f42b19a0` |

Reproduce with the documented report CLI, the frozen sample/proof hashes above and the same
inventory manifests. New samples require a new report snapshot; they must not silently change
interpretation of this interim artifact.
