# Collection-cost audit: final 1,018-record campaign report

**Sampling is complete. All 24 complete system/framework forecasts and the fleet total remain unknown where coverage or setup qualification is missing.**
This snapshot contains 924 passed, 49 failed, 28 partial and 17 timed-out observations. Passed samples qualify only where native forwards and runtime protocols are observed.
Failed/partial/timeout records and returned clocks without forward evidence remain
diagnostic spend, never successful fits.
No GPU allocation, queue or reserved-idle time is charged.

B is pinned #676 coverage. S is the previous shared expansion unioned with B. M is the optional model-owned
extension plus the independent MoE/KDA/NCCL changes. Counts below are exact planned physical
cells after grouped-axis expansion; they include unqualified planned routes and are not promises
of runtime admission. NCCL is outside these op-plan CSVs; measured system-shared components and their limits are shown separately below.

Added columns show **history / non-history** cells relative to B. Point columns show
**qualified added cells / inclusive GPU-hours** for that measured or held-out-qualified subset.
These point GPU-hours exclude separately recorded shared setup and are not total collection costs.
Each operator benchmark uses one GPU; TP metadata describes local geometry and is not a GPU-hour
multiplier. **S and M have different covered subsets: their displayed subset sums establish no S−M savings.**
Pair columns show **covered added cells / observed native-group difference in GPU-hours**.
Unknown strata count operations with any uncovered additions/removals; this count does not
quantify finer unmeasured geometry, precision or checkpoint gaps. Held-out columns show passed/tested point
strata with a numeric error; a dash means no usable held-out test, not a pass.

| System | Framework | B cells | S added H/N | M added H/N | Removed S/M | S points / GPUh | M points / GPUh | S paired GPUh | M paired GPUh | Unknown ops S/M | Held-out S/M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| a100_sxm | sglang | 60,303 | 610/4,314 | 95/4,314 | 0/0 | 0 / — | 0 / — | — | — | 4/4 | — / — |
| a100_sxm | trtllm | 82,848 | 7,780/3,864 | 232/3,864 | 0/0 | 0 / — | 0 / — | — | — | 5/4 | — / — |
| a100_sxm | vllm | 61,388 | 9,654/1,734 | 262/1,734 | 0/0 | 0 / — | 0 / — | — | — | 7/6 | — / — |
| b200_sxm | sglang | 350,952 | 30,832/16,764 | 1,794/16,764 | 0/0 | 12 / 0.069349293 | 6 / 0.066691667 | 2 / -0.002171…+0.001528 | 2 / -0.002171…+0.001528 | 12/10 | — / — |
| b200_sxm | trtllm | 395,923 | 32,363/18,396 | 921/18,396 | 0/0 | 6 / 0.031617873 | 5 / 0.035380556 | — | — | 10/7 | — / — |
| b200_sxm | vllm | 245,209 | 29,110/6,906 | 872/6,906 | 0/0 | 6 / 0.016446563 | 5 / 0.020744444 | 2 / -0.000254…+0.000349 | 2 / -0.000254…+0.000349 | 9/8 | — / — |
| b300_sxm | sglang | 350,952 | 30,832/16,764 | 1,794/16,764 | 0/0 | 19 / 0.015421460 | 7 / 0.012539136 | 2 / -0.004641…+0.004272 | 2 / -0.004641…+0.004272 | 12/10 | 1/2 / 1/2 |
| b300_sxm | trtllm | 395,923 | 32,363/18,396 | 921/18,396 | 0/0 | 23 / 0.166619732–0.166974726 | 15 / 0.031773491–0.032128485 | — | — | 10/7 | 2/3 / 2/3 |
| b300_sxm | vllm | 245,209 | 29,110/6,906 | 872/6,906 | 0/0 | 19 / 0.083894492–0.086672518 | 11 / 0.024742245–0.027520271 | 2 / -0.004602…+0.000069 † | 2 / -0.004602…+0.000069 † | 9/8 | 1/3 / 1/3 |
| gb200 | sglang | 350,952 | 30,832/16,764 | 1,794/16,764 | 0/0 | 9 / 0.016308822 | 3 / 0.013861111 | 2 / +0.000477 † | 2 / +0.000477 † | 12/10 | — / — |
| gb200 | trtllm | 395,923 | 32,363/18,396 | 921/18,396 | 0/0 | 6 / 0.019907636 | 5 / 0.021188889 | — | — | 10/7 | — / — |
| gb200 | vllm | 245,209 | 29,110/6,906 | 872/6,906 | 0/0 | 6 / 0.027197872 | 5 / 0.030886111 | 2 / +0.000192 † | 2 / +0.000192 † | 9/8 | — / — |
| gb300 | sglang | 350,952 | 30,832/16,764 | 1,794/16,764 | 0/0 | 6 / 0.003434278 | 0 / — | 2 / -0.001996…+0.000152 | 2 / -0.001996…+0.000152 | 12/10 | — / — |
| gb300 | trtllm | 395,923 | 32,363/18,396 | 921/18,396 | 0/0 | 6 / 0.022244684 | 5 / 0.024036111 | — | — | 10/7 | — / — |
| gb300 | vllm | 245,209 | 29,110/6,906 | 872/6,906 | 0/0 | 6 / 0.027894249 | 5 / 0.031200000 | 2 / -0.000311…+0.000237 | 2 / -0.000311…+0.000237 | 9/8 | — / — |
| h100_sxm | sglang | 245,123 | 27,326/9,726 | 1,372/9,726 | 0/0 | 6 / 0.022503372 | 3 / 0.019861111 | 2 / +0.000445…+0.001002 | 2 / +0.000445…+0.001002 | 12/10 | — / — |
| h100_sxm | trtllm | 270,336 | 27,299/11,790 | 755/11,790 | 0/0 | 6 / 0.004072661 | 4 / 0.020013889 | 2 / -0.000654…+0.000125 | 2 / -0.000654…+0.000125 | 10/6 | — / — |
| h100_sxm | vllm | 186,409 | 25,660/4,746 | 948/4,746 | 0/0 | 5 / 0.017633193 | 4 / 0.026783333 | 2 / -0.000183…+0.000070 | 2 / -0.000183…+0.000070 | 9/8 | — / — |
| h200_sxm | sglang | 245,123 | 27,326/9,726 | 1,372/9,726 | 0/0 | 9 / 0.081486964 | 6 / 0.079266667 | 2 / +0.000724…+0.001486 | 2 / +0.000724…+0.001486 | 12/10 | — / — |
| h200_sxm | trtllm | 270,336 | 27,299/11,790 | 755/11,790 | 0/0 | 6 / 0.003659610 | 4 / 0.015577778 | 2 / -0.001242…-0.000280 | 2 / -0.001242…-0.000280 | 10/6 | — / — |
| h200_sxm | vllm | 186,409 | 25,660/4,746 | 948/4,746 | 0/0 | 6 / 0.020536028 | 5 / 0.029955556 | 2 / +0.000036…+0.000696 | 2 / +0.000036…+0.000696 | 9/8 | — / — |
| l40s | sglang | 211,847 | 23,061/9,726 | 429/9,726 | 0/0 | 3 / 0.003444043 | 0 / — | 2 / +0.004258…+0.004340 | 2 / +0.004258…+0.004340 | 8/6 | — / — |
| l40s | trtllm | 168,468 | 21,165/7,038 | 543/7,038 | 0/0 | 3 / 0.002642695 | 0 / — | 2 / +0.000390…+0.000515 † | 2 / +0.000390…+0.000515 † | 5/4 | — / — |
| l40s | vllm | 149,305 | 20,528/4,746 | 524/4,746 | 0/0 | 3 / 0.004821827 | 0 / — | 2 / +0.000111…+0.000374 | 2 / +0.000111…+0.000374 | 7/6 | — / — |

No S or M comparison removes baseline cells. S/M non-history paired components are reused only when exact physical members and complete generated tasks match. These are alternative comparisons, not additive collection spend; history setup never transfers.

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

## Conditional components and interpretation

The separate [dense allocation study](dense-component-fit.md) qualified only H100/TRT
context execution. Intersecting its unchanged third-replicate predictions with B/S/M
identifies six added points and six retained controls. The six added points measured
**0.001839915 GPUh** execution; the conditional pass-2 model predicts **0.001851390 GPUh**,
for both M−B and S−B. Controls and training repeats are excluded. These six points occur
in both candidates, establishing no S−M savings. Execution is nested in inclusive point
costs; **never add this subtotal or forward latency to an inclusive cost again**.
Preparation, cold initialization, normal one-pass worker and fleet costs remain unknown.

The [B52 warm-history preparation fit](warm-history-preparation-fit.md) remains a separate
conditional NVFP4 checkpoint component. Neither it nor the new targeted NVFP4 worker
measurements establishes canonical BF16 startup/capacity equivalence. The record-specific
source, loader, phase and runtime qualifications remain binding.

Point fits and holdouts are isolated by native source, driver/observer hashes, runtime
and clock scope, actual dispatch/repetitions and first/subsequent pass where observed.
Unknown historical metadata stays in a distinct unknown scope. Returned clocks lacking
native forward evidence remain diagnostic spend; they cannot qualify successful shapes.
The two-pass allocation study is excluded entirely from this one-pass report population.

## Final measured NCCL components

NCCL is collected once per system through the canonical native harness, not separately
for vLLM/SGLang/TRT. The [component artifact](nccl-cost-components.json) records 1 GiB new
calls with 512 MiB controls, four collectives, half/int8 and actual participating ranks.
Native clocks include nccl-tests initialization, warmup, timed iterations and logging.
Missing/censored calls and unproven parent setup increments keep complete NCCL totals
unknown. L40S has no supported multi-GPU local topology in this campaign.

| System | Measured new calls / expected local calls | Measured new native component GPUh |
|---|---:|---:|
| a100_sxm | 0/24 | — |
| l40s | 0/0 | — |
| h100_sxm | 18/24 | 0.100763934 |
| h200_sxm | 18/24 | 0.083085794 |
| b200_sxm | 16/24 | 0.071507475 |
| b300_sxm | 18/24 | 0.100567457 |
| gb200 | 16/16 | 0.041510716 |
| gb300 | 16/16 | 0.033116289 |

These measured system-shared components are not complete fleet estimates and must not
be multiplied by three frameworks or added twice through other inclusive clocks.

## Final frozen provenance

The original 597-record attribution is superseded. Forty-four fresh context labels
were corrected to actual native execution identities, and holdout roles followed their
predeclared physical cases. Independent review verified all final 1,018 records, including
recovered sparse/MLA prefill identities, disjoint TRT dry-run phases and targeted NVFP4
points. Native inclusive clocks were preserved; no fit was retuned to make a gate pass.

The [final input archive](fit-inputs-1018-final.zip) and
[manifest](fit-inputs-1018-final.manifest.json) preserve exact input/raw/producer evidence.
The machine report is [losslessly compressed](cost-report.json.gz); its
[report manifest](cost-report.manifest.json) records compressed/decompressed hashes,
lengths, exact input members and source identities. The CLI still emits ordinary JSON:

```bash
gzip -dc cost-report.json.gz > /tmp/cost-report.json
```

- Native candidate: `c569abb1cd034c934f47974209515e3507f60b8a`.
- Executed helper revision: `eb54cd5c40fcc22f0167c660c530ab7cc06fa869`; public byte-identical equivalent: `44251a59fd9a146d1ae87d9a6576e1a4dfc0d082`; SHA256 of `collection_cost_report.py`: `4aa3d81834dabd42031ea6992f40ee111c2bcbfd7aea8657ad7a655bb65eaec3`.
- B/M inventories: pinned #676 / `4eaef24ff72a4dce5d4acf2976b746a65de3ca19`. S uses corrected final5 exports from shared-policy fix `07caae244`; it unions every baseline generation point. Exact scalar CSV rows are the comparison contract.
- Decompressed report SHA256: `7b94b003b32515a623bc3db3f1e1b26d3c4b0767d7698288506a391bc0ff1440`.
- Final input archive SHA256: `749b8563e84453a718866434731325b767913cc864bdfbc91d7b69f6a398ea65`.
- Frozen sample JSONL SHA256: `9210a7908f5142a9b33f7b3ceb8b0727939bd1bd629df615c9b6740c17404309`.
- Frozen source-equivalence proof SHA256: `412af791ab15503b0e34514a5fd8fba614b7a3026de55ccb37dc463237659ac2`.
- Actual native source revisions retained in the compressed report: `95a52e8f520ee2d25ead27cfb04cfcb78ad89cd9`, `c4b44d81f526b3c5299f7c15ccafe02d90b39229`, `c569abb1cd034c934f47974209515e3507f60b8a`.
- Older op evidence is admitted only by the recorded per-operation immutable file-hash proofs; actual sample SHAs, repetitions or null values, raw artifact references and phase clocks are preserved.

The inventory manifest digest below hashes each sorted relative CSV path, a NUL byte and the
binary SHA256 of its contents. It identifies exactly the CSV population used by this snapshot.

| Inventory | CSV files | Manifest SHA256 |
|---|---:|---|
| B | 264 | `84f96489845f8b5a7ef1c0c5b9a340134489158c1619b4e2287e4e29e3c0b21b` |
| S | 264 | `005623c3e65908aa2d9c325efa8c4f5eed4d8d99a852655cc0cd49fe13b25005` |
| M | 264 | `425b69cbf1158b0c31bd62b838cfe4feb3254164afaff1d3ed365e18f42b19a0` |

Runtime evidence retains observed image/framework/Torch/CUDA/power metadata where exported;
unobserved host concurrency or threadpool facts remain unknown. Requested CPU/OMP settings
do not prove host isolation. All jobs are terminal; no additional GPU jobs are planned.

The audit measured **7.263664 GPUh** collection time. Its conservative charged audit upper
bound is **7.324775 GPUh**; adding **0.475278 GPUh** correctness-job administrative upper
bounds gives **7.800053 GPUh**, within the authorized 12 GPUh. This is campaign-budget
accountability, not the estimated cost of a production sweep. Scheduler/allocation/reserved
idle time is excluded from measured collection costs. See the frozen ledger for details.

Reproduce with the documented report CLI, the frozen sample/proof hashes above and the same
inventory manifests. New samples require a new report snapshot; they must not silently change
interpretation of this frozen artifact.
