# DeepSeek-R1 SGLang: 84-point TTFT diagnosis

## Result

The large aggregate error combines two different problems. H200 has a false
KV-admission cliff; Blackwell underpredicts prefill even without queueing.
Replacing only KV capacity with each original server's logged value removes the
H200 cliff. It leaves every Blackwell prediction unchanged.

| Hardware | Points | Original TTFT MAPE | Logged-capacity TTFT MAPE | Original TPOT MAPE | Logged-capacity TPOT MAPE |
|---|---:|---:|---:|---:|---:|
| H200 | 10 | 1612.27% | 33.69% | 19.59% | 10.36% |
| B200 | 29 | 65.15% | 65.15% | 5.61% | 5.61% |
| B300 | 45 | 60.29% | 60.29% | 5.86% | 5.86% |
| All | 84 | 246.73% | 58.80% | 7.41% | 6.31% |

These are equal-point means of absolute percentage errors against original
silicon **mean** TTFT/TPOT, in milliseconds. All 84 observations are retained.
Both replay arms complete 21,430 requests without output truncation. The new arm
is a diagnostic input correction, not a shipped memory-estimator fix. There is
no fitted timing multiplier, new GPU trace, or matched-version accuracy claim.

## H200: confirmed capacity and admission error

All ten original logs expose a usable capacity: 509,285 tokens for nine points,
and 509,157 for one. AISim estimates 57,906 tokens. The original replay did not
import the logged capacity and used the estimate for admission instead.

For 8k input / 1k output / concurrency 64:

- Silicon TTFT: 2245.58 ms.
- Original replay: 103251.69 ms, including 102930.85 ms before first admission.
- Logged-capacity replay: 1943.10 ms, including 1622.13 ms before admission.
- Mean admission-to-first-token interval stays about 321 ms.

For 1k / 1k / concurrency 64, TTFT changes from 14961.07 to 168.20 ms;
silicon is 359.75 ms. Only five of the ten H200 points change. The other five
already fit, so their residual error remains. The 8k/concurrency-64 replay still
records ten readmissions after capacity correction; this experiment does not
establish exact scheduler alignment or eliminate every admission discrepancy.

### Why the estimate is small

The native memory diagnostics report these rank-local quantities:

| Component | Bytes |
|---|---:|
| GPU capacity | 151397597184 |
| Weights | 90845872128 |
| Activations/workspace | 24310185984 |
| Runtime overhead | 4509715660 |
| Communication overhead | 411041792 |
| CUDA Graph reservation | 0 |
| Remaining KV budget | 4069214126 |
| KV bytes per token | 70272 |

AISim applies `mem_fraction_static=0.82`, then subtracts weights, a token-budget
activation estimate, and overheads from that pool. The 32,768-token prefill budget
alone induces 24.31 GB of activation/workspace subtraction.

Original SGLang 0.5.12 instead profiles available memory after loading the model
and reserves a fraction of the pre-load free memory. It does not subtract AISim's
extra peak-activation estimate from that static pool. This is a backend memory
budget mismatch, beyond simply choosing different activation coefficients.
The log also reports 79.80 GiB loaded weights versus AISim's 84.61 GiB estimate;
that difference needs separate reconciliation. Removing just the activation
term would therefore not reproduce the exact observed capacity.

Source checks:

- AISim `sdk/memory.py`, `KVCacheEstimator.from_request` and capacity calculation;
  `sdk/backends/base_backend.py::_get_memory_usage`; and
  `sdk/backends/sglang_backend.py::memory_fraction_of_free`, at runtime commit below.
- [SGLang 0.5.12 memory profiling](https://github.com/sgl-project/sglang/blob/127b9e3283f7c2a43234b852ff5c9f1796d53624/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L62).
- [Original H200 benchmark 409939 log](https://inferencex.semianalysis.com/inference/logs/409939).

Cause attribution: **GAP-005 confirmed**, mitigated in the diagnostic replay;
**GAP-017 confirmed budget-contract mismatch**. The capacity substitution above
is a diagnostic experiment only. The production estimator correction is isolated
in [PR #263](https://github.com/ai-dynamo/aisimulate/pull/263), using static-pool
semantics and resident-weight accounting without importing logged capacity.
Its independent estimate is 497973 tokens (2.22% below the typical log); the
original 84-point replay reaches H200 TTFT/TPOT MAPE 33.92%/10.29% and overall
58.83%/6.30%. This data/precision PR contains no KV estimator correction.
Changing attention precision or latency alone cannot repair the admission error.

## Blackwell: prefill timing gap, not the H200 capacity cliff

All 74 Blackwell predictions are below measured TTFT. Every prediction is
unchanged by the logged-capacity experiment. Single-request cases have zero
simulated queue time and already show substantial error:

| Hardware / model precision / TP | Input | Silicon TTFT ms | Replay TTFT ms |
|---|---:|---:|---:|
| B200 / FP8 / 8 | 1024 | 160.36 | 43.22 |
| B200 / FP8 / 8 | 8192 | 314.28 | 166.33 |
| B200 / FP4 / 4 | 1024 | 155.27 | 34.15 |
| B300 / FP4 / 4 | 1024 | 77.36 | 33.99 |
| B300 / FP8 / 8 | 1024 | 70.92 | 42.07 |

Confirmed observations:

- All 74 original logs select `trtllm_mla` attention and
  `flashinfer_trtllm` MoE, disable piecewise graphs for that MoE runner, and
  report eager prefill. Across their logs, 8904 prefill entries report graph off;
  these include warmups and are not counted as benchmark requests. Decode uses
  graphs. A global graph-enabled flag would miss this distinction.
- Single-request replay TTFT equals the native per-op sum, within timestamp
  rounding. Its boundary has no separately measured frontend/host/rank-wait
  interval. In the first B200 FP8 1k request, the attention block contributes
  5.86 ms of the 42.88 ms native total; the silicon cell mean is 160.36 ms.
  These different populations are not an isolated attention error measurement.
- SGLang GEMM, MoE, and the newly collected standalone MLA attention use the
  graph-enabled `benchmark_with_power` default. Complete attention-module
  prefill collection has a separate eager path; do not classify every selected
  attention contribution as graph-timed. Original per-row GEMM/MoE collection
  logs are not reconstructed here.
- Repricing the single-request inputs at the nominal 1024/8192 token lengths
  produces only modest increases. For B200 FP8 1k, 43.22 ms mean becomes a
  45.73 ms nominal-length forward. Input-length variation cannot explain the
  roughly 117 ms missing interval in that cell.

**GAP-007 is supported**, with remaining attribution open: isolated graph-timed
ops do not establish eager whole-forward latency, including host submission and
rank synchronization. **GAP-014 remains suspected**: client TTFT also includes
serving/frontend work, but aggregate logs cannot split that interval. Neither
explains a quantified share of the residual without a matched R1 trace.
Do not transfer the earlier DeepSeek-V4 trace's overhead number to R1.

### Collector-path clarification

A phase-level rule cannot be applied to every collector. The full SGLang MLA
module collector does use eager prefill and graph decode (subject to decode
coverage). Its normal context path sets `use_module_cuda_graph=False` and times
back-to-back eager calls. The separate standalone MLA collector calls
`benchmark_with_power` without overriding `use_cuda_graph=True`; the outer
benchmark captures the operation even though mock server args disable the
runtime's own graphs. GEMM/MoE microbenchmarks also use that graph default and
do not apply a universal prefill/decode switch.

Crucially, inspecting the **original installed replay wheel** at `2cfe6f83`
with `get_database(system, "sglang", "0.5.14")` and
`MLAModule.load_data(database)` gives empty context and generation module views
for both `b200_sxm` and `b300_sxm`. Thus `context_mla_block` in the per-op report
is a fallback wrapper name, not proof of a module-table hit: these predictions
use the granular projection/MLA path. Both profile sidecars pin the refreshed
`context_mla_perf` collector to `35b5292364a0c8d004d3450af27f6259a17aa668`, whose
`collect_mla.py::benchmark_layer` uses the outer graph-enabled benchmark.

The user's eager-prefill/graph-decode description is correct for the full module
collector. The broader claim that all prefill profiles use graphs is incorrect.
The narrower graph-versus-eager exposure is established for the standalone MLA
path used here; historical GEMM/MoE row provenance remains partly unresolved.
This does not quantify the TTFT contribution or justify adding a blanket CPU
launch penalty. No performance data, runtime behavior, or reported MAPE changes
in this clarification.

The next discriminating measurement is a warm single-request R1 prefill with
per-request stage timestamps and all-rank CUDA/CPU tracing, starting with B200
FP8 TP8 at 1k and 8k. Compare selected op costs with the forward critical path;
keep overlapping host gaps and collective waiting from being counted twice.

## Version and kernel limits

Silicon uses 65 SGLang 0.5.12-cu130 observations and 19 0.5.12.post1 observations;
the prediction corpus is 0.5.14. H200 explicitly uses historical FlashInfer
attention, while the 0.5.14 mapping/data is FA3 BF16. The dtype correction recovers
coverage but does not align those kernels. Blackwell matches the named TRT-LLM
MLA/MoE family, but package versions still differ. **GAP-013 exposure is confirmed
for H200; its latency contribution is unmeasured.**

The old AIC analytical estimate is not a second silicon measurement. Its
aggregate path includes a heuristic TTFT queue multiplier (1.9 even at one
request and one prefill step); native replay schedules requests explicitly.
A closer analytical value does not justify copying that multiplier into replay.

## Reproduction and artifacts

- [Paired results, memory diagnostics, op costs and hashes](sglang-mla-ttft-20260917.json).
- Runtime wheel/source: `2cfe6f836fa3f2c81786d250125406cc624c1357`;
  wheel SHA256 `006a3bba530a4555e8b4c2d5ea47eb363e3dc3182c2e4eebfb146c343c813756`.
- Investigated PR code/data head: `50c012460c2feef5d61cda51f0fdce8bbbe815a0`;
  the later commit only recorded replay evidence.
- Original 84-point selection and native baseline are unchanged from the
  [mapping replay](../python/aisimulate/collector/sglang/mla-precision-20260917.md).
- Raw logs, log/benchmark identities, original and overridden replay specs,
  per-request reports, diagnostic scripts and output are retained under
  `/Users/simonec/.cache/aisim-e2e-gym/mla-fp8-20260917/ttft-gap/`.
- `diagnose.py` resolves the exact benchmark from the original database dump,
  saves the public server log, and calls the same `EngineReplayRunnerFactory`
  with only `num_gpu_blocks = logged_tokens // block_size` changed.
  `op_breakdown.py` uses the same wheel and specs to inspect all 13 concurrency-1
  cells. Run both with `../mapping-84/venv/bin/python` from that artifact directory.
- Every raw file has a SHA256 entry in the paired artifact. Raw external source
  and server logs remain in the local evidence directory; they are not vendored
  into this repository.
