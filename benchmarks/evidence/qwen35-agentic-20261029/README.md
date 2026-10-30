# Qwen3.5 agentic E2E alignment

Controlled, self-collected silicon evidence. These are not official InferenceX
points. Collection, exact-token replay and comparison are complete.

## Completed results

- **12/12 operating points; 36/36 repeated runs; 2,016/2,016 measured requests.**
  No failed, missing or unattempted cases. All predictions completed.
- **2,142/2,142 cache checks match:** 2,016 measured requests plus 126 primers.
  Another 252 kernel-warmup requests are excluded from the measured cohort.
- A real full-block hybrid-state checkpoint bug was found and fixed in
  `6618b91f7f46969b51cbb65ecb3ada07f264fbd8`. That commit is the predictor used
  for every result below. The original baseline remains recorded separately.
- Slurm `946660`: **COMPLETED, exit 0:0**, node `pool0-0213`, eight B300s.
  The allocation lasted 43m23s; the measured campaign including preparation
  between cases lasted 14m49s (2026-10-30 03:37:03–03:51:52 UTC).

Functional parity is established for this matrix. Timing is still inaccurate:
request-level MAPE over all 2,016 pairs is **4,041.1% TTFT, 1,019.9% TPOT,
1,490.4% request E2E**. No numerical acceptance threshold or timing correction
was applied. Native capability metadata still marks agentic qualification as
`functional_only`; these op-level timing predictions are diagnostic, not a
claim of qualified agentic timing accuracy.

Each table entry is **silicon / AISim**. Latencies are request-weighted means
across three repeats. Throughput is the mean of the three finite cohorts,
including their 100ms between-turn delays and excluding preparation.

| Initial tokens | Lanes | Cache | Requests | TTFT (ms) | TPOT (ms/token) | Request E2E (s) | Output tokens/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4,096 | 1 | cold | 24 | 229.8 / 3040.9 | 6.46 / 4.54 | 1.878 / 4.199 | 130.2 / 59.7 |
| 4,096 | 1 | warm | 24 | 229.8 / 2759.2 | 6.45 / 4.54 | 1.874 / 3.917 | 130.5 / 63.9 |
| 4,096 | 4 | cold | 96 | 338.8 / 3839.4 | 7.84 / 6.97 | 2.337 / 5.617 | 422.1 / 179.5 |
| 4,096 | 4 | warm | 96 | 332.5 / 2829.4 | 7.69 / 5.47 | 2.292 / 4.225 | 430.1 / 237.5 |
| 4,096 | 16 | cold | 384 | 463.3 / 5404.5 | 10.26 / 14.07 | 3.081 / 8.992 | 1291.9 / 451.1 |
| 4,096 | 16 | warm | 384 | 414.3 / 2900.1 | 9.96 / 13.58 | 2.955 / 6.363 | 1345.3 / 635.0 |
| 32,768 | 1 | cold | 24 | 319.1 / 44543.6 | 6.64 / 4.65 | 2.013 / 45.730 | 121.9 / 5.6 |
| 32,768 | 1 | warm | 24 | 257.5 / 15731.6 | 6.62 / 4.65 | 1.945 / 16.918 | 125.9 / 15.1 |
| 32,768 | 4 | cold | 96 | 531.5 / 97770.3 | 8.21 / 165.47 | 2.625 / 139.966 | 377.3 / 7.3 |
| 32,768 | 4 | warm | 96 | 389.3 / 16001.9 | 7.89 / 5.83 | 2.402 / 17.489 | 411.1 / 58.3 |
| 32,768 | 16 | cold | 384 | 940.2 / 258142.4 | 17.50 / 1071.48 | 5.403 / 531.370 | 741.2 / 7.7 |
| 32,768 | 16 | warm | 384 | 637.6 / 15951.1 | 10.93 / 41.29 | 3.425 / 26.480 | 1163.9 / 154.2 |

The known FlashInfer source-selection problem is the first timing issue to fix:
its context table favors decode-labelled measurements with only ISL=1 coverage
for this shape. Multi-token prefill must use the runtime-appropriate prefill
source; one-token queries, including APC tails, still use decode. Preserve the
backend/version identity and rerun these unchanged traces after correcting the
selection. A global prefill-first reorder would miss that distinction.

At 32K/16-lane cold, the native median token interval is 9.83ms but the maximum
is 101.22s. This is consistent with large scheduling stalls; attributing the
complete E2E amplification to particular mixed steps requires step/op evidence.
Some timing donors are vLLM 0.30.0, and exact allreduce dispatch parity remains
unverified. See the provenance and limitations in the summary.

Independent raw-data checks verified all literal token chains, token counts,
streaming reconstruction, cache resets, preparation barriers and zero measured
preemptions. Realized tool delays had median 100.61ms and p99 101.88ms. Initial
HTTP dispatch spread reached 28.17ms; native lanes release simultaneously.
Cold/warm starting prompts match, but 930/1,008 paired turns differ in realized
tokens. This is not an identical-trajectory cache ablation or a real-AgentX
semantic workload.

Artifacts:

- [Machine-readable summary](results-summary.json),
  [12-point metric table](operating-points.csv),
  [artifact paths and SHA-256 manifest](artifact-manifest.json).
- [Observed versus predicted chart](alignment-results.png).
- Raw `campaign/`, `replay-timing/`, `comparison.json`, `paired-requests.csv`,
  `raw-integrity-audit.json`, runtime/cache receipts and logs are outside Git.
  The exact local and cluster locations are in `results-summary.json`.
- Cluster results:
  `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec/results/qwen35-agentic-alignment-20261029/gpu-946660`.
  Paired analysis is saved beneath `analysis-6618b91/` there.

Validation of the native repair: 61 state-cache tests and 220 vLLM scheduler
tests passed. The rebuilt Python runner passed all 24 campaign tests and the
separate boundary/chain diagnostics. Cache pressure, eviction/preemption,
MTP, branch/join, offload, P/D, multi-node and FPM-step accuracy remain outside
this campaign.

## Plan and gates

1. Pin AISim `8ae6b65801c4993febb3c827a0e68518b20b379e`, the vLLM 0.31.0
   container/source identity, and the Qwen3.5-397B-A17B BF16 checkpoint/tokenizer.
2. On AWS PDX-1, reuse exact complete shared model/image artifacts or prepare
   them on a CPU data-mover allocation. Keep all remote files under the approved
   InferenceX storage root.
3. Probe an eight-B300, TP8 server. Record actual cache groups, padded page
   sizes, block granularity, usable pool capacity, state dtypes, graph mode,
   package versions and GPU identity. Do not infer pool capacity from nominal
   GPU memory.
4. Qualify manual Qwen state geometry with cold and repeated-prefix requests
   around physical cache-block boundaries. Compare measured reuse against
   AISim. A failed gate is a recorded functional gap, not a timing result.
5. Collect the twelve cells below, three repeats each, using the same workload
   construction and matched cold/warm measured frontier. Keep smoke evidence
   separate from campaign evidence.
6. Run AISim engine replay over the exact recorded token-prefix relationships,
   output lengths, dependencies and delays. Compare cache behavior before
   interpreting TTFT/TPOT differences. Preserve failures and missing predictions.
7. Save paired metrics, raw records, commands, job identities and hashes. Keep
   calibration and held-out validation distinct; no numerical acceptance
   threshold has been specified.

## Fixed configuration

- Model: `Qwen/Qwen3.5-397B-A17B`, BF16 weights and attention KV, BF16
  convolution state, FP32 recurrent state, text only.
  The pinned checkpoint has 60 layers: 15 full attention and 45 recurrent.
- Hardware/topology: one eight-B300 node, TP8, PP1, attention-DP1, context
  parallelism 1, aggregated serving. Expert-parallel switching is disabled.
- vLLM 0.31.0; prefix caching and chunked prefill enabled; Mamba `align`;
  synchronous scheduling; no speculation, CPU/storage offload or ReplaySSM.
- Context limit 65,536; maximum 64 sequences; prefill token budget 8,192.
- Initial GPU memory utilization 0.90; AISim uses the measured resolved pool.
- Physical block size is discovered at startup. AISim explicitly sets
  `prefix_match_unit` equal to that physical block size.

## Matrix

| Dimension | Values |
| --- | --- |
| Initial input tokens | 4,096; 32,768 |
| Agent lanes | 1; 4; 16 |
| Initial cache | cold; warmed |
| Repeats | 0; 1; 2 |
| Measured turns per lane | 8 |
| Output tokens per turn | 256, ignore EOS, temperature 0 |
| Tool tokens appended between turns | 512 |
| Completion-relative tool delay | 100 ms |

There are 12 operating points and 36 repeated runs, with 2,016 planned measured
requests. Each next prompt includes the complete previous prompt and its actual
output token IDs. Lanes have distinct initial prompts. Cold/warm counterparts
use matching prompt construction and measured suffixes.

Cold means empty caches at the start, not disabled prefix caching. Warm means
unmeasured preparation followed by the same measured suffix; preparation output
is not appended to that suffix. Kernel/graph warmup is separate. Drain all
preparation work and reset both attention and recurrent caches at case boundaries.

## Outputs and interpretation

The campaign saves request/phase identities, complete token IDs, streaming event
times, actual lengths, cached-token counts where exposed, metrics snapshots,
errors, server receipts and per-case summaries. TTFT starts at client dispatch;
TPOT uses the interval from first to last output token divided by output tokens
minus one. Record streaming bursts rather than assuming one token per chunk.

Prompt construction and tokenization happen before the measured request window.
Client request encoding, transport and server detokenization remain within the
measured client boundary and must not be silently subtracted to improve agreement. Paired
finite cohorts support request-level inspection. Performance-data absence is a
separate result from replay rejection or a cache mismatch.

The initial campaign excludes MTP, offload, fine-grained prefix matching, P/D,
multi-node execution and quantized checkpoints. MTP needs separate recurrent
scratch-state and prefix-rewind qualification. Branch/join graphs, cache-pressure
experiments and held-out real AgentX plays follow this baseline.

## Execution

`job.sbatch` starts `run_server.sh` inside the verified image. The server writes
its command, package/GPU identity, cache receipt, GPU telemetry and raw logs. It
collects the boundary smoke first and waits for a `CACHE_GATE_PASSED` file in its
run directory. Create that file only after the local functional replay passes:

```bash
python/aisimulate/.venv/bin/python benchmarks/evidence/qwen35-agentic-20261029/replay.py \
  --requests "$RESULTS/smoke" --cache-receipt "$RESULTS/cache-receipt.json" \
  --functional-only --output-dir "$RESULTS/replay-smoke"
```

The replay uses the public AISim engine API. Its fixed timing is only for cache
qualification and must not be used in latency comparisons. Full timing replay
requires the explicit `--engine-config` file. Cache-pressure and eviction
behavior are not qualified by the boundary smoke.

`collect.py` also supports direct use against this dedicated server:

```bash
python collect.py --base-url http://127.0.0.1:8800 \
  --model Qwen/Qwen3.5-397B-A17B --cache-receipt cache-receipt.json \
  --output campaign --matrix --repeats 3
```

The collector requires `aiohttp` (provided by the vLLM image). Each output
directory must be new. Raw data stay outside Git; small receipts and summaries
identify the source artifacts. No performance threshold or timing correction is
applied automatically.

For timing replay and comparison, from the repository root:

```bash
python/aisimulate/.venv/bin/python benchmarks/evidence/qwen35-agentic-20261029/replay.py \
  --requests "$RESULTS/campaign" --cache-receipt "$RESULTS/cache-receipt.json" \
  --engine-config benchmarks/evidence/qwen35-agentic-20261029/engine-config.json \
  --output-dir "$RESULTS/replay-timing"
python/aisimulate/.venv/bin/python benchmarks/evidence/qwen35-agentic-20261029/compare.py \
  --observed "$RESULTS/campaign" --replay "$RESULTS/replay-timing" \
  --output "$RESULTS/comparison.json"
```

`compare.py` retains all 36 planned cases and distinguishes missing, failed,
cache-mismatched and timing-qualified results. It reports paired request-level
TTFT/TPOT, total request latency, cohort completion time and finite-cohort
throughput; percentages include explicit denominator
counts. Fixed-timing cache replays never contribute accuracy scores.

`timing-data-provenance.json` records the packaged measurement tables and their
hashes. vLLM 0.31.0 is the serving identity; some attention and recurrent timing
tables come from older releases. These donors remain visible as an accuracy
limitation. The preflight's default cache block is only a setup check: replay
replaces it with the measured cache geometry. Online timing correction is off.

## Qualification evidence

CPU preparation job `946518` verified the image and all 94 model shards.
GPU attempt `946625` failed before model loading because the image has `python3`
without a `python` alias; the launcher was corrected. Job `946660` completed on
`pool0-0213` and resolved block size 528, 18,955 usable blocks, 15,360 bytes per
token and 24,330,240 state bytes per request per rank. All eight ranks agree.

`smoke-baseline.json` preserves the pre-fix functional failure: repeated
1,056-token requests reuse `0, 528` measured tokens in vLLM but `528, 528` in the
baseline AISim engine. The 527-, 528- and 529-token controls pass. The native
repair stops inventing an intermediate state during cold aligned prefill
and creates it only when a later shared-prefix request actually computes it.
After the native repair, `smoke-fixed.json` records all four boundary cases and
four diagnostic chains passing: 46 cache comparisons, including 40 measured
requests and six primers. The full campaign is released only after this gate.

`qualify_chains.py` collects four additional single-lane diagnostic cases
(4K/32K, cold/warm, eight turns each) using the same collector. These 32 measured
requests are separate from the 2,016-request campaign. Their exact token chains
also serve as regression evidence for the native repair. Their cache counts
match the baseline replay. Cold/warm generated tokens diverge despite temperature
zero, so each run is replayed independently from its actual tokens. Differences
between cold and warm timings do not isolate cache effects alone.

The server selected the vLLM V2 model runner, `FULL_AND_PIECEWISE` CUDA graphs,
FlashInfer TRT-LLM attention, FlashInfer GDN prefill and BF16 TRT-LLM MoE. A long temporary
socket path caused FlashInfer allreduce workspace initialization to fall back
to its TRT-LLM backend; server logs retain the effective path. The timing-data
provenance records a separate FlashInfer prefill-table selection limitation.
