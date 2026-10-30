# Qwen3.5 agentic E2E alignment

Controlled, self-collected silicon evidence. These are not official InferenceX
points. Performance scores are pending until collection and replay finish.

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

Client transport/tokenization overhead remains within the measured client
boundary and must not be silently subtracted to improve agreement. Paired
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
TTFT/TPOT and finite-cohort throughput; percentages include explicit denominator
counts. Fixed-timing cache replays never contribute accuracy scores.

`timing-data-provenance.json` records the packaged measurement tables and their
hashes. vLLM 0.31.0 is the serving identity; some attention and recurrent timing
tables come from older releases. These donors remain visible as an accuracy
limitation. The preflight's default cache block is only a setup check: replay
replaces it with the measured cache geometry. Online timing correction is off.
