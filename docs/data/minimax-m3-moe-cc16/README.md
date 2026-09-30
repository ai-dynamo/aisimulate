# MiniMax-M3 B300 16-token MoE correction

The vLLM 0.24.0 table records 0.909133 ms for a 16-token NVFP4 MoE call,
while its 8/32-token neighbors cost 0.086317/0.114458 ms. A B300 remeasurement
with the same CUTLASS kernel identity did not reproduce that spike.

This change replaces **one latency value** with 0.10723513793945312 ms,
the median of three extended-timing measurements. All other 63,405 rows,
column types, metadata, row order, and kernel labels are preserved. The
8/32-token rows serve as controls and are not replaced.

## Measured evidence

| Tokens | Historical ms | Short-timing median ms | Extended median ms | Extended range ms |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.086317 | 0.094310 | 0.092930 | 0.091495–0.095320 |
| 16 | 0.909133 | 0.108403 | 0.107235 | 0.097462–0.115944 |
| 32 | 0.114458 | 0.112557 | 0.110788 | 0.108542–0.123461 |

- NVFP4; hidden/intermediate width 6144/3072; top-k 4; 128 experts;
  logical MoE TP1/EP8; `power_law_1.01` routing.
- One B300 measures the collector's rank-local expert path, not EP network
  communication. Routing, activation, quantization and weights are generated
  by the pinned native collector without kernel substitutions.
- vLLM 0.24.0 and FlashInfer 0.6.12. Collector snapshot:
  [aiconfigurator 2ed278a9](https://github.com/ai-dynamo/aiconfigurator/tree/2ed278a91bc599c5149ddfcd527a20c3421b471d/collector).
  The wrapper redirects the snapshot's stale model-config directory after its
  repository move and explicitly selects FlashInfer CUTLASS via vLLM config.
- Three seeded fresh processes per timing setting, rotating shape order.
  Short timing: one warmup and one timed graph replay. Extended timing:
  20 warmups and 100 timed replays. Each replay contains five routing
  populations. CUDA Graph use was verified for all 18 successful rows;
  eager fallback was forbidden.
- Optional TRTLLM comparison failed before timing: stock vLLM 0.24 does not
  support `SWIGLUOAI` for that backend. Its 18 planned shape measurements
  remain unmeasured; no activation substitution or fabricated timing is used.

`remeasurement.json` retains all 18 successful measurements, timing evidence,
source hashes, pinned image digests, exact replaced key, old/new values and
table hashes. Full raw logs and the standalone runner remain in AISim E2E Gym
commit `1e49b5923562ddd256e83424141f818cb57a14c8`, under
`benchmark_results/audits/minimax-m3-moe-cc16-20260929/`.

The old value is 8.48 times the extended median. The original collection-time
cause remains unknown: both short and extended remeasurement avoid the spike.
The table's original legacy provenance is preserved; this scoped correction
does not attest to the rest of the corpus or match the historical 0.23.1
serving runtime exactly.

## Paired engine replay

Both arms use a fresh native build of AISimulate
`7e2f3024bf394c2c0effc0f57a5f7b1b170181c9`, the same archived checkpoint
snapshot and serving controls, and all the same performance data except the
one corrected row. These are native engine replays, not analytical estimates.
Config 1747 uses TP8/PP1/attention-DP1/MoE-TP1/EP8, BF16 KV, block size 128,
1024 max sequences, chunked prefill and no prefix caching. Request count is
10 times concurrency; input/output lengths use NumPy sampling at ratio 0.8,
seed 0. The recorded token budget is 2048 for 1k/1k and 16384 for 8k/1k.

| ISL/OSL | CC | Silicon TPOT ms | Before ms | After ms |
| --- | ---: | ---: | ---: | ---: |
| 1k/1k | 8 | 6.791 | 9.116 | 9.116 |
| 1k/1k | 16 | 8.209 | 58.541 | 11.661 |
| 1k/1k | 32 | 9.852 | 14.470 | 13.784 |
| 8k/1k | 8 | 9.912 | 9.769 | 9.769 |
| 8k/1k | 16 | 9.859 | 59.999 | 13.011 |
| 8k/1k | 32 | 13.334 | 17.806 | 16.891 |

- All 12 replays succeeded and completed the requested populations.
- Six-point TPOT MAPE: **206.30% → 29.38%**. No TPOT APE regression.
- Six-point TTFT MAPE: **21.77% → 23.30%**. TTFT APE worsens at 1k/1k
  CC32 and 8k/1k CC16/CC32; improves at 1k/1k CC16; CC8 is unchanged.
  The correction does not eliminate other modeling gaps.
- `replay.json` preserves each point's TTFT/TPOT, signed error and APE,
  workload, engine args, checkpoint identity, native hash and limitations.
  KV capacity is estimated consistently in both arms. Historical runtime
  differences and client/graph controls remain limitations; this is not
  full serving or whole-forward qualification.

## Integrity checks

- Exactly one `latency` cell changed across 63,406 rows; all other values,
  Arrow schema metadata and row order are identical to the base table.
- Zero null cells, duplicate physical keys or nonfinite/nonpositive latencies.
- `python/aisimulate/tools/perf_database/check_collector_data.py`: all seven
  rules pass (coverage, reuse, comm placement, family, identity, legacy-marker
  and collection-event checks).
