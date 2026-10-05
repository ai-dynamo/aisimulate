# dsv411 — DeepSeek-V4.1 module producers (family `dsv411`)

The `dsv411` family is the from-scratch, parallel decomposition of DeepSeek-V4.1 selected by the
Rust-owned switch `dsv41_family="dsv411"` (SDK model `DEEPSEEKV411`, Rust `operators/dsv411.rs`,
table `perf_database/dsv411.rs`). It coexists with the legacy `dsv41` family; nothing here touches
`dsv41` data or predictions.

| piece | file |
|---|---|
| contract (identities, grid expansion, plan schema, row validation, admission, parquet) | `collector/dsv411/contract.py` |
| GPU-side helpers (device witness, source pins, interval timers, graphed sub-calls, raw rows) | `collector/dsv411/runtime.py` |
| grid + budgets (DSA/DSV4 conventions) | `collector/cases/base_ops/dsv411_module.yaml` |
| plan freezer (SDK manifest + cases + pins) | `python -m collector.dsv411.plan` |
| producers | `collector/sglang/collect_dsv411_module.py`, `collector/vllm/collect_dsv411_module.py` |
| publisher | `python -m collector.dsv411.publish` |

## Rows

`dsv411_module_perf.parquet`, one row per physical key
`(component, structure columns, phase, tp_size, batch_size, query, kv_len)`:

| component | measured boundary | coordinates |
|---|---|---|
| `attention_core` | the native attention layer forward minus the nested indexer interval; TP output all-reduce after the end event | context `(batch, query, past kv)`; generation `(batch, 1, past kv + 1)` |
| `indexer` | index-query preparation + scoring + top-k / candidate selection (SGLang `_low_ratio_index_topk`; vLLM `DeepseekV4Indexer.forward` + `indexer_op`) | same |
| `engram`, `mhc`, `shared_linear` | the native sub-calls (collectives outside) | `(1, tokens, 0)` |

Structure columns are exported from the SDK graph (`contract.build_manifest`) and proven against
the loaded native modules before any timer is installed; one representative layer (the first that
carries a structure) produces the row, the other 39 layers still execute for KV ownership.

## Measurement regimes (fixed by the contract, stored per row)

* context: `eager_drained` — eager forward, device drained before every representative layer.
* generation: `cuda_graph` — SGLang: the serving decode graph (`ModelRunner.init_cuda_graphs`),
  per-layer *external* CUDA events recorded during capture, read after each replay; vLLM: the
  serving piecewise split (graph: projections / KV insert / index-query prep → eager sparse
  indexer + MLA → graph: `_o_proj`), replayed per layer. Token components: one CUDA graph per
  native sub-call. An eager generation measurement is admitted only when the plan declares
  `--regime-exception component=reason` (`measurement_regime=eager_exception`).
* KV seeding (`plan.kv_seed_regime`, stored per row as `kv_seed_regime`): `synthetic_kv` (default) runs
  the serving allocation bookkeeping for the prefix (slots, pages, window sliding) without the forwards
  and fills the caches once with bounded random contents - same kernels and shapes, seconds instead of
  hours at 1M kv; the indexer's top-k then gathers a uniform selection (slightly pessimistic locality),
  which a top-k delta calibration corrects if the A/B against `real_kv` rows shows a bias. `real_kv`
  seeds every cached-prefill / decode row by chunked prefills (8192) of corpus tokens (the TP2 H20
  calibration of 2026-10-04 was collected this way).
  SGLang: the out-of-window SWA slots are released before every chunk and before the measured
  extend, where the serving scheduler releases them (`slide_windows`), so the hybrid SWA pool only
  holds the windows plus one extend; the pool limits live in the plan (`plan.py` `DEFAULT_POOL`) and
  `validate_plan` refuses a pool that cannot hold a case's resident KV.

## Running (H20 box)

```
python -m collector.dsv411.plan --backend sglang --tp 2 --purpose calibration \
    --model-path <ckpt-meta> --pins <sha256sum of every .py in the image package> \
    --image-digest <hex> --collector-revision <git sha> --shard 0/4 --out plans/sglang-tp2-s0
torchrun --standalone --nproc-per-node=2 -m collector.sglang.collect_dsv411_module \
    --plan plans/sglang-tp2-s0/plan.json --manifest plans/sglang-tp2-s0/manifest.json \
    --model-path <ckpt-meta> --prompt-file <corpus with >= 1.06M tokens> --output raw/sglang-tp2-s0 \
    --runtime-digest sha256:<hex>            # inside the pinned image, DSV411_* env as in run.sh
python -m collector.sglang.collect_dsv411_module --plan ... --manifest ... --output raw/... --admit out.parquet
python -m collector.dsv411.publish --system h20_3e --backend sglang --systems-root <systems> \
    --runs raw/sglang-tp2-s0 ... --image lmsysorg/sglang:v0.5.21 --torch 2.13.0 --nccl 2.30.7
```

The full grid is ~3.3k cases / ~29k physical keys per (backend, TP); shard it across GPU pairs
(`--shard i/n` round-robins whole seed groups: the sglang producer seeds one (batch, past kv) prefix
and measures every query length on it).

## Interrupted runs, failures, admission

* A run dies as a whole (torchrun). `--resume` on the same `--output` skips the attention cases every
  rank finished (intersection of the `progress-rank-*.jsonl` files), keeps their rows, archives the
  interrupted attempt's receipt as `rank-N.attempt-K.json`, and re-measures the interrupted case and
  the token components.
* A case that kills an attempt is a framework-side observation, not a retry target: each rank keeps
  the running case in `current-rank-N.case` (archived per rank on resume, never deleted, so every rank
  derives the same set); the next attempt records it under `failed_cases` in its receipt and skips it.
  `aggregate_run` drops the keys of the failed cases when all ranks name the same set and reports them,
  and the publisher writes them into the admission record.
* Known limits recorded this way on H20 (sglang 0.5.21 / vLLM 0.30.0): the generation ladder tops at
  kv_len 1,048,575 (past 1048574), the last decode of a request that fills vLLM's `max_model_len`
  1,048,576 (vLLM runs it); sglang admits at most `context_len - 2` tokens per request
  (`managers/tp_worker.py:404` `max_req_len = context_len - 1`, `managers/scheduler.py:2533`
  `max_new_tokens <= max_req_len - input_len - 1` at v0.5.21), so its deepest serving decode reads
  kv_len 1,048,573 and its captured decode graph asserts out of bounds on the 1,048,575 cell - recorded
  as a failed case per sglang TP, not a measurement; sglang's Triton w8a8 block-fp8 GEMM (the only path for this checkpoint's 32-wide
  weight blocks) forms int32 offsets, so context forwards above `2^31 / max weight dim` tokens
  (131072 at TP2) are refused up front as `KernelLimit` failures (`FIXME(kernel-limit)` in the
  producer); vLLM's engram at 262144 tokens does not fit a TP2 rank next to its table shard; at TP4 batch 1024 with a cached
  prefix, some (query, prefix) shapes return all-zero / non-finite output in one of sglang's two sliding-window
  layers (q256-kv16, q128-kv256; neighbours pass) - recorded in-process as `NonFiniteOutput`
  (`OutputQualificationError`), cause open (SWA extend-input parity audit + real_kv A/B pending).
