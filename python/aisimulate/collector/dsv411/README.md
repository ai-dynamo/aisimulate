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
* Real KV: every cached-prefill / decode row is seeded by chunked prefills (8192) of corpus tokens.
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

The full grid is ~3.3k cases / ~29k physical keys per (backend, TP); shard it across GPU pairs.
