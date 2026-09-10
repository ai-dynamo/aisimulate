# DeepSeek-V4.1 native component collection

This collector is experimental until the checked-in validation report qualifies
the exact runtime and hardware. It loads the actual 40-layer text checkpoint
through SGLang's model builder, then uses the framework's own request and KV
allocation helpers. Prefix extension retains the same request and its cache;
decode follows actual prefill. Inputs are tokenized text with recorded hashes.

The integration is pinned to SGLang commit
`1aa0e962b206102b7c439a4a0c4981cfec6e87bc`. The published runtime reports a
development version, so an immutable image digest and installed Python source
hash manifest are mandatory. Configuration must match the production model
graph's canonical SHA-256. An incompatible runtime raises a recorded failure.

## Measurement boundaries

- Attention includes local projections, compressor/indexer and attention. The
  native `wo_b.reduce_results` boundary excludes output all-reduce, which still
  executes immediately after the measured interval. TP, local heads and output
  groups come from the native builder.
- Engram sums `_owned_rows`, native `wkv` and `apply_gate`; the intervening real
  lookup all-reduce remains outside the measured intervals. Host tables and
  shared host layouts are rejected.
- mHC sums both `_hc_mix_and_combine` and `hc_post` sites. The optional statistics
  stream is disabled for eager local measurement, preserving selected kernels.
  Layer 2 represents ordinary predecessor-pre-mix behavior; layer 0's initial
  copy is not used as the representative sample.
- Dense shared projections retain their native quantization method and local
  dimensions. Replicated or fused shared-expert layouts are rejected because
  they differ from the decomposed graph's sharded projections.

The raw stream records each rank, measured method, source/config/runtime
fingerprints and finite-logit checks. These are component timings, not a
whole-forward overlap or throughput validation. `used_cuda_graph=false` is
explicit. The graph's separate collective nodes remain active.

## Data contract

`dsv41_module_perf.parquet` keys component, canonical native geometry excluding
the display name, batch size and prefix exactly. Only `x` interpolates. For
attention `x` is query length in prefill and absolute sequence length in decode;
all other components use total processed tokens with batch 1 and prefix 0.
Integer Parquet columns are INT64 constrained to uint32. Latency is float64 ms.

Replay profiles are distinct campaign outputs. The bounded profile executes
the native last-min(current-extend,128) tail for each request after layer 20;
the absolute context remains unchanged. Full and bounded raw observations may
not be silently combined under duplicate physical keys. The table writer
rejects duplicates and mixed immutable provenance. Decode and positive-prefix
attention require real KV; other components use `kv_seed_regime=n/a`.

Sources: [`one_batch.py`](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/benchmark/one_batch.py),
[`deepseek_v4.py`](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/srt/models/deepseek_v4.py),
[`engram.py`](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/srt/layers/engram.py).
See the repository's canonical third-party notices for attribution.
