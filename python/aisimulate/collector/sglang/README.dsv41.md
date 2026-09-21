# DeepSeek-V4.1 native component collection

This collector is experimental until validation qualifies the exact runtime
and hardware. Historical measurements and reports are preserved in the
[validation archive](https://github.com/ai-dynamo/aisimulate/tree/24faa2e263c75c137c091b8e80b7c2d36740b864/data/experimental/deepseek-v41/gb300-silicon).
It loads the actual 40-layer text checkpoint
through SGLang's model builder, then uses the framework's own request and KV
allocation helpers. Prefix extension retains the same request and its cache;
decode follows actual prefill. Inputs are tokenized text with recorded hashes.

The integration is pinned to SGLang commit
`1aa0e962b206102b7c439a4a0c4981cfec6e87bc`. The published runtime reports a
development version, so an immutable image digest and installed Python source
hash manifest are mandatory. Configuration must match the production model
graph's original cached checkpoint configuration SHA-256, before SDK-inferred
quantization fields are attached. This differs intentionally from FPM's
normalized execution identity. An incompatible runtime raises a recorded failure.

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

The optional `--collect-baselines` sweep observes the loaded expert method.
Blackwell's native TRTLLM MXFP4/MXFP8 path retains
`moe_dtype=w4a8_mxfp4_mxfp8`; Hopper's native CUTLASS MXFP4/BF16 path uses
`moe_dtype=w4a16_mxfp4_cutlass`. Both use the existing perf-table contract and
record the actual kernel source. Explicit native Humming uses the separate
`moe_dtype=w4a16_mxfp4_humming` / `sglang_mxfp4_humming_moe` identity. It
preserves the framework's weight padding and requires an extra untimed
qualification forward that observes both actual BF16 expert-projection inputs,
absence of activation scales, and full-precision accumulation. The observer
does not replace native methods or run during the timing samples. Aggregation
requires the matching qualification receipt from every rank. Quantized Humming
activations and the CUTLASS method's SM120 path remain rejected.
See the pinned framework's
[`Fp8Config` selector](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/srt/layers/quantization/fp8.py#L421)
and [CUTLASS method](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py#L39).

This dispatch support does not qualify a complete model run. The native
collector still loads the full text backbone with both Engram tables in GPU
memory; `decoder_bounded` does not reduce resident weights. The pinned
CUTLASS constructor also requires each local expert intermediate dimension
to be divisible by 128. For V4.1 TP4, the dimension is 576, which that
constructor rejects. Native Humming performs its own supported weight padding
when that backend is explicitly selected; this is a distinct execution and
perf-table identity. Keep startup failures as collection evidence and do not
add collector-side padding or silently substitute a backend.

## Data contract

`dsv41_module_perf.parquet` keys component, canonical native geometry excluding
the display name and analytical KV-layout selector, batch size and prefix
exactly. Precision and physical dimensions remain exact keys. Only `x` interpolates. For
attention `x` is query length in prefill and absolute sequence length in decode;
all other components use total processed tokens with batch 1 and prefix 0.
Integer Parquet columns are INT64 constrained to uint32. Latency is float64 ms.

Replay profiles are distinct campaign outputs. Set `DSV41_EXECUTION_PROFILE`
to `full` or `decoder_bounded` for a collector invocation. A locked output
marker rejects a second profile targeting the same table. The bounded profile executes
the native last-min(current-extend,128) tail for each request after layer 20;
the absolute context remains unchanged. Full and bounded raw observations may
not be silently combined under duplicate physical keys. The table writer
rejects duplicates and mixed immutable provenance. Decode and positive-prefix
attention require real KV; other components use `kv_seed_regime=n/a`.

Sources: [`one_batch.py`](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/benchmark/one_batch.py),
[`deepseek_v4.py`](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/srt/models/deepseek_v4.py),
[`engram.py`](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/srt/layers/engram.py).
See the repository's canonical third-party notices for attribution.
