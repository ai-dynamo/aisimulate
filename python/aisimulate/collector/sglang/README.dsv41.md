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

Select collected Humming rows explicitly with
`ModelConfig(moe_quant_mode=MoEQuantMode.w4a16_mxfp4_humming)` or the same
`moe_quant_mode` in `Task`. The default Hopper selection remains CUTLASS.
The canonical Humming dtype is written directly in `moe_perf.parquet`, so
task validation and native lookup do not alias it to another kernel. The
temporary fixtures in `test_dsv41_humming_consumer.py` exercise loading,
task validation and a complete V4.1 decode prediction on H100/H200; their
synthetic timings test selection only and are not measured profiles.

## Isolated native operators

`dsv41_isolated_runner` constructs one checkpoint-shaped native module at a
time when the complete checkpoint cannot reside on the requested GPUs. This
is a separate distributed producer for the existing component and baseline
tables. It retains the unchanged model configuration and native quantization
selector and invokes native post-load processing. Floating weights use native
random initialization on bounded parameter views; packed E2M1 bytes are random,
and scales remain positive. Its observations do not validate checkpoint accuracy,
whole-model residency, attention coverage, or FPM.

The declared components are `baselines` (router, LM head, experts and native
collectives), `linear` (sharded shared-expert projections), `engram` (both
physical tables, loaded sequentially), and `mhc`. Engram retains each complete
native GPU table and its real all-reduce. Inputs come from the serving
tokenizer and native `EngramHasher`; hashing and rank agreement checks run
outside the timed module. Each independent sequence uses the native EXTEND
history contract without a prefix. The mHC measurement uses native predecessor
mix coefficients, seeded synthetic attention/FFN outputs, and both real
mix/post sites with no statistics stream. It measures the serial operator
boundary, without an overlap-aware block-latency claim.

Freeze a JSON plan before launch with schema `dsv41.isolated-collection.v1`,
`purpose` (`smoke` or `calibration`), `tp_size`, `execution_profile`, increasing
`token_counts`, `components`, `warmup`, `iterations`, `seed`, explicit
`moe_runner_backend`, `expected_gpu`, `expected_sm`, `framework_commit`,
`collector_revision`, `weight_initializer`, `source_pins`, `metadata_pins`, `runtime_digest`, and
`image_sha256`. Freeze the producer commit and exact copied files before
formal collection; a dirty worktree must not claim its previous revision. Source
pins must describe the actual installed image, including any documented
source modifications. Metadata pins include the unchanged checkpoint config
and tokenizer files. Generate the consumer manifest with
`dsv41_contract.build_manifest(tp_size, decoder_bounded)` and preserve its
exact bytes alongside the plan and token corpus.

On an owned allocation with the matching visible GPU count, bind the actual
HOME cache and `/root/.cache` to the same allocation-private `home-cache`
directory, set `DSV41_PRIVATE_CACHE` to its parent, and set
`SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE=env://` before native imports. Verify
the container's SHA-256 before setting `DSV41_LAUNCH_IMAGE_SHA256`. Launch with
the pinned framework installed:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m collector.sglang.dsv41_isolated_runner \
  --plan /campaign/plan.json --manifest /campaign/manifest.json \
  --model-path /campaign/checkpoint-metadata \
  --prompt-file /campaign/token-corpus.txt --output /campaign/raw \
  --runtime-digest "$VERIFIED_IMAGE_DIGEST"
```

Each rank preserves its native source, allocated GPU UUID/driver, parameter
shapes, input hashes, measurement scope and failure receipt. The raw output
is never published automatically. `aggregate_isolated_records(raw, plan,
manifest)` rejects smoke data and requires every declared token, geometry,
sample and rank, including both Engram tables and matching native Humming
qualification. Its component rows and separate baseline rows still require
the normal perf-table provenance and consumer validation before publication.
Freeze and execute a new calibration plan; changing an old smoke receipt's
description does not admit it as calibration data.

The `native_random_chunks_v1` initializer calls the pinned native
`weight_utils.initialize_dummy_weights` with its default uniform range on
views of at most 8 Mi elements. Each chunk uses `seed + chunk_index`; this is
reproducible but does not claim the full-parameter initializer's exact RNG
sequence. It bounds the native FP8 conversion temporary without replacing the
native conversion. Packed bytes contain uniformly selected valid E2M1
nibbles; E8M0 scales encode a declared synthetic positive one, avoiding invalid signed uniform
initialization of an unsigned exponent format. Per-parameter sample hashes,
nonzero counts, dtype and chunk seeds are recorded before native post-load
processing. Historical zero-weight capability probes are not formal data.

Input population references below are in `python/sglang/srt/` at the pinned
SGLang commit; the AMD image's documented vision-only guard does not change
these sites:

| Input contract | Native source |
| --- | --- |
| Random weights, conversion and post-load order | `model_loader/weight_utils.py:1647-1682`; `model_loader/loader.py:1601-1621` |
| Packed E2M1 values/storage and positive block scales | `layers/quantization/fp8.py:169-213,1266-1284,1354-1377`; `layers/quantization/mxfp4_flashinfer_cutlass_moe.py:74-81` |
| EXTEND mode, tokens and no-prefix sequence lengths | `managers/schedule_batch.py:2553-2595`; `model_executor/forward_batch_info.py:780-798` |
| EXTEND starts and positions | `model_executor/forward_batch_info.py:903-928,1826-1840` |
| One legal isolated history slot, reset history and missing-predecessor PAD | `layers/engram.py:232-243,285-340,399-404` |
| Hasher-only `out_cache_loc=None`, distinct from zero-slot graph padding | `layers/engram.py:346-356` |
| Serving tokenizer parity and Engram input/gate shape | `layers/engram.py:245-268,923-942` |
| mHC predecessor pre-mix, both residual/post sites and stream join | `models/deepseek_v4.py:2651-2661,2734-2785` |

Slot zero belongs to this initialized isolated history fixture; native
`alloc_for_extend` (`schedule_batch.py:2618,2770`) may allocate another slot.
The recipe measures only the native hasher/module contract, not scheduler
allocation, prefix-cache behavior, attention, or model hidden-state quality.

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
