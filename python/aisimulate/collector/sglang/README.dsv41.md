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

## Attention without whole-model residency

`dsv41_attention_runner` is a separate attention-only producer. It constructs
all 40 native `MQALayer` modules from the unchanged configuration so compressor,
indexer, borrowed-KV and bounded-tail ownership remain native. It omits the
embedding, MoE, Engram, mHC and LM head. Native dummy initialization supplies
random weights; normalization weights are explicitly set to one. Each layer
receives the same seeded synthetic BF16 hidden tensor through native RMSNorm,
outside the attention interval. These are declared operator inputs, not hidden
states from an executed DeepSeek model. A zero-logit stub serves only the native
request lifecycle; no logits, forward latency, FPM or model accuracy are admitted.

The native `_TorchBenchRunner` and existing `run_workload` create actual requests,
allocate KV, seed positive prefixes and extend the same request with its actual
pool slot. Decode seeds exactly the declared past KV before the native inclusive
decode. The native backend selects bounded tail rows and positions after the
last source layer. No collector builds cache metadata or substitutes a guessed
request slot. Each TP needs its own `build_manifest(tp, bounded)` result; both
phase manifests are checked against loaded native dimensions before wrapping.

The recorder reuses `ComponentRecorder.finish` and the existing native MQA
interval. It requires pure TP and executes the real output all-reduce after the
CUDA end event. Normalization, RNG, validation, output writing and collectives
are outside that interval. The collector adds no per-layer host synchronization
or output validation to timed forwards. Each case first executes a separate
untimed native replay;
all 40 actual attention outputs, including prefix seed forwards, must be finite
and nonzero. Checking the logits stub cannot satisfy this gate.

Freeze an attention plan with schema `dsv41.attention-collection.v1`, purpose
`smoke`, `calibration` or `heldout`, and the identity fields described above for
isolated operators. Replace `components`/`token_counts` with `input_method`,
`workloads_sha256` and `prompt_sha256`; use the module's explicit
`WEIGHT_INITIALIZER` and `INPUT_METHOD`. The declared limits are
`context_length=8192`, `max_total_tokens=8192`, `max_requests=4`, with at least
two warmups and five measured repetitions. Freeze workload bytes produced by
`freeze_workloads`, all required `ATTENTION_SOURCES` pins, original config and
tokenizer bytes, immutable image/archive identities and the actual producer
revision. The recorded source digest includes the entire installed SGLang Python
tree, using the same algorithm as the whole-model producer.

Prepare a three-case smoke with uncached prefill, batch-two cached prefill whose
query exceeds the 128-token tail, and batch-two real-KV decode, for each TP and
both profiles. Qualify this producer on the target allocation before a separate
calibration run. Capability evidence from a different adapter is not timing
acceptance. With the same private cache and verified-image setup as above:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m collector.sglang.dsv41_attention_runner \
  --plan /campaign/attention-plan.json --manifest /campaign/manifest-tp2.json \
  --workloads /campaign/workloads.json --model-path /campaign/checkpoint-metadata \
  --prompt-file /campaign/token-corpus.txt --output /campaign/attention-raw \
  --runtime-digest "$VERIFIED_IMAGE_DIGEST"
```

All requested cases and rank samples are retained. `aggregate_attention_records`
checks exact plan, actual-output qualification and per-case/rank/sample coverage;
it rejects smoke/heldout promotion and whole-model rows mixed into its raw stream.
Different invocations sharing a physical key are reported with every owner and
remain a publication gate pending a source equivalence audit. They are never
silently dropped or merged, even for canonical bounded-tail collisions. Full
and bounded outputs remain separate. `--admit NEW_TABLE` writes only a fresh
destination after those gates; it refuses to overwrite an existing whole-model
or isolated table. The ordinary consumer contract and provenance checks still
apply before publishing any measured profile.

Native tensor population references at the pinned SGLang commit:

| Contract | Native source under `python/sglang/` |
| --- | --- |
| Native attention geometry, compressor/indexer ownership | `srt/models/deepseek_v4.py:957-1090` |
| Random weights and native post-load order | `srt/model_loader/weight_utils.py:1647-1682`; `srt/model_loader/loader.py:1592-1621` |
| Normalization immediately before MQA | `srt/models/deepseek_v4.py:2743-2764` |
| Actual prefix request slot and native extension | `benchmark/one_batch.py:428-472` |
| Native schedule/forward batch and decode state | `benchmark/one_batch.py:493-543,563-583` |
| Pure-TP output reduction boundary | `srt/models/deepseek_v4.py:781-793,2078-2082` |
| Native bounded tail rows/positions and source KV | `srt/models/deepseek_v4.py:3542-3563`; `srt/layers/attention/deepseek_v4_backend.py:1647-1704`; `srt/mem_cache/deepseek_v4_memory_pool.py:1141-1171` |

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

## Collected H100/H200/B200/GB200 TP2 and TP4 profiles

The collected `full` execution profiles use SGLang
`dev-1aa0e962b206102b7c439a4a0c4981cfec6e87bc`. Select these MoE kernels explicitly:

| System | TP2 | TP4 |
| --- | --- | --- |
| `h100_sxm`, `h200_sxm` | `w4a16_mxfp4_cutlass` | `w4a16_mxfp4_humming` |
| `b200_sxm`, `gb200` | `w4a8_mxfp4_mxfp8_trtllm` | `w4a8_mxfp4_mxfp8_trtllm` |

Set GEMM to `fp8_block`, FMHA to `fp8`, MoE TP equal to model TP, and MoE EP to 1.
Profiles measure native serving modules with synthetic weights/hidden states and real
native token/KV metadata. GB200 TP4 attention was measured in the whole model with the
immutable checkpoint. All primitive families use the isolated collector. TP2 supplies
duplicate mHC/router physical keys after an explicit shape/dtype/source/scope parity
audit; every TP4 raw sample remains in the private collection archive. Selection did not
depend on latency.

The communication table measures `torch.distributed.all_reduce` with the loaded NCCL
2.30.7 provider. PyTorch's reported build version is 2.29.7 and does not identify this
loaded library. Native SGLang PyNccl is a separate API. Keep the packaged default
systems intact and use the existing `systems_paths` override to select the measured NCCL
version:

```python
from importlib.resources import files
from pathlib import Path
import yaml

systems = Path(str(files("aisimulate_core") / "systems"))
overlay = Path("dsv41-systems").resolve()
overlay.mkdir(exist_ok=True)
for name in ("h100_sxm", "h200_sxm", "b200_sxm", "gb200"):
    config = yaml.safe_load((systems / f"{name}.yaml").read_text())
    config["data_dir"] = str((systems / config["data_dir"]).resolve())
    config["misc"]["nccl_version"] = "2.30.7"
    (overlay / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
```

Pass `systems_paths=[str(overlay)]` to the SDK or `--systems-paths dsv41-systems` to the
legacy `aiconfigurator cli estimate` command, along with the explicit backend version
and `SILICON` mode. For example:

```bash
aiconfigurator cli estimate --estimate-mode static \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash --system gb200 --backend sglang \
  --perf-db-version dev-1aa0e962b206102b7c439a4a0c4981cfec6e87bc \
  --systems-paths dsv41-systems --database-mode SILICON \
  --tp-size 4 --moe-tp-size 4 --moe-ep-size 1 \
  --gemm-quant-mode fp8_block --moe-quant-mode w4a8_mxfp4_mxfp8_trtllm \
  --fmha-quant-mode fp8 --isl 128 --osl 2 --batch-size 1
```

All eight hardware/TP full-profile cells pass strict prediction checks for the frozen
145 calibration geometries (108 prefill and 37 decode, batches 1/2/3) and 38 independent
heldout geometries (28 prefill and 10 decode). This is the explicit case grid, not every
Cartesian combination or arbitrary batch/length coverage. Primitive token counts are 1,
2, 4, 8, 16, 32, 64, 128, 129, 256, 512, 1024, 2048, 4096, and 8192. Successful heldout
geometry prediction is coverage unless a separate whole-model truth run is reported.
Whole-model memory feasibility must be evaluated separately from isolated operator
coverage. No extrapolated H100 or TP2 whole-model accuracy is reported. The
`decoder_bounded` raw collection retains eight colliding physical-key groups and is
excluded from these admitted tables pending native-equivalence validation.

Independent full-checkpoint validation used 38 cases, five repetitions and four ranks
for each of these full-profile cells:

| System / TP | Overall MAPE | Prefill (28 cases) | Decode (10 cases) |
| --- | --- | --- | --- |
| GB200 / TP4 | 6.96% | 4.52% | 13.78% |
| H200 / TP4, native Humming | 12.68% | 7.47% | 27.27% |

Ground truth uses each repetition's maximum rank wall time, then the median across
five repetitions. Its native synchronized wall boundary includes preparation.
Predictions and database hashes were frozen before reading each system's truth,
with separate text/tokenized corpora and no geometry overlap. No accuracy threshold
was asserted; the larger H200 decode error remains visible without fitting the
tables to these heldout observations.

These measurements use immutable AMD64 image manifest
`sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5` and ARM64
image manifest
`sha256:800cc9adea5be1e18f48185451220c4bc487c545b7095c720d2ccc9ba9bb3b5d`. Each
collection event records the actual SquashFS archive SHA-256; independently repacked
archives are distinct even when they have the same OCI manifest. The installed
`srt/models/deepseek_v4.py` carries the runtime image's vision-only attention-DP guard:
upstream SHA-256 `48c8a718d90e3136aa7c0fa1373432eaa274f5dd013fdf154e0ddc5719b5c888`
becomes installed SHA-256
`d69b85051bcf4535993d9c2a6625a1e86386e99e1954bb9c2c25e47cd577a8a9`. The recorded source
audit verified this change does not alter the measured text-model path. Both
architectures verify the installed source bytes; their runtimes are not described as an
unmodified upstream checkout.
