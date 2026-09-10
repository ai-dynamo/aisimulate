# DeepSeek-V4.1 Flash: GB300 component calibration

Draft experimental data for the text AR path, TP4 / MoE-TP4 / EP1 / DP1 / PP1.
Both decoder replay profiles completed native prefill, cached extension and
real-KV decode. These tables qualify the listed component samples; they do not
establish whole-serving accuracy or broader workload coverage.

## Runtime and provenance

- Checkpoint: `deepseek-ai/DeepSeek-V4.1-Flash`, revision
  `fb2764a5cf321eaa5070ca8f9e892818f477c16d`; original config SHA-256
  `d7637228d27528f6bd259781b5a27258068f50bf637c9c83aab784d81579669d`.
- ARM64 image: `lmsysorg/sglang`, digest
  `sha256:800cc9adea5be1e18f48185451220c4bc487c545b7095c720d2ccc9ba9bb3b5d`.
  Actual SGLang version `0.0.0.dev0`, PyTorch `2.13.0+cu130`, NCCL `2.29.7`.
- Text-serving source corresponds to SGLang
  `1aa0e962b206102b7c439a4a0c4981cfec6e87bc`; the image has a different vision-DP
  guard. The installed-source hashes, rather than an invented image Git SHA,
  identify the actual runtime in each evidence directory.
- Native experts selected `Mxfp4FlashinferTrtllmMoEMethod` with MXFP8 activations.
  Logical expert intermediate 2304 / TP4 = 576 is physically padded to 640 by
  that native method. The consumer key remains logical 2304.
- Native shared projections retain FP8 block 32×32. Attention `wo_a` is BF16
  after the runtime's streaming dequantization. Engram tables reside in HBM,
  sharded by TP. Text requests execute no vision or DSpark work; unused vision
  weights are still resident because this runtime's language-only loader flag
  rejects V4.1.

The hardware fields match the bundled GB300 specification. Only
`misc.nccl_version` changes to the measured `2.29.7` in these explicit overlays.
They do not replace curated backend defaults.

## Samples and boundaries

Each profile has **16 workload cells**: batch `{1,2}` × actual extension
`{3,128,129,256}` × cached prefix `{0,256}`. Every cell has one warmup, three
measured repetitions and two real decode iterations. That produces 48 measured
extension invocations and 96 decode invocations per profile, with 1,584 raw
component records on each of four ranks. The 309 physical table points are
264 attention, 18 shared-linear, 18 Engram and 9 mHC points; they are not 309
independent workloads. Both profiles use the same tokenized corpus.

Each physical invocation requires exactly ranks 0–3 once. Its latency is the
maximum rank latency; the table stores the median across matching invocations.
Raw compressed JSONL, input hashes, source hashes and completion receipts remain
adjacent. A separate profiled canary retained actual CUDA kernel names; its
perturbed timings are excluded from the tables.

Attention excludes its output collective. Engram includes lookup/dequantization,
projection and gate, with lookup all-reduce outside the timed intervals. mHC
includes both mix/combine and post sites; its statistics stream is serialized
for the eager local measurement. Shared dense projections are timed separately.
All component tables use `used_cuda_graph=false`.

The separate baseline sweep has nine token counts `{1,2,3,6,128,129,256,258,512}`:
18 BF16 GEMM points, 9 MXFP4/MXFP8 MoE points and 18 NCCL payload points. It uses
loaded native weights and explicit seeded uniform synthetic expert routing;
recorded real-text routing is never relabeled uniform. The two profile overlays
share these replay-independent local baselines, with the original bounded-run
provenance retained. NCCL raw evidence records bytes, while its consumer table
keys **elements** (`bytes / 2`); `wire_dtype=bfloat16` identifies the actual
payload behind the existing 16-bit `half` communication category.

The analytical text TP graph explicitly uses NCCL after attention, Engram,
embedding, and the combined routed/shared expert result. Matching whole-serving
validation must disable both custom all-reduce and FlashInfer all-reduce fusion.
Component collection kept default communication outside the local intervals;
its whole-forward configuration therefore differs from that validation setting.
No NCCL timing is relabeled as custom-all-reduce data.

The SGLang pure-TP eager graph runs shared and routed experts sequentially.
The verified runtime has `enable_single_batch_overlap=false` and both CUDA graph
backends disabled. In the actual `srt/models/deepseek_v2.py` (SHA-256
`4be6f035f7a573ef4016eb38dd08ac1dede74fb3bcc13e8d529bd0bd983f22dd`),
`_can_dual_stream_graph` requires graph capture/replay; `forward_normal` executes
the shared and routed calls on the same stream. See the pinned
[dispatch and eager implementation](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/srt/models/deepseek_v2.py#L885).
The prediction grid uses their summed cost. Graph/SBO configurations require a
separately qualified execution contract.

## What the checks establish

All 618 V4.1 physical table points round-trip through the strict Rust SILICON
consumer with the measured latency and `source=silicon`. All 32 profile/workload
combinations compile and execute a complete prefill plus one decode estimate
without a missing table or donor source. Reproducing calibration points is an
admission check, not an accuracy test. `prediction_grid.json` contains those
predictions; independent E2E/FPM comparison is pending and will be recorded in PR #160.

The shared Engram hash/history update and framework metadata preparation are
outside the timed module boundary and are not modeled explicitly. Embedding,
norm/activation and other memory operations retain the existing empirical
formulas; stage totals consequently include `source=mixed`. Graph-enabled serving's mHC/shared-expert overlap and fused communication can
differ from this eager graph.
Long KV, candidate saturation, larger batches, other TP/EP layouts, additional
content/routing distributions and CUDA graphs remain unqualified.

## Use and reproduce

Select exactly one `full/systems` or `decoder_bounded/systems` directory through
`--systems-paths`, use backend version `0.0.0.dev0`, and keep shared-layer reuse
disabled. Set `decoder_replay` to match that directory. Do not combine the two
module files: unchanged early-layer/decode keys overlap.

From the repository root, run `verify_tables.py` using the project Python
environment with `PYTHONPATH=python/aisimulate/src:python/aisimulate`. It rebuilds
tables in a temporary directory from raw rank observations, checks the emitted
values against the checked-in tables, then exercises strict native queries.

Source licenses and adapted serving contracts are documented in the root
`THIRD_PARTY_NOTICES.md`. No checkpoint weights or internal infrastructure
identifiers are distributed with this artifact.
