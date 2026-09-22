<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek V4.1 collection on H100, H200, B200 and GB200

This campaign requests DeepSeek-V4.1-Flash TP2 and TP4, separately for `full`
and `decoder_bounded`. Its execution contract keeps the native checkpoint
precision, GPU-resident TP-sharded Engram, DP1, PP1, EP1, eager execution,
text inputs and speculation disabled. GB200 TP4 full and decoder_bounded now
have independently validated HF tables; other requested profiles remain pending
or exceed weight capacity. A requested cell is not an available FPM profile.

The checkpoint is pinned to
[`fb2764a5cf321eaa5070ca8f9e892818f477c16d`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/fb2764a5cf321eaa5070ca8f9e892818f477c16d).
See the [FPM contract](deepseek-v41.md) for execution identity, real-KV provenance
and immutable Hugging Face dataset pins.

## Residency before scheduling

The SDK's `DeepSeekV41Model.get_resident_weights_bytes()` includes the full
model inventory and native MXFP4 scale bytes. With pure tensor parallelism it
reports 251,900,739,528 bytes (234.60 GiB) per rank at TP2 and 126,489,849,160
bytes (117.80 GiB) at TP4. These are weight-only lower bounds: KV, activation,
workspace and runtime allocations need additional memory. Decoder-bounded
execution retains the same weights as full execution.

The following comparison uses the checked-in system capacities. Before running
an admitted candidate, archive the actual allocation's GPU name, UUID, memory,
architecture and topology; a system YAML is not a live allocation receipt.

| System | Capacity per GPU in system YAML | TP2, both profiles | TP4, both profiles |
| --- | ---: | --- | --- |
| `h100_sxm` | 80 GiB | Weights exceed capacity | Weights exceed capacity |
| `h200_sxm` | 141 GiB | Weights exceed capacity | Native Humming full smoke passed at static fraction 0.98; bounded qualification pending |
| `b200_sxm` | 180 GiB | Weights exceed capacity | Full smoke observed; accepted FPM profiles pending |
| `gb200` | 185.03 GiB | Weights exceed capacity | Full and decoder_bounded tables admitted; independent coverage below |

Capacity rejection is not a failed GPU experiment and must not be reported as
an observed OOM. A bounded decoder profile does not cure weight residency.
Increasing TP, moving Engram to host memory, or changing weight precision would
create a different campaign identity; none is an implicit substitute for the
requested cell.

GB200 TP4 uses separate 145-geometry calibration and 38-geometry heldout runs
for each profile, with ten fixed attempts per point. Full predicts all 38 heldout
geometries: MAPE **1.18418035%**, maximum error **14.2899673%**. Bounded with the
explicit per-request native API predicts 38/38: MAPE **1.12313769%**, maximum error
**8.85221426%**. Its aggregate API separately supports 26/38 geometries with
conditional MAPE **1.23254318%**; the unchanged guard rejects 12 multi-prefill
geometries that need per-request extend lengths. Each MAPE equally weights
geometry errors against medians of ten heldout samples. These results cover
the predeclared grid and the measured 4-slot / 5120-token pool, not arbitrary
loads. See the [profile pins and coverage](../../src/aisimulate_core/systems/profiles/dsv41_fpm/README.md)
and [source replay instructions](../../collector/fpm_forward/README.md).

H200 TP4 uses the separately qualified native Humming route with BF16
activations. The default Hopper CUTLASS route rejects the local intermediate
width of 576. Both Humming and Blackwell TRTLLM pad that width to 640; each stores
2,005,401,600 bytes of routed expert weights and scales per layer and rank. This
adds 7.47 GiB across 40 layers relative to the unpadded checkpoint accounting.
Do not add that padding twice when comparing a native loaded model.

The native module shapes, Engram tables, a conservative subset of vision
weights, and shared RoPE buffers give a TP4 residency lower bound of 126.94 GiB.
On the observed H200 allocation, CUDA reports 139.84 GiB total; a static memory
fraction of 0.9 permits at most 125.86 GiB before other runtime allocations.
That budget cannot fit the lower bound. It does not prove that the model exceeds
physical H200 memory. A separate native load diagnostic at fraction 0.98 observed
127.18 GiB of unique resident tensor storage per rank. The subsequent full
ordinary-serving qualification proved a physical 5120-token KV pool on all four
ranks, prefix reuse and clean client, frontend and worker exits. This establishes
that this configuration can run; calibration and independent held-out accuracy
remain separate requirements.

For that H200 qualification, untimed startup requests used the same live serving
worker and exact token content as the ordinary canary, with distinct request
identities. The original 90-second canary deadline was unchanged, and all startup
and canary FPM records remained in the lifetime audit. Earlier independent-engine
startup attempts and failed canaries remain failed evidence. Humming profiles
require their distinct `w4a16_mxfp4_humming` consumer identity; they must not be
loaded as Blackwell TRTLLM profiles.

These dimensions follow SGLang
[`1aa0e962b206102b7c439a4a0c4981cfec6e87bc`](https://github.com/sgl-project/sglang/tree/1aa0e962b206102b7c439a4a0c4981cfec6e87bc):
[`DeepseekV4Model` and its native attention](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/srt/models/deepseek_v4.py),
[`EngramEmbedding`](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/srt/layers/engram.py),
and the [RoPE factory](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/srt/layers/rotary_embedding/factory.py).
RoPE tables share two variants across layers, about 1 GiB combined; summing 40
independent copies would overcount them. Actual installed source hashes and
unique storage addresses must still accompany the runtime qualification.

## Independent collection and validation

Run eligible systems on separate allocations when resources permit. Each
allocation must have a private output directory and compilation caches. Run only
one whole-model measurement at a time on a given GPU set, and cancel only the
campaign's own receipted jobs or steps.

For each system, TP and execution profile:

1. Verify the immutable container digest for the actual CPU architecture,
   installed framework source hashes, checkpoint identity and native timing
   instrumentation. Use a source-bound development version for published
   profiles rather than treating `0.0.0` as a release identity.
2. Execute the ordinary-serving load and small real-text canary. Check real KV,
   prefix reuse, the selected kernels, actual decoder-bounded execution when
   enabled, and the physical KV pool on every rank. Preserve failed evidence.
3. Freeze separate calibration and held-out corpora, seeds and request geometry
   before collecting measurements. Keep warmups and canaries separate from both
   datasets. Retain token plans and actual scheduled geometry for every result.
4. Collect calibration and held-out runs independently with the same native
   timing boundary. Build FPM tables from calibration only. Evaluate held-out
   observations through the canonical native performance-model interface;
   report supported counts, rejected or missing cases, and MAPE separately.
5. Publish accepted tables and metadata to the FPM Hugging Face dataset. Update
   the implementation's manifest only after exact artifact hashes and immutable
   dataset revisions exist. Verify a clean materialization and native predictions
   with the published pin before marking a profile usable.

Existing quarantined GB200 data remains historical evidence. A fresh successful
load, completed calibration, or transfer to Hugging Face does not by itself
resolve its previously observed serving-latency mismatch. Admission requires the
new run's independent validation results.
