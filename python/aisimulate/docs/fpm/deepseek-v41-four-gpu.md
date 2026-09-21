<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek V4.1 collection on H100, H200, B200 and GB200

This campaign requests DeepSeek-V4.1-Flash TP2 and TP4, separately for `full`
and `decoder_bounded`. Its execution contract keeps the native checkpoint
precision, GPU-resident TP-sharded Engram, DP1, PP1, EP1, eager execution,
text inputs and speculation disabled. New measurements and accuracy results are
pending. A requested cell is not an available FPM profile.

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
| `h200_sxm` | 141 GiB | Weights exceed capacity | Runtime qualification required |
| `b200_sxm` | 180 GiB | Weights exceed capacity | Runtime qualification required |
| `gb200` | 185.03 GiB | Weights exceed capacity | Runtime qualification required |

Capacity rejection is not a failed GPU experiment and must not be reported as
an observed OOM. A bounded decoder profile does not cure weight residency.
Increasing TP, moving Engram to host memory, or changing weight precision would
create a different campaign identity; none is an implicit substitute for the
requested cell.

H200 TP4 also needs a framework dispatch audit. An MXFP4 checkpoint does not by
itself establish which activation precision or MoE implementation executes on
Hopper. Record the runtime-selected implementation and precision before admitting
measurements. Blackwell kernel selection and its measured timings cannot qualify
Hopper. A memory fit alone qualifies neither platform.

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
