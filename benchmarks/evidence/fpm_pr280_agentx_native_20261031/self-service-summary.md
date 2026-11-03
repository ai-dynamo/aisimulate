<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# Completed P/D native FPM collection

Two separate deployments completed native FPM capture on November 1, 2026,
using real prefill/decode disaggregation and AgentX session traffic. Each used
one eight-B200 node, with four GPUs per role, TP4/EP4 and attention DP1.
Both roles loaded the full model and maintained their own KV cache. These
results support the collection procedure for the tested configurations;
they do not establish prediction accuracy or qualify every runtime.

Follow the [P/D setup reference](../../../docs/perf-model/fpm-self-service/native-pd-setup.md)
with your own cluster, model and workload. Its runtime pins and operational
settings explain this example; reproducing these exact request counts is not
an acceptance requirement for another collection.

## Tested identities

| Identity | vLLM | SGLang |
| --- | --- | --- |
| Model | `deepseek-ai/DeepSeek-V4-Flash` | `deepseek-ai/DeepSeek-V4.1-Flash` |
| Model/tokenizer revision | `60d8d70770c6776ff598c94bb586a859a38244f1` | `fb2764a5cf321eaa5070ca8f9e892818f477c16d` |
| Engine package | `0.26.0` | `0.0.0.dev1+ge087e662b` |
| Installed Dynamo | `1.4.0` | `1.5.0` |

The setup reference records the immutable container digests. The SGLang run
also used the [native five-field producer correction](native-patches/README.md);
the image digest alone does not identify that corrected producer. The models
differ, so these runs are not a controlled comparison of backend performance.

## Completion and retained data

Both deployments completed all four concurrency tiers. Every reported warmup
and profiling response passed client validation, with no reported request
errors, cancellations, context skips or grace timeout. Outstanding requests
were allowed to drain after each 1,200-second sending interval.

| Concurrency | Warmup responses, each backend | vLLM profiling responses | SGLang profiling responses |
| ---: | ---: | ---: | ---: |
| 16 | 16 | 173 | 178 |
| 32 | 31 | 837 | 817 |
| 64 | 62 | 1,306 | 1,277 |
| 128 | 126 | 503 | 431 |
| **Total** | **235** | **2,819** | **2,703** |

| Capture/export check | vLLM | SGLang |
| --- | ---: | ---: |
| Original retained gzip files | 5 | 5 |
| Total native records | 690,638 | 637,502 |
| Busy / idle records | 685,827 / 4,811 | 631,877 / 5,625 |
| Excluded startup/smoke busy records | 34 | 33 |
| Selected workload busy records | 685,793 | 631,844 |
| Prefill-role export records | 25,586 | 20,254 |
| Decode-role export records | 660,207 | 611,590 |
| Reported raw-stream validation issues | 0 | 0 |

The smoke used two sequential uncached requests with 16 output tokens each.
vLLM produced 2 prefill and 32 decode rows; SGLang produced 2 prefill and
30 decode rows plus one separately identified startup prefill row. Selection
excluded those exact prefixes using worker/counter identity and original row
hashes. Subsequent native warmup, sending and drain rows remained included;
strict per-phase membership was not claimed.

vLLM's smoke recorded eight completed KV transfers. SGLang's real transfer is
supported by its source/configuration and matching role sequences; direct
transfer-completion telemetry was not retained for that smoke.

Independent streaming checks compared every selected row with its original
role export. They verified all five numeric counters, unchanged native duration
in seconds, provenance, ordering, file hashes, complete consumption and no
extra rows. Export used the `fpm5` feature contract with `join-ranks none`.

## Measurement and training boundaries

- **Native timers differ.** vLLM normally measures host time between busy
  output-update arrivals; initialization or an empty output resets the baseline,
  so the next busy row uses its matching schedule timestamp. SGLang drains
  completed device-timer intervals with overlap disabled in this qualification.
  Neither convention is a replacement for end-to-end request latency, and
  exporting them does not make their timing boundaries identical.
- **Decode context differs.** vLLM uses prior computed-token counts; SGLang's
  ordinary decode lengths include the current token. Preserve backend identity
  rather than applying a shared one-token correction.
- **Role is not batch class.** The vLLM decode export contains 333 pure-prefill
  admission rows and 2,680 mixed rows, in addition to 657,194 ordinary decode
  rows. The documented decode trainer excludes those 3,013 admission/mixed rows;
  their preservation in the export is not evidence of training coverage. The
  SGLang exports contain only pure-prefill work on P and pure-decode work on D.
- **Smoke has bounded coverage.** The accepted numeric smoke covers two short
  uncached requests. Independent timing-to-batch alignment for cached, chunked
  and concurrent work remains separate qualification work.
- **Retained integrity is not delivery completeness.** vLLM's inner process
  shutdown escalated after client drain. SGLang's role cleanup reported `-9`
  without a supervisor SIGKILL, leaving the sender/internal shutdown path
  unresolved. Persisted idle tails and valid files do not prove every forward
  reached the sink; publisher losslessness remains unproven.
- **No model was trained by this campaign.** Held-out FPM error, matched Replay
  serving accuracy and full source-workload coverage were not established.
  Use the guide's [training qualification](../../../docs/perf-model/fpm-self-service/e2e-gbdt.md#3-train-and-qualify-accuracy)
  before claiming accuracy for a fitted model.

## Archived evidence identity

The following receipt paths identify the original campaign archive, not files
required to run the portable guide. The full archive is not distributed with
this summary. SHA256 values bind the reviewed evidence; hashes alone do not
make those artifacts publicly retrievable.

| Archived receipt, relative to this campaign directory | SHA256 |
| --- | --- |
| `slurm/completed-campaign-verification.json` | `6e4021d8f8862297a4a4115b6c1a9f24159a8e310c0e1393ecb8bafd12c9f2f6` |
| `analysis/vllm-5508775/fpm5-independent-verification.json` | `f93e7de21a0206e1788ba431f7d45e55361fc5655a2d3832bdc4c8e811c54712` |
| `analysis/sglang-5520049/fpm5-independent-verification.json` | `64723482df11f4b011c8816e4173a222051e3cff9164c7ac21db91938eeb762e` |
| `analysis/vllm-5508775/native-capture-quality.json` | `8c3d067f17a449589e8ddc65320c569a487d8fe9620d9c7add12cac82e95efc7` |
| `slurm/sglang-5520049-native-verification.json` | `4d2af214da723b30d968ac943f5efdf419bb049798c00475322dbd5165a9d839` |
| `preflight/native-label-review.md` | `7a4968e5ef193b479af05c5cef1aa981b8de2f9bda48e6221df3148aba6936a0` |
