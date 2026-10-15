<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Speculative decoding

Replay supports two speculative-decoding inputs. They cannot be combined.

| Input | Drafts from | Configure with |
| --- | --- | --- |
| [Prompt lookup (ngram)](#prompt-lookup-ngram) | Earlier tokens of the request | `engine.speculation` |
| [MTP / EAGLE](#mtp) | A draft head of the model | `engine.nextn`, `engine.nextn_accepted` |

In both cases you provide how often draft tokens are accepted. That rate
depends on the workload, so Replay does not predict it. Each decode round then
emits a sampled number of tokens, and its cost is a verification pass over the
draft tokens.

## Speculation fields

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.speculation` | Unset (off) | fixed | Optional ngram block below. Rejected together with `engine.nextn > 0`. |
| `engine.speculation.kind` | Required | fixed | `ngram`. |
| `engine.speculation.num_speculative_tokens` | Required | fixed | Integer from 1 to 5. |
| `engine.speculation.acceptance_rates` | Required | fixed | Exactly one value per draft token, each in `[0, 1]`. |
| `engine.speculation.seed` | `42` | fixed | Unsigned 64-bit sampling seed. |
| `engine.nextn` | `0` (off) | fixed | MTP/EAGLE draft tokens, 0 to 5. |
| `engine.nextn_accepted` | Unset | fixed | Required when `nextn > 0`. Expected accepted **draft** tokens per round, excluding the base token, in `[0, nextn]`. |

`recommend` keeps these values fixed for every candidate; it does not search
draft length or acceptance.

<a id="prompt-lookup-ngram"></a>
<a id="prompt-lookup-ngram-speculative-decoding"></a>

## Prompt lookup (ngram)

Merge this fragment into an existing `engine` block from the [engine overview](README.md):

```yaml
engine:
  backend: vllm
  speculation:
    kind: ngram
    num_speculative_tokens: 3
    acceptance_rates: [0.8, 0.6, 0.4]
    seed: 42
```

Or from the command line:

```bash
aisimulate predict -c prediction.yaml \
  --set 'engine.speculation={kind: ngram, num_speculative_tokens: 3, acceptance_rates: [0.8, 0.6, 0.4], seed: 42}' \
  --output-dir ./ngram-prediction
```

`acceptance_rates[i]` is the probability that draft token `i` is accepted given
that all earlier draft tokens were accepted. Each round samples until the first
rejection and is clipped to the remaining output length. The example averages
`1 + 0.8 + 0.8×0.6 + 0.8×0.6×0.4 = 2.472` tokens per round.

Each round is priced as one target-model pass over `num_speculative_tokens + 1`
tokens. There is no draft model, draft weights or draft KV cache. Replay assumes
a draft is available in every steady-state round; fresh and recomputing
requests first emit one base token without draft verification. Real n-gram
matching, other rounds without a draft, and CPU lookup time are not modeled.
With `fixed` or `polynomial` timing,
the decode time you supply is used per verification round as-is.

Supported: vLLM aggregated or disaggregated workers with `op_level`, `fixed` or
`polynomial` timing, offline `engine` stack. Rejected: SGLang, TensorRT-LLM, FPM timing, AFD/EPD, and host
or G3 offload. HBM-only agentic traffic is supported with the same functional-only
qualification as ordinary AgentX replay. The Dynamo stack is not supported for ngram.
Deployment artifacts are not generated for ngram candidates.

Prompt lookup is unrelated to `kv_cache.prefix_caching`.

<a id="mtp"></a>

## MTP / EAGLE

Merge this fragment into an existing `engine` block from the [engine overview](README.md):

```yaml
engine:
  nextn: 3
  nextn_accepted: 2.2
```

`nextn` sets the draft-token count used by the timing model, and
`nextn_accepted` sets the expected number of accepted draft tokens per round.
Replay turns it into per-position acceptance: the whole part is always
accepted and the fraction is the chance of one more. The example accepts
2 draft tokens every round and a third in 20% of rounds, so each round emits 3
or 4 tokens. Output is clipped to the request's remaining length. The selected timing model must price
verification passes; see
[performance-model configuration](../../perf-model/configuration.md#flat-engine-precision-and-execution-controls).

For a measured mean total acceptance length of 2.78 with three draft tokens,
set `nextn: 3` and `nextn_accepted: 1.78`: the base token accounts for the
remaining one token per round. A mean is an assumption, not a complete
empirical acceptance distribution.

The vLLM scheduler charges known and draft verification positions against
`max_batched_tokens`. A partial round can verify fewer drafts when its token
budget is small. Fresh requests and requests recomputing after preemption have
no drafts yet: their first output is one base token. An aggregated request's
first token comes from its final prefill forward, without a second decode
forward. Rejected speculative reservations are released, and preemption clears
draft readiness. TensorRT-LLM retains its existing scheduling approximation.

These are scheduling/progress semantics, not a new timing estimator: the
current timing callback still prices the configured verification round for a
partial-width decode. It does not interpolate dynamically by actual draft
width or enable measured full-SD-round FPM ingestion. These limits matter
when benchmarking tiny token budgets.

HBM-only agentic MTP/EAGLE replay is supported by the offline engine stack,
including aggregated and P/D configurations supported by the selected backend.
This is functional replay, not a claim of AgentX hardware accuracy.
Rejected with MTP: grouped caches, G2/G3 offload, and state-cache prefix matching
with `prefix_match_unit`. Combinations accepted by the
timing model are not necessarily supported by Replay; see
[feature support](../features.md).
