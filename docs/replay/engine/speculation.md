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
| `engine.nextn_accepted` | Unset | fixed | Required when `nextn > 0`. Expected accepted tokens per round, in `[0, nextn]`. |

`recommend` keeps these values fixed for every candidate; it does not search
draft length or acceptance.

<a id="prompt-lookup-ngram"></a>
<a id="prompt-lookup-ngram-speculative-decoding"></a>

## Prompt lookup (ngram)

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
a draft is available in every round; real n-gram matching, rounds without a
draft and CPU lookup time are not modeled. With `fixed` or `polynomial` timing,
the decode time you supply is used per verification round as-is.

Supported: vLLM aggregated or disaggregated workers with `op_level`, `fixed` or
`polynomial` timing, offline `engine` stack. Rejected: SGLang, TensorRT-LLM, FPM timing, AFD/EPD, and host
or G3 offload. Agentic traffic and the Dynamo stack are not supported.
Deployment artifacts are not generated for ngram candidates.

Prompt lookup is unrelated to `kv_cache.prefix_caching`.

<a id="mtp"></a>

## MTP / EAGLE

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

Rejected with MTP: grouped caches, G2/G3 offload, state-cache prefix matching
with `prefix_match_unit`, and agentic traffic. Combinations accepted by the
timing model are not necessarily supported by Replay; see
[feature support](../features.md).
