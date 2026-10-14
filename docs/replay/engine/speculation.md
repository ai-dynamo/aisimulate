<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Speculative decoding

Replay supports three speculative-decoding inputs. They cannot be combined.

| Input | Drafts from | Configure with |
| --- | --- | --- |
| [Prompt lookup (ngram)](#prompt-lookup-ngram) | Earlier tokens of the request | `engine.speculation` |
| [Fixed-length DSpark](#dspark) | A parallel draft backbone | `engine.speculation` |
| [MTP / EAGLE](#mtp) | A draft head of the model | `engine.nextn`, `engine.nextn_accepted` |

For each scheme you provide how often draft tokens are accepted. That rate
depends on the workload, so Replay does not predict it. Each decode round then
emits a sampled number of tokens, and its cost is a verification pass over the
draft tokens.

## Speculation fields

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.speculation` | Unset (off) | fixed | Optional ngram or DSpark block below. Rejected together with `engine.nextn > 0`. |
| `engine.speculation.kind` | Required | fixed | `ngram` or `dspark`. |
| `engine.speculation.num_speculative_tokens` | Required | fixed | Ngram: 1–5. DSpark: positive integer below `2^32 - 1`. |
| `engine.speculation.acceptance_rates` | Required for ngram | fixed | Exactly K conditional probabilities in `[0, 1]`. DSpark requires this or `accepted_tokens`, never both. |
| `engine.speculation.accepted_tokens` | Unset | fixed | DSpark only. Expected accepted draft tokens in `[0, K]`, excluding the bonus token. |
| `engine.speculation.adaptive_verification` | `false` | fixed | DSpark only; `true` is rejected. |
| `engine.speculation.draft_model_path` | Target checkpoint | fixed | DSpark draft checkpoint; omit for bundled drafts. |
| `engine.speculation.draft_revision` | Unset | fixed | DSpark HF draft revision. Separate checkpoints require this or an explicit config snapshot. Use an immutable commit for reproducibility. |
| `engine.speculation.draft_config` | Resolved from checkpoint | fixed | Optional complete draft config snapshot. Retained in cost identity and provenance. |
| `engine.speculation.seed` | `42` | fixed | Unsigned 64-bit sampling seed. |
| `engine.nextn` | `0` (off) | fixed | MTP/EAGLE draft tokens, 0 to 5. |
| `engine.nextn_accepted` | Unset | fixed | Required when `nextn > 0`. Expected accepted tokens per round, in `[0, nextn]`. |

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
a draft is available in every round; real n-gram matching, rounds without a
draft and CPU lookup time are not modeled. With `fixed` or `polynomial` timing,
the decode time you supply is used per verification round as-is.

Supported: vLLM aggregated or disaggregated workers with `op_level`, `fixed` or
`polynomial` timing, offline `engine` stack. Rejected: SGLang, TensorRT-LLM, FPM timing, AFD/EPD, and host
or G3 offload. Agentic traffic and the Dynamo stack are not supported.
Deployment artifacts are not generated for ngram candidates.

Prompt lookup is unrelated to `kv_cache.prefix_caching`.

<a id="dspark"></a>

## Fixed-length DSpark

```yaml
engine:
  estimation_mode: op_level
  fallback_policy: deny
  speculation:
    kind: dspark
    num_speculative_tokens: 6
    adaptive_verification: false
    accepted_tokens: 2.77
    seed: 42
```

K counts proposed drafts, excluding the bonus. The example commits an average
of 3.77 tokens per round. Fractional `accepted_tokens` uses adjacent-integer
sampling: 2 or 3 accepted drafts, plus one bonus. Conditional rates instead
sample successive acceptance until the first rejection. Output/context limits
clip committed progress; they do not change the configured cost width.

Supported paths are vLLM and SGLang aggregated/P/D workers, including AgentX
multi-turn replay, with op-level timing. Fixed/polynomial round costs can test
scheduler behavior. FPM, adaptive verification, external cache/offload, AFD,
EPD and online deployment generation remain unsupported. Legacy MTP limits
and layer scaling do not apply to DSpark.

The graph prices context projection/KV preparation, one parallel K-token
draft, logits, sequential Markov steps and K+1 target verification. V4 context
preparation follows the pinned vLLM padded-query contract. SGLang uses this
cost approximation pending matched operation measurements. V4 draft attention
uses a window-capped HCA proxy; target wide-query attention remains a token-basis
proxy. These are coverage limitations, not measured kernel equivalence. Host
sampling/launch overhead and eager-versus-graph differences are not calibrated.

Automatic capacity includes draft weights and draft KV, reserving speculative
positions for every configured active sequence. Shared target embedding/head
weights are counted once. The scalar capacity and target prefix-cache model
remain approximations to V4's grouped cache. Fixed block capacities are user
inputs and must already account for the draft's resident memory.

Recommendations retain DSpark and acceptance settings unchanged. The source
checkpoint must actually contain DSpark metadata; the older bundled V4 target
configs alone are insufficient. Use the matching checkpoint or an explicit
resolved draft snapshot.

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

Rejected with MTP: grouped caches, G2/G3 offload, state-cache prefix matching
with `prefix_match_unit`, and agentic traffic. Combinations accepted by the
timing model are not necessarily supported by Replay; see
[feature support](../features.md).
