<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Speculative decoding

Prediction Replay accepts the following speculative-decoding inputs. Select
one input; `engine.speculation` cannot be combined with `engine.nextn` or
`engine.nextn_accepted`.

| Input | Drafts from | Configure with |
| --- | --- | --- |
| [Prompt lookup (ngram)](#prompt-lookup-ngram) | Earlier tokens of the request | `engine.speculation` |
| [Registered speculative schemes](#registered-schemes) | The selected scheme's draft head or checkpoint | `engine.speculation` with `params` and `expected_accepted_tokens` |
| [MTP / EAGLE](#mtp) | A draft head of the model | `engine.nextn`, `engine.nextn_accepted` |

In each case you provide how often draft tokens are accepted. That rate
depends on the workload, so Replay does not predict it. Each decode round then
emits a sampled number of tokens. The selected cost model prices the configured
verification round, including draft operations for schemes that model them.

## Speculation fields

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.speculation` | Unset (off) | Ngram only, fixed | Ngram block or registered-scheme block below. Rejected together with `nextn` or `nextn_accepted`; omit it to disable speculation. |
| `engine.speculation.kind` | Required | Ngram only, fixed | `ngram` for prompt lookup; registered schemes include `mtp`, `eagle3`, `dflash`, `dspark`, and `draft_model`, subject to model/backend and replay limits below. |
| `engine.speculation.params` | `{}` | Rejected | Scheme-specific cost parameters. The scheme must resolve to 1–5 draft tokens; EAGLE requires a chain. |
| `engine.speculation.draft_model_path` | Unset | Rejected | Draft checkpoint path or model ID, when required by the scheme. |
| `engine.speculation.draft_config` | Unset | Rejected | Resolved draft geometry retained with the cost identity. See the [cost configuration](../../perf-model/configuration.md#speculative-decoding-configuration) for checkpoint requirements. |
| `engine.speculation.expected_accepted_tokens` | Required for registered-scheme input | Rejected | Expected accepted draft tokens per round, excluding the base token, in `[0, resolved draft depth]`. |
| `engine.speculation.num_speculative_tokens` | Required for flat ngram input | fixed | Integer from 1 to 5. |
| `engine.speculation.acceptance_rates` | Required for flat ngram input | fixed | Exactly one conditional acceptance probability per draft token, each in `[0, 1]`. |
| `engine.speculation.seed` | `42` | fixed | Unsigned 64-bit sampling seed. |
| `engine.nextn` | `0` (off) | fixed | MTP/EAGLE draft tokens, 0 to 5. |
| `engine.nextn_accepted` | Unset | fixed | Required when `nextn > 0`. Expected accepted **draft** tokens per round, excluding the base token, in `[0, nextn]`. |

`recommend` retains the flat ngram and legacy `nextn` inputs, fixed for every
candidate. Registered-scheme blocks are prediction-only. Recommendation does
not search draft length or acceptance.

## Registered schemes

The prediction `engine.speculation` block combines the existing
[canonical cost configuration](../../perf-model/configuration.md#speculative-decoding-configuration)
with a workload acceptance assumption and a sampling seed. For example, merge
this fragment into an aggregated prediction for a supported model:

```yaml
engine:
  speculation:
    kind: mtp
    params: {depth: 3}
    expected_accepted_tokens: 1.5
    seed: 42
  workers:
    aggregated:
      kv_cache:
        capacity: {type: fixed, blocks: 4096}
```

The block count is illustrative: supply the target KV capacity available after
reserving memory for the selected draft weights and draft KV cache. Every
configured language-worker role requires explicit fixed capacity for a generic
scheme, including MTP configured through this block. Replay does not infer that
capacity. Legacy `nextn` and flat ngram keep their existing capacity paths.

For a full draft-ready round, `expected_accepted_tokens: 1.5` means one draft
token is always accepted and a second is accepted with probability 0.5, before
clipping to the remaining output length. The base token is additional.
Acceptance and `seed` control progress;
they are excluded from the cost configuration passed to
`ForwardPassPerfModel::best_available`. The scheme parameters and resolved draft
identity are preserved through that constructor and saved configuration.
Direct replay descriptors using a generic canonical timing identity must also
provide explicit `aic_nextn_accept_rates` on the worker rank. They do not inherit
the legacy `nextn` acceptance defaults. The rank may omit `aic_nextn`; native
replay derives it from the canonical scheme. An explicit rank depth must match.
The legacy `aic_nextn_accepted` shorthand still requires an explicit rank depth
to convert the expected count to acceptance rates.

For a linear EAGLE3 draft, use `kind: eagle3`,
`params: {num_speculative_tokens: 3}`, and the draft checkpoint path or validated
configuration required by the scheme. Standalone `draft_model` still requires
its checkpoint path; retained `draft_config` does not replace checkpoint access.
Scheme-specific model and backend checks remain unchanged. Explicit MTP is
rejected for Kimi-K3 (`KimiK3ForConditionalGeneration`), the architecture in the
DSPARK capability gate. MiniMax EAGLE3 remains unsupported.
Selecting a scheme does not override these model-family gates.

Replay uses the existing NextN executor for a linear chain of 1–5 draft tokens.
EAGLE trees, a verification budget inconsistent with the chain, and wider
blocks are rejected even when the cost API can estimate them. A scheme name
does not enable a separate scheduler, tree traversal, dynamic verification
width, or draft offload behavior. The first-token, partial-round and preemption
semantics described under [MTP](#mtp) still apply.

Supported compositions use the offline `engine` stack, vLLM or SGLang language
workers, aggregated or P/D topology, and synthetic or HBM-only agentic traffic.
The selected scheme may impose a narrower backend/model constraint. Use default
timing so the canonical constructor retains draft costs; fixed and polynomial
timing are rejected for generic schemes. OP/SOL support and estimator fallback
follow the [cost-model contract](../../perf-model/configuration.md#speculative-decoding-configuration):
an explicit FPM request with `fallback_policy: deny` fails, while an allowed
fallback can select OP without discarding the scheme.

Grouped caches, G2/G3 offload, state-cache `prefix_match_unit`, AFD/EPD,
TensorRT-LLM, and optional Dynamo-stack execution are outside this generic
replay path. Functional replay does not establish AgentX prediction accuracy,
model quality, or measured hardware speedup; see [feature support](../features.md).

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

An engine-level FPM `estimation_mode` request with `fallback_policy: allow` can
fall back to OP timing through the canonical estimator. With `deny`, that
unsupported request fails instead.

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
