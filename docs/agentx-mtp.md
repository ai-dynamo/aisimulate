<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Speculative decoding in AgentX replay

AgentX Weka replay uses the existing SDK speculative schemes and native token
sampler in aggregated or prefill/decode deployments. This is a `functional_only`
simulation with modeled costs, not a GPU benchmark or hardware accuracy result.

Start with the [AgentX quickstart](agentx-quickstart.md), keep its traffic and
deployment configuration, and add an explicit speculative assumption:

```yaml
engine:
  model: nvidia/GLM-5.2-NVFP4
  hardware: b300_sxm
  backend: sglang
  backend_version: 0.5.14
  estimation_mode: op_level
  speculation:
    kind: mtp
    num_speculative_tokens: 3
    expected_accepted_tokens: 1.99
    seed: 42
```

`num_speculative_tokens` is the draft count, from 1 to 5.
`expected_accepted_tokens` counts accepted drafts, excluding the base token;
1.99 therefore means an expected acceptance length of 2.99 including the base.
The existing sampler realizes fractional expectations using adjacent integer
counts. Acceptance changes token progress; it does not discount the modeled
cost of draft layers and widened target verification. A shared seed reproduces
the same configuration; changing routing or worker count can change its draws.

The GLM value references K3, thinking-on AL2.99 in
[InferenceX's GLM-5.2 MTP reference](https://github.com/SemiAnalysisAI/InferenceX/blob/400dfe463877e50a1c4bb9705d99919b773c33bb/inferencex-e2e/infx/golden_al_distribution/glm5.2_mtp.yaml).
That immutable source measured FP8/vLLM on B300 using SPEED-Bench coding data,
temperature 1.0, top-p 0.95 and output length 4096. Applying its mean to
NVFP4/SGLang is a simulation assumption, not a measured acceptance result.

The same MTP interface accepts a different configured target. For example, set
`model: deepseek-ai/DeepSeek-V4-Pro` and `expected_accepted_tokens: 1.5` to model
hypothetical target-shaped MTP while retaining DSV4's attention and MoE graph.
This does not model its real DSpark draft architecture. Unsupported targets
remain errors; Kimi-K3's DSpark-specific NextN graph cannot be relabeled MTP.

For other schemes, use SDK parameters and a draft checkpoint or its parsed
`draft_config`. For example, with a compatible Qwen3-8B EAGLE3 draft:

```yaml
engine:
  model: Qwen/Qwen3-8B
  speculation:
    kind: eagle3
    params: {tree_shape: [1, 4, 4], verify_token_budget: 10}
    draft_model_path: /path/to/eagle3-draft
    expected_accepted_tokens: 1.5
    seed: 42
```

The generic form also accepts `mtp`, `ngram`, `dflash`, `draft_model` and
`dspark`, with their existing SDK `params`. Specify exactly one acceptance
input: `expected_accepted_tokens`, or per-depth conditional `acceptance_rates`.
A tree's accepted path depth differs from its verification budget; this example
accepts at most three drafts while pricing ten target verification tokens.
Resolved draft configuration survives saving and reloading. Acceptance and
seed affect progress, independently of the draft cost identity.

Existing SDK family/backend checks apply: EAGLE3 supports the LLAMA family on
vLLM/SGLang; DFlash supports LLAMA/DEEPSEEKV4 on vLLM; standalone draft models
use vLLM; dense DSpark supports LLAMA on vLLM/SGLang. DSpark V4's query overrides
remain unsupported by the compiled engine. MiniMax EAGLE3 is also unsupported;
an MTP surrogate on MiniMax is not a measurement of its real Eagle draft.

Automatic capacity uses the existing target and draft memory hooks. Explicit
fixed blocks must already account for the intended deployment's reservations.
AgentX speculative replay retains HBM-only ordinary KV and its
prefix/grouped-cache restrictions. Existing non-SD G2 replay remains available.
The Weka hash block size remains independent of the engine KV block size.

Run the saved configuration with `aisimulate predict --config config.yaml`.
With the matching Dynamo adapter, add `--stack dynamo`; existing round-robin
and KV-router paths use the same speculative inputs. This change does not add
Dynamo profiles, snapshots, warmup or session/sibling routing capabilities.
P/D keeps the same configured target in both roles and its existing transfer
model; DSV4 transfer still uses the scalar bytes-per-token approximation.

The legacy `nextn` and `nextn_accepted` inputs remain supported. Use one input
form per engine; do not combine them with an explicit `speculation` block.
Saved configurations retain the chosen method. Compare completed requests,
delivered output tokens and modeled latency; output limits can truncate a
speculative burst. See [the core API](core-api.md#choosing-a-forward-pass-api)
for the canonical cost interface and Rust source compatibility notes.
