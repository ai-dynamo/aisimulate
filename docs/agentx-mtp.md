<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MTP in AgentX replay

AgentX Weka replay can use the existing MTP/NextN scheduler on vLLM and SGLang,
with aggregated or prefill/decode deployments. This is a `functional_only`
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

The interface also accepts a different configured target. For example, set
`model: deepseek-ai/DeepSeek-V4-Pro` and `expected_accepted_tokens: 1.5` to model
hypothetical target-shaped MTP while retaining DSV4's attention and MoE graph.
This does not model its real DSpark draft architecture. Unsupported targets
remain errors; Kimi-K3's DSpark-specific NextN graph cannot be relabeled MTP.

For each worker, supply explicit fixed KV capacity, for example
`kv_cache: {block_size: 64, capacity: {type: fixed, blocks: 8192}}`.
Choose capacity for the intended deployment after draft and graph reservations;
the example block count is not a measured memory fit. Agentic MTP requires
HBM-only ordinary KV and AIC op-level timing. Host offload, grouped KV/FPM,
incompatible prefix-match settings and other draft methods remain unsupported.
The Weka trace's hash block size is independent of the engine KV block size.

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
