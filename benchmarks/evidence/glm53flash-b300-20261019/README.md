<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Experimental GLM-5.3-Flash B300 FP8 TP4

**Admission is development; default materialization is blocked.** The independent
prefill prediction gate failed. Two held-out cases are outside the collected
domain, and supported prefill cases have large interpolation errors. The failed
result is retained unchanged in `validation.json`.

This contribution adds measured native forward-step timings for B300 SXM, vLLM
0.31.0, FP8 weights and sparse KV, text only, with TP4 and DP = PP = EP = 1.
The collection uses prefix caching, Mamba align mode, and the unchanged official
Flash producer from AISimulate `8d021d5a41f6f47fd3d4c8064be6361b7c8b3c85` with
Dynamo Python instrumentation `54960177085413259859c88bd34ed0734d4c2ea9`.
`collection.json` records the physical hardware, runtime image, installed packages,
source overlays, model revision, token generator and evidence hashes.

The calibration table has 212 prefill and 147 decode cases. Each case runs five
warmups and ten measurements; its table value is the median native FPM wall time
in milliseconds. The three input JSON files preserve the frozen request lengths
and decode contexts. The measured context limit is 131,072 tokens per request;
the runtime limit includes seven tokens of admission headroom.

`validation.json` reports calibration loading checks, 30 disjoint held-out cases,
and a separate full-grid engine launch. `holdout-results.json` preserves every
held-out prediction and measured median. Calibration self-queries check the data
path and are not an accuracy test. The two-launch, source-inclusive comparison
is diagnostic; it does not establish the canonical fresh-population repeatability
qualification. No valid slow measurements were removed.

The CPU consumer was built with Python and Rust from source commit
`c9e7363bf5983b980fe6622f7b064ee7adff5f5d`. `replay-validation.json` records finite
standalone request replay and exact completion/token counts.
`hf-roundtrip-validation.json` verifies all three uploaded file hashes, default
serving admission rejection, and two completed requests using the downloaded data. These tests exercise
no cache reuse and do not establish production serving accuracy, 1M context
coverage, other parallel layouts, or a supported DynoSim dependency pairing.

To consume the data, materialize `b300-vllm-fp8-tp4` from the checked-in
`glm53flash_fpm/hf_dataset.json`, then use its returned root as `systems_paths`
with explicit `--allow-unqualified` diagnostic opt-in, system `b300_sxm`, vLLM `0.31.0`, TP4, FP8 KV, `fpm_interpolation`,
`fallback_policy="deny"`, strict provenance and shared layers disabled.
The [model guide](../../../docs/glm53flash-fpm.md) documents the native producer.
Raw native artifacts and all individual repetitions are retained by the collector;
this directory contains their hashes and the bounded validation evidence.

Recommended next work is to extend the small-prompt coverage and sample the
64-scheduled-token graph/eager transition, then test new blind cases. The review
asks whether the recommended FPM grid or interpolator should separate dispatch
regimes. The existing holdout results must remain failed; they are not a new
blind validation after any tuning.
