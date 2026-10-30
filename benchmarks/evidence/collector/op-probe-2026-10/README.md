<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Op-probe findings (2026-09 to 2026-10)

This non-packaged bundle holds the serving observations that the collector
and generator changes for the vLLM 0.30.0, SGLang 0.5.21 and TensorRT-LLM
1.3.0rc29 pins cite as provenance. The op-probe harness (`opharness`) produced
them by booting the real serving framework per model and recording the
kernels and backends it selected. The harness lands separately.

Citations in the code name either a finding key or a results file below. Grep
a finding key, such as `sm100_probe_failures_recheck_2026_10_04`, to reach its
statement.

## Files

- `findings.yaml`: curated conclusions, one entry per key. Each statement gives
  its own versions, dates and hardware inline, and `applies_to` lists the
  models it was observed on. Evidence paths inside statements refer to
  workspace bundles that are not committed.
- `results/sm100/sglang-0.5.21.yaml`: B200 boot-and-run verdict per model on
  SGLang 0.5.21 under the default generator render.
- `results/retests/sm100/sglang-0.5.21.moe_auto.yaml`: the B200 SGLang 0.5.21
  retest of the MoE models under an explicit `--moe-runner-backend auto`.
- `results/sm100/trtllm-1.3.0rc29.yaml`: B200 boot-and-run verdict per model on
  TensorRT-LLM 1.3.0rc29.

Findings describe framework releases and rot with them. Re-verify them on the
next version bump instead of extending them.
