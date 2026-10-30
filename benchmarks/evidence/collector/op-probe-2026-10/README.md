<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Op-probe findings (2026-09 to 2026-10)

This non-packaged bundle holds the serving observations that the collector
and generator changes for the vLLM 0.30.0, SGLang 0.5.21 and TensorRT-LLM
1.3.0rc29 pins cite as provenance. The op-probe harness (`opharness`) produced
them by booting the real serving framework per model and recording the
kernels and backends it selected. The harness lives in
`python/aisimulate/collector/opharness/`.

Citations in the code name either a finding key or a results file below. Grep
a finding key, such as `sm100_probe_failures_recheck_2026_10_04`, to reach its
statement.

`findings.yaml` is the complete archive of the campaign's 100 findings. The
results files are excerpts: each keeps `_meta` plus the models the citing
comments name, and `_meta.summary` still counts the full matrix.

The harness's intermediate outputs are not committed: the full per-GPU
matrices, kernel-path verdicts, decompositions, retests and campaign ledgers.
Paths such as `results/pathdiff/...` inside finding statements name those
outputs. The 2026-10 set is preserved at commit `76bacff5b71db00be053a0f61697daf7622ef6e9` of the
`feat/op-probe-harness` branch on `tianhaox/aisimulate`, under
`python/aisimulate/collector/opharness/results/`. Later campaigns pack them
into their evidence bundles with `components/evidence_bundle.py`.

## Files

- `findings.yaml`: curated conclusions, one entry per key. Each statement gives
  its own versions, dates and hardware inline, and `applies_to` lists the
  models it was observed on. Evidence paths inside statements refer to
  harness outputs and workspace bundles that are not committed.
- `results/sm100/sglang-0.5.21.yaml`: B200 boot-and-run verdict per model on
  SGLang 0.5.21 under the default generator render.
- `results/retests/sm100/sglang-0.5.21.moe_auto.yaml`: the B200 SGLang 0.5.21
  retest of the MoE models under an explicit `--moe-runner-backend auto`.
- `results/sm100/trtllm-1.3.0rc29.yaml`: B200 boot-and-run verdict per model on
  TensorRT-LLM 1.3.0rc29.

Findings describe framework releases and rot with them. Re-verify them on the
next version bump instead of extending them.
