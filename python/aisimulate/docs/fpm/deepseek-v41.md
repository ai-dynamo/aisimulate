<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1 FPM collection

Published experimental measurements cover the text backbone on one GB200 node
with pure TP4, native checkpoint precision, GPU-resident Engram, eager execution,
and DSpark disabled. The [GB200 calibration](../../../../data/experimental/deepseek-v41/gb200-fpm/calibration-v1/README.md)
contains 126 points; [38 independent forward holdouts](../../../../data/experimental/deepseek-v41/gb200-fpm/holdout-sol-review-v2/README.md)
provide descriptive accuracy evidence. GB200 ordinary HTTP serving verification
remains in progress. Separate [GB300 SGLang OFF/ON packets](../../../../data/experimental/deepseek-v41/gb300-fpm/README.md)
contain whole-forward calibration and serving comparisons, including their
remaining coverage and precision limits. These data use explicit experimental
systems overlays; the curated defaults are unchanged.

## Execution identity

Schema 7 appends four strings to the existing exact model/backend/topology
identity: `model_config_sha256`, `execution_profile`, `engram_residency`, and
`input_modality`. The config hash uses canonical JSON after the same quantization
normalization used by the SDK. V4.1 uses `hbm_tp_sharded` and `text`; its profile
is `full` or `decoder_bounded`. Changing the config or profile cannot borrow
another cell. Schema 6 loads with the legacy values `""`, `full`, `none`, `text`.
Publication upgrades a validated schema-6 pair in memory before emitting schema 7.

The shared model retains the entire resident-weight inventory even when a replay
profile executes fewer decoder tokens. FPM interpolation uses the original
stage-aware SOL graph. Vision and speculative decoding remain outside this
campaign's measurement contract.

When a native FPM table labels FMHA by its cache precision, select that table
with `fpm_fmha_dtype: "fp8"` in native engine/replay JSON, or
`ModelConfig(fpm_fmha_quant_mode=FMHAQuantMode.fp8, forward_model="fpm")`.
This option requires `forward_model="fpm"` and changes only the exact FPM cell
selector. The checkpoint's analytical attention graph, interpolation SOL
anchors, and memory inventory remain unchanged. `activation_dtype` retains its
existing arithmetic-override meaning; it is not a substitute for this selector.
The qualified GB200 comparison uses the table selector with no activation
override, and its optional op-level SOL comparison uses checkpoint precision.

`--fpm-decoder-replay` describes true bounded decoder execution. The current
vLLM route rejects it because the verified preview executes the full backbone.
Prefix-cache/SWA tail recomputation does not establish true decoder replay.
Replay OFF data must never be relabeled as ON data.

## Slurm transport

`--fpm-executor slurm` runs inside a caller-owned Slurm allocation with Pyxis.
It requires an explicit `--fpm-slurm-container-image` and accepts repeated
`--fpm-slurm-container-mount SOURCE:TARGET` arguments. The frozen Generator
manifest determines the node count; the allocated count must agree. A shared
campaign directory must be visible at the same path on each node.

The transport reuses Generator scripts, native result validation, attempt
identity, recovery, and formal publication. It launches one engine per node
with explicit ranks and a common master, and collects under stable `nodeNNNN`
execution-unit names. Cleanup cancels only receipted, campaign-specific steps
and verifies their exit; it never cancels the caller's allocation. Kubernetes
remains the default transport.

## Qualification and publication gates

Use a digest-pinned ARM64 image for GB200. The official preview recipe is at
[vLLM recipes commit ce19de141d448df3e739de977da0830e9a175de5](https://github.com/vllm-project/recipes/blob/ce19de141d448df3e739de977da0830e9a175de5/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml).
The model config revision is
[fb2764a5cf321eaa5070ca8f9e892818f477c16d](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/fb2764a5cf321eaa5070ca8f9e892818f477c16d).
Image support for loading the model does not prove that its Dynamo instrumentation
satisfies the Collector's native FPM contract. The ordinary preflight must pass.

Engram makes token history part of the workload. Qualification therefore needs
a reproducible tokenizer-generated text corpus and real model-computed KV for
both cached prefills and decode. V4.1 cached-prefill and decode publication reject `fake_fallback`,
legacy, and skipped-warm provenance; the consumer rejects them too. A producer
must earn the `real_kv` marker through execution. The initial canary should use
small batch/context limits, then expand the matrix only after native artifacts,
config identity, input provenance, and timing checks pass.

Record the immutable checkpoint/image/instrumentation revisions, corpus hash,
resolved engine settings, native artifacts and measured coverage with the data.
Keep internal cluster names, account names, filesystem paths, and raw operational
logs in private campaign storage.


The pinned producer is packaged under
`collector/fpm_forward/runtime/dsv41/`. It executes seed chunks on the same
request and block tables as the measured step, retaining Engram history and
compressor ring state. Source hashes cover the installed ARM vLLM files and
Dynamo scheduler/point/FPM definitions. `ai-dynamo-runtime==1.4.2` has passed
an actual import check with Dynamo source `54960177085413259859c88bd34ed0734d4c2ea9`.
This import check does not establish GPU numerical or timing correctness.

Its initial native grid is bounded to batch 2, per-request context 2048, total
scheduled prefill tokens 512, and global benchmark warmup 0. Use model limit
2050 to include the native decode context-2048 endpoint. Every completed point
archives actual prompt/output tokens in an adjacent JSONL file; the Collector
checks the file checksum, point coverage, per-request computed counts, and
completed seed witness. Unsupported or incomplete points fail the campaign.
The original mixed English/Chinese fixture is reproducible and is not a
representative production workload; Engram locality and routing sensitivity
remain limits on generalizing any resulting curve.
