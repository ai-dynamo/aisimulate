---
name: aisimulate-agentx-collect
description: Collect and qualify self-collected AgentX end-to-end silicon benchmarks with complete serving-feature evidence, request accounting, and reproducible workloads. Use when preparing or reviewing an AgentX GPU collection; not for parsing official exports or running prediction-only campaigns.
license: Apache-2.0
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Collect AgentX silicon evidence

Produce a reproducible silicon bundle whose eight configured feature settings
are resolved from the actual serving deployment. Read the collection's existing
recipe and budget first. This skill does not authorize a GPU allocation, restart
a stopped campaign, enlarge a sweep, upload evidence, or publish a dataset.

Use the existing [Qwen collection](../../../benchmarks/evidence/accuracy/qwen-weka/README.md)
and its `collect.py`, `receipt_probe.py`, and `normalize.py` as a starting point
when they match the task. They are a version-specific example, not defaults for
another model/backend. Preserve the requested workload and framework version.
For replay qualification and gap diagnosis, use the adjacent
[AgentX accuracy skill](../aisimulate-agentx-gap-debug/SKILL.md).

## Capture settings before the measured sweep

After server initialization, capture effective settings for every worker/rank,
not just non-default command-line flags. Keep exact launch arguments, the selected
recipe/override, relevant environment, full resolved server configuration,
frontend/router/connector topology, and runtime/cache implementation inventory.
Preserve requested and effective values separately when initialization changes
them. Reject unexplained rank disagreements; do not use rank zero as a substitute
for checking the other ranks.

Pin the installed framework version and source revision (including local patches),
container digest, checkpoint/tokenizer and draft-checkpoint revisions. A database
version, a moving image tag, or a model name cannot establish these identities.
Capture feature evidence after final configuration resolution and verify it is
still the configuration used by each measured profile.

Each feature record uses `aisimulate.benchmark.features.v1` and has `id`, `state`,
`parameters`, `observed`, and `evidence`. Store exactly the eight rows below. If an
existing normalizer emits a broader inventory, use the worker to project these
rows into `features.json`; preserve the other dimensions in its workload artifact.
Configured state is
`enabled`, `disabled`, `not_applicable`, or `unknown`; every resolved state needs
SHA-256 and JSON-pointer references to retained evidence. Never put schema versions
in directory, module, command, or artifact names.

| Feature identity | Required capture |
| --- | --- |
| `cache.g1_prefix` | Effective prefix/radix caching flag, including forced compatibility overrides. Cold cache does not mean caching is disabled. |
| `cache.sub_block_prefix_reuse` | Effective lookup/match granularity and allocation block size, plus the setting or implementation evidence establishing reuse within a block. A block size or cache-hit count alone is insufficient. |
| `cache.g2` | Effective host KV-offload enablement and, when on, local/shared scope, capacity and implementation settings. Do not confuse weight offload or a host staging buffer with KV offload. |
| `cache.g3` | Effective storage KV-offload enablement, backend/connector and local/shared scope when on. A missing export is not Off. |
| `cache.hybrid_state` | Actual cache-group/spec inventory; evidenced recurrent-state, sliding-window, or compressed-cache types; relevant implementation, state dtype and cache settings. Model architecture alone does not prove cache behavior. |
| `speculation` | Effective method and numeric configured draft length when on; checkpoint identity in configuration/provenance. Methods include MTP, DSPARK, EAGLE3, DFlash, DFlash2, n-gram and other draft models. When off use `method: off`, `draft_length: null`. Acceptance length/rate and step count are separate measurements. |
| `cache.kvcr` | Explicit Dynamo KVCR/connector configuration, or complete native-serving topology establishing N/A. A generic transfer connector does not prove KVCR. |
| `routing.kv_aware` | Actual frontend/router deployment and routing policy, or complete native-serving topology establishing N/A. Planner/autoscaling flags do not establish KV-aware routing. |

For G2/G3 that are Off/N/A, use `scope: not_applicable`. When hybrid caching is
On, record `cache_types`; when sub-block reuse is On, record
`match_granularity_tokens`. Preserve simultaneous settings and their actual
values even when AISim cannot simulate them. Do not clamp to simulator limits.

For a direct native launch, retain a complete recipe with its actual `engine`,
`frontend.type`, `frontend.enable_multiple_frontends`, roles and launched arguments.
The dataset worker can recognize a native single-role deployment from that
evidence. Do not manufacture a native topology receipt for a deployment that
actually includes Dynamo, an external router, or unrecorded wrappers.

If a framework does not expose an effective field, capture the consumer's runtime
value through a read-only probe, or retain the exact installed source and a
reviewed derivation including all applicable overrides. Distinguish a source-
resolved value from a runtime observation. Do not silently default a missing flag
to Off. If a setting remains unknown, retain the attempt and its reason, fix the
capture, and stop qualification before spending the remaining sweep budget.

## Keep configured state and observed use separate

An enabled setting does not prove exercise. Keep observed state as `unknown`
unless retained telemetry supports `exercised` or `not_observed` with a count.
Name each count's phase, scope, denominator and instrumentation coverage. Missing
instrumentation is not a zero count. Cache snapshots remain snapshots; never
invent continuous scheduler/cache series between them.

Keep model/topology/precision, runtime, graph policy, workload, warmup, delays and
conversation structure outside the eight-row checklist. Retain per-request
identities, actual input/output lengths, stream timestamps/counts, parent/child
relationships, source selections, corpus cursors and cuts, seeds, cache receipts,
warmup and preparation/profile/grace boundaries. Account for every attempted,
failed, cancelled and unmatched request. Preserve the actual finite-cohort or
continuous-profile boundary and tool delays.

## Qualify and hand off

1. Before the full sweep, inspect a small authorized pilot's complete feature
   receipt and verify every referenced hash/pointer against the raw evidence.
2. Run the dataset worker's strict feature check from an
   `aisim-hf-dataset-ci` checkout:

   ```sh
   python -m ci.benchmark_features /absolute/path/to/features.json
   ```

   This checks feature completeness, not the existence of referenced payloads.
   The worker's full bundle validation checks artifact hashes and JSON pointers.
   A new active self-collected intake must pass both checks. Historical/failed
   evidence may retain unknowns and must remain explicitly unqualified.
3. Verify the pilot's request accounting, measurement boundaries and, if the task
   requires it, native replay qualification before releasing the rest of the
   authorized sweep. Keep failed attempts and their original qualification state.
4. Hand off silicon evidence, actual settings, source provenance and pinned
   upstream corpus references to the dataset worker. Keep simulation configs,
   predictions, performance database versions, support verdicts and generated
   replay artifacts outside silicon bundles. Raw prompts, token arrays, streams
   and logs stay private; public numerical projections use the existing allowlist.

The canonical storage/intake contract lives in the dataset worker's
[`docs/agentx-schema.md`](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-hf-dataset-ci/-/blob/simonec/agentx-e2e-intake/docs/agentx-schema.md).
Use bundle-relative references and stable corpus descriptor IDs so self-collected
bundles can later move unchanged to `nvidia/aisim-dataset/data/e2e/`.
Downloaded official AgentX evidence remains separate in the private cache.
