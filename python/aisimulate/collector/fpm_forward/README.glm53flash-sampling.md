<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Versioned GLM formal sampling candidates

Generate the proposed campaign inputs from the collector checkout:

```sh
PYTHONPATH=python/aisimulate python -m collector.fpm_forward.glm53flash_sampling \
  --output /results/glm53flash-candidates-v1
```

The generator performs no GPU work and makes no admission or accuracy claim.
Every candidate begins `NOT_EVALUATED`. The output directory must be new so a
previous inventory or attached failure evidence cannot be overwritten.

The default has **547 geometries per configuration**: 210 prefill and 116 decode
calibration points; 144 prefill and 77 decode holdouts. The same requested matrix
covers vLLM/SGLang × original FP8/NVIDIA NVFP4 × TP2/TP4, with DP/PP/CP/EP=1.
Immutable model and backend revisions come from the shared GLM descriptor.

## Files and identities

- `calibration-points.json` and `holdout-points.json` use the existing FPM
  planner's **schema 3**, with `prefill` and `decode` arrays. Tests pass both
  exact payloads through `FPMCollectionOptions.from_args`; their canonical
  hashes are the planner's benchmark-point hashes.
- `candidate-inventory.json` records every candidate, phase, context length,
  selection reason, and the calibration brackets for every holdout. Its
  campaign hash binds generator version, options, immutable revisions, corpus
  hashes and both complete point payloads.
- `calibration.txt` and `holdout.txt` are distinct original prose/arithmetic/
  procedural/Chinese text, copied byte-for-byte from `glm53flash_corpora/`.
  Their provenance and limitations are documented there. They are not upstream
  model training text or a claim of representative production routing.
- `qualification-template.json` has a slot for every requested candidate in
  each of the eight configurations, plus actual allocator/device-capacity and
  runtime-configuration receipts. Receipt slots are unfilled; this file never
  certifies qualification by itself.

The role-prefixed candidate IDs and proposed request namespaces are disjoint.
Native producers still own actual request creation and request IDs; use each
role's corpus and dataset-role when preparing its collection plan. Existing
holdout validation checks the observed native request IDs and corpus hashes.
Namespaces alone do not prove independent execution.

## Point selection

Base batches are 1, 2, 4, 8, 16 and 32. Context anchors are 1K, 2K, 4K, 8K, 16K,
32K, 64K and inclusive 128K. Cached-prefill queries are 32 and 256 tokens per
request. Full-prefill aggregate new-token anchors are 128, 512, 2048 and 8192.
The aggregate prefill budget stays at 8192, so long contexts at large batches
are cached-prefill requests, not oversized full-prefill requests.

The context limit includes the current query. Decode therefore allows at most
131071 past KV tokens per request, plus its one current token. Its FPM table
feature `total_prefill_tokens` remains zero.

Additional probes cover all four IndexPool prefix residues and nearby prefix
block anchors 128/4352; aggregate new-token chunk anchors 2048/8192; and decode
batch sizes immediately below/at/above graph anchors 4/8/16/32. The default edge
batch subset is 1/4/32; all six base batches retain the full context grid.
The 4352-token anchor is an explicit candidate for the observed hybrid-block
boundary, not a universal declaration about either backend. Actual native block,
chunk and graph configuration must appear in runtime qualification receipts.
Out-of-bounds neighbors, such as B33 or more than 8192 scheduled prefill tokens,
are listed separately as exclusions by declared mathematical limits. They were
never GPU-qualified or queued failures.

Each holdout lies strictly between two calibration points on one coordinate
axis, keeping batch, phase and the other coordinate unchanged. The inventory
records both endpoint IDs. This places holdouts inside the proposed geometric
interpolation domain; it does not prove that a runtime or consumer supports
interpolation across a particular kernel dispatch transition. Adjacent integer
calibration points with no disjoint interior are explicitly recorded. Identical
physical tuples across probe families are deduplicated with all reasons retained.
No calibration geometry can occur in the holdout set, even across families or
configurations.

## Qualification, then freeze

Use the complete inventory to request bounded native qualification. A successful
single native request and actual capacity check may qualify a candidate; formal
five-warmup/ten-measurement collection is not required before the plan is frozen.
Attach real file paths and SHA256s for the capacity/runtime receipts and each
candidate's exact scheduled geometry or failure. Capacity evidence must identify
the actual device/allocation and actual allocator outcome, including relevant
weights, KV/recurrent state, graph and workspace allocations. It is not a SOL
memory estimate. Native scheduling evidence must show that the requested batch,
query and prefix actually occurred under the declared runtime configuration.

Keep `NOT_EVALUATED`, `NOT_ADMITTED` and failed outcomes in the inventory. A
failure is not permission to delete a difficult point from the denominator.
If producer repairs or an explicitly revised campaign are required, retain the
original inventory and failure receipts. In particular, unchanged SGLang native
scheduling may not produce a long homogeneous B>1 decode or cached-prefill target;
those points remain requested qualification candidates until observed.

After reviewing actual qualification, the existing planner freezes the exact
schema-3 file and selected corpus bytes. Retain the candidate inventory hash and
qualification receipt references with that plan. This module intentionally does
not implement another native reader, collection loop or automatic freeze gate.
Formal native/raw and installed-consumer accuracy acceptance remain owned by
`glm53flash_validation.py` (FPM 10%, Ops 20%). Missing qualification or formal data
must never be described as passing.

Grid density and probe anchors can change **before** freeze using CLI options
such as `--contexts`, `--cached-queries`, `--edge-batches`, `--block-anchors`,
`--chunk-anchors`, and `--graph-batch-anchors` (comma-separated ascending integers).
Required six batch anchors, 1K/64K/128K anchors and declared limits are retained.
Changing inputs changes the campaign hash; the generator refuses to overwrite an
existing bundle. Increase density when native dispatch evidence requires it.
