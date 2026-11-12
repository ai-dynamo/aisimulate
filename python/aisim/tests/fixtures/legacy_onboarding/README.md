<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

These synthetic requests and checkpoint were serialized with the actual
`SupportRequest` and `Checkpoint` implementations at pre-rename commit
`4d961bab6164b18c0752d7b5f67ceb2ac8e3e933` in this repository, using the synthetic
profile from `tests/unit/test_onboard_checkpoint.py`. They are not measured data
or a real user's approval. The two configurations exercise persisted v1 and v2
requests, accepted-profile hashes, request-file hashes and collection-context
hashes. `collection.json` is a synthetic immutable artifact.

Keep these bytes and hashes independent of the current writer. Resume tests
copy them to a new directory and exercise both public module entry points.

`plans-v2.json` and `policy-v1.json` / `policy-v2.json` were emitted by the
original plan and validation-policy writers at
`ba02b3071c1ed09e65c7d8682e959c292dedb677`. Plan fixtures cover aggregated,
prefill, decode and unresolved-runtime-version campaigns. Only the original
writer's package-resource lookup was pointed at the moved, byte-identical
hardware YAML files. The fixed plan-root placeholder makes these outputs
portable; tests substitute a temporary absolute root and recompute only the
generated-file hashes affected by that substitution. They never regenerate
the expected plan with the current writer.
