<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MiniMax-M2.7 pinned FPM profiles

[`hf_dataset.json`](hf_dataset.json) is an adapted index of NVIDIA-collected
measurement metadata from
[`nvidia/aisimulate-fpm-dataset/data/MiniMaxAI--MiniMax-M2.7`](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/tree/dbfcdd9eb62ea733efbc3f0cbd78277f2960061a/data/MiniMaxAI--MiniMax-M2.7)
at immutable revision `dbfcdd9eb62ea733efbc3f0cbd78277f2960061a`. Each entry records the
original artifact paths and SHA256 digests; table contents are fetched explicitly
and are not bundled in this directory. The dataset card declares Apache-2.0.
The repository's canonical `THIRD_PARTY_NOTICES.md` records this adaptation.

Nine profiles are `development`; two are `quarantined` for artifact retrieval
only. All require `allow_unqualified=True`. Publication does not establish serving
qualification, and each deployment retains a separate cache root.

The quarantined profiles are
`h200_sxm-vllm-fp8-block-tp4-ep4-kv-off` and
`h200_sxm-vllm-fp8-block-tp4-ep4-kv-on`. Their original sidecars lack
`coordinate_system`, `measurement_policy`, and top-level `system` and
`backend_version`, so the native FPM consumer cannot load them. Opting in
retrieves the unchanged artifacts; it does not make predictions possible.
Profile listings and opt-in warnings expose this limitation in `admission_reason`.

`packaged_system` selects the canonical hardware YAML from the installed
`aisimulate_core/systems` directory. The materializer copies it into the profile
root and includes its actual hash in cache identity and verification. This is a
current hardware reference, not an archived collection configuration. No
collection system YAML is published for these tables at the pinned dataset
revision. Native schema and exact-cell validation still apply after
materialization; profile names do not override table identity.
