<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Inkling-NVFP4 pinned FPM profiles

[`hf_dataset.json`](hf_dataset.json) is an adapted index of NVIDIA-collected
measurement metadata from
[`nvidia/aisimulate-fpm-dataset/data/thinkingmachines--Inkling-NVFP4`](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/tree/dbfcdd9eb62ea733efbc3f0cbd78277f2960061a/data/thinkingmachines--Inkling-NVFP4)
at immutable revision `dbfcdd9eb62ea733efbc3f0cbd78277f2960061a`. Each entry records the
original artifact paths and SHA256 digests; table contents are fetched explicitly
and are not bundled in this directory. The dataset card declares Apache-2.0.
The repository's canonical `THIRD_PARTY_NOTICES.md` records this adaptation.

All 3 profiles are `development` and require `allow_unqualified=True`.
Publication does not establish serving qualification. KV variants and comparator
tables retain separate profiles and cache roots.

`packaged_system` selects the canonical hardware YAML from the installed
`aisimulate_core/systems` directory. The materializer copies it into the profile
root and includes its actual hash in cache identity and verification. This is a
current hardware reference, not an archived collection configuration. No
collection system YAML is published for these tables at the pinned dataset
revision. Native schema and exact-cell validation still apply after
materialization; profile names do not override table identity.

The current built-in model-name construction path rejects the
`InklingForConditionalGeneration` architecture. These pins provide artifact
access for a compatible consumer; they do not add that architecture to the
canonical model registry.
