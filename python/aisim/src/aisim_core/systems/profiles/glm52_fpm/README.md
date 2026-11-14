<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.2-NVFP4 pinned FPM profiles

[`hf_dataset.json`](hf_dataset.json) is an adapted index of NVIDIA-collected
measurement metadata from
[`ai-dynamo/aisim-data/data/fpm/nvidia--GLM-5.2-NVFP4`](https://huggingface.co/datasets/ai-dynamo/aisim-data/tree/main/data/fpm/nvidia--GLM-5.2-NVFP4)
at the immutable revision pinned in `hf_dataset.json` (the files moved byte-identically from `nvidia/aisimulate-fpm-dataset@dbfcdd9eb62ea733efbc3f0cbd78277f2960061a`, `data/<rest>` to `data/fpm/<rest>`). Each entry records the
original artifact paths and SHA256 digests; table contents are fetched explicitly
and are not bundled in this directory. The dataset card declares Apache-2.0.
The repository's canonical `THIRD_PARTY_NOTICES.md` records this adaptation.

All 7 profiles are `development` and require `allow_unqualified=True`.
Publication does not establish serving qualification. KV variants and comparator
tables retain separate profiles and cache roots.

`packaged_system` selects the canonical hardware YAML from the installed
`aisim_core/systems` directory. The materializer copies it into the profile
root and includes its actual hash in cache identity and verification. This is a
current hardware reference, not an archived collection configuration. No
collection system YAML is published for these tables at the pinned dataset
revision. Native schema and exact-cell validation still apply after
materialization; profile names do not override table identity.
