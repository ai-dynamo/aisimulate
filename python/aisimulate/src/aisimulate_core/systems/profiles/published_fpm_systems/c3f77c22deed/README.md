<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reference system configurations for published FPM tables

These files are byte-identical copies of
[`python/aisimulate/src/aisimulate_core/systems/` at AISimulate commit
`c3f77c22deed711e6f7fb3f1ea730bdf3d7fee53`](https://github.com/ai-dynamo/aisimulate/tree/c3f77c22deed711e6f7fb3f1ea730bdf3d7fee53/python/aisimulate/src/aisimulate_core/systems).
Their original copyright and Apache-2.0 headers are retained.

The published MiniMax-M2.7, DeepSeek-V4-Pro, DeepSeek-V4-Flash-0731,
GLM-5.2-NVFP4, Kimi-K3 and Inkling-NVFP4 tables have no archived collection
system YAML at the manifests' pinned dataset revision. These copies provide
reference hardware and query policy for explicitly opted-in reproduction.
They do not establish the hardware configuration used during collection or
grant serving qualification.

The manifests pin every file by SHA256. Keep this snapshot unchanged when the
default system configurations evolve; a future manifest update must use a new
snapshot with its own source revision and qualification evidence.
