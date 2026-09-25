<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

These two text files were written originally for the GLM-5.3-Flash sampling
campaign. They are not copied or adapted from an upstream model, dataset, book,
or paper. Both are distributed under this repository's Apache-2.0 license.

They are distinct tokenizer inputs for calibration and independent holdout.
The native producers rotate/repeat the selected text's token stream to reach
requested lengths. These small corpora exercise varied prose, arithmetic,
procedural explanations and Chinese text; they are not a claim of production
traffic representativeness or an independently measured routing distribution.
Their exact byte hashes are recorded by `glm53flash_sampling.py`.
