<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FPM v1 aggregate admission: frozen-input audit

The stricter Decoder ON guard rejects multiple prefill requests with fresh
tokens because FPM v1 prompt-length variance cannot identify their individual
current extends and cached prefixes. This audit applies the old and new
predicates to **69,168 preserved native inputs** from the six GB300 union
verification populations. It does not rerun prediction or recalculate accuracy.

| Population | Inputs | Old rejections | New rejections | Newly rejected |
|---|---:|---:|---:|---:|
| OFF core | 15,756 | 0 | 0 | 0 |
| OFF field | 3,503 | 0 | 0 | 0 |
| OFF service | 3,509 | 0 | 0 | 0 |
| ON core | 39,393 | 100 | 396 | 296 |
| ON field | 3,503 | 0 | 37 | 37 |
| ON service | 3,504 | 0 | 36 | 36 |
| Total | 69,168 | 100 | 469 | 369 |

The 369 newly rejected inputs were predicted in the historical reports. All
have batch 2, zero prefix and zero native variance, with 384, 768 or 1536 total
new tokens. This conservative admission change does **not** assert that these
physical executions had heterogeneous extends. The generic FPM v1 input does
not establish their per-request geometry. OFF, single-prefill, decode-only and
explicit homogeneous static queries retain their separate supported paths.

The original observations, predictions, errors, confidence intervals and
coverage denominators remain unchanged. Historical claims that the older guard
excludes only 100 intervals do not describe this stricter admission rule.

## Reproduce

From the repository root, using Python 3.11 or later and a fresh output path:

```bash
python3 data/experimental/deepseek-v41/fpm-aggregate-admission-v1/audit.py \
  --output /tmp/dsv41-fpm-admission-result.json
cmp data/experimental/deepseek-v41/fpm-aggregate-admission-v1/result.json \
  /tmp/dsv41-fpm-admission-result.json
```

The result SHA-256 is
`78ea636b031157498df8bc5eb166b6c802dfc6169d3017637a44ca3ceba3d321`.
The script refuses to overwrite an existing output. It pins each compressed
input's SHA-256 before decompression and records both compressed and decoded
hashes in the result. The snapshot commit identifies those input files; the
separate predicate-reference commits identify the reviewed admission rules,
not a new source identity for the historical predictions.

Each `(physical_run_id, counter_id)` must be unique, the original six population
counts must match, and native/predictor input bridges must produce identical
predicate outcomes. The result contains population and geometry counts, with
no copy of the 69,168 rows and no measurement-timing values. This is validation
evidence about support scope, not a new product or Collector schema.
