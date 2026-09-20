<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Per-request FPM producer patches

Stock Dynamo and SGLang emit aggregate-only `ForwardPassMetrics`
(`num_*_requests`, `sum_*_tokens`, variances). The per-request feature presets
of the learned forward-pass model (`sglang18`, `hisim`) need two additional,
optional, aligned lists in `scheduled_requests`:

| Field | Meaning |
| --- | --- |
| `extend_lengths[i]` | tokens computed for scheduled request `i` in this iteration |
| `past_kv_lengths[i]` | KV tokens already present for request `i` before the iteration |

One entry per scheduled request, prefill requests first, sorted by
`past_kv_lengths` descending. The FPM `version` stays 1; consumers that only
read aggregates are unaffected. **Neither patch is merged upstream yet.**
Streams from unpatched producers train with `--features v1`.

| Patch | Applies to | Files |
| --- | --- | --- |
| `dynamo-per-request-fpm-fields.patch` | ai-dynamo/dynamo `bcec7eae7117` (2026-09-20) | `components/src/dynamo/common/forward_pass_metrics.py`, `components/src/dynamo/vllm/instrumented_scheduler.py` |
| `sglang-per-request-fpm-fields.patch` | sgl-project/sglang `9f3d2759407f` (2026-09-20) | `python/sglang/srt/observability/forward_pass_metrics.py`, `python/sglang/srt/managers/scheduler_components/metrics_reporter.py` |

```bash
cd dynamo && git apply /path/to/dynamo-per-request-fpm-fields.patch
cd sglang && git apply /path/to/sglang-per-request-fpm-fields.patch
```

In a container image the same files can be overlaid on the installed package
(`site-packages/dynamo/...`, `site-packages/sglang/...`) without rebuilding.

Notes:

- SGLang takes the values from the schedule-time `batch.extend_lens` /
  `batch.prefix_lens` (the per-request attributes are already reset when the
  metrics are emitted). Mixed batches append decode requests as `(1, seqlen)`.
- SGLang FPM must be collected with `--disable-overlap-schedule` regardless of
  this patch; see the "Collect the FPM stream" section of
  [learned-forward-pass-model.md](../learned-forward-pass-model.md).
