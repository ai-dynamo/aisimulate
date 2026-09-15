<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1 GB300 preview operator databases

These two systems roots contain the measured operator tables used by the
DeepSeek-V4.1 SILICON predictor. They are included in the Python distribution.
End-to-end ground truth, raw experiment logs, and prediction reports remain in
the immutable archive linked below.

These measurements are specific to a preview build, not a released SGLang
version. As of September 14, 2026, the
[official V4.1 guide](https://github.com/sgl-project/sglang/blob/07e1918924b11223c185544507669f9eb02c9b65/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx)
states that V4.1 support has not shipped in a release and directs NVIDIA users
to `dev-dsv41`. A release-specific database requires a new collection on a
supporting release; these measured files must not be relabeled as that release.

| Systems root | `decoder_replay` | V4.1 module rows | GEMM / MoE / NCCL rows |
| --- | --- | ---: | ---: |
| `full/` | `false` | 848 | 32 / 16 / 32 |
| `decoder_bounded/` | `true` | 948 | 32 / 16 / 32 |

Each root contains `gb300.yaml` and the standard
`data/gb300/<family>/<backend>/<version>/` layout, with a collection sidecar
and SHA-256 for each Parquet file. Latencies are in milliseconds. These are
physical operator points, not independent workload counts.

## Selecting a database

Use `system_name="gb300"`, `backend="sglang"`, `backend_version="0.0.0.dev0"`,
`database_mode="SILICON"`, `forward_model="op_level"`, `strict_provenance=True`,
and `enable_shared_layer=False`. Set TP and MoE TP to 4, and PP, MoE EP and
attention DP to 1. Set `systems_path` to the matching root:

```python
from pathlib import Path

import aiconfigurator_core

decoder_replay = False
profile = "decoder_bounded" if decoder_replay else "full"
systems_path = str(
    Path(aiconfigurator_core.__file__).parent / "systems" / "profiles" / "dsv41" / profile
)
```

Pass both `systems_path` and `decoder_replay` into the prediction configuration.
The flag does not select a database automatically. The profiles contain
overlapping physical keys with different measured timings and must stay in
separate roots; a mismatched root is not guaranteed to fail every lookup.
They do not replace the general GB300 database or its default runtime versions.

Exact points return measured timings. The reader interpolates within an
existing curve and uses an existing measured boundary with SOL ratios for
extrapolation. A missing curve is a typed coverage error. Existing empirical
operators can still contribute to a whole-model total.

## Measurement scope and provenance

The September 2026 collection used four GB300 GPUs, the SGLang ARM64 image
`sha256:800cc9adea5be1e18f48185451220c4bc487c545b7095c720d2ccc9ba9bb3b5d`,
and `deepseek-ai/DeepSeek-V4.1-Flash` at checkpoint revision
`fb2764a5cf321eaa5070ca8f9e892818f477c16d`. The reported package version
`0.0.0.dev0` is a development placeholder, not a release identity. The actual
image's OCI revision, SGLang build commit and source-overlay commit are all
`unknown`; its version label is `local/sglang:dev`. The immutable image digest
above and the captured Python source-manifest digest
`d50217d8f78e4bd173774c36713650bbf44b058c9575ac8babba208a5c5173a2`
identify these measurements. The current mutable preview tag may point to a
different image, and the reference source commit below does not identify the
entire captured runtime.

Collection used eager text autoregressive execution, HBM-resident Engram,
unfused shared experts, and separate Torch NCCL 2.29.7 collectives. Local module
timings exclude collectives; the MoE baseline uses seeded uniform expert
routing. CUDA graphs and fused all-reduce were disabled. Coverage is limited
to the measured geometry and runtime; these tables do not qualify alternative
topologies, offload, output equivalence, or general serving accuracy.

All 18 database files are byte-identical copies from AISimulate commit
[`24faa2e263c75c137c091b8e80b7c2d36740b864`](https://github.com/ai-dynamo/aisimulate/tree/24faa2e263c75c137c091b8e80b7c2d36740b864/data/experimental/deepseek-v41/gb300-silicon/indexer-identity-v2/prefix-refinement),
under `data/experimental/deepseek-v41/gb300-silicon/indexer-identity-v2/`
`prefix-refinement/{full,decoder_bounded}/systems/`. The archive preserves the
[sampling and collection records](https://github.com/ai-dynamo/aisimulate/blob/24faa2e263c75c137c091b8e80b7c2d36740b864/data/experimental/deepseek-v41/gb300-silicon/prefix-refinement-v1/README.md)
and [original prediction input bindings](https://github.com/ai-dynamo/aisimulate/blob/24faa2e263c75c137c091b8e80b7c2d36740b864/data/experimental/deepseek-v41/prediction-refresh-20260914/silicon/original-input-bindings.json).

The module geometry includes a documented metadata correction: indexer heads
changed from 8 to 32 after auditing the captured runtime source. Latency bits
and all other columns were preserved; this was not a new measurement of loaded
module dimensions. The
[derivation and source audit](https://github.com/ai-dynamo/aisimulate/blob/24faa2e263c75c137c091b8e80b7c2d36740b864/data/experimental/deepseek-v41/gb300-silicon/indexer-identity-v2/README.md)
bind original and derived table hashes. Collection sidecars retain the original
producer identities, rather than attributing old measurements to the current
collector.

The geometry follows SGLang's modified serving contracts at
[`1aa0e962b206102b7c439a4a0c4981cfec6e87bc`](https://github.com/sgl-project/sglang/tree/1aa0e962b206102b7c439a4a0c4981cfec6e87bc),
including `python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py` and the
model, Engram and compressor paths identified in `THIRD_PARTY_NOTICES.md`.
Copyright 2023-2024 SGLang Team and SGLang contributors, Apache-2.0. These
tables contain AISimulate measurements and adapted geometry, not upstream
model execution code. The distribution includes the applicable license and
canonical third-party notices.
