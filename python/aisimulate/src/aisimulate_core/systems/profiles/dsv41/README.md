<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1 operator databases

These systems roots contain the measured operator tables used by the
DeepSeek-V4.1 SILICON predictor. They are included in the Python distribution.
End-to-end ground truth, raw experiment logs, and prediction reports remain in
external verification archives.

The database versions identify immutable collection images: `dev-800cc9adea5b`
for GB300 and `dev-c4ca651192e5` for B300. Each suffix uses the first 12 characters
of that image's SHA-256 digest. Complete image and source identities are recorded
below and in the collection sidecars. Both systems contain TP4 data. B300 TP2
full calibration is also available in the separate `tp2/full/` systems root described below. TP2 bounded is available in `tp2/decoder_bounded/`.

| System | Systems root | `decoder_replay` | V4.1 module rows | GEMM / MoE / NCCL rows |
| --- | --- | --- | ---: | ---: |
| GB300 | `full/` | `false` | 848 | 32 / 16 / 32 |
| GB300 | `decoder_bounded/` | `true` | 948 | 32 / 16 / 32 |
| B300 | `full/` | `false` | 985 | 46 / 23 / 46 |
| B300 | `decoder_bounded/` | `true` | 1029 | 46 / 23 / 46 |

Each root contains `gb300.yaml`, `b300_sxm.yaml` and the standard
`data/<system>/<family>/<backend>/<version>/` layout, with a collection sidecar
and SHA-256 for each Parquet file. Latencies are in milliseconds. These are
physical operator points, not independent workload counts.

## Selecting a database

Choose the matching system and image version below. Use `backend="sglang"`,
`database_mode="SILICON"`, `forward_model="op_level"`, `strict_provenance=True`,
and `enable_shared_layer=False`. Set TP and MoE TP to 4, and PP, MoE EP and
attention DP to 1. Set `systems_path` to the matching root:

```python
from pathlib import Path

import aisimulate_core

decoder_replay = False
system_name = "b300_sxm"  # or "gb300"
backend_version = {
    "gb300": "dev-800cc9adea5b",
    "b300_sxm": "dev-c4ca651192e5",
}[system_name]
profile = "decoder_bounded" if decoder_replay else "full"
systems_path = str(
    Path(aisimulate_core.__file__).parent / "systems" / "profiles" / "dsv41" / profile
)
```

Pass `system_name`, `backend_version`, `systems_path` and `decoder_replay` into the prediction configuration.
The flag does not select a database automatically. The profiles contain
overlapping physical keys with different measured timings and must stay in
separate roots; a mismatched root is not guaranteed to fail every lookup.
They do not replace the general hardware databases or their default runtime versions.

Exact points return measured timings. The reader interpolates within an
existing curve and uses an existing measured boundary with SOL ratios for
extrapolation. A missing curve is a typed coverage error. Existing empirical
operators can still contribute to a whole-model total.

## B300 TP2 full operator database

Use `systems_path` ending in `systems/profiles/dsv41/tp2/full`,
`system_name="b300_sxm"`, `backend="sglang"`,
`backend_version="dev-c4ca651192e5-tp267c0e788b4b2"`, TP and MoE TP 2,
PP/MoE EP/attention DP 1, and `decoder_replay=False`. Retain the strict
SILICON/OP settings described above. This separate root preserves the TP4
and GB300 tables, including their NCCL observations.

The TP2 database contains 985 V4.1 module, 46 GEMM, 23 MoE and 46 NCCL rows.
All 145 frozen workloads completed on both ranks with one warmup and five
measured iterations; baselines used two warmups and five measured iterations.
All 1,100 aggregated records passed strict native exact lookup. The actual
runtime used request cap 4 and physical KV capacity 5120 on B300 driver
`610.57.04`. The version suffix binds the immutable image and qualified
startup-loader source/policy; it does not reuse the TP4 runtime identity.

[TP2 full provenance](b300-tp2-full-provenance.json) records the source identity,
raw archive/admission hashes and every database file hash. Raw observations
remain external. Complete calibration and exact lookup establish table
coverage. Independent full native-forward heldout validation covers 38 geometries
and 322/380 fixed attempts, with SILICON MAPE **5.0551%** (prefill **2.1298%**,
decode **13.2460%**). The 58 native batch-split failures remain in the denominator.
The bounded root below has separate measurements.

## B300 TP2 bounded operator database

Use `systems_path` ending in `systems/profiles/dsv41/tp2/decoder_bounded`
with the same TP2 backend version and parallelism above, and
`decoder_replay=True`. This profile contains 1,029 V4.1 module, 46 GEMM,
23 MoE and 46 NCCL records from 154 frozen workloads, including the nine
bounded supplemental geometries. All 1,144 records passed strict native
exact lookup. Warmup, measured repetitions and physical-pool limits match
the TP2 full protocol; the actual driver was `610.57.04`.

[TP2 bounded provenance](b300-tp2-bounded-provenance.json) identifies its
separate raw archive, admission and table hashes. These measured tables are
separate from the full profile. Independent bounded native-forward validation
covers 38 geometries and 326/380 fixed attempts, with SILICON MAPE **6.3582%**
(prefill **2.1822%**, decode **18.0512%**). The 54 native batch-split failures
remain in the denominator.

Both TP2 profiles use predeclared heldout geometries disjoint from FPM calibration.
MAPE is the unweighted mean geometry error against the median of every qualified
fixed heldout attempt. No observed latency is a predictor input; no fitting or
failed-attempt removal occurs. These are native-forward results for the recorded
source-qualified runtime, not a blind benchmark or HTTP serving validation.
Actual wheel readers reproduce all 2,244 TP2 exact lookups and every heldout
prediction from both profiles.

## B300 TP4 measurement scope and provenance

The September 17, 2026 collection used four B300 SXM6 GPUs on driver `610.57.04`,
the AMD64 image
`sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5`,
and the checkpoint revision listed below. Full and bounded-decoder collection
covered 145 and 154 frozen workloads respectively, including nine predefined
bounded-profile supplemental workloads. Each workload had one warmup and five
measured iterations on all four ranks. Baselines had two warmups and five
measured iterations. Aggregation takes the maximum across TP ranks per physical
invocation, then the median within the same profile and physical key. The raw
package version `0.0.0.dev0` remains provenance; loading uses the image dev selector.

Both profiles use eager text execution, a physically observed 5120-token KV
capacity, HBM-resident Engram, unfused shared experts and separate NCCL 2.29.7
collectives. CUDA graphs and fused all-reduce are disabled. Each complete
calibration passed source, workload, physical-pool and exact native-reader
checks. These checks establish database coverage, not heldout prediction accuracy.
The TP4 tables contain no TP2 observations. The separately admitted TP2 full
root above uses its own source-qualified startup runtime and raw measurements.

[B300 TP4 provenance](b300-tp4-provenance.json) records table hashes, runtime and
checkpoint identities, collection settings, and hashes binding the external
raw archive and admission receipts. The tables and sidecars are unchanged
copies of the admitted outputs. The build's Git revision is unknown; the
serving-contract reference below is not asserted to identify the entire image.

## GB300 measurement scope and provenance

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

The database selector uses the image hash, while `runtime.version` in the
unchanged historical sidecars records the package-reported `0.0.0.dev0`.
That field is collection evidence, not the version to request when loading
these databases. Do not use the mutable `dev-dsv41` image tag to identify them.

Collection used eager text autoregressive execution, HBM-resident Engram,
unfused shared experts, and separate Torch NCCL 2.29.7 collectives. Local module
timings exclude collectives; the MoE baseline uses seeded uniform expert
routing. CUDA graphs and fused all-reduce were disabled. Coverage is limited
to the measured geometry and runtime; these tables do not qualify alternative
topologies, offload, output equivalence, or general serving accuracy.

All 18 GB300 database file contents are byte-identical copies from AISimulate commit
[`24faa2e263c75c137c091b8e80b7c2d36740b864`](https://github.com/ai-dynamo/aisimulate/tree/24faa2e263c75c137c091b8e80b7c2d36740b864/data/experimental/deepseek-v41/gb300-silicon/indexer-identity-v2/prefix-refinement),
under `data/experimental/deepseek-v41/gb300-silicon/indexer-identity-v2/`
`prefix-refinement/{full,decoder_bounded}/systems/`. Only the SGLang version
directory names changed from `0.0.0.dev0` to `dev-800cc9adea5b`; the NCCL
`2.29.7` paths and all measurements and sidecars are unchanged. The archive preserves the
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
