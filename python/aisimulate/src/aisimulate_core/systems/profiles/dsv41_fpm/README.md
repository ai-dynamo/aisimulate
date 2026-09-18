<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1 FPM databases

This directory defines the loading contract for DeepSeek-V4.1 whole-forward
measurements on GB300 at TP2 and B300 at TP2 or TP4. Published tables
cover B300 TP2/TP4 and GB300 TP2 in both the `full` and
`decoder_bounded` execution profiles.

The six calibration tables and collection sidecars live in
[Hugging Face PR #12](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/discussions/12).
`hf_dataset.json` pins dataset commit
`8c070411dfeba2b9211a448c67c8651c76431d27` and every downloaded file's SHA-256.
The wheel contains this manifest and the hardware configuration; it does not
bundle FPM tables. Downloads are explicit, hash-verified and reusable offline.

Profiles are selected by system, TP width and execution scope, for example
`b300_sxm-tp2-full`, `b300_sxm-tp4-decoder_bounded` and `gb300-tp2-full`.
Each materialized root retains the original `data/<system>/sglang/<backend_version>/`
layout expected by the native reader. TP2 and TP4 keep distinct identities and
source-qualified runtime selectors. The paths in the coverage notes below refer
to this original layout, now stored externally and selected through the manifest.

## Published coverage

`full/data/b300_sxm/sglang/dev-c4ca651192e5/` contains 145 measured
calibration geometries at TP4. Each row is the first geometry-qualified
sample by fixed attempt index, with no selection based on latency. Of
1,450 planned calibration attempts, 1,275 qualified; 175 native batching
failures remain in the retained validation evidence. One separate whole-grid
warmup precedes collection; each published row contains one native timing sample.

Independent held-out validation covers 38 geometries and retains all 380
planned attempts: 345 qualified and 35 failed native geometry admission.
FPM MAPE is **5.4528%** (prefill **6.8378%**, decode **1.5747%**), computed
as the unweighted mean of geometry errors against each geometry's median
qualified held-out latency. Calibration and held-out run, request, and
geometry identities are disjoint. Geometries were predeclared; this is not
a blind unseen-geometry evaluation. These results apply only to the recorded
runtime and execution profile. The sidecar records immutable collection
identity, coverage, selection policy, and validation hashes.
The profile records SGLang PyNCCL **2.30.7**, observed in both native worker
startup logs and package inventories; an operator database may use a different
collective implementation and version.

## B300 TP2 full validation

`full/data/b300_sxm/sglang/dev-c4ca651192e5-tp267c0e788b4b2/` contains
145 measured calibration geometries at TP2. Of 1,450 fixed calibration attempts,
1,224 qualified and 226 failed native geometry admission. Rows use the first
geometry-qualified attempt by index, with no latency selection; a separate
whole-grid warmup precedes the ten formal attempts per geometry.

Independent heldout validation retains all 380 attempts: 322 qualified and
58 failed native geometry admission. Both the aggregate and explicit static
FPM APIs predict all 38 geometries, with conditional MAPE **0.7977%**
(prefill **0.8325%**, decode **0.7002%**). These are unweighted geometry errors
against medians of all qualified fixed heldout attempts. Calibration and heldout
run, request and geometry sets are disjoint; the grid was predeclared and no
heldout fitting was performed. Exact calibration self-queries are integrity
checks, not accuracy results.

The TP2 runtime uses the qualified startup bridge and request-capacity policy;
its selector includes the source-qualified runtime identity suffix. It is
separate from the TP4 base-image selector. Raw collection configs and the
package-reported version are preserved privately; only admitted output rows
receive the dev selector. Both actual TP2 worker startup logs and package
inventories confirm SGLang PyNCCL **2.30.7**, matching the existing profile YAML.
Collection and prediction hashes, actual reader/native identity, source-qualified
runtime and all failed-attempt counts are recorded in the compact sidecar.

## B300 TP4 bounded validation

`decoder_bounded/data/b300_sxm/sglang/dev-c4ca651192e5/` contains 145
calibration geometries. Of 1,450 planned calibration attempts,
1280 qualified and
170 native geometry failures remain in the retained evidence.
The same first-qualified selection and single-sample policy applies. Native
static self-queries validate all 145 rows; they are integrity checks, not accuracy.
The two actual bounded runs report SGLang PyNCCL **2.30.7**.

Independent bounded validation uses actual per-request native seed/chunk/context
witnesses with the existing **explicit static API**. It predicts
38/38 geometries and 346/380 planned attempts;
34 native geometry failures are retained. Its conditional MAPE is
**1.5628%** (prefill **1.4889%**,
decode **1.7697%**), using the same unweighted geometry-median definition
as the full profile. Calibration and heldout run, request and geometry sets are
disjoint; the geometry grid was predeclared and no heldout fitting was performed.

The **aggregate FPM-v1 API remains guarded for multiple prefill requests**:
55/145 calibration and 12/38 heldout grid geometries require per-request extend
lengths. Aggregate results separately predict 26/38 geometries and
260/380 attempts, with 86 qualified attempts rejected by that guard.
Its conditional MAPE is **1.6108%** over its 26 supported geometries;
this denominator is separate from explicit static coverage. Static coverage does
not establish general aggregate-telemetry support for bounded multi-prefill.

These historical static comparisons used the native engine with actual
per-request lengths. They do not establish aggregate-telemetry support for
bounded multi-prefill. Production construction uses
`RustForwardPassPerfModel.best_available` with
`estimation_mode="fpm_interpolation"`, `decoder_replay=True`, and the pinned
`systems_paths`; the aggregate admission guard remains enforced. The FPM
selector does not override arithmetic dtype. The validation above reports
static and aggregate coverage separately.

## B300 TP2 bounded validation

`decoder_bounded/data/b300_sxm/sglang/dev-c4ca651192e5-tp267c0e788b4b2/`
contains 145 calibration geometries: 1,248 of 1,450 fixed attempts qualified;
202 native geometry failures remain in the evidence. Rows retain the first
qualified attempt by index, without latency selection. The separate whole-grid
warmup and ten formal attempts per geometry match the full profile.

The explicit static native API uses actual per-request seed, chunk and context
witnesses. It predicts all 38 heldout geometries and 326 of 380 attempts, with
conditional MAPE **5.2327%**; 54 native geometry failures remain in the denominator.
The aggregate API retains its bounded multi-prefill guard: it predicts 26 of 38
geometries and 260 of 380 attempts, with conditional MAPE **6.7164%**. The other
66 qualified attempts are rejected by that guard. These two MAPE values use
separate supported geometry sets, each compared with medians of all qualified
fixed heldout observations; static coverage does not widen aggregate support.

The largest geometry error is **138.5442%** at decode batch 2, total past-KV
count 3,584: prediction **272.4125 ms**, heldout median **114.1979 ms**.
The prediction interpolates the first-qualified calibration samples at 3,072
(**433.5766 ms**, trial 0) and 4,096 (**111.2484 ms**, trial 0). Later faster
samples do not replace those observations. The measurements are retained without
attributing the difference to an unverified cause.

Calibration and heldout run, request and geometry sets are disjoint; the grid
was predeclared, and no heldout fitting was performed. Calibration self-queries
are integrity checks, not accuracy results. Both actual bounded TP2 worker logs
and their package inventories confirm SGLang PyNCCL **2.30.7**. The compact
sidecar records separate static/aggregate coverage, raw and validation hashes,
and the actual reader/native identity. Raw campaign config `backend_version`
remains null; the original package-reported version is retained, and only the
admitted output row selector is transformed.

## GB300 TP2 validation

`{full,decoder_bounded}/data/gb300/sglang/dev-800cc9adea5b/` each contains
145 calibration geometries measured on GB300 at TP2. All 1,450 fixed
calibration attempts and all 380 independent heldout attempts per profile
passed the original token, native geometry and lossless transport audits.
Each published row uses the first qualified attempt by index; later faster
observations do not replace it. Calibration and heldout run, request and
geometry sets are disjoint. The grid was predeclared, and heldout observations
were not used for fitting. Native self-queries are integrity checks.

| Profile | Explicit static FPM MAPE | Static coverage | Aggregate FPM MAPE | Aggregate coverage |
| --- | ---: | --- | ---: | --- |
| Full | 0.9063% | 38/38 geometries, 380/380 attempts | 0.9063% | 38/38 geometries, 380/380 attempts |
| Decoder bounded | 5.1727% | 38/38 geometries, 380/380 attempts | 1.5201% | 26/38 geometries, 260/380 attempts |

These are unweighted means of geometry errors against the median of all ten
qualified heldout observations. The bounded aggregate API rejects the other
12 multi-prefill geometries (120 qualified attempts) because it lacks their
per-request extend lengths; static coverage does not widen aggregate support.
The largest bounded static error is 142.2716% at prefill batch 2, total extend
96 and total past-KV 3,072: prediction 428.0247 ms versus heldout median
176.6714 ms. The original first-qualified samples and this outlier are retained.

All four workers exited with code 1 during redundant endpoint unregister,
after the final campaign acknowledgement. Client/frontend exit codes were 0.
Complete control sequences, original source/runtime identities, native timing
transport and request/geometry audits passed; this qualifies the measured
samples but **does not establish clean worker shutdown**. The actual native
handler's finalization after delivering its ACK was not directly proved.
Original logs retain startup warnings, FlashInfer tuning fallbacks and shutdown
discovery warnings. The sidecars record these limits and evidence hashes.

Both profiles record SGLang PyNCCL **2.30.7**, observed in all four worker logs.
The selector identifies the ARM collection image; its source-qualified cohort
adapter and unchanged native timing producer remain in provenance. Raw package
version `0.0.0.dev0` is retained as evidence, not used as the database selector.
No measurements from B300 or GB200 are substituted for GB300.

## Loading a published table

Use `estimation_mode="fpm_interpolation"`, `backend="sglang"`, the measured system, and the
exact `backend_version` recorded in the publication metadata. Versions use
`dev-` followed by the first 12 characters of the actual collection image's
SHA-256 manifest digest. B300 TP2 bridge-qualified data additionally includes
`-tp2` and the first 12 characters of its canonical runtime identity; use the
complete recorded selector. The full digest, captured runtime source identities,
and original package-reported version remain collection provenance. An image
hash identifies a build; it is not a source Git commit.

Materialize a pinned profile, then pass the returned `systems_path` to prediction:

```python
import json
from pathlib import Path

import aisimulate_core
from aisimulate_core.sdk.fpm_dataset import materialize_fpm_profile
from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel

system = "b300_sxm"
tp = 2
decoder_replay = False
profile = "decoder_bounded" if decoder_replay else "full"
manifest = (
    Path(aisimulate_core.__file__).parent
    / "systems" / "profiles" / "dsv41_fpm" / "hf_dataset.json"
)
key = f"{system}-tp{tp}-{profile}"
identity = json.loads(manifest.read_text())["profiles"][key]["identity"]
systems_path = str(materialize_fpm_profile(manifest, key))
config = ForwardPassPerfModelConfig(
    model=identity["model_path"], system=system, backend="sglang",
    backend_version=identity["backend_version"], worker_type="aggregated",
    systems_paths=(systems_path,), tp=tp, moe_tp_size=tp, moe_ep_size=1,
    decoder_replay=decoder_replay, estimation_mode="fpm_interpolation",
    fpm_fmha_quant_mode="fp8", enable_shared_layer=False, strict_provenance=True,
)
model = RustForwardPassPerfModel.best_available(config)
```

For offline reuse, set `local_files_only=True`; an absent or corrupted cache
fails instead of downloading or silently changing datasets. The fixed commit
works before and after the HF PR is merged; no floating `main` revision is used.

Pass the pinned root through `systems_paths` and preserve `decoder_replay` in
the canonical prediction configuration.
Set attention TP and MoE TP to the measured width; PP, attention DP, MoE EP,
and CP are 1. Match the table's precision selectors, including
`fpm_fmha_dtype="fp8"` for native engine/replay JSON. This selects the FPM
attention cell without overriding the checkpoint's analytical arithmetic.
See the [execution contract](../../../../../docs/fpm/deepseek-v41.md) for the
corresponding Python model option.

Coverage is established only by published measurements and their validation.
A missing topology, runtime, or geometry remains a coverage error. Decoder
profiles retain separate execution identities. Calibration self-queries check
data integrity; accuracy is measured on separate held-out observations.

## Upstream attribution

The execution identity and geometry are modified AISimulate adaptations of
SGLang's serving contracts at
[`1aa0e962b206102b7c439a4a0c4981cfec6e87bc`](https://github.com/sgl-project/sglang/tree/1aa0e962b206102b7c439a4a0c4981cfec6e87bc),
including `python/sglang/srt/models/deepseek_v4.py`, `deepseek_v2.py`,
`python/sglang/srt/layers/engram.py`, and the attention, compressor and memory
pool paths listed in `THIRD_PARTY_NOTICES.md`. Copyright 2023-2024 SGLang Team
and SGLang contributors; Apache-2.0.

The checkpoint configuration and architecture derive from `config.json`,
`inference/model.py`, and `DeepSeek_V41_Tech_Report.pdf` in
[`deepseek-ai/DeepSeek-V4.1-Flash@fb2764a5cf321eaa5070ca8f9e892818f477c16d`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/fb2764a5cf321eaa5070ca8f9e892818f477c16d).
Copyright (c) 2023 DeepSeek; MIT. The latencies are new AISimulate measurements;
the tables contain adapted execution identity and geometry, with no upstream
model execution code. The distribution includes the applicable licenses and
canonical third-party notices. The SGLang reference revision identifies the
serving-contract sources, not the entire captured runtime image.
