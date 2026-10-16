<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Collector data format and identity

## Identity layers

Keep three identities distinct:

1. **Recipe:** which model/YAML profile requested a measurement.
2. **Invocation:** everything that changes the executed kernel or setup,
   including checkpoint quantization and runtime selection.
3. **Physical key:** the consumer-visible operation dimensions stored in a table.

Deduplicate only when both invocation and physical key are equivalent. Two model
aliases colliding on a table key do not prove they run the same kernel. Correlated
structural tuples (heads, KV heads, head dimension, window, TP) are expanded as
model shapes; unrelated axis lists must not manufacture nonexistent models.
Shared batch/sequence sweeps are applied afterward.

Population combines base and model YAML, resolves artifact precision policy,
applies universal mathematical/hardware constraints, derives invocation keys,
and deduplicates stable equivalents. Registry maturity and hang-only exclusions
then determine runnable cases. Ordinary backend errors and OOMs are observed
outcomes, not reasons to remove intended coverage before collection.

## Table layout and runtime pins

```text
python/aisimulate/src/aisimulate_core/systems/data/
  <system>/<family>/<backend>/<version>/
    <table>_perf.parquet
    collection_meta.yaml
    reuse.yaml                  # only when reuse is declared
```

`collector/<backend>/registry.py` names runnable operations; `PerfFile` names
tables; `collector/op_backend_catalog.yaml` maps tables to management families.
`collector/framework_manifest.yaml` resolves each producing framework/family to
its default or overridden version/image. A registry operation with no family
or runtime is an error. The consumer backend directory is not a substitute for
recording the producing runtime, especially for special/WideEP images.

A Parquet file is a table with an operation-specific identity, not just a numeric
latency grid. Preserve precision, phase, architecture/kernel, parallel dimensions,
and units. Collector finalization owns staging-to-Parquet conversion; use its
validation and [review tooling](reviewing-data.md). Whole-forward pairs use the
separate [FPM self-service format](../fpm-self-service/implementation.md).

## Head-axis keying

Attention/module rows persist rank-local `num_heads`. Native model identity is
an additional key where local dimensions alone do not determine computation:

| Family | Identity before numeric interpolation | Native geometry |
| --- | --- | --- |
| GQA attention | Local Q/KV heads, dimensions, window | Numeric key is computation-complete. |
| MLA kernels | Local heads with the supported fixed MLA geometry | Do not add an unnecessary native bucket. |
| MLA modules | Native model heads, then local heads | Explicit model-name pin, not a synthetic `num_heads * tp_size`. |
| DSV4 modules | Native heads, then local heads | Genuine TP sweep: `num_heads * tp_size`. |
| DSA / MiniMax MSA modules | Architecture, then local heads | Guardrails require one native geometry per architecture/table. |
| DSV4 sparse calibration | Native geometry by its own contract | Do not reinterpret its key as the module's local head axis. |

MLA module head sweeps can use `tp_size=1` as provenance while changing local
heads; their native geometry therefore comes from an explicit model pin.
Unpinned models fail. Genuine TP rows must agree with their native pin. The
module resolver chooses exact native, then the sole bucket, then nearest not
larger, then smallest, as implemented by the native loader.

MLA kernel reuse assumes the supported common latent/RoPE dimensions. A model
changing those dimensions needs a reviewed key/geometry contract. The existing
DeepSeek builder's Kimi-K2.5 kernel convention uses its fixed head reference;
module paths and Kimi-K3 use true geometry. Do not silently change that
prediction convention while reorganizing table keys.

DSV4 Flash/Pro share an architecture string but have different native module
geometry, so both native and local axes are required. See
[DeepSeek-V4](deepseek-v4.md) for its exact-match calibration rule.

## Provenance and population coverage

`collection_meta.yaml` records producing runtime, collector source/content hash,
case-plan hash, collection time, rows, and completion status per table. A single
event uses schema 1. Schema 2 retains `collections` histories for tables assembled
from several events; each event needs its own attestation, with optional explicit
runtime override and source-campaign subset information.

The case-plan hash attests the attempted set, not the universe of possible shapes.
A completed filtered campaign is distinguishable from a full plan. Module failure
or no produced rows marks partial collection; classified individual failures can
remain valid observed outcomes. Never infer complete model coverage from a
`complete` table status. A table without corresponding provenance is a coverage
failure. Legacy provenance remains explicitly legacy, not newly measured.

Changed-operation manifests identify which family/system evidence a collector or
runtime change requires. CI checks that evidence separately from per-directory
execution status. See [CI](../../ci/README.md) for the repository gates.

## Reuse and source priority

With shared-layer sourcing enabled, primary measured rows take precedence and
approved donors fill missing physical shapes. The source chain distinguishes:

- Explicit same-backend `reuse.yaml` donors, including reviewed forward reuse.
- Eligible earlier same-backend versions; a newer version is not implicitly a donor.
- Cross-backend rows only through the kernel-identity reuse manifest and eligible
  shared/shared-fallback source labels.

A declaration needs table, donor version, reason, and review ownership; a
version directory can contain both primary rows and a declaration. Reuse does
not relabel the donor's runtime as a new measurement. Some operation loaders
have pinned overwrite semantics; preserve their source order and parity tests
instead of assuming every raw loader uses the same merge algorithm.

The authoritative implementations are
[`collector`](../../../python/aisimulate/collector/),
[`perf_database.py`](../../../python/aisimulate/src/aisimulate_core/sdk/perf_database.py),
and the [native table loaders](../../../crates/core/src/perfmodel/perf_database/).
