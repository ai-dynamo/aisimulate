<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Context-parallel DSA prefill

The native [DSA module](../../../crates/core/src/perfmodel/operators/dsa.rs)
composes context-parallel prefill from a measured per-card module, sparse
indexer/top-k corrections, and attention communication. This describes the
implemented mechanism; data coverage remains specific to architecture,
head geometry, system, runtime version, precision, and batch.

## Work and timing composition

A round-robin CP split gives each card query positions distributed across the
full causal context. KV is gathered so a local query still sees its actual full
past. For total fresh length `S`, cached prefix `P`, and CP width `C`, the busiest
rank has `s_local = ceil(S/C)` fresh tokens. The native implementation uses at
least one token for its nonempty per-card lookup.

For a layer that executes its indexer:

```text
DSA = module(s_local, P)
    + mqa(S, P)/C - mqa(s_local, P)
    + topk_last(S, P)/C - topk_flat(s_local, P)
    + AG_index_keys + AG_compressed_kv
```

Sparse corrections are queried at the actual batch slice, without an extra
assumed linear batch multiplier. Both full and local lookups use the real cached
prefix. `full/C` is not interchangeable with multiplying the local top-k cost by
`C`: top-k does not have quadratic context scaling. For skip-indexer layers,
the matching skip-module base and attention all-gathers remain, while the mqa
and top-k deltas are omitted.

Projections and sparse FMHA keep their per-card module contribution. The
sparse attention is capped by the selected-key geometry; indexer scoring is
still over its full causal domain. Missing required sparse mqa/flat/top-last
tables raises a data error rather than silently leaving the uncorrected base.

## Communication and MoE

Attention all-gathers carry the current chunk's index keys and compressed KV,
using BF16 payload geometry, actual batch, and CP group size. Cached-prefix KV
is already replicated and is not sent again. Hidden-state all-gather and
reduce-scatter belong to MoE dispatch, outside this attention correction.

For a TP+sequence-parallel MoE group, hidden states are gathered to the full
token set, expert compute runs under its MoE TP layout, then one reduce-scatter
both reduces TP partials and restores per-card token shards. Do not add another
TP all-reduce. An EP path instead uses its dispatch/combine contract. The model
assembles these operations with its explicit serial/overlap structure; an
unrelated decode or post-processing collective is not prefill layer work.

## Precision and limits

Choose the module row by the executed attention arithmetic and projection
precision, not by KV storage dtype alone. A runtime storing FP8 KV can dequantize
and run a BF16 attention kernel. Its module, sparse calibration, and collective
rows must refer to the intended execution identity.

The CP sparse deltas are latency-only; `SOL_FULL` decomposition for CP DSA is
explicitly unsupported. CP prefill and DCP decode use different geometry and
resource contracts. Check [configuration](../configuration.md) and the
[support matrix](../support-matrix.md) before extrapolating this mechanism to
another family or claiming full Replay support.
