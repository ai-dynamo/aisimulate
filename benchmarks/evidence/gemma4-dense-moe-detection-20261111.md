<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dense Gemma 4 MoE classification evidence

Measured on 2026-11-13 on macOS arm64 (Python 3.13.18) against `origin/main`
`f85518b36` (the AISim rename) with the prediction-only Python and Rust sources
of this branch. No GPU was run, and no performance data was added, re-collected,
or re-fitted.

The code and test changes were measured at `29bb28580`; later commits add only
this write-up and docstrings, with no behavior change.

## Scope

This evidence covers parallel-shape selection, the generator producer
enumeration, the CLI and SDK consumers of the classification, and the public
recommendation path for the two Gemma 4 checkpoint kinds. It does not measure
serving accuracy, latency, or throughput. The recommendation probe supplies
constant metrics through offline runner and performance-model doubles, so
candidate scores are placeholders; only the selected shape set and the
feasible/infeasible status are meaningful.

## Inputs and provenance

`check_is_moe()` classifies `GEMMA4MIX` from the parsed `num_experts`/`topk`
pair. The bundled checkpoints pin that pair:

| Bundled checkpoint | Git blob (`git hash-object`) | `text_config.num_experts` | `text_config.top_k_experts` | Classification |
| --- | --- | ---: | ---: | --- |
| `nvidia--Gemma-4-31B-IT-NVFP4_config.json` | `b8a5581721139016753fb549f1d7083393202348` | `null` | `null` | dense |
| `nvidia--Gemma-4-26B-A4B-NVFP4_config.json` | `1247fb379f4ac91070ebdc083389c6d005b72ccb` | `128` | `8` | routed |
| `google--gemma-4-26B-A4B_config.json` | `29512a0d8108fc06ea4cec3cf2be250cfe0b960a` | `128` | `8` | routed |

All three declare `architectures=["Gemma4ForConditionalGeneration"]` and
`model_type="gemma4"`, so the architecture alone cannot separate dense from
routed. The configs also carry `text_config.enable_moe_block` (`false` dense,
`true` routed), but no Python or Rust source reads that key; it appears only in
a `gemma4.py` docstring. The parsed `num_experts`/`top_k_experts` fields that
`check_is_moe()` uses are therefore the only dense/routed signal the model build
consumes.

The blob hashes are identical on `origin/main` (`f85518b36`); this branch does
not modify the bundled configs.

## Reproduction

Shape enumeration, from the repository root with the built native extension:

```python
from aisim.sweeper import model_hw

for budget in (1, 2, 4, 8):
    model_hw.parallel_configs_for(
        "nvidia/Gemma-4-31B-IT-NVFP4",
        "gb300",
        gpu_budget=budget,
        deployment_mode="agg",
        backend="vllm",
        max_seq_len=2048 + 128,
    )
```

The "before" column was produced by the same snippet in a clean
`git worktree add --detach /tmp/aisim-origin-main origin/main` checkout with
`PYTHONPATH=/tmp/aisim-origin-main/python/aisim/src`, not by overriding
`check_is_moe` in this tree.

The public recommendation path is covered by
`test_dense_gemma4_public_recommendation_is_expert_parallel_free` in
`python/aisim/tests/unit/test_modern_recommend_context.py`. That test fails
on `origin/main` with the `ValueError` below and passes on this branch.

## Shape enumeration

`gb300`, `vllm`, `deployment_mode="agg"`, `max_seq_len=2176`:

| GPU budget | `origin/main` `f85518b36` | This branch |
| ---: | --- | --- |
| 1 | 1 shape: `tp1/dp1/moe_tp1/moe_ep1` | 1 shape: `tp1/dp1/moe_tp1/moe_ep1` |
| 2 | `ValueError: ... dense Gemma 4 variants require moe_ep_size=1, got 2` | 3 shapes, all `moe_tp=1, moe_ep=1` |
| 4 | same `ValueError` | 7 shapes, all `moe_tp=1, moe_ep=1` |
| 8 | same `ValueError` | 15 shapes, all `moe_tp=1, moe_ep=1` |

The branch shapes are pure TP/DP/replica combinations (`tp{1,2,4,8}/dp1`); no
expert-parallel dimension is offered for the dense checkpoint.

Routed control, `nvidia/Gemma-4-26B-A4B-NVFP4`, is unchanged by the fix:

| GPU budget | `origin/main` | This branch |
| ---: | ---: | ---: |
| 2 | 6 shapes | 6 shapes (identical set) |
| 4 | 16 shapes | 16 shapes (identical set) |
| 8 | 36 shapes | 36 shapes (identical set) |

## Public recommendation path

`run_recommendation(..., stack="engine")` with offline doubles,
`max_candidate_gpus=8`, `max_trials=8`, `parallelism=1`, `concurrency=1`,
2048 input / 128 output tokens, `gb300`/`vllm`/aggregated:

| Checkpoint | `origin/main` `f85518b36` | This branch |
| --- | --- | --- |
| `nvidia/Gemma-4-31B-IT-NVFP4` | `ValueError: ... dense Gemma 4 variants require moe_ep_size=1, got 2` | 8 evaluated, 8 feasible, 8 selected, `moe` dims `{(1, 1)}` |
| `nvidia/Gemma-4-26B-A4B-NVFP4` | 8 evaluated, 8 feasible, 8 selected, `moe` dims `{(1, 1), (1, 2), (1, 4), (1, 8), (2, 1), (4, 1)}` | identical |

The dense checkpoint no longer aborts the whole recommendation, and the routed
checkpoint keeps every expert-parallel candidate it had before.

## Generator producer path

`enumerate_profiling_configs()` is the other consumer of `check_is_moe()`: the
result decides the wideEP gate, the multi-node memory floor, and
`build_disagg_parallel_lists(is_moe=...)`. The probe replaces the system-config,
weight, hardware-support and DGD-rendering lookups with fixed stubs (8 GPUs per
node, 180 GiB per GPU, 1.5x memory multiplier, 60 GiB dense / 800 GiB routed
weight so the routed case crosses the wideEP threshold) and calls it for
`gb300`/`vllm`:

| Checkpoint | `origin/main` `f85518b36` | This branch |
| --- | --- | --- |
| `nvidia/Gemma-4-31B-IT-NVFP4` | 6 prefill + 6 decode candidates, all 6 shapes expert-parallel (`moe_ep` 2/4/8 at `tp` 1/2/4/8), `is_moe=True`, wideEP off | 4 prefill + 4 decode candidates, `moe_tp=moe_ep=1`, `is_moe=False`, wideEP off |
| `nvidia/Gemma-4-26B-A4B-NVFP4` | 4 + 4 candidates, 4 expert-parallel shapes, `is_moe=True`, wideEP auto-enabled | identical |

The dense producer candidates were shapes a dense checkpoint cannot serve; the
routed ladder is unchanged.

## Cross-layer consumers of the classification

`check_is_moe()` also gates the search-space, CLI, and capacity paths. Each row
is pinned by a test rather than argued from reachability:

| Consumer | Dense Gemma 4 behavior | Test |
| --- | --- | --- |
| `generator/enumerate.py::enumerate_profiling_configs` | wideEP gate and multi-node floor stay off, and `build_disagg_parallel_lists(is_moe=False)` yields only pure TP/DP candidates | `test_dense_gemma4_producer_configs_are_expert_parallel_free`, `test_routed_gemma4_producer_configs_keep_expert_parallel_shapes` |
| `generator/request/sweeper.py::_resolve_model_facts` | `ModelFacts.is_moe` is `False`, so the rule engine never selects MoE fields | `test_generator_model_facts_follow_the_classification` |
| `sdk/task_v2.py::Task._is_moe` → `attention_dp_size` in `run_single_agg` / `run_single_disagg` | a requested `dp` never reaches the attention axis (`attention_dp_size=1`); routed keeps `dp>1` on both roles | `test_run_single_agg_attention_dp_follows_the_gemma4_classification`, `test_run_single_disagg_attention_dp_follows_the_gemma4_classification` |
| `legacy_cli/api.py::_resolve_moe_parallelism` | MoE widths pass through untouched instead of being resolved | `test_dense_gemma4_moe_parallelism_is_passed_through_untouched` |
| `legacy_cli/api.py` static and disagg `check_is_moe` gates | neither role calls `resolve_model_config_moe_comm`; routed static and disagg still resolve every required phase | `test_dense_gemma4_static_estimate_skips_moe_comm_resolution`, `test_dense_gemma4_disagg_estimate_skips_moe_comm_resolution`, `test_routed_gemma4_static_estimate_still_resolves_moe_comm`, `test_disagg_resolves_each_role_moe_comm_before_session_run` |
| `sdk/moe_comm_resolver.py::resolve_model_config_moe_comm` | returns `None` before any database access | `test_dense_gemma4_never_resolves_moe_communication` |
| `sdk/memory/kv_cache.py` preflight | classification and the dense `moe_ep_size>1` rule run before the perf-DB lookup, so neither corrupt metadata nor unsupported expert parallelism can reach the naive fallback | `test_estimate_kv_cache_rejects_corrupt_gemma4_metadata_before_the_database`, `test_estimate_kv_cache_rejects_dense_gemma4_moe_ep_before_the_database` |
| `sdk/models/blocks/moe.py::MoEBlockShape.from_model_info` | reached only behind `_is_moe` or the resolver's early return, and its exception is tolerated | covered by the `task_v2` and resolver tests above |

The `gemma4.py` dense guard now raises `InvalidEngineConfigurationError` instead
of asserting, so `python -O` cannot disable it and the Rust bridge treats an
explicit dense `moe_ep_size>1` as non-fallback-safe. The diagnostic is built by
the shared `dense_gemma4_moe_ep_error` helper, which the KV-cache preflight also
uses before it resolves the perf database. The support-matrix classifier matches
it by the same anchored diagnostic
(`tools/support_matrix/fpe_support_matrix.py`).

## Regression tests added

| Test | Contract |
| --- | --- |
| `test_dense_gemma4_producer_configs_are_expert_parallel_free` | Generator enumeration: non-empty prefill/decode producer candidates, every candidate `moe_tp=moe_ep=1`, `is_moe=False`, wideEP never enabled |
| `test_routed_gemma4_producer_configs_keep_expert_parallel_shapes` | Generator enumeration keeps the MoE branch, the wideEP auto-enable, and `moe_ep>1` candidates for the routed checkpoint |
| `test_generator_model_facts_follow_the_classification` | `ModelFacts.is_moe` (generator rule-engine input) is `False` dense / `True` routed |
| `test_dense_gemma4_never_resolves_moe_communication` | `resolve_model_config_moe_comm` returns `None` for dense Gemma 4 before it touches the database |
| `test_dense_gemma4_public_recommendation_is_expert_parallel_free` | Final consumer: at least one feasible candidate, non-empty selection, every candidate `moe_tp=1`/`moe_ep=1`, no dense-rejection reason |
| `test_dense_gemma4_rejects_moe_controls_before_database_or_capacity` | `compile_engine`, `KVCacheEstimator.from_request`, and `estimate_num_gpu_blocks` reject `moe_backend`/`enable_eplb`/`wideep_num_slots`/`moe_kernel_source` before database or capacity fallback |
| `test_routed_gemma4_still_accepts_the_same_moe_controls` | The same four controls stay valid for the routed checkpoint |
| `test_estimate_kv_cache_never_falls_back_after_corrupt_checkpoint_metadata` | `allow_naive_fallback=True` cannot turn corrupt expert metadata into a naive capacity |
| `test_estimate_kv_cache_rejects_corrupt_gemma4_metadata_before_the_database` | Classification precedes the perf-DB lookup: corrupt metadata raises even when the database is unavailable and the fallback is allowed |
| `test_estimate_kv_cache_keeps_the_naive_fallback_for_a_valid_dense_gemma4_checkpoint` | A well-formed dense checkpoint with a missing database still takes the documented naive fallback |
| `test_estimate_kv_cache_rejects_dense_gemma4_moe_ep_before_the_database` | A dense checkpoint with `moe_ep_size=2` and `allow_naive_fallback=True` is rejected before the perf-DB lookup and never priced by the naive estimator |
| `test_dense_gemma4_static_estimate_skips_moe_comm_resolution` | Static CLI gate: no `resolve_model_config_moe_comm` call for dense Gemma 4 |
| `test_dense_gemma4_disagg_estimate_skips_moe_comm_resolution` | Disagg CLI gate: neither role resolves MoE communication for dense Gemma 4 |
| `test_routed_gemma4_static_estimate_still_resolves_moe_comm` | Routed Gemma 4 keeps the static-path resolution for both required phases |
| `test_dense_gemma4_moe_parallelism_is_passed_through_untouched` | `_resolve_moe_parallelism` leaves dense MoE widths untouched and still resolves routed ones |
| `test_dense_variant_rejects_moe_ep_gt_1` | The dense guard raises `InvalidEngineConfigurationError` (not `AssertionError`) for an explicit `moe_ep_size>1`, so `python -O` cannot disable it and the Rust bridge keeps the rejection non-fallback-safe |
| `test_run_single_agg_attention_dp_follows_the_gemma4_classification` | Agg `run_single_agg` forces `attention_dp_size=1` for dense and preserves `dp` for routed |
| `test_run_single_disagg_attention_dp_follows_the_gemma4_classification` | Both disagg roles apply the same dense/routed attention-DP rule |
| Inconsistent-metadata cases in `tests/unit/sdk/models/test_model_config.py` | `(128, 0)`, `(0, 8)`, `(None, 8)`, `(128, None)`, `(128, 129)`, `(128, -8)` and `(-128, 8)` raise `InvalidEngineConfigurationError`; `(None, 0)`, `(0, 0)`, and `(128, 8)` classify as dense/dense/routed |
| `test_no_current_cli_matrix_row_records_the_dense_gemma4_rejection` | Generated-artifact guard: no CLI support-matrix row the generator still enumerates records the dense rejection for a checkpoint the classifier calls dense |
| `test_the_current_cli_matrix_still_covers_a_dense_gemma4_checkpoint` | The same guard cannot pass vacuously: a currently enumerated dense Gemma 4 row must exist |
| `test_dense_gemma4_coverage_check_ignores_other_dense_families` | Dropping every Gemma 4 row fails the coverage assertion even though other dense families remain in the matrix |
| `dense_gemma4_expert_parallelism_is_not_fallback_safe` (Rust) | The bridge's `InvalidEngineConfig` mapping is not fallback-safe, so `best_available` surfaces the dense rejection instead of using the regression estimator |

## Support-matrix artifacts

The bundled support matrices recorded the pre-fix rejection
(`dense Gemma 4 variants require moe_ep_size=1`) for
`nvidia/Gemma-4-31B-IT-NVFP4`. The CLI matrix is reproducible on any host, so
the affected rows were regenerated in this branch; the op-level FPE matrix is a
campaign artifact and is left as an explicit blocking follow-up.

### CLI matrix: regenerated here

Twelve rows moved from `FAIL` (a `sweep_agg` / `_get_disagg_worker_candidates`
traceback ending in the dense rejection) to `HYBRID_PASS`:

| File | Rows |
| --- | ---: |
| `systems/support_matrix/b200_sxm.csv` | 2 |
| `systems/support_matrix/b300_sxm.csv` | 2 |
| `systems/support_matrix/gb200.csv` | 2 |
| `systems/support_matrix/gb300.csv` | 2 |
| `systems/support_matrix/rtx_pro_6000_server.csv` | 4 |

They were produced with the repository's own generator, one invocation per
`(system, backend, version)`, and only the previously failing rows were
replaced with the generated rows:

```bash
python python/aisim/tools/support_matrix/generate_support_matrix.py \
  --model nvidia/Gemma-4-31B-IT-NVFP4 --system gb300 \
  --backend trtllm --backend-version 1.3.0rc20 --output <tmp-dir>
```

Each run reports 2/2 `Hybrid estimable` (`HYBRID_PASS`, source
`hybrid:xshape`): SILICON is still missing perf data for that backend version,
so the documented hybrid retry answers, and the dense rejection is gone.

Ten `FAIL` rows remain, all for `trtllm 1.3.0rc23` on the same five systems.
That backend version is retired: `SupportMatrix().generate_combinations()` now
enumerates `1.3.0rc20` and `1.3.0rc29` for `trtllm`, so the generator cannot
produce those rows. They are historical records in a cumulative matrix, not a
statement about the current classifier.

### Op-level FPE matrix: blocking follow-up

103 `FAIL` rows across ten `systems/fpe_support_matrix/*.csv` files still carry
the dense rejection text (counts are per file): `a100_sxm` 7, `b200_sxm` 9,
`b300_sxm` 8, `b60` 2, `gb200` 10, `gb300` 9, `h100_sxm` 16, `h200_sxm` 17,
`l40s` 13, `rtx_pro_6000_server` 12.

These rows are **stale and must not be read as current support**, but they
cannot be refreshed from a single row, system, or PR:
`build_fpe_support_matrix.py` rejects shards whose `source_sha` or workload
identity differ, so the split CSVs come from one campaign that probes every
model on every system with a single source identity
([`docs/ci/accuracy.md`](../../docs/ci/accuracy.md#fpe-support-qualification)).
A per-model shard would also change the per-shard probe and topology counts that
sibling rows in the same file carry, so patching the affected rows alone would
produce an internally inconsistent artifact.

Blocking follow-up: rerun the FPE campaign on the fixed sources, rebuild the
split CSVs, and confirm no Gemma 4 row keeps the dense rejection. Rows that
still fail for a real reason (`PERF_DATA_MISSING`, `SDK_UNREPRESENTABLE`,
`BUILD_FAILED`) stay failed with their own cause.

### Guard

`tests/unit/support_matrix/test_support_matrix_dense_gemma4.py` fails when a
row the CLI generator still enumerates records the dense rejection for a
checkpoint the classifier calls dense, and also fails if the dense checkpoint
drops out of the matrix entirely (so the first check cannot pass vacuously). It
reports the ten rows above before the regeneration and passes after it. The
Rust counterpart, `dense_gemma4_expert_parallelism_is_not_fallback_safe` in
`crates/core/src/perfmodel/fpm/model.rs`, pins that the bridge maps the typed
Python error to a non-fallback-safe `AicError`, so `best_available` surfaces the
dense rejection instead of degrading to the regression estimator.

## Limitations

- Only `gb300`/`vllm` `0.30.0` and the two bundled checkpoints above were
  exercised. Shape availability on other databases is not claimed.
- Generator producer numbers use fixed stubs for the system config, model weight
  and DGD rendering. They pin the branch taken and the shape sets, not absolute
  candidate counts for a real cluster.
- Recommendation scores come from constant offline metrics. This document makes
  no accuracy claim, and it does not compare against a GPU serving run.
- Local measurements used a release native extension built with
  `cargo build -p aisim-core --release --features extension-module`; the CI wheel
  uses the standard `maturin` build with `jemalloc`. Neither the allocator nor
  the build path affects classification or shape enumeration.
