---
description: >
  Adding or changing collector case coverage: case YAML, op plans, skips,
  running subsets.
paths:
  - "collector/**"
  - "tests/unit/collector/**"
---

# Collector Case Authoring

How to add or change collection coverage. The declaration surface is the case
YAML under `collector/cases/`, the op plans in `collector/op_plans.py` and the
skip list in `collector/skips.py`. If you need a new kind of rule, re-read
`layer_permissions.md` first. Two principles from `overview.md` govern every
change: generate the same cases for all three frameworks, and leave cases out
only through `collector/skips.py`.

## The equation

```text
plan cases = expand(base sweep grid ∪ model shapes), one case per persisted key
runnable   = plan cases − what collector/skips.py skips
```

- **Base grids** (`cases/base_ops/<op>.yaml`) give interpolation its uniform
  support: token, batch and sequence sweeps, parallelism sizes, routing
  distributions. Density here is the knob for collection cost.
- **Model shapes** (`cases/models/<architecture>_cases.yaml`,
  `model_case_values`) guarantee exact hits for real models and keep
  correlated dimensions together: query heads, KV heads, head_dim and window
  are one tuple, never crossed with another model's values.
- **Deduplication** keeps one case per persisted key. When several models
  write the same key, the plan keeps one owning model.

## Adding coverage

| Goal | Action |
|---|---|
| New architecture | a new `cases/models/<architecture>_cases.yaml` |
| New model, same architecture | add its path to `model_paths` in the existing file |
| New checkpoint quant artifact | a new `model_case_values` row with its `quantization_modes` (MoE); never merge quant-distinct artifacts into one row |
| Change a shared sweep | edit the op's `cases/base_ops/<op>.yaml` |
| New op collector | only when no existing op can produce the points; give it an `OP_PLANS` entry ("Op plans" below) |
| A dtype, op or model one backend cannot run on some SMs, or a quant mode serving routes elsewhere on some SMs | a `collector/skips.py` entry, its reason and source as the comment above it |
| A case one backend runs on two kernels, and only one kernel cannot run it | a `collector/skips.py` entry naming that `dispatch`; the case still runs on the other kernel |

## Model file structure

- `architecture`, `model_path`, `model_paths` (aliases), `include_base`,
  `base_ops`, `model_ops`, `framework_specific_base_ops`,
  `model_specific_base_ops`: op activation.
- `all_frameworks_op_cases` / `framework_specific_op_cases`: op sections. The
  only meaningful value is `cases: all`; there are no selectors.
- `model_case_values` / `framework_specific_model_case_values`: shapes.
  `model_aliases` means one physical case under many artifact names
  (shape-only ops); `model_paths` inside a row means one case per path
  (runtime-sensitive).

## Narrowing shapes: declare, never filter

There are no selectors, so "fewer shapes for this model" has one legal form:
declare the correlation and let the generator expand only valid combinations.

| The constraint belongs to | Where it goes | Example |
|---|---|---|
| the model (head count, valid TP shards, windows) | a field on its `model_case_values.<op>` row | attention `tensor_parallel_sizes` |
| the op's universal math (identities, budgets) | a generator constraint or base-op budget field | `tp*ep == num_gpu`; `batch*seq <= max_context_tokens` |
| the platform (dtype floor) | `collector/skips.py` | `layer_permissions.md`, "Leaving cases out" |
| device memory | the worker's run-time refusal | `layer_permissions.md`, "Capacity" |
| the framework (kernel limits) | nothing: the cases fail and are recorded, unless "Leaving cases out" applies | |

If an op's generator cannot honor a model-correlated field you need, extending
it is a mechanism change: propose it. Do not approximate it with
post-generation filtering. Over-collection in the meantime is acceptable:
extra valid points are interpolation support, and invalid ones fail into the
classified log.

## Unresolvable declarations fail loudly

Cases with the wrong identity are worse than no cases: they benchmark
successfully and poison the database with mislabeled rows.

1. When a declared input cannot be resolved (a model row, a quant mode, an
   attention or MLA profile, an artifact's config), the generator raises. It
   never substitutes defaults, another model's geometry or a "close enough"
   quant mode.
2. `model_aliases` is the only sanctioned aliasing, and only for shape-only
   ops where the artifact provably cannot change the invoked kernel or the
   persisted key. Everything else uses `model_paths` or fails.
3. An op that plans zero cases must be explained by `collector/skips.py`
   entries; zero cases with no logged reason is a population bug.

## Op plans

Every registered op except moe_a2a has one `OpPlan` in `collector/op_plans.py`
(`OP_PLANS`). The moe_a2a collectors keep their own population: each backend
measures a different all-to-all library for one launched world.

An op plan declares:

- **Case type**: a named tuple whose string form is the list form
  (`['fp8', 64, 4096, 7168]`), so case IDs, checkpoints and `--case-filter`
  strings keep working.
- **Expand function**: every case, in run order, from the YAML above.
- **`dispatch`**: the kernel or runner a backend measures a case on, the one
  serving selects. `--plan-only` counts cases per dispatch, and a skip entry
  may name one.
- **`rows`**, for a backend whose run function takes more than the case: the
  run arguments for one case. Rows say how a case runs, never whether; a case
  with no row is an error. A field that enters the persisted key or changes
  the kernel belongs in the case or the dispatch. A row may be an
  `{id, params}` mapping; a run records it under its `id`.
- **`batch_rows`**, for a backend that times several cases on one module
  build: tasks of (row, the cases it times). A row lists its points, so the
  worker never derives them, and every kept case is timed by exactly one row.

`plan(op, backend, sm)` removes what `collector/skips.py` skips for that
backend and SM; each getter is one line,
`return plan_cases("<op>", "<backend>", get_sm_version())`. A plan does not
depend on the device: the worker refuses what the device cannot hold
(`layer_permissions.md`, "Capacity").

Planning imports no framework, so `--plan-only` runs on any CPU. Rows that
build framework objects are built only when the getter runs; on a CPU,
`--plan-only` still counts those cases but reports `plan_hash: null`.

`tests/unit/collector/test_op_plans.py` runs over every entry: the one-line
getters, that the backends differ only by `collector/skips.py` entries naming
them, that every left-out case is counted under its entry, that every kept
case is timed exactly once, and the case string form.

To add an op, write its case type and expand function from shared YAML, add
the `OP_PLANS` entry with `dispatch` and `rows` or `batch_rows` as needed, and
make each getter the one-line call. Inspect a plan without a GPU:

```bash
python3 collect.py --backend sglang --model-cases-full --sm 120 --ops gemm \
  --plan-only --plan-table /tmp/plan
```

The `plans` block gives the cases per dtype, what was left out under which
`collector/skips.py` entry, and `plan_hash`, the `case_plan_hash` of a run that
attempts every case. `--plan-table` writes one CSV row per case for diffing.

## Running subsets

Subset selection is a run-time concern, never persisted to YAML:

```bash
python3 collect.py --backend sglang --model-path <model> --gpu b200_sxm        # one model
python3 collect.py --backend sglang --ops gemm --case-filter "['fp8_block', "  # one GEMM dtype
python3 collect.py --backend trtllm --resume                                   # finish an interrupted run
```

A case filter matches a substring of the case or row string; with
`batch_rows`, it selects whole rows.

## What not to do

1. No generation recipes inside model op sections: shapes go in
   `model_case_values`, recipes in `base_ops/`.
2. No per-model narrowing of another op's grid "to save time"; tune the base
   grid instead.
3. No YAML keys that condition on batch, sequence, token or feature values.
4. No skip entries for scattered failures, and no per-framework narrowing of a
   shared grid beyond `collector/skips.py` entries.
