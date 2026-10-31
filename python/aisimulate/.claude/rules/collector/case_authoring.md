---
description: >
  Adding/changing collector case coverage: base grids, model shapes, capability floors.
paths:
  - "collector/**"
  - "tests/unit/collector/**"
---

# Collector Case Authoring

How to add or change collection coverage. The declaration surface is exactly
two kinds of YAML plus one capability table — if you feel the need for a new
kind of rule, re-read `layer_permissions.md` first.

Read `overview.md` first. Two of its principles govern every change below:
generate the same cases for all three frameworks (a framework that cannot run
part of a grid fails those cases and the failures are recorded), and remove
only whole regions that cannot run, such as an SM without a dtype, when the
plan is expanded.

## The equation

```text
plan cases = dedup( expand(base sweep grid) ∪ expand(model shapes) )
runnable   = plan cases − whole regions (capabilities.yaml floors and backend_gaps)
```

- **Base grids** (`cases/base_ops/<op>.yaml`) give interpolation its uniform
  support: token/batch/seq sweeps, parallelism sizes, routing distributions.
  One file per op. Density here is the only knob for collection cost.
- **Model shapes** (`cases/models/<architecture>_cases.yaml`,
  `model_case_values`) guarantee exact hits for real models and keep
  correlated dimensions together (query heads / KV heads / head_dim / window
  are ONE tuple — never crossed with another model's values).
- **Dedup** happens on physical tuples when base and model expansions overlap.

## Adding coverage

| Goal | Action |
|---|---|
| New architecture | one new `cases/models/<architecture>_cases.yaml` |
| New model, same architecture | add path to `model_paths` in the existing file |
| New checkpoint quant artifact | new `model_case_values` row with its `quantization_modes` (MoE) — never merge quant-distinct artifacts into one row |
| Change a shared sweep | edit the op's `cases/base_ops/<op>.yaml` |
| New op collector | only when no existing op can produce the data points |
| Op needs newer SM than some GPUs | axis `min_sm` (quant modes) or `cases/capabilities.yaml` |
| One backend's collector cannot run a whole (op, dtype) on some SMs | `capabilities.yaml` `backend_gaps` entry naming a guard note `G-<backend>-<slug>` in `collector/notes/<backend>.yaml` |
| New GEMM dtype | add it to `gemm_types` in `cases/base_ops/gemm.yaml`, plus a floor if the hardware needs one |
| Quant mode exists on an SM interval, not a floor (framework routes the platform elsewhere) | quant-mode `max_sm_exclusive` with a mandatory serving-dispatch citation (owner decision 2026-07-13: negative gates are sanctioned only on the quant-mode axis, only with a file:line@version citation, and every citation is re-verified on version bumps — the default-open behavior lets NEW SMs be probed and fail loudly instead of being silently whitelisted out) For an op on an op plan (GEMM, MoE), the same fact is a `backend_gaps` entry whose guard note carries the citation, so R8 re-checks it at every pin bump |

## Model file structure (unchanged from v2 core)

- `architecture`, `model_path`, `model_paths` (aliases), `include_base`,
  `base_ops`, `model_ops`, `framework_specific_base_ops`,
  `model_specific_base_ops` — op activation.
- `all_frameworks_op_cases` / `framework_specific_op_cases` — op sections;
  the only meaningful value is `cases: all` (activation). There are no
  selectors: `case_ids` / `contains` / `indices` / `ranges` / `limit` /
  `rules` no longer exist.
- `model_case_values` / `framework_specific_model_case_values` — shapes.
  `model_aliases` = one physical case, many artifact names (shape-only ops).
  `model_paths` inside a row = one case per path (runtime-sensitive).

## Legitimate shape narrowing: declare, never filter

There are no selectors, so "I need fewer shapes for this model" has exactly
one legal form: **declare the correlation and let the generator expand only
valid combinations**. Route by who owns the fact:

| The constraint belongs to | Where it goes | Existing example |
|---|---|---|
| the model (head count, valid TP shards, windows) | a field on its `model_case_values.<op>` row | `attention` profiles: `tensor_parallel_sizes` — the generator expands only valid shards of that tuple |
| the op's universal math (identities, budgets) | generator constraint / base-op budget field | `tp*ep == num_gpu`; `batch*seq <= max_context_tokens` |
| the platform (dtype floor, memory) | `capabilities.yaml` / the collector memory filter | see `layer_permissions.md` |
| the framework (kernel limits) | nothing: the cases fail and are recorded. A whole (op, dtype, SM) region goes in `backend_gaps`; a case that would mismeasure, hang or waste time gets a guard | see `layer_permissions.md` |

If the op's generator does not yet support the model-correlated field you
need (e.g. `mla_bmm` sweeps a global head grid with no model axis), the fix
is to extend that generator to honor a declared field — a mechanism change,
so propose it to the human. Do NOT approximate it with any post-generation
filtering; that is how the selector engine was born. Over-collection in the
meantime is acceptable: extra valid points are interpolation support, and
invalid ones fail into the classified log.

## Unresolvable declarations fail loudly (no generation-side fallbacks)

Fallback deception has a generation-side twin: cases that carry the wrong
identity are worse than no cases, because they benchmark successfully and
poison the database with mislabeled rows.

1. When a declared input cannot be resolved — a model row, a quant mode, an
   attention/MLA profile, an artifact's config — the generator RAISES. It
   never substitutes defaults, another model's geometry, or a "close enough"
   quant mode.
2. `model_aliases` is the only sanctioned aliasing, and only for declared
   shape-only ops where the artifact provably cannot change the invoked
   kernel or the persisted key. Everything else uses `model_paths` (one
   physical case per artifact) or fails.
3. A planned op that expands to zero cases must be explainable from logged
   drops (capability floors, memory filter). Zero cases with no logged
   reason is a population bug to fix — not a clean completion.

## Op plans (GEMM and MoE so far)

An op plan expands one case list for every CUDA backend and leaves out only
whole regions. `collector/op_plans.py` holds one `OpPlan` entry per migrated
op: its case type (a named tuple whose string form is still the list form,
`['fp8', 64, 4096, 7168]`, so case IDs, checkpoints and `--case-filter`
strings do not change), its expand function (every case, in run order, from
the YAML above), its dtype field and the fields that share one module build.
`plan(op, backend, sm)` removes the regions a floor or a backend gap rules
out; each backend's getter is one line,
`return plan_cases("<op>", "<backend>", get_sm_version())`.

A backend whose run function takes more arguments than the shared case may
declare `rows`: the run arguments for one case. Rows say how a case runs,
never whether; a case with no row is an error.

`tests/unit/collector/test_op_plans.py` runs over every entry, so moving an
op onto a plan needs no new test: it checks the one-line getters, that the
backends differ only by backend gaps, that only whole regions are left out,
and the case string form.

To move an op: write its case type and expand function from the existing
shared YAML, add the `OP_PLANS` entry, make each getter the one-line call and
delete what the getters filtered. Decide each deleted filter by
`overview.md`: it either goes (the cases run and failures are recorded),
becomes a `backend_gaps` entry with a guard note (a whole region one backend
cannot run), or becomes a guard that raises in the worker (mismeasure, hang,
or significant wasted time).

Inspect a plan without a GPU:

```bash
python3 collect.py --backend sglang --model-cases-full --sm 120 --ops gemm \
  --plan-only --plan-table /tmp/plan
```

The `plans` block gives cases per dtype, the regions left out with the reason
or note id, and `plan_hash`, the case_plan_hash of a run that attempts every
case. `--plan-table` writes one CSV row per case for diffing.

## Running subsets (healing)

Subset selection is a RUNTIME concern, never persisted to YAML:

```bash
python3 collect.py --backend sglang --model-path <model> --gpu b200_sxm   # one model
python3 collect.py --backend trtllm --ops moe --case-filter "tp=4"        # substring filter
python3 collect.py --backend sglang --ops gemm --case-filter "['fp8_block', "  # one GEMM dtype
python3 collect.py --backend trtllm --resume                              # finish an interrupted run
```

## What NOT to do

1. No generation recipes inside model op sections — shapes go in
   `model_case_values`, recipes in `base_ops/`.
2. No per-model narrowing of another op's grid "to save time" — extra points
   are cheap; missing interpolation support is not. Tune the base grid instead.
3. No new YAML keys that condition on batch/seq/token/feature values.
4. Do not put a framework gap in `dtype_min_sm` or `op_min_sm`: those hold
   hardware facts only. A backend that cannot run a whole (op, dtype) region
   goes in `backend_gaps` with a guard note; anything finer fails and is
   recorded.
5. No per-framework narrowing of a shared grid. If one framework cannot run
   part of it, those cases fail and are recorded.
