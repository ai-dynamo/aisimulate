---
description: >
  Which collector layer may hold which kind of rule; leaving cases out;
  capacity; kernel selection and serving parity; deviations; version
  contract; module boundary.
paths:
  - "collector/**"
  - "tests/unit/collector/**"
---

# Collector Layer Permissions

Which layer of the collector may hold which kind of rule. When "fixing" a
failing case, pick the layer first, then act. Read `overview.md` first for the
vocabulary and the failure principle.

## The one-line architecture

```text
plan cases = expand(cases/base_ops grids ∪ cases/models shapes), one case per persisted key
runnable   = plan cases − what collector/skips.py skips
result     = perf rows + classified failure records   (a failure is data, not a defect)
```

## Permission table

| Layer | Allowed | Forbidden |
|---|---|---|
| `cases/base_ops/*.yaml` | sweep density and ranges; op budgets such as `batch*seq <= max_context_tokens` | per-model special cases; shape exclusion rules; SM or version gates |
| `cases/models/*_cases.yaml` | model structural shapes, correlated tuples, artifact and quant policy, op activation (`cases: all`) | selectors, match rules, a second generator recipe |
| `collector/op_plans.py` | an op's case type, expand function, `dispatch`, `rows` or `batch_rows` | deciding whether a case runs |
| `collector/skips.py` | one entry per reason, naming what it skips (`backend`, `op`, `sm`, `model`, `dtype`, a case field, or a `dispatch` kernel), with the reason as the comment above it | an entry without a reason; skipping scattered failures; version pins or IDs |
| `<backend>/registry.py` | version routing (`VersionRoute`); maturity markers `unverified=True` (op × backend) and `unverified_sms=(...)` (op × backend × SM) | any shape-level information |
| `collect_*.py` | dispatch: run the kernel path serving selects and record `kernel_source`; raise a classified exception; refuse capacity before allocation | a silent `continue` or skip of a queued case; any other case filtering; patching code so one shape passes |
| executor (`collect.py`) | — off-limits during case-fixing tasks | changing checkpoint, case-ID or failure-classification logic to make a case pass |
| failure records (`errors_*.json`, summary) | append-only observation | feeding failures back into a declaration layer as new rules |

## Dispatch, not skip

> A legal branch changes HOW a case runs. An illegal branch changes WHETHER it runs.

- Legal: `if sm >= 100: use kernel A else kernel B`, when serving does the
  same. Both sides produce a row and record which kernel ran.
- Illegal: `if sm < 90: continue`. The case vanishes with no row and no
  failure record.

A collector has two legal responses to a queued case: **execute it, or
raise**. Whether a case runs is decided only in `collector/skips.py` and the
registry `unverified` markers.

## Leaving cases out

A framework kernel limit usually needs nothing: the case runs, fails, lands in
the failure log, and the next run is compared with this one. Do not write a
probe, a guard or a YAML rule for it.

Leaving cases out is legal only when running them would:

1. **mismeasure**: succeed on a different kernel than serving selects,
   producing a wrongly labeled row;
2. **hang** or kill the worker;
3. **fail as a whole block** or **waste significant time**, such as a full
   module load per case before a failure that depends only on the case fields
   and the SM; or
4. time a model the backend **cannot serve**.

Put such cases in `collector/skips.py`, with the reason and its source (a
framework file and symbol, or a failing run) as the comment above the entry.
No IDs and no versions: whoever moves a pin re-checks the comment. The plan
leaves the cases out, and `--plan-only` counts them under the entry.

A guard that needs the running kernel may raise in the worker instead, before
the expensive step. Its message starts with `FIXME(kernel-limit):` (the
failure classifier keys on that prefix), and the comment beside it says why.

## Capacity: refuse at run time, before allocation

Plans do not depend on the device, so no getter drops a case on memory. The
worker refuses a case or point the device cannot hold:

1. **Before allocating it**, in the worker; never a run-time `continue`.
2. **By size against capacity only**: an estimated footprint against live
   device memory, or a shape against the device's prefill chunk or KV pool.
   Never by framework version or model name.
3. **Classified**: the reason starts with `FIXME(kernel-limit): does not fit`.

## Kernel selection: framework truth, never a guess

A collector measures what the framework runs in serving. The kernel or backend
invoked for a (model, phase, dtype or quant, shape, SM) case must be the one
the framework's own dispatch selects under those conditions. A benchmark that
invokes the wrong backend produces silently wrong data, which is worse than a
crash.

1. **Prefer the framework's own dispatch path.** Build modules through the
   framework's builder or selector, as serving does, instead of instantiating
   a backend class by hand.
2. **A manual pin needs a source citation**: a comment naming the framework
   file and symbol that select this backend under these conditions, SM- and
   dtype-dependent branches included. When the plan picks it, it is the op's
   `dispatch`.
3. **`kernel_source` records the kernel that ran**, never the intended one.
4. **No invented fallbacks.** If the framework-selected path cannot be built
   or invoked, raise a classified error; never swap in another backend to keep
   collecting. Replicate a fallback chain only when serving performs it, and
   record what ran. A measurement-method fallback (CUDA graph to eager) is
   allowed only when the row records it. An API-compat shim may change how
   the same kernel is built, never which kernel runs.

## Metadata and input parity: serving truth, never reconstruction

A collector that hand-builds framework metadata or input tensors (attention
metadata, KV-cache parameters, position ids, workspace shapes) replicates
serving-internal state, and every replicated field can drift silently.

1. **Every hand-built field or input cites its serving population site**
   (framework file and symbol), like a pinned backend. A field changed without
   a citation is a review finding even when the change makes cases pass.
2. **Audit serving parity before blaming the framework.** Before calling a
   crash a framework bug, or guarding it, diff every collector-built field
   against the serving population code for the failing request shape, field
   by field. A traceback shows where it faulted, not whose contract broke.
3. **A pass on one SM validates only the fields its kernel path reads.** A
   failure on one SM only should first suspect the fields only that path
   reads.

## Deviations are notes

When a table cannot follow the serving-parity rules (it times an unfused path
where serving fuses, or a row is derived from another kernel), write a
deviation note: a comment at the code that measures the deviating path, giving
what differs and on which systems, the direction of the error when known, its
sources, and how it could be retired. The owner accepts it in review. The
comment is the one home for that fact; data comments and docs point to it.
Field reference: `docs/perf-model/collector/data-format.md` ("Deviations").

A note documents and bounds a deviation; it does not weaken the rule.

## Version contract: one release per collector file

Version history lives in git, not in side-by-side files or version ranges.

1. **Each collector module targets exactly one framework release.** Its
   `__compat__` is a single exact pin, `"<framework>==<version>"`, equal to
   the runtime `framework_manifest.yaml` resolves for its op (the framework
   default or a family pin). No ranges, no `!=` exclusions, no per-version
   branches.
2. **Upgrade in place.** A pin bump rewrites the module for the new release
   and moves its pin. Prior-version data stays under its version key. Do not
   add `_<version>` fork modules.
3. **Pin consolidation.** A model or family pin below the target version
   upgrades with it; a pin above the default is the next version for that
   family and needs its own GPU validation.
4. **A pin bump re-checks** every manual backend pin, serving citation,
   `collector/skips.py` entry and deviation note in scope against the new
   framework source.
5. **Enforced by tests**: `tests/unit/collector/test_framework_manifest.py`
   fails when a stock vLLM, SGLang or TRT-LLM registry module declares
   anything but `==` its resolved runtime.

## Module boundary: a collector task may only touch collector/

The collector's outward surface is its data contract: the perf row schema
(columns written by `helper.log_perf`), the canonical filenames
(`registry_types.PerfFile`) and the parquet finalization format. The SDK
(`src/aisimulate_core/sdk/perf_database.py`), the Rust perf database in
`crates/core`, `tools/support_matrix/` and the packaged data under
`src/aisimulate_core/systems/data/` consume it.

1. A collector task may modify `collector/` and `tests/unit/collector/`,
   nothing else, even when the change "obviously needs" a consumer-side
   change.
2. **Changing the data contract needs explicit human approval.** A new
   column, perf file or key dimension, or a rename of anything the SDK parses,
   is a coordinated producer and consumer change. If you find the need
   mid-task, stop and hand the human a proposal: the change, why, and which
   consumers must follow. An approved PR that declares the contract change may
   carry both sides.
3. SDK and modeling tasks do not reach into `collector/` either.
4. These rule files are human-owned policy. Do not edit them as a side effect
   of a fix; propose changes instead.

## Meta rules

1. **The default action for a failing case is no change.** Confirm it is
   recorded and classified, then stop. Escalate only per
   `failure_handling.md`.
2. **Mechanism changes need explicit human approval**: the executor, the
   op-plan mechanism (`OpPlan`, `plan()`), the case generator, failure
   classification and the case-ID format.
3. **Inexpressibility is the guardrail.** The case YAML schemas cannot express
   shape conditions or version predicates. Do not extend them; a case that
   must not run goes in `collector/skips.py`.
4. **Mark new collector tests `pytest.mark.unit`**: the documented local unit
   run (`-m unit`) skips unmarked tests.
