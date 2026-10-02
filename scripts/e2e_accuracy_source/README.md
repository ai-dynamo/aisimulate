# InferenceX accuracy source resolution

This repository-only package prepares source evidence before prediction. It
uses the source-resolution and cohort contracts from `aisim-e2e-gym` revision
`2ad1ec4287ad9fb5857680901eb65b4ff0cb4098`, under
`tools/silicon_predictor/src/silicon_predictor/`. The replay projection follows
`tools/aisimulate_predictor/run_predictions.py` at that revision.

The resolver reads immutable InferenceX recipes without executing their shell,
checks reviewed framework-default source hashes, resolves checkpoint metadata,
and retains source settings, runtime evidence, and workload controls. The
[`manifests/`](manifests/) JSON files identify the upstream paths, revisions, and hashes for
reviewed defaults. Missing evidence is an unresolved outcome, never an invented
serving default. Unknown kernel/quantization mappings remain unsupported.

## Package layout

Keep Python modules flat and group the reviewed data in `manifests/`:

| Responsibility | Modules |
| --- | --- |
| Cohort and records | `cohort.py`, `filter.py`, `staleness.py`, `schema.py`, `mapping.py` |
| Source I/O and hash verification | `sources.py`, `inferencex_recipe.py` |
| Static launchers and shell values | `legacy_recipe.py`, `shell_recipe.py`, `shell_values.py` |
| Runtime evidence | `runtime_recipe.py`, `single_node_runtime.py` |
| Effective defaults | `framework_defaults.py`, `sglang_additional_defaults.py`, `trt_additional_defaults.py`, `workload_defaults.py` |
| Checkpoint identity | `checkpoint_quantization.py`, `model_config_snapshot.py` |
| Source resolution | `deployment.py` |
| Prediction projections | `estimate.py` (historical wheels), `replay.py` |

`sources.py` loads manifests once per process and verifies upstream bytes against
reviewed hashes. Callers keep backend-specific rules, cache scope, and error
context. Treat loaded manifests as immutable; restart the campaign after editing
them. Manifest contents and recorded filenames are unchanged by their folder
location. The campaign hashes Python and JSON files recursively, so moving a
manifest cannot silently exclude it from driver provenance.

Prediction workers import the small projection modules; source resolution runs
in the parent before those workers start.

Keep backend rules separate: similar knob names can have different release,
hardware, and runtime conditions. Further package nesting would add import
churn without simplifying those rules.

## Campaign boundary

- `cohort.py` joins workflow provenance, applies gym filtering, deduplication,
  image coherence, and the 180-day configuration freshness window. P/D and
  multinode records stay eligible. Every input is selected or counted as excluded.
- Resolution runs once per selected point before predictor subprocesses. Source
  files and framework/checkpoint lookups are cached, so there is no discovery in
  the simulated request/token loop.
- `ResolvedInferenceXSource` lowers resolved evidence into the wheel's public
  estimate contract. Older wheels use the equivalent source-derived estimate
  arguments, with backend memory aliases normalized consistently and dense-model
  MoE overrides omitted. Both paths materialize the same verified checkpoint bytes.
- `replay.py` carries gym's modeled per-role engine controls and source workload
  into `ReplaySpec`, including the NumPy length sampler and benchmark seed.
  Performance-database versions are selected independently of measured framework
  versions. Estimate failure does not suppress replay or its coverage.
- The driver writes full resolved evidence and outcomes only to `--evidence`.
  Nothing in that directory is uploaded by the workflow. For policy
  `gym-resolved-config-v2`, `cohort_sha256` hashes the complete resolved inputs,
  including source evidence; `driver_sha256` includes resolver code and source
  manifests. Only the validated aggregate summary is public.

`--source-cache` may contain gym-compatible `runtime-evidence/` indexes and
checksum-verified runtime artifacts. Without matching runtime artifacts, the
resolver uses pinned recipes and verified defaults where sufficient; otherwise
it reports the missing knobs. It does not guess a dynamic request limit from
benchmark concurrency or run gym's optional old-revision KV capacity estimator.

Configuration parity is distinct from simulation support and prediction
accuracy. Original graph/kernel controls and unmodeled client behavior remain
in the resolved evidence even when gym's replay projection cannot model them.
Unpinned historical checkpoint revisions are explicitly identified by the
resolver; fetching current metadata cannot prove the measured revision.

## Validation

Run `pytest -c /dev/null tests/e2e_accuracy_source tests/test_e2e_accuracy_nightly.py`.
Tests use synthetic inputs and pinned expected behavior; PR checks do not need
the internal gym repository or network access. The package's hashed dependency
lock is installed by Pages and the nightly campaign.

On the September 28 dump, the new policy matches gym's **2,281 measurement IDs**
exactly. A live differential source check matched complete deployment and
evidence objects for configs **202, 618, 909, and 1553**, including the explicit
unresolved-knob outcome for 618. These are input-parity checks, not a latency
qualification or a refreshed public accuracy snapshot.
