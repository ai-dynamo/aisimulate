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
reviewed defaults. Verified mode excludes missing evidence. Estimated mode fills
only allowlisted gaps and labels every assumption; unknown kernel/quantization
mappings and conflicting evidence remain unsupported.

## Package layout

Group recipe readers and defaults one level below the package root. Keep reviewed
JSON data in `manifests/` and orchestration/shared contracts at the root:

| Responsibility | Location |
| --- | --- |
| Static launchers and shell values | `recipes/inferencex_recipe.py`, `recipes/legacy_recipe.py`, `recipes/shell_recipe.py`, `recipes/shell_values.py` |
| Runtime evidence | `recipes/runtime_recipe.py`, `recipes/single_node_runtime.py`, `recipes/runtime_evidence.py` |
| Effective defaults | `defaults/framework_defaults.py`, `defaults/sglang_additional_defaults.py`, `defaults/trt_additional_defaults.py`, `defaults/workload_defaults.py` |
| Opt-in research assumptions | `defaults/research_defaults.py` |
| Reviewed source manifests | `manifests/` |
| Cohort and records | `cohort.py`, `filter.py`, `staleness.py`, `schema.py`, `mapping.py` |
| Source I/O and hash verification | `sources.py` |
| Checkpoint identity | `checkpoint_quantization.py`, `model_config_snapshot.py` |
| Source resolution | `deployment.py` |
| Prediction projections | `estimate.py` (historical wheels), `replay.py` |

`defaults/` groups related responsibilities. Verification status is recorded in
code and evidence: framework/workload defaults require reviewed sources, while
`research_defaults.py` fills allowlisted gaps only in estimated mode and labels
those values as unverified assumptions.

`sources.py` loads manifests once per process and verifies upstream bytes against
reviewed hashes. Callers keep backend-specific rules, cache scope, and error
context. Treat loaded manifests as immutable; restart the campaign after editing
them. Manifest contents and recorded filenames are unchanged by their folder
location. The campaign hashes Python and JSON files recursively, so moving a
manifest cannot silently exclude it from driver provenance.

Prediction workers import the small projection modules; source resolution runs
in the parent before those workers start.

Keep backend rules separate: similar knob names can have different release,
hardware, and runtime conditions. Keep these groups one level deep; further
nesting would add import churn without simplifying those rules.

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
  Production runs do not upload that directory. Explicit PR preview runs retain
  it as a separate diagnostic artifact for local review. For policy
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
lock is installed by Pages, Full CI's repository-contract shard, and the nightly
campaign. Install it locally before running these tests:

```bash
python -m pip install --require-hashes -r scripts/e2e_accuracy_source/requirements.txt
```

On the September 28 dump, the new policy matches gym's **2,281 measurement IDs**
exactly. A live differential source check matched complete deployment and
evidence objects for configs **202, 618, 909, and 1553**, including the explicit
unresolved-knob outcome for 618. These are input-parity checks, not a latency
qualification or a refreshed public accuracy snapshot.

## Runtime evidence and research coverage

CI defaults to `--configuration-mode estimated`, matching the research preview.
Use `--configuration-mode verified` for source-evidence-only coverage. Both modes
apply measured overrides and verified defaults first. Estimated mode then fills
missing server/workload controls using `coverage-experiment/1`, recording each
value with `historical_value_verified: false`. Explicit false, zero, and KV dtype
values survive. An unresolved `auto` KV dtype receives a separate assumption.
Missing recipes can use observed database topology and an explicitly assumed
model mapping; ambiguous topology, corrupt evidence, quantization conflicts,
and unsupported mappings still block resolution. Predictor failures stay visible.

The assumptions and topology/model mapping follow `research_defaults.py`,
`mapping.py`, and `filter.py` under the gym source directory above at immutable
revision `a5862f9d0e516fd920c130dfa8d6828c7ac319c4`. The original repository is
[aisim-e2e-gym](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-e2e-gym).
They fill preparation inputs, never the replay request/token loop.

`--fetch-runtime-evidence` downloads public InferenceX artifacts and fixed
run-attempt logs before resolution. The reviewed `runtime_agg_index.json` and
`runtime_disagg_index.json` are unchanged copies of
`benchmark_results/runtime-evidence/db-dump__2026-09-14/{agg,disagg}_index.json`
at that gym revision. Downloads enforce recorded SHA-256 hashes, size limits,
and fixed repository API routes. Expired or inaccessible archives are recorded
as unavailable; checksum mismatches remain errors. `runtime-fetch.json` records
the download outcomes in private campaign evidence.

`runtime_observations.json` preserves 136 parsed deployment records and 36
workload-only records from retained,
checksum-verified copies of those artifacts. These are serving/workload facts,
not cached predictions or latency values. `freeze_records(points, source,
read_deployment_recipe)` generates the file through the existing identity,
benchmark-match, checkout, and job-log checks, with `source.archived_runtime`
disabled. Inputs are the selected September 28 cohort and the retained source
cache. Regenerate with those original artifacts; do not edit records manually.
Each record binds the complete input `SiliconRow` by SHA-256, including measured
metrics, so another measurement cannot reuse it. Available raw artifacts take
precedence. Archived facts explicitly record
`historical_artifacts_revalidated: false`: CI did not revalidate expired raw
artifacts, but the archived values were validated during generation.

The full cohort resolves to 964 verified and 1,107 estimated candidates, with
210 unresolved. Summary counts and each published point label their evidence
quality. Coverage parity does not promise equal predictions across predictor
revisions; fresh replay outcomes and errors determine the report.
