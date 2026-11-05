# Config Adapter

This package converts supported server configurations into validated,
versioned AIC estimate requests. Adaptation is deterministic and does not run an
estimate or execute commands from source recipes.

For the full contract and mapping details, see the
[config adapter guide](../../../../../../docs/aic-backward-compatibility/configuration-import.md).

## Public API

Import public types and functions from `aisimulate.sdk.config_adapter`:

```python
from pathlib import Path

from aisimulate.legacy_cli.api import cli_estimate
from aisimulate.sdk.config_adapter import (
    AdapterOverrides,
    DynamoRecipeSource,
    adapt_config,
    to_cli_estimate_kwargs,
)

report = adapt_config(
    DynamoRecipeSource(
        deployment=Path("deploy.yaml"),
        performance=Path("perf.yaml"),
    ),
    AdapterOverrides(backend_version="0.19.0"),
)

for outcome in report.outcomes:
    if outcome.status == "adapted":
        result = cli_estimate(**to_cli_estimate_kwargs(outcome.request))
```

Call `cli_estimate` only when estimation is explicitly requested. Adaptation
itself never invokes it.

## Supported sources

- `InferenceXSource`: one DB-export config record and one benchmark operating
  point.
- `ResolvedInferenceXSource`: an evidence-qualified `resolved-deployment/1`
  object, matching config/benchmark records, and an immutable source reference.
  Source acquisition remains outside the adapter. The caller may supply a
  verified local checkpoint through `AdapterOverrides.model_path`.
- `DynamoRecipeSource`: standard aggregate or prefill/decode-disaggregated
  DynamoGraphDeployment YAML, optional performance YAML, and concrete
  `dynamo-ci` benchmark recipes.

The adapters support vLLM, SGLang, and TRT-LLM. Unsupported or ambiguous
topologies are returned as rejected outcomes with structured diagnostics.
For Dynamo performance Jobs, literal `CONCURRENCIES` values are expanded in
order and shell comments are ignored during command parsing. TRT-LLM engine
ConfigMaps are inspected only when their volume mount resolves the exact engine
argument path. Conflicting active depths are rejected, zero depth stays
disabled, and active speculation requires a caller-supplied `nextn_accepted`.

## Contract

Every adapted request uses schema version `aic-estimate-request/1.0.0`.
`EstimateRequestV1.schema_path()` locates the packaged JSON Schema snapshot.
The Python model remains authoritative for cross-field validation.

Adaptation follows these rules:

1. Explicit overrides take precedence.
2. Unambiguous source values are used next.
3. Documented source-specific defaults are used last.
4. Every discovered operating point produces one ordered outcome.
5. Missing or conflicting required values are rejected; no point is silently
   dropped.

## Resolution strategy: source to replay

Source resolution and replay configuration are both required for serving
configuration parity with e2e-gym. Public accuracy CI now uses
`gym-resolved-config-v2`: the shared evaluator resolves source evidence before
calling either predictor, carries per-role replay settings and source workload,
and records estimate and replay outcomes independently. Historical campaigns
retain their original `latest-complete-config-run-v1` policy and fixed settings.
See the [resolver contract](../../../../../../scripts/e2e_accuracy/source/README.md)
for the pinned gym revision, validation evidence, and remaining modeling limits.

### Source resolution

- Resolve each benchmark's matching recipe, launcher, and available runtime
  evidence. Pin source revisions and artifact hashes; a recipe fingerprint
  alone cannot reconstruct its settings. Fetch and cache artifacts outside the
  deterministic adapter; never execute source shell commands.
- Resolve settings separately for aggregate, prefill, and decode workers.
  Preserve explicit overrides, source values, and verified backend/version
  defaults with their provenance. Reject unresolved conflicts. Keep source
  framework/image versions separate from predictor performance-database versions.
- Track model/checkpoint and quantization identity alongside serving settings.
  Missing evidence must stay explicit. Verified mode never substitutes an
  unverified default. CI's separate estimated mode applies the documented
  `coverage-experiment/1` assumptions and labels affected configurations as
  estimated, including any interpretation of `auto` KV dtype. It cannot
  override conflicting evidence or unsupported mappings.
- Restore runtime artifacts with hash and run-attempt checks. When upstream
  artifacts expire, reviewed archived settings may be reused only for the
  identical measurement hash; provenance states that raw artifacts were not
  revalidated in this run. See the resolver contract for generation and sources.

### Replay configuration

- Carry the resolved deployment into the final per-role replay engine arguments: sequence limits, batched/prefill token
  budgets, prefix caching, memory fraction, KV dtype, and chunked prefill.
  Keep graph controls in source evidence when the engine cannot model them. Preserve the workload and worker topology with those settings.
- Replace CI's fixed settings only when the corresponding source setting or
  verified default is resolved. A documented replay approximation must remain
  distinguishable from a source-matched configuration. Published reports carry
  verified/estimated counts and a quality label on every operating point.
- Validate each setting against the evaluated wheel and backend. If a required
  setting cannot be represented or modeled, report an explicit unsupported
  outcome instead of silently dropping it or claiming parity. Existing estimate
  schema fields do not by themselves prove replay support.

### CI validation

- **PR checks:** compare resolved values and final per-role engine arguments
  with reviewed, pinned gym fixtures for vLLM, SGLang, and TRT-LLM. Include missing,
  conflicting, and unsupported settings. Keep these checks CPU-only and usable
  without access to the internal gym repository. Preserve fixture attribution.
- **Nightly accuracy:** resolve and cache pinned source artifacts, then exercise
  the complete adaptation-to-replay path with supported baseline and candidate
  wheels. Record configuration provenance and account for every excluded point.
- Claim parity only for matched measurement IDs, resolved settings, workload,
  predictor/data revisions, and metric boundaries. Configuration parity and
  measured prediction accuracy are separate results.

The new cohort follows gym's row deduplication, image coherence, and 180-day
configuration freshness window, including P/D and multinode evidence. Missing
runtime settings remain explicit exclusions. Graph/kernel controls outside the
modeled replay API, unmodeled client behavior, and unknown historical checkpoint
revisions remain visible in the local evidence; matching gym inputs does not
establish that every source behavior is simulated.

See the [accuracy audit](../../../../../../pages/e2e-accuracy/README.md#adapter-parity-audit-2026-10-02)
for validation evidence and modeling limits.

## Package layout

| Module | Responsibility |
| --- | --- |
| `schema.py` | Canonical request, override, report, and diagnostic models |
| `api.py` | Public dispatch and lowering to `cli_estimate` keyword arguments |
| `inferencex.py` | InferenceX record adaptation |
| `resolved.py` | Source-resolved InferenceX estimate adaptation |
| `dynamo.py` | DynamoGraphDeployment and performance YAML adaptation |
| `dynamo_ci.py` | Concrete `dynamo-ci` recipe adaptation |
| `schemas/` | Language-neutral JSON Schema snapshot |

Only the Python package and canonical schema belong in the unified wheel. Agent
skills, fixtures, datasets, reports, and gap-analysis infrastructure remain
repository-only.

## Tests

Run the focused suite from the repository root:

```bash
uv run --extra dev pytest -q \
  tests/unit/sdk/config_adapter \
  tests/integration/test_config_adapter_estimate.py
```
