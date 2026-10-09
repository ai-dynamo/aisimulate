# FPM accuracy evaluation

Development-only evaluation code adapted from NVIDIA AISim FPM Gym at
`e8221729db2802e822f6919fd68bc2941743385b`:
https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-fpm-gym/-/tree/e8221729db2802e822f6919fd68bc2941743385b/src/aisim_fpm

Licensed under Apache-2.0 with maintainer-confirmed migration permission.
See the root THIRD_PARTY_NOTICES.md and LICENSE.

HF manifest/protocol loaders, rank-aware measurement types, FPM staging, and
worker-isolated regression adapters retain upstream behavior. Imports were
renamed to `fpm_accuracy`; the op-based adapter and experimental registry were
removed. Unused presentation metadata and its conversion helpers are omitted;
the worker schema and MoE mapping required for evaluation are retained.
Unsupported measurement protocols remain visible as unsupported
configurations; missing protocol identities and corrupt inputs still fail closed. `evaluate.py` reduces each shared measurement stream directly into
overview and per-variant heatmap aggregates. Optional notification evidence retains observation-order hashes
and compressed percentage-error sequences separately from public Pages artifacts; it does not retain
raw measured or predicted latencies. See [daily reporting](../../docs/ci/accuracy.md#daily-accuracy-report).
The public `skipped_count` combines excluded and unavailable source observations;
the Overview labels this count “excluded or unavailable.” It does not mean
that all of these observations were deliberately filtered out.

Native AISim imports remain deferred in the adapted adapters so parser and fake
predictor tests work without an installed native extension. The real campaign
checks that the native SDK is installed before evaluating any case.
When the evaluated wheel provides `ForwardPassPerfModelConfig`, both FPM and
regression use `best_available(config)`, with explicit mode, worker identity,
and migrated tuning options. Native FPM retains its compatibility adapter for
older branch wheels that do not provide the canonical configuration type.

Gym regression selects the measured signed 4×1 lazy configuration through
`estimator_config`: attention/MoE features and retention axes, 64 observations
per store, minimum five observations, ridge 1e-9, no scheduled rebuilds, signed
coefficients, and lazy updates with 1% relative / 0.1 ms absolute tolerance,
window 8, trigger 2, cooldown 1, and startup 10. Explicit adapter options for
capacity, minimum observations, ridge, rebuild interval, or legacy bucket grid
remain supported. An explicit canonical `fpm_regression.sampling` block retains
its Rust-resolved axes, grid, and capacity, including an explicitly chosen 4×4
grid; the Gym grid applies only when neither sampling nor legacy grid options
are supplied. This selection is local to the Gym regression adapter; the
shared AISim 4×4 eager defaults and AgentX/ShareGPT/LongBench recommendations
are unchanged. See the [measured configuration](../../benchmarks/evidence/fpm-regression/README.md).

Rust normalizes and validates this configuration before construction. Wheels
that cannot represent or preserve its signed/lazy controls report regression
as unsupported, with all measurements retained in the coverage denominator.
The evaluator never substitutes an older regression policy. Resolved native
diagnostics retain the complete estimator configuration, and each public result
records the evaluator commit. Historical results keep their original policy.

Hub cache loading supports repository-local blobs and the marked cache-wide
shared blob store used by huggingface-hub 1.32. Manifest hashes still bind the
measurement and FPM bytes; arbitrary symlink targets outside these stores are
rejected. Local dataset checkouts retain their strict root boundary. Catalogs,
configuration and measurement manifests, and FPM sidecars share the strict
public-contract JSON parser: duplicate keys (including nested keys) and
non-finite constants fail even when the pinned bytes match their hashes.

Daily campaigns load only current configuration snapshots, including all their
eligible FPM variants and hash-verified measurement evidence. Unrelated archived
snapshots do not gate the current overview. Explicit history reads still require
the recorded manifest hashes and validate all historical snapshots; their cache
is separate from current campaign membership. Optional override files retain
full-catalog selector validation, including historical bindings.

Schema v7 FPM pairs retain hash, row-count, base-configuration, and internal
sidecar/Parquet consistency validation. Selector flags must be booleans, the
model-config hash must be a lowercase SHA-256 (or the producer's empty legacy
identity), and other execution fields must be nonblank strings. This does not
establish a binding to an authoritative execution identity in the selected
configuration. Every v7 pair is therefore rejected by native staging, even if
its sidecar and rows agree. Supporting v7 prediction requires that binding and
an adapter that carries the full identity; this change does not add either.
Measurements still participate in coverage and worker regression, which does
not consume FPM pairs. Unknown schema versions and corrupt pairs fail closed.

Decode context parallelism (`dcp`) is a separate identity dimension from `cp`.
Legacy manifests, sidecars, and parquet files without `dcp` mean `dcp=1`;
non-default values must agree across all three. Native FPM currently has no
DCP input, so `dcp>1` is reported as unsupported with measurements retained in
coverage. Worker-isolated regression continues to score the same observations.

Listener window and single-rank chronology keys use milliseconds so mixed
streams preserve predict → score → tune ordering. Missing MoE parallelism
defaults to one; malformed values and unknown recorded precisions fail closed.
Worker regression verifies required store names before scoring and reports
contract mismatches in Actions logs while retaining measurement coverage.

The `fpm-accuracy` group in [`../pyproject.toml`](../pyproject.toml) declares evaluator dependencies. `requirements.txt` locks
their transitive dependencies and distribution hashes for Python 3.12; CI
installs it with `--require-hashes`. Regenerate with:

```bash
uv pip compile --group scripts/pyproject.toml:fpm-accuracy --generate-hashes \
  --python-version 3.12 --universal \
  --output-file scripts/fpm_accuracy/requirements.txt
```

Install test tools such as pytest separately from the hash-checked campaign
requirements; they are not part of the daily evaluation environment.

The selected AISim wheel and its runtime dependencies are installed separately
because evaluated branches can declare different runtime requirements. The
campaign checks the wheel hash and runs `pip check` after both installs.

## Dashboard migration

The files under `dashboard/` adapt the corresponding
`src/aisim_fpm/dashboard/{data,measurement_heatmaps,visualization,visualization_diagnostics}.py`
from [Gym commit f934c030afc3a03cb04d8f3ff4709194f7445c98](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-fpm-gym/-/tree/f934c030afc3a03cb04d8f3ff4709194f7445c98).
Copyright NVIDIA CORPORATION & AFFILIATES; Apache-2.0, under the existing
maintainer-confirmed migration permission. Imports were renamed, unused data
contracts and the GitLab publication helper removed, and publication moved to
qualified GitHub artifacts. Tests in `tests/fpm_accuracy/test_visualization.py`
adapt the upstream file of the same name; dashboard fixtures are synthetic
outputs from those test cases.

Measurement bins and native-rank axis semantics are preserved. Error heatmaps
share measured bins and accumulate alongside prediction, without a second
scoring pass. Only the selected variant contributes to Overview and Trends;
Slice Detail retains every evaluated variant. Measurement-only 3D assets are
produced once per pinned campaign by `scripts/fpm_accuracy/prepare_fpm_measurements.py`.
That script traverses current HF snapshots, matching branch accuracy.
Archived source snapshots do not gate fresh current measurement publication. See the public
[dashboard README](../../pages/fpm-accuracy/README.md) for storage and rollout.
