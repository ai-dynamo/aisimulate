# Per-layer expert popularity consumer (Stage 4)

Depends on Collector [#89](https://github.com/ai-dynamo/aisimulate/pull/89).
This consumer connects its bundles to **DeepEP-LL dispatch, combine and expert
compute**. It does not change the collector or WideEP measurements.

## Selection

CLI flags, Python `ModelConfig` / estimate API and Task YAML share:

| Setting | Meaning |
| --- | --- |
| `moe_routing_mode: auto` (default) | Valid measured profile, otherwise the model's existing distribution |
| `moe_routing_mode: uniform` | Existing balanced Monte Carlo semantics |
| `moe_routing_mode: random` | **Random-input measured**; missing compatible measurements are errors |
| `moe_routing_mode: power-law` | Existing model-family alpha, or explicit `moe_power_law_alpha` |
| `moe_model_revision` | Optional exact bundle model revision requirement |

`random` does **not** generate new uniform-random expert probabilities. Alpha
must be finite and positive and can only accompany `power-law`. An explicit
legacy `workload_distribution` is honored; conflicting new settings are errors.
Neither setting enables or overrides `enable_eplb`.

```yaml
moe_routing_mode: power-law
moe_power_law_alpha: 1.2
```

```python
from aiconfigurator_core.sdk.config import ModelConfig

config = ModelConfig(moe_routing_mode="auto", moe_ep_size=4, moe_tp_size=1,
                     attention_dp_size=4, num_gpus_per_node=4,
                     moe_comm_backend={"context": "deepep_ll", "generation": "deepep_ll"})
```

The native simulation `EngineConfig` JSON/YAML also accepts these flat fields.
`AicEngineBuilder.moe_routing(mode, alpha, revision)` forwards them through
Python compilation into the serialized Rust graph. Native callers choose
communication independently with `moe_comm_backend` (a context/generation map)
or the builder's corresponding method. Distribution selection does not switch
an existing fused graph to LL. Without LL selection, native `auto` records the
unsupported-backend fallback. CLI/Task retain their existing coverage resolver.
Compiled engine `extra.moe_routing_provenance` retains the selection record;
invalid profiles surface as hard native configuration errors, not a
best-available regression fallback.

Bundle selection uses the canonical model ID or an explicitly declared alias,
not name similarity or a guessed quantized-checkpoint relationship. It checks
revision (when requested), layer count, expert count and Top-K. Without a
requested revision, provenance records `bundle_pinned` and its immutable
revision; the bundle's collection checkpoint remains distinct from canonical ID.
A missing model is ordinary fallback. A present damaged or incompatible bundle
is an error, including a missing file inside an existing bundle.

Same-phase measurements are preferred. **By explicit design decision, `auto`
permits prefill measurements as a decode proxy**, labeled
`measured_prefill_proxy`, retaining both phases in provenance. This differs from
the reviewer's original strict phase-coverage requirement. It does not establish
decode accuracy. Strict `random` does not permit this proxy; constructing a
two-phase model with prefill-only data therefore requires `auto`, not `random`.
Other communication backends (and whole-model FPM) keep their previous behavior
under `auto` with `unsupported_consumer_backend`; `random` rejects them.

## Numerical path

Python validates and expands aggregated LL operators into separate MoE layers.
Dense layers are absent. Expert vectors retain **expert ID order**, with
contiguous expert-to-rank placement. The old aggregate layer scale is removed
before summing individual layer costs. Other operators and shared/overlap
structure remain in place.

Rust holds the measured marginal vector fixed. For each trial, randomized
integer quotas sum to `global_tokens * top_k` and each expert receives at most
`global_tokens`. The existing bipartite route constructor realizes quotas with
distinct experts per token. No power-law weights are sampled, and no hottest-rank
rotation is applied. Communication uses the same quota RNG/seed as compute.

The LL model retains startup calibration, exact-topology curves and NVLink/IB
endpoint bounds. Measured trials are averaged; the explicit legacy power-law
path retains its existing P50 statistic. Compute queries the balanced curve at
`ceil(max_rank_assignments * EP / top_k)` for each trial, then averages latency.
Repeated coordinates share one curve lookup. This avoids applying skew twice.
Compute uses the exact attention-sharding/globalization convention of its
paired communication operators.

Balanced compute calibration first uses a shape-matched
`moe_expert_compute_perf.parquet` curve, then a shape-matched `moe_perf.parquet`
curve. Both require `balanced` (not a uniform or power-law substitution), the
same quantization, dimensions and EP. Missing calibration remains a typed
performance-data error. Latencies are labeled **estimated**, not silicon.

Content bits/digest, layer, contiguous topology, tokens, seed and simulation
parameters participate in the native communication cache. The compiled engine
identity includes routing provenance. `model.moe_routing_provenance`,
`summary.get_moe_routing_provenance()` and estimate result
`moe_routing_provenance` expose mode, phase/proxy, revisions and profile digest.

TODO(EPLB): future placement, expert replication and dynamic rebalancing are
separate strategies, not alternative names for probability selection.

## Accuracy gate and held-out evidence

Run `tools/moe_routing_accuracy.py observations.json --output report.json`.
It invokes the production Rust operators twice (model-family power-law and
measured marginals) and reports dispatch, combine, compute and summed-latency
MAPE for **every** model. It exits nonzero for missing models, failed samples,
per-model total-MAPE regression or overall regression. The required first set is
DeepSeek-Coder-V2-Lite-Instruct, DeepSeek-R1 and GLM-5.2.

Each observation is a JSON record with:

- `model_id`, pinned `revision`, `layer_id`, `phase`, `num_experts`, `top_k`,
  `per_rank_tokens`;
- `model_config`: `tp_size`, `attention_dp_size`, `moe_tp_size`, `moe_ep_size`,
  `num_gpus_per_node`, and quantization settings matching the timed kernel;
- `database`: `system`, `backend`, `version` (and optional `systems_paths`);
- `route_file`: JSON array `[global_token][top_k]`, in source-rank token order;
  `route_sha256`, held-out `workload_sha256`, and nonempty
  `bundle_workload_sha256s` from the collector campaign manifest;
- `measurement`: `kind` (`routing_replay` or `native_decode`),
  `kernel_revision`, `gpu_type`, `driver`, `timing_method`,
  `placement: contiguous_expert_id`, and positive `latency_ms` values for
  `dispatch`, `combine`, `compute` under identical conditions.

Keep raw route/workload files, timing logs and infrastructure identity private.
The manifest's independence and timing attestations must be audited against
those originals; a schema checker cannot establish experimental independence.
Routes sampled from the measured probabilities are not independent evidence.
Prefill route replay proves only the replay scenario. Decode accuracy requires
native decode observations. The report never converts absent evidence to PASS.

### Current acceptance status

The implementation has deterministic selection/serialization/numerical tests;
these are **correctness tests, not accuracy measurements**. Independent paired
kernel timings for the three required models are not yet available. Existing
aggregate recorder outputs cannot recover token co-selection or provide the
three component timings. The PR must remain **Draft** pending this gate.

A local production-operator preflight at GB200 / SGLang 0.5.14 / EP4 resolved
all three measured components for R1 and GLM-5.2. Coder Lite's graph builds, but
its 2048-hidden / 1408-intermediate / 64-expert / Top-K-6 BF16 shape lacks both
LL communication and balanced compute calibration at that coordinate. These
are typed data misses, not accuracy results. A separate new-seed Coder Lite
diagnostic route capture completed successfully; it does not time the three kernels.
The held-out canary used four new workload seeds and two repeats: 4,604 prompt
tokens per repeat, 24 requests per repeat, and 48 raw per-request route arrays.
All repeat counts matched exactly. Raw requests, routes and validation evidence
remain private; the packaged training bundles are unchanged.

Remaining review/acceptance gaps are explicit: paired held-out kernel timings
for all three models (and the missing Coder Lite calibration); decode-specific
accuracy; and reporting the exact executed balanced compute donor table rather
than only the strict calibration policy. The strict `random` two-phase
construction limitation above also remains until phase-scoped construction is
available. None of these is masked by a synthetic accuracy result.

Marginal probabilities cannot recover expert co-selection, correlations across
tokens or all router constraints. The joint routing distribution remains a
simulation assumption. Neither measured marginal reuse nor the prefill proxy
is a promise of improved accuracy before held-out validation.
