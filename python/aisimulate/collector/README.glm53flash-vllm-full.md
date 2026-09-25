# Native V2 FULL graph Ops evidence

`glm53flash_vllm_graph_export.py` connects original V2 FULL capture registries,
CUPTI clone callbacks, unique per-forward traces, completed request histories,
and separately executed logits to the common graph exporter. Its native contract
is vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`; the exact installed runtime,
source closure and worker binaries remain subject to the shared native reader.
The stock version and the precisely admitted IndexPool repair remain distinct.

Schema 2 graph policy includes only the actual initialized `full_graphs` region.
The complete source-validated descriptor and candidate snapshot remains hashed
in the raw evidence. Native NONE and PIECEWISE seed forwards are validated
against that snapshot, but supply no FULL target costs. A PIECEWISE-only capture
bucket is not FULL coverage. SGLang retains its separate schema 1 contract.

Each target binds its rank, invocation, native run, request set and sampling role
to one original trace. Original captured node ownership and completed zero-node
boundaries are checked again. GPU activity outside the replay is joined through
CUDA launch correlations to the source-pinned logits scope or runtime setup;
missing device activity fails validation. One native whole-forward interval
selects a coherent TP rank per repetition, then that rank's disjoint operation
activity unions are composed as an explicit additive approximation. Setup is
charged once per decode step. No cost comes from a whole-forward residual.

The common `export_graph` entry requires an independent unprofiled control with
the same actual initialized policy and public native EngineArgs, excluding only
`seed`. It retains the timing difference without scaling measured units.
`bind_calibration` returns exact table, policy and evidence hashes. The strict
public consumer uses the calibration capture policy to select padding; holdout
traces can only reveal disagreement. Missing shapes, another runtime, and
geometry-losing mixed APIs reject. A native past length P remains the table axis;
the public static API supplies P and the internal inclusive RuntimeContext
supplies P+1. SOL and eager arithmetic are unchanged.

The tests use explicitly authored TEST_ONLY traces and tables, including the
complete production model's public query path. They do not provide GPU kernel
qualification, graph timing equivalence or independent MAPE acceptance. FULL
calibration/control/holdout GPU evidence and the separate PIECEWISE measurement
path remain required for complete native-serving coverage.

An explicitly frozen calibration can set
`AISIM_GLM53_PIECEWISE_CAPTURE_ONLY=1` before worker imports to qualify the native
PIECEWISE initialization alongside FULL targets. This records distinct segment
and eager-callable ownership artifacts; it does not enable PIECEWISE timing or
FULL table reuse. The default remains disabled, and controls/holdouts reject
this profiling opt-in. The native inventory records the chosen capture setting.

## Explicit bounded schema 3 serving lookup

Schema 3 has a separate native-serving reader for measured NONE, PIECEWISE and
FULL operations. To enable bounded analysis lookup, pass
`lookup_contract="vllm_serving_bounded_p_q_v1"` to
`glm53flash_vllm_serving_export.export_serving(..., control_root=..., control_run=...)`,
`glm53flash_serving_shards.publish_calibration(parent, children, destination, ...)`,
or the shared `publish_sharded_calibration(..., parent_run=parent, ...)` entry.
Use the original frozen run objects, complete calibration children and
independent controls; output must be a new canonical table path. The exporter
revalidates original evidence and adds only the analysis columns
`lookup_contract` and `source_ownership_sha256`. It does not change native policy,
runtime admission, producer loops or original timing receipts. Unknown contracts
reject before evidence writes. Schema 1/2 keep their original meaning; schema 3
without this opt-in retains its existing exact/same-dispatch P-only rule.

The Rust consumer first selects an exact point. Otherwise each of the 277 named
physical units and the one setup unit requires two enclosing measured endpoints:
nearest P brackets at fixed B/Q, or Q brackets at fixed B/P0. P0 initialization
cannot bracket a cached prefix. Runtime/checkpoint/precision/TP, actual policy,
source-call ownership, measurement method and existing per-unit state/layout
partitions must match. FULL and PIECEWISE keep the actual native descriptor and
padding; NONE physical tokens may vary with B*Q. Different observed kernel
fingerprints or contribution counts remain in the evidence without becoming an
equality requirement. There is no B interpolation, cached-P Q interpolation,
extrapolation, SOL fallback, whole-forward residual or substitution for a
missing unit. Independent controls and original ten-sample reduction still apply.

After compiling the public `EngineHandle` with that table,
`engine.glm53flash_lookup_audit("context", B, Q, P)` or
`engine.glm53flash_lookup_audit("generation", B, 1, P)` reports the same Rust
selection used by prediction. The output contains the native and graph policy
hashes and every unit's exact/P/Q choice, endpoint coordinates and weights,
original latency, measurement method, contribution count, dispatch fingerprint,
rank-selection hash, raw-evidence hash, policy-evidence hash and ownership hash.
`predict_homogeneous` retains these records in `prediction_evidence` for every
successful original holdout point. Sharded predictions additionally retain
`prediction_evidence_origins` with the child cell, native benchmark ID and
original parent point ID. Failed points remain error rows with no fabricated audit.

These analysis APIs do not establish GPU coverage or accuracy acceptance. Actual
PIECEWISE coverage depends on the initialized capture policy and measured
endpoints. Independent full holdout prediction and the existing accuracy gate
remain required; TEST_ONLY fixtures do not supply those measurements.

The serving reader parses each shared PIECEWISE callback receipt once per rank
within one capture-validation call. Every segment still requires the original
receipt hash, and all source, executable, node and complete-capture checks run
unchanged. File identity is checked on reuse and complete contents are hashed
again before returning validated captures. The cache never spans reader calls
and cannot reuse another run's evidence. This avoids repeatedly parsing the
same callback stream for every segment without changing recorded costs or
acceptance thresholds.
