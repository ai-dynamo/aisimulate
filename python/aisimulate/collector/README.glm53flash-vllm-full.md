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
