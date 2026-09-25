# GLM-5.3-Flash native operation measurements

This collector observes the framework's loaded text decoder while the serving
scheduler executes actual requests. It never constructs attention metadata,
fills synthetic KDA state, changes a native dispatch choice, or moves a collective
outside its native call. MTP, multimodal, EP and DP execution are outside this
initial contract. TP2 and TP4 are required; NVFP4 TP1 is optional after admission.

`glm53flash_contract.py` obtains the complete operation identities from the
production graph. The parquet key is `(component, geometry, batch_size, prefix,
x)`. Geometry is canonical sorted JSON with the display name removed, including
backend and checkpoint format. Context uses query length `x` and actual cached
`prefix`; decode uses past-KV length `x` and `prefix=0`. Token-only components use
total tokens `x`, `batch_size=1`, `prefix=0`. Latency is milliseconds; integer
columns are physical INT64 constrained to uint32. The Rust reader admits exact measured keys, or bounded linear interpolation
with every required corner measured under the same actual CUDA dispatch
fingerprint, runtime, graph policy and state mode. KDA initial-state and
IndexPool short-path/tail partitions cannot be crossed. Missing kernel evidence,
an incomplete interpolation cell or extrapolation is an explicit coverage gap;
there is no analytical fallback.

`glm53flash_native_hooks.py` binds all 45 loaded decoder layers to
`NativeOperationObserver`. The serving adapter must pass native request IDs,
prefix execution receipts and actual phase coordinates to `begin(NativeWorkload)`
before the real forward and call `end()` after it. `close()` restores every
original callable. These first hooks admit eager observations only; graphs must
use capture-bound events and replay evidence. Eager data cannot be labelled as
production graph data. Missing native invocations fail complete-graph coverage.

Attention includes local projections, KDA or pooled-index NoPE sparse MLA, cache
writes, and the local output projection. Separately observed synchronous
same-stream collective intervals are excluded; overlap or fused communication
must receive a separate qualified contract. SGLang's saved latent projection
may execute lazily inside attention. That explicitly declared same-operation
callback shares the enclosing attention interval and preserves its native
source attribution; a hoisted execution contributes a disjoint interval. It is
never counted twice. Different-operation nested compute remains an error. KDA conv/recurrent state and
the sparse indexer's incomplete pool tail must come from the real request's
native prefix execution.

mHC is measured at native boundaries, including the output RMSNorm: vLLM has one
pre, 89 fused post/pre and one post interval, plus expand/contract; SGLang has 90
pre and 90 post intervals plus expand/contract. If SGLang leaves RMSNorm outside
its pre call, the current hook rejects that path rather than undercounting it.
At the pinned SGLang revision, `hc_attn_pre` and `hc_ffn_pre` unconditionally
forward to the same `_hc_pre` implementation and `mhc.hc_pre` native operation
([source, lines 710–746](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/models/glm5_next.py#L710)).
The reader normalizes only these two entrypoints, with the exact source-manifest
digest and mHC-pre geometry required. Original entrypoint names remain in the
hashed raw records. This source equivalence does not supply a CUDA kernel
fingerprint: observations without one remain eligible for exact lookup only.
The pinned vLLM `hc_pre`, `hc_post` and `hc_fused_post_pre` similarly call only
their corresponding mHC CustomOp
([source, lines 557–619](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/glm5next/nvidia/model.py#L557)).
Their source witness includes that loaded CustomOp dispatch. Historical raw
witnesses also enumerated unrelated attention and FFN descendants of the owning
decoder layer. The reader retains those original records and narrows only these
three pinned forwarding identities to the observed callee, rejecting a missing,
changed or ambiguous callee instead of merging it.
The whole native FFN includes FP32 gate projection, sigmoid/top-k routing,
shared and routed experts, and clamp10. Its analytical children never query
generic measured GEMM/MoE rows. All45 FFNs are required on every rank.
The complete graph additionally observes embedding, final norm, the complete
native logits processor and 91 TP allreduces. Logits include native BS-row
selection, BF16 vocabulary all-gather and SGLang's FP32 output conversion;
communication is included in this boundary. Allreduces use separate primitive
rows and are excluded from enclosing local compute only with a same-stream,
synchronous interval witness. The observer rejects nested compute, asynchronous
collectives and missing occurrences. The graph has 277 vLLM and 366 SGLang
occurrences per phase/rank; decoder observations alone cannot certify it.

Raw rank JSONL records retain every layer occurrence, request/history identity,
sample, invocation and excluded collective. Publication first requires complete
graph occurrences on every rank. New collection uses
`whole_forward_slowest_rank_v1`: within each exact native forward, select the
rank with the largest recorded whole-forward interval (ties choose the lowest
TP rank), retain all its operation intervals, then apply the existing shape/layer
median. `rank-selection.json` preserves every rank's whole interval, the exact
forward/request join, selected rank and raw file hashes; the table binds its
digest. SGLang eager uses its native DeviceTimer; vLLM uses embedding-to-logits
GPU events, also recorded during calibration. No operation sum, fitted target
or residual selects or scales a rank. This is an explicit representative-rank
approximation, assessed against the independent 20% gate.

Historical `per_operation_tp_max_v1` tables retain their conservative behavior;
their maxima need not belong to one rank timeline. The reader rejects mixed
aggregation policies. Old vLLM calibration without whole-forward intervals
cannot be retrospectively relabelled with coherent rank selection. All original
raw records and failed attempts remain available. Distinct checkpoint formats, runtimes, graph modes, seed policies or
state modes never silently collapse onto the same physical key. Failed attempts
remain separate evidence. A collector success is not accuracy acceptance; the
formal Ops gate is phase/cell MAPE <=20% against independent whole-forward truth.

## Source-grounded interpolation partitions

KDA keeps zero-prefix and initialized-state workloads separate. Native tile
counts change the work grid and scratch size; they are not an arbitrary
`ceil(query/128)` dispatch boundary. vLLM's pinned
[FlashKDA implementation](https://github.com/vllm-project/FlashKDA/blob/b59532f1f464fbd536272780e30df5bf6a2ccc02/csrc/flash_kda.cpp)
uses tiles of 16 and has a real `use_vsplit` specialization depending on heads,
request count and SM count. SGLang's selected Triton implementation uses tiles
of 64 in [chunk_kda_fwd](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/kernels/ops/attention/fla/kda.py).
Observed CUDA kernel fingerprints must still match; changing native backend or
specialization is not admitted by this geometric rule.

For full prefill with no previous tokens, partial IndexPool tails of one, two
and three tokens use runtime masks in the same native kernels. The reader
separates no complete pool versus at least one complete pool, empty versus
partial tail, and the inclusive 2048-token short path. Cached prefill and decode
retain exact prefix/total residues modulo four. This follows native
[vLLM pool writes and tail seeding](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/sparse_attn_indexer_kpool.py)
and [SGLang runtime-masked tail writes](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/layers/attention/dsa/kpool_fp8_index.py).
These rules only establish interpolation eligibility. They do not establish
numerical correctness of native cached starts or the required independent
whole-forward MAPE <=20%.

## Stock vLLM cached-start qualification failure

GB300 native correctness probe 604200 tested the pinned vLLM mapping, prefill
compression and tail-seed APIs. One-shot writes matched the independent
uniform-gate pooling oracle in every case. Split execution matched for
P4096/Q3 and P4096/Q4, but P4097/Q3 at B1/B2 and P4097/Q4 at B1 produced an
incorrect pool1024. Tail contents matched. The preserved result, source hashes
and cache-snapshot hashes are in `docs/glm53flash-kpool-native-gb300.json`.
This diagnostic does not measure model quality or latency.

The native prefill helper gathers current-chunk K and assumes pool-aligned
starts; previous raw-tail reconstruction occurs only on the decode path.
Current stock-vLLM measurement admission conservatively rejects cached-prefill
P%4!=0 with Q>=2, including starts not individually proven wrong. Q1 follows
the native decode-threshold path and requires its own normal admission. The
collector preserves requested points and writes `qualification-failures.json`;
the reader rejects both rows and queries. It does not delete points, silently
round prefixes, substitute a kernel, or claim the complete requested matrix.
SGLang and explicit analytical SOL are separate contracts.

## Upstream integration sources

These are original observation adapters calling upstream implementations, not
copies of the model or kernels. The bindings and architectural interpretation
are based on these immutable sources, under Apache-2.0:

- vLLM, Copyright contributors to the vLLM project, revision
  `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (v0.30.0):
  [model.py](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/glm5next/nvidia/model.py),
  `attention.py`, `kda.py` in the same directory, and
  `vllm/distributed/parallel_state.py`.
- SGLang, Copyright 2023-2024 SGLang Team and SGLang contributors, revision
  `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` (v0.5.20):
  [glm5_next.py](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/models/glm5_next.py),
  `python/sglang/srt/models/deepseek_v2.py`, and
  `python/sglang/srt/layers/communicator_mhc.py`.

Checkpoint identities come from `config.json` and model cards at
`zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`
and `nvidia/GLM-5.3-Flash-NVFP4@09b04e5e74bca08ca8549fc736d4cdd8624bfde3`.
Their config/license attribution is maintained with the production model's
packaged configuration files. See the canonical root `THIRD_PARTY_NOTICES.md`.

## SGLang serving integration

In each native worker, call `collector.glm53flash_sglang_runtime.install()`
through the shared driver's spawn-safe native scheduler entrypoint. Set `AISIM_GLM53_TRACE_DIR` to an
attempt-private output directory and `AISIM_GLM53_PROVENANCE` to a JSON file
with the immutable row provenance. Optional `AISIM_GLM53_OPS_MANIFEST` enables
the eager operation observer; omission records native whole-forward telemetry
without module instrumentation.

`AISIM_GLM53_REQUEST_MANIFEST` points to a frozen JSON mapping with top-level
`request_set`, `dataset_role`, `corpus_sha256`, and `requests`. Each request ID
maps to `benchmark_id`, `repetition`, `sampling_role` (`warmup` or
`measurement`), `target_phase` (`context` or `generation`), `target_query`
(new tokens per request), `target_prefix` (past tokens per request), and
`target_batch_size`. Only an exact, complete native batch with real same-request
history receives `stage=measure`; all other forwards remain `stage=seed`.

Outputs are `forward-rank-N.jsonl`, `failed-rank-N.jsonl`, optional
`inventory-rank-N.json`, and operation observations `rank-N.jsonl`. Forward
records preserve the actual per-request query/prefix lengths, resolved input
tokens, prompt/output history, predecessor forward ID, native DeviceTimer
interval and category, actual graph mode and captured decode bucket. Reading
actual sampled token IDs synchronizes after the native timing interval; this
telemetry policy must be retained in comparison controls. Request host output
IDs may lag under overlap scheduling, so histories use the resolved native
input tensor and observed sampled IDs. Unknown piecewise graph padding rejects
a formal target rather than being inferred from its unpadded size.

Additional integration sources at the pinned SGLang revision above are
`python/sglang/srt/managers/tp_worker.py`,
`python/sglang/srt/model_executor/{model_runner,forward_batch_info}.py`,
`python/sglang/srt/model_executor/runner/{eager_runner,decode_cuda_graph_runner}.py`,
and `python/sglang/srt/utils/device_timer.py`. Their implementations are called
without modification; the wrappers and trace schema are original adapter code.


## vLLM eager worker integration

The pinned release selects `vllm.v1.worker.gpu.model_runner.GPUModelRunner`
(V2), as confirmed by actual job 604200. The source-checked import loader installs
`install_v2()` on that actual class. It observes the returned native `InputBatch`,
wraps the loaded model forward, and finalizes only after the separate native
`sample_tokens` call has executed logits. `execute_model` itself returns before
logits in V2. Dummy profiling runs are excluded. The legacy V1 adapter remains
explicitly separate. Actual worker activation receipts and native source hashes
are retained; import success alone does not qualify a GPU observation. The scheduler atomically updates
`AISIM_GLM53_REQUEST_MANIFEST` before each request cohort; workers reload it for
every real forward. Required files are `AISIM_GLM53_OPS_MANIFEST` and
`AISIM_GLM53_PROVENANCE`; output uses `AISIM_GLM53_TRACE_DIR`. Every cache tensor's
allocated dtype/shape/stride is preserved separately. The current adapter rejects
actual graph dispatch. This integration remains unverified until target-GPU
instrumented smoke and independent whole-forward accuracy checks pass.


## Population, calibration and independent acceptance

The public `glm53flash_module` route uses the dedicated framework family pins
(vLLM0.30.0 / SGLang0.5.20, GB300 ARM64 platform digests). Repository YAML emits
eight native campaign tasks per backend: two checkpoint formats, TP2/TP4, and
two phases. Those tasks contain 120 declared workload points per backend;
a targeted checkpoint has four tasks/60 points. There is no deduplication.
The registry remains `unverified=True`, so the ordinary scheduled queue is zero
until the native collection paths are qualified. These population counts are
not measured coverage.

The launch route requires `AISIM_GLM53_MODEL_PATHS` (a JSON file mapping the two
pinned Hub IDs to local snapshots), `AISIM_GLM53_INPUT_TEXT`, and the immutable
`AISIM_GLM53_RUNTIME_DIGEST`. vLLM also requires the allocation-local Dynamo
runtime and `ETCD_ENDPOINTS`. Actual checkpoint config, runtime source files and
GB300 device names are checked. Every frozen point requires five real warmups
and ten measurements; an exact requested query/prefix which never occurs under
native scheduling is a missing point and fails the campaign.

When `AISIM_GLM53_DISPATCH_PROFILING=1`, the final excluded warmup uses the native
PyTorch CUDA profiler to attribute launched kernel names to module/collective
ranges. Retained measurements run with the profiler stopped. Its raw kernel
lists and dispatch fingerprint are retained; missing attribution permits only
exact lookups. This profiler path requires target-GPU qualification separately
from the initial eager event smoke.

`calibration-evidence.json` hashes request/operation manifests, raw observations,
forward histories, source preflight and actual allocated state layouts. Table
rows preserve calibration role, corpus digest, request set and the evidence
file's digest. The acceptance adapter rechecks those files, joins each module
observation to its admitted native forward, verifies the declared rank-selection
policy and reaggregates the original intervals before accepting the table.
Content hashes alone are insufficient.

Independent `ops_holdout` runs disable module hooks and record one GPU event
window from embedding through logits. The separate native scheduler/DeviceTimer
interval is retained but never substituted as the Ops comparator. Device-event
elapsed windows can include host enqueue gaps and cross-rank synchronization;
they must not be described as summed GPU busy time. A separate ordinary Engine
control with no retained observer reproduced close native DeviceTimer/model-event
windows; this checks the timing boundary, not the cause of native latency. The common
holdout validator checks disjoint workload geometry and request/corpus evidence,
then calls the installed public strict Ops consumer with a 20% per-cell/phase
MAPE gate. No graph-mode Ops data or accuracy pass is claimed by this code.

## Bounded calibration shards

The shared shard manifest preserves each original point ID and complete native
5+10 repetition cohort. `glm53flash_shards.physical_ownership` computes each
physical key's lowest original point ID from the frozen manifest before timing.
All child shards must finish and retain separate native runs, request identities
and evidence receipts. Shared keys require compatible runtime, dispatch, state
and corpus; no measured latency influences ownership. Other observations remain
in their original raw files.

After `load_native` has admitted every child, call
`publish_sharded_calibration(children, frozen_shard_manifest, destination)` with
`children=[(child_frozen_run, child_native_receipt), ...]`. Child runs carry the
verified `original_point_ids` mapping. Publication writes the complete parquet
and its ownership/evidence sidecar, refusing an existing destination. The common
acceptance utility calls `bind_sharded_calibration` with the same inputs to
reaggregate original observations and reproduce ownership independently. Missing
shards or unsupported requested points prevent full publication and acceptance.


Worker hardware is part of the hashed state-layout receipt. Each worker records
its selected CUDA device's native name, compute capability, memory size, device
index and UUID when the runtime exposes one, together with its TP rank. Formal
Ops evidence requires GB300/sm103, binds all allocated cache tensor devices to
that selected device and rejects repeated physical UUIDs across ranks. A dataset
directory or launch flag cannot substitute for this receipt. Earlier immutable
smoke bundles remain historical and are not upgraded to formal hardware evidence.

## Native graph whole-forward diagnostic

`glm53flash_graph_observer.NativeFullGraphWindow` supplies a separate, bounded
vLLM FULL decode diagnostic. The caller supplies the actual native prepared
request receipt and captured graph object/descriptor; start is immediately
before `ModelCudaGraphManager.run_fullgraph`, and end is immediately after
`compute_logits`, before sampling. Completed native request/sample receipts
and exact padding are mandatory. Repeated dispatch, recapture, changed stream,
eager/piecewise relabels and a missing logits endpoint are rejected.

This window measures the complete native graph model interval, including any
in-graph metadata, plus logits. It does not observe constituent operations or
certify a Python module execution count during replay. It emits diagnostic
receipts only; the existing eager reader still rejects graph data. Graph Ops
accuracy and eager-calibration reuse remain NOT_EVALUATED/NOT_ADMITTED. Before
any reuse, actual kernels, physical inputs, runtime/fusion/communication policy
and complete operation coverage need an explicit equivalence certificate,
followed by the unchanged independent <=20% graph whole-forward gate.


The vLLM V2 observer activates serving measurements only after native
`Worker.compile_or_warm_up_model` returns successfully. At the pinned revision,
`gpu/warmup.py:355` intentionally executes scheduler-shaped initialization
requests without `dummy_run=True`; those original native calls execute outside
observation while the explicit worker warmup lifecycle is active. Native dummy
calls also remain unobserved. A non-dummy call outside that lifecycle before
successful readiness fails, as do missing request manifests or unknown serving
request IDs afterward. The historical 604896 initialization failure remains
preserved; this correction is not a GPU collection qualification.


Native SGLang FULL decode graph collection has separate experimental purposes
`ops_graph` and `ops_graph_holdout`. They require explicit FULL decode and
disabled prefill capture; captured Python ownership rejects torch.compile.
`glm53flash_sglang_graph_ops.py` installs at the real scheduler entry, before
model initialization captures any graph. It uses the complete existing native
operation hooks for all45 layers, whole FFNs, mHC, embedding/norm/logits and
collectives. Native capture warmups and ordinary eager seed forwards execute
unchanged. Capture hooks only enumerate existing CUDA graph nodes; they add no
CUDA event, kernel, dependency or stream wait. Native same-operation lazy
callbacks are included once; collectives own disjoint captured nodes.

`glm53flash_graph_nodes.py` binds those nodes to replay CUPTI `graph id`,
`graph node id` and actual launch correlation. `glm53flash_graph_hooks.py`
retains native shape/padding and full operation boundaries. Kernel launch
geometry, interval overlaps and in-graph setup nodes are kept explicitly.
Kernels, memcpy and memset require complete activity records; structural nodes
remain separately visible. For each operation, the raw diagnostic reports the
union of its owned activity intervals (including memory activity). It also
reports their explicitly approximate additive sum, global activity union and
interval envelope. These distinct values are never labeled a critical path.
Cross-operation overlap and PDL edges remain intact, and no whole-forward
measurement is distributed back into operation costs. Runtime preparation and
output-copy nodes retain `native_graph_setup` ownership; they cannot disappear
from a future complete measured prediction. Future graph admission also needs
actual active batch/padding, runtime/phase/cache/capture identity and the same
independent 20% whole-forward gate, without changing the SOL theoretical graph.
Missing nodes, recaptured/unregistered graphs, child graphs and unknown fusion
scopes fail. Native selected requests/sample histories and hardware/state
receipts are still required from the shared retained lifecycle.

The graph calibration path uses CUPTI; the separate graph holdout installs no
module hooks or profiler and measures the unchanged native FULL replay with
outer GPU events. Both start at native `DecodeCudaGraphRunner.execute`, before
`load_batch` prepares GPU buffers/attention metadata, and end after its model
replay returns logits, before sampling or observer readback. Preparation outside
the model graph must join its actual CUDA API correlations within this exact
execution range and has explicit `native_graph_setup` ownership. Additional
metadata glue graphs require their own native node registry; the initial
single-graph adapter rejects them. CPU enqueue gaps remain elapsed diagnostics,
not costs apportioned to operators. Its boundary is explicitly
`native_full_graph_metadata_to_logits_gpu_v1`. The current measured consumer
**does not admit these experimental raw graph observations**: native mechanism
qualification, profiling controls, complete metadata-node ownership/cost and
the independent <=20% whole-forward comparison are pending. Kernel sums are
not treated as whole-forward critical-path times. Existing eager admission is
unchanged and all8 deployment accuracy cells remain NOT_EVALUATED.

The original primitive external-event GPU606363 probe recorded identical kernel
launches/outputs and updated events, but outer median0.020992→0.030864ms (about47%
slower). It proves event mechanics, not low-perturbation native GLM timing. The
read-only node path is a separate implementation and preserves that failure to
establish equivalence.

CUDA API references (original bindings; no upstream code copied):
https://docs.nvidia.com/cuda/archive/13.0.2/cuda-runtime-api/group__CUDART__STREAM.html
(`cudaStreamGetCaptureInfo`, seven arguments including edge metadata),
https://docs.nvidia.com/cuda/archive/13.0.2/cuda-runtime-api/group__CUDART__GRAPH.html
(`cudaGraphGetNodes`, five-argument `cudaGraphGetEdges`, `cudaGraphNodeGetType`),
https://docs.nvidia.com/cuda/archive/13.0.2/cuda-runtime-api/structcudaGraphEdgeData.html
(eight-byte dependency metadata, retained including nondefault PDL edges), and
https://docs.nvidia.com/cupti/13.1.0/api/group__CUPTI__ACTIVITY__API.html
(`cuptiGetGraphId`, `cuptiGetGraphNodeId`). Actual loaded CUDA/CUPTI binary hashes
are recorded separately; API documentation is not substituted for runtime proof.
CPU607166 verified both frozen ARM64 containers export the CUDA13 names and
installed declarations above. Their actual `libcudart.so.13` SHA256 is
`7bdba2b5b08cbdc85203c41cc94598adedb1bcfea7cb574ca693ac73599e4e63`;
the selected CUDA13 header SHA256 is
`3f91d3f84f1aafb17cccc475803f9c489b6474f2a3bf0fbc75a5380821c966e0`.
The original CPU606857 failure remains recorded: the CUDA12-era
`cudaStreamGetCaptureInfo_v2` symbol is absent. This ABI correction neither
qualifies GPU graph timing nor changes any already frozen attempt. Other CUDA
major versions fail explicitly before calling the capture-query ABI.
The subsequent tiny GPU607527 probe reached capture but failed on the second
node-enumeration call with an empty graph: CUDA13 rejects a nonnull output array
with zero capacity. Empty node/edge snapshots now use the actual native count
query without submitting a zero-capacity array. Every later boundary queries
again, and count/fill inconsistencies or edges to absent nodes fail explicitly.
No model stage ran after that failed tiny prerequisite.
`glm53flash_graph_callbacks.py` binds capture nodes to the executable through
actual CUPTI resource callbacks. CUPTI13.0.85 passes `CUpti_ResourceData` to
resource subscribers; its `resourceDescriptor` carries `CUpti_GraphData`.
The independent ctypes declarations use the exact installed 24-byte wrapper
(descriptor offset16) and 56-byte graph descriptor. The loaded CUPTI library
must match SHA256 `a55e03ccab21830f5b9d1ca7a02ecd59c557e0d54c769a181ad1140a3cff8ac1`.
The subscriber is active only around native capture/instantiation and is
removed before Kineto starts. Callbacks perform only CUPTI ID queries and copy
scalars; original callback progress and source registries are retained.

Instantiation's source graph ID differs from the executable graph ID reported
by activity records. The resolver joins the actual creation callback and all
node-cloned callbacks by opaque native handles, verifies their original graph,
and requires a complete one-to-one node/type mapping. It does not infer IDs
from integer bit patterns or kernel order. Foreign-source nodes, clone chains,
and unknown structural node kinds fail. Native dependencies, including PDL
metadata, remain attached to the mapped nodes.

SG GPU611706 calibration rejected a source EventRecord node (native type7)
whose clone callback reported type0. The separate native CUDA13 GPU612570
experiment retained all original handles/IDs and queried after unsubscribe,
while source and executable graphs were alive: source types5/6/2/7 all matched;
clone API types were0/6/2/7, while every callback type was0. Thus neither the
callback field nor the internal clone API is a universal source-type authority.
The observer accepts only the demonstrated source EventRecord7/callback0 case,
and only after a successful deferred `cudaGraphNodeGetType` query returns7 for
that exact clone handle. Source type and callback bytes remain unchanged.
The full unique source/clone ID and handle mapping, exact qualified provider
hashes, closed subscription, live executable handle and query rc/type are
retained. The hashed query receipt is independently rechecked by the exporter.
All other mismatches, including Empty5/clone0, still reject. No potentially
freed source handle is queried after Torch capture returns, and no CUDA API
is invoked inside a callback. The original failed model attempt is preserved;
this narrow repair still needs a new full-model capture/profile qualification.

The wrapper ABI was checked against NVIDIA's original
`extras/CUPTI/samples/cuda_graphs_trace/cuda_graphs_trace.cu` in the immutable
CUPTI13.0.85 Linux SBSA archive (archive SHA256
`f6f34d534cce56f91b1496abf51be3b1559ba879985d34eb89c808004b77513a`,
sample SHA256 `0458254b6d8ada6c6db82492f14f6bf029f70cafaf97a07dfc6168c95dc312f8`).
No sample implementation is copied. Historical tiny GPU608141 failed from the
incorrect direct descriptor cast; GPU608687 captured callbacks but its earlier
resolver rejected the source/executable namespace difference. Both failures
remain unchanged. A separately hashed, independently reviewed offline resolver
mapped all2 nodes in both GPU608687 traces with unchanged launch inventories.
This is node-identity evidence, not qualification of native GLM profiling or
prediction accuracy; a fresh native mechanism gate and matched profiled versus
unprofiled GLM controls are still required.

Native SGLang source pin94602c9 additionally binds
`python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py`
and `python/sglang/srt/model_executor/runner/{shape_key,decode_cuda_graph_runner}.py`.
The latter
([execute/load_batch source](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py))
has SHA256 `55892739b9c577ae43a60d5d31eac53f81e2b4aeca57ef5368b9c881117889d8`.
The earlier experimental wrapper started within `backend.replay` and therefore
omitted metadata preparation; no model graph profile was admitted from it.
The corrected boundary has CPU interval/ownership tests and still requires
actual native-class/GPU qualification in a new immutable attempt.

The experimental `glm53flash_vllm_graph_ops.py` capture adapter covers the
corresponding native V2 FULL initialization boundary. Source:
https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu/cudagraph_utils.py
(`CudaGraphManager.capture` and `ModelCudaGraphManager.capture`). It wraps the
native forward factory and observes it only while the original FULL capture is
active. Original warmups and PIECEWISE execution remain unchanged. Installation
must occur before native graph initialization. Actual compiled model submodules
and cross-thread microbatch runners are rejected: compiled leaf implementations
inside otherwise uncompiled GLM units are left intact.
The actual offloader must be the exact pinned `NoopOffloader`: native capture
calls `join_after_forward` after the observed forward factory returns, and only
the reviewed noop implementation guarantees no unobserved tail work there.
Source at the same immutable revision: `vllm/model_executor/offloader/base.py`.

Native memory sizing precedes serving initialization. The pinned
`profile_cudagraph_memory` sets `_max_full_descs_to_capture=2` and an empty
`_capture_mem_samples` list, invokes the original capture, then discards its
temporary graphs and manager. Both observation paths recognize this exact
native lifecycle before installing operation hooks or retaining serving graph
objects. The sizing capture still executes once with unchanged arguments and
return value; its memory estimate and cleanup remain native. Missing or mixed
markers fail. The later serving capture must still pass the complete initialized
descriptor validation. A sizing graph is never serving-policy evidence, and
this CPU-tested distinction requires a new producer identity and GPU attempt.

This vLLM graph contains hidden states; its raw registry explicitly lists logits
as an uncaptured operation. Native output copies remain visible as setup nodes.
The capture adapter alone writes no measured table. Child graphs, changed graph
objects and incomplete capture observations still fail. CPU scope/identity
tests are not native GPU qualification.

The V2 adapter also records `vllm-graph-policy-rank-N.json` immediately after
native initialization. `glm53flash_vllm_graph_policy.py` binds the original
manager's capture sizes, complete descriptors, candidate priorities and native
breakable PIECEWISE entries, with the exact source hashes. Its ordinary Q1
decode contract rejects compiled model wrappers, speculative decoding, LoRA,
microbatching and other unsupported eligibility predicates. This is a policy
inventory, not measured PIECEWISE coverage. The native source is
[`_init_candidates` and `dispatch`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu/cudagraph_utils.py)
and the original
[`BreakableCUDAGraphWrapper`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/compilation/breakable_cudagraph.py).
The same three new tokens can select FULL with three decode requests, or
PIECEWISE for a prefill; physical padding never replaces the actual request
geometry. Out-of-range tokens retain the native NONE identity. No holdout
dispatch is used to populate this pre-request policy.

The separate `ops_graph_holdout` V2 purpose observes unprofiled native FULL
decode without installing Python operation hooks. After original
`GPUModelRunner.prepare_inputs` returns, it verifies the actual descriptor,
unpadded per-request tokens/KV history and physical padding against that
initialized policy. Its GPU window starts after observer input readback,
before native attention metadata/state preparation, and ends when original
`compute_logits` returns. Sampling and completed-token readback remain outside
the window. The same runner must complete exactly one replay of its original
registered graph. Native dummy/compile warmups are excluded by the existing
worker lifecycle. Seed forwards keep their actual NONE/PIECEWISE/FULL identity;
measured targets require FULL Q1 decode. This producer boundary has CPU tests;
actual native class/GPU qualification and the vLLM graph exporter/consumer
integration remain pending. It does not supply measured PIECEWISE operations
or replace the independent whole-forward error gate.

The separate experimental `ops_graph` calibration purpose combines the FULL
capture registry with actual CUPTI activity from that same metadata-to-logits
window. It retains one original trace per rank/forward and emits its graph
receipt only after the actual replay and native sampled-token completion.
Profiling starts before the first GPU timing event and stops after the final
event, before sampling. Neither independent control nor holdout installs this
profiler. Actual `LogitsProcessor.forward` calls get a CPU profiler range outside
the captured graph; their GPU activities are owned by their CUDA launch
correlations. The pinned source is
[`logits_processor.py`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/logits_processor.py),
SHA256 `6b0603d67b0c756253c2fdc882a3896d2e873a16e9aa2ef877aabca8d36bdb5f`.
The observer verifies its exact class, source, BF16 head policy, full vocabulary
projection and all-gather; the native local-argmax alternative is not covered.

The binder requires exactly one complete source-bound logits range, forbids
straddling launches, missing device activity and extra unregistered graphs, and
charges every remaining observed device activity to `native_graph_setup`.
Logits cannot be owned both inside and outside the graph. Unit interval unions
and cross-unit overlap remain explicit; CPU gaps and whole-forward residuals
are never allocated to operations. Initialization registries stay in separate
hashed files instead of being repeated in every forward. Failed native calls
retain their failed trace and original exception. This path currently has CPU
contract tests only: actual calibration-class/GPU qualification, vLLM
raw-to-table export and independent accuracy are pending. Measured PIECEWISE
coverage remains a separate required implementation.

`glm53flash_vllm_piecewise.py` adds an optional capture observer, enabled only
with `install(..., include_piecewise=True)` during adapter qualification. The
current FULL producer keeps this option off. It observes the original
[`BreakableCUDAGraphWrapper` and segment APIs](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/compilation/breakable_cudagraph.py),
SHA256 `3cc427612a08e2b9b3fee47548026400c1d0776e2d4747535e59ef5512bdf1e8`.
The native capture and replay methods still own every graph and eager call.
Physical operation identity spans segment transitions; captured nodes and
native eager callables retain their individual source and position evidence.
A measured replay can activate CPU scopes around the original eager callable,
without adding graph nodes or counting initialization work as measurements.

PIECEWISE source and bound files use distinct names from FULL registries. One
shared complete callback receipt plus actual per-segment executable identities
avoids repeating the full initialization callback stream for every segment.
Missing or changed callable identities, operation coverage, graph nodes, source
pins or clone evidence reject capture admission. This is capture infrastructure
with CPU contract tests: actual model/GPU qualification, replay activity binding,
table export and independent PIECEWISE accuracy remain required.


## Native graph measured consumer and dispatch policy

The separate `glm53flash_graph_perf.parquet` contract consumes complete physical
operation rows plus one `native_graph_setup` runtime row at each measured decode
point. Every row carries the same canonical source/config/capture policy,
calibration evidence and coherent-forward rank-selection receipt. The cost is
an explicitly approximate sum of each unit's disjoint-node activity union;
setup includes observed preparation and memory activity. It does not reconstruct
a critical path or distribute any whole-forward residual. No graph table is
included yet, and raw-to-table qualification and independent accuracy remain
pending.

`glm53flash_graph_policy.py` reads the actual initialized native decode runner,
all captured ShapeKeys and eligibility flags before timing. It verifies the
pinned runner sources plus the model source-manifest hash. Padding is selected
from that actual capture list, with the native disable-padding rule: the pilot
list `[1, 2, 4]` yields B3/pad4, whereas a deployment captured at every size1–32
yields B3/pad3. Unsupported variants, extra metadata graphs, torch.compile,
unknown flags and missing all-rank source/state/capture receipts fail. Holdout
observations never supply a missing prediction policy. A formal deployment
must collect and validate its own frozen capture policy.

The Rust consumer retains complete homogeneous one-token `RuntimeContext`
coordinates through scalar, per-op, compiled-spec and stride paths. A selected
graph file cannot fall back to an eager row. History interpolation keeps active
batch, actual padding, physical shape, kernel fingerprint, activity count and
IndexPool state/short-path partition fixed, requires measured brackets, and
rejects extrapolation. Legacy token-only queries, mixed-step composition and
aggregate decode telemetry with more than one request lack sufficient geometry
and explicitly reject graph data.

The append-only `Glm53Runtime` enum value46 is held by the model's operation list
once per phase. It contributes zero in SOL and eager execution; FULL mode must
query the measured setup row. Existing variant values0–45 and the numerical SOL
costs are unchanged. The 277 vLLM /366 SGLang physical operation counts exclude
this additional runtime marker. A decode stride charges the marker once per
represented step, consistently in scalar and breakdown output.

The native dispatch predicates above are independently expressed from
`sgl-project/sglang@94602c9c2b7cbdb8efd5c52802dac6a1c180089e`, original paths
`python/sglang/srt/model_executor/runner/{base_cuda_graph_runner,decode_cuda_graph_runner,shape_key}.py`
and `python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py`
(Apache-2.0, SGLang Team and contributors). No native compute is copied or changed.


Graph queries preserve `RuntimeContext.s` as the inclusive current decode
position; their native past-KV key is checked `s - 1`. Static prediction's
`isl + step + 1` therefore maps to actual past `isl + step`. The observed B1
telemetry bridge explicitly converts its past-only total with checked `+1`
before entering that same generation walk. This matches pinned SGLang
`ScheduleBatch.prepare_for_decode` in `srt/managers/schedule_batch.py:3471–3473`,
which increments native sequence lengths before the current token, and the
observer's `actual_coordinates` subtraction. No collected table key changes.
Inclusive position131072 means past131071; position0 is invalid. Scalar,
breakdown, diagnostics and stride use this common boundary; aggregate B>1
telemetry still cannot prove per-request histories and is rejected.

Graph-backed compiled specs, including older schema20 payloads, must contain
exactly one direct generation setup marker matching every generation unit's
backend/checkpoint/TP/phase identity. Missing, duplicate, nested, context-phase
or rebound markers are rejected at Engine construction and generation-session
execution. Recompile an old GLM spec before selecting graph data. SOL/eager
specs remain compatible without the marker.

### Reproducible graph export and independent controls

`glm53flash_graph_export.export_graph(root, frozen_run, output,
control_root=..., control_run=...)` exports a new
`glm53flash_graph_perf.parquet` only after the shared native loader verifies
real request/token/state continuity on every rank. The frozen run explicitly
sets `spec.ops_execution_mode="native_full_graph"`. Its calibration role retains
the profiled run; `control_run.role="control"` identifies a separate unprofiled
run over the same calibration corpus, token histories and geometries. Control
requests and run IDs must be distinct. Independent accuracy uses a third run
with role `holdout` and separate frozen corpus/geometries.

SGLang eager and graph acceptance compare the complete actual resolved
ServerArgs across calibration, unprofiled controls, independent holdout and
shards. Only the native per-process `random_seed` is excluded. Different memory,
kernel or scheduler settings are rejected before export/prediction, with every
original requested holdout point retained. Reports contain the normalized digest
and original file receipts; private configuration values remain outside reports.
If an FPM-authored plan explicitly requests `sglang_mem_fraction_static`, Ops
also requires the same value in its selected frozen cell and both actual native
declared/resolved receipts. An omitted value retains native-default semantics;
the cross-run actual-policy comparison still applies. This reader support does
not import the separate FPM campaign orchestration into the Ops producer.

The exporter reconstructs each executable node registry from the original
capture plus native clone callbacks, then reconstructs all unit activity unions
from saved CUPTI traces. The source-bound native policy snapshot includes every
initialized capture bucket, even when unused by calibration. Empty boundaries
require an explicitly completed native call with an empty owned-node set; missing
observations never become zero-cost rows. Setup includes the actual captured and
uncaptured metadata/memory activity, without allocating CPU gaps or a residual.
Per-forward selection uses the slowest actual whole-forward rank and deterministic
ties, followed by physical-row medians. Different actual dispatch signatures
or activity counts cannot collide on one physical key.

`graph-calibration-evidence.json`, `graph-rank-selection.json` and
`graph-profile-control.json` bind original inputs, all-rank intervals and selected
rows. The control report retains profiled/unprofiled whole-forward ratios and
does not assume timing equivalence or rescale unit costs. The existing independent
20% phase gate remains required. The reader recomputes this proof and the control
from original files whenever binding a graph table. Full trace/token arrays remain
on disk; aggregation retains only compact timing, identity and dispatch summaries.

The common holdout evaluator uses the public homogeneous static decode API for
this explicit graph mode, after native per-request geometry validation. Padding
comes from the calibration table's policy; holdout dispatch only diagnoses policy
mismatch. Calibration and holdout must use the same actual native policy. FPM and
eager prediction paths retain their prior APIs. SGLang uncompiled FULL decode uses one fully receipted consumer data root.
Named SGLang calibration shards can populate that root as described below;
vLLM graph shard publication is not supported by this helper. No graph performance data
or accuracy acceptance is bundled with this implementation.


### Quarantined repaired vLLM Ops identity

`0.30.0+glm53kpool.bf5f6b0e689d` is quarantined after diagnostic612960
observed native generic slot-mapping access beyond the circular tail table.
The earlier four-cell bounded Engine receipts remain immutable historical
functional evidence. They do not admit this runtime for any production phase.
Python producer/readers and Rust eager/graph/FPM consumers reject the exact
version before measured lookup, including exact rows with the old source hash.
The shared repaired-runtime registry is empty. New model GPU collection on this
runtime is paused; a new repair requires independent native qualification.

Stock-runtime identity, its existing unaligned cached-prefill rejection, SGLang,
and SOL behavior remain unchanged. Historical build/qualification proof readers
retain their strict byte and receipt checks for auditing. No formal Ops coverage
or independent whole-forward accuracy is accepted by this quarantine change.

### PIECEWISE activity ownership

The separate PIECEWISE activity binder joins the initialized graph segments
and original eager callable scopes to their actual CUDA launch correlations.
It preserves native segment order and checks GPU activity in both directions.
An operation spanning multiple graph segments and eager work receives one
union of its measured device intervals, with overlap and interval sums retained
for diagnosis. Setup and external logits remain separate source-owned units.
Zero-duration CUDA API records exactly at ownership boundaries are rejected
as ambiguous; they cannot silently become setup. Structural-only graph segments
remain unsupported until their launch identity has independent native evidence.
These trace-contract tests do not qualify PIECEWISE serving collection or table
consumption. Native capture/replay, export, and independent accuracy gates remain.

The graph execution helper has a separate, default-disabled PIECEWISE trace
lifecycle. It resolves the selected V2 descriptor to the original initialized
breakable entry, checks unchanged callables, enables eager scopes only for that
target and clears them on completion or failure. A result requires original
GPU completion, sampled tokens and explicit completed native entry replay.
The serving adapter exposes this lifecycle only with
`AISIM_GLM53_PIECEWISE_REPLAY=1` and an explicit `ops_graph` or
`ops_graph_holdout` purpose. It is mutually exclusive with capture-only
qualification. It wraps the source-pinned native `BreakableCUDAGraphWrapper._replay`
once, checks the actual entry and segment identities before and after that call,
and preserves the original return and subsequent external logits computation.
The independent control retains its own initialized entry and callable objects;
it installs no operation hooks or profiler. Missing, repeated or substituted
replay is rejected. Default FULL behavior is unchanged. This opt-in has no
native GPU qualification yet. PIECEWISE schema3 export requires the separate
source/capture/activity checks below. NONE targets use their own explicit route
and cannot borrow enforced-eager calibration.

### Native serving NONE diagnostic

`AISIM_GLM53_SERVING_NONE_DIAGNOSTIC=1` selects a separate, default-disabled
raw observation for native prefill outside the initialized capture buckets.
It requires an `ops_graph` or `ops_graph_holdout` purpose and rejects either
PIECEWISE opt-in. The native graph policy and dispatch remain unchanged.
Both roles retain their independently initialized policy; calibration installs
inactive operation hooks only after initialization and activates them solely
for an actual unpadded NONE target. It installs no graph operation hooks.

The actual conditional model inherits `forward` and `compute_logits` from
`vllm/model_executor/models/glm4_1v.py` at the pinned vLLM revision. The diagnostic
binds that file, the GLM5 text source, exact original classes and methods, and
rejects compiled or substituted callables. It requires one completed original
model call and all 277 physical operation calls with the same native forward.
Its independent control installs no operation hooks or profiler. Whole events
retain the prepared-inputs-to-logits boundary; additional events directly record
prepared inputs to raw model entry, and raw model return to logits entry.
These are elapsed GPU event windows and can include host enqueue gaps.

Rows go only to `serving-none-ops-rank-N.jsonl`, carrying the actual dispatch,
model-source receipt hash, and `DIAGNOSTIC_ONLY_NO_TABLE_EXPORT`. They are not
eager query tables. The raw call inventory, setup windows and independent whole
timing are the next native qualification evidence, not accepted costs or accuracy.
This diagnostic format remains permanently excluded from table export, including
exact hits. Actual native class, GPU call coverage, control overhead and the
independent 20% gate remain required after runtime requalification.

### Native serving NONE event measurement

`AISIM_GLM53_SERVING_NONE_MEASURED=1` enables a separate prefill producer with
`measurement_contract=native_serving_none_events_v1`. It is default-disabled,
requires `ops_graph` or `ops_graph_holdout`, and rejects the diagnostic NONE and
PIECEWISE opt-ins. It retains the original native initialization, dispatch,
request state and model calls. A target must actually dispatch to unpadded NONE;
its real model class and original methods must match the source receipt.

Calibration writes `serving-none-measured-ops-rank-N.jsonl`. Every original
forward must complete each of the 277 named physical operations exactly once.
Only the fifth excluded warmup uses the observer's single profiler. Its original
trace binds the native run, rank and invocation to completed operation calls,
source-owned CPU scopes and matching CUDA API/GPU activity correlations.
Synchronous nested collectives have exclusive ownership established by their
actual parent call, excluded source and correlation. Unknown, missing or
ambiguous ownership fails. The profiler stops at logits completion before
sampling; all ten retained repetitions use the original per-operation CUDA
events without profiling.

Setup is the sum of two directly recorded GPU event windows: prepared inputs to
raw model entry, and raw model return to logits entry. Neither a whole-forward
residual nor profile-derived scaling supplies setup or operation latency.
The independent control installs no operation hooks or profiler and retains the
same whole-forward interval. Its actual policy, configuration and requests must
match calibration. Failures preserve original traces and partial call evidence.

The schema3 reader independently rederives the warmup ownership, validates every
retained event row and selects one coherent rank per actual forward. Its 277
named costs plus setup can produce exact rows only after the common runtime,
request/state/hardware and control checks pass. NONE fingerprints remain empty;
kernel names alone do not qualify P/Q interpolation. This implementation and
TEST_ONLY interoperability establish no native model qualification, runtime
admission, measured coverage or independent accuracy acceptance.

### CUDA graph runtime provider

The qualified SGLang GPU import path can load two separate instances of the
same CUDA 13 runtime binary. For that case, the graph observer reads the actual
ELF graph/stream/event relocation slots in the active PyTorch package's
`libtorch_cuda.so` and `libc10_cuda.so`. All inspected slots, including graph
launch, instantiation and stream capture, must identify one mapped runtime
instance by both path and load address. Selecting the first path or merely
finding identical file hashes is insufficient. Mixed, unresolved, missing or
different-binary providers fail. Multiple CUPTI instances remain unsupported.

This inspection requires Linux ELF64 (aarch64 or x86_64), readable process maps
and relocation memory, and `readelf` from binutils. Direct `dlopen` with
`RTLD_NOLOAD | RTLD_LAZY` preserves the existing loader state; it avoids
`ctypes.CDLL(path)` adding `RTLD_NOW`. Raw evidence retains all observed runtime
and caller hashes, load identities and actual relocation bindings. No CUDA
function runs during provider selection. The production graph API then binds
the selected existing runtime and still verifies the CUDA 13 API contract.
Full-model capture and independent accuracy require their separate GPU gates.

These are original bindings to documented interfaces, not copied implementation:
[dlopen](https://man7.org/linux/man-pages/man3/dlopen.3.html),
[dlinfo](https://man7.org/linux/man-pages/man3/dlinfo.3.html),
[dladdr](https://man7.org/linux/man-pages/man3/dladdr.3.html), and
[CPython ctypes loader flags](https://github.com/python/cpython/blob/v3.12.9/Modules/_ctypes/callproc.c).

### Captured memcpy ownership and replay proof

Actual CUDA13/CUPTI13.0.85 diagnostic613725 observed a source memcpy node whose
clone callback and internal clone query both reported type0, while the exact
executable graph/node/API correlation produced a real D2D memcpy activity.
This does not establish type equivalence or qualify a model capture. SGLang
may retain this source1/callback0 mapping as pending only with the complete
original-to-executable bijection, the qualified providers, and successful
`cudaGraphMemcpyNodeGetParams` on the live **source** node. The supported scope
is a positive-size, one-dimensional D2D copy between linear device pointers;
arrays, default-direction, offset/3D and unknown cases reject. No internal
clone pointer query substitutes for source copy parameters.

Every measured replay, including each retained repetition, must contain exactly
one matching `gpu_memcpy` activity with the source byte count and D2D direction.
The direction spelling comes directly from CUPTI copyKind/srcKind/dstKind in
[Kineto CuptiActivity.h](https://github.com/pytorch/kineto/blob/094d3c1d072362d0a919a77299459eee94f97931/libkineto/src/CuptiActivity.h#L548)
and [cupti_strings.cpp](https://github.com/pytorch/kineto/blob/094d3c1d072362d0a919a77299459eee94f97931/libkineto/src/cupti_strings.cpp#L15),
the gitlink pinned by actual Torch cf30153c4c131c8164ee7798e5022d810682e2cb.
The exporter independently rebuilds both mappings from original source/callback
records and the actual forward trace. A capture-only pending mapping, missing
activity, kernel substituted for copy, wrong direction, or byte mismatch cannot
produce a measured row. Native source parameters, callback0, and all failed
proofs remain evidence; copying is charged once to its actual operation owner.
The new native GetParams ABI and full SGLang B1 replay remain subject to actual
CPU and GPU qualification. vLLM mapping keeps the previous strict default.


### Native serving schema3

`glm53flash_vllm_serving_export.py` implements an explicit `native_serving`
reader for vLLM homogeneous prefill/decode. Existing SGLang schema1 and vLLM
FULL-only schema2 retain their original meanings. Schema3 stores each of the
277 physical operation names separately plus exactly one runtime setup term;
it does not merge different layers merely because their geometries match.
Every point uses all intervals from the rank with the largest actual
metadata-to-logits GPU interval for that same forward, with the lowest rank
breaking exact ties. Five original warmups remain in the evidence and ten
measured repetitions form each median. The result is an additive measured-unit
approximation, not an exact critical path or an arbitrary PDL scheduler.

The stable policy contains the complete initialized native capture descriptors,
priority candidates, PIECEWISE entries, source pins, checkpoint/runtime identity,
and actual EngineArgs hash with only seed excluded. Its native snapshot has no
invented schema-version field. Per-attempt file hashes, native executable IDs
and rank selection are retained separately. FULL uses its native padded request
count; PIECEWISE uses its captured token bucket and actual active request count;
NONE uses actual B*Q tokens and B requests. Context Q1 never becomes decode FULL.
Tables key phase/B/Q/past/physical tokens/physical requests as well as the full
operation geometry and name. RuntimeContext context `num_tokens` remains B*Q,
including logits; the native logits primitive independently selects B rows.

The reader reconstructs every FULL/PW executable from original source nodes and
CUPTI callbacks, including unique source and live executable identities across
descriptors. PIECEWISE keeps its native graph/eager segment order; one physical
operation can own activity in several segments and is charged by its interval
union once. Every replay trace has a unique native run/rank/invocation identity,
its GPU activities and runtime API correlations are rechecked, and independent
control executes the same requests/configuration without operation profiling.
Zero-activity operations require actual completed native call evidence. Setup
contains source-owned device work; no whole-forward residual creates a cost.

`export_serving(..., control_root=..., control_run=...)` writes a new canonical
`glm53flash_graph_perf.parquet` only after these checks and the common native
request/state/hardware reader. `bind_calibration` recomputes original evidence
and compares every row. The public homogeneous prediction helper requires that
binding, verifies current table hashes, and returns it with the predictions.
Holdout policy checks cannot supply prediction dispatch or missing measurements.
The shared acceptance path preserves the independent whole-forward 20% gate.

Initial interpolation is history-P only, with fixed phase/B/Q/name/mode/padded
geometry and equal nonempty actual dispatch fingerprints within the native
state partition. The frozen 397 calibration / 221 holdout geometry has only a
185-point structural P-only upper bound; all 36 uncovered points are P0 with
unsampled Q. Existing Q neighbors do not prove kernel compatibility. Q-axis
support requires measured source-bound specialization/workload-extent evidence;
there is no geometry-only waiver or residual scaling. NONE diagnostic files
remain `DIAGNOSTIC_ONLY_NO_TABLE_EXPORT` and are rejected even for exact hits.
Its measured producer admission remains a separate native qualification step.
No schema or TEST_ONLY test admits a repaired runtime or establishes GPU accuracy.

Schema3 shard publication uses `glm53flash_serving_shards.py`. Supply the
original parent run and every strictly admitted child to
`publish_sharded_calibration(..., parent_run=parent)`. The complete parent point
payload, child plan digests, native-to-original point map, corpus and initialized
runtime policy are checked again. Every child is rederived from its original
native capture/activity, rank selection and independent control. Named rows keep
their own evidence hashes; repeated geometry, missing children and reused native
request/run IDs fail instead of being averaged or supplied by another attempt.

The shared acceptance path binds the final table to that complete point union
and preserves the group of actual native run IDs. A group is not assigned a
fictional native run ID. Holdout children predict using the same calibrated
policy, then map results back to the original parent IDs. Errors and missing
brackets remain in those results. A passing shard publication or protocol test
does not meet the independent accuracy gate or qualify a runtime.

### Named FULL graph analysis

`export_graph(..., lookup_contract="graph_named_operations_v1")` preserves each
native operation occurrence in legacy SG FULL/schema 1 and vLLM FULL/schema 2
measurements. The Parquet key adds `operation_name`; `graph_lookup_contract`
records this analysis choice. The original native `graph_policy`, its hash,
source/capture identity and measurement boundary stay unchanged. In particular,
`embedding_allreduce` is measured and queried separately from the 90 layer
all-reduces even when their physical geometry and kernel fingerprint match.
Schema 3 serving and schema 4 prefill already preserve names and are unchanged.

For existing original evidence, call
`republish_named_graph(native_root, frozen_calibration_run, new_table_path)`.
The new path must use `glm53flash_graph_perf.parquet`. This verifies the original
native state, capture/clone/trace bindings, independent control and evidence
receipts, then derives named rows from the original traces. It writes only the
new table; original sidecars and any prior pooled table remain unchanged. A
pooled table cannot recover individual measurements or supply this export.

Named points require the complete 367 SG or 278 vLLM model/setup units and all
original five warmups plus ten measured repetitions. Rust requires exact names
and physical geometries, rejects missing/duplicate/mixed-contract rows, and
composes each measured unit once. Exact lookup and the existing FULL P-bracket,
padding, state and fingerprint constraints are unchanged; there is no geometry
fallback, SOL replacement or extrapolation. Tables without the explicit contract
keep their historical geometry-pooling behavior.

The existing public `glm53flash_lookup_audit("generation", B, 1, P)` API returns
named selected endpoints, weights, original evidence/rank-selection hashes and
activity fingerprints from the same Rust selector that computes the price.
The public validation adapter requires the exact calibration binding and retains
this audit in `prediction_evidence`. An offline re-export is an analysis result,
not new GPU measurement or accuracy acceptance. Graph controls remain
`REPORTED_NOT_ASSUMED`; profiled launch gaps and collective arrival waits are not
constants to add to independent holdout predictions.


### Named SGLang FULL decode shards

`publish_sharded_calibration(children, frozen_shard_manifest, destination,
parent_run=parent, lookup_contract="graph_named_operations_v1")` admits only
SGLang `native_full_graph` decode shards. `glm53flash_graph_shards.py` verifies
the original FPM parent and every child plan, corpus, role, complete point map
and original phase-local IDs. Each child must retain all 367 named operations
and its original five warmups plus ten measured repetitions. The helper
rederives capture/clone/trace measurements and each independent control; it
rejects reused roots, run IDs, requests, missing points and duplicate physical
rows. It never averages duplicate rows or reconstructs names from pooled data.

Without the separate group opt-in below, all calibration children must have
exactly equal complete `graph_policy` objects, including original resolved-configuration, capture/evidence and state
identities. Different capture evidence is rejected even when it might describe
the same serving configuration. The helper does not normalize those differences
or claim that actual separately collected shards are compatible. An actual
campaign using this single-policy contract must satisfy that condition. The
separate group contract below retains independent policies explicitly.

The sidecar retains each child's native run and control identities, evidence
SHA, point ownership and original parent/shard manifest hashes. Binding
recomputes the complete table from the original children. Grouped holdout
prediction uses the existing public Rust graph selector for each child and
maps successes and errors back to every original requested point. Successful
named predictions require endpoint audits with their original evidence hashes;
`prediction_evidence_origins` retains child, native and original point IDs. No
aggregate native run is invented. Controls remain `REPORTED_NOT_ASSUMED`, and
independent full-coverage accuracy acceptance remains required. This source
implementation includes TEST_ONLY query evidence, not a qualified GPU campaign.


### Explicit independent graph policy groups

For independently captured SGLang FULL decode shards, set the analysis-only
parent spec `ops_graph_group_contract="sglang_named_graph_group_v1"` and call
`publish_sharded_calibration(..., parent_run=parent,
lookup_contract="sglang_named_graph_group_v1")`. This option is not part of the
original producer plan, native arguments or native graph policy. The public
parent loader propagates it only to analysis of each original child.

The reader rederives every child's source/clone/trace and independent control
before comparing actual serving/allocator settings, native capture descriptors,
physical state layouts and named source-call ownership. Native CUDA/CUPTI
library content identities remain part of that comparison. Original process
addresses, node IDs, GPU UUIDs and receipt hashes remain in each child's original
evidence; they are not rewritten into a common native policy. Every GPU must
independently qualify as GB300. This first compatibility contract conservatively
requires equal tensor group counts, dtypes, dimensions, strides and capacities;
an actual capacity mismatch is rejected and requires a separate source-backed
rule. It is not an accuracy or cross-allocation performance guarantee.

The named table adds `graph_group`, `graph_group_sha256` and `graph_member_id`.
Each row retains its complete original `graph_policy` and SHA, measurement
evidence and rank-selection receipt. The group manifest binds the complete
original parent point union, child policies, native runs, independent controls,
and actual compatibility. Binding rederives the entire table; declared metadata
alone cannot admit measurements. The grouped native parent has no aggregate
run ID or native graph policy.

The Rust reader rejects mixed, missing, duplicate or undeclared membership. It
selects an exact complete 367-unit point first, then the nearest compatible
complete P bracket at the same B and native padded B; equal-width ties use the
lower prefix first. All units use the same selected point pair. Existing native
state partitions, dispatch fingerprint/activity guards and no-extrapolation
rules remain. The public audit uses the same selector and reports a group hash
plus each endpoint's original member, policy SHA, native run, measurement and
rank-selection SHAs, native benchmark/original point IDs and weight. It never
presents one member policy as an aggregate policy. Calibration/control identities
cannot be reused by independent holdout. Legacy unopted schema1/2 and schema3/4
contracts retain their existing behavior.

Tests use explicitly authored synthetic measurements and a rebuilt public Rust
consumer. Real multi-shard compatibility and full holdout accuracy still require
original GPU evidence. No historical failed pilot is relabeled as accepted.

### Pre-freeze vLLM observation-family partitions

`glm53flash_observation_partition.build_partition(parent, child_plans,
runtime_identity, parent_cell_id=..., role=..., campaign_id=...)` creates the
explicit `glm53flash_ops_observation_partition_v1` analysis contract. Supply the
complete original parent and every original FPM child plan. The unchanged
`validate_point_union` rechecks their content hashes, original phase-local IDs,
point geometry, topology and corpus before deriving any leaf. The public helper
has no campaign paths or fixed campaign point/child counts.

This source-bound version declares FULL for decode, PIECEWISE for prefill BQ up
to 2048, and NONE above 2048, within the original batch32/8192-token/context128K
bounds. It never forces native dispatch. Every real target on every rank must
pass the existing strict source, snapshot, descriptor, state, physical ownership
and timing readers, then match its declared family and all five warmup plus ten
retained repetitions. The actual snapshot must retain FULL_AND_PIECEWISE,
max32 requests and capture2048. A mismatch fails; it cannot trigger repartition,
retry, fallback or reuse of a pilot's rows.

`leaf_plan(partition, leaf_id)` emits a new `glm53flash_ops_observation_leaf_v1`
plan. It is not an original FPM child. The original source plans remain inside
the partition. Native benchmark IDs are local to each new leaf; the unchanged
scheduler therefore assigns new token offsets. The contract preserves original
**geometry and corpus**, not historical token-byte equality. New per-leaf
requests must be frozen by the native scheduler. The independent calibration
and control leaves must use the same new point map/corpus and distinct run IDs;
actual cohort/token/dispatch equality remains mandatory. Holdout preserves its
complete independent original point union and corpus. Errors remain in the
original parent denominator.

Use these parent acceptance-spec fields with the existing public plan loader:

- `plan` and `cell_id`: original parent receipt and selected phase cell.
- `ops_execution_mode`: `native_serving`.
- `ops_observation_partition`: path/SHA receipt for the new partition.
- `observation_runtime_sources`: exact path/SHA references for every declared
  entry source file; the loader verifies bytes and size.
- `observation_children`: every real leaf's custom `plan` receipt, `cell_id`,
  `raw_root` and `ops_execution_mode=native_serving`.

Do not supply an aggregate `raw_root`, original FPM `shards` field, or a fake
FPM plan for a leaf. Load calibration, control and holdout as distinct roles.
The common `glm53flash_formal_observation_runtime_v1` source declaration binds
the immutable producer commit/wheel/source map, shared native runtime/cache/
overlay and both entry mechanisms. Actual native provenance must match its
runtime digest and the leaf's unique run ID. Source declaration and entry byte
checks do not replace installed-wheel, startup-chain or GPU qualification.
Render a new experiment with the repaired public `build_run_provenance` helper
and exact leaf `run_id`/`FPM_RUN_ID`; keep both native BenchmarkPoints phase lists,
including the empty unused phase. Historical renders remain historical.

The existing `publish_sharded_calibration(..., parent_run=parent)` and binding
routes re-read every original real leaf and its independent control. The new
`glm53flash_serving_observation_ownership_v1` sidecar preserves each row's new
leaf/native ID plus original child/local/parent IDs, raw evidence and native
policy. All leaves require the same complete actual serving policy and separate
native requests/runs. Each successful prediction keeps the unchanged Rust
endpoint audit, with these distinct origin IDs in `prediction_evidence_origins`.
A group has no invented native run. Legacy original-FPM routing and schema3
row keys, lookup rules and formulas are unchanged. Source/TEST_ONLY checks do
not establish GPU collection, full coverage or accuracy acceptance.
