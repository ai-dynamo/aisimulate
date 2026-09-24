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
must receive a separate qualified contract. SGLang's hoisted latent projection
is a disjoint part of its attention measurement. KDA conv/recurrent state and
the sparse indexer's incomplete pool tail must come from the real request's
native prefix execution.

mHC is measured at native boundaries, including the output RMSNorm: vLLM has one
pre, 89 fused post/pre and one post interval, plus expand/contract; SGLang has 90
pre and 90 post intervals plus expand/contract. If SGLang leaves RMSNorm outside
its pre call, the current hook rejects that path rather than undercounting it.
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
graph occurrences on every rank, then takes the median of per-invocation rank
maxima. Distinct checkpoint formats, runtimes, graph modes, seed policies or
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
observation to its admitted native forward, and reaggregates rank maxima and
medians before accepting the table. Content hashes alone are insufficient.

Independent `ops_holdout` runs disable module hooks and record one GPU event
window from embedding through logits. The separate native scheduler/DeviceTimer
interval is retained but never substituted as the Ops comparator. The common
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
