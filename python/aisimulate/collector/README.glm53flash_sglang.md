# GLM-5.3-Flash SGLang serving telemetry

Formal collection uses `sglang_retained_request_benchmark_v1`. Engine still
receives real tokenizer-generated prompts and constructs native `Req` objects.
The benchmark replaces only the scheduler event loop with a serialized cohort
controller. It retains the configured native forward streams, TP worker,
ModelRunner, graph dispatch and DeviceTimer. These measurements describe the
native forward interval, including native graph load/replay work where the
DeviceTimer includes it; they do not measure production scheduler throughput.
Ordinary Engine smoke traces without this protocol cannot be formal data.

For Ops prefill under the production decode policy, the driver accepts
`--ops-native-prefill` only with a prefill target and `ops`/`ops_holdout`
observation. It requires disabled prefill capture and FULL decode in both the
declared and resolved native configuration. Decode graphs are constructed by
the original runtime; prefill still reaches its native `EagerRunner` through
`model_executor/model_runner.py:1774–1865` at the pinned revision below.
The command builder exposes this as `ops_execution_mode="native_eager_prefill"`;
an optional explicit `sglang_mem_fraction_static` is preserved in native argv.
The evidence reader requires the same mode, actual NONE forwards, real request
histories, and the existing embedding-to-logits GPU boundary. A separate control
uses the calibration requests without operation hooks. This mode adds no graph
fallback, state reconstruction or qualification by itself.

For each repeat, the controller executes every prefix in real native chunks
of at most 8192 new tokens, keeping the same request, KV row and KDA state slot.
It then combines the parked requests into one native `ScheduleBatch` for the
requested B×Q extension (total new tokens <=8192). For decode it seeds P−1
tokens, runs the final prompt token for the complete cohort, and feeds the
actual samples into native decode preparation. No fake KV blocks, random state
or manually advanced KV counters are used. B=1/2/4/8/16/32, arbitrary nonnegative
cached-prefix lengths and positive query lengths are representable, subject to
the frozen inclusive context limit and actual native allocation/kernel success.

The population and lifetime calls follow SGLang revision
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`:

- `managers/schedule_batch.py:1484,1510,2632` owns requested extend ranges,
  input preparation and native allocation; continuing requests call
  `init_next_round_input()` without another radix match.
- `managers/scheduler.py:3456,3588,3991` and
  `managers/scheduler_components/batch_result_processor.py:415` own the
  middle-chunk result flag and stashing. The controller sets only the native
  scheduling range and middle-chunk flag; allocation counters are written by
  `mem_cache/allocation.py:344-450`.
- `mem_cache/chunk_cache.py:86` reads committed prefix indices from the real
  GPU request row. `mem_cache/memory_pool.py:1422` preserves existing request
  and Mamba allocations across further native preparations.
- `managers/scheduler.py:4101,4245,4594` retains native decode preparation,
  TP execution, sample relay and result processing; `mem_cache/common.py:276`
  releases KV and Mamba state on normal request completion.

`retained-rank-N.jsonl` links every seed/target forward to the observed GPU
completion and records actual GPU KV-index digests, request-to-Mamba mapping,
stable logical slots, parked state and release. Raw traces and envelopes carry
the same producer protocol, model execution identity, run, context and timing
policy. Readers require every rank and reject missing seed, changed slot/index
history, incomplete release or an ordinary-scheduler trace. Physical layout
receipts separately describe KDA conv/recurrent state, sparse latent/index and
IndexPool tails. CPU lifecycle tests validate the contract; actual GPU capacity,
dispatch and full-matrix accuracy remain to be measured.

In each native worker, call `collector.glm53flash_sglang_runtime.install()`
from the campaign's `sitecustomize.py`. Set `AISIM_GLM53_TRACE_DIR` to an
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

Token histories are retired from worker memory only after the native request
releases both KV and Mamba state and its retained-state receipt is written.
The original token arrays remain in the raw JSONL files. FPM publication and
validation stream those files one record at a time and retain compact timing
witnesses after validating each complete request chain. File SHA256 checks,
all-rank token/dispatch equality, lifecycle coverage and request-reuse rejection
remain mandatory. Streaming changes host memory consumption, not the native
timing interval, raw evidence format or admission thresholds.

Additional integration sources at SGLang revision `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` (v0.5.20, Apache-2.0, Copyright SGLang Team and contributors) are
`python/sglang/srt/managers/tp_worker.py`,
`python/sglang/srt/model_executor/{model_runner,forward_batch_info}.py`,
`python/sglang/srt/model_executor/runner/{eager_runner,decode_cuda_graph_runner}.py`,
and `python/sglang/srt/utils/device_timer.py`. Their implementations are called
without modification; the wrappers and trace schema are original adapter code.


### Event-loop entry state

The pinned native `Scheduler.__init__` calls `init_overlap` before serving;
that method owns FutureMap, forward/copy streams and the two-slot batch lifetime
ring. `run_event_loop` owns the schedule-stream context and WAR-barrier policy.
The retained loop leaves both native initialization stages intact. The native
`event_loop_overlap` initializes its `result_queue` only at loop entry
(`scheduler.py:1944–1948`); the replacement also initializes this empty queue
before request ingestion and idle checks, and refuses to discard a pending
result. Each completed result is drained synchronously, so this queue remains
empty. Idle gaps and paused-engine accounting call the native paths. These are
scheduler bookkeeping changes; they do not seed or repair GPU hybrid state.
The first retained GPU attempt exposed this missing queue before any request or
timing; its failure remains a failed qualification, not a data observation.

### Source-bound native prefill units

`--ops-native-prefill` retains disabled prefill graphs and FULL decode graph
initialization. Its new observation contract is
`native_sglang_prefill_events_v1`. Only the fifth excluded warmup uses the
existing observer profiler, started at the actual native model entry and
stopped after logits. The original Chrome document is retained with exact
run/rank/forward identity before deriving CPU-scope → CUDA-API → GPU-activity
ownership. Names are never used to assign kernels to operations. Every one of
the 366 physical units must execute; qualified hoisted MLA projections can
contribute disjoint source parts, and synchronous nested collectives own their
activities exclusively. Unknown, missing or unowned device activity rejects
export and leaves the trace and error evidence intact.

The native `BumpAllocator.__init__` in the model's forward calls `torch.zeros`.
The observer directly times its one actual 90-element FP32 CUDA allocation and
records the constructor arguments as `native_graph_setup`. This interval is
not a residual or a host allocator estimate. TBO, auxiliary hidden states,
multimodal inputs, a second conditional zero allocator and a configured
`input_embeds` copy buffer require separate observation contracts. The current
route rejects these states without changing native execution. The remaining
ten retained samples contain original CUDA-event intervals and no profiler.
Independent control and holdout use only the original complete model timer.

`glm53flash_sglang_prefill_export.export_prefill` rederives every trace, original
event row, same-request state chain and all-rank 5+10 identity. It requires a
separate control with identical native configuration (only random_seed is
normalized), distinct requests, identical input tokens and the original 5%
timing-equivalence threshold. Terminal sampled outputs remain in the original
evidence and state-chain checks; they are produced after the timed boundary
and need not equal the independent control's terminal outputs. One
whole-forward slowest rank is selected per
repetition for all its unit costs; retained medians never mix per-unit maxima.
No ratio rescales measured latency.

The dedicated `glm53flash_sglang_prefill_perf.parquet` schema4 keeps operation
name and full B/Q/P coordinates for all 366 units plus the existing runtime
marker. Its public consumer initially supports exact homogeneous prefill
queries, including SG's unaligned prefixes. Missing points, incomplete phases,
wrong source/config/checkpoint/runtime, direct token-only queries and legacy
eager fallback are rejected. Schema1 SG decode and vLLM schemas2/3 retain their
meanings. Complete independent holdout coverage and accuracy acceptance still
require real GPU evidence; CPU and TEST_ONLY tables do not establish them.

The source boundaries additionally inspect immutable SGLang revision
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`, paths
`python/sglang/srt/models/glm5_next.py`,
`python/sglang/srt/managers/mm_utils.py` and
`python/sglang/srt/utils/common.py` (Apache-2.0, SGLang Team and contributors).
No upstream computation is copied or substituted. Original failed native
prefill evidence without Chrome traces cannot be upgraded to this contract.


### Independent graph control with matching native inputs

The graph driver records its original public `Engine.generate` arguments in
`sglang-request-inputs.jsonl` before each call, outside the model timer. It also
records the actual collector source closure and the five pinned native
sampling files, plus each worker's original loaded sampling methods. These
receipts are part of the raw evidence. Existing runs without these receipts
cannot be relabeled as this new producer.

After a fresh calibration completes, call
`glm53flash_sglang_control.freeze_reference(native_root, frozen_run, new_path)`.
This revalidates the original complete native state and graph activity proofs
and writes a new reference containing all 5+10 point repetitions, actual
submitted request order, prompts, Q1 query tokens, input-history hashes and
source/TP forward identities. The returned SHA identifies its original bytes.
It does not imply that calibration has passed its independent timing control.

Only a separate decode `ops_graph_holdout` process on the calibration corpus
may pass `--ops-graph-control-inputs NEW_REFERENCE.json` together with
`--ops-graph-control-inputs-sha256 SHA256`. Its public per-request sampling
parameters add exactly `logit_bias={str(original_target_token): 100.0}` to the
existing temperature-zero, two-token, ignore-EOS request. This is a fixed finite
experimental value, not a documented native magnitude bound or a guarantee of
selection. Before a GPU qualification, the actual pinned native CPU
`SamplingParams(**kwargs).verify(vocab_size)` must accept those exact arguments.
The GPU reader must then prove every real target query and complete input
history equals the reference. A mismatch fails; there is no adaptive bias,
retry, request-field mutation, tensor write or fabricated KV state.

Native sampling applies this bias after the original metadata-to-logits model
interval. The bias allocation can nevertheless change execution conditions;
the original independent 5% timing control remains mandatory and cannot be
used to rescale measured units. Terminal sampled outputs remain in the raw
TP and state-chain evidence, but equality between terminal outputs is not an
input constraint. Actual query/history disagreement still rejects export.
Calibration and control must have the same complete producer and loaded
sampling identity, native configuration and initialized graph policy. The
control reference and its SHA remain in the original provenance/forward rows
and are rederived against the calibration files during publication.

This mechanism still needs actual native CPU and GPU qualification. Original
failed controls stay failed. No new timing data, formal matrix coverage or
accuracy acceptance follows from CPU fixtures.


### Recorded native allocator policy

The shared native driver accepts `--sglang-allocator-max-split-size-mb N`, an
exact integer of at least 20 MiB under the pinned native PyTorch parser. It sets
`PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:N` before framework
imports. Conflicting inherited CUDA/HIP/unified allocator settings and disabled
caching are rejected, including explicitly empty settings. Omission follows
the original default and does not reclassify historical unknown environments.

Each new worker records the actual native backend, effective split limit,
allowlisted environment, Torch source and loaded library identities, native
run/precision identity and GPU. Every forward binds the original rank receipt
SHA. Shared readers compare the normalized actual allocator with ServerArgs;
known-policy evidence cannot be paired with legacy unknown evidence. These
adapters are ported from FPM commit636206a73f391a1340b57acd8c396c557061c9db.
Ops keeps its existing orchestration; externally frozen public FPM plans/cells
must retain any explicit allocator option. This records execution identity,
not capacity qualification, an OOM repair or performance acceptance.


Ops additionally cross-checks the requested allocator option in the original
plan, selected cell and runtime cell, reads every original rank receipt against
its GPU and checkpoint, and checks every seed and measured forward's allocator
SHA. Both graph and prefill readers derive the same actual allocator-inclusive
execution-policy digest; independent controls, holdouts and existing shard
unions reject known/unknown or differing actual policies. Schema4 uses its
existing `execution_policy_sha256`. Schema1 retains the original meaning of
`resolved_config_sha256` and binds allocator evidence through its Python
execution-policy and raw/control evidence closure. No public allocator query
axis or competing deployment policy is introduced.
