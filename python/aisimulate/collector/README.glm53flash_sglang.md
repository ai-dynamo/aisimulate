# GLM-5.3-Flash SGLang serving telemetry

Formal collection uses `sglang_retained_request_benchmark_v1`. Engine still
receives real tokenizer-generated prompts and constructs native `Req` objects.
The benchmark replaces only the scheduler event loop with a serialized cohort
controller. It retains the configured native forward streams, TP worker,
ModelRunner, graph dispatch and DeviceTimer. These measurements describe the
native forward interval, including native graph load/replay work where the
DeviceTimer includes it; they do not measure production scheduler throughput.
Ordinary Engine smoke traces without this protocol cannot be formal data.

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

Each state-layout receipt also contains its native TP rank and selected CUDA
device properties. The existing layout digest binds that hardware evidence to
every forward. Formal readers require actual GB300/sm103 on every rank, matching
tensor devices and distinct native UUIDs when available. Rejected hardware
receipts are retained before startup fails. Earlier frozen smoke payloads remain
historical; missing hardware fields cannot be added after measurement to qualify
them as formal data.

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
