# GLM-5.3-Flash SGLang serving telemetry

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

Additional integration sources at SGLang revision `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` (v0.5.20, Apache-2.0, Copyright SGLang Team and contributors) are
`python/sglang/srt/managers/tp_worker.py`,
`python/sglang/srt/model_executor/{model_runner,forward_batch_info}.py`,
`python/sglang/srt/model_executor/runner/{eager_runner,decode_cuda_graph_runner}.py`,
and `python/sglang/srt/utils/device_timer.py`. Their implementations are called
without modification; the wrappers and trace schema are original adapter code.
