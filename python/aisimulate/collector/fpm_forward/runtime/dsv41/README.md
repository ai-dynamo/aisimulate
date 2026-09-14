# DeepSeek V4.1 native FPM canary producer

Status: the historical producer completed **126 GB200 calibration points**
(100 prefill, 26 real decode) in eager TP4/DP1 mode. The retained
[collection receipt](../../../../../../data/experimental/deepseek-v41/gb200-fpm/calibration-v1/collection-receipt.json)
binds producer SHA256 `771d2591eb5893f636a4208d60eb10daf51192eb4062bfec22da2f981cbe8ec0`
and runtime-source manifest `a201055338b2e185db52b44744b0033aca850452580279bde4d3a4d8f8dceaef`.
The earlier allocation-timeout and eager warmup failures remain failed evidence;
they are not the completed calibration run. Subsequent ordinary-serving
[validation](../../../../../../data/experimental/deepseek-v41/gb200-fpm/ordinary-serving-retention128-v1/README.md)
exposed a large latency mismatch, so completed collection does not establish
serving prediction accuracy or general scheduler qualification. Current changes
to activation, startup configuration and explicit identity attestation have CPU
regression coverage; they have not been collected again on a GPU. This remains
a bounded collection extension, not a general serving scheduler.

`dsv41_scheduler.py` extends the native Dynamo `InstrumentedScheduler`. It keeps
native `BenchmarkPoint`, FPM messages, schema-v2 rank artifacts, coverage checks,
and second-step decode timing. The producer writes additive provenance directly.
The Collector must continue to reject failed, incomplete, fake, or unmarked rows.

The DP1 adapter bounds each complete real-KV point at 120 seconds, including
all same-request seed forwards and first-use kernel compilation. This deadline
is independent of Dynamo's short synthetic-point result timeout; that native
timeout, native interval timing and the campaign timeout remain unchanged.
Both timeout values and the adapter source hash are recorded in producer
provenance. Expiration raises a retained failure and never publishes a partial
point or switches to synthetic KV. The first formal-geometry canary exposed
the old eight-second deadline during Engram JIT; that failed attempt is retained.

## Source and attribution

The scheduler adapter is modified code derived from NVIDIA's Apache-2.0 Dynamo
implementation: https://github.com/ai-dynamo/dynamo/blob/54960177085413259859c88bd34ed0734d4c2ea9/components/src/dynamo/vllm/instrumented_scheduler.py

Original copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. Original SPDX attribution remains in the derived module.
`LICENSE` preserves the upstream Apache-2.0 license. No root NOTICE file exists
in that upstream checkout. The original text fixture and test driver were authored
for this change; they contain no third-party corpus content. The fixture is also
Apache-2.0 licensed under the adjacent license and this attribution.

Compatibility was initially inspected against vLLM preview commit
79a7108d9aea27ddab99ce1779290d300b17fc23:
https://github.com/vllm-project/vllm/tree/79a7108d9aea27ddab99ce1779290d300b17fc23

Relevant original vLLM files carry Apache-2.0 and `Copyright contributors to the
vLLM project`. The adapter calls their request, scheduler-output, and KV-manager
APIs; it does not vendor vLLM implementation files. Current source hashes are in
`runtime-source-sha256.json`. The actual ARM image reports build version
`0.1.dev20904+g179dd0fa9`; that abbreviated SHA does not resolve in the public
upstream repository. Actual installed sources were exported and compared: the six request/scheduler/
KV-manager files match the inspected preview exactly, as does model_state.py.
The actual model and Engram files add DP-shared table sharding/gathers and per-ubatch
staging; config/engram.py adds the model_has_engram_layers helper. The source
manifest now pins the exported image files. The exported DP helper explicitly returns shard size 1 for DP1, creates no
Engram DP group in that case, and gather_engram_hashes returns its input unchanged.
The exported parallel config confirms num_ubatches=0 for this canary. The actual
ModelConfig preflight passed with outer/text model_type=deepseek_v41, architecture
DeepseekV41ForCausalLM, and resolved Engram cpu_offload=false. Full producer import
passes with the composite runtime below. Completed calibration and subsequent
serving-accuracy limitations are recorded above. Do not substitute the package
version string for source verification.
Artifacts report vllm_revision=null, the actual package version, the inspected API
revision, and the source-manifest digest rather than inventing a global git revision.

The derived files and source pin are recorded in root THIRD_PARTY_NOTICES.md
and its byte-identical packaged copy. Run scripts/check_packaged_legal_files.py
after changing either notice.

## Runtime

Stage this directory on PYTHONPATH with Python `ai-dynamo==1.4.2` and matching
`ai-dynamo-runtime==1.4.2`, then overlay these four unchanged files from Dynamo
`54960177085413259859c88bd34ed0734d4c2ea9`:

- `components/src/dynamo/vllm/instrumented_scheduler.py`
- `components/src/dynamo/vllm/benchmark_points.py`
- `components/src/dynamo/vllm/gc_policy.py`
- `components/src/dynamo/common/forward_pass_metrics.py` (already identical in 1.4.2)

Keep both distributions' licenses and the source manifest. This is a composite
Python runtime: the instrumentation revision is not the revision of the entire
worker. Using all Python modules from that newer commit with native runtime 1.4.2
failed worker import at `update_model_taints`. The verified composition passes
frontend help, worker help, native runtime initialization/shutdown and the
source-checked scheduler import. Generator 1.3/0.24 metadata selects a supported
launch template; it does not describe the deployed backend versions.

Install the matching AISimulate
wheel, including its native extension, so the shared execution_identity helper
can import. A source-only Python path is insufficient. The image must provide
the pinned vLLM source and native GPU dependencies.

Set:

- DYN_FPM_DSV41_REAL_KV=1
- DYN_FPM_INPUT_TEXT=/staged/path/fpm_text.txt
- DYN_FPM_TOKENIZER_REVISION=fb2764a5cf321eaa5070ca8f9e892818f477c16d

Use the native Dynamo scheduler class. The lightweight `sitecustomize` hook
defers source verification and subclass activation until that exact scheduler
module is imported. Compiler/helper interpreters do not import Dynamo or vLLM
through this hook. Both native-first and adapter-first imports are tested. An
activation failure raises a chained `RuntimeError` at that later import. It is
outside Python's startup `sitecustomize` exception handler, so ordinary imports
fail with a traceback and allow `finally`/exit cleanup. The preflight catches
activation and shared-SDK import failures, saves a failed audit, and re-raises;
it also rejects a missing scheduler activation before model loading. It never
falls back to the unmodified scheduler.

Runtime gates require a local pinned V4.1 checkpoint, TP4, DP1, PP1, no EP, no
speculation/DSpark, no ubatching/DBO or context parallelism, no KV or encoder connector, and explicit Engram cpu_offload=false.
The last requirement matters: the preview defaults to CPU UVA offload. All requests
are text-only; no vision inputs or decoder replay are injected.

This first V4.1 dataset contract supports eager execution only. Pass
`--fpm-enforce-eager`; the frozen plan and rendered arguments record it, the
producer checks the actual model configuration, and the native reader requires
the producer's `execution_mode=eager` evidence. Graph data cannot be published
under this contract. Supporting graph execution later requires a distinct query
identity, as well as separate calibration and validation. The earlier graph
startup attempt is retained as runtime qualification evidence only.

Native benchmark warmup_iterations must be 0. Native per-shape eager warmups remain
in place and run through the same real-forward path. The token-stream sidecar's
schema 2 labels each history as `warmup` or `measurement`, records the frozen
native warmup IDs, and keeps completed warmup witnesses separately in
`warmup_results`. Only the native measured `results` enter calibration tables.
The reader requires exact coverage of both sets and rejects leaked warmup
timings or unclassified extra histories. This mirrors the eager-replica identity
and save boundaries in [Dynamo's pinned scheduler](https://github.com/ai-dynamo/dynamo/blob/54960177085413259859c88bd34ed0734d4c2ea9/components/src/dynamo/vllm/instrumented_scheduler.py#L2376).
The first eager canary's original mixed sidecar remains failed qualification
evidence; its extra warmup records must not be silently removed.

An explicit native point
manifest is accepted. For an automatic native grid, use max_model_len<=2050,
max_num_seqs<=2 and max_num_batched_tokens<=512. The complete study requires
2050/2/512 and enough native page capacity. Every grid point is checked against
batch<=2, decode past KV<=2048, prefill prefix-plus-new<=2048 per request, and
total newly scheduled prefill tokens<=512. Unsupported points fail the run
instead of being silently removed.
A native decode point with context<2 fails instead of being relabeled.

## Startup configuration

The adapter's image-layout default lives in the staged `runtime-paths.json`.
To use another verified composition, set `PYTHONPATH` through the existing
`K8sConfig.extra_env` deployment input. The Collector prepends its staged adapter
directory and writes the identical resolved path into `run.sh` and
`collector-runtime-env.sh`; the latter also configures preflight in Slurm, where
Kubernetes Pod environment injection is absent. Required source hashes still
apply at the configured paths; changing a path does not bypass verification.

`FPM_READINESS_TIMEOUT_SECONDS` uses the same `K8sConfig.extra_env` input, defaults
to 900 seconds, and accepts integer budgets from 1 through 3600. This is the
bounded etcd rendezvous wait after each node's preflight, covering staggered
container startup. For example, `{name: FPM_READINESS_TIMEOUT_SECONDS, value: "600"}`
sets a ten-minute budget. The outer execution timeout still bounds the entire
step; this is not an additional native benchmark timing or per-point budget.
Slurm `wait_ready` separately waits for the exact existing allocation to become
RUNNING and validates its node count within the caller's single deadline; it does
not claim the container or model has loaded.

## State and timing contract

A request is registered once, with the full tokenizer-produced prompt history.
Its prefix is executed in real forward chunks from position zero. The request and
all its block tables remain alive, including the non-prefix-cacheable compressor
ring and Engram lookback state. Each seed forward drains before measurement starts.
Prefill measures the suffix on that same request. Decode performs native admission
at context-1 then measures the following steady step at the exact requested context.
Those two decode steps can pipeline as in the native benchmark.

No code assigns num_computed_tokens. vLLM advances it only through the normal
post-schedule accounting, and the producer counts seed tokens only after their
model output callbacks complete. It refuses synthetic-prefix/decode entry points,
allocation failure, lost requests, unexpected CoW, timeout, or missing native FPMs.
The `kvwarm_real_kv` point annotation is earned immediately before successful native
save. Each saved result also has completed_seed_tokens and same_request witnesses.

Input provenance includes the original UTF-8 text SHA, the SHA of compact JSON token
IDs produced by the actual runtime tokenizer, tokenizer revision, token count and
unique count. The producer also writes benchmark_results.token-streams.jsonl
with actual prompt and sampled output token IDs for every completed point. Its
SHA and record count are in input_provenance.token_stream_manifest; individual
point witnesses include the stream SHA. Per request, offset is 131*request_index+17*benchmark_id, wrapping the
same real-text token stream to the requested length. This fixture is reproducible;
it is not a representative production corpus. Routing/locality sensitivity and
numerical equivalence remain live validation work.

From the repository root, run
`python/aisimulate/.venv/bin/pytest -p no:timeout -c python/aisimulate/pytest.ini python/aisimulate/tests/unit/collector/test_fpm_dsv41_producer.py`
to execute the 14 isolated CPU lifecycle and producer-consumer contract checks.
The test wrapper sets the producer and native artifact module paths. They cover actual
seed completion before measure, retained request identities, cached-prefill state,
uneven request lengths, decode pipeline timing sequence, failure paths, and earned
annotations. These are state-machine tests, not substitutes for GPU correctness.

Same-request seed forwards qualify retained KV state for cached-prefill points.
Cross-request prefix-cache reuse needs separate serving verification with a warm
request, a subsequent request sharing its prefix, and a cold control. End-to-end
latency and full Dynamo FPM traces from those verification runs remain separate
from the calibration grid and its published curves.


The pinned Linux ARM64 image is
`vllm/vllm-openai@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58`.
Mount the verified composite Dynamo tree at `/opt/dsv41-dynamo` read-only.
Install the matching AISimulate wheel, including its native extension, in the
runtime image. Importing its shared SDK identity through a source-only Python
path is insufficient: the package loads the native extension at import time.
Stage the adapter, source manifest, and fixture through the Collector. The adapter,
Dynamo source and AISimulate wheel must be available during runtime preflight.
The installed native `ai-dynamo-runtime` is separately pinned and import-tested;
the Python package version alone does not establish scheduler compatibility.
