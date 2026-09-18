# GPT-OSS TTFT: eager submission and rank waiting

## Status

- The requested trace collection is complete. The main missing execution
  boundary is supported by two-rank kernel traces and separate CPU timings.
- **The TTFT prediction is not fixed.** The frozen point still predicts
  16.53 ms versus the original unprofiled reproduction's 48.86 ms.
  No kernel rows, latency constants, or 263-point predictions are changed.
- The proposed op-based correction needs an execution-mode/CPU-timing
  data contract; see the [reviewable plan](../../../../docs/trtllm-eager-execution-plan.md).
  The user chose to defer this extension and retain the collection evidence.
  No producer/consumer implementation is approved.

## Matched collection

- GPT-OSS-120B, B200 TP2/EP1, ISL1024/OSL1024, concurrency4; NumPy seed0,
  length ratio0.8, eight warmup and 40 measured requests per repetition.
- TensorRT-LLM PyTorch backend rc14, source
  `93cb6518b6d6dbd6095748189e626db731f44545`; image child digest
  `fe2f17d0c9698bafeb9f437929003c6badc56d17e8382a02505233145a07f61f`.
  Model revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, InferenceX
  client `2baba8e27be8529b4453afc953f9859ec092c73d`.
- Job 1996517 completed 0:0 in 3m06s on two B200s. Both Torch traces parse;
  all 40 measured input/output token vectors match frozen replay point0028.
  Capture covers 201 forwards per rank: two mixed and 199 decode steps.
- Profiling changes timing: client TTFT 57.46 ms, TPOT 3.0572 ms. This is
  diagnostic evidence, not a replacement for the unprofiled E2E baseline.

## GPU attribution

Associate each CPU launch inside a forward range with GPU work by its CUDA
correlation ID. Use the projected GPU envelope for that forward, and union
GPU intervals within each rank. Do not select kernels merely because they
fall inside a CPU timestamp window. CPU and GPU annotation copies are
explicitly separated. Sampler/copy work outside the forward is retained as
unmatched evidence, not assigned to a transformer operation.

| Mixed step | Rank | GPU span (ms) | GPU busy union (ms) | Gaps (ms) | Gap before next launch starts (ms) | MoE kernel sum (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 886 prefill + 3 decode | 0 | 50.529 | 33.425 | 17.105 | 16.364 | 7.492 |
| 886 prefill + 3 decode | 1 | 50.597 | 11.443 | 39.154 | 35.765 | 7.483 |
| 1014 prefill + 3 decode | 0 | 41.100 | 35.328 | 5.771 | 5.499 | 8.040 |
| 1014 prefill + 3 decode | 1 | 41.128 | 12.306 | 28.822 | 25.672 | 8.062 |

- The 886-token native op probe is 12.569 ms, including 8.281 ms of MoE.
  This is a controlled shape probe; its decode KV is an approximation, not
  a reconstructed exact scheduler query. The MoE coordinate is exact:
  889 combined tokens, TP2/EP1, MXFP4 weights and MXFP8 activation.
- MoE kernel sums are close to the corrected prediction. Increasing the
  shared MoE graph table is not supported by this trace.
- Fused all-reduce/norm kernel sums differ sharply between ranks:
  23.547/1.601 ms for the first mixed step and 24.870/1.772 ms for the second.
  Rank 0 also has elongated router GEMMs. Kernel durations include dependency
  and peer waiting; these differences are not evidence for a 20x collective
  throughput correction. Summing ranks or adding waiting to host gaps would
  double count parts of the dependency path.
- The pinned [GPT-OSS TP block](https://github.com/NVIDIA/TensorRT-LLM/blob/93cb6518b6d6dbd6095748189e626db731f44545/tensorrt_llm/_torch/models/modeling_gpt_oss.py#L365)
  fuses all-reduce, residual add and RMSNorm. Compare that combined boundary.
- Across 199 decode steps, mean GPU spans are 2.893/2.890 ms; gap unions are
  0.254/0.191 ms. Decode does not exhibit the mixed step's large eager gaps.

## Lightweight CPU timing

- Separate job 1996606 completed 0:0 in 4m14s. No Torch profiler or per-op
  device synchronization is enabled. Wrappers record CPU call intervals for
  the forward, attention, MLP, fused all-reduce and decoder layer.
- All 120 measured requests across three repetitions match replay token
  vectors. Native request timestamps exclude warmup and identify 114
  measured prefill/mixed forwards in each worker.

| CPU boundary | Worker A mean (ms) | Worker B mean (ms) |
|---|---:|---:|
| Complete model-engine forward call | 31.116 | 30.632 |
| 36 decoder-layer calls | 24.389 | 23.740 |
| Attention calls | 8.716 | 8.502 |
| MLP calls | 9.320 | 9.138 |
| All-reduce calls, including embedding | 5.065 | 4.855 |
| Work outside decoder-layer calls | 6.728 | 6.892 |

- These are nested boundaries. Layer totals contain their attention, MLP
  and all-reduce calls; CPU time overlaps GPU work. Neither columns nor
  CPU/GPU totals may be added as independent latency penalties.
- Client TTFT 55.06 ms and TPOT 2.8368 ms remain diagnostic. Worker labels
  follow the ordered artifact names in the JSON; they do not assert MPI rank.
- The first trace's custom MoE CPU wrapper produced no ranges in this runtime.
  MoE classification therefore uses the observed routing/quantization/expert
  GEMM/finalize kernel names and verifies 36 layer occurrences. The separate
  lightweight MLP wrappers do execute, with 36 calls per measured forward.

## Same-allocation control

- Job 1996733 completed 0:0 in 8m22s, with two successive server launches
  on the same two-GPU allocation. Both retain the whole-forward CPU timer
  and native request metrics; the first disables only module timing.
- Both phases complete 120/120 measured requests with identical replay token
  vectors. Each worker has 114 measured prefill/mixed forwards. Match CPU
  records to the post-startup native iteration sequence, assert every batch
  state agrees, and select the native measured requests' `first_iter` values.
  This independently agrees with the timestamp-based warmup exclusion.

| Observation | Module timing off | Module timing on |
|---|---:|---:|
| Client mean TTFT (ms) | 53.1475 | 56.2429 |
| Client mean TPOT (ms) | 2.8536 | 2.8423 |
| Worker A mean forward CPU call (ms) | 30.0409 | 30.8428 |
| Worker B mean forward CPU call (ms) | 30.3245 | 31.2880 |

- The module-timed phase has 5.82% higher TTFT. Sequential phases can also
  have temporal drift; this is an observed difference, not a causal estimate
  of instrumentation overhead alone. Even without module timing, CPU
  submission remains about 30 ms. None of these values is used as a fitted
  latency penalty or published kernel measurement.
- Host metadata records Intel Xeon Platinum 8568Y+ and 16 allowed logical
  CPUs in every server/worker process, in both phases. The launcher did not
  confine the two workers to one CPU. CPU clocks and the historical host
  environment remain uncontrolled.
- Total new collection: 400/400 measured requests across the trace,
  lightweight timing and paired control jobs. All three jobs completed 0:0.

## Evidence and remaining work

- Raw traces, scripts, logs and native metrics:
  `${CAMPAIGN_ARTIFACT_ROOT}/gpt-ttft-fix-20260918/` and the same relative
  directory under `${B200_ARTIFACT_ROOT}/results/trt-62-20260917/`.
- [Compact measurements and hashes](gym-gpt-eager-20260918.json).
- Freeze the new measurement contract before formal eager collection.
  Qualify CPU/device composition on independent shapes and complete ranks,
  then rerun GPT-OSS and all shared-code controls. Queue/frontend costs
  identified in the [earlier report](gym-residual-20260918.md) remain separate.
