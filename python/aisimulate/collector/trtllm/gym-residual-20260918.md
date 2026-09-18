# TRT residual diagnosis after activation correction

## Status

- Review cleanup is complete: final MoE provenance, NumPy seed bounds, and
  artifact-location aliases. The three changes do not alter the seed-0
  prediction results in the [263-point report](gym-w4a8-20260918.md).
- Qwen scheduling and kernel-selection contracts remain incomplete. The
  evidence below supports a scheduling contribution; it does not quantify the
  contribution or establish a corrected Qwen MAPE.

## Qwen: source controls and mixed-step amplification

- Frozen point: config1553, B200, attention DP8 / EP8, ISL8192, OSL1024,
  concurrency512, 5,120 completed requests and zero readmissions.
- Historical TPOT **35.0937 ms**, replay **116.8493 ms**. Replay ITL median
  **21.6066 ms**, p75 **315.7660 ms**, p90 **349.6741 ms**.
  Historical TTFT **2,915.4292 ms**, replay **1,399.7996 ms**.
- Controlled native per-op probes, using the same model, topology and rc20
  data: 64 decode requests cost **21.2968 ms**; 8,192 prefill tokens plus
  63 decode requests cost **348.0735 ms**; 16,384 prefill tokens plus
  62 decode requests cost **686.1578 ms**. These are model probes, not
  observed silicon batches. Their agreement with the replay tail supports
  investigating prefill interruptions before treating the full gap as a
  pure-decode kernel error.
- The immutable [InferenceX recipe](https://github.com/SemiAnalysisAI/InferenceX/blob/f9426a550344a64a9d9b11fc58f10cb5c61d015e/benchmarks/single_node/fixed_seq_len/qwen3.5_fp4_b200_trt.sh)
  explicitly disables chunked prefill and selects the controls below.
  Gym's resolved source arguments retain them, but the frozen engine spec
  does not express them.

| Recipe scope | Effective source control | Replay limitation |
|---|---|---|
| Seven attention-DP points | Balance enabled, batching wait 10 iterations, timeout 500 iterations | Missing TRT cross-rank prefill waiting |
| Twelve non-DP points | Batch wait timeout 50 iterations, token threshold 0.45 | Missing TRT token-threshold batching wait |
| Seven attention-DP points | CUTEDSL MoE, low-precision combine | Kernel selection absent from standard replay identity |
| All 19 points | MAX_UTILIZATION capacity policy | Engine supports GUARANTEED_NO_EVICT only |
| All 19 points | Serving rc18 | Frozen performance data is rc20 |

- Pinned [TRT rc18 scheduling implementation](https://github.com/NVIDIA/TensorRT-LLM/blob/15d06c0923b63ac1781784d5f59e1747bb47d5f1/tensorrt_llm/_torch/pyexecutor/py_executor.py#L3682)
  confirms that balanced DP prefill can wait for context requests on all
  ranks, then apply the configured batching delay while generation continues.
  Non-DP batching uses a token threshold and iteration timeout. A fixed vLLM
  prefill cadence is not an equivalent implementation.
- Next discriminating check: capture matched rc18 per-rank batch composition
  and CUTEDSL timings, then implement the supported scheduling contract and
  rerun all 19 points. The capacity-policy difference is a confirmed config
  mismatch; its numerical effect is not established by zero readmissions.

## GPT-OSS: matched TTFT reproduction and timing boundaries

- Slurm job **1996181**, two B200 GPUs, completed `0:0` in 9m27s.
  Config155, TP2/EP1, ISL1024/OSL1024, concurrency4; each repetition has
  eight warmup and 40 measured requests, NumPy seed0 and length ratio0.8.
  All **240/240 measured requests** across six repetitions completed.
  Every input and output length matches the frozen replay request vector.
- The previous reproduction's rc14 image and checkpoint were reused. Pin
  the image child digest
  `fe2f17d0c9698bafeb9f437929003c6badc56d17e8382a02505233145a07f61f`,
  checkpoint `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, and InferenceX client
  `2baba8e27be8529b4453afc953f9859ec092c73d`. Runtime confirms rc14,
  Torch 2.11.0a0+eb65b36914.nv26.02, CUDA 13.1 and SM100.
  The historical recipe did not pin its image digest or checkpoint revision;
  the historical host environment is not recreated exactly.
- Baseline keeps native request metrics disabled. A separate server enables
  `return_perf_metrics` and retains all 48 request records per repetition.
  Sort by server arrival, exclude the first eight warmups, and verify that
  every warmup finished before any measured request arrived.

| Observation | Mean TTFT (ms) | Mean TPOT (ms) |
|---|---:|---:|
| Historical observation | 48.3129 | 2.8212 |
| Frozen corrected replay | 16.5326 | 2.9374 |
| Uninstrumented repeat 1 | 50.3386 | 2.8087 |
| Uninstrumented repeat 2 | 48.4294 | 2.8042 |
| Uninstrumented repeat 3 | 47.8015 | 2.8132 |
| Uninstrumented repeats, equal-weight mean | 48.8565 | 2.8087 |
| Native-metrics repeats, equal-weight mean | 50.5588 | 2.8374 |

- The historical TTFT residual reproduces; TPOT remains close to silicon.
  Instrumented runs average 3.48% higher TTFT than the uninstrumented runs,
  so retain them as diagnostic evidence rather than replace the baseline.
- Native timing means over 120 measured diagnostic requests:

| Boundary | Mean time (ms) |
|---|---:|
| Server arrival to executor arrival | 2.6915 |
| Executor arrival to first scheduled iteration | 5.6543 |
| First scheduled iteration to first token | 34.9355 |
| First token to server response timestamp | 2.0356 |
| Server arrival to server response | 45.3170 |
| Client mean minus server mean | 5.2418 |

- The client/server difference is a difference of means, not an ID-paired
  transport measurement. Native timestamps and response boundaries follow
  [rc14 openai_server.py](https://github.com/NVIDIA/TensorRT-LLM/blob/93cb6518b6d6dbd6095748189e626db731f44545/tensorrt_llm/serve/openai_server.py#L830).
- Each diagnostic repetition has 38 distinct first-prefill iterations for
  40 requests. Match `first_iter` to rank0 iteration logs and use the next
  iteration's `prev_device_step_time`, as defined by the pinned
  [ping-pong CUDA-event logger](https://github.com/NVIDIA/TensorRT-LLM/blob/93cb6518b6d6dbd6095748189e626db731f44545/tensorrt_llm/_torch/pyexecutor/py_executor.py#L926).
  Mean prefill-step device intervals are **33.44 / 31.40 / 31.75 ms**.
  They can include host launch gaps and synchronization; they are not sums
  of GPU kernel durations. Frontend-only overhead cannot explain the full gap.
- [Matched two-rank traces and CPU timing](gym-gpt-eager-20260918.md) now
  support eager submission gaps and peer waiting as a missing boundary.
  The user deferred the execution-mode/CPU-timing extension; the prediction
  correction remains open.
  Adding a constant TTFT offset is not justified by this single point.
- [Timing summary, per-request diagnostic durations and artifact hashes](gym-residual-gpt-20260918.json).
  Remote results: `${B200_ARTIFACT_ROOT}/results/trt-62-20260917/residual-20260918/gpt-ttft/`;
  logs: `${B200_ARTIFACT_ROOT}/logs/gpt-ttft-tp2-0918-1996181.{out,err}`.

## Evidence retention

- Raw artifacts: `${CAMPAIGN_ARTIFACT_ROOT}/residual-20260918/`.
- [Machine-readable Qwen probes and hashes](gym-residual-qwen-20260918.json).
- Runtime, data and original-cohort identities remain those in the linked
  full replay report. No performance rows or prediction values were changed.
