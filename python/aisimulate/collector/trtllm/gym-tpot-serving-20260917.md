# GPT-OSS/B200 serving reproduction

## Finding

The reproduced rc14 server uses **W4A8 MXFP4/MXFP8 MoE**, while the frozen
Gym specification explicitly requests **W4A16 MXFP4**. The checkpoint declares
MXFP4 weights; TRT-LLM chooses activation quantization at runtime. On SM100,
`ModelConfig.get_mxfp4_quant_algo()` selects `W4A8_MXFP4_MXFP8` for the TRTLLM
backend unless `OVERRIDE_QUANT_ALGO` is set. The original recipe sets no such
override. Both the real server log and GPU kernel trace confirm that path.

The earlier bias/routing/autotuning experiments used the W4A16 operator.
They establish a valid collector improvement, but do not establish full
GPT-OSS/B200 serving parity. This reproduction supersedes that interpretation.

## Matched point

- Original InferenceX config131: GPT-OSS-120B, one B200, TP1/EP1, ISL/OSL 1024,
  concurrency 256, 2,560 requests, random range ratio 0.8, seed 0.
- Recipe/client: SemiAnalysisAI/InferenceX
  `2baba8e27be8529b4453afc953f9859ec092c73d`; unmodified benchmark client.
  Preserve 512 warmups, ignore-EOS, streaming interval 20, FP8 KV, max batch 512,
  max sequence 2304, max tokens 20000, graph padding through batch 256, and PDL1.
- Model revision: `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`; configuration
  exactly matches the frozen replay checkpoint configuration. CPU job 1994868
  verified all 15 checkpoint shards; completed 0:0.
- rc14 amd 64 image:
  `sha256:fe2f17d0c9698bafeb9f437929003c6badc56d17e8382a02505233145a07f61f`.
  Source `93cb6518b6d6dbd6095748189e626db731f44545`.
- Uninstrumented baseline job 1994872, one B200 on gpu78, completed 0:0 in 6m30s.
  All 2,560 requests succeeded; complete per-request token counts retained.
  Adaptations are local model/output paths, localhost endpoint, detailed output,
  CPU/GPU allocation and storage. No serving algorithm or quantization override.
- The historical model revision and image digest are unavailable. This run
  pins the replay's resolved checkpoint and the current rc14 image digest;
  it does not prove historical artifact identity.

## Results

| Run | TTFT (ms) | TPOT (ms) | TPOT APE versus reproduction |
| --- | ---: | ---: | ---: |
| Historical silicon | 183.61345 | 16.79212 | — |
| Uninstrumented reproduction | 189.62701 | 16.54925 | — |
| Replay, W4A16, exact client lengths | 487.35848 | 65.16758 | 293.78% |
| Replay, W4A8, exact client lengths | 120.92177 | 17.17212 | 3.76% |

The reproduced TPOT is 1.45% below the historical value. The two replay arms
use the same exact 2,560 client input/output lengths, concurrency and scheduler
settings; only `aic_moe_dtype` changes. Runtime remains
`89d2051137b772944a3e303c452108811790bef4`; performance data remains PR#264
`a85b9e10f5117439eafd7a5d113118b781229920`. No source/table mutation or
latency scaling is applied. Corrected TTFT still has 36.23% error.

A separate first ablation preserves the frozen synthetic workload and changes
only precision: TPOT 65.15635→17.17169ms. Its request lengths are identical
between arms but differ from the real client: the frozen runner uses Python
`random.Random` even though the workload contains `length_sampler=numpy_random_state`.
The exact-client-length ablation changes TPOT by only 0.00043ms for W4A8.
This excludes that sampler mismatch as the cause of this point's large gap.

## Kernel and step checks

- The original `print_iter_log` output yields 9,233 pure-decode batch 256
  iterations. Restricting to adjacent pure-decode iterations aligns the
  previous-device event correctly: 8,246 samples, median 14.05802ms.
- Diagnostic job 1994898 completed 0:0 in 5m59s on the same B200 node. Native
  profiler range 3000–3010 produced 11 graph replays and 396 MoE layer instances.
  The diagnostic E2E score is not used as silicon truth.
- Each layer uses native `bmm_MxE4m3_MxE2m1MxE4m3...` followed by
  `bmm_Bfloat16_MxE2m1MxE4m3...`. The latter's BF16 names the output;
  its input is still MXFP8. Quantization kernels are present.
- MoE GEMM, routing, quantization and finalize kernels sum to 10.01317ms
  per decode step, or 0.27814ms per layer. These are summed GPU durations,
  not an asserted full critical path. The selected W4A8 table row is 0.265798ms;
  the W4A16 row is 1.336288ms for the same logical batch 256 shape.
- The initial routing hook produced no snapshots. Its missing result is not
  treated as routing agreement; the retry and its outcome are recorded below.

## Live routing snapshots

- Retry job `1995012` completed `0:0` in 6m16s. It captures router outputs in the decode CUDA graph and
  saves them after live replays 100, 150 and 200. All three samples contain
  all 36 layers at batch 256, with 128 experts and top-k 4 (108 layer samples).
- Across the 108 samples, active experts range from 77 to 116 (median 105);
  the hottest expert receives 45–198 of 256 tokens (median 84.5).
- The model itself reports `W4A8_MXFP4_MXFP8`. Input/output request lengths
  retain the original benchmark sampling; this diagnostic omits the 512 client
  warmups and is not used for the E2E accuracy comparison.
- Reconstructed top-k histograms are retained with every snapshot. Native
  tie-breaking may differ from `torch.topk`; no independent routing-distribution
  latency contribution is claimed. Full logits remain in the raw artifacts.

## Scope and next correction

This confirms the activation-precision mapping defect for the reproduced TP1
point. It does not produce a new 31-point or 263-point MAPE. The automatic Gym
mapping still needs a backend/version/architecture-aware default rule, honoring
explicit runtime overrides. Before the full rerun, collect missing W4A8 TP2/4/8
profiles under their correct keys; do not relabel the prior W4A16 measurements.
Hopper defaults and non-TRT backends must remain independently resolved.

## Evidence

- Source rule: https://github.com/NVIDIA/TensorRT-LLM/blob/93cb6518b6d6dbd6095748189e626db731f44545/tensorrt_llm/_torch/model_config.py#L342-L353
- Original recipe: https://github.com/SemiAnalysisAI/InferenceX/blob/2baba8e27be8529b4453afc953f9859ec092c73d/benchmarks/single_node/gptoss_fp4_b200_trt.sh
- Raw client/server results, telemetry, profile, replay reports and runners:
  `${CAMPAIGN_ARTIFACT_ROOT}/serving-repro/`.
- Remote: `${B200_CLUSTER}`,
  `${B200_ARTIFACT_ROOT}/results/trt-62-20260917/serving-repro/`.
- `comparison.json`, `trace-summary.json`, `manifest.json`, `model-manifest.json`
  and `client-lengths-replay.json` retain the numeric results and provenance.

[Machine-readable evidence](gym-tpot-serving-20260917.json) records paired
metrics, routing summaries, source identity, and artifact hashes.

## Full correction follow-up

The mapping, W4A8 collection, and complete 263-point rerun are recorded in
[the 2026-09-18 report](gym-w4a8-20260918.md). Earlier metrics above remain
historical ablations.
