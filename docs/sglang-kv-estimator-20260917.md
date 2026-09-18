# SGLang KV estimator: static pool and resident weights

The estimator now derives capacity from SGLang's static-pool semantics and model
geometry. It does not import measured server capacities. Performance data and
attention-precision changes remain in [PR #262](https://github.com/ai-dynamo/aisimulate/pull/262).

## Three accounting corrections

1. Peak activation/workspace memory belongs to execution headroom outside the
   static pool. Keep it in diagnostics, but do not subtract it again from KV.
2. SGLang measures free memory after distributed/CUDA setup, before weight
   loading. Apply the static fraction to estimated pre-load free memory, then
   subtract resident weights. With an explicit additional graph reservation,
   the budget is:
   `(GPU capacity - resident overhead) * fraction - weights - extra graph bytes`.
3. Ordinary SGLang DeepSeek-V3/R1 weights must follow checkpoint layer counts and
   TP sharding. R1 has 58 MoE layers and 3 dense MLP layers, and a TP-sharded
   embedding. The latency graph currently approximates all 61 layers as MoE and
   a full embedding lookup. A model-owned weight calculation corrects memory
   without changing that timing graph.

The resident-weight correction is limited to DeepSeek-V3/R1 on ordinary SGLang
TP/EP layouts without CP, PP, or speculative decoding. Other layouts retain their
prior weight accounting. vLLM and TRT-LLM memory formulas are unchanged. Explicit
additional CUDA Graph reservation keeps its existing meaning.

## H200 capacity reconciliation

All ten original H200 configurations estimate the same capacity. Nine logs report
509285 tokens and one reports 509157. The logs are validation evidence only.

| Estimator stage | KV tokens | Difference from 509285 |
|---|---:|---:|
| Original estimator | 57906 | -88.63% |
| Exclude transient activations only | 403850 | -20.70% |
| Correct resident weights and pre-load pool | 497973 | -2.22% |

For TP8, model weight bytes change from 90845872128 (84.6068 GiB) to
85117435904 (79.2718 GiB). The original server reports a 79.80 GiB **free-memory
change during weight loading**, which includes allocations beyond parameter
storage. It is not an exact tensor-byte inventory. Quantization metadata,
allocator behavior, and temporary/resident buffers remain approximations.

The model's resident overhead is 4.5828 GiB, leaving 136.4172 GiB before weight
loading. The original log reports approximately 137.98 GiB at that boundary.
This difference and the loading-footprint difference partly offset each other.
No constant is tuned to remove the remaining capacity residual.

Pinned runtime source checks:

- [SGLang memory-pool profiling](https://github.com/sgl-project/sglang/blob/127b9e3283f7c2a43234b852ff5c9f1796d53624/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L62).
- [SGLang distributed initialization and weight-loading boundary](https://github.com/sgl-project/sglang/blob/127b9e3283f7c2a43234b852ff5c9f1796d53624/python/sglang/srt/model_executor/model_runner.py#L1299).
- [SGLang DeepSeek layer selection and embedding construction](https://github.com/sgl-project/sglang/blob/127b9e3283f7c2a43234b852ff5c9f1796d53624/python/sglang/srt/models/deepseek_v2.py#L1916).
- [Original benchmark 409939](https://inferencex.semianalysis.com/inference/logs/409939).

## Validation contract

The code branch is based on main `2832e6b8756b8b83f18aee2f8e8fcaf23eea8675`.
The two estimator commits are `d21e46a8` and `17686b36`. The standalone PR
contains no performance-data files or attention-precision mapping changes.

For the controlled E2E comparison, a detached checkout starts from data PR #262
at `35050da11ad675ed612e61626b170b2db6c430bd` and applies only these estimator
changes, producing validation revision `cd2f1e38`. This preserves the original
latency model/data while isolating the memory change. The paired baseline is the
84-point mapping replay at runtime `2cfe6f836fa3f2c81786d250125406cc624c1357`.
All replay specs must remain byte-identical; no `num_gpu_blocks` override is added.

The selected tests cover prefill-budget invariance for the SGLang pool,
vLLM/TRT-LLM behavior, explicit extra graph reservations, native capacity and
block conversion, independent R1 tensor-shape accounting at TP1/4/8 and EP1/4/8,
unchanged latency-op counts, and existing large-EP/NVFP4 model graphs.

## Original 84-point replay

All 84 points succeeded: 10 H200, 29 B200, and 45 B300. All 21430 requests
completed with zero truncation. All 84 replay specs are byte-identical to the
baseline. The installed wheel's 279 Python sources match the validation checkout;
profile hashes and native-runtime identity are recorded in the adjacent JSON.

MAPE averages each point's absolute percentage error against its silicon mean.
No points are dropped; TTFT and TPOT are scored separately.

| Hardware | Points | Baseline TTFT MAPE | Fixed TTFT MAPE | Baseline TPOT MAPE | Fixed TPOT MAPE |
|---|---:|---:|---:|---:|---:|
| H200 | 10 | 1612.27% | 33.92% | 19.59% | 10.29% |
| B200 | 29 | 65.15% | 65.15% | 5.61% | 5.61% |
| B300 | 45 | 60.29% | 60.29% | 5.86% | 5.86% |
| All | 84 | 246.73% | 58.83% | 7.41% | 6.30% |

The activation-only intermediate replay at `9f11a386` reached H200
77.74%/9.23% and overall 64.05%/6.18% TTFT/TPOT MAPE. Correcting resident
weights and the pre-load boundary improves TTFT further, but TPOT aggregate
error increases relative to that intermediate arm. Capacity accuracy is not a
latency-fitting objective.

Five H200 points change relative to the original baseline; five H200 and all
74 Blackwell predictions remain unchanged. No TTFT APE worsens. H200 8k/1k
CC8 TPOT APE increases from 2.18% to 9.29%; it is retained in the aggregate.
The historical logged-capacity diagnostic reached H200 33.69%/10.36%, close
to this independently estimated result, but does not supply a production input.

Silicon used SGLang 0.5.12-cu130 for 65 points and 0.5.12.post1 for 19; the
queried performance tables are 0.5.14. This is a controlled estimator comparison,
not a version-matched accuracy claim. The Blackwell prefill timing gap remains
open in the separate data/precision investigation.

Validation: 237 focused unit/integration tests passed, covering the contracts
listed above. Ruff lint/format checks and `git diff --check` passed.

- [Paired point results and provenance](sglang-kv-estimator-20260917.json).
- Raw scripts, manifests, wheel, specs, and native per-request reports:
  `/Users/simonec/.cache/aisim-e2e-gym/mla-fp8-20260917/kv-fixed-84/`.
- Intermediate activation-only artifacts: the sibling `static-pool-84/` directory.
- Replay commands: run `run.py`, `finalize.py`, then `score.py` with the fresh
  `kv-fixed-84/venv/bin/python`. The runners preserve the original selection
  and record input, profile, installed-source, and output hashes.
