<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MiniMax-M3 / AgentX 439922 — B200 vLLM reproduction and results

Native vLLM FPM-off/on comparison on the same four allocated B200 GPUs, with **G2 disabled in both cases**. The image updates [FPM PR #52061](https://github.com/vllm-project/vllm/pull/52061) to `b3563fc65ae0f5359802593d78e7ea097e1fed31`, preserving the previous Dynamo nightly's vLLM/CUDA binaries and isolated AIPerf installation. It does not use Dynamo's `InstrumentedScheduler`.

## Configuration

The reference is [AgentX 439922](https://inferencex.semianalysis.com/inference/agentic/439922). Configuration was checked against the [published API](https://inferencex.semianalysis.com/api/v1/benchmarks?model=MiniMax-M3), its [server log](https://inferencex.semianalysis.com/api/v1/server-log?id=439922&file=results%2Fserver.log), and the [InferenceX launcher](https://github.com/SemiAnalysisAI/InferenceX/blob/5c3e65cf4c59db9966a9b16eb0035702bc5cf692/benchmarks/single_node/agentic/minimaxm3_fp4_b200_mtp.sh). [reference-point.json](reference-point.json) preserves the retrieved point.

The API labels speculation as `mtp`, but the actual server uses **EAGLE3-GQA**, not MiniMax's native MTP head. Reproduce the actual target/draft pair and command.

| Setting | This experiment |
| --- | --- |
| Hardware / topology | 4 B200, TP4, DP1, PP1, no expert-parallel flag |
| Target | `nvidia/MiniMax-M3-NVFP4`, revision `901464083161bf8612a29ff7ad29914cd4ab4a85` |
| Draft | `Inferact/MiniMax-M3-EAGLE3-GQA`, revision `96692486b5fd38ebf8fd2a5f6bb53427d30819a8` |
| Speculation | `eagle3`, 3 speculative tokens, draft `FLASH_ATTN`, synthetic acceptance length 2.78 |
| G2 / weight CPU offload | No KV connector; no CPU KV cache; `--cpu-offload-gb 0` |
| Frontend / scheduler | Native Python API server and native vLLM scheduler; V1 model runner, matching the reference execution path |
| Model mode | Language-model-only; thinking enabled; native `minimax_m3` tool/reasoning parsers |
| Attention | `FLASHINFER`, TRT-LLM attention enabled, FP8 indexer cache, Triton sparse MSA decode |
| GPU KV | FP8, block size 128, prefix caching enabled, max context 1,048,576 |
| Batching / graphs | Max batched tokens 16384; max CUDA graph capture size 512; native model-specific graph setup |
| Streaming | `--stream-interval 20`, as in the reference |
| Memory / all-reduce | GPU memory utilization 0.9; FlashInfer all-reduce backend `trtllm` |
| Replay | Weka 393 entries; c15, seed 42; 3600 measured seconds per case |
| Warmup / trajectory | 10 warmup requests/lane, grace 1800 seconds; trajectory start 0.25–0.75; idle-gap caps 300/10 seconds |
| Client behavior | Streaming chat; server token counts; `ignore_eos=true`; first-turn-prefix cache bust |
| Client revision | SemiAnalysisAI/aiperf `754356e9a39acc6cc6afb242d123bb57c3fb6f75`, inherited unchanged in `/opt/agentx-aiperf` |
| Native FPM | Job-scoped port, one DP-rank-0 publisher for the TP4 engine; one buffered recorder |

The reference uses vLLM `0.27.2rc1.dev77+gac7509e2b`; this image retains nightly vLLM `0.28.0` at `2cf0a6915ce544dc493a0990f2ea38d81601128a`. The reference server's checkpoint was a local directory with `revision=None`, so its exact weight revision is not independently established. Synthetic acceptance tests performance, not model quality. These boundaries apply to SA comparisons even though both local cases share the same image and weights.

## Reproduce

1. Reuse the target and draft in `/home/scratch.hongkuanz_gpu/models/hub/`. [checkpoint-result.json](checkpoint-result.json) records the immutable revisions, paths and verified file sizes: target 88 weight shards / 250,137,296,832 total bytes; draft one weight file / 6,149,993,396 total bytes. CPU-only download job `4244099` completed `0:0`. No previous model was deleted. Weight licenses are retained by the HF cache; weights and model code are not vendored here.
2. If the checkpoints are absent in another scratch space, adapt the paths in [download-checkpoints.py](download-checkpoints.py), then run [submit-download.sh](submit-download.sh). It requests eight CPUs, 32 GiB RAM and **zero GPUs**, queries metadata at the pinned revisions, resumes downloads and validates every file's size plus the indexed weight shards. Preserve the backing HF blobs.
3. Use image `nvcr.io/nvidian/dynamo-dev/vllm-agentx@sha256:2aab1ad231c052eacc53f430b37261b01b31abc1319cd72e0e5cf9936292199a`. The Slurm cache is `/home/scratch.hongkuanz_gpu/images/vllm-agentx-fpm-b3563fc65a-minimax-layout-amd64.sqsh`, SHA256 `673d21cdb48ccbf0cb7610c5e31e1784d44104b1cc85444d1f830a51060ffe3f`. This includes the singleton FP8 descale compatibility fix described below. See [image-published.json](image-published.json).
4. Stage these scripts, the image manifest and checkpoint result in `/home/scratch.hongkuanz_gpu/minimax-m3-439922-vllm-20260911/`; adapt user/account/partition paths as needed. Submit:

```bash
ssh hongkuanz@computelab-sc-01 \
  'sbatch /home/scratch.hongkuanz_gpu/minimax-m3-439922-vllm-20260911/submit-benchmark.sh'
```

[submit-benchmark.sh](submit-benchmark.sh) requests **4 GPUs, 112 CPUs, 768 GiB host RAM and four hours**. Host RAM is for model loading, file cache and client work, not G2. This is not a whole-node exclusive allocation; other workloads may use the remaining GPUs/host resources. Record this limitation when interpreting small performance differences. API and FPM ports are `20000 + job_id % 10000` and `10000 + job_id % 10000`, with availability checked before each case; the exact ports are saved in `protocol.json`.

[benchmark.py](benchmark.py) verifies the checkpoint manifest and four CUDA-visible B200 devices, runs a short-lived GPU probe so the controller retains no CUDA context, saves allocation/topology/command metadata, and checks the native CLI has no KV connector or CPU offload. `VLLM_PLUGINS=""` prevents bundled Omni plugins from replacing native vLLM classes. The script unblocks inherited `SIGCHLD`, uses a job-local AIPerf mmap cache, and starts load only after health and chat smoke checks.

The order is off then on, each with a fresh engine/KV cache and the same warmup and 3600-second measured phase. Compilation caches may be reused within the allocation. The on case subscribes before engine startup, checks rank 0 and active decode after smoke, then preserves raw FPM/checksum before server teardown. Final validation checks received counters and metric sanity; publication remains best-effort and is not an exact GPU-length oracle.

Results appear under `/home/scratch.hongkuanz_gpu/agentx-minimax-m3-results/job-<JOB_ID>/`: exact server/client commands, `protocol.json`, allocated GPU identities, AIPerf exports/logs, `on/fpm.jsonl`, checksum and validation. Reproduce the comparison on a workstation, not on measured GPUs:

```bash
uv run --no-project analyze.py /path/to/job-ID --reference-api reference-point.json
```

### Rebuild the updated image

[Dockerfile](Dockerfile) extends the previous image at `b58174af2daaf4d02c4845fc90dd3f18615fc7c3ef930fb852ed0d29e76f1d12`. [parent.sha256](parent.sha256) checks every changed parent file before applying [update-fpm.patch](update-fpm.patch) without fuzz. CUDA binaries and the isolated client environment are unchanged; `multiprocess==0.70.18` is added only to the FPM test environment for the upstream shared-memory test.

The native FPM module is byte-identical to PR revision `b3563fc`; compatibility edits preserve older core/Ray imports and the older locations of CLI and scheduler hooks. The update includes corrected async SD decode lengths, current prefill attention variance, `torch.Event`, and nonblocking tail-timing polling that preserves ordered multiprocessing RPC responses. `var_prefill_length` now means the population variance of `kv_read + scheduled_query_tokens / 2` in a scheduled prefill batch, unlike the older DSv4 capture in this branch.

Run [submit-image.sh](submit-image.sh) and [prepare-image.sh](prepare-image.sh) inside a one-B200 allocation. They build, test, publish and export the named image; use a new tag/cache path for a new revision rather than overwriting recorded artifacts. [cli-preflight-node.sh](cli-preflight-node.sh) performs CLI and real CUDA-event smoke checks inside the same allocation. The checked-in test differs from the upstream test only by accepting this older base's earlier `model_config` validation error.

### Singleton FP8 descale compatibility

The raw FPM-updated image hit a first-request FA4/CuTe ABI error with the GQA draft: `Expected strides[leading_dim] == 1, but got 0`. A scalar expanded to `(1,1)` has strides `(0,0)`; PyTorch considers it contiguous, so `.contiguous()` does not canonicalize the leading stride. The draft has four KV heads globally and one per TP4 rank, making this shape reachable. The failure also occurs with FPM disabled and is unrelated to FPM state collection.

[Dockerfile.descale](Dockerfile.descale) extends the FPM image at `ab1c5a6e1b2b63743c8a3f8ec4621ca244f049f754cdba8f5ff4b96a57d067c9` with [flash-attn-descale.patch](flash-attn-descale.patch): three scale expressions become `scale.view(-1).expand(descale_shape)`. This preserves storage and values, gives the singleton head dimension stride 1, and adds no data copy, D2H or CUDA kernel. It retains the reference's FA4/GQA algorithm rather than selecting another attention backend. The FPM module remains byte-identical to `b3563fc`. This image-only compatibility patch is separate from PR #52061 and is applied identically to both cases.

[prepare-layout-image.sh](prepare-layout-image.sh) builds/tests/exports this final layer in a one-B200 allocation. [flash-attn-parent.sha256](flash-attn-parent.sha256) guards the original vLLM backend file. [test_flash_attn_descale.py](test_flash_attn_descale.py) checks view aliasing, reproduces the unmodified singleton ABI failure, and verifies real FA4 outputs at batch 1/2 match dense descale storage **bitwise**.

## Results and validation

The off baseline from job `4244685` completed 3600 measured seconds: 2,491 successful requests, one client `Broken pipe` error, zero output-length mismatches/cancellations, and `submission_valid=true`. Its wrapper nevertheless exited `FAILED 1:0` because an additional zero-error assertion ran after the valid export. The assertion has been removed: the harness now follows AIPerf's validity result and preserves every reported error in `client-validation.json`, without changing the client or retry policy.

To avoid repeating a valid hour, supplementary **on-only job `4245705`** runs the same configuration on `umbriel-b200-048`. These are **separate allocations and different GPU UUID sets**, not a strict same-allocation pair; fresh compilation caches and changes in co-tenant activity also limit attribution of small differences to FPM. The on job's `baseline-link.json` preserves this boundary and the original baseline error/status. Its performance/FPM results are pending.

The recovery invocation is:

```bash
sbatch --nodelist=umbriel-b200-048 \\
  --export=ALL,MINIMAX_CASES=on,MINIMAX_BASELINE_JOB_ID=4244685 \\
  /home/scratch.hongkuanz_gpu/minimax-m3-439922-vllm-20260911/submit-benchmark.sh
```

For a new controlled pair, leave `MINIMAX_CASES` unset to run both cases in one allocation. To analyze the preserved split runs:

```bash
uv run --no-project analyze.py /path/to/job-4245705 \\
  --baseline-root /path/to/job-4244685 --reference-api reference-point.json
```

[monitor.sh](monitor.sh) is a read-only workstation monitor that queries exact-job `sacct` state (including when `squeue` is empty), prints final campaign/capture evidence, and stops at a supplied UTC deadline. Log scans alone must not be treated as proof that a job is still running.

Preparation validation:

- Build job `4244288`: 22 focused FPM/shared-memory tests passed in the matching image, including nonblocking tail collection, new-request arrival, ordered RPC responses, SD corrections and prefill variance.
- Native off/on CLI preflight passed with no G2 or weight CPU offload.
- A real B200 `torch.Event` timing smoke passed; it is not a model/performance result.
- Final layout-image build `4244616`: 27 tests passed, including the above FPM/RPC regressions and real FA4 descale equivalence; allocation completed `0:0`.
- The changed-file source/response-order review found no new compatibility issue. No additional inference CUDA synchronization was introduced by this port.

One ordered pair measures an observed difference, not universal or statistically proven zero overhead. Report request errors, drain cancellations, metric-duration coverage and publisher/capture limitations alongside performance; do not conflate `was_cancelled=false` with zero cancelled requests.

## Source and attribution

Serving/replay configuration is adapted from SemiAnalysisAI/InferenceX `5c3e65cf4c59db9966a9b16eb0035702bc5cf692`, `benchmarks/single_node/agentic/minimaxm3_fp4_b200_mtp.sh` and `benchmarks/benchmark_lib.sh`. Apache-2.0; copyright 2025 SemiAnalysis LLC, Advanced Micro Devices, NVIDIA CORPORATION. Modifications include immutable checkpoint/image selection, G2-off validation, paired native FPM capture, Slurm allocation and artifact preservation.

The patch and tests/subscriber derive from vllm-project/vllm PR #52061 at `b3563fc65ae0f5359802593d78e7ea097e1fed31`, updating the earlier `996fed467139edd7719a0063d57709b8a7fa6989` compatibility port on base `2cf0a6915ce544dc493a0990f2ea38d81601128a`. Apache-2.0; copyright contributors to the vLLM project. The root and packaged `THIRD_PARTY_NOTICES.md` record the source paths and modifications. Model downloads retain the target's [MiniMax Community License](https://huggingface.co/nvidia/MiniMax-M3-NVFP4/blob/901464083161bf8612a29ff7ad29914cd4ab4a85/LICENSE) and the draft's separately distributed licenses.
