<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Pro / AgentX 440246 — B200 vLLM reproduction and results

This experiment compares native vLLM forward-pass metrics (FPM) disabled and enabled on the same eight-B200 node. The FPM implementation is [vLLM PR #52061](https://github.com/vllm-project/vllm/pull/52061), revision `996fed467139edd7719a0063d57709b8a7fa6989`, including async speculative-decoding length and timing corrections. It does not use Dynamo's `InstrumentedScheduler`.

## Reference and experiment configuration

The reference is [AgentX 440246](https://inferencex.semianalysis.com/inference/agentic/440246), using the [InferenceX B200 launcher](https://github.com/SemiAnalysisAI/InferenceX/blob/4552491d40b179c3323a3485c63090e5b8c964ad/benchmarks/single_node/agentic/dsv4_fp4_b200_vllm_mtp.sh). The serving command, not the UI's TP labels alone, determines the topology.

| Setting | Our paired experiment |
| --- | --- |
| GPUs and parallelism | One node, 8 B200; TP1, DP8, EP8, PP1 |
| Model | `deepseek-ai/DeepSeek-V4-Pro`, cached revision `b5968e9190ef611bbf34a7229255be88a0e937c1` |
| Runtime | Latest Dynamo vLLM nightly resolved September 10, 2026; native FPM compatibility port |
| vLLM base | `2cf0a6915ce544dc493a0990f2ea38d81601128a` (`0.28.0`); not a wholesale checkout of the PR's newer base |
| Frontend and router | Native vLLM Rust frontend; `vllm-router==0.1.14`, DP-aware consistent hashing, correlation ID mapped to session ID |
| Attention and KV | `FLASHINFER_MLA_SPARSE_DSV4`, FP8 KV, quantized prefill queries, FP4 indexer cache, block size 256 |
| GPU memory and context | Utilization 0.90; max model length 1,048,576 |
| CPU KV offload (G2) | Native `SimpleCPUOffloadConnector`, eager offload, cross-layer blocks; **128 GiB/rank, 1 TiB total** |
| MoE | Current nightly's `deep_gemm_mega_moe` FP4-expert backend |
| Speculative decoding | Native MTP, 3 speculative tokens, synthetic acceptance length 2.49 |
| Prefill scheduling | Interval 8; long-prefill threshold 512; max batched tokens 8192/rank |
| Decode scheduling | Max 16 sequences/rank; full decode CUDA graphs at 4, 8, ..., 64 tokens; compilation mode 0 |
| Prefix retention | `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768`, matching the reference launcher |
| Load | AgentX, 64 replay lanes, 3600 measured seconds per case, seed 42 |
| Dataset and replay | Weka 393 entries; trajectory start 0.25–0.75; 10 warmup requests/lane; first-turn-prefix cache bust; `ignore_eos=true` |
| Client | SemiAnalysisAI/aiperf `754356e9a39acc6cc6afb242d123bb57c3fb6f75`, isolated in `/opt/agentx-aiperf` |
| FPM-on | Native TCP publishers at ports 20380–20387; separate buffered recorder; scope `model_step_cuda` |

The reference CPU KV budget is 356,125,000,000 bytes/rank, or 2,849,000,000,000 bytes total (approximately 2.59 TiB). Available eight-B200 Computelab hosts have about 1.97 TiB usable host memory; the user approved reducing G2. Both local cases use exactly 137,438,953,472 bytes/rank. This and the newer runtime prevent claiming an exact reproduction of the reference's cache capacity or performance.

The reference's old `deep_gemm_amxf4_mega_moe` name is replaced by `deep_gemm_mega_moe` in this nightly. The historical `VLLM_DSV4_MEGA_FP8_COMBINE` and `VLLM_RPC_TIMEOUT` environment options are not consumed by this base; no equivalence to those old implementation switches is assumed. Synthetic acceptance is a performance configuration, not model-quality validation. The selected checkpoint is explicit, but the reference server used a local directory and does not independently prove its exact weight revision.

## Reproduce

1. Reuse the cached checkpoint at `/home/scratch.hongkuanz_gpu/models/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/b5968e9190ef611bbf34a7229255be88a0e937c1`. The benchmark verifies all 64 indexed weight shards exist and are nonempty. Preserve the backing Hugging Face blobs; do not redownload the approximately 805 GiB checkpoint.
2. Use the immutable image `nvcr.io/nvidian/dynamo-dev/vllm-agentx@sha256:b58174af2daaf4d02c4845fc90dd3f18615fc7c3ef930fb852ed0d29e76f1d12`. [image-published.json](image-published.json) records the nightly base, patch revision and validation. The scratch image cache is `/home/scratch.hongkuanz_gpu/images/vllm-agentx-fpm-996fed4671-amd64.sqsh`, SHA256 `660a517a6589189c15aac602d53b11f99eac208aad3e39029ee5f607c0de9b44`.
3. Stage this directory's files in `/home/scratch.hongkuanz_gpu/vllm-fpm-sd-20260910/`. Adapt scratch paths, account and partitions for another user. Do image imports, builds and heavy checks inside a compute allocation, never on the login frontend.
4. Submit [submit-benchmark.sh](submit-benchmark.sh):

```bash
ssh hongkuanz@computelab-sc-01 \
  'sbatch /home/scratch.hongkuanz_gpu/vllm-fpm-sd-20260910/submit-benchmark.sh'
```

The job requests an exclusive eight-B200 node, 224 CPUs, all host RAM, and four hours. It runs FPM-off then FPM-on, each with a fresh engine/router/KV cache and the same warmup and measured duration. Compilation caches may be reused within the allocation. Results are written to `/home/scratch.hongkuanz_gpu/agentx-vllm-results/job-<JOB_ID>/`; existing job directories are never overwritten.

[benchmark.py](benchmark.py) captures the hardware, topology, NUMA mapping, exact server/router/client commands, protocol and client package versions. It starts AIPerf only after server/router readiness and a successful chat smoke request. The on case starts its recorder before the engine, verifies all eight publisher ranks, and copies the capture and SHA256 to shared scratch before engine teardown. Review the final client exports, recorder counters, FPM validation and Slurm exit code before interpreting a run as successful.

### Required compatibility settings

- Set `VLLM_PLUGINS=""` in every engine process. This text-only experiment needs no external vLLM plugins. The bundled vLLM-Omni registration plugin otherwise replaces the native 16-field `EngineCoreOutput` with a 19-field Omni variant, which the Rust frontend cannot decode. [protocol-preflight.py](protocol-preflight.py) checks the native schema before loading weights. The underlying import-time replacement is visible in [vLLM-Omni's patch](https://github.com/vllm-project/vllm-omni/blob/v0.26.0rc1/vllm_omni/patch.py).
- Pass explicit per-GPU `--numa-bind-nodes`, read from PCI sysfs. Automatic NUMA discovery can be skipped under Slurm's CPU affinity; do not disable binding or assume a universal GPU/socket mapping.
- Set `VLLM_ENGINE_READY_TIMEOUT_S=3600`. Weight loading, JIT compilation and allocating 1 TiB of G2 can exceed the frontend's 600-second default.
- Unblock inherited `SIGCHLD` before starting AIPerf and use a job-local mmap cache. This is a preventive client startup setting, not proof that every dataset reconstruction hang is fixed.

### Rebuild the image

[Dockerfile](Dockerfile) pins the original nightly amd64 manifest and installs the isolated client and native FPM implementation. [base.sha256](base.sha256) guards the unmodified upstream files before applying [native-fpm.patch](native-fpm.patch) without fuzz. It includes the standalone subscriber and buffered recorder; its default `bash` command allows Enroot import.

Inside an allocated compute node, from this directory:

```bash
docker build --build-arg FPM_PR_REVISION=996fed467139edd7719a0063d57709b8a7fa6989 \
  -t vllm-agentx-fpm-local .
```

The compatibility port preserves the nightly's compiled CUDA extensions and engine base. Only Python FPM modules/hooks are backported; this image is not the full upstream PR checkout. Rebuilding can resolve different unpinned transitive client/test dependencies, so use the published digest for the recorded experiment.

## Results and validation

Full DSv4 FPM-off/on performance results are pending; no valid measured pair has been produced yet. Job `4228930` was submitted September 10, 2026 with the compatibility settings above. This is submission provenance, not a claim of current Slurm state; check its logs and final exports.

| Validation | Result |
| --- | --- |
| Changed-file PR pre-commit checks | Passed |
| Focused native FPM tests in matching nightly compatibility image | 20 passed |
| Native IPC plugin preflight, job 4228882 | Unfiltered plugins reproduced the 19-field Omni schema; empty allowlist restored native 16-field output; all 20 FPM tests passed again; allocation completed 0:0 |
| B200 native MTP functional smoke | Qwen3.5-0.8B; V2; 3 speculative tokens; decode CUDA graphs; 24/24 requests successful |
| Smoke capture | 242 records, 238 active, 234 decode; positive active timings/KV sums; zero observed gaps, resets or rejected messages |

The smoke used the Python frontend and does not establish Rust-frontend, DSv4, DP8/EP8 or large G2 performance. The new SD correction adds CPU bookkeeping but no extra D2H, GPU kernels, CUDA event pairs or hot-path CUDA synchronization relative to the earlier PR revision. FPM itself still uses CUDA events and asynchronous publication. `model_step_cuda` spans execution through sampling/drafting on the recorded stream; it is not GPU kernel-busy time or client TTFT/ITL. One ordered off/on pair cannot establish statistical zero overhead.

## Source and attribution

Serving/replay configuration is adapted from SemiAnalysisAI/InferenceX `4552491d40b179c3323a3485c63090e5b8c964ad`, `benchmarks/single_node/agentic/dsv4_fp4_b200_vllm_mtp.sh` and `benchmarks/benchmark_lib.sh`, Apache-2.0. Copyright 2025 SemiAnalysis LLC, Advanced Micro Devices, NVIDIA CORPORATION. Changes include the pinned Dynamo runtime, reduced G2, native FPM pair, recording, validation, NUMA/plugin compatibility and allocation cleanup.

The FPM patch, subscriber and tests derive from vllm-project/vllm PR #52061 at `996fed467139edd7719a0063d57709b8a7fa6989`, adapted onto `2cf0a6915ce544dc493a0990f2ea38d81601128a`, Apache-2.0, copyright contributors to the vLLM project. The root and packaged `THIRD_PARTY_NOTICES.md` identify the files and revisions. Runtime packages retain their separately distributed licenses. No checkpoint, trace dataset, credentials or raw request content is vendored here.
