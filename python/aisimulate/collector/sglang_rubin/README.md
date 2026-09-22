<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Vera Rubin SGLang pilot

This directory contains isolated collectors for `nvidia/GLM-5.2-NVFP4` on SM107, using one frozen Dynamo CI runtime image. The serving target is one four-GPU node, TP4/EP1, with data-parallel attention disabled. GEMM measures the checkpoint's BF16 projections; MoE measures its NVFP4 experts; DSA measures full and shared-index layers plus the three existing sparse-kernel tables. All-reduce uses a separate four-rank launcher outside the nine-op model entrypoint.

The collectors reuse the shared case plan, executor, checkpoints, error records, perf schemas, parquet finalization, and `collection_meta.yaml` transaction. Shared integration includes producer hash/provenance registration and the separately approved DSA publication/resume repair: worker filename selectors are distinct from physical output ownership, so full and skip-indexer producers publish into the same two canonical tables with joint provenance. This also corrects the two stock SGLang skip-indexer registrations. Stock version routing, framework manifests, performance schemas, and kernel execution remain unchanged by that shared repair.

**Historical eager-prefill baseline, 2026-09-17:**

- The exact image passed ARM64/SM107 inventory and AISimulate/native-extension import preflight ([job 443897220](https://gitlab-master.nvidia.com/dl/jet/ci/-/jobs/443897220)).
- The frozen serving configuration below passed all 492 measured requests: input lengths 1,024/8,192/32,768, concurrency 1/8/32, and exactly 500 output tokens ([job 443959525](https://gitlab-master.nvidia.com/dl/jet/ci/-/jobs/443959525)). Each point used four measured requests per client after two matching-workload warmup requests per client. Prompts reuse one repetitive exact-length string per input length with prefix caching disabled; this is a bounded synthetic timing baseline.
- Compute collection produced 518 BF16 GEMM and 81 NVFP4 MoE rows ([job 444037945](https://gitlab-master.nvidia.com/dl/jet/ci/-/jobs/444037945)). After correcting the collector's native dense Tensor handling, the DSA rerun passed all 388 planned rows: 152 full-context, 152 skip-context, 42 full-generation, and 42 skip-generation rows ([job 444073831](https://gitlab-master.nvidia.com/dl/jet/ci/-/jobs/444073831)). All 14 outer cases completed with zero errors and empty failed/attempted ledgers; four installed native source files matched the audited revision.
- The shared publication repair passed independent review and 654 tests. The completed DSA run confirmed full/skip union provenance for both canonical tables. After correcting the standalone communication sidecar to use the canonical filename stem, a fresh run passed all 14 communication rows ([job 444098306](https://gitlab-master.nvidia.com/dl/jet/ci/-/jobs/444098306)). All five tables, totaling 1,001 measured rows, passed strict native source admission with their original sidecars and content hashes.
- CPU replay completed all nine workloads and 492 requests, but failed the proposed pilot gate of at most 20% absolute relative error for each metric at every point. Throughput passed 9/9 points (errors -18.8% to +14.6%), TTFT 4/9 (-45.0% to -10.3%), and TPOT 6/9 (-11.9% to +31.2%). Only the three 32,768-token workloads passed all three metrics. Errors are `100 * (prediction / serving - 1)`; no serving-result calibration was applied.
- Seventeen hardware-profile stress variants completed 153 workloads and 8,364 replay requests. No predicted metric changed by as much as 1.642% from baseline; halving or doubling either inferred network bandwidth left predictions unchanged within relative tolerance `1e-9`. These tests do not explain or resolve the accuracy gaps.

The historical collectors and external data established the initial bounded pilot. The following historical end-to-end results remain separate from the graph-forward qualification described below; progress is tracked in the [implementation plan](https://linear.app/nvidia/document/glm-52-nvfp4-on-vera-rubin-aisimulate-implementation-plan-c3a86136763e). The experimental hardware profile and CPU replay script passed independent review. The accepted tables and hardware assumptions are now packaged for the bounded graph-forward profile; raw collection evidence and the historical replay campaign remain external artifacts. These results do not establish general VR200 support, and a CPU `--plan-only` run is not GPU qualification. The bounded module replay uses five tables across four families: GEMM, MoE, DSA context/generation, and communication. Raw sparse component tables are separate smoke evidence. Preserve each run's original sidecars and content hashes when importing data; compute, DSA, and communication artifacts need not share one collector source snapshot.

That initial replay represented the checkpoint's three initial dense layers as MoE and omitted decode graph padding. The current GLM model composition preserves the three dense layers. Interpolation/extrapolation, empirical memory and fusion approximations, and unmodeled serving overhead also limit the comparison. These are investigation targets, not established causes of the observed errors.

## Exact prefill graph profile

The opt-in `sglang_glm52_nvfp4_vr200_tp4_graph_v1` profile supplies direct homogeneous prefill latency through the canonical `RustForwardPassPerfModel.best_available(config)` constructor and its `predict_prefill_latency(bs, isl, prefix)` method. Set `estimator_config.op_level.prefill_graph_profile`, explicit `op_level`/`deny`/`SILICON`, worker type `prefill`, and `estimator_config.correction.enabled=false`; the [Core API example](../../../../docs/core-api.md#vera-rubin-glm-52-graph-prefill-pilot) provides the complete identity. Saved canonical configurations retain the resolved profile SHA-256. Here `isl` is total input length, including the cached prefix. The seven admitted calls are `(1,1024,0)`, `(2,1024,0)`, `(1,2048,1024)`, `(1,8192,0)`, `(2,8192,0)`, `(1,16384,0)` and `(1,32768,16384)`. This selector stays outside normal CLI/scheduler configuration. Other shapes, runtimes, mixed/decode steps, energy and SOL fail explicitly. Selecting this graph profile is opt-in. The pilot also includes the separately validated dense-prefix and projection correctness fixes described in the PR.

The frozen graph-forward comparison passed all seven measured contexts at the 15% criterion, with worst absolute relative error 5.2334%. This is a forward-step mean-latency result for the exact pinned runtime. That prefill comparison does not establish scheduler TTFT, model quality or general Vera Rubin accuracy. Decode is assessed separately below. The full profile binds deterministic FlashInfer top-k, forced DSA, breakable prefill graphs, buckets `[1024,2048,8192,16384]`, TP4/EP1 and the source/runtime/checkpoint identities. The full native argv and environment are retained in the profile; the historical eager launch below is a different collection path.

`publish_prefill_graph.py` is a CPU-only offline publisher for the already qualified evidence. It invokes the exact frozen native producer verifiers, checks both independent operator decisions, separate fixed-formula accuracy acceptance and explicit public-contract approval, then derives 15 rows. No new default GPU sweep is registered.

- `sglang_prefill_attention_sequence_perf.parquet` contains seven complete 78-call attention-sequence means, each using all 30 joint-arm timings once. Its sequence is `PP + 19×PSSS` with 21 producers, 57 shared consumers, 100 graph segments and 99 eager breaks. All control arms remain qualification evidence.
- `sglang_prefill_comm_norm_boundary_perf.parquet` contains eight exact `(num_tokens, boundary_role)` means. Each sample takes the maximum of four aligned whole-block rank timings, divides by 100 calls, then all 30 samples are averaged. The roles are `post_attention` (78 uses) and `following_mlp` (77 uses), at 1024/2048/8192/16384 tokens.

Both tables are latency-only and use exact keys with no interpolation or source inheritance. The identical `.profile.json` sidecars contain all row payloads without `profile_id`; the SHA-256 of their exact UTF-8 bytes including the final LF becomes every row's profile ID. The consumer binds that reviewed ID, exact Float64 row values and the original system/table hashes. Existing tables and empirical coefficients are unchanged. Publication receipts separately hash the final parquet files, avoiding a profile/table hash cycle.

From the Python package directory, publish a fresh systems bundle using the portable, immutable evidence bundle and the original qualified systems files:

```bash
python -m collector.sglang_rubin.publish_prefill_graph \
  --evidence /artifacts/prefill-graph-evidence \
  --base-systems /artifacts/qualified-systems \
  --output-systems /artifacts/graph-systems
```

`--validate-only` runs complete admission and row derivation without publishing. Publication stages both families and metadata before one atomic directory rename. Exact repetition is idempotent; a changed row, profile, input or partial bundle fails. The external verification bundle retains every raw export, candidate and decision at relative locations; a documentary file originally outside its candidate directory has one exact name-hash/content-hash relocation. No public runtime depends on a user's workspace or Vault. Ship the small systems tables, profile sidecars and receipt; raw GPU evidence remains an external artifact.

The approximation reuses synthetic attention weights/activations/KV without intervening MLP/communication queue work, uses repeated-state communication throughput and ideal router/routed-versus-shared overlap, and retains empirical norm/add terms. Original full-model structural weight inventory is preserved; these measurements do not qualify peak graph/KV memory.

## Decode forward-step validation

The default op-level predictor is validated separately against 18 native CUDA-graph decode steps on the same pinned runtime. The grid is batches `[1,3,8,29,31,32]` × past-KV lengths `[1024,8192,32768]`, with one current token per request and exact native graph buckets. Seventeen cases meet the original ±15% criterion; batch 1 at K=1,024 is −17.09%, accepted for the initial pilot. No prefill graph selector or observed-MoE distribution override is used for decode prediction. The [combined prefill/decode report](../../../../docs/vr200-glm52-accuracy.md) contains every row, the canonical API calls, measurement method, evidence identities and repeatability limits. This comparison does not qualify arbitrary graph padding or end-to-end TPOT.

## Reproduce qualified observed-MoE data

The two fixed observed-MoE profiles retain the exact publisher source that their consumers admit. Shared collector code has since changed, so reproduction requires the original `evidence/<distribution>/publisher-source` directory as well as the approved raw archive, review and base systems tree. The current public Python `publish()` APIs and CLIs require `publisher_source` / `--publisher-source`; omitting it fails before creating output.

From the Python package directory:

```bash
python -m collector.sglang_rubin.publish_observed_moe \
  --publisher-source /artifacts/v1-evidence/publisher-source \
  --base-systems /artifacts/original-81-row-systems \
  --archive /artifacts/v1-native.tar.gz --review /artifacts/v1-review.json \
  --output-systems /artifacts/replayed-v1-systems > /artifacts/v1-replay.json

python -m collector.sglang_rubin.publish_observed_moe_v2 \
  --publisher-source /artifacts/v2-evidence/publisher-source \
  --base-systems /artifacts/qualified-original-84-row-systems \
  --v1-archive /artifacts/v1-native.tar.gz --v1-review /artifacts/v1-review.json \
  --tail-archive /artifacts/tail-native.tar.gz --tail-review /artifacts/tail-review.json \
  --output-systems /artifacts/replayed-v2-systems > /artifacts/v2-replay.json
```

The v1 closure must match `sha256:067ad7797474518eab028911d4f0d6f314e1dd0456b08be5bae45de267f3e332`; v2 must match `sha256:bbe3ebf2d450053f524d38b0a8ef97f0e55df000b6c6f61430b0207afc622eaf`. Missing, extra, changed or symlinked source files fail admission. Replay copies the verified sources unchanged and executes them in an isolated subprocess; collector imports are restricted to that private package. Current registry/catalog support is staged explicitly and recorded separately from the historical publisher. The JSON result records both source sets, imported files, replay launcher and dependency versions. Preserve it beside the output bundle.

The unmodified historical publisher rechecks the complete approved archive, source, case plan, raw measurements, independent review and base before writing. V2 requires the exact originally qualified 84-row systems tree, including its historical evidence; a fresh v1 republication has a different publication timestamp and is not that base. `--validate-only` performs admission and derivation without publication. These CPU workflows preserve fixed profile IDs and performance-data contracts and do not qualify a new workload or generate GPU measurements.

## Frozen inputs

- Runtime tag: `gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo-ci:8a8fb0687160e73169fd6236d93ed19dec80ef54-68286189-rubin-sglang-arm64`.
- Runtime index digest: `sha256:53299500a280c8de34bd484507a45b2f83b4d5e7c999b77284fa31930f7e63ab`; ARM64 manifest: `sha256:1c7ffbccde1dd9a3ec894db29a1aa3721e3ba393fc7337b2148b2f184aa0bcab`.
- Observed SGLang distribution: `0.5.18+nvinternal.rubin.0.8full.66997102`. Its build label is separately `0.5.18+02c5a855`.
- Audited SGLang source: [02c5a855aceb968c310e6fbc6632270e26edc84b](https://gitlab-master.nvidia.com/dl/sglang/sglang/-/tree/02c5a855aceb968c310e6fbc6632270e26edc84b). The selected installed source files checked during preflight matched this revision.
- NVFP4 checkpoint: `/artifacts/model/nvidia_glm-5.2-nvfp4/hf/hf-aec724e_orig`. The entrypoint checks its `config.json` and `hf_quant_config.json` SHA-256 values against the observed snapshot recorded in `registry.py`, and checks the cached model's structural and quantization policy.
- Serving and collection require `SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=0` before launching Python. The existing MoE perf key has no phase dimension; disabling deferred finalization keeps the measured finalized TRTLLM expert path consistent across prefill and decode. At the audited revision, `srt/models/deepseek_v2.py:904-1023` can otherwise defer finalization during captured forwards, `srt/layers/moe/fused_moe_triton/layer.py:387-391` reads the environment flag, and `srt/layers/moe/moe_runner/flashinfer_trtllm.py:1031-1155` chooses `do_finalize`. An unset or different value fails preflight; collectors never overwrite it.
- The standalone eager-prefill DSA collection path requires `--disable-prefill-cuda-graph`, matching the DSA collector's explicit `disable_prefill_cuda_graph=True`. At the audited revision, `srt/layers/attention/dsa_backend.py:3313-3339` and `srt/layers/attention/dsa/dsa_indexer.py:1601-1608` change dense-attention and K-only indexer dispatch inside prefill graphs. Decode retains the native CUDA graph policy. The default TRTLLM MoE runner, runtime image, and perf schemas are unchanged.
- MoE collection targets the frozen server's `--max-running-requests 32 --cuda-graph-max-bs-decode 32`, enabled FlashInfer autotuning, and disabled extend autotuning. At the audited revision, `srt/model_executor/runner/base_runner.py:242-248` and `decode_cuda_graph_runner.py:516-528` warm up using the maximum decode shape; `flashinfer_autotune.py:249-265` skips the extend pass when `SGLANG_FLASHINFER_AUTOTUNE_EXTEND` resolves to false, its default in `srt/environ.py:919`. The collector validates that native environment value, tunes views of at most 32 rows from each of its five synthetic inputs, then measures every full-size input with the existing graph benchmark. It clears the dedicated worker's FlashInfer cache before and after each case, so other collected shapes cannot leave prefill tactics behind. Larger token counts still execute and use native fallback selection. These rows are scoped to this server policy, including eager serving prefill; they do not establish tactic equality for checkpoint inputs or for other decode limits. Smaller cases tune only their reachable decode buckets. The existing synthetic distributions, timing method, case grid, and perf schema are unchanged; fresh rows and their new collector provenance are required before revalidation.

- GEMM collection treats `(N, K) = (256, 6144)` as this exact checkpoint's replicated router. Its TP4 dense, shared, attention, indexer, and vocabulary projections have different physical widths; this selection must be audited again before extending the pilot to another checkpoint or topology. The collector builds native `MoEGate` from the hash-checked cached checkpoint config and uses its unchanged forward path, retaining BF16 inputs/weights and FP32 logits. At the audited revision, `srt/models/deepseek_v2.py:458-550` selects the JIT router for SM107 batches of at most 16 tokens and `linear_bf16_fp32` for larger batches when deterministic inference and context parallelism are disabled. An untimed Python call observes the actual JIT or cuBLAS leaf and validates output shape and dtype, then removes its observer before the existing graph benchmark. Unknown leaves and invalid outputs fail before publication. Other GEMMs retain `UnquantizedLinearMethod.apply`, and the existing input-dtype/shape key and `kernel_source` column are unchanged. Fresh rows with the new collector provenance are required; historical rows remain evidence of the earlier ordinary-linear measurement.

The launcher must enforce the image digest. `--launcher-image` records that reference; neither this argument nor build environment variables attest the container from inside it. The `-test` sibling image has a separate dependency environment and is outside this pilot.

## Stage the source and inspect the plan

Mount the complete AISimulate checkout at `/aisimulate` in the pinned Linux/ARM64 GPU container, together with the checkpoint and a writable `/results` on a node-local filesystem. On Hecate, collect under `/tmp` or mount node-local storage at `/results`: Lustre `/work` does not support the atomic no-replace rename required by the shared checkpoint writer. After collection exits, preserve or archive the complete output directory to Lustre, including finalized tables, provenance, inventories, plans, logs, and checkpoints. Restore that complete directory to suitable node-local storage before resuming. The released wheel does not include these GPU collector files. Use the image's existing Python environment; preserve the checked-out `collector/` tree, YAML cases, hash closures, and cached `src/aisimulate_core/model_configs/` files. Keeping `.git` allows the standard sidecar to record the collector commit; content hashes are recorded independently.

```bash
export PYTHONPATH=/aisimulate/python/aisimulate
export VR_IMAGE='gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo-ci@sha256:53299500a280c8de34bd484507a45b2f83b4d5e7c999b77284fa31930f7e63ab'
export VR_CHECKPOINT=/artifacts/model/nvidia_glm-5.2-nvfp4/hf/hf-aec724e_orig
export SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=0

python -m collector.sglang_rubin --plan-only
python -m collector.sglang_rubin.runtime \
  --checkpoint-dir "$VR_CHECKPOINT" --launcher-image "$VR_IMAGE" \
  --check-imports --validate > /results/runtime-inventory.json
```

The plan also works on a CPU development environment with AISimulate's normal dependencies. It does not import Torch or SGLang and reports both the required serving configuration and the observed environment without validating them. Runtime inventory separates those same declarations and observations; it does not attest an external server's settings. Unknown models and operations fail explicitly.

## Collect

This focused attention/MoE smoke run selects the first outer case of each requested operation and bounds the DSA inner sweeps to one context and one decode shape. The context has 4,096 cached tokens plus 128 new tokens, so it exercises the sparse path. `--limit` alone only bounds outer cases; it does not bound an attention module's inner sweep.

```bash
AIC_DSA_CONTEXT_PREFIX_LENS=4096 \
AIC_DSA_CONTEXT_SEQ_LENS=128 \
AIC_DSA_CONTEXT_BATCH_SIZES=1 \
AIC_DSA_GENERATION_PREFIX_LENS=4096 \
AIC_DSA_GENERATION_SEQ_LENS=1 \
AIC_DSA_GENERATION_BATCH_SIZES=1 \
python -m collector.sglang_rubin \
  --checkpoint-dir "$VR_CHECKPOINT" --launcher-image "$VR_IMAGE" \
  --ops moe dsa_context_module dsa_context_module_skip_indexer \
    dsa_generation_module dsa_generation_module_skip_indexer \
  --output-dir /results/vr200-glm52-smoke --limit 1 --sequential
```

Run raw sparse-kernel proof separately with the same bounded sparse shapes:

```bash
AIC_DSA_CONTEXT_PREFIX_LENS=4096 \
AIC_DSA_CONTEXT_SEQ_LENS=128 \
AIC_DSA_CONTEXT_BATCH_SIZES=1 \
AIC_DSA_GENERATION_PREFIX_LENS=4096 \
AIC_DSA_GENERATION_SEQ_LENS=1 \
AIC_DSA_GENERATION_BATCH_SIZES=1 \
python -m collector.sglang_rubin \
  --checkpoint-dir "$VR_CHECKPOINT" --launcher-image "$VR_IMAGE" \
  --ops glm5_mqa_logits_module glm5_topk_module glm5_dsa_attn_module \
  --output-dir /results/vr200-glm52-sparse-smoke --limit 1 --sequential
```

Raw MQA/top-k collectors intentionally raise when native serving selects dense attention or the K-only indexer, because those paths do not execute the requested kernels. Use the full DSA modules to measure those paths; do not run the raw sparse operations over the full module grid.

The shared GEMM grid contains generic large shapes, so a prefix selected by `--limit` is not representative GLM coverage. Use the standard runtime substring filter to select a declared shape for a separate GEMM smoke run:

```bash
python -m collector.sglang_rubin \
  --checkpoint-dir "$VR_CHECKPOINT" --launcher-image "$VR_IMAGE" \
  --ops gemm --case-filter "['bfloat16', 128, 6144, 6144]" \
  --output-dir /results/vr200-glm52-gemm-smoke --sequential
```

`--case-filter` is available only with `--ops gemm`, uses the existing executor's OR substring semantics, and is included in the plan and resume identity. Supply filters for the GLM projection shapes and token counts needed by a collection campaign; this one shape only establishes a smoke result. `--processes N` selects the number of independent GPU workers; the default is one, and `--sequential` runs on device 0. `--smoke` instead shuffles outer cases and samples one by default; avoid combining that random selection with a single-batch inner filter, because the selected outer batch may differ. Inner filters intersect the declared grid and cannot inject shapes.

After smoke validation, use this bounded DSA module scope in a fresh directory. It matches the pilot's operational `vr-pilot/dsa_campaign.py`: 152 context shapes and 42 generation shapes for each full/skip variant, or 388 planned rows. The environment filters intersect the existing sweep; they do not inject new shapes.

```bash
AIC_DSA_CONTEXT_PREFIX_LENS=0,4096,8192,16384 \
AIC_DSA_CONTEXT_SEQ_LENS=128,512,1024,2048,4096,8192,16384 \
AIC_DSA_CONTEXT_BATCH_SIZES=1,2,4,8,16,32 \
AIC_DSA_GENERATION_PREFIX_LENS=512,1024,2048,4096,8192,16384,32768,65536 \
AIC_DSA_GENERATION_SEQ_LENS=1 \
AIC_DSA_GENERATION_BATCH_SIZES=1,2,4,8,16,32 \
python -m collector.sglang_rubin \
  --checkpoint-dir "$VR_CHECKPOINT" --launcher-image "$VR_IMAGE" \
  --ops dsa_context_module dsa_generation_module \
    dsa_context_module_skip_indexer dsa_generation_module_skip_indexer \
  --output-dir /results/vr200-glm52-dsa --processes 4 --limit 6
```

The stock context sweep caps batch size at 8 for input lengths of at least 8,192. Its generation token budget excludes past-KV 32,768 at batch 32, so this scope does not give exact coverage for that decode point. Collect MoE separately with `--ops moe --processes 4` and a fresh output directory; collect GEMM with explicit projection/token filters as above. The visible runtime filters restrict MoE to TP4/EP1 and DSA modules to TP4; sparse kernels use TP4/head-count 16. GEMM retains the existing physical shape grid. There are no YAML shape exclusions. Queued failures remain errors, and any error gives a nonzero exit even when successful rows are finalized.

After the independent workers exit, collect TP4 communication with all four GPUs available:

```bash
torchrun --standalone --nproc-per-node=4 \
  --module collector.sglang_rubin.collect_all_reduce \
  --checkpoint-dir "$VR_CHECKPOINT" --launcher-image "$VR_IMAGE" \
  --output-dir /results/vr200-glm52-communication \
  --tokens 1 8 32 128 1024 8192 16384 --modes graph eager
```

This produces 14 plain all-reduce rows, excluding fused residual/RMSNorm and expert all-to-all. The standalone collector requires a fresh directory and does not support resume. The native communication loader filters out `_eager` rows on systems other than `b60`; for this pilot, the seven graph rows supply replay in both phases. All fourteen graph/eager measurements remain collected evidence, but the eager rows are not prediction inputs.

Each model-entrypoint output directory holds inventory and plan records, a `pilot_identity.json`, normal per-op logs/checkpoints, canonical parquet tables, and the standard provenance sidecar. Reusing that directory requires `--resume`; resume rejects changes to the image, observed package versions, checkpoint metadata, collector content hashes, declared serving configuration, observed serving environment, runtime case filters, or inner-sweep environment. Every pilot producer's provenance closure includes `runtime.py`, which declares the required serving configuration. Use the same scope/environment when resuming, and `--resume-retry-failed` to retry recorded failures. Run only one collector invocation per output directory. Smoke and full collection use separate directories.

## Historical eager-prefill serving and end-to-end measurement

For the original eager-prefill baseline, start the checkpoint with TP4/EP1 and FP8 KV in the same pinned image, with both required serving constraints. Keep the default MoE runner selected by this SGLang build. Preserve the launch environment, resolved server arguments, and logs; verify that `disable_prefill_cuda_graph` resolves to `true` and the MoE layer log reports deferred finalization disabled. A successful load alone is insufficient.

```bash
SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=0 python -m sglang.launch_server \
  --model-path "$VR_CHECKPOINT" --served-model-name glm52 \
  --tp-size 4 --ep-size 1 --trust-remote-code \
  --host 127.0.0.1 --port 30000 \
  --kv-cache-dtype fp8_e4m3 --disable-radix-cache \
  --max-running-requests 32 --cuda-graph-max-bs-decode 32 \
  --mem-fraction-static 0.80 --context-length 40000 \
  --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --disable-prefill-cuda-graph
```

Preserve the runtime inventory, launch command/environment, `/get_server_info`, `/get_model_info`, and server log. The frozen baseline resolved DSA `trtllm`, MoE `flashinfer_trtllm`, quantization `modelopt_fp4`, TP4/EP1/DP1/CP1, decode graph sizes 1/2/4/8/12/16/24/32, and 2,552,000 KV tokens in 64-token pages (39,875 blocks).

Prepare `/results/prompt-1024.txt` with one exact 1,024-token prompt using this checkpoint's tokenizer and its default special-token policy. The benchmark independently requires the server to report that exact input length. Record the launch options and environment, including both dispatch constraints and cache settings, in `/results/server-config.json`. The following two invocations reproduce the matching-workload warmup followed by measurement. The client's additional short sequential warmup does not replace this full-grid warmup.

```bash
python -m collector.sglang_rubin.benchmark \
  --api-base-url http://127.0.0.1:30000/v1 --model glm52 \
  --prompt-file /results/prompt-1024.txt --expected-prompt-tokens 1024 \
  --output-tokens 500 --ignore-eos --require-output-length \
  --concurrencies 1,8,32 --requests-per-worker 2 \
  --serving-config /results/server-config.json --output /results/workload-warmup-1024.json

python -m collector.sglang_rubin.benchmark \
  --api-base-url http://127.0.0.1:30000/v1 --model glm52 \
  --prompt-file /results/prompt-1024.txt --expected-prompt-tokens 1024 \
  --output-tokens 500 --ignore-eos --require-output-length \
  --concurrencies 1,8,32 --requests-per-worker 4 \
  --serving-config /results/server-config.json --output /results/e2e-1024.json
```

Repeat both invocations with exact 8,192- and 32,768-token inputs and separate artifacts. There are 164 measured requests per input length, or 492 total; exclude the separate warmup artifacts from measured throughput. The frozen baseline reused its prompt at each input length with prefix caching disabled. A request must finish its SSE stream, emit text, provide a finish reason, and report a positive actual output-token count. Empty output, missing usage, truncated streams, wrong token counts, and any failed request invalidate the measured wave and return nonzero. Compare measured TTFT, TPOT, and throughput with predictions only after the experimental VR200 hardware profile and genuine tables pass the normal data/provenance checks. That historical end-to-end comparison failed its proposed accuracy gate. The separately qualified prefill graph profile does not establish scheduler TTFT; current decode forward-step results are documented in the [combined accuracy report](../../../../docs/vr200-glm52-accuracy.md).
