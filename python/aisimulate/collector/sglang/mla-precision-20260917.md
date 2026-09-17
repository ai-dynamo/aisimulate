# SGLang 0.5.14 MLA precision collection

## Result

All 9,240 prefill cases passed on dedicated Slurm nodes. Each system has
3,080 fresh rows, zero failed or unattempted cases, and zero duplicate physical
keys. Every previously stored shape and KV-storage combination is covered.

| GPU | Production job | Node | Elapsed | Compute / KV precision: rows |
| --- | --- | --- | --- | --- |
| B200 | 1993054 | nsc-svg-slurm-1-gpu-200 | 2m53s | BF16/BF16: 1,540; FP8/FP8: 1,540 |
| B300 | 465988 | pool0-0051 | 3m08s | BF16/BF16: 1,540; FP8/FP8: 1,540 |
| H200 | 8873 | neb-cdg-slurm-1-gpu-1 | 2m39s | BF16/BF16: 1,540; BF16/FP8: 1,540 |

All three jobs and their GPU steps completed with exit code `0:0`.
The shared grid contains 14 local-head counts and 110 batch/sequence pairs
per precision. The previous B300 table had 1,760 duplicate physical keys;
the replacement removes them and adds the shared 96-head-family grid.

## Precision evidence

Separate smoke jobs B200 `1992964`, B300 `465927`, and H200 `8870` each passed
four cases: prefill/decode with BF16/FP8 KV storage. Wrappers observed tensor
dtypes at the native attention calls; PyTorch's CUDA profiler captured the
kernel names. Production measurements ran without these wrappers or profiling.

- B200/B300 FP8-KV prefill passed FP8 Q/K/V to
  `flashinfer.prefill.trtllm_ragged_attention_deepseek`. CUDA kernels contained
  `QkvE4m3OBfloat16` on SM100/SM103. The old collector's fixed BF16 label was
  incorrect. BF16-KV controls used `QkvBfloat16OBfloat16`.
- H200 passed BF16 Q, QV, K and V to FA3 with either KV-storage dtype.
  The captured SM90 kernel used `cutlass::bfloat16_t`.
- Decode probes confirmed the same compute distinction. This campaign
  replaces only `context_mla_perf.parquet`; existing generation tables retain
  their previous status-only provenance. The native generation loader keys
  on KV precision, not `mla_dtype`.

Source: SGLang commit `4289f36ef960fad8268a6b94935686e792a81432`,
[Blackwell prefill dispatch](https://github.com/sgl-project/sglang/blob/4289f36ef960fad8268a6b94935686e792a81432/python/sglang/srt/layers/attention/trtllm_mla_backend.py#L683-L692)
and [Hopper absorbed MLA](https://github.com/sgl-project/sglang/blob/4289f36ef960fad8268a6b94935686e792a81432/python/sglang/srt/layers/attention/flashattention_backend.py#L1176-L1181).
The installed Blackwell backend source hash matches that immutable revision:
`b83137cfc31a095425062aab3d80835cd670b4190cf13ada14c7802b63fddb8d`.

The 74 Blackwell failures are resolved by measured FP8/FP8 profiles. The 10
H200 failures are resolved by mapping inferred attention to the observed BF16
execution dtype before building native ops. No FP8/BF16 rows were fabricated.

## Profile-only 84-point replay

The exact point set was recovered from the E2E Gym
`inferencex-coverage-20260917` audit, release `db-dump/2026-09-14`.
The source `predictions.json.gz` SHA256 is
`dca0cee0e750295e6d7f7d33af8b085d38276ecb63a00a0acdee8c8deebc2d6b`.

| GPU | Original points | Baseline failures | New-profile successes | Remaining failures |
| --- | ---: | ---: | ---: | ---: |
| B300 | 45 | 45 | 45 | 0 |
| B200 | 29 | 29 | 29 | 0 |
| H200 | 10 | 10 | 0 | 10 |
| Total | 84 | 84 | 74 | 10 |

- Both arms used the original AISimulate runtime from
  `93419f7ca0cf971b56c843d19a456b8af82e4bcc`. Python runner, AIC materializer,
  and both native extension hashes match the original audit.
- Comparing 1,592 installed package files found exactly three differences:
  the B200/B300/H200 `context_mla_perf.parquet` files. This isolates the
  profile change from subsequent simulator changes on main.
- All 84 original error messages were reproduced. Baseline and new-profile
  replay specs were byte-identical for every point. Original source fields,
  query version `0.5.14`, quantization, and workload settings were preserved.
  The Gym adapter was pinned to `734e9cf31fdbca74a1be496a309026ab7a58f119`;
  its hash is recorded separately from the original audit adapter.
- The 74 successful points completed all 18,950 expected requests, with
  positive finite metrics and zero truncated outputs. All 10 H200 failures
  retain the original error and have no prediction metrics. No fallback was used.
- Source silicon images were SGLang `0.5.12-cu130` (65 points) and
  `0.5.12.post1` (19 points), while the original profile query was `0.5.14`.
  This replay establishes coverage recovery, not version-matched accuracy.
- An initial audit wrapper incorrectly required optional `max_model_len`.
  Its logs and native reports are retained; both arms were rerun after fixing
  the wrapper. No engine or deployment setting changed for that retry.

Point identities, metrics, errors, commands, and hashes are in the
[replay evidence](mla-precision-20260917-replay.json). Full inputs, native
per-request reports, scripts, and retry logs are retained locally at
`/Users/simonec/.cache/aisim-e2e-gym/mla-fp8-20260917/replay-84/`.

## Execution mapping and final replay

`resolve_sglang_mla_compute()` resolves BF16 compute before model construction
for SGLang 0.5.14 SM90 FA3 and the measured DeepSeek-V3/R1 BF16 model geometry
(512-rank KV, 64-dimensional RoPE). Native compilation, KV memory construction,
and the estimate-path FMHA resolver share this rule. It does not depend on
which precision tables are available. Explicit FMHA overrides, Blackwell,
whole-model FPM, and unaudited backend/version/geometry combinations are preserved.

A fresh wheel from `2cfe6f836fa3f2c81786d250125406cc624c1357` replayed all
84 original points successfully: B300 **45/45**, B200 **29/29**, H200 **10/10**.
All 21,430 requests completed, with zero truncated outputs and positive finite
metrics. The 84 input replay specs and original source fields were unchanged.
The three installed profile hashes match the collected data, and all 124 Python
source files in the replay/core packages match the branch checkout.

This final run uses the complete branch wheel, including the main updates;
the profile-only experiment above separately isolates the data contribution.
The original silicon/query-version mismatch remains, so these results establish
coverage recovery rather than version-matched accuracy.

- [Final replay evidence](mla-precision-20260917-mapping-replay.json) records
  point identities, metrics, wheel/runtime hashes, and commands.
- Full per-request reports and scripts are retained in
  `/Users/simonec/.cache/aisim-e2e-gym/mla-fp8-20260917/mapping-84/`.
- 106 focused mapping, compilation, and memory tests passed; all eight native
  version-resolution tests passed. Tests verify both primary and fallback MLA
  op serialization, explicit precision, and Blackwell behavior.

## Runtime and measurement

- Image: `lmsysorg/sglang:v0.5.14`, amd64.
- OCI index: `sha256:5027e95bf6ec536856b1b52a91d1f35ff5c564ab83e8a94758a169ff09bb8df3`.
- Pinned amd64 manifest: `sha256:9611bd4c5624b0e9e17829506188a12f17205f2083de0dd44d6c521733553a50`.
- Actual packages: SGLang 0.5.14, sglang-kernel 0.4.4, FlashInfer 0.6.12,
  PyTorch 2.11.0+cu130, CUDA 13.0.
- Collector commit: `35b5292364a0c8d004d3450af27f6259a17aa668`.
- Source archive SHA256: `f8e4ff64ce4223db8353bf479d092593ffd565a7e49e78ff69c3b6d2b99422c9`.
- Direct runner SHA256: `001c384038d4ee684e4e0671f0ac80ac55fda4118b47bae8c171ca933cd33a56`.
- Eight GPU workers called the unchanged `run_mla` timing path; no model
  weights or server were required. Smoke rows were excluded from publication.
- Clocks were not locked. Five-second samples with nonzero GPU utilization
  observed SM clocks of 1,447–1,965 MHz (B200), 1,252–2,032 MHz (B300), and
  1,515–1,980 MHz (H200). Power limits were 1,000/1,100/700 W respectively.
  These samples describe run conditions, not per-kernel clock measurements.

## Artifacts and recovery

On B200 `nsc-svg-slurm-1-login-02.nvidia.com` and B300
`aws-pdx-slurm-1-login-01.nvidia.com`, the storage root is:

`/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec`

On H200 `neb-cdg-slurm-1-login-02.nvidia.com`, the storage root is:

`/lustre/fsw/portfolios/coreai/projects/coreai_comparch_lights-out-inf/users/simonec`

Under each root:

- `results/mla-fp8-20260917/production-direct/`: plan, eight raw CSV shards,
  per-case outcomes, summary, clock/power samples, hashes, validated parquet.
- `results/mla-fp8-20260917/smoke-clean/probe.json`: tensor and CUDA-kernel evidence.
- `results/mla-fp8-20260917/collect_direct.py`: exact production runner.
- `jobs/mla-fp8-20260917-direct.sh`: Slurm launcher.
- `logs/mla-fp8-direct-<job>.out` and `.err`: production logs.

All evidence is also retained locally under
`/Users/simonec/.cache/aisim-e2e-gym/mla-fp8-20260917/`.

Initial smoke retries fixed macOS archive metadata and an optional package
metadata lookup. H200 job `8864` failed before runner startup on GPU node 11
with signal 53, repeating the same day's recorded node failure; the retry
excluded that node. B300 job `465950` exposed unsupported atomic no-replace
rename on Lustre before any case ran. The task runner therefore used separate
CSV shards and per-case outcomes; canonical parquet finalization ran locally.
Collector checkpoint and failure-classification code was not changed.

## Validation

- All planned cases matched exactly one successful outcome and one positive,
  finite timing row; source archive and runner hashes matched the workers.
- Native Rust table views loaded and matched all 9,240 row latencies with
  sibling inheritance disabled. Precision slices matched the observed kernels.
- All seven collector-data checks and the backend-facts registry check passed.
- 40 focused collector and MLA loader tests passed; Ruff and diff checks passed.
- Detailed measurements and hashes: [evidence JSON](mla-precision-20260917.json).
