# H200 SXM vLLM 0.25.0 collection

This is an in-progress data checkpoint for PR #244. It does not claim complete operator coverage or whole-model accuracy.

- Completed full sweeps: BF16 GEMM and ordinary FP8 GEMM, 37,518 cases each.
- Published checkpoint: one GEMM table, 75,036 measured rows, zero failed or unattempted cases in these two sweeps.
- FP8-block GEMM and the other PR #219 operator families are still being collected. Smoke measurements are not included.
- Hardware: NVIDIA H200, SM90, 143,771 MiB, 700 W maximum board power, NV18 links between all eight GPUs.
- Runtime: vLLM 0.25.0, source `dd10e03f95f94edbea1975c67ace3a35ec9a8a40`; Torch 2.11.0+cu130; CUDA 13.0; FlashInfer 0.6.13.
- Image: `vllm/vllm-openai@sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97`.
- Collector: unchanged `cbaf51b64fa460e5ec6146bde407a4c64958212d`.
- Cluster: `neb-cdg-slurm-1`; measurement jobs `8674` and `8686`. Each used one exclusive eight-GPU node and one Slurm task; the collector managed eight workers.
- Raw evidence: `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_lights-out-inf/users/simonec/results/vllm025-pr244-multigpu/h200_sxm/production/`.
- Slurm requested the highest GPU frequency; sampled clocks are retained. A hard clock lock was not verified.
- Validation: hashes, schema, finite positive latencies, unique physical keys, complete case IDs, strict provenance, and exact native GEMM queries passed with shared-layer fallback disabled.

All 0.24.0 data remain present. Its GEMM donor policy retains BF16 and ordinary FP8 and excludes only the 35,741 FP8-block rows measured eagerly by the old collector. Explicit 0.24.0 primary queries retain historical values. The query-version defaults are unchanged.
