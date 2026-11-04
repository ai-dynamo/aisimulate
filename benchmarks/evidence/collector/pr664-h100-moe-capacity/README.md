# H100 MoE capacity refusal and native control

These bounded module checks used one clean H100 80GB HBM3 on AWS IAD Slurm,
node `pool0-0006`, TensorRT-LLM 1.3.0rc29. Actual device capacity was
85,019,590,656 bytes. Probe rows are not packaged performance data.

| Check | Result |
|---|---|
| Nemotron Ultra non-gated BF16, H8192/I5120/E512, TP1/EP1 | Refused before expert allocation: two resident matrices need 85,899,345,920 bytes. Peak torch allocation 0; collection call 0.999 s. |
| Same geometry, FP8, 16K tokens | Dry run completed, but native autotuning failed: `Can't allocate profile workspace for MoE GEMM profile`. Peak torch allocation 78,744,654,848 bytes; call 39.543 s. No performance row. |
| GLM BF16, H6144/I2048/E256/top-k8, TP4/EP1, 16K tokens | Native CUTLASS row **6.037243 ms**. Peak torch allocation 11,736,094,720 bytes; call 19.286 s. |

The weight bound uses actual gated/non-gated matrix counts, bit width and
TP/EP sharding. It refuses an impossible allocation without removing the case
from the plan. Passing this lower bound does not establish native workspace
fit: the FP8 failure demonstrates that distinction. No guessed workspace
multiplier or model-name exclusion was added.

Jobs **28267** (guard source `c66ce3ba3`) and **28269** (final integrated source
`c4b44d81f526b3c5299f7c15ccafe02d90b39229`) preserve their native result,
source hash, harness, logs and GPU receipts below. The guard source predates
the integrated strict token-sweep failure behavior; each probe requested only
one token count, so it never used the retired smaller-token fallback.

The safe control persisted its row and requested the collector's normal
`WORKER_RESTART` with exit **10** because 6,180 MiB remained allocated. The
standalone harness did not translate that executor protocol into process
success; its raw result has no `status` field. `summary.json` explicitly
derives native-row success from that persisted row and the logged restart,
rather than claiming a zero scheduler exit. The failed FP8 control remains a
failure even though its enclosing job script completed.

Measured collection calls total **59.828 seconds = 0.016619 GPU-hours**,
including the failed native attempt, excluding scheduler/allocation time.
An earlier setup-only job 28261 failed to import the mounted collector because
the image overrides `PYTHONPATH`; 28267 fixes the harness import path. That
setup failure is not a collector timing anchor.

`executed-artifacts.zip` retains exact executed bytes. Readable published
copies add SPDX headers and normalize whitespace; `artifact-hashes.json`
records both hashes. `summary.json` is derived evidence, not a measurement
record. Native remote roots:
`/lustre/fsw/portfolios/coreai/projects/coreai_comparch_lights-out-inf/users/simonec/results/pr664-capacity-20261103/{28267,28269}`.

The separate FP8 head-512 attention n4/kv1 stride failure remains unqualified.
The installed vLLM 0.31.0 `set_default_quant_scales` in
`model_executor/layers/attention/attention.py` creates scalar scale tensors,
matching the collector; changing their rank merely to bypass CuteDSL's
stride check would not establish serving parity. Earlier n16/kv2 positive
controls and n4/kv1 failures remain the evidence for that nonblocking gap.
