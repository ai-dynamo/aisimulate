<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mamba mixed-step accounting

These are simulator predictions for the modeled `b200_sxm` platform, for
Nemotron-3-Nano-30B-A3B BF16 at TP1. They are accounting diagnostics. They are
not silicon measurements, calibration or accuracy claims.

## Setup

| Variant | Commit | Content |
| --- | --- | --- |
| Candidate | `686275e7350e4b4a39e88dec55c135312b671038` | The fix. |
| Baseline | `49d93f5a4aec1ec6033ae63228548348167b4e12` (not published) | The Candidate with `session.rs` and the mixed-step hunks of `runtime.rs` reverted. |
| Main | `bf373a132b6ce72e9fdd2a5d1b5124cefb84ed0c` | Unqualified control only. |

Baseline and Candidate share all qualification code. The accounting code is the
only difference. No built-in model sets `state_dtype` or `token_width` yet, so
every qualified run applied this wrapper before it built the model:

```python
def apply(nextn):
    """Same wrapper in every variant: record the recurrent-state dtype and the decode width."""
    import aisimulate_core.sdk.models.nemotron_h as nh
    real = nh.ops.Mamba2Kernel
    def wrapped(name, scale, kernel, phase, *args, **kw):
        kw.setdefault("state_dtype", "float16")
        kw.setdefault("token_width", (nextn + 1) if phase == "generation" else 1)
        return real(name, scale, kernel, phase, *args, **kw)
    class _Shim:
        def __init__(self): self._o = nh.ops
        def __getattr__(self, n): return wrapped if n == "Mamba2Kernel" else getattr(self._o, n)
    nh.ops = _Shim()
```

## Operation sources

| Data | Context conv1d | Context scan | Generation |
| --- | --- | --- | --- |
| TensorRT-LLM `1.3.0rc20` | Measured table, exact shape key | Analytical bound | Analytical bound |
| vLLM `0.24.0` | Analytical bound | Analytical bound | Analytical bound |

`fallback_policy: deny` does not prove measured Mamba coverage, because the
analytical bound is an allowed result.

## Results

Qualified Baseline to qualified Candidate.

`InferenceSession.run_agg`, batch 16, output 64, 2048-token prefill budget
(8128 for the split budget), TensorRT-LLM:

| MTP | Case | TTFT ms | tok/s/GPU |
| --- | --- | --- | --- |
| 0 | chunked, 6000 input | 249.54 to 250.68 (+0.46%) | 930.2 to 926.4 (-0.41%) |
| 0 | split budget, 1024 cached | 100.69 to 101.24 (+0.54%) | 1370.2 to 1366.2 (-0.29%) |
| 3 | chunked, 6000 input | 251.95 to 253.57 (+0.64%) | 1021.5 to 1015.0 (-0.64%) |
| 3 | split budget, 1024 cached | 101.51 to 102.40 (+0.87%) | 1820.7 to 1809.4 (-0.62%) |

On vLLM, TTFT changes by +0.11% to +0.75% and tok/s/GPU by -0.12% to -1.28%.
With a 3000-token cached prefix, `run_agg` returned the same values as without
it in both variants, so it does not show partial-cache behavior.

Packed telemetry (`RustForwardPassPerfModel.estimate_forward_pass_time_ms`),
both backends:

| Shape | MTP0 | MTP3 |
| --- | --- | --- |
| Pure prefill | 0% | 0% |
| Pure decode, 8 requests | 0% | +2.03% to +2.14% |
| Mixed, 1x2048 + 8 decode | +0.25% to +0.27% | +0.68% to +0.75% |
| Mixed, 2 requests of 1500 tokens + 8 decode | +0.24% to +0.36% | +0.62% to +0.71% |

Per-phase `mixed_step_latency`, TensorRT-LLM, MTP3: a 6000-token chunk with a
3000-token prefix rises 1.21% (26.013 to 26.328 ms). The decode component rises
from 0.20 to 0.35 ms because decode Mamba now runs at decode coordinates. Pure
prefill does not change.

## Controls

- Replay (`predict --stack engine`) does not change. It prices prefill and
  decode as separate passes. Over 14 configurations, requests, output tokens,
  cache reuse and per-request status match. Latency metrics differ only by
  rounding, about 1e-16.
- Main, Baseline and Candidate give identical results for unqualified ops.
- Qualified and unqualified runs differ without the fix, because the analytical
  bound adds recurrent-state bytes. Compare Baseline with Candidate only.
- Nemotron-3-Super-120B-A12B FP8 on TensorRT-LLM fails in both variants with a
  missing MoE row. No other estimate replaces it.

## Request-count fix

Commit `b2b489d999e97749c2932c3f91b1dddb544c81a8` spreads the telemetry prefill
token sum over the observed request count. Before it, an uneven sum added a
request. The shapes above divide evenly, so they do not change. A same-host check
of `686275e7` against `b2b489d9` changed only qualified uneven shapes. On vLLM,
2 prefills of 17 tokens plus 1 decode went from 5.860174 to 5.853910 ms. That
difference is one FP16 state pair (2 MiB) for each of 23 Mamba layers at 7.7e12
B/s, within 1e-13. All other shapes were identical.

## Host cost

Six alternating Baseline and Candidate pairs ran. The retained replay results
record the in-process runner time (`wall_s`) and the process peak memory
(`maxrss_kb`):

| Field | Baseline | Candidate |
| --- | --- | --- |
| `wall_s` | median 1.627 s (1.622 to 1.660) | median 1.637 s (1.635 to 1.645) |
| `maxrss_kb` | median 171 MiB (162 to 198) | median 162 MiB (162 to 198) |

Six samples cannot separate the variants. The whole-process times (1.70 s and
2.37 s) and the `run_agg` memory (about 700 MB) are transcribed observations.
Their raw log was not saved. No speed, memory or significance claim is made.

## Limits

- Summing uniform-batch queries can overcount launch overhead for
  variable-length batches.
- One model family, one modeled platform, synthetic shapes.
