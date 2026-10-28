<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mamba mixed-step accounting

This page records predicted-latency changes from the Mamba mixed-step accounting
fix. Every number is a simulator prediction for the modeled platform
`b200_sxm`. None of it is a silicon measurement, a calibration, or an accuracy
claim. The sources of each Mamba operation are listed below because they decide
what the numbers mean.

## What was compared

| Variant | Identity | Content |
| --- | --- | --- |
| Candidate | `686275e7350e4b4a39e88dec55c135312b671038` | The code tested for this page. |
| Baseline | `49d93f5a4aec1ec6033ae63228548348167b4e12` (not published) | The Candidate tree with the accounting hunks reverted (see below). |
| Main | `bf373a132b6ce72e9fdd2a5d1b5124cefb84ed0c` | Used only for the unqualified control. |

To rebuild the Baseline tree, start from the Candidate and restore
`crates/core/src/perfmodel/session.rs` and
`crates/core/src/perfmodel/engine/runtime.rs` from Main. Then keep only the
qualification parts of the Candidate in `runtime.rs`: the
`state_dtype: "unrecorded"` and `token_width: 1` fields in test `Mamba2Op`
literals and the qualified-Mamba `token_width` check in `Engine::build`. The
reverted hunks are the qualified-Mamba branches in the three mixed-step passes,
the earlier prefix check, and the telemetry changes in `session.rs`.

Both Baseline and Candidate contain the same qualification code: the
`Mamba2Op` fields `state_dtype` and `token_width`, the recurrent-state bytes in
the analytical cost, and the `Engine::build` width check. The accounting code
is the only difference between them.

No built-in model sets `state_dtype` or `token_width` yet. In every qualified
run, the test script called `apply(nextn)` from this file before it built the
model. This is the exact file that was used:

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

The unqualified control ran without the wrapper.

All results below come from the Candidate `686275e7`. A later commit,
`b2b489d999e97749c2932c3f91b1dddb544c81a8`, stops the packed-telemetry prefill
split from inventing an extra request when the token sum does not divide evenly
by the request count. Every telemetry shape on this page divides evenly, so the
numbers below do not change. The [current-head check](#current-head-check)
confirms this.

Retained result archives (SHA-256, kept outside the repository):

| Archive | SHA-256 |
| --- | --- |
| ARM host functional matrix and timing block | `1f0ebdaa251ca012f130ce4122630e7f284fc302bbf5451d65cf864cd00dc437` |
| x86 host functional matrix | `306bc0aed4aa69a1c607b382226472264a727a14164c9dc3a92c43f17ecd7cc8` |

## Which public paths are affected

| Path | Result |
| --- | --- |
| `predict --stack engine` replay | No change. Operation-level replay prices prefill and decode as separate passes on both backends. |
| `InferenceSession.run_agg` (calls `mixed_step_latency`) | Changed. |
| `RustForwardPassPerfModel.estimate_forward_pass_time_ms` (packed telemetry) | Changed. |
| Whole-model FPM | Different path. It does not run `Mamba2Op` and is not part of this evidence. |

Replay control: 14 configurations ran. They cover two backends, MTP0 and MTP3,
mixed, pure-prefill and pure-decode traffic, and a second model. In all of them,
completed requests, output tokens and prefix-cache reuse are identical between
Baseline and Candidate. Per-request output length and terminal status are also
identical.

Aggregate latency metrics differ only by floating-point rounding, about 1e-16.
The one exception is a standard deviation of near-zero values in one
configuration. Per-request timestamps are identical after sorting.

## Operation sources

| Backend and data | Context conv1d | Context scan | Generation |
| --- | --- | --- | --- |
| TensorRT-LLM `1.3.0rc20`, `b200_sxm` | Measured table, exact key | Analytical bound (no scan rows) | Analytical bound |
| vLLM `0.24.0`, `b200_sxm` | Analytical bound | Analytical bound | Analytical bound |

The key is exact for Nemotron-3-Nano-30B-A3B and Nemotron-3-Super-120B-A12B.
Values between grid points are interpolated. `fallback_policy: deny` does not
prove measured Mamba coverage, because the analytical bound is an allowed
result. These are accounting diagnostics. They are not vLLM 0.31 or GB200
silicon calibration.

## Results: Nemotron-3-Nano-30B-A3B BF16, TP1

Qualified Baseline to qualified Candidate. Batch 16, output 64, 2048-token
prefill budget (8128 for the split-budget case).

### `run_agg`, TensorRT-LLM

| MTP | Case | TTFT ms | TPOT ms | tok/s/GPU |
| --- | --- | --- | --- | --- |
| 0 | chunked, 6000 input | 249.54 to 250.68 (+0.46%) | 16.74 to 16.81 (+0.41%) | 930.2 to 926.4 (-0.41%) |
| 0 | split budget, 1024 cached | 100.69 to 101.24 (+0.54%) | 9.91 to 9.93 (+0.23%) | 1370.2 to 1366.2 (-0.29%) |
| 3 | chunked, 6000 input | 251.95 to 253.57 (+0.64%) | 7.79 to 7.84 (+0.64%) | 1021.5 to 1015.0 (-0.64%) |
| 3 | split budget, 1024 cached | 101.51 to 102.40 (+0.87%) | 7.31 to 7.35 (+0.55%) | 1820.7 to 1809.4 (-0.62%) |

On vLLM `0.24.0` the changes are +0.11% to +0.75% for TTFT and -0.12% to -1.28%
for tokens per second per GPU. The `chunked_partial_cached` input (prefix 3000)
returned the same values as the uncached input in both variants, so `run_agg`
does not show partial-cache behavior here. The per-phase shapes below do.

### Packed telemetry, forward-pass time ms

| Backend | MTP | Pure prefill | Pure decode, 8 requests | Mixed, 1x2048 + 8 decode | Mixed, 2 requests 3000 tokens + 8 decode |
| --- | --- | --- | --- | --- | --- |
| TensorRT-LLM | 0 | 29.421 (0%) | 4.980 (0%) | 20.582 to 20.633 (+0.25%) | 25.660 to 25.753 (+0.36%) |
| TensorRT-LLM | 3 | 29.421 (0%) | 5.247 to 5.354 (+2.03%) | 20.588 to 20.728 (+0.68%) | 25.665 to 25.848 (+0.71%) |
| vLLM | 0 | 29.098 (0%) | 4.736 (0%) | 18.525 to 18.576 (+0.27%) | 23.546 to 23.602 (+0.24%) |
| vLLM | 3 | 29.098 (0%) | 4.989 to 5.096 (+2.14%) | 18.530 to 18.670 (+0.75%) | 23.551 to 23.697 (+0.62%) |

Pure decode at MTP0 is unchanged because the telemetry decode path already used
request counts. At MTP3 the candidate now converts requests to flattened tokens
for qualified Mamba, which raises the decode cost.

### Per-phase `mixed_step_latency` (TensorRT-LLM, MTP3)

| Shape | Baseline ms | Candidate ms | Change |
| --- | --- | --- | --- |
| Pure prefill, 4096 | 29.421 | 29.421 | 0% |
| Pure decode, 8 (through the mixed API) | 8.758 | 8.659 | -1.12% |
| Mixed, 2048 + 8 decode | 20.996 | 21.131 | +0.64% |
| Mixed, ctx 6000, prefix 3000, + 8 decode | 26.013 | 26.328 | +1.21% |
| Mixed, ctx 8128, prefix 1024, + 8 decode | 43.953 | 44.237 | +0.65% |

The decode component of the mixed step rises from 0.20 ms to 0.35 ms because
decode Mamba now runs at decode coordinates. The shared component rises for
partially cached and split-budget prefills because each chunk pays its own
recurrent state.

## Controls

- Unqualified parity: main, Baseline and Candidate give identical results for
  every shape, in `run_agg`, telemetry and per-phase output, for both backends.
- Qualification effect: qualified and unqualified runs differ even without the
  accounting change, because the analytical bound adds recurrent-state bytes for
  qualified operations. Compare Baseline with Candidate, not qualified with
  unqualified.
- Strict coverage failure: Nemotron-3-Super-120B-A12B FP8 on TensorRT-LLM
  `1.3.0rc20` and `b200_sxm` fails in both variants with a missing MoE row
  (`fp8`, 512 experts, hidden 1024, intermediate 2688, `moe_ep_size` 1). It is
  not replaced by another estimate.

## Host cost

Six alternating Baseline and Candidate pairs per workload ran on an otherwise
idle ARM host (16 pinned CPUs, one thread each). The workloads were the
`run_agg` script and one replay (vLLM, MTP3).

Recorded in the retained replay result files (12 files):

| Field | Boundary | Baseline | Candidate |
| --- | --- | --- | --- |
| `wall_s` | In-process time of the replay runner call, measured with `time.perf_counter()`. It excludes interpreter start-up, imports and configuration parsing. Unit: seconds. | median 1.627, range 1.622 to 1.660 | median 1.637, range 1.635 to 1.645 |
| `maxrss_kb` | Peak resident memory of the whole process at the end of the run (`ru_maxrss`, kilobytes on Linux). Shown here in MiB. | median 171, range 162 to 198 | median 162, range 162 to 198 |

The ranges overlap, and the six samples per variant are too few to separate the
two variants. This page makes no speed, memory or significance claim.

Transcribed observations: the whole-process times for `run_agg` (1.70 s for
both variants) and for replay (2.37 s for both variants), and the `run_agg` peak
memory (about 700 MB), were read from the printed timing log. That log was not
saved, and the `run_agg` result files have no timing or memory field. These
values are not independently auditable raw timing artifacts. No log was
reconstructed from them.

## Current-head check

`686275e7` and `b2b489d9` ran on the same x86 host through
`RustForwardPassPerfModel.estimate_forward_pass_time_ms`, with the same
qualification wrapper and inputs (Nemotron-3-Nano-30B-A3B, `b2b489d9` source
SHA-256 `f4c0650700bc7f1c7416553e84594db820e9ad1f1fe6baf81095dbbe42415ec4`).
The run covered vLLM `0.24.0` and TensorRT-LLM `1.3.0rc20`, qualified and
unqualified, MTP0 and MTP3.

| Shape | Qualified result, `686275e7` to `b2b489d9` |
| --- | --- |
| 2 prefills, 17 tokens, plus 1 decode | vLLM: 5.860174 to 5.853910 ms (MTP0) |
| 3 prefills, 4097 tokens, plus 8 decode | vLLM: 29.136896 to 29.130632 ms (MTP0) |
| Evenly divisible, pure prefill, pure decode | identical |
| All unqualified shapes | identical |

On vLLM every Mamba operation uses the analytical bound. The old-minus-new
difference for both uneven shapes is 6.264220e-03 ms. That is exactly one FP16
state pair (2 MiB) for each of the 23 Mamba layers at 7.7e12 B/s (relative
error 1e-13), the same at MTP0 and MTP3. Both heads price all 17 activation
tokens, so the old split charged three state pairs for two requests and the new
split charges two.

On TensorRT-LLM the difference is larger (0.16 ms for the 17-token shape),
because the context conv1d is a table lookup that depends on the batch and
sequence of each chunk group.

## Limits

- Summing uniform-batch queries can overcount launch overhead for
  variable-length batches. This is an accounting approximation. It is not
  silicon qualification.
- Results come from one model family, one modeled platform and synthetic
  shapes.
- A CPU-only host ran every number. The host type has no effect on the
  predictions, which match to the printed precision on two different hosts.
