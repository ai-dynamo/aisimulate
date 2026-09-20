<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang Rust frontend timing patch

`sglang-v0.5.19-timing.patch` adds optional timing to the multimodal workers
of SGLang's embedded Rust server so `aisimulate.vl.calibrate` can measure the
Rust frontend's service time per request. It applies to
[sgl-project/sglang v0.5.19, commit `0bcd822377da7b5718e674eaf9c870d349424dd1`](https://github.com/sgl-project/sglang/tree/0bcd822377da7b5718e674eaf9c870d349424dd1)
and touches `rust/sglang-mm/src/driver.rs`,
`rust/sglang-server/src/multi_modality/worker.rs`,
`rust/sglang-server/src/tokenizer_manager/{to_scheduler,channel}.rs` and
`rust/sglang-server/src/lib.rs`; the added `rust/sglang-server/src/ais_timing.rs`
is original helper code. Worker count, image order, CPU affinity and the
request/result protocol are unchanged, and nothing is recorded unless
`AIS_MM_TIMING_PATH` is set.

The upstream code is copyright SGLang Team and SGLang contributors, Apache-2.0;
the full license is in `SGLANG-LICENSE` and the attribution in the repository
`THIRD_PARTY_NOTICES.md`.

## Use

Apply the patch to a dedicated SGLang checkout, never to the source tree that
is serving:

```bash
git rev-parse HEAD   # 0bcd822377da7b5718e674eaf9c870d349424dd1
git apply --check /path/to/aisimulate/tools/frontend/sglang-v0.5.19-timing.patch
git apply /path/to/aisimulate/tools/frontend/sglang-v0.5.19-timing.patch
```

Launch the server from that checkout with `AIS_MM_TIMING_PATH` pointing at a
writable JSONL file, drive it with the calibration workload (same image size,
count and encoding the prediction will use), then lower the recording:

```bash
AIS_VL_BOUNDARY_TRACE=1 AIS_MM_TIMING_PATH=/tmp/rust-timing.jsonl python -m sglang.launch_server ...
python -m aisimulate.vl.calibrate --frontend rust --model Qwen/Qwen3-VL-8B-Instruct \
  --images 1024x1024 --encoding jpeg --rust-timing /tmp/rust-timing.jsonl --output profile.json
```

The calibrator takes a request's service time from its `rust_worker`
`boundary_span` row (`started_ns`/`ended_ns` on `CLOCK_MONOTONIC`), which
covers everything the multimodal worker does for the request: payload
conversion, fetch, content hash, decode, patchify, tokenization and token
layout, M-RoPE, feature packing, the optional shared-memory copy and the
sidecar park. Those rows exist only with `AIS_VL_BOUNDARY_TRACE=1`; without
them the calibrator refuses the recording. The rows carrying
`image_timings_ns` (decode and patchify nanoseconds per image) describe the
inner steps and are kept as provenance only. HTTP handling and the scheduler
handoff are not worker time. The recording must cover concurrency levels
`1..mm_workers`, each with enough steady samples, or the lowering fails
naming the missing levels.
