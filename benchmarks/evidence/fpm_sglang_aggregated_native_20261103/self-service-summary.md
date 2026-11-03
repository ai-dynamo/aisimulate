<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Aggregated SGLang native collection: self-service evidence

This summary records the completed collection used to inform the
[self-service worker example](../../../docs/perf-model/fpm-self-service/examples.md#example-d-collect-native-fpm-from-an-aggregated-worker).
The complete raw capture and experiment helper package are separate artifacts;
this file is a summary of inspected receipts, not a replacement for those bytes.

The deployment used `deepseek-ai/DeepSeek-V4-Flash-0731` revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`, four NVLink-connected B200 GPUs,
and one aggregated TP4/EP4/DP1/PP1 worker. SGLang source was
`e087e662ba1ac4ef7747537e2a9141085efd4561`; Dynamo source was
`7931147d403c554575438c2451da4248e53e10b8`. The imported
[corrected native producer](../fpm_pr280_agentx_native_20261031/native-patches/README.md)
hash was verified. No FPM data hook was installed, and duration labels retain
native CUDA time in seconds.

| Check | Recorded result |
| --- | --- |
| AgentX source | 393 native roots from `semianalysisai/cc-traces-weka-062126-256k`, revision `8fecd2fc56694469f758f0afbbb6335ad3043740` |
| Traffic | Concurrency 15, seed 42, 3600.000050157-second sending window |
| Drain | 21.813574614 seconds |
| Successful requests | 164 warmup and 1,039 profiling; no errors, cancellations or skips |
| Request/server-usage matching | All 1,203 requests matched |
| Persisted input | 352,764 native records in two closed gzip segments; both segment hashes rechecked |
| Profiling work rows | 1,329 prefill and 348,058 decode; 349,387 total |
| Drain work rows | 2,522 decode |
| Decode accounting | 935,813 expected and observed request-steps across warmup, profiling and drain |
| Cleanup | Aggregate worker exited zero after publisher cleanup; owned processes closed |

Overlap, mixed scheduling, speculation, attention DP and prefill CP were
disabled. This run supplies no mixed-row coverage and does not qualify timing
with overlap enabled. Thirteen busy rows outside client phases were smoke before
cache reset. Phase exports use relay-observation windows; request totals include
drain and describe a different interval from the profiling-only export.

**Publisher losslessness remains unproven.** This producer can discard events
before assigning a counter. Valid gzip, continuous counters, drained shutdown
and matching decode accounting cannot prove that every prefill or idle event
was published. This collection performed no GBDT fit or measured prediction
accuracy assessment, and does not establish SGLang Replay compatibility.

The separately retained experiment package has SHA256
`9e998b6cc61cae7ef6680903d3367f3410b153cf1623a1c7853eda737e50f112`
and manifest SHA256
`2551e15dede65cd21d1d42980f69e3c6b7e661a95039243d0b040de10afb6aa5`.
Its extraction receipt verified 755 payload files. No public download URL is
recorded for it. The public guide therefore uses the native worker settings and
local validation helpers with a user's own deployment and workload; it does not
promise exact reproduction of that experiment's client and supervisor.
