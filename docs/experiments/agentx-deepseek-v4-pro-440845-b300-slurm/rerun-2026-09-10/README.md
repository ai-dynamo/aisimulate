<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AgentX DSv4 rerun after the fixed-workload FPM sweep

Job `4227233` targets `umb-b300-dp-142`, the same node used for the completed
[fixed8K/1K experiment](../../sglang-fpm-fixed-8k1k/README.md). The fixed sweep
showed small throughput gaps; this rerun checks whether AgentX reproduces its
prior gap and whether request context/cache distributions explain it.

The workload is [AgentX440845](https://inferencex.semianalysis.com/inference/agentic/440845):
DeepSeek-V4-Pro, c32, all393 Weka trajectories, seed42, TP8EP8attentionDP8PP1,
HiCacheoff, synthetic EAGLE/native-MTP acceptance2.49. Both runs use the same
FPM-fixed amd64 image digest
`sha256:f856a45537f82e1900ea7607edcbaa7f77fbb2e70220eae522d1d50d0046727e`.

Run off then on, each with fresh engine/KV and equivalent warmup, then600s of
measured sending, per the request for shorter diagnostic iterations. FPM-on records all eight rank streams. Serving/router settings
match the original AgentX experiment, including maxrunningrequests64; this differs
from the fixed sweep's256-slot pool. The goal is to reproduce the AgentX comparison,
not treat the fixed workload and replay as otherwise identical configurations.

Dataset setup uses `AIPERF_DATASET_WEKA_PARALLEL_WORKERS=1` to avoid the previously
verified blocked-SIGCHLD forkserver cleanup bug. The client documents identical
reconstruction semantics for serial/parallel paths. Its content-addressed mmap
cache is located at `/home/scratch.hongkuanz_gpu/agentx-dsv4-rerun-mmap-cache-20260910/`
to avoid small-home ENOSPC. Both cases use the same setup settings; these do not
change the engine or the authored trace. Startup wait is bounded at3600s.

Files are staged at `/home/scratch.hongkuanz_gpu/agentx-dsv4-rerun-20260910/` with
the pinned thinking template/license and existing image-squashfs checksum.
`submit.sh` requests one exclusive8B300node for at most2.5hours. Raw results:
`/home/scratch.hongkuanz_gpu/agentx-dsv4-rerun-results/job-4227233/`.

After both runs, inspect final client validity/errors/cancellations and FPM
integrity, then compare matched source-request keys and distributions of total
prompt length, uncached prompt tokens and prefix reuse. Keep unmatched/recycled
requests visible. Stratified replay evidence can support a workload-effect
hypothesis but does not by itself establish causality.

Configuration provenance remains SemiAnalysisAI/InferenceX
`fb85931b1edec09f9498509835a8c814bebe3c65`,
`benchmarks/single_node/agentic/dsv4_fp4_b300_sglang_mtp.sh`, Apache-2.0.
The original thinking template is a runtime dependency, not vendored here.

## Short diagnostic window

The user requested approximately10-minute measurements to accelerate iteration.
Both off/on cases use `--benchmark-duration 600 --unsafe-override`: the pinned
AgentX scenario normally enforces a900-second minimum. Consequently these runs
are explicitly diagnostic, and `submission_valid=false` is expected from that
protocol override. Request errors, cancellations, coverage and FPM validity still
need independent checks. Keep warmup separate, and do not compare these as formal
one-hour leaderboard submissions. A short run may miss late cache-pressure or
eviction effects.
