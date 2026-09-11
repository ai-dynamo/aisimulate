<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AgentX hardware experiments

Each directory identifies the model, published AgentX point and hardware. These
are hardware measurement records and preparation logs, not AISimulate accuracy
claims. Follow each experiment's log for current validation and limitations.

| Model | AgentX point | Hardware / experiment |
| --- | --- | --- |
| GLM-5.2 | [440958](https://inferencex.semianalysis.com/inference/agentic/440958) | [B200 Slurm reproduction and results](agentx-glm-5.2-440958-b200-slurm/README.md) |
| GLM-5.2 | [440082](https://inferencex.semianalysis.com/inference/agentic/440082) | [GB200 Kubernetes HiCache](agentx-glm-5.2-440082-gb200-hicache/README.md) |
| DeepSeek-V4-Pro | [440845](https://inferencex.semianalysis.com/inference/agentic/440845) | [B300 Slurm reproduction and results](agentx-deepseek-v4-pro-440845-b300-slurm/README.md) |
| DeepSeek-V4-Pro | [440246](https://inferencex.semianalysis.com/inference/agentic/440246) | [B200 vLLM native FPM reproduction and results](agentx-deepseek-v4-pro-440246-b200-vllm/README.md) — off/on completed, job `4228930` |
| MiniMax-M3 | [439922](https://inferencex.semianalysis.com/inference/agentic/439922) | [B200 vLLM G2-off reproduction and results](agentx-minimax-m3-439922-b200-vllm/README.md) — off/on measured, split allocations; FPM captured |

On September 9, 2026, `agentx-b200-slurm` was renamed to
`agentx-glm-5.2-440958-b200-slurm`, and `agentx-gb200-hicache` was renamed to
`agentx-glm-5.2-440082-gb200-hicache`. Existing result files, launcher behavior,
remote artifact paths and deployed resource names were preserved.

[Fixed 8K/1K SGLang FPM concurrency sweep](sglang-fpm-fixed-8k1k/README.md)
measures repeated off/on gaps at concurrency1/32/128, separately from AgentX replay.
