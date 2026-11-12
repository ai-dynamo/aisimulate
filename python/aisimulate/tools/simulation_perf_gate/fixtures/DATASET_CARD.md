---
license: apache-2.0
pretty_name: CC Traces — Weka, With Subagents, 256k cap, v7 only (Jun 21 2026)
task_categories:
  - text-generation
tags:
  - llm
  - inference
  - benchmarking
  - kv-cache
  - agentic
  - multi-turn
  - claude
  - subagents
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: traces.jsonl
---

# semianalysisai/cc-traces-weka-062126-256k

> Abridged by NVIDIA from the
> [upstream dataset card](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/blob/8fecd2fc56694469f758f0afbbb6335ad3043740/README.md)
> at revision `8fecd2fc56694469f758f0afbbb6335ad3043740`: the distribution-plot
> section is omitted. The statistics below describe the full upstream dataset;
> see [README.md](README.md) for this benchmark's four-play subset.

WekaTrace corpus derived from SemiAnalysis Claude Code proxy traces. Built 2026-06-21 17:49:45 UTC via `utils/agentic/build_weka_hf_dataset.py`.

Derived from [semianalysisai/cc-traces-weka-062126](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126) by applying the 256k per-request cap and preserving the surviving requests' relative timestamps.

## Filters

- Trace version: exactly v7
- min Anthropic requests per session: 20
- Claude Code CLI ≥ 2.1.139 (every row)
- peak concurrent sub-agent groups ≤ 10
- Non-image rows only (image content excluded at source)
- Classifier calls excluded (`max_tokens<=64 AND no tools` → SUGGESTION MODE, title-gen, Security Monitor)
- Exact-duplicate proxy rows deduped by `(timestamp, model, in, out, dur_ms, agent_id)`
- Dynamic-workflow-bug sessions excluded: Claude Code CLI < 2.1.174 emitted dynamic-workflow subagents without a subagent-label header, so they appear as many interleaved unlabeled trajectories in one session. Dropped when peak concurrent unlabeled multi-turn trajectories ≥ 3 (changelog 2.1.174).
- 256k per-request cap (see *256k filter rule* below)

## 256k filter rule

- Per-request `input + output ≤ 256,000` tokens. Applied at request granularity, not conversation. Main-agent and sub-agent inner requests are evaluated independently.
- Sub-agent groups where *every* inner request was filtered → entire group dropped.
- Sub-agent groups with only *some* inners filtered → partial group kept (surviving inners retained).
- Timeline preservation: surviving request `t` values keep their original relative offsets, including sub-agent overlap. If the first request was filtered, all surviving timestamps are shifted by one uniform offset so the earliest survivor starts at `t = 0`.

## Stats

```
traces:             393
main_turns:          28,444
subagent_groups:           1,697
subagent_inner_requests:          39,822
total_model_requests:          68,266
total_input_tokens:   6,891,228,864
total_output_tokens:      58,728,807
```

## Source script

```
python utils/agentic/sample_proxy_traces.py --out '<workdir>/proxy' --sampling top --min-trace-version 7 --max-trace-version 7 --min-requests 20 --require-cli-min 2.1.139 --max-parallel-subagents 10 --exclude-dynamic-workflow-bug
```

## Loader plugin

Load in aiperf via:

```
--public-dataset semianalysis_cc_traces_weka_with_subagents_256k
```
