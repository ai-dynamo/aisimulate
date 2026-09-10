# Frozen calibration and independent native forward study

This extends the earlier pilot without replacing it. Both text AR profiles ran
126 frozen calibration configurations and 38 separate forward holdouts on four
GB300 GPUs. Every calibration and holdout configuration has one native warmup
and three measured repetitions. No holdout module timing enters a calibration
table.

| Profile | Calibration configurations | Module table points | GEMM / MoE / NCCL points | Independent forward holdouts |
| --- | ---: | ---: | ---: | ---: |
| `full` | 126 | 836 | 32 / 16 / 32 | 38 |
| `decoder_bounded` | 126 | 830 | 32 / 16 / 32 | 38 |

The module totals are 756 / 750 attention points, plus 32 shared-linear,
32 Engram and 16 mHC points per profile. These are physical operator keys,
not independent workload counts. Each profile preserves 378 measured model
invocations and 504 warmup/measured progress records per rank. Baselines use
16 token counts and two warmups plus three repetitions for each kernel.

The frozen configuration grid and its sampling rationale remain in
[`../study-plan/`](../study-plan/) and [`../SAMPLING.md`](../SAMPLING.md).
Calibration uses cached prefixes 0, 128, 512 and 1536; queries include 127, 128
and 129 around the replay boundary. The common domain is batch 1–2, at most
512 new tokens in a batch, and at most 2048 cached tokens per decode request.
Dynamo's decode coordinate is **past KV**: native SGLang reads that prefix plus
its new token. Both coordinates are retained. Capacity was increased to 8192
slots and the native prefill chunk limit to 4096 so the maximum two-request
seed and inclusive decode fit without changing any planned geometry.

## Measurement boundaries

`calibration/evidence/` contains local-component CUDA event observations and
separate instrumented forward observations. Component recording serializes the
mHC statistics stream and keeps collectives outside local component intervals.
The instrumented forward observations are not independent validation results.

`heldout/evidence/` contains native synchronized wall measurements around the
existing `one_batch` prepare/forward/sample boundary. It contains no component
recorder or baseline rows. This boundary retains the native mHC stream schedule,
shared Engram hash/history work, metadata preparation and sampling. Its results
are distinct from HTTP latency and the serving runtime's GPU FPM intervals.
For every repetition, the published latency is the maximum across ranks 0–3;
all three maxima and their median remain in `heldout/forward-results.json`.

The three repetitions are an initial precision check. Bounded case
`prefill-0017` (batch 2, 96 new tokens per request, prefix 0) has rank maxima
548.985, 379.867 and 187.826 ms after a roughly 181.9 ms warmup. All ranks
show the variation and pass the source, finite-logit and completion checks;
the available evidence does not establish its cause. These observations are
retained. A separate uniform ten-repeat followup has now completed all 38 cases
in each profile, with one warmup per case. It uses the same producer, source,
checkpoint, input and geometry. The new observations are in
`<profile>/precision-v2/`, alongside exact comparisons with the original attempt;
neither attempt is pooled or replaced. Each profile preserves 380 measured
model invocations and 418 warmup/measured progress records per rank.

| Precision attempt | Median within-point CV | Maximum within-point CV |
| --- | ---: | ---: |
| `full`, ten repeats | 0.404% | 3.570% |
| `decoder_bounded`, ten repeats | 0.408% | 7.857% |

CV is sample standard deviation divided by the mean of the ten rank maxima.
These statistics describe within-attempt repeatability, not an accuracy bound
or confidence interval. The formerly unstable bounded `prefill-0017` now has a
187.947 ms median and 0.353% CV; its original 379.867 ms median remains recorded.
The largest new bounded CV is `prefill-0004` (batch 1, query 48, prefix 128):
nine repetitions are 192.527–194.363 ms and the tenth is 242.580 ms. All ten are
retained. The median signed change in per-case medians between attempts is
−5.284% for `full` and +1.936% for `decoder_bounded`; additional repetitions do
not eliminate between-attempt variation or identify its cause. Frozen
calibration-table hashes and the unchanged original observation hashes are in
[`precision-v2-receipt.json`](precision-v2-receipt.json).

Both phases explicitly disable custom all-reduce and FlashInfer all-reduce
fusion, disable shared-expert fusion, and disable prefill/decode CUDA graphs.
Shared and routed experts follow the native serial eager path. Engram tables
remain TP-sharded in HBM. The runtime, checkpoint, corpus and source hashes match
the earlier pilot; capacity and communication flags are recorded explicitly.
The direct baseline uses Torch NCCL 2.29.7. The image also initializes PyNccl
2.30.7; those version labels must not be conflated. The explicit eager flags
and source dispatch support the Torch collective path, as documented in
[`collective-dispatch-receipt.json`](collective-dispatch-receipt.json).
The pilot is retained separately and is not silently merged into this study.

## Admission and prediction comparison

All 1666 module table points reproduce their measured latency through the
strict native SILICON reader. Raw observations rebuild every emitted module and
baseline table. Both original and precision-v2 sets of 38 independent forward holdouts pass
rank, repetition, warmup, finite-logit, source, input and execution-contract checks.
These checks establish artifact integrity, not prediction accuracy.

The frozen key analysis predicts full interpolation coverage for 38/38 `full`
holdouts and 28/38 `decoder_bounded` holdouts. Ten bounded cases need late-layer
attention prefix buckets absent from calibration. Strict SILICON must retain
those missing results; HYBRID may report explicit SOL fallback. Independent
prediction errors, source breakdowns and the separate HTTP/FPM comparison are
reported in PR #160. A missing result must not become a zero-error observation.

Shared Engram hash/history and framework metadata are not explicit operators in
the current graph. Existing embedding, normalization and activation estimates
remain empirical. Component calibration therefore does not establish measured
coverage of every part of a complete forward pass. Other content distributions,
heterogeneous requests, larger batches, longer KV and graph execution remain
outside this study's qualified domain.

Select one `study/<profile>/systems` overlay explicitly and set decoder replay
to match it; keep shared-layer reuse disabled. From the repository root, run:

```bash
PYTHONPATH=python/aisimulate/src:python/aisimulate \
  python/aisimulate/.venv/bin/python \
  data/experimental/deepseek-v41/gb300-silicon/verify_study.py
```

The native producer is pinned to AISimulate
`6853dab695eede0ca02ea3fb901312766b0d08c6`; each table sidecar records the producer
and workload-helper hashes. The original image digest and actual source hashes
are retained in each profile. Root `THIRD_PARTY_NOTICES.md` covers source-derived
contracts. No checkpoint weights or internal infrastructure identifiers are
included.
