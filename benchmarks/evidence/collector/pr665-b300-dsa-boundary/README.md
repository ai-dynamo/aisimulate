# B300 DSA timing boundary and physical-coordinate normalization

PR #665 retains the native-QKV measurements at the 308 physical coordinates
that overlap the previous SGLang 0.5.21 table. The shipped table has **6,312
rows: 5,812 unchanged earlier rows plus all 500 unchanged new rows**. Of the
500 additions, 192 introduce new physical coordinates. No global loader
precedence changes. `original-append.parquet` archives the complete unmodified
6,620-row append, including all superseded rows. `normalization.json` records
the physical key and before/after SHA256 values.

The physical key includes kernel source, architecture, compute/KV/projection
precision, local heads, batch, ISL, TP and step. Model labels do not separate
the loader's physical coordinates. Normalization makes the selected value
independent of parquet row order. Original collector revisions, hashes and
plan hashes remain unchanged; the older event's retained row count decreases
by 308. The archived table preserves its complete original event.

## Same-node evidence

Job **1001596**, one B300 on AWS PDX Slurm, compared collectors
`ee219cc60e13e9f60db50b63f13bab536c598781` and
`0bba6709ed4ef3a0fcae878d2ad2fb61bbd11a04` in old/new/new/old order,
with ten overlapping coordinates per full/skip-indexer mode. Both used the
same SGLang 0.5.21 image and eight local heads, FP8 KV, BF16 projections,
TP8-local geometry, two dummy layers. Sources/image hashes are recorded in
`input-manifest.json`; full paired values and telemetry summaries are in
`paired-summary.json`.

- Median new/old ratios: **1.14885 full-indexer, 1.15487 skip-indexer**;
  maximum process-repeat spread 2.10%.
- Serving uses native `prepare_qkv_latent` and fresh `AttentionInputs`.
  The newer collector includes this work; the older collector reused
  synthetic warmup QKV. Pinned SGLang revision
  `e00930c5489053f26d86b179cee0d087f846acbb`,
  `python/sglang/srt/models/deepseek_v2.py` and
  `python/sglang/srt/layers/layer_boundary/adapters/attention.py`.
- Changing only the QKV callback diagnostically reduces median ratios to
  **1.04322 / 0.99590**. This explains most of the gap; the remaining 4.3%
  full-indexer difference is unisolated, and remains a stated limitation.
- Collector head `2579763d1882b3259843c7d2ee4084b8b95d1ef5` agrees with the
  measurement revision at median 1.00277. This is a bounded experiment, not
  validation of every historical overlap or full TP8 serving.
- Exact module B52/B56 probes do not establish serving's padded-graph parity.
  Batch-52 rows remain subject to the coordinated #664 serving-graph fix;
  these archived historical measurements must not be relabeled as padded.

Diagnostic rows in `paired-summary.json` are **not publishable performance
data**. They are not read by the shipped performance database. The archive is
the historical measurement table, not the diagnostic table.

## Reproduction

Run `python benchmarks/evidence/collector/pr665-b300-dsa-boundary/validate.py`
from the repository root with pandas and pyarrow installed. It verifies exact
row preservation, physical-key uniqueness and both table hashes.

`paired_worker.py`, `run_pairs.py`, `paired.sbatch` and `summarize.py` preserve
the original experiment harness. Their historical remote source archives are
hash-identified in `input-manifest.json`. Raw result root:
`/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec/results/pr664-gpu-review-20261102/sglang/pairs-1001596`.
The run used about 0.09 allocation GPU-hours; that historical allocation
figure is not a collection-cost estimate.
