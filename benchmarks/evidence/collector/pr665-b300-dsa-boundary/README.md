# B300 DSA timing boundary and physical-coordinate normalization

PR #665 retains the native-QKV measurements at the 308 physical coordinates
that overlap the previous SGLang 0.5.21 table. The shipped table has **6,312
rows: 5,812 unchanged earlier rows, 456 unchanged November 2 additions and
44 batch-52 rows recollected on November 3**. Of the 500 additions, 192
introduce new physical coordinates. No global loader
precedence changes. `original-append.parquet` archives the complete unmodified
6,620-row append, including all superseded rows. `normalization.json` records
the physical key and before/after SHA256 values.

The physical key includes kernel source, architecture, compute/KV/projection
precision, local heads, batch, ISL, TP and step. Model labels do not separate
the loader's physical coordinates. Normalization makes the selected value
independent of parquet row order. Original revisions and hashes are retained
for their own rows. The older event's retained count decreases by 308; each
November 2 event retains 228 of its original 250 rows. `source_campaign_rows`
preserves raw campaign counts. Two new 22-row events identify the actual
recollection. The shipped table is a derived normalization, not a raw
6,312-row collector output. The archive preserves the complete original events.

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
- Historical exact module B52/B56 probes do not establish serving's padded-graph
  parity. The 44 historical batch-52 rows are archived and replaced by fresh
  measurements below, rather than relabeled as padded.

Diagnostic rows in `paired-summary.json` are **not publishable performance
data**. They are not read by the shipped performance database. The archive is
the historical measurement table, not the diagnostic table.

## Corrected batch-52 recollection

Job **1024142** on B300 completed both native worker groups with **44/44 rows,
zero errors and zero refused points**. Each group requested the exact 22
batch-52 histories present in the final signed collector's model-owned plan,
through past length 720,896. Source:
`c4b44d81f526b3c5299f7c15ccafe02d90b39229`; collector hash and canonical
executor-row plan hashes are retained in each `recollection/cases/*/result.json`
and the packaged sidecar. This is a declared subset, not a full-model sweep.

The corrected collector executes a native 56-slot module graph for 52 real
requests, with four native sentinel slots. The native `load_batch` buffer
comparison and consumer-key regression are in #664's
`benchmarks/evidence/collector/pr664-padded-serving-b300`. The row key remains
52 real requests; a B56 row would include four additional real histories.
This is module-level evidence, not full-model TP8 E2E qualification.

The two `run_mla_module_worker` calls took **353.263 and 350.622 seconds**:
**703.885 seconds = 0.195524 GPU-hours** total. These calls include child
startup/imports, model initialization, preparation, warmup and timed forwards.
Parent bootstrap and CPU plan enumeration before the calls were not separately
timed. Allocation time is excluded; the scheduler's 13m15s duration is not
used as collection cost.

`recollection/production_worker.py` verifies every requested physical case
against the canonical opt-in plan before invoking the unchanged worker. The
attempted subset's real executor IDs and their canonical `case_plan_hash` are
recorded; no full-grid or empty-checkpoint hash is substituted. The exact
executed artifacts are archived with executed/published hashes. None of the
paired diagnostic measurements is published.

## Reproduction

Run `python benchmarks/evidence/collector/pr665-b300-dsa-boundary/validate.py`
from the repository root with the AISimulate native extension, pandas and
pyarrow installed. It verifies original/published artifact hashes, the exact
44-row replacement, preservation of the other 456 additions and 5,812 older
rows, physical-key uniqueness, and real Rust consumer lookups for added,
recollected and unchanged controls.

`paired_worker.py`, `run_pairs.py`, `paired.sbatch` and `summarize.py` preserve
readable copies of the original experiment harness, with SPDX headers and
normalized whitespace. `executed-artifacts.zip` preserves exact executed
bytes; `artifact-hashes.json` records executed and published SHA256 values.
Their historical remote source archives are
hash-identified in `input-manifest.json`. Raw result root:
`/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec/results/pr664-gpu-review-20261102/sglang/pairs-1001596`.
The run used about 0.09 allocation GPU-hours; that historical allocation
figure is not a collection-cost estimate.

## Runtime image identity

`image-index.json` is the unmodified registry manifest at immutable digest
`sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`.
Its AMD64 child is `sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75`,
matching the native image used by the paired experiment and recollection. The
packaged sidecar records the index; the harness receipt identifies the executed
child. Hashing this manifest reproduces the sidecar's index digest.
