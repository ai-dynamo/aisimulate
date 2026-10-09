# FPM measurement storage migration

2026-10-08 local CPU checks on macOS arm64, Python 3.11, PyArrow 25.0.1.
Legacy HF source: `68fa3add95b32a0399d781b043cb0f1008c8040d`.
Migrated source: the manifest v5 working tree accompanying this change.

All 39 current/historical snapshots have identical ordered predictor-facing rank
payloads, actual latencies, logical source membership and exclusion counts:
3,185,145 accepted observations including historical snapshots. Paths and their
identity hashes are intentionally excluded from this equality check.
The [machine-readable comparison](reader-comparison.json) includes each snapshot.

| Representative current configuration | Truth size MiB, legacy → Parquet | Reader seconds | Peak MiB |
|---|---:|---:|---:|
| DeepSeek V4 Pro GB300 DEP8 (reduced records) | 6.98 → 7.71 | 8.79 → 8.75 | 449 → 486 |
| Kimi K3 GB300 TEP8 (rank observations) | 6.06 → 4.94 | 10.45 → 10.83 | 559 → 605 |
| GLM 5.2 GB200 DEP16 (grouped iterations) | 80.61 → 52.47 | 55.58 → 58.28 | 1,444 → 2,057 |

Each reader ran in a fresh process, sequentially on the same host. Timings include
checksum validation and existing semantic decoding; OS caches were not cleared.
Peak memory includes the evaluator's full in-memory observation population.
These single-run measurements do not establish a speedup: Parquet reduces some
compressed sizes but increases others and currently uses more peak memory here.
Conversion and decoding use bounded batches; the evaluator's existing population
materialization is unchanged.

Kimi is the first verified request-chart capture. The published migration audit
records the exact source hashes, run boundaries, unknown fields, and renamed assets.
Request charts describe profiling requests; warmup/drain records remain separate.
No new measurements, rank alignment, prediction policy or evaluation selection
was introduced.
