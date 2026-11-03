<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Synthetic seeded decode GBDT example

This small decode-only B bundle demonstrates source verification and a seeded
B→B+C snapshot workflow. Targets come from arithmetic formulas authored for
AISimulate. It contains no hardware measurements, downloaded dataset bytes, model
weights or third-party-derived fixtures. It does not establish real-model accuracy,
memory capacity, replay fidelity or serving suitability.

The identity `aisimulate/synthetic-fpm-lifecycle-v1` on
`synthetic-functional-only`, with worker role `decode` and attention DP 1, is a
functional example. It is not a deployable model or supported hardware profile.
This directory has no operational catalogue entry. Its single pure-decode store
covers the complete execution requirements of this example's role; it makes no
prefill, aggregated or cross-rank coverage claim.

The installed generator is `aisimulate_core.sdk.gbdt_example`. Inputs and bundle
contents use this repository's Apache-2.0 license. `inputs/inputs.json` records
formula, units, generator digest, exact input hashes and counts. Source inputs
remain alongside the derived seed. `build-info.json` records the actual generation
build, dependencies, training recipe and completed functional checks.

| Input | Iterations | Purpose |
| --- | ---: | --- |
| `inputs/training.jsonl` | 48 | Synthetic B training only |
| `inputs/evaluation.jsonl` | 16 | Functional checks excluded from training |
| `inputs/c_feedback.jsonl` | 40 | Application-owned C demonstration |

The decode target is `700 + 35*batch + batch*context` integer microseconds,
converted to FPM `wall_time` seconds. Record identities are disjoint between
splits. Periodic feature coordinates can repeat; separate evaluation records test
API behavior rather than statistical generalization. These formulas are not
estimates calibrated to a real system.

The accepted B seed uses 12 trees, at most seven leaves, learning rate 0.1,
minimum leaf count 4, log-ms targets, seed 0 and no early stopping. The bundle's
`training_recipe` records that actual offline estimator, trainer version, feature
ABI and hyperparameters. It does not declare runtime first-fit capability. The
packaged seed contains B only, with no C admission lineage.

Verify installed bytes without importing the optional trainer:

```sh
python -m aisimulate_core.sdk.gbdt_example verify-bundle --output-dir /tmp/gbdt-bundle-verification
```

This validates all inputs, lengths/counts, training/evaluation lineage and generator
digest. Explicit, privately installed and catalogue-discovered sources must have
complete decode readiness and identical native prediction bits on all sixteen
evaluation queries. The command writes a resolved config and verification receipt.
Destinations must be new. Editing the generator requires regenerating this bundle
because its input manifest pins the exact installed generator bytes.

To exercise B training and later application-owned feedback, use the `learned`
extra with supported scikit-learn 1.9.1:

```sh
python -m aisimulate_core.sdk.gbdt_example generate --output-dir /tmp/gbdt-example-inputs
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m aisimulate_core.sdk.gbdt_example run \
  --input-dir /tmp/gbdt-example-inputs --output-dir /tmp/gbdt-seeded-example
```

The application reads C from the external input file, predicts before admission,
and updates the actual B-seeded instance. It explicitly flushes after twenty C
observations, saves/relocates a ready B+C snapshot, and records the feedback digest,
consumed offset and snapshot digest in a separate `feedback-state.json`. Twenty
is an example checkpoint boundary, not a training threshold. A fixed restore
rejects tuning; a continued restore resumes at the saved offset and must match an
uninterrupted model bit for bit after all forty observations. The final snapshot
and its config are written separately. `run` can generate these same synthetic
inputs itself when `--input-dir` is omitted.

The input cursor illustrates application ownership and digest binding, not a
production transaction or exactly-once delivery system. Applications own telemetry,
rank synchronization, sampling, deduplication, order, initial fitting, admission
cadence, durable cursor commits and error recovery. AISimulate owns seeded updates
and model-state persistence. A failure after partial admission needs reconciliation
before retry; source-free collection and missing-store training are not provided
by this runtime.

`receipt.json` records the loaded native digest, SDK origin, dependencies, input
and bundle hashes, readiness and checks. Cross-platform offline training need not
produce identical floating-point bits; distributed bytes are pinned by hashes.
Fixed and continued snapshots preserve the model that actually created them.

This example excludes prefill/mixed/cross-rank stores, original-A fallback,
selector experts, independent architecture/resources validation, live telemetry
transport and hardware/performance qualification. Those boundaries must be
validated separately for an actual deployment.
