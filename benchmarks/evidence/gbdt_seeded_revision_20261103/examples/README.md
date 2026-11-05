# Seeded example generation and verification

Final structured receipts, reproduction commands, the packaging helper and
installed-wheel verification remain in this review. Duplicate rendered output
and preparation-run details are preserved in the
[immutable example evidence](https://github.com/ai-dynamo/aisimulate/tree/6fe7a3f5304c8fa707ed76571a4d51caec7d236a/benchmarks/evidence/gbdt_seeded_revision_20261103/examples).
Executed JSON receipts are unchanged: their historical log paths and hashes
refer to files in that checkout. Use it to reproduce the recorded runs with
their original sources and native build; this cleanup reruns no measurements.

The final packaged source was regenerated through the public example using
producer PR3 `3e6e82c1b67e4448da01d60ae9d06893d1f8489f` and native
`6a12ba8d1af12a1930d8a030a400b22aa2aacd439f458002488adb71a2f42d42`.
The [packaging record](final-packaging.json) verifies the producer SDK bytes
against that immutable revision; the PR4 generator is pinned separately by its
source digest. No artifact embeds its own future commit hash.

| Final check | Result | Record |
| --- | --- | --- |
| Public B→C→B+C lifecycle | All ten checks pass; complete decode readiness and 40 accepted C observations | [Receipt](final-receipt.json) |
| Packaged verification | Seven checks pass, including source lineage and explicit/installed/discovered prediction equality | [Verification](final-verification.json) |
| Base runtime verification | Passes with every sklearn import rejected | [Trainer-free log](https://github.com/ai-dynamo/aisimulate/blob/6fe7a3f5304c8fa707ed76571a4d51caec7d236a/benchmarks/evidence/gbdt_seeded_revision_20261103/examples/final-trainer-free.log) |
| Tamper rejection under `python -O` | Altered evaluation lineage is rejected | [Tamper log](final-tamper.log) |
| Deterministic source bytes | Bundle, seed and inputs exactly match preparation; build-info now records final provenance | [Byte comparison](final-artifact-determinism.json) |
| Installed application wheel | End-to-end workflow passes outside the checkout, including seeded source/snapshot APIs, public example, trainer-free verification and tamper rejection | [Installed wheel](installed-wheel.json), [log](installed-wheel.log) |

The bundle manifest SHA-256 is
`800ad6f0d1c1f850fe5fc907db9216b62f2fe8425723f69e3c665e1497a1d71c`.
[final-commands.json](final-commands.json) records the exact commands and outcomes.
The initial manual tamper command incorrectly supplied a directory to
`--bundle-path`; that invocation is retained in `final-tamper-invalid-cli.log`
and is not counted as a validation result. The corrected command supplies
`bundle.json` and verifies the intended failure.

To reproduce using the selected source/interpreter, choose new output paths:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m aisimulate_core.sdk.gbdt_example run \
  --input-dir /private/tmp/gbdt-seeded-revision-inputs-20261103 \
  --output-dir NEW_RUN
python benchmarks/evidence/gbdt_seeded_revision_20261103/examples/package_example.py \
  --repo . --run NEW_RUN \
  --inputs /private/tmp/gbdt-seeded-revision-inputs-20261103 \
  --expected-native-sha256 6a12ba8d1af12a1930d8a030a400b22aa2aacd439f458002488adb71a2f42d42 \
  --producer-revision 3e6e82c1b67e4448da01d60ae9d06893d1f8489f
python -m aisimulate_core.sdk.gbdt_example verify-bundle --output-dir NEW_VERIFICATION
```

`package_example.py` is a maintainer artifact-generation helper and writes the
packaged example source directory. Ordinary users can run the public module
against the shipped bundle without this step.

The final wheel SHA-256 is
`f8cc4bd57e8d8a9c08af28d345dd129bdf2eea548308e3cc6dc6c9146c62495e`.
Its native extension is byte-identical to qualified build 6a12ba8d. The
[release build](wheel-build.json) and installed test both pass; the latter takes
29.88 seconds and uses a separately installed environment outside the source
checkout. Receipts for [the installed lifecycle](installed-example-run.json),
[packaged verification](installed-example-verify-bundle.json) and
[trainer-free verification](installed-example-without-trainer.json) are retained.
The initial collection attempt lacked registration for the disabled macOS timeout
marker; its log is preserved. Registering that marker in the command line allowed
collection without modifying product or test code.

## Preserved preparation evidence

The preparation run uses candidate native binary
`7fe75475aca65819a9174942ded464ff49060f082357753bcde5286a2028a0d0`.
Its producer and generator hashes are in `preparation-packaging.json`. This was
a dirty working tree based on the merged PR #658 inputs, not a clean build of
the recorded old HEAD. These preparation receipts remain unchanged and must
not be relabeled as final receipts.

The public module generated an external 48/16/40-iteration training, evaluation
and C-feedback dataset under `/private/tmp/gbdt-seeded-revision-inputs-20261103`.
The decode-only B seed then received C on that same model instance. The run checked
complete decode execution readiness, prediction before admission, explicit flush,
relocated fixed and continued B+C snapshots, rejection of fixed-state tuning,
and bit-identical continued feedback against an uninterrupted twin. Its separate
application cursor binds the consumed input offset to the feedback and snapshot
digests. This is a recovery example, not an exactly-once delivery service.

Executed from the candidate worktree with its own `.venv/bin/python`:

```sh
python -m aisimulate_core.sdk.gbdt_example generate \
  --output-dir /private/tmp/gbdt-seeded-revision-inputs-20261103
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m aisimulate_core.sdk.gbdt_example run \
  --input-dir /private/tmp/gbdt-seeded-revision-inputs-20261103 \
  --output-dir /private/tmp/gbdt-seeded-example-preparation-20261103
python benchmarks/evidence/gbdt_seeded_revision_20261103/examples/package_example.py \
  --repo . \
  --run /private/tmp/gbdt-seeded-example-preparation-20261103 \
  --inputs /private/tmp/gbdt-seeded-revision-inputs-20261103 \
  --expected-native-sha256 7fe75475aca65819a9174942ded464ff49060f082357753bcde5286a2028a0d0
python -m aisimulate_core.sdk.gbdt_example verify-bundle \
  --output-dir /private/tmp/gbdt-seeded-bundle-verification-20261103
```

`preparation-receipt.json` records the ten successful lifecycle checks.
`preparation-verification.json` records native verification of explicit,
installed and catalogue-discovered sources, all with complete decode coverage.
`preparation-trainer-free.log` repeats verification with a Python import finder
that raises for every `sklearn` import. `preparation-tamper.log` records the
expected rejection of altered evaluation provenance under `python -O`.
Documentation links, Ruff and formatting checks are recorded separately.

`package_example.py --producer-revision REV` checks producer SDK bytes against
that immutable revision before recording final provenance. The generator has
its own source digest because it is delivered in PR4 after the producing PR3
revision; generated files do not embed their own Git commit hash. Native bytes
are pinned independently. The final generation above reran the public example
and recorded a new receipt before using this option.

These self-authored arithmetic inputs establish functional behavior only.
They contain no hardware measurements and provide no serving accuracy, memory,
mixed/cross-rank, original-A-prior, selector or performance qualification.
