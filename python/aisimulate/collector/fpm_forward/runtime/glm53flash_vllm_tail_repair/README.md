# GLM circular-tail runtime repair: qualification candidates

The old `0.30.0+glm53kpool.bf5f6b0e689d` runtime is quarantined after actual generic slot-mapping reads beyond its one-block circular tail table. These separately built distributions provide a reviewable candidate and controlled reference. Neither is admitted to production collection/prediction.

- `candidate/`: `0.30.0+glm53tail.eb4704514fdf` combines the retained unaligned-pooling repair with native `KpoolTailSpec.uses_slot_mapping=False`.
- `reference/`: `0.30.0+glm53tailref.4e4a40c2a838` changes only the tail mapping property. It can reference complete one-shot prompt prefill followed by single-token decode; its original helper must not process unaligned cached multi-token prefill.

Each directory preserves the immutable upstream commit, original/patched source hashes, reviewable diff, encoded patch, Apache-2.0 license, binary/source input identity, genuine-version build command, and actual ARM job 613836 build/install receipts. No binary wheel or credentials are vendored. The build uses the exact image-derived stock wheel and upstream precompiled-source route, verifies all 19 native binaries, and preserves their legal material. The actual verifier checks imported distribution/source identity and real native tail/ordinary spec behavior without initializing CUDA.

Reproduce inside the pinned ARM image with the original stock distribution installed:

    python candidate/build.py --source-archive UPSTREAM_ARCHIVE --stock-wheel ORIGINAL_BINARY_WHEEL --stock-wheel-sha256 fb53683eddeaddf1b069cad685bacd8d68e1c6b4a1eb30dddbe57ae2a69f5493 --output NEW_BUILD
    python -m pip install --no-deps --target PRIVATE_INSTALL NEW_BUILD/dist/ACTUAL_WHEEL.whl
    CUDA_VISIBLE_DEVICES= PYTHONPATH=PRIVATE_INSTALL python candidate/verify_install.py --build-receipt NEW_BUILD/build-receipt.json --wheel NEW_BUILD/dist/ACTUAL_WHEEL.whl --install-root PRIVATE_INSTALL --output NEW_INSTALL_RECEIPT.json

Use the corresponding reference directory in a separate clean process/installation. Source-archive and binary-origin JSON files provide exact source URLs and hashes. Actual build receipts provide the produced filenames/digests. A copied expected digest alone is not actual install evidence.

Required model qualification remains separate: sixteen independent cache-arithmetic cases; all four checkpoint/TP cells with tail-only one-shot, candidate one-shot and candidate split native requests; actual cache-group flags/positions/circular addresses and complete native sampling histories; ordinary 128K regression; then five-warmup/ten-measurement retained-state/capacity controls. Any new build or runtime identity requires its own qualification. All failures remain visible. No complete cell or accuracy acceptance follows from the CPU receipts.

`original-build-inputs.json` records exact task-local sources executed for ARM job 613836. Repository formatting and completion of the NVIDIA copyright header on the original install-verifier wrapper is documented by `packaging-lineage.json`; the patch and runtime source bytes are unchanged. Qualification status and production admission remain closed independently of these files.

ARM job 614605 subsequently executed the exact packaged verifiers against both unchanged 613836 installations. Each `actual-packaged-verifier-receipt.json` records the imported version, all 31 source and 19 binary hashes, native spec behavior, and that CUDA remained uninitialized. Its adjacent launch receipt preserves the actual read-only mounts, command, exit status, and frozen input inventory. `packaged-verifier-executed-inputs.json` describes the files at execution time, before the packaging lineage was updated with this result. The original 613836 receipts remain unchanged. This verifies the packaged wrapper; model correctness and formal collection admission remain separate gates.
