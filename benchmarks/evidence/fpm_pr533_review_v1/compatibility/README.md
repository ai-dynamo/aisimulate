<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

The candidate at `1de653c9ef28788a877a660cb81cb3379edb8c6a` preserves ordinary
configuration compatibility with prior main
`3327f7c6214234c6cd2c6f9734a75129694c7a38`. The actual prior crate's
`ForwardPassPerfModelConfig` deserializer accepted all eight default cases:
`auto`, `op_level`, `fpm_interpolation`, and `fpm_regression`, each with learned
controls omitted or explicitly `{}`. The candidate omits the default learned
block from saved JSON. Seven nondefault learned controls remain serialized and
are explicitly rejected by the older reader. All 15 candidate roundtrips are
exact; both readers also construct the two regression cases.

`result.json` records compact per-case results and hashes. `source-manifest.json`
pins both revisions, binaries, source files, and the audit of all 258 archived
prior-source files against immutable Git objects. The prior schema is neither
mocked nor reconstructed: `prior_reader.rs` links the unchanged prior crate.
`check_readers.py` contains every fixture input and runs each reader in a separate
process. These checks cover configuration compatibility, not serving accuracy.

To repeat from a repository checkout with the recorded commits and cached Cargo
dependencies, first build the candidate native extension from its recorded
revision using the repository's normal build instructions. Then run:

```bash
COMPAT_DIR="$PWD/benchmarks/evidence/fpm_pr533_review_v1/compatibility"
COMPAT_WORK="$(mktemp -d)"
git archive 3327f7c6214234c6cd2c6f9734a75129694c7a38 Cargo.toml Cargo.lock crates | tar -x -C "$COMPAT_WORK"
mkdir -p "$COMPAT_WORK/crates/core/examples"
cp "$COMPAT_DIR/prior_reader.rs" "$COMPAT_WORK/crates/core/examples/pr533_prior_reader.rs"
cargo build --manifest-path "$COMPAT_WORK/Cargo.toml" --locked --offline --no-default-features \
  -p aisimulate-core --example pr533_prior_reader --target-dir "$COMPAT_WORK/target"
python3 "$COMPAT_DIR/check_readers.py" \
  --candidate-runtime /absolute/path/to/candidate/_runtime.abi3.so \
  --prior-reader "$COMPAT_WORK/target/debug/examples/pr533_prior_reader" \
  --output "$COMPAT_WORK/result.json"
```

Expected counts are **8 accepted defaults, 7 preserved/rejected nondefaults,
15 exact roundtrips**. Rebuilt binary hashes may differ with compiler, paths,
target, or profile; the recorded source hashes and behavioral checks remain
the comparison criteria. The reproduction harness adds only an SPDX comment
header to the originally executed harness, with both hashes recorded.
