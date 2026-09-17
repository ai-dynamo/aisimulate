# FPM accuracy evaluation

Development-only evaluation code adapted from NVIDIA AISim FPM Gym at
`e8221729db2802e822f6919fd68bc2941743385b`:
https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-fpm-gym/-/tree/e8221729db2802e822f6919fd68bc2941743385b/src/aisim_fpm

Licensed under Apache-2.0 with maintainer-confirmed migration permission.
See the root THIRD_PARTY_NOTICES.md and LICENSE.

HF manifest/protocol loaders, rank-aware measurement types, FPM staging, and
worker-isolated regression adapters retain upstream behavior. Imports were
renamed to `fpm_accuracy`; the op-based adapter and experimental registry were
removed. Unused presentation metadata and its conversion helpers are omitted;
the worker schema and MoE mapping required for evaluation are retained.
Unsupported measurement protocols remain visible as unsupported
configurations; missing protocol identities and corrupt inputs still fail closed. `evaluate.py` reduces each shared measurement stream directly into
overview aggregates, without local reports, raw result exports, or history.

Native AISim imports remain deferred in the adapted adapters so parser and fake
predictor tests work without an installed native extension. The real campaign
checks that the native SDK is installed before evaluating any case.

Hub cache loading supports repository-local blobs and the marked cache-wide
shared blob store used by huggingface-hub 1.32. Manifest hashes still bind the
measurement and FPM bytes; arbitrary symlink targets outside these stores are
rejected. Local dataset checkouts retain their strict root boundary.

Listener window and single-rank chronology keys use milliseconds so mixed
streams preserve predict → score → tune ordering. Missing MoE parallelism
defaults to one; malformed values and unknown recorded precisions fail closed.
Worker regression verifies required store names before scoring and reports
contract mismatches in Actions logs while retaining measurement coverage.

`requirements.in` declares evaluator dependencies. `requirements.txt` locks
their transitive dependencies and distribution hashes for Python 3.12; CI
installs it with `--require-hashes`. Regenerate with:

```bash
uv pip compile scripts/fpm_accuracy/requirements.in --generate-hashes \
  --python-version 3.12 --universal \
  --output-file scripts/fpm_accuracy/requirements.txt
```

Install test tools such as pytest separately from the hash-checked campaign
requirements; they are not part of the daily evaluation environment.

The selected AISim wheel and its runtime dependencies are installed separately
because evaluated branches can declare different runtime requirements. The
campaign checks the wheel hash and runs `pip check` after both installs.
