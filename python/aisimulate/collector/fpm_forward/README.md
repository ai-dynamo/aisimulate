# Native FPM source recipes

`python -m collector.fpm_forward.recipe` reproduces a separately published,
source-pinned collection recipe. It complements the existing vLLM benchmark
entry point; it does not treat a SGLang HTTP campaign as that benchmark.

Use the pin shipped with the selected dataset. It specifies a Hugging Face
**dataset** repository, a full 40-character commit, the source archive filename,
and its SHA256. Authenticate with `hf auth login` if required. The wrapper uses
the existing HF CLI authentication and never accepts a token on its command line.

```bash
python -m collector.fpm_forward.recipe \
  --pin /path/to/published-source-pin.json \
  --destination /path/to/new-reproduction
```

This downloads and verifies source only. Add `--run`, followed by `--` and the
recipe's documented arguments, to execute its pinned Python or Bash entry point.
The destination must be new. An already downloaded archive can be supplied with
`--archive`; the same SHA256 and complete member verification still apply.
Run the recipe in its documented environment and owned allocation. The wrapper
does not request GPUs or alter the recipe's model, native backend, input plan,
startup, timings, deadlines, or acceptance policy.

The archive must contain regular files only, including `recipe.json`. Its schema
is `aisimulate.fpm-source-recipe.v1`, with these fields:

- `files`: each source path mapped to its exact `sha256` and integer `bytes`.
  This covers every archive member except `recipe.json` itself; the outer archive
  SHA binds that manifest.
- `entrypoint`: one `path` from `files`, and `interpreter` equal to `python` or
  `bash`. Arguments are passed as a list without shell interpolation.
- `readme`: a source member documenting prerequisites, invocation, input roles,
  runtime/source identities, and native acceptance.
- `licenses`: applicable license, attribution and notice members from `files`.
  The archived recipe must retain upstream revisions and modified-file notices.

Use a tar or compressed tar archive without directory, symlink, hardlink, or
device entries. Each source file remains unchanged and read-only in `source/`;
write collection outputs to a separate directory. The retained archive is
rehashed against the supplied pin before execution, and every materialized
source member is checked again. Python bytecode creation is disabled.

The wrapper retains materialization failures and actual subprocess exit codes.
An exit of zero only reports that the pinned process exited successfully. GPU,
HTTP, native geometry, complete calibration, heldout validation and prediction
accuracy must pass the archived recipe's own gates. Startup observations and
heldout samples must not enter calibration tables. Historical failed runs remain
failed evidence when a later source revision succeeds.

These archives are executable source, so review the published source and its
provenance before using `--run`. Hashes establish identity; they do not establish
trust or replace the native qualification required by the dataset.

## DeepSeek V4.1 GB200 TP4 full and decoder bounded

From `python/aisimulate`, with the project environment installed:

```bash
python -m collector.fpm_forward.recipe \
  --pin collector/fpm_forward/recipes/dsv41_gb200_tp4_full.json \
  --destination /path/to/new-reproduction
python -m aisimulate_core.sdk.fpm_dataset \
  src/aisimulate_core/systems/profiles/dsv41_fpm/four_gpu_hf_dataset.json \
  gb200-tp4-full --cache-dir /path/to/fpm-cache
```

The first command pins the HF source artifact commit; the second pins the
separate data commit and prints the systems root for prediction. The source
archive's `README.md`, `entrypoint.py` and `replay-contract.json` document CPU
reduction and prediction, including the exact raw/prepared archive paths and
hashes. Download those evidence archives from the data manifest's immutable HF
revision. CPU replay verifies the original native admission, reproduces the
145-row table byte for byte, and evaluates 38 independent heldout geometries
with ten observations each. It needs no GPU or private campaign directory.

For the independently collected bounded profile, use
`recipes/dsv41_gb200_tp4_decoder_bounded.json` in the same recipe directory and
`gb200-tp4-decoder_bounded` as the dataset key. Use a separate materialization
directory for each profile. The bounded archive adds its own CPU replay entry;
the original full entry and pin are unchanged. Its explicit native API predicts
38/38 heldout geometries (380 observations), MAPE **1.12313769%**. The aggregate
API predicts 26/38 (260 observations), conditional MAPE **1.23254318%**, and keeps
its existing rejection of 12 multi-prefill geometries lacking per-request extend
lengths. Per-request witnesses come from the frozen request and native geometry
proof, never from latency or fitted residuals.

New GPU collection requires the archive's original qualified runtime and
checkpoint. Its retained Slurm launch files record the original site paths;
they are not a portable allocation service. Other GPU/TP/profile combinations
are not implied by this pin. The `dev-800cc9…-cohort…` selector identifies the
image and cohort adapter; the recorded SGLang Git revision is
`1aa0e962b206102b7c439a4a0c4981cfec6e87bc`.
