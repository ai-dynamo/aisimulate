# GLM IndexPool runtime repair candidate

**Status: quarantined after actual circular-tail slot-mapping out-of-bounds evidence.**
Diagnostic612960 copied original native inputs on both GB300 TP ranks and showed
position-derived reads exceeding the actual tail block-table tensor extent.
The first three calls returned; the fourth failed inside the unchanged original
slot-mapping call. The common KPool tail source is shared by FP8 and NVFP4.

The earlier four-cell functional suite below remains historical evidence. It
no longer grants runtime admission, and partial FPM measurements are retained
outside delivered tables. Python's production repair allowlist is empty; Rust
rejects this exact runtime for both phases, including exact hits and decode
baselines. A new immutable runtime and new qualification are required. The
original receipts are preserved without rewriting their historical status.

The stock GB300 probe observed incorrect pooled cache entries for retained
prefix4097 followed by query3 or4, while aligned-prefix controls, one-shot
prefill and retained tails passed. The repair uses the request's actual retained
circular tail to complete the first partially filled pool, masks each request's
own block, and writes each completed pool once before the native tail update.
It preserves the native pooling/FP8 transform and aligned path. The patch is
not a collector-side reconstruction of attention state.

## Immutable source and attribution

The lossless `retained-tail-prefill.patch.b64` decodes to the exact original
patch bytes/hash. `retained-tail-prefill.review.diff` is a readable display with
blank context-line trailing whitespace removed; it is not a build input. This
keeps the original candidate identity unchanged while respecting repository
whitespace checks. The build decodes and verifies the original patch before
applying it. Reproduce the lossless conversion with Python:

```python
import base64
from pathlib import Path
original = Path("retained-tail-prefill.patch").read_bytes()
Path("retained-tail-prefill.patch.b64").write_bytes(base64.b64encode(original) + b"\n")
restored = base64.b64decode(Path("retained-tail-prefill.patch.b64").read_bytes().strip(), validate=True)
assert restored == original
```

The patch modifies
[vllm/model_executor/layers/sparse_attn_indexer_kpool.py](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/sparse_attn_indexer_kpool.py)
at revision `ced6857afa0ea7b2e3f0846a62e1394e90f15607`. Copyright contributors to
the vLLM project, Apache-2.0. The patch preserves the upstream copyright and
license identifiers and marks the NVIDIA modification. Full upstream LICENSE
is adjacent; the immutable upstream root contains no NOTICE. Original build and
qualification wrappers reference the same native revision; no native compute
implementation is copied into those wrappers.

`patch-identity.json` is the historical, immutable pre-build identity input. Its
`formal_runtime_version=NOT_BUILT` describes that input's creation time.
`build-receipt.json` and `qualification-status.json` separately record the actual
subsequent build and current admission status. They must not be conflated.

## Actual candidate and binary lineage

The genuine built distribution is `0.30.0+glm53kpool.bf5f6b0e689d`. Its wheel SHA256
is `a3b63cb3c95cf976f717077102e8172a33501c7d05092bc84cc58e3aaef47d36`.
The modified helper SHA256 is
`2aa61ce832e530f07a33ee2884a92bc30fca4f693b83174546020b70dca211e8`.

CPU job604885 used upstream's own source build, `VLLM_VERSION_OVERRIDE` and
`VLLM_PRECOMPILED_WHEEL_LOCATION`; both distribution METADATA and `_version.py`
contain the new version. No installed package version was spoofed. All19 native
binaries and binary-base legal files survived byte-for-byte. The task-private
installation passed actual imports/source/binary checks in CPU job604941.
The wheel is intentionally excluded from Git; its exact external location is
recorded in `qualification-status.json`.

The binary input is transparently **image-derived**, not the original PyPI
artifact. The official PyPI wheel matched18 installed `.so` files, but its
extensionless `vllm-rs` executable differed from the immutable stock image.
Job604724 rejected that mismatch. A private binary-input archive replaces only
that executable with the actual image export, updates its RECORD entry, and
checks every other member is unchanged. `binary-base-origin.json` preserves both
origins, hashes and the two changed archive members. The official source build
then creates the actual new distribution. Never install the private binary-input
archive as if it were the original release.

## Rebuild

Run in the pinned ARM64 base image after obtaining the exact source archive from
`source-archive.json` and exact binary input from `binary-base-origin.json`:

```sh
python build.py --source-archive /inputs/upstream.tar.gz \
  --stock-wheel /inputs/image-derived-binary-base.whl \
  --stock-wheel-sha256 fb53683eddeaddf1b069cad685bacd8d68e1c6b4a1eb30dddbe57ae2a69f5493 \
  --output /new/build-output
```

The checked-in portable driver applies the patch to the verified source archive;
the historical job604885 driver copied the already-patched helper. Both require
identical original/patch/result hashes and use the same upstream build route.
The driver never modifies the installed base. Required build helpers must exist
in an isolated build environment; job604885 added task-private
`setuptools-rust==1.11.1` and `semantic-version==2.10.0` without changing Torch.
All original failed jobs remain historical evidence.

## Qualification gates

The source-overlay cache diagnostic passed16/16 GB300 cases in job604684,
including heterogeneous requests, nonuniform gates and short tails. The adjacent
receipt records actual bytewise pooled-cache/tail versus one-shot/oracle results.
That is a cache correctness diagnostic, not full Engine or performance acceptance.

`qualification/` contains the ordinary public Engine probe: original requests,
real native retained state and completed worker/sample witnesses; stock one-shot
versus candidate one-shot, then candidate split versus candidate one-shot, with
exact output-token comparison. Its README describes the fixed geometries and
source/binary/runtime checks. All four production-policy cells passed: FP8 TP4
(job 606669), FP8 TP2 (608011), NVFP4 TP2 (608013), and NVFP4 TP4 (608014). Each
cell retained all three profiles, 20 requests per profile, 32 greedy output
tokens per request, and 40 exact comparisons with no differences. The current
strict validator rechecked all original raw files; an independent review also
compared original prompts, outputs and finish status across all four cells.

This qualifies the frozen functional geometry suite only. Formal campaign
hardware/capacity qualification, performance rows, graph Ops support, full
eight-configuration coverage, FPM 10% and Ops 20% accuracy remain NOT_EVALUATED.


Offline request identity validation follows native `InputProcessor.assign_request_id`
(input_processor.py:262–279), `random_uuid` (utils/__init__.py:11–12), and
`OutputProcessor` (output_processor.py:384) at the immutable revision above.
The exact source hashes are in `qualification/request-id-source.json`. Native
worker IDs must be a bijection with the complete external ID followed by a
hyphen and exactly eight lowercase hexadecimal characters. Every pair also
requires identical prompt bytes/hash and the complete native sampled-token
chain on every TP rank. This is an offline validator correction: historical
GPU logs and the original failed validator result remain unchanged; revalidation
receipts identify the new validator hash separately. That historical correction
alone did not qualify the candidate runtime or relax its native-state/functional
gates; the completed four-cell evidence below establishes the bounded pass.


New qualification runs additionally hash-check the parent request-ID APIs and
observe the original `InputProcessor.assign_request_id` before/after IDs in
`request-id-map.jsonl`. They retain default native randomization and record
only after the original assignment returns. The declared
`native_assign_request_id_v1` protocol requires that witness to agree with
every completed worker chain; missing or altered assignments fail validation.
Historical runs without that declaration continue using the explicitly named
`pinned_source_offline_bijection` correction above.

The historical repaired-runtime consumer contract preserved stock manifest bytes
and extends its own closure with the native Engine qualification sources plus
`v2-source-sha256.json`. These additional hashes identify upstream
vllm-project/vllm@ced6857afa0ea7b2e3f0846a62e1394e90f15607 under
`vllm/v1/worker/gpu/{cudagraph_utils,attn_utils,warmup}.py` and
`vllm/v1/worker/gpu/model_states/mamba_hybrid.py` (Apache-2.0). They cover the
actual V2 graph, attention-cache initialization, warmup and hybrid state path;
the legacy runner file alone is insufficient. No upstream implementation is
copied by this manifest. Each repaired worker must hash the effective vLLM
source files and all 19 native binaries before timing. The former exception
applied only to `0.30.0+glm53kpool.bf5f6b0e689d` and its immutable qualification
summary. That exception is now revoked; the production repair allowlist is empty.

## Historical reviewed qualification evidence

The reviewed `qualification/admission-summary.json` has SHA256
`d43dfdcfabe870cc51983fa41fada4897b4d84d64ac57fafe2236e7753435e67`. The historical evidence validator
requires these exact bytes and validates the complete packaged receipt chain;
this validation no longer grants production admission. The original raw
directories remain external. Small receipts and validator sources were copied
to task-owned Lustre and verified by reading their actual remote bytes before
constructing the final summary URIs.

The summary has `schema_version: 1`, status
`native_engine_qualification_passed`, and scope
`native_Engine_functional_correctness_for_frozen_geometry_suite_only`. It binds
`backend_version`, `build_receipt_sha256`, `wheel_sha256`, and
`expected_runtime_sha256` to the constants in `glm53flash_runtime_identity.py`.
`accuracy_acceptance` and `formal_8_cell_coverage` must both remain
`NOT_EVALUATED`; native functional correctness does not establish performance
accuracy or eight-configuration coverage.

The receipt contract also requires:

- `source_commit`: the immutable 40-character revision of the strict validator.
  `validator_sources` maps `validate.py`, `probe.py`, `worker_probe.py`,
  `request-id-source.json`, and `cohort-source.json` to SHA256 values of the
  actual files in this packaged `qualification/` directory. These identify the
  current full-raw revalidation, independently of historical collection code.
- `cells`: exactly four entries with `checkpoint` (`fp8` or `nvfp4`) and `tp`
  (`2` or `4`). Each `comparison` contains `path`, `sha256`, and `source_uri` for
  the original strict comparison JSON. It must report production policy,
  20 requests per profile, 32 greedy output tokens per request, 40 comparisons,
  `status: passed`, and no differences.
- Each cell's `profiles` contains, in order, stock/reference,
  candidate/reference, and candidate/split. Each profile has `runtime_kind`,
  `mode`, `raw_uri`, `preflight: {path, sha256}`, and
  `native_receipt: {path, sha256}`. Paths are distinct flat JSON basenames in
  `qualification/`; the existing wheel include rule ships them. Preserve the
  original preflight and native receipt bytes, even when the original receipt
  was written by an older validator.
- Each profile's `original_validator: {sha256, source_uri}` records the actual
  frozen historical `validate.py` bytes and their external location. Obtain this
  identity from the original frozen job bundle, not from the current checkout.
  `revalidation_added_fields` explicitly lists only the newly derived fields:
  `checkpoint_identity` when absent from the historical receipt, followed by
  `files.native-receipt.json`. No other native evidence or original raw file
  hash may change. A new original receipt already containing checkpoint proof
  lists only `files.native-receipt.json`.

All external URIs use `https`, `s3`, `gs`, or `ssh`, without credentials, query
strings, fragments, or traversal. Each profile has a distinct raw directory.
The comparison retains the complete original raw inventory: all worker,
prompt, forward, output, request-ID, cohort, and configuration files remain
externally available by location and SHA256. The consumer checks every packaged
small-file SHA against that inventory, the actual checkpoint config and both
loader revisions, the runtime/source/19-binary closure, native cohort protocol,
all TP ranks, and FULL graph evidence.

The accepted summary records current strict revalidation from source commit
`aae82cc48f2e05a1a077893ba42152d8708771de` and preserves the exact four comparison
JSON files. It separately binds the original frozen v3 validator
(`839af86e26a41a14a79ed301c71e495c571d16970f6c37d0802a30fd7d3fb5e2`),
original invocations and unmodified raw inventories. Historical failed jobs and
receipts remain historical failures. Newly derived checkpoint identity and
receipt-file hashes are explicitly labeled rather than attributed to the old
validator. This small-file consumer verifies the reviewed evidence chain offline;
it does not reread large
external token arrays or independently recreate a GPU execution. Unit tests
construct only temporary `TEST_ONLY` contracts and do not supply publication or
admission artifacts.
