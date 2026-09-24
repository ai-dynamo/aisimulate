# GLM IndexPool runtime repair candidate

**Status: NOT_QUALIFIED.** This directory preserves a separately versioned native
vLLM repair candidate. It does not select or admit that runtime in a collector.
The stock-runtime rejection for cached prefill beginning inside a pool remains
active. Requested failing geometries stay visible; no point is dropped.

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
source/binary/runtime checks. Full Engine qualification is pending. Only after
that result may the shared runtime contract admit the exact new identity.
No performance rows, graph Ops support, full-matrix coverage, FPM10% or Ops20%
accuracy pass is claimed by this bundle.


Offline request identity validation follows native `InputProcessor.assign_request_id`
(input_processor.py:262–279), `random_uuid` (utils/__init__.py:11–12), and
`OutputProcessor` (output_processor.py:384) at the immutable revision above.
The exact source hashes are in `qualification/request-id-source.json`. Native
worker IDs must be a bijection with the complete external ID followed by a
hyphen and exactly eight lowercase hexadecimal characters. Every pair also
requires identical prompt bytes/hash and the complete native sampled-token
chain on every TP rank. This is an offline validator correction: historical
GPU logs and the original failed validator result remain unchanged; revalidation
receipts identify the new validator hash separately. It does not qualify the
candidate runtime or relax its native-state/functional gates.


New qualification runs additionally hash-check the parent request-ID APIs and
observe the original `InputProcessor.assign_request_id` before/after IDs in
`request-id-map.jsonl`. They retain default native randomization and record
only after the original assignment returns. The declared
`native_assign_request_id_v1` protocol requires that witness to agree with
every completed worker chain; missing or altered assignments fail validation.
Historical runs without that declaration continue using the explicitly named
`pinned_source_offline_bijection` correction above.

The dormant repaired-runtime consumer contract preserves stock manifest bytes
and extends its own closure with the native Engine qualification sources plus
`v2-source-sha256.json`. These additional hashes identify upstream
vllm-project/vllm@ced6857afa0ea7b2e3f0846a62e1394e90f15607 under
`vllm/v1/worker/gpu/{cudagraph_utils,attn_utils,warmup}.py` and
`vllm/v1/worker/gpu/model_states/mamba_hybrid.py` (Apache-2.0). They cover the
actual V2 graph, attention-cache initialization, warmup and hybrid state path;
the legacy runner file alone is insufficient. No upstream implementation is
copied by this manifest. Each repaired worker must hash the effective vLLM
source files and all 19 native binaries before timing. Python and Rust repair
allowlists remain empty until an immutable native Engine qualification receipt
has been reviewed and explicitly admitted.
