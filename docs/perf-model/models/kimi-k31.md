<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kimi K3.1 Pro

Kimi K3.1 Pro reuses the Kimi K3 hybrid backbone and replaces its 24 full MLA
layers with gated DSA. The 69 KDA layers, latent MoE geometry, hidden dimension,
and attention-head geometry are unchanged in the configurations inspected at
[K3 revision f831ab6](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)
and [K3.1 Pro revision d9e99b0](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/config.json).
Load the Pro configuration dynamically using its own model identifier or a
local checkpoint directory. The Pro configuration is not bundled in AISimulate.

The parser recognizes `KimiK31ForConditionalGeneration` within the existing
KIMIK3 family, preserves its DSA configuration, and compiles sparse layers into
DSA module operators. Renaming Pro to K3 would discard this distinction.
The supported DSA geometry has 32 index heads of dimension 128, top-k 2048,
no index RoPE, and an independent indexer on every sparse layer.

The native module boundary includes attention projections, index selection,
sparse attention, the output gate, and output projection. Input normalization
and the attention tensor-parallel reduction remain separate model operators.
The gate projection and its parameter inventory are included once. Rust owns
the SOL calculation and uses the existing DSA roofline with K3.1 dimensions
and the additional gate work; hand-calculated tests cover prefill, cached-prefix
extension, decode, and parameter residency.

Top-k limits sparse attention reads, not resident history. Each sparse layer
keeps its full latent KV history and a separate 132-byte FP8 index entry per
token. The analytic model also permits hypothetical BF16 latent entries of
1152 bytes; the pinned SGLang runtime rejects BF16 KV, so this mode does not
claim serving support or measured coverage. For FP8 KV, an explicit
`flashmla` or `trtllm_mla` attention backend identifies the physical layout:
FlashMLA uses 656 bytes (FP8 latent, FP32 scales, BF16 RoPE), while TRTLLM
uses 576 raw FP8 bytes. Index storage is additional in both cases. KDA state
continues to use the existing K3 state model.

Initial modeling covers ordinary decoding (`nextn=0`) with CP=1 and DCP=1.
K3's DSPARK contract does not establish K3.1 speculative support. Automatic
vLLM physical state-cache sizing also needs a K3.1-specific runtime contract;
use measured/manual state capacity for that path.

The pinned native runtime requires FP8 KV and defaults both prefill and decode
to FlashMLA. Its TRTLLM option changes the physical cache format and uses raw
FP8 values; both phases must use the same layout. Collector rows record the
observed native pool layout in `kernel_source`, independently of the executed
attention kernel. In particular, a dense TRTLLM prefill kernel can consume a
FlashMLA packed cache. Rows with missing or ambiguous K3.1 layout markers are
rejected instead of being assigned by a kernel-name substring.

Op-based coverage is keyed by the Pro architecture and actual backend. K3,
DeepSeek, or GLM attention observations do not supply Pro DSA coverage. The
collector uses the native SGLang K3.1 module and a pinned runtime; initial
measurements and graph qualification are tracked separately from analytic SOL
support. Module measurements establish execution of their tested points, not
full-grid coverage or prediction accuracy. FPM data and its validation are
maintained in a separate change.

The model case declaration retains TP1/2/4/8 (96/48/24/12 local heads).
The native full-node adapter owns its visible GPU group and runs each selected
case in a fresh subprocess, using the standard selection, checkpoint and
failure accounting. B300 (SM103) has native TP1/2/4 measurements with FP8 KV
and eager forwards. TP8 remains unmeasured; other targeted SMs retain the
registry's unverified marker. Graph serving is not qualified by these eager
measurements.

Decode history is established by real native prefill through bounded request
continuations, preserving every latent KV and index-K entry. The final native
decode batch retains the requested total history and batch size. This untimed
preparation avoids the monolithic one-batch helper's long-history scratch and
projection failures; it does not reduce the measured workload or replace the
native attention path. The original failed campaign and the independently
collected continuation results retain separate source and case-plan receipts.

Runtime source inspected for the module boundary and cache layout:
SGLang source commit `afcd7d6b487a9cbe73e25941971b4cc107e6f9ae`, embedded in the
[official K3.1 image](https://hub.docker.com/r/lmsysorg/sglang/tags?name=dev-kimi-k31-pro)
at index digest `sha256:67fc77d1cbe67de27ca54d874c82757a4844743fea899ea7d0df547e30c2c391`,
`python/sglang/srt/models/kimi_k3.py` and the DSA memory-pool implementations.
The source identity comes from the official image's `.k31-source-commit`, not
from its mutable tag alone.

## Measured coverage and accuracy limits

The November 9 B300 dataset contains 39 context and 33 generation points across
TP1/2/4, with BF16 projections and FP8 packed FlashMLA KV. Context includes
128–16384 new tokens and cached-prefix extensions; generation includes
2048–131072 total history. The selected batch/length combinations, original
12 preparation failures and their successful follow-up are documented in the
[dataset README](../../../python/aisimulate/src/aisimulate_core/systems/data/b300_sxm/sparse_attention/sglang/0.0.0.dev1+g41208373c.d20261104/README.md).
All 72 published rows passed strict provenance and exact native Rust `silicon`
lookup. This checks data integrity; it does not measure end-to-end prediction
error against SemiAnalysis or qualify a trained full model.

The runtime uses one dummy-initialized DSA layer, eager execution, CP=1, DCP=1
and no speculation. TP8, other GPU systems, serving graph/overlap behavior and
SemiAnalysis configurations requiring speculation or context parallelism remain
outside this measurement scope. No prediction-accuracy result is claimed.
