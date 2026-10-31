# Qwen3.5-397B-A17B-NVFP4-V2 configuration provenance

`nvidia--Qwen3.5-397B-A17B-NVFP4-V2_config.json` and
`nvidia--Qwen3.5-397B-A17B-NVFP4-V2_hf_quant_config.json` are unmodified
copies of `config.json` and `hf_quant_config.json` at
https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4-V2/tree/8f590eae8f10bf55d9a46f79ea0280bde435c9f8

| Local file | Upstream file | sha256 |
| --- | --- | --- |
| `nvidia--Qwen3.5-397B-A17B-NVFP4-V2_config.json` | https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4-V2/blob/8f590eae8f10bf55d9a46f79ea0280bde435c9f8/config.json | `9408a9e559cc2f05f0b357738213666353e6651160ce8ff477b1c26982bc4f63` |
| `nvidia--Qwen3.5-397B-A17B-NVFP4-V2_hf_quant_config.json` | https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4-V2/blob/8f590eae8f10bf55d9a46f79ea0280bde435c9f8/hf_quant_config.json | `2be0a371edb1983ace9b776b12ff3a19e2b4841efc15c69df8d9f4aa5ec1c56c` |

The files describe an NVIDIA ModelOpt MIXED_PRECISION export of
Qwen/Qwen3.5-397B-A17B. Its `text_config` is identical to that of the bundled
`nvidia/Qwen3.5-397B-A17B-NVFP4` configuration, and its quantization recipe
differs. The routed experts are NVFP4 with 4-bit activations. In the 60
decoder layers, the Gated DeltaNet (GDN) projections `in_proj_qkv`,
`in_proj_z` and `out_proj`, the full-attention projections and the
shared-expert projections are static per-tensor FP8. The GDN `in_proj_a`,
`in_proj_b` and `conv1d`, the router, the shared-expert gate, the LM head, the
embeddings, the multi-token prediction (MTP) layer and the vision tower stay
BF16. The KV cache is FP8.

Copyright owners: NVIDIA CORPORATION & AFFILIATES (the quantized export and
its quantization metadata) and Alibaba Cloud (the base Qwen3.5-397B-A17B
configuration).

The upstream model card (`README.md` at the same revision) declares
`license: apache-2.0` and `base_model: Qwen/Qwen3.5-397B-A17B`, and its
"License/Terms of Use" section names the Apache License 2.0. The upstream
repository ships no separate license or NOTICE file. The base model
repository declares the same license, and its `LICENSE` at revision
`8472618112abcbd45acbcdc58436aff4233c23f7` states "Copyright 2026 Alibaba
Cloud". The text of the Apache License, Version 2.0 is in the `LICENSE` file
that is distributed with AISimulate.
