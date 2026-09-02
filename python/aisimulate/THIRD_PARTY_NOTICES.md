<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Notices

AISimulate contains source material derived from the projects identified below.
These notices apply only to the identified third-party material. Dependencies
installed separately by package managers are governed by the license material
distributed with those packages.

## vLLM

The following files are derived from vLLM's attention test utilities at tag
`v0.11.0` (commit `b8b302cde434df8c9289a2b465406b47ebab1c2d`):

- `collector/vllm/utils.py`
- `collector/vllm/utils_xpu.py`

Upstream source:
https://github.com/vllm-project/vllm/blob/v0.11.0/tests/v1/attention/utils.py

Copyright contributors to the vLLM project.

vLLM is licensed under the Apache License, Version 2.0. The full Apache-2.0
license text is reproduced in `LICENSE`.

## DeepEP

The following files are derived from DeepEP test utilities at commit
`73b6ea4a439ba03a695563f9fd242c8e4b02b37c` and contain NVIDIA modifications:

- `collector/wideep/sglang/deepep/test_internode.py`
- `collector/wideep/sglang/deepep/test_intranode.py`
- `collector/wideep/sglang/deepep/test_low_latency.py`
- `collector/wideep/sglang/deepep/utils.py`

The patch `collector/wideep/vllm/patches/deepep_73b_nvl4.patch` is also a
modification of DeepEP source at that commit.

Upstream source:
https://github.com/deepseek-ai/DeepEP/tree/73b6ea4a439ba03a695563f9fd242c8e4b02b37c

MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## DeepSeek model configuration files

The following model configuration files are copied from, or formatting-only
adaptations of, the named DeepSeek model repositories:

| AISimulate file | Upstream revision |
| --- | --- |
| `src/aiconfigurator_core/model_configs/deepseek-ai--DeepSeek-R1_config.json` | `deepseek-ai/DeepSeek-R1@56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad` |
| `src/aiconfigurator_core/model_configs/deepseek-ai--DeepSeek-V3_config.json` | `deepseek-ai/DeepSeek-V3@e815299b0bcbac849fa540c768ef21845365c9eb` |
| `src/aiconfigurator_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json` | `deepseek-ai/DeepSeek-V3.2@c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3` |
| `src/aiconfigurator_core/model_configs/deepseek-ai--DeepSeek-V4-Flash_config.json` | `deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1` |
| `src/aiconfigurator_core/model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json` | `deepseek-ai/DeepSeek-V4-Pro@b5968e9190ef611bbf34a7229255be88a0e937c1` |

Upstream repositories:
https://huggingface.co/deepseek-ai

MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Hugging Face Transformers

The `rotate_half` function in `collector/trtllm/collect_mla.py` is copied from
`transformers.models.llama.modeling_llama.rotate_half` in the Hugging Face
Transformers project at tag `v4.57.1`:

https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py#L109-L113

Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.

Transformers is licensed under the Apache License, Version 2.0. The full
Apache-2.0 license text is reproduced in `LICENSE`.
