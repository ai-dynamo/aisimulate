# GLM-5.3-Flash configuration provenance

The adjacent `zai-org--GLM-5.3-Flash_config.json` is an unmodified copy of
https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json

`nvidia--GLM-5.3-Flash-NVFP4_config.json` and its `_hf_quant_config.json`
are unmodified copies of `config.json` and `hf_quant_config.json` at
https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/09b04e5e74bca08ca8549fc736d4cdd8624bfde3
The NVIDIA derivative identifies MIT as its license and Z.AI as its upstream author.
The configs describe the full checkpoint; AISimulate's initial graph covers text AR only.

MIT License

Copyright (c) 2026 Z.AI Co., Ltd

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

Execution grouping and analytical adaptations reference vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`,
`vllm/models/glm5next/nvidia/{model,attention,kda}.py` and
`vllm/model_executor/layers/sparse_attn_indexer_kpool.py`, and SGLang
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`,
`python/sglang/srt/models/glm5_next.py`, `python/sglang/srt/models/deepseek_v2.py`,
`python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py`, and
`python/sglang/srt/layers/communicator_mhc.py`. These projects use Apache-2.0;
see the canonical root THIRD_PARTY_NOTICES.md and packaged identical notice.
The source implementations are not copied into the model.
