# GLM-5.3-NVFP4 configuration provenance

`nvidia--GLM-5.3-NVFP4_config.json` and `nvidia--GLM-5.3-NVFP4_hf_quant_config.json`
are unmodified copies of `config.json` and `hf_quant_config.json` at
https://huggingface.co/nvidia/GLM-5.3-NVFP4/tree/e3b8f9f2fad29727dfa95e766c5cac18ea1414bf

| Local file | Upstream file | sha256 |
| --- | --- | --- |
| `nvidia--GLM-5.3-NVFP4_config.json` | https://huggingface.co/nvidia/GLM-5.3-NVFP4/blob/e3b8f9f2fad29727dfa95e766c5cac18ea1414bf/config.json | `b0b6b9ab922a82e3688bc39a4117bc443e106fcf88969ef48b3627778f4bdf7b` |
| `nvidia--GLM-5.3-NVFP4_hf_quant_config.json` | https://huggingface.co/nvidia/GLM-5.3-NVFP4/blob/e3b8f9f2fad29727dfa95e766c5cac18ea1414bf/hf_quant_config.json | `13799d4fed353c20dde2264fbdd7f68a780b514782f8d80bba96ac5e44aa9bba` |

The files describe an NVIDIA ModelOpt MIXED_PRECISION export of
zai-org/GLM-5.3: FP8 dense MLPs in layers 0-2, NVFP4 routed experts in layers
3-77, the MTP layer excluded, and FP8 KV cache. Modules absent from
`quantized_layers` (attention including the indexer, shared experts) are
modeled BF16, as vLLM and SGLang load them.

Copyright owners: NVIDIA CORPORATION & AFFILIATES (the quantized export and
its quantization metadata) and Z.AI (the base GLM-5.3 configuration).

The upstream model card (`README.md` at the same revision) states
"License type: NVIDIA Proprietary" and "Use of this model is governed by
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/)". Its metadata declares `license: other`
and `license_name: nvidia-open-model-license`. The card also states: "The
original GLM-5.3 model are licensed under the GLM-5.3 License, included in
this repository. The NVIDIA Open Model Agreement does not modify or supersede
the GLM-5.3 License or its conditions." Both texts are reproduced below.

## GLM-5.3 License

Reproduced verbatim from `LICENSE` at
https://huggingface.co/nvidia/GLM-5.3-NVFP4/blob/e3b8f9f2fad29727dfa95e766c5cac18ea1414bf/LICENSE

```text
GLM-5.3 License

Copyright (c) 2026 Z.AI

Permission is hereby granted, free of charge, to any person or entity (the "Licensee") obtaining a copy of this software — including the model weights, parameters, configuration files, inference and training code, and associated documentation (collectively, the "Software") — to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software; to run, deploy, fine-tune, or otherwise modify the Software and create derivative works from it; and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. The Licensee's use of the Software must comply with applicable laws and regulations.

2. "Model as a Service" means giving a third party access to language model inference or fine-tuning (e.g., via API) in a manner that allows such third party to exercise meaningful control over the inputs, parameters, or training data. This does not include (a) end-user products with model capabilities solely embedded within specific features or harnesses, or (b) mere relaying of requests to models hosted by others.
If the Licensee or any of its affiliates operates a Model as a Service business, and the aggregate revenue of the Licensee and its affiliates exceeds 10 billion US dollars (or the equivalent in other currencies) in total over any consecutive 12 months, the Licensee must pass Z.AI's security review before using the Software or its derivative works for any commercial purpose. The scope and method of the security review shall be reasonably determined by Z.AI.

3. THE SOFTWARE AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL Z.AI OR ITS AFFILIATES OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For any questions regarding this license, please contact glmlicense@z.ai.

-----

版权所有 (c) 2026 Z.AI

特此免费授予任何获得本软件副本的个人或实体（"被许可方"）——包括模型权重、参数、配置文件、推理和训练代码及相关文档（统称"软件"）——不受限制地处理本软件的权利，包括但不限于：使用、复制、修改、合并、发布、分发、再许可和/或销售软件副本；运行、部署、微调或以其他方式修改软件并创建衍生作品；以及允许获得软件的其他人行使上述权利，但须遵守以下条件：

1. 上述版权声明和本许可声明应包含在软件的所有副本或实质性部分中。被许可方对软件的使用必须符合适用法律法规。

2. "模型即服务"指以允许第三方对输入、参数或训练数据行使实质性控制的方式，向第三方提供语言模型推理或微调服务（如通过API）。不包括：(a) 模型能力仅嵌入特定功能或框架中的终端用户产品，或(b) 单纯转发请求至他人托管的模型。
若被许可方或其关联方运营"模型即服务"业务，且被许可方及关联方在任意连续12个月内累计总收入超过100亿美元（或等值其他货币），则被许可方在使用软件或其衍生作品进行任何商业用途之前，须通过Z.AI的安全审查。安全审查的范围和方式由Z.AI合理确定。

3. 软件及其任何输出和结果均按"现状"提供，不附带任何形式的保证，无论是明示还是暗示，包括但不限于适销性、特定用途适用性和不侵权的保证。在任何情况下，Z.AI或其关联方或版权持有人均不对任何索赔、损害或其他责任承担责任，无论该责任是基于合同、侵权或其他方式，因软件或使用软件而产生或与之相关。

如对本许可有任何疑问，请联系 glmlicense@z.ai。
```

## NVIDIA Open Model Agreement

Reproduced from https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
(retrieved 2026-10-27). The page shows "Release Date: April 2, 2026" and
versions the terms "(v. March 9, 2026)". Site navigation is omitted. Section
numbers and list markers are the ones the page renders through its list
styling; the wording is unchanged.

```text
NVIDIA Open Model Agreement

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

Release Date: April 2, 2026

NVIDIA Works released under this License are intended to be used permissively and enable the further development of AI technologies. Subject to the terms of this License, NVIDIA confirms that:

- Works are commercially usable.

- You are free to create and distribute Derivative Works.

- NVIDIA does not claim ownership to any outputs generated using the Works or Derivative Works.

By using, reproducing, modifying, distributing, performing or displaying any portion or element of the Works or Derivative Works, or otherwise accepting the terms of this License, you agree to be bound by this License.

1. Definitions.

   “License” shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 10 of this document.

   “Legal Entity” shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, “control” means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.

   “You” (or “Your”) shall mean an individual or Legal Entity exercising permissions granted by this License.

   “Work” shall mean the work of authorship, including machine learning model, software, checkpoints, learnt weights, algorithms, parameters, configuration files and documentation, made available under the License, as indicated by a copyright notice that is included in or attached to the work (an example is provided in the Appendix below).

   “Derivative Works” shall mean any work, whether in source or object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.

2. Grant of License. Subject to the terms and conditions of this License, NVIDIA hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in source or object form.

   If You institute patent or copyright litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or an output from the Work constitutes direct or contributory patent or copyright infringement, then any licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.

3. Redistribution. You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in source or object form, provided that You meet the following conditions:

      a. You must give any other recipients of the Work a copy of this License; and

      b. You must retain, in the source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works; and

      c. If the Work includes a “NOTICE” text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the following attribution notice within a “Notice” text file with such copies and the following statement: “Licensed by NVIDIA Corporation under the NVIDIA Open Model Agreement.”

   You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Derivative Works as a whole, provided Your use, reproduction, and distribution of the Work otherwise complies with the conditions stated in this License.

4. Trademarks. This License does not grant permission to use the trade names, trademarks, service marks, or product names of NVIDIA, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.

5. Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, NVIDIA provides the Work on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

6. Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall NVIDIA be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work, a Derivative Work or an output from the Work or Derivative Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if NVIDIA has been advised of the possibility of such damages.

7. Accepting Warranty or Additional Liability. While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of NVIDIA, and only if You agree to indemnify, defend, and hold NVIDIA harmless for any liability incurred by, or claims asserted against, NVIDIA by reason of your accepting any such warranty or additional liability. You will indemnify and hold harmless NVIDIA from and against any claim by any third party arising out of or related to your use or distribution of the Works, Derivative Works thereof, or output from the Works or Derivative Works.

8. Feedback. NVIDIA appreciates your feedback, and You agree that NVIDIA may use it without restriction or compensation to You.

9. Governing Law. This Agreement will be governed in all respects by the laws of the United States and the laws of the State of Delaware, without regard to conflict of laws principles or the United Nations Convention on Contracts for the International Sale of Goods. The state and federal courts residing in Santa Clara County, California will have exclusive jurisdiction over any dispute or claim arising out of or related to this Agreement, and the parties irrevocably consent to personal jurisdiction and venue in those courts; except that, either party may apply for injunctive remedies or an equivalent type of urgent legal relief in any jurisdiction.

10. Trade and Compliance. You agree to comply with all applicable export, import, trade and economic sanctions laws and regulations, as amended, including without limitation U.S. Export Administration Regulations and Office of Foreign Assets Control regulations. These laws include restrictions on destinations, end-users and end-use.

(v. March 9, 2026)

End of terms and conditions
```
