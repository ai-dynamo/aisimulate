# NVIDIA Nemotron 3 Super and Ultra NVFP4 configuration provenance

The following files are unmodified copies of `config.json` and
`hf_quant_config.json` at the immutable Hugging Face revisions shown.

| Local file | Upstream file | sha256 |
| --- | --- | --- |
| `nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4_config.json` | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/blob/ff433f5493e25d631c9f12b5d55c674229923d02/config.json | `14a1ad3ab6ec49e09e9c30f8686c8e3ca5b5338ee0eb0c79dff4f285fc29778d` |
| `nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4_hf_quant_config.json` | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/blob/ff433f5493e25d631c9f12b5d55c674229923d02/hf_quant_config.json | `dbabb3970ad65a6cca51488db15d66d976c2486ff5448c4aee906389a02df316` |
| `nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4_config.json` | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/blob/02462641f13d3af838b904f48195b9bb8a1e4ebc/config.json | `0c939f324c8910f5ebdafbe2a56d7e4e074c50042a3b4f26326bf71a3fe33929` |
| `nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4_hf_quant_config.json` | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/blob/02462641f13d3af838b904f48195b9bb8a1e4ebc/hf_quant_config.json | `68f0c66d927e5e9115ad475a7c1dbfe40f7e7e72d9ab9c3b6a33d8c52da2c596` |

The files describe NVIDIA ModelOpt MIXED_PRECISION exports. The routed experts
are NVFP4 with group size 16. The listed Mamba projections and shared experts
are FP8. The KV cache is FP8. Modules absent from `quantized_layers`, or listed
in `exclude_modules`, stay BF16.

Copyright owner: NVIDIA CORPORATION & AFFILIATES. The copied files are
unmodified. Neither upstream repository ships a NOTICE file at these revisions.

Licenses, as declared by the upstream model cards at these revisions:

- Super: NVIDIA Nemotron Open Model License (card metadata
  `license_name: nvidia-nemotron-open-model-license`),
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
- Ultra: OpenMDW License Agreement, version 1.1 (card metadata
  `license_name: openmdw-1.1`), https://openmdw.ai/license/1-1/

Both texts are reproduced below.

## NVIDIA Nemotron Open Model License

Reproduced from
https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
(retrieved 2026-10-28, terms "v. December 15, 2025"). Site navigation is
omitted. The wording is unchanged.

```text
NVIDIA Nemotron Open Model License

Last Modified: December 15, 2025

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

NVIDIA Works released under this License are intended to be used permissively and enable the further development of AI technologies. Subject to the terms of this License, NVIDIA confirms that:

Works are commercially usable.

You are free to create and distribute Derivative Works.

NVIDIA does not claim ownership to any outputs generated using the Works or Derivative Works.

By using, reproducing, modifying, distributing, performing or displaying any portion or element of the Works or Derivative Works, or otherwise accepting the terms of this License, you agree to be bound by this License.

1. Definitions.

“License” shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 10 of this document.

“Legal Entity” shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.

“You” (or “Your”) shall mean an individual or Legal Entity exercising permissions granted by this License.

“Work” shall mean the work of authorship, including machine learning model, software, checkpoints, learnt weights, algorithms, parameters, configuration files and documentation, made available under the License, as indicated by a copyright notice that is included in or attached to the work (an example is provided in the Appendix below).

“Derivative Works” shall mean any work, whether in source or object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.

2. Grant of License. Subject to the terms and conditions of this License, NVIDIA hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in source or object form.

If You institute patent or copyright litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or an output from the Work constitutes direct or contributory patent or copyright infringement, then any licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.

3. Redistribution. You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in source or object form, provided that You meet the following conditions:

a. You must give any other recipients of the Work a copy of this License; and

b. You must retain, in the source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works; and

c. If the Work includes a "NOTICE" text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the following attribution notice within a “Notice” text file with such copies and the following statement: “Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License.”

You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Derivative Works as a whole, provided Your use, reproduction, and distribution of the Work otherwise complies with the conditions stated in this License.

4. Trademarks. This License does not grant permission to use the trade names, trademarks, service marks, or product names of NVIDIA, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.

5. Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, NVIDIA provides the Work on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

6. Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall NVIDIA be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work, a Derivative Work or an output from the Work or Derivative Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if NVIDIA has been advised of the possibility of such damages.

7. Accepting Warranty or Additional Liability. While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of NVIDIA, and only if You agree to indemnify, defend, and hold NVIDIA harmless for any liability incurred by, or claims asserted against, NVIDIA by reason of your accepting any such warranty or additional liability. You will indemnify and hold harmless NVIDIA from and against any claim by any third party arising out of or related to your use or distribution of the Works, Derivative Works thereof, or output from the Works or Derivative Works.

8. Feedback. NVIDIA appreciates your feedback, and You agree that NVIDIA may use it without restriction or compensation to You.

9. Governing Law. This Agreement will be governed in all respects by the laws of the United States and the laws of the State of Delaware, without regard to conflict of laws principles or the United Nations Convention on Contracts for the International Sale of Goods. The state and federal courts residing in Santa Clara County, California will have exclusive jurisdiction over any dispute or claim arising out of or related to this Agreement, and the parties irrevocably consent to personal jurisdiction and venue in those courts; except that, either party may apply for injunctive remedies or an equivalent type of urgent legal relief in any jurisdiction.

10. Trade and Compliance. You agree to comply with all applicable export, import, trade and economic sanctions laws and regulations, as amended, including without limitation U.S. Export Administration Regulations and Office of Foreign Assets Control regulations. These laws include restrictions on destinations, end-users and end-use.

(v. December 15, 2025)

END OF TERMS AND CONDITIONS
```

## OpenMDW License Agreement, version 1.1

Reproduced verbatim from
https://raw.githubusercontent.com/OpenMDW/OpenMDW/refs/heads/main/1.1/LICENSE.OpenMDW-1.1
(retrieved 2026-10-28).

```text
OpenMDW License Agreement, version 1.1 (OpenMDW-1.1)

By exercising rights granted to you under this agreement, you accept and agree
to its terms.

As used in this agreement, "Model Materials" means the materials provided to
you under this agreement, consisting of: (1) one or more machine learning
models (including architecture and parameters); and (2) all related artifacts
(including associated data, documentation and software) that are provided to
you hereunder.

Subject to your compliance with this agreement, permission is hereby granted,
free of charge, to deal in the Model Materials without restriction, including
under all copyright, patent, database, and trade secret rights included or
embodied therein.

If you distribute any portion of the Model Materials, you shall retain in your
distribution (1) a copy of this agreement, and (2) all copyright notices and
other notices of origin included in the Model Materials that are applicable to
your distribution.

If you file, maintain, or voluntarily participate in a lawsuit against any
person or entity asserting that the Model Materials directly or indirectly
infringe any patent or copyright, then all rights and grants made to you
hereunder are terminated, unless that lawsuit was in response to a
corresponding lawsuit first brought against you.

This agreement does not impose any restrictions or obligations with respect to
any use, modification, or sharing of any outputs generated by using the Model
Materials.

THE MODEL MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE, NONINFRINGEMENT, ACCURACY, OR THE
ABSENCE OF LATENT OR OTHER DEFECTS OR ERRORS, WHETHER OR NOT DISCOVERABLE, ALL
TO THE GREATEST EXTENT PERMISSIBLE UNDER APPLICABLE LAW.

YOU ARE SOLELY RESPONSIBLE FOR (1) CLEARING RIGHTS OF OTHER PERSONS THAT MAY
APPLY TO THE MODEL MATERIALS OR ANY USE THEREOF, INCLUDING WITHOUT LIMITATION
ANY PERSON'S COPYRIGHTS OR OTHER RIGHTS INCLUDED OR EMBODIED IN THE MODEL
MATERIALS; (2) OBTAINING ANY NECESSARY CONSENTS, PERMISSIONS OR OTHER RIGHTS
REQUIRED FOR ANY USE OF THE MODEL MATERIALS; OR (3) PERFORMING ANY DUE
DILIGENCE OR UNDERTAKING ANY OTHER INVESTIGATIONS INTO THE MODEL MATERIALS OR
ANYTHING INCORPORATED OR EMBODIED THEREIN.

IN NO EVENT SHALL THE PROVIDERS OF THE MODEL MATERIALS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE MODEL MATERIALS, THE
USE THEREOF OR OTHER DEALINGS THEREIN.
```
