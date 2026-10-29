# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4, text path only.

The packaged configuration files are unmodified copies of the upstream files at
revision da5b2650a835292c678994a1fa3fb653250fb508
(model_configs/NEMOTRON35_SUPER_VL_NVFP4_PROVENANCE.md). Vision inputs are out of
scope: AISimulate models the text path of this vision-language model.
"""

import hashlib
from pathlib import Path

import pytest

import aisimulate_core

pytestmark = pytest.mark.unit

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
_CONFIGS = Path(aisimulate_core.__file__).parent / "model_configs"
# nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@da5b2650a835292c678994a1fa3fb653250fb508, byte for byte.
_PACKAGED_SHA256 = {
    "config": "35189a8038d1b1f934e58cba50eb7e97b42a4da569424e35b9fd1048058bc7ea",
    "hf_quant_config": "b797654d89e25e9bd13617cb1d216ff5fad78cb6a8b99ba0cd0fbcf20050d75a",
}


@pytest.mark.parametrize("suffix", list(_PACKAGED_SHA256))
def test_bundle_is_the_pinned_upstream_bytes(suffix):
    data = (_CONFIGS / f"nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4_{suffix}.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == _PACKAGED_SHA256[suffix]
