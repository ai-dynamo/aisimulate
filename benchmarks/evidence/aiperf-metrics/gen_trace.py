# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded Mooncake trace with variable input/output lengths and unique prompts."""

import json
import math
import random
import sys

n, block, seed, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rng = random.Random(seed)
next_hash = 0
with open(out, "w") as f:
    for i in range(n):
        isl = rng.randint(128, 1536)
        # Mixed short and long outputs separate request-weighted from token-weighted TPS.
        osl = rng.choice([8, 24, 64, 160, 400])
        k = math.ceil(isl / block)
        f.write(
            json.dumps({"input_length": isl, "output_length": osl, "hash_ids": list(range(next_hash, next_hash + k))})
            + "\n"
        )
        next_hash += k
