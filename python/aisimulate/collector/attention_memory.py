# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan dense attention against the collecting device, before cases are queued."""


def filter_attention_cases(cases, estimate_bytes, cuda, op):
    """Keep CPU-only planning complete; a CUDA plan uses 90% of total memory.

    The backend estimate includes its visible tensors and a workspace reserve.
    Admission is not a guarantee against native workspace growth or fragmentation;
    queued cases must still execute or raise, never silently skip in the worker.
    """
    if not cuda.is_available():
        return cases
    total = cuda.get_device_properties(cuda.current_device()).total_memory
    kept = [case for case in cases if estimate_bytes(case) <= int(total * 0.9)]
    if len(kept) != len(cases):
        print(
            f"{op}: dropped {len(cases) - len(kept)}/{len(cases)} cases (memory budget, device={total / 2**30:.1f}GiB)"
        )
    return kept
