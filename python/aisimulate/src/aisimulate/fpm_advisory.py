# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FPM collection advisories shared without importing collector or estimator runtime."""


def prefill_graph_advisory(policy: str, *, enforce_eager: bool) -> str | None:
    if policy != "explicit" or enforce_eager:
        return None
    return (
        "Explicit CUDA-graph captures apply to prefill only; decode keeps runtime-selected captures. "
        "A shared memory profile requires matching effective graph settings in both phases. "
        "Compatibility is unresolved until observed: use prefill_cudagraph_policy=runtime, "
        "choose explicit captures verified to match decode, or select enforce_eager=true for both "
        "phases when that matches the serving target."
    )
