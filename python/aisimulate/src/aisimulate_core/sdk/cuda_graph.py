# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISimulate alias for the canonical CUDA graph reservation API."""

from aiconfigurator_core.sdk.cuda_graph import (
    CudaGraphProfileDatabaseError,
    CudaGraphReservationEstimate,
    CudaGraphReservationRequest,
    estimate_cuda_graph_reservation,
)

__all__ = [
    "CudaGraphProfileDatabaseError",
    "CudaGraphReservationEstimate",
    "CudaGraphReservationRequest",
    "estimate_cuda_graph_reservation",
]
