// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AFD (Attention-FFN Disaggregation) timing provider support. `pipeline` is
//! the verbatim `evaluate_afd_phase` port; `timing` is the `"afd"` replay
//! timing provider injected into the monolithic backend core.

pub mod pipeline;
pub mod timing;
