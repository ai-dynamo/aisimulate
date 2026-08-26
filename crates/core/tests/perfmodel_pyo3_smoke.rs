// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Cargo only discovers integration-test crates directly under `tests/`.
// Keep the imported AIC test in its mirror subtree and use this glue module
// solely to make it discoverable.
#[path = "perfmodel/pyo3_smoke.rs"]
mod pyo3_smoke;
