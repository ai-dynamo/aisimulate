// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Execute the exact shared AIPerf semantic tests against AISim's owned binding.
//! Modules are path-attached to the selected AIPerf source, never copied or
//! weakened. This target has no AIPerf application/simulation dependency.

use aisim_aiperf_workload_runner::model as model_binding;

include!(concat!(env!("OUT_DIR"), "/agentx_semantics_modules.rs"));
