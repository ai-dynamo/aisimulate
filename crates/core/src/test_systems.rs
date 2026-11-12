// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Where unit tests find the packaged systems directory and the op-latency tables.
//!
//! Operation tables are no longer in the repository. `AICONFIGURATOR_SYSTEMS_PATH` names a composed
//! root (specs plus the pinned op-latency data; build one with
//! `python -c "from aisim_core.data.default_systems import default_systems_path as p; print(p())"`).
//! System spec YAML files always come from the packaged in-tree directory.

use std::path::PathBuf;

fn in_tree() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/aisim/src/aisim_core/systems")
}

/// `relative` is `""` (the systems root), `data/...` (operation tables) or a top-level spec file.
pub(crate) fn path(relative: &str) -> PathBuf {
    let is_data = relative.is_empty() || relative == "data" || relative.starts_with("data/");
    let root = match std::env::var_os("AICONFIGURATOR_SYSTEMS_PATH") {
        Some(root) if is_data => PathBuf::from(root),
        _ => in_tree(),
    };
    if relative.is_empty() {
        root
    } else {
        root.join(relative)
    }
}
