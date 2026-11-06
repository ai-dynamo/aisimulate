// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bind provenance to the source actually used by this standalone build.

use std::{path::{Path,PathBuf},process::Command};
fn revision(path:&Path) -> String {
    let output = Command::new("git").arg("-C").arg(path).args(["rev-parse","HEAD"]).output();
    match output {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout).trim().to_owned(),
        _ => "unavailable".into(),
    }
}
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    let package = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let manifest = std::fs::read_to_string(package.join("Cargo.toml")).unwrap();
    let dependency = manifest.lines().find(|line|line.starts_with("aiperf-config =")).and_then(|line|line.split('"').nth(1));
    let aiperf = dependency.map(|path|package.join(path));
    let ais = revision(&package);
    let upstream = aiperf.as_ref().map_or_else(||"unavailable".into(),|path|revision(path));
    println!("cargo:rustc-env=AISIM_SOURCE_REVISION={ais}");
    println!("cargo:rustc-env=AIPERF_SOURCE_REVISION={upstream}");
}
