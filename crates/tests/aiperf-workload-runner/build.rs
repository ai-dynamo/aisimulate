// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bind provenance to the source actually used by this standalone build.

use std::{path::{Path,PathBuf},process::Command};
fn git(path:&Path,args:&[&str]) -> Option<String> {
    let output = Command::new("git").arg("-C").arg(path).args(args).output().ok()?;
    output.status.success().then(||String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
fn revision(path:&Path) -> String {
    let Some(revision) = git(path,&["rev-parse","HEAD"]) else { return "unavailable".into(); };
    for name in ["HEAD","index","packed-refs"] {
        if let Some(file) = git(path,&["rev-parse","--git-path",name]) {
            let file = PathBuf::from(file);
            let file = if file.is_absolute() { file } else {path.join(file)};
            println!("cargo:rerun-if-changed={}",file.display());
        }
    }
    if let Some(reference) = git(path,&["symbolic-ref","-q","HEAD"]) {
        if let Some(file) = git(path,&["rev-parse","--git-path",&reference]) {
            let file = PathBuf::from(file);
            let file = if file.is_absolute() {file} else {path.join(file)};
            println!("cargo:rerun-if-changed={}",file.display());
        }
    }
    if let Some(root) = git(path,&["rev-parse","--show-toplevel"]) {
        // Watch source files as well as refs: source edits can change the
        // provenance even when HEAD and the staging index remain unchanged.
        if let Some(files) = git(path,&["ls-files"]) {
            for file in files.lines() { println!("cargo:rerun-if-changed={}",Path::new(&root).join(file).display()); }
        }
    }
    match git(path,&["status","--porcelain","--untracked-files=normal"]) {
        Some(status) if status.is_empty() => revision,
        Some(_) => format!("{revision}+dirty"),
        None => format!("{revision}+worktree-state-unavailable"),
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
