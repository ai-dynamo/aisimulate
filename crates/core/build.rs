// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sha2::{Digest, Sha256};
use std::{env, fs, path::Path};

fn sources(path: &Path, paths: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(path).expect("read core source directory") {
        let path = entry.expect("read core source entry").path();
        if path.is_dir() {
            sources(&path, paths);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            paths.push(path);
        }
    }
}

fn main() {
    let root = std::path::PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let mut paths = vec![root.join("Cargo.toml"), root.join("build.rs")];
    sources(&root.join("src"), &mut paths);
    paths.sort();
    let mut digest = Sha256::new();
    for path in paths {
        let relative = path.strip_prefix(&root).unwrap().to_str().unwrap();
        let data = fs::read(&path).expect("read core source");
        digest.update((relative.len() as u64).to_le_bytes());
        digest.update(relative.as_bytes());
        digest.update((data.len() as u64).to_le_bytes());
        digest.update(data);
    }
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rustc-env=AISIMULATE_CORE_SOURCE_SHA256={:x}",
        digest.finalize()
    );
}
