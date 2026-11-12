// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};

fn record(files: &mut BTreeMap<String, String>, path: &Path, name: String) {
    println!("cargo:rerun-if-changed={}", path.display());
    let bytes = fs::read(path).expect("read native build input");
    files.insert(name, format!("{:x}", Sha256::digest(bytes)));
}

fn record_sources(files: &mut BTreeMap<String, String>, directory: &Path, crate_root: &Path) {
    // Watching directories also invalidates Cargo's cached build-script output
    // when a source file is added, removed, or renamed.
    println!("cargo:rerun-if-changed={}", directory.display());
    for entry in fs::read_dir(directory).expect("read native source directory") {
        let path = entry.expect("read native source entry").path();
        if path.is_dir() {
            record_sources(files, &path, crate_root);
        } else if path.is_file() {
            let relative = path.strip_prefix(crate_root).expect("crate-local source");
            record(
                files,
                &path,
                format!(
                    "crates/core/{}",
                    relative.to_string_lossy().replace('\\', "/")
                ),
            );
        }
    }
}

fn main() {
    let crate_root = env::var_os("CARGO_MANIFEST_DIR").expect("Cargo manifest directory");
    let crate_root = Path::new(&crate_root);
    let mut files = BTreeMap::new();
    record_sources(&mut files, &crate_root.join("src"), crate_root);
    for name in ["build.rs", "Cargo.toml"] {
        record(
            &mut files,
            &crate_root.join(name),
            format!("crates/core/{name}"),
        );
    }
    // The published crate can also be built outside this repository workspace.
    // Such a build records its available manifests; workspace qualification
    // requires an exact match with the current checkout's input inventory.
    for name in ["Cargo.toml", "Cargo.lock"] {
        let path = crate_root.join("../..").join(name);
        if path.is_file() {
            record(&mut files, &path, name.to_owned());
        }
    }
    let output = env::var_os("OUT_DIR").expect("Cargo output directory");
    fs::write(
        Path::new(&output).join("build-source-files.json"),
        serde_json::to_vec(&files).expect("serialize native build inputs"),
    )
    .expect("write native build identity");
}
