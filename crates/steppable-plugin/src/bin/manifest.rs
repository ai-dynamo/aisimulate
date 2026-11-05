// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Writes the `*.plugin.json` manifest for a built Steppable plugin library.
//!
//! Every identity field is read from the library itself: it is loaded, its
//! descriptor is checked with the canonical ABI validator, and the provider ID,
//! ABI version and capability bits it reports become the manifest. A manifest
//! therefore cannot claim a capability that the artifact beside it lacks, no
//! matter which Cargo features this tool was built with.
//!
//! ```text
//! aisimulate-steppable-plugin-manifest <LIBRARY> [--output PATH] [--provenance TEXT]
//! ```

use std::ffi::CStr;
use std::io::Read;
use std::path::{Path, PathBuf};

use aiperf_steppable_abi::{CAPABILITY_NAMES_V1, PluginEntryV1, validate_descriptor_v1};
use aisimulate_steppable_plugin::ENTRY_SYMBOL;
use anyhow::{Context, Result, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Manifest file name written beside the library unless `--output` is given.
const MANIFEST_FILE: &str = "aisimulate-steppable-plugin.plugin.json";

/// Manifest document, schema version 1, in AIPerf's field order.
#[derive(Debug, Serialize)]
struct ManifestV1 {
    schema_version: u32,
    tier: &'static str,
    plugin_id: String,
    plugin_version: &'static str,
    abi: Abi,
    capabilities: Vec<&'static str>,
    platform: String,
    library: String,
    sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    provenance: Option<String>,
}

#[derive(Debug, Serialize)]
struct Abi {
    major: u32,
    minor: u32,
}

/// What the loaded descriptor reports.
struct Reported {
    provider_id: String,
    abi_major: u32,
    abi_minor: u32,
    capabilities: u64,
}

struct Arguments {
    library: PathBuf,
    output: Option<PathBuf>,
    provenance: Option<String>,
}

fn main() -> Result<()> {
    let arguments = parse(std::env::args().skip(1))?;
    let library = arguments
        .library
        .canonicalize()
        .with_context(|| format!("resolving {}", arguments.library.display()))?;
    let directory = library
        .parent()
        .context("library path has no parent directory")?;
    let output = arguments
        .output
        .unwrap_or_else(|| directory.join(MANIFEST_FILE));
    let output_directory = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .context("resolving manifest output directory")?;
    // AIPerf resolves `library` relative to the manifest and refuses anything
    // outside that directory, so the two must sit side by side.
    if output_directory != directory {
        bail!(
            "manifest {} must be written in the library's directory {}",
            output.display(),
            directory.display()
        );
    }
    if let Some(provenance) = arguments.provenance.as_deref()
        && (provenance.trim().is_empty() || provenance.trim() != provenance)
    {
        bail!("--provenance must be non-empty and carry no surrounding whitespace");
    }

    let reported = load(&library)?;
    let capabilities = capability_names(reported.capabilities)?;
    let manifest = ManifestV1 {
        schema_version: 1,
        tier: "steppable",
        plugin_id: reported.provider_id,
        plugin_version: env!("CARGO_PKG_VERSION"),
        abi: Abi {
            major: reported.abi_major,
            minor: reported.abi_minor,
        },
        capabilities,
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        library: library
            .file_name()
            .and_then(|name| name.to_str())
            .context("library file name is not UTF-8")?
            .to_owned(),
        sha256: digest(&library)?,
        provenance: arguments.provenance,
    };
    let mut bytes = serde_json::to_vec_pretty(&manifest)?;
    bytes.push(b'\n');
    std::fs::write(&output, bytes).with_context(|| format!("writing {}", output.display()))?;
    println!("{}", output.display());
    Ok(())
}

fn parse(mut arguments: impl Iterator<Item = String>) -> Result<Arguments> {
    const USAGE: &str =
        "usage: aisimulate-steppable-plugin-manifest <LIBRARY> [--output PATH] [--provenance TEXT]";
    let mut library = None;
    let mut output = None;
    let mut provenance = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--output" => output = Some(PathBuf::from(arguments.next().context(USAGE)?)),
            "--provenance" => provenance = Some(arguments.next().context(USAGE)?),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            flag if flag.starts_with('-') => bail!("unknown option {flag}\n{USAGE}"),
            _ if library.is_some() => bail!("more than one library given\n{USAGE}"),
            _ => library = Some(PathBuf::from(argument)),
        }
    }
    Ok(Arguments {
        library: library.context(USAGE)?,
        output,
        provenance,
    })
}

fn load(path: &Path) -> Result<Reported> {
    // Safety: loading the plugin is the purpose of this tool; its initializers
    // are this crate's own code.
    let library = unsafe { libloading::Library::new(path) }
        .with_context(|| format!("loading {}", path.display()))?;
    // Safety: the symbol name is NUL terminated and the ABI fixes its type.
    let entry = unsafe { library.get::<PluginEntryV1>(ENTRY_SYMBOL) }
        .with_context(|| format!("resolving the Steppable entry point in {}", path.display()))?;
    // Safety: the entry contract returns null or a static descriptor.
    let descriptor = unsafe { entry() };
    // Safety: the validator handles null and checks every required field.
    unsafe { validate_descriptor_v1(descriptor) }
        .map_err(|error| anyhow::anyhow!("invalid descriptor in {}: {error:?}", path.display()))?;
    // Safety: validated above; the descriptor and its NUL-terminated provider
    // ID are static for as long as `library` stays loaded.
    let reported = unsafe {
        Reported {
            provider_id: CStr::from_ptr((*descriptor).provider_id)
                .to_str()
                .context("provider ID is not UTF-8")?
                .to_owned(),
            abi_major: (*descriptor).abi_major,
            abi_minor: (*descriptor).abi_minor,
            capabilities: (*descriptor).capabilities,
        }
    };
    drop(library);
    Ok(reported)
}

/// Names every advertised bit through the ABI's canonical table.
///
/// A bit with no canonical name means the library was built against a newer
/// ABI than this tool knows, so writing a partial list would be untruthful.
fn capability_names(bits: u64) -> Result<Vec<&'static str>> {
    let mut remaining = bits;
    let mut names = Vec::new();
    for (name, bit) in CAPABILITY_NAMES_V1 {
        if bits & bit != 0 {
            names.push(*name);
            remaining &= !bit;
        }
    }
    if remaining != 0 {
        bail!("descriptor advertises capability bits {remaining:#x} with no canonical name");
    }
    Ok(names)
}

fn digest(path: &Path) -> Result<String> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("reading {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
