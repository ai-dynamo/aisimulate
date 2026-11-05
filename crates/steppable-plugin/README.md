<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# aisimulate-steppable-plugin

AISimulate's replay engine packaged as a shared library that AIPerf loads at
run time through its Steppable plugin ABI. AIPerf needs only the built library
(or its manifest); it does not build AISimulate or need this source tree.

## Artifact

```bash
cargo build --release -p aisimulate-steppable-plugin
```

| Platform | Library | Manifest |
| --- | --- | --- |
| Linux | `target/release/libaisimulate_steppable_plugin.so` | `aisimulate-steppable-plugin.plugin.json` |
| macOS | `target/release/libaisimulate_steppable_plugin.dylib` | same name |
| Windows | `target/release/aisimulate_steppable_plugin.dll` | same name |

The library exports one symbol, `aiperf_steppable_plugin_v1`, which returns a
static `PluginDescriptorV1` with provider ID `aisimulate`, ABI 1.0.

Write the manifest beside the library after building it:

```bash
cargo run --release -p aisimulate-steppable-plugin \
  --bin aisimulate-steppable-plugin-manifest -- \
  target/release/libaisimulate_steppable_plugin.so \
  --provenance "aisimulate $(git rev-parse HEAD)"
```

The tool loads the library, checks its descriptor with the ABI's own
validator, and writes the provider ID, ABI version and capability names that
the library reports, plus its SHA-256 and the build platform. It does not
derive anything from its own Cargo features, so the manifest is truthful for
whichever variant it is pointed at. Run it on the machine that built the
library: `platform` is the tool's own `os-arch`.

## Loading it from AIPerf

Set `dynamic_provider.library_path` in an `aisimulate` transport to the
library or to the `.plugin.json` manifest. A manifest is checked before the
library is opened (schema version, tier, ABI major, platform, digest, a
`library` path confined to the manifest's directory), and after opening AIPerf
refuses it if the reported provider ID, ABI minor or any asserted capability
disagrees with the loaded descriptor.

## Capability variants

AIPerf negotiates each optional route by descriptor capability bit.

| Build | Advertised capabilities |
| --- | --- |
| `--no-default-features` | `final_report_bundle`, `request_reuse_facts`, `timestamped_events` |
| `--no-default-features --features compact` | the above plus `compact_request` |
| default (`leases`) | the above plus `compact_buffer_leases` |
| `--features without-timestamped-events` | default minus `timestamped_events`: a negative control |

Every variant implements the full operation table; the features only change
what the descriptor advertises, which is what selects AIPerf's materialized,
copied-compact or leased-compact submission route.

## The ABI dependency

The C layout is AIPerf's contract and is defined once, in AIPerf's
`aiperf-steppable-abi` crate. This crate compiles against that crate: it does
not vendor, copy or regenerate the types, so the plugin cannot disagree with
the definitions AIPerf validates in `crates/engine/steppable-abi/tests/layout.rs`.

**The git pin is transitional.** `Cargo.toml` pins
`https://github.com/NVIDIA-dev/aiperf-rs.git` at an exact revision only because
the ABI crate is not yet published. When it is, the dependency becomes a
registry version requirement (`aiperf-steppable-abi = "0.x"`) with no other
change. Moving the pin is an ABI review, not a routine bump:
`tests/abi_conformance.rs` checks the exported descriptor with the pinned
validator, and `tests/abi_layout.rs` freezes the record layouts this plugin
exchanges, so a layout change in a newer ABI fails here first.
