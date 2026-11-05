// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The exported descriptor, checked with AIPerf's own validator.
//!
//! These run once per feature variant. They prove the descriptor a host reads
//! is the one the canonical ABI accepts, that it advertises exactly the
//! capabilities its features promise, and that every advertised bit has a
//! canonical manifest name.

use std::ffi::CStr;

use aiperf_steppable_abi::{
    CAPABILITY_COMPACT_BUFFER_LEASES_V1, CAPABILITY_COMPACT_REQUEST_V1,
    CAPABILITY_FINAL_REPORT_BUNDLE_V1, CAPABILITY_NAMES_V1, CAPABILITY_REQUEST_REUSE_FACTS_V1,
    CAPABILITY_TIMESTAMPED_EVENTS_V1, PLUGIN_ABI_MAJOR_V1, PluginDescriptorV1, PluginVTableV1,
    validate_descriptor_v1,
};
use aisimulate_steppable_plugin::{ENTRY_SYMBOL, PROVIDER, aiperf_steppable_plugin_v1};

fn descriptor() -> &'static PluginDescriptorV1 {
    let descriptor = aiperf_steppable_plugin_v1();
    // Safety: the entry returns a pointer to a static descriptor.
    unsafe { validate_descriptor_v1(descriptor) }.expect("canonical validator accepts it");
    // Safety: validated non-null static.
    unsafe { &*descriptor }
}

#[test]
fn descriptor_passes_the_canonical_validator_at_full_size() {
    let descriptor = descriptor();
    assert_eq!(descriptor.abi_major, PLUGIN_ABI_MAJOR_V1);
    assert_eq!(
        descriptor.struct_size as usize,
        std::mem::size_of::<PluginDescriptorV1>()
    );
    // Safety: validated descriptor; its vtable is static.
    let vtable = unsafe { &*descriptor.vtable };
    assert_eq!(
        vtable.struct_size as usize,
        std::mem::size_of::<PluginVTableV1>(),
        "the plugin publishes every tail of the ABI it was built against"
    );
    // Safety: validated descriptor; the provider ID is a static C string.
    let provider = unsafe { CStr::from_ptr(descriptor.provider_id) };
    assert_eq!(provider.to_str().unwrap(), PROVIDER);
    assert_eq!(ENTRY_SYMBOL, b"aiperf_steppable_plugin_v1\0");
}

#[test]
fn capabilities_follow_the_feature_set_exactly() {
    let mut expected = CAPABILITY_FINAL_REPORT_BUNDLE_V1 | CAPABILITY_REQUEST_REUSE_FACTS_V1;
    if cfg!(feature = "compact") {
        expected |= CAPABILITY_COMPACT_REQUEST_V1;
    }
    if cfg!(feature = "leases") {
        expected |= CAPABILITY_COMPACT_BUFFER_LEASES_V1;
    }
    if !cfg!(feature = "without-timestamped-events") {
        expected |= CAPABILITY_TIMESTAMPED_EVENTS_V1;
    }
    assert_eq!(descriptor().capabilities, expected);
}

#[test]
fn every_advertised_capability_has_a_canonical_manifest_name() {
    let named = CAPABILITY_NAMES_V1
        .iter()
        .fold(0_u64, |bits, (_, bit)| bits | bit);
    assert_eq!(descriptor().capabilities & !named, 0);
}
