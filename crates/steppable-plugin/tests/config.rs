// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_steppable_plugin::{BackendConfig, BackendTopology};

#[test]
fn backend_config_defaults_to_a_single_round_robin_worker() {
    let config: BackendConfig = serde_json::from_str("{}").unwrap();

    assert_eq!(config.topology, BackendTopology::Single);
    assert_eq!(config.workers, 1);
    assert_eq!(config.prefill_workers, 1);
    assert_eq!(config.decode_workers, 1);
}

#[test]
fn aggregated_config_accepts_an_explicit_dynamic_placement_locator() {
    let config: BackendConfig = serde_json::from_str(
        r#"{
            "topology": "aggregated",
            "dynamic_placement": {
                "library_path": "/opt/providers/libplacement.so",
                "selector_seed": [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "options_namespace": [100, 121, 110, 97, 109, 111],
                "provider_options": [1, 2, 3],
                "limits": {
                    "max_mutations": 1,
                    "max_admission_results": 1,
                    "max_released": 1,
                    "max_diagnostic_bytes": 32
                }
            }
        }"#,
    )
    .expect("explicit dynamic-placement configuration is accepted");

    let placement = config
        .dynamic_placement
        .expect("configuration retains the explicit locator");
    assert_eq!(
        placement.library_path.to_string_lossy(),
        "/opt/providers/libplacement.so"
    );
    assert_eq!(placement.selector_seed[0], 7);
    assert_eq!(placement.options_namespace, b"dynamo");
    assert_eq!(placement.provider_options, [1, 2, 3]);
    assert_eq!(placement.limits.max_diagnostic_bytes, 32);
}
