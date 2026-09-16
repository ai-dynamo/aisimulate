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
