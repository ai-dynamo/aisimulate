// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_core::{AicError, ForwardPassSpeculationConfig};
use serde_json::{Value, json};

fn config(value: Value) -> ForwardPassSpeculationConfig {
    serde_json::from_value(value).unwrap()
}

// Keep this as one test in a separate executable: other Rust tests may
// initialize Python, masking the cold-interpreter failure for Rust callers.
#[test]
fn replay_depth_respects_native_and_python_feature_boundaries() {
    #[cfg(feature = "python")]
    assert_eq!(unsafe { pyo3::ffi::Py_IsInitialized() }, 0);

    assert_eq!(config(json!({"kind": "none"})).replay_depth().unwrap(), 0);
    for depth in [1, 5] {
        assert_eq!(
            config(json!({"kind": "ngram", "params": {"num_speculative_tokens": depth}}))
                .replay_depth()
                .unwrap(),
            depth
        );
    }
    for invalid in [
        json!({"kind": ""}),
        json!({"kind": " "}),
        json!({"kind": "none", "params": {"depth": 1}}),
        json!({"kind": "none", "draft_model_path": "draft"}),
        json!({"kind": "none", "draft_config": {}}),
        json!({"kind": "ngram"}),
        json!({"kind": "ngram", "params": {"num_speculative_tokens": 0}}),
        json!({"kind": "ngram", "params": {"num_speculative_tokens": 6}}),
        json!({"kind": "ngram", "params": {"num_speculative_tokens": 1.5}}),
        json!({"kind": "ngram", "params": {"num_speculative_tokens": true}}),
        json!({"kind": "ngram", "params": {"num_speculative_tokens": 2, "unknown": 1}}),
        json!({"kind": "ngram", "params": {"num_speculative_tokens": 2}, "draft_model_path": "draft"}),
        json!({"kind": "ngram", "params": {"num_speculative_tokens": 2}, "draft_config": {}}),
    ] {
        assert!(matches!(
            config(invalid).replay_depth(),
            Err(AicError::InvalidEngineConfig(_))
        ));
    }

    #[cfg(feature = "python")]
    assert_eq!(unsafe { pyo3::ffi::Py_IsInitialized() }, 0);

    #[cfg(not(feature = "embed-python"))]
    {
        let error = config(json!({"kind": "mtp", "params": {"depth": 2}}))
            .replay_depth()
            .unwrap_err();
        assert!(matches!(error, AicError::InvalidEngineConfig(_)));
        #[cfg(not(feature = "python"))]
        assert!(error.to_string().contains("requires the python feature"));
        #[cfg(feature = "python")]
        {
            assert!(
                error
                    .to_string()
                    .contains("requires an initialized Python interpreter")
            );
            assert_eq!(unsafe { pyo3::ffi::Py_IsInitialized() }, 0);
        }
    }

    #[cfg(feature = "embed-python")]
    {
        // An unknown scheme always fails validation, even if the SDK is not
        // installed. The important contract here is that embed-python still
        // initializes its interpreter instead of taking the cold-host guard.
        let error = config(json!({"kind": "not_a_registered_scheme"}))
            .replay_depth()
            .unwrap_err();
        assert!(matches!(error, AicError::InvalidEngineConfig(_)));
        assert!(error.to_string().contains("speculation:"));
        assert_eq!(unsafe { pyo3::ffi::Py_IsInitialized() }, 1);
    }
}
