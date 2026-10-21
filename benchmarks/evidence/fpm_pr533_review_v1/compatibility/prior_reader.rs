// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compatibility harness only; production sources are unchanged at recorded main.

use std::io::{self, Read};

use aisimulate_core::{ForwardPassPerfModel, ForwardPassPerfModelConfig};
use serde_json::{Value, json};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let input: Vec<Value> = serde_json::from_str(&input).unwrap();
    let cases: Vec<Value> = input
        .into_iter()
        .map(|case| {
            let mut result = json!({"name": case["name"]});
            match serde_json::from_value::<ForwardPassPerfModelConfig>(case["config"].clone()) {
                Ok(config) => {
                    result["accepted"] = true.into();
                    result["normalized"] = serde_json::to_value(&config).unwrap();
                    if case["construct"].as_bool() == Some(true) {
                        match ForwardPassPerfModel::best_available(config) {
                            Ok(model) => result["diagnostics"] = serde_json::to_value(model.diagnostics()).unwrap(),
                            Err(error) => {
                                result["accepted"] = false.into();
                                result["error"] = error.to_string().into();
                            }
                        }
                    }
                }
                Err(error) => {
                    result["accepted"] = false.into();
                    result["error"] = error.to_string().into();
                }
            }
            result
        })
        .collect();
    println!("{}", json!({"cases": cases}));
}
