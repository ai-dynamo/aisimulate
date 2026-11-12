// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AFD phase accounting over the shared ordered-stage pipeline simulator.

use crate::common::error::AicError;
use crate::perfmodel::afd::model::{AfdPhaseMetrics, AfdPipelineLayout};

#[derive(Clone, Copy, Debug)]
pub struct AfdPipelineKnobs {
    pub pipeline_layout: AfdPipelineLayout,
    /// In-flight sequences across all A workers.
    pub total_batch_size: u32,
    pub total_gpus: u32,
    /// Multiplies the complete pass latency, after any KV averaging.
    pub latency_correction: f64,
}

impl Default for AfdPipelineKnobs {
    fn default() -> Self {
        Self {
            pipeline_layout: AfdPipelineLayout::default(),
            total_batch_size: 1,
            total_gpus: 1,
            latency_correction: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AfdPhaseEvaluation {
    pub step_latency_ms: f64,
    pub sequence_rate: f64,
    pub tokens_per_second: f64,
    pub balance_ratio: f64,
    pub pipeline_layout: AfdPipelineLayout,
    pub total_gpus: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfdEvalPhase {
    Prefill,
    Decode,
}

/// Consume already evaluated, possibly stride-averaged pass latencies.
/// Never run the simulator on the averaged stage times or scale transfers again.
pub(crate) fn evaluate_afd_metrics(
    knobs: &AfdPipelineKnobs,
    metrics: &AfdPhaseMetrics,
    phase: AfdEvalPhase,
    input_length: u32,
    output_length: u32,
) -> Result<AfdPhaseEvaluation, AicError> {
    if input_length == 0 || output_length == 0 {
        return Err(AicError::InvalidEngineConfig(
            "AFD input_length and output_length must be positive".into(),
        ));
    }
    if !knobs.latency_correction.is_finite() || knobs.latency_correction <= 0.0 {
        return Err(AicError::InvalidEngineConfig(
            "AFD latency_correction must be positive and finite".into(),
        ));
    }
    let step = metrics.billed_step_ms * knobs.latency_correction;
    if !step.is_finite() || step <= 0.0 {
        return Err(AicError::InvalidEngineConfig(
            "AFD step latency must be positive and finite".into(),
        ));
    }
    let rate = f64::from(knobs.total_batch_size) * 1000.0 / step;
    let (sequence_rate, tokens_per_second) = match phase {
        AfdEvalPhase::Decode => (rate / f64::from(output_length), rate),
        AfdEvalPhase::Prefill => (rate, rate * f64::from(input_length)),
    };
    let times = &metrics.layer_times;
    let compute_max = times.attention_ms.max(times.ffn_ms);
    Ok(AfdPhaseEvaluation {
        step_latency_ms: step,
        sequence_rate,
        tokens_per_second,
        balance_ratio: if compute_max == 0.0 {
            0.0
        } else {
            times.attention_ms.min(times.ffn_ms) / compute_max
        },
        pipeline_layout: knobs.pipeline_layout,
        total_gpus: knobs.total_gpus,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perfmodel::afd::model::AfdLayerTimes;

    #[test]
    fn balance_ratio_is_scale_independent_and_zero_without_compute() {
        // One stage is twice the other, regardless of the unit scale. A
        // transfer-only pass has no compute balance and reports zero.
        for (attention_ms, ffn_ms, expected) in [(1e-12, 2e-12, 0.5), (0., 0., 0.)] {
            let times = AfdLayerTimes {
                attention_ms,
                ffn_ms,
                a_to_f_ms: 1.,
                f_to_a_ms: 1.,
                num_layers: 1,
            };
            let metrics =
                AfdPhaseMetrics::from_layer_times(AfdPipelineLayout::FourStage, times, 1, 1.)
                    .unwrap();
            let result = evaluate_afd_metrics(
                &AfdPipelineKnobs::default(),
                &metrics,
                AfdEvalPhase::Decode,
                1,
                2,
            )
            .unwrap();
            assert_eq!(result.balance_ratio, expected);
        }
    }

    #[test]
    fn transfer_factor_and_final_correction_land_once() {
        // m=1: two layers each cost 1 + 2*2 + 1 + 2*2 = 10;
        // correction 1.5 makes the complete pass 30, for either layout.
        let times = AfdLayerTimes {
            attention_ms: 1.,
            a_to_f_ms: 2.,
            ffn_ms: 1.,
            f_to_a_ms: 2.,
            num_layers: 2,
        };
        for layout in [AfdPipelineLayout::FourStage, AfdPipelineLayout::TwoStage] {
            let knobs = AfdPipelineKnobs {
                pipeline_layout: layout,
                latency_correction: 1.5,
                total_batch_size: 6,
                ..Default::default()
            };
            let metrics = AfdPhaseMetrics::from_layer_times(layout, times, 1, 2.).unwrap();
            let decode =
                evaluate_afd_metrics(&knobs, &metrics, AfdEvalPhase::Decode, 100, 10).unwrap();
            assert_eq!(decode.step_latency_ms, 30.);
            assert_eq!(decode.tokens_per_second, 200.);
            assert_eq!(decode.sequence_rate, 20.);
            let prefill =
                evaluate_afd_metrics(&knobs, &metrics, AfdEvalPhase::Prefill, 100, 10).unwrap();
            assert_eq!(prefill.sequence_rate, 200.);
            assert_eq!(prefill.tokens_per_second, 20000.);
        }
    }

    #[test]
    fn averaged_metrics_are_consumed_without_resimulation() {
        // Two four-stage samples [4,1,1,1] and [1,1,4,1], m=4, L=2,
        // factor=2: each first completion is 9 then seven completions 4 apart.
        // Their mean billed pass is 37; averaged stages would instead cost 26.5.
        let times = AfdLayerTimes {
            attention_ms: 2.5,
            a_to_f_ms: 1.,
            ffn_ms: 2.5,
            f_to_a_ms: 1.,
            num_layers: 2,
        };
        let knobs = AfdPipelineKnobs::default();
        let metrics = AfdPhaseMetrics {
            layer_times: times,
            t_step_ms: 35.,
            billed_step_ms: 37.,
        };
        assert_eq!(
            evaluate_afd_metrics(&knobs, &metrics, AfdEvalPhase::Decode, 100, 10)
                .unwrap()
                .step_latency_ms,
            37.
        );
        assert_eq!(
            knobs
                .pipeline_layout
                .step_latency_ms(&times, 4, 2.)
                .unwrap(),
            26.5
        );
    }

    #[test]
    fn invalid_inputs_reject() {
        let times = AfdLayerTimes {
            attention_ms: 1.,
            a_to_f_ms: 1.,
            ffn_ms: 1.,
            f_to_a_ms: 1.,
            num_layers: 2,
        };
        let metrics =
            AfdPhaseMetrics::from_layer_times(AfdPipelineLayout::FourStage, times, 1, 1.).unwrap();
        assert!(
            evaluate_afd_metrics(&Default::default(), &metrics, AfdEvalPhase::Decode, 0, 1)
                .is_err()
        );
        for value in [0., -1., f64::NAN, f64::INFINITY] {
            assert!(
                evaluate_afd_metrics(
                    &AfdPipelineKnobs {
                        latency_correction: value,
                        ..Default::default()
                    },
                    &metrics,
                    AfdEvalPhase::Decode,
                    1,
                    1
                )
                .is_err()
            );
        }
    }
}
