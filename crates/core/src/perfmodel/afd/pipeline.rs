// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordered stages with one server per stage and `m` circulating microbatches.

use serde::{Deserialize, Serialize};

use super::model::AfdLayerTimes;
use crate::common::error::AicError;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AfdPipelineLayout {
    /// A, A→F, F and F→A can each work independently.
    #[default]
    FourStage,
    /// A→F immediately follows A on one resource; F→A follows F on another.
    TwoStage,
}

// Ready queues contain at most one task per circulating microbatch.
#[derive(Clone, Copy, Debug, Default)]
struct Task {
    ready: f64,
    duration: f64,
    layer: usize,
    micro: usize,
    stage: u8,
}
impl Task {
    fn advance(&mut self, layout: AfdPipelineLayout, layers: usize) -> bool {
        if self.stage == 5 {
            return false;
        }
        if self.stage == 4 {
            if self.layer + 1 == layers {
                self.stage = 5;
            } else {
                self.layer += 1;
                self.stage = 0;
            }
        } else {
            self.stage += if layout == AfdPipelineLayout::TwoStage {
                2
            } else {
                1
            };
        }
        true
    }
}
impl PartialEq for Task {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for Task {}
impl PartialOrd for Task {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Task {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.ready
            .total_cmp(&other.ready)
            .then_with(|| (self.stage != 4).cmp(&(other.stage != 4)))
            .then_with(|| self.layer.cmp(&other.layer))
            .then_with(|| self.micro.cmp(&other.micro))
    }
}
enum ReadyQueue {
    Inline([Option<Task>; 4]),
    Heap(std::collections::BinaryHeap<std::cmp::Reverse<Task>>),
}
impl ReadyQueue {
    fn new(small: bool) -> Self {
        if small {
            Self::Inline([None; 4])
        } else {
            Self::Heap(Default::default())
        }
    }
    fn push(&mut self, task: Task) {
        match self {
            Self::Inline(slots) => {
                *slots
                    .iter_mut()
                    .find(|s| s.is_none())
                    .expect("one task per microbatch") = Some(task)
            }
            Self::Heap(heap) => heap.push(std::cmp::Reverse(task)),
        }
    }
    fn pop(&mut self) -> Option<Task> {
        match self {
            Self::Inline(slots) => {
                let i = slots
                    .iter()
                    .enumerate()
                    .filter_map(|(i, t)| t.map(|t| (i, t)))
                    .min_by_key(|(_, t)| *t)
                    .map(|(i, _)| i)?;
                slots[i].take()
            }
            Self::Heap(heap) => heap.pop().map(|t| t.0),
        }
    }
}

impl AfdPipelineLayout {
    /// Heterogeneous source layer sequence. Empty stages are bypassed rather
    /// than queued on an unrelated resource (dense layers use A only).
    pub(crate) fn plan_latency_ms(
        self,
        templates: &[[f64; 5]],
        layers: &[usize],
        microbatches: u32,
        factor: f64,
        prefix: f64,
        suffix: f64,
    ) -> Result<f64, AicError> {
        if microbatches == 0
            || layers.is_empty()
            || !factor.is_finite()
            || factor <= 0.0
            || [prefix, suffix].iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid AFD layer pipeline geometry".into(),
            ));
        }
        self.schedule(
            layers,
            microbatches,
            |_, template| templates.get(template).copied(),
            |_| prefix,
            |_| suffix,
            factor,
            |_, _, _, _, _, _| {},
        )
    }

    /// A microbatch visits A twice: before dispatch and after the return.
    /// Queue only ready work; reserving a future restore would stall attention
    /// which could run while that microbatch is still on F.
    pub(super) fn schedule(
        self,
        layers: &[usize],
        microbatches: u32,
        durations: impl Fn(usize, usize) -> Option<[f64; 5]>,
        prefix: impl Fn(usize) -> f64,
        suffix: impl Fn(usize) -> f64,
        factor: f64,
        mut observe: impl FnMut(usize, usize, usize, u8, f64, f64),
    ) -> Result<f64, AicError> {
        if microbatches == 0 || layers.is_empty() || !factor.is_finite() || factor <= 0.0 {
            return Err(AicError::InvalidEngineConfig(
                "invalid AFD pipeline geometry".into(),
            ));
        }
        let small = microbatches <= 4;
        let mut queues: [ReadyQueue; 4] = std::array::from_fn(|_| ReadyQueue::new(small));
        let mut running: [Option<(f64, Task)>; 4] = [None; 4];
        let mut now = 0.0_f64;
        let mut finished = 0;
        let enqueue = |mut task: Task, queues: &mut [ReadyQueue; 4]| -> Result<bool, AicError> {
            loop {
                let d = durations(task.micro, layers[task.layer]).ok_or_else(|| {
                    AicError::InvalidEngineConfig("invalid AFD layer template".into())
                })?;
                if d.iter().any(|v| !v.is_finite() || *v < 0.0) {
                    return Err(AicError::InvalidEngineConfig(
                        "invalid AFD stage duration".into(),
                    ));
                }
                let (resource, mut duration) = match (self, task.stage) {
                    (_, 5) => (0, suffix(task.micro)),
                    (Self::FourStage, 0) => (0, d[0]),
                    (Self::FourStage, 1) => (1, d[1] * factor),
                    (Self::FourStage, 2) => (2, d[2]),
                    (Self::FourStage, 3) => (3, d[3] * factor),
                    (_, 4) => (0, d[4]),
                    (Self::TwoStage, 0) => (0, d[0] + d[1] * factor),
                    (Self::TwoStage, 2) => (2, d[2] + d[3] * factor),
                    _ => unreachable!(),
                };
                if task.layer == 0 && task.stage == 0 {
                    duration += prefix(task.micro);
                }
                if !duration.is_finite() || duration < 0.0 {
                    return Err(AicError::InvalidEngineConfig(
                        "invalid AFD stage duration".into(),
                    ));
                }
                if duration > 0.0 {
                    task.duration = duration;
                    queues[resource].push(task);
                    return Ok(false);
                }
                if !task.advance(self, layers.len()) {
                    return Ok(true);
                }
            }
        };
        for micro in 0..microbatches as usize {
            finished += usize::from(enqueue(
                Task {
                    micro,
                    ..Task::default()
                },
                &mut queues,
            )?);
        }
        loop {
            // Settle ALL completions at this timestamp before dispatching.
            for slot in &mut running {
                if slot.is_some_and(|(end, _)| end == now) {
                    let (_, mut task) = slot.take().unwrap();
                    task.ready = now;
                    if task.advance(self, layers.len()) {
                        finished += usize::from(enqueue(task, &mut queues)?);
                    } else {
                        finished += 1;
                    }
                }
            }
            if finished == microbatches as usize {
                break;
            }
            for (resource, (queue, slot)) in queues.iter_mut().zip(&mut running).enumerate() {
                if slot.is_none() {
                    if let Some(task) = queue.pop() {
                        let end = now + task.duration;
                        if !end.is_finite() {
                            return Err(AicError::InvalidEngineConfig("AFD pass overflow".into()));
                        }
                        observe(resource, task.micro, task.layer, task.stage, now, end);
                        *slot = Some((end, task));
                    }
                }
            }
            now = running
                .iter()
                .flatten()
                .map(|(end, _)| *end)
                .min_by(f64::total_cmp)
                .ok_or_else(|| {
                    AicError::InvalidEngineConfig("AFD pipeline has unfinished work".into())
                })?;
        }
        if now <= 0.0 {
            return Err(AicError::InvalidEngineConfig(
                "invalid AFD pass latency".into(),
            ));
        }
        Ok(now)
    }

    /// Apply the transfer factor once, before grouping stages.
    pub fn step_latency_ms(
        self,
        times: &AfdLayerTimes,
        microbatches: u32,
        comm_overhead_factor: f64,
    ) -> Result<f64, AicError> {
        if !comm_overhead_factor.is_finite() || comm_overhead_factor <= 0.0 {
            return Err(AicError::InvalidEngineConfig(
                "AFD comm_overhead_factor must be positive and finite".into(),
            ));
        }
        self.plan_latency_ms(
            &[[
                times.attention_ms,
                times.a_to_f_ms,
                times.ffn_ms,
                times.f_to_a_ms,
                0.0,
            ]],
            &vec![0; times.num_layers as usize],
            microbatches,
            comm_overhead_factor,
            0.0,
            0.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restoration_contends_with_attention_not_f_or_return() {
        for (layout, expected) in [
            (AfdPipelineLayout::FourStage, 40.),
            (AfdPipelineLayout::TwoStage, 42.),
        ] {
            let mut intervals: [Vec<(f64, f64)>; 4] = std::array::from_fn(|_| Vec::new());
            let step = layout
                .schedule(
                    &[0],
                    2,
                    |_, _| Some([10., 1., 1., 1., 10.]),
                    |_| 0.,
                    |_| 0.,
                    1.,
                    |r, _, _, _, start, end| intervals[r].push((start, end)),
                )
                .unwrap();
            // A executes two 10ms attentions and two 10ms restores; in
            // two_stage it also executes the two 1ms outgoing transfers.
            assert_eq!(step, expected);
            for resource in intervals {
                for pair in resource.windows(2) {
                    assert!(pair[0].1 <= pair[1].0);
                }
            }
        }
    }

    #[test]
    fn restoration_feedback_preserves_source_layers_and_ties() {
        // With unit durations, four_stage completes the two layer-1 restores
        // at 5/6, then layer-2 restores at 11/12. two_stage completes at 13.
        for (layout, expected) in [
            (AfdPipelineLayout::FourStage, 12.),
            (AfdPipelineLayout::TwoStage, 13.),
        ] {
            let mut returns = [[0.; 2]; 2];
            let mut first_visit = [[0.; 2]; 2];
            let step = layout
                .schedule(
                    &[0, 0],
                    2,
                    |_, _| Some([1.; 5]),
                    |_| 0.,
                    |_| 0.,
                    1.,
                    |resource, micro, layer, stage, start, end| {
                        if stage == 4 {
                            assert_eq!(resource, 0);
                            returns[micro][layer] = end;
                        }
                        if stage == 0 {
                            first_visit[micro][layer] = start;
                        }
                    },
                )
                .unwrap();
            assert_eq!(step, expected);
            for micro in 0..2 {
                assert!(first_visit[micro][1] >= returns[micro][0]);
            }
        }
    }

    #[test]
    fn endpoints_execute_once_on_the_a_resource() {
        // m=2, L=2: routed returns are 8/11 then 14/17. The A head
        // starts at 14 and 19, costs 5 each, and completes the pass at 24.
        assert_eq!(
            AfdPipelineLayout::FourStage
                .plan_latency_ms(&[[1., 1., 3., 1., 0.]], &[0, 0], 2, 1., 2., 5.)
                .unwrap(),
            24.
        );
        // Amortizing the seven endpoint milliseconds into two layers changes
        // the resource ordering and gives 23.5 instead.
        assert_eq!(
            AfdPipelineLayout::FourStage
                .step_latency_ms(&times([4.5, 1., 3., 1.], 2), 2, 1.)
                .unwrap(),
            23.5
        );
    }

    #[test]
    fn heterogeneous_layers_skip_idle_resources_and_preserve_source_order() {
        let templates = [[1., 1., 10., 1., 0.], [2., 0., 0., 0., 0.]];
        // Routed microbatches return at 13 and 23. The first dense layer
        // completes at 15, independently of the F server still busy to 22.
        assert_eq!(
            AfdPipelineLayout::FourStage
                .plan_latency_ms(&templates, &[0, 1], 2, 1., 0., 0.)
                .unwrap(),
            25.
        );
        assert_eq!(
            AfdPipelineLayout::FourStage
                .plan_latency_ms(&templates, &[1, 0], 2, 1., 0., 0.)
                .unwrap(),
            27.
        );
        assert_eq!(
            AfdPipelineLayout::TwoStage
                .plan_latency_ms(&templates, &[0, 1], 2, 1., 0., 0.)
                .unwrap(),
            26.
        );
        assert_eq!(
            AfdPipelineLayout::TwoStage
                .plan_latency_ms(&templates, &[1, 0], 2, 1., 0., 0.)
                .unwrap(),
            28.
        );
    }

    #[test]
    fn return_restore_and_one_time_work_are_not_transfer_scaled() {
        // One routed layer costs 1+2+3+4+5, dense costs 7, prefix/suffix
        // cost 2+3: 27. Doubling transfers adds only 2+4, giving 33.
        for layout in [AfdPipelineLayout::FourStage, AfdPipelineLayout::TwoStage] {
            let templates = [[1., 2., 3., 4., 5.], [7., 0., 0., 0., 0.]];
            assert_eq!(
                layout
                    .plan_latency_ms(&templates, &[0, 1], 1, 1., 2., 3.)
                    .unwrap(),
                27.
            );
            assert_eq!(
                layout
                    .plan_latency_ms(&templates, &[0, 1], 1, 2., 2., 3.)
                    .unwrap(),
                33.
            );
        }
    }

    fn times(stages: [f64; 4], layers: u32) -> AfdLayerTimes {
        AfdLayerTimes {
            attention_ms: stages[0],
            a_to_f_ms: stages[1],
            ffn_ms: stages[2],
            f_to_a_ms: stages[3],
            num_layers: layers,
        }
    }

    /// Four unit stages finish the first microbatch at 4 and others one
    /// unit apart. With two merged stages, completions are two units apart.
    /// Feedback permits a two-stage pipeline to stay busy with m>=2.
    #[test]
    fn hand_timeline_for_common_depths() {
        for (layout, single, double) in [
            (
                AfdPipelineLayout::FourStage,
                [4., 5., 6., 7.],
                [8., 9., 10., 11.],
            ),
            (
                AfdPipelineLayout::TwoStage,
                [4., 6., 8., 10.],
                [8., 10., 14., 18.],
            ),
        ] {
            for m in 1..=4 {
                assert_eq!(
                    layout.step_latency_ms(&times([1.; 4], 1), m, 1.).unwrap(),
                    single[m as usize - 1]
                );
                assert_eq!(
                    layout.step_latency_ms(&times([1.; 4], 2), m, 1.).unwrap(),
                    double[m as usize - 1]
                );
            }
        }
    }

    /// For [2,4,3,5], first-layer completions are 14,19,24,29.
    /// For m=2, second-layer completions are 28,33 (feedback causes a gap).
    /// Merging yields [6,8], whose m>=2 cadence is 8 after the first 14.
    #[test]
    fn unequal_stages_and_feedback() {
        for (layout, expected) in [
            (AfdPipelineLayout::FourStage, [28., 33., 39., 49.]),
            (AfdPipelineLayout::TwoStage, [28., 38., 54., 70.]),
        ] {
            for m in 1..=4 {
                assert_eq!(
                    layout
                        .step_latency_ms(&times([2., 4., 3., 5.], 2), m, 1.)
                        .unwrap(),
                    expected[m as usize - 1]
                );
            }
        }
    }

    #[test]
    fn zero_transfers_compute_and_one_way_transfer_bottlenecks() {
        // Only one nonzero server: every job consumes exactly that duration.
        for stages in [
            [5., 0., 0., 0.],
            [0., 5., 0., 0.],
            [0., 0., 5., 0.],
            [0., 0., 0., 5.],
        ] {
            for layout in [AfdPipelineLayout::FourStage, AfdPipelineLayout::TwoStage] {
                for m in [1, 2, 3, 4, 7] {
                    assert_eq!(
                        layout.step_latency_ms(&times(stages, 3), m, 1.).unwrap(),
                        15. * f64::from(m)
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_times_and_counts_reject() {
        let layout = AfdPipelineLayout::FourStage;
        for bad in [-1., f64::NAN, f64::INFINITY] {
            assert!(
                layout
                    .step_latency_ms(&times([1., bad, 1., 1.], 2), 2, 1.)
                    .is_err()
            );
        }
        assert!(layout.step_latency_ms(&times([0.; 4], 2), 2, 1.).is_err());
        assert!(layout.step_latency_ms(&times([1.; 4], 0), 2, 1.).is_err());
        assert!(layout.step_latency_ms(&times([1.; 4], 2), 0, 1.).is_err());
    }
}
