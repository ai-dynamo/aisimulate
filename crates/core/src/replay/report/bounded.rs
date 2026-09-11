// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact external-memory quantiles for batch replay. No additional sketch error.
//! Temporary files are anonymous and removed on success, error, or process exit.
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

use super::*;
use anyhow::{Context, Result, bail};

const MEMORY_SAMPLES: usize = 4096;
// Four distributions together consume at most 1 GiB of temporary disk space.
const MAX_SAMPLES: usize = (256 * 1024 * 1024) / 8;
pub(super) const MAX_RETAINED_REQUESTS: usize = 100_000;

#[derive(Debug, Default)]
struct Samples {
    memory: Vec<f64>,
    disk: Option<BufWriter<File>>,
    count: usize,
}

impl Samples {
    fn push(&mut self, value: f64) -> Result<()> {
        if self.count == MAX_SAMPLES {
            bail!(
                "resource_limited: exact report samples exceed the 1 GiB temporary-storage limit"
            );
        }
        if self.disk.is_none() && self.memory.len() == MEMORY_SAMPLES {
            let mut file =
                BufWriter::new(tempfile::tempfile().context("create exact report sample file")?);
            for sample in self.memory.drain(..) {
                file.write_all(&sample.to_le_bytes())?;
            }
            self.disk = Some(file);
        }
        if let Some(file) = &mut self.disk {
            file.write_all(&value.to_le_bytes())
                .context("write exact report sample")?;
        } else {
            self.memory.push(value);
        }
        self.count += 1;
        Ok(())
    }

    fn finish(self) -> Result<TraceDistributionStats> {
        let Some(mut file) = self.disk else {
            return Ok(build_distribution_stats(self.memory));
        };
        file.flush().context("flush exact report samples")?;
        let mut file = file.into_inner().map_err(|error| error.into_error())?;
        let mut sum = 0.0;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        scan(&mut file, self.count, |value| {
            sum += value;
            min = min.min(value);
            max = max.max(value);
        })?;
        let mean = sum / self.count as f64;
        let mut squared = 0.0;
        scan(&mut file, self.count, |value| {
            squared += (value - mean).powi(2)
        })?;

        // Simultaneous radix selection needs five 256-bin histograms and eight
        // sequential file passes, independent of the number of samples.
        let mut ranks = [50., 75., 90., 95., 99.].map(|p| percentile_rank(self.count, p));
        let mut prefixes = [0_u64; 5];
        let mut mask = 0_u64;
        for byte in (0..8).rev() {
            let shift = byte * 8;
            let mut bins = [[0_usize; 256]; 5];
            scan(&mut file, self.count, |value| {
                let key = ordered_key(value);
                for target in 0..5 {
                    if key & mask == prefixes[target] {
                        bins[target][((key >> shift) & 255) as usize] += 1;
                    }
                }
            })?;
            for target in 0..5 {
                for bucket in 0..256 {
                    if ranks[target] < bins[target][bucket] {
                        prefixes[target] |= (bucket as u64) << shift;
                        break;
                    }
                    ranks[target] -= bins[target][bucket];
                }
            }
            mask |= 255_u64 << shift;
        }
        let [median_ms, p75_ms, p90_ms, p95_ms, p99_ms] = prefixes.map(from_ordered_key);
        Ok(TraceDistributionStats {
            mean_ms: mean,
            min_ms: min,
            max_ms: max,
            median_ms,
            p75_ms,
            p90_ms,
            p95_ms,
            p99_ms,
            std_ms: (squared / self.count as f64).sqrt(),
        })
    }
}

fn ordered_key(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits >> 63 == 0 {
        bits ^ (1 << 63)
    } else {
        !bits
    }
}
fn from_ordered_key(key: u64) -> f64 {
    f64::from_bits(if key >> 63 == 0 {
        !key
    } else {
        key ^ (1 << 63)
    })
}
fn scan(file: &mut File, count: usize, mut visit: impl FnMut(f64)) -> Result<()> {
    file.seek(SeekFrom::Start(0))?;
    let mut reader = BufReader::new(file);
    for _ in 0..count {
        let mut bytes = [0; 8];
        reader
            .read_exact(&mut bytes)
            .context("read exact report samples")?;
        visit(f64::from_le_bytes(bytes));
    }
    Ok(())
}

#[derive(Debug, Default)]
pub(super) struct BoundedSummary {
    total: usize,
    completed: usize,
    input: usize,
    output: usize,
    reused: usize,
    first_reused: usize,
    duration_ms: f64,
    good_requests: usize,
    good_output: usize,
    ttft: Samples,
    ttst: Samples,
    tpot: Samples,
    e2e: Samples,
}

impl BoundedSummary {
    pub(super) fn duration_ms(&self) -> f64 {
        self.duration_ms
    }

    pub(super) fn add(&mut self, stats: &TraceRequestStats, sla: SlaThresholds) -> Result<()> {
        self.total += 1;
        if stats.first_admit_ms.is_none()
            || stats.terminal_status != Some(ReplayTerminalStatus::Completed)
        {
            return Ok(());
        }
        let Some(terminal_ms) = stats.terminal_time_ms else {
            return Ok(());
        };
        self.completed += 1;
        self.input += stats.input_length;
        let output = stats.actual_output_length();
        self.output += output;
        self.reused += stats.reused_input_tokens;
        self.first_reused += stats.first_admission_reused_input_tokens;
        self.duration_ms = self.duration_ms.max(terminal_ms);
        let (Some(first), Some(last)) = (stats.first_token_ms(), stats.last_token_ms()) else {
            if sla.is_set()
                && sla.is_good_without_tokens((terminal_ms - stats.arrival_time_ms).max(0.0))
            {
                self.good_requests += 1;
            }
            return Ok(());
        };
        let ttft = (first - stats.arrival_time_ms).max(0.0);
        let e2e = (last - stats.arrival_time_ms).max(0.0);
        self.ttft.push(ttft)?;
        self.e2e.push(e2e)?;
        if let Some(value) = stats.ttst_ms() {
            self.ttst.push(value)?;
        }
        if let Some(value) = stats.mean_tpot_ms() {
            self.tpot.push(value)?;
        }
        if sla.is_set() && sla.is_good(ttft, e2e, output) {
            self.good_requests += 1;
            self.good_output += output;
        }
        Ok(())
    }

    pub(super) fn finish(self, mut report: ReplayReport) -> Result<ReplayReport> {
        report.request_counts = TraceRequestCounts {
            num_requests: self.total,
            completed_requests: self.completed,
            total_input_tokens: self.input,
            total_output_tokens: self.output,
        };
        let seconds = (self.duration_ms / 1000.0).max(1e-9);
        let throughput = &mut report.throughput;
        throughput.duration_ms = self.duration_ms;
        throughput.request_throughput_rps = self.completed as f64 / seconds;
        throughput.input_throughput_tok_s = self.input as f64 / seconds;
        throughput.output_throughput_tok_s = self.output as f64 / seconds;
        throughput.total_throughput_tok_s = (self.input + self.output) as f64 / seconds;
        report.prefix_cache_reused_ratio = if self.input == 0 {
            0.0
        } else {
            self.reused as f64 / self.input as f64
        };
        report.first_admission_prefix_cache_reused_ratio = if self.input == 0 {
            0.0
        } else {
            self.first_reused as f64 / self.input as f64
        };
        if let Some(goodput) = &mut report.goodput {
            goodput.completed_requests = self.good_requests;
            goodput.request_throughput_rps = self.good_requests as f64 / seconds;
            goodput.output_throughput_tok_s = self.good_output as f64 / seconds;
        }
        report.latency.num_ttft_samples = self.ttft.count;
        report.latency.num_tpot_samples = self.tpot.count;
        report.latency.num_e2e_latency_samples = self.e2e.count;
        report.latency.ttft = self.ttft.finish()?;
        report.latency.ttst = self.ttst.finish()?;
        report.latency.tpot = self.tpot.finish()?;
        report.latency.e2e = self.e2e.finish()?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spilled_quantiles_match_exact_selection() {
        let values: Vec<_> = (0..20_003)
            .map(|i| ((i * 1237) % 2003) as f64 / 7.0)
            .collect();
        let expected = build_distribution_stats(values.clone());
        let mut samples = Samples::default();
        for value in values {
            samples.push(value).unwrap();
        }
        assert!(samples.disk.is_some());
        assert!(samples.memory.capacity() <= MEMORY_SAMPLES);
        let actual = samples.finish().unwrap();
        for (left, right) in [
            (actual.median_ms, expected.median_ms),
            (actual.p75_ms, expected.p75_ms),
            (actual.p90_ms, expected.p90_ms),
            (actual.p95_ms, expected.p95_ms),
            (actual.p99_ms, expected.p99_ms),
        ] {
            assert_eq!(left, right);
        }
        assert!((actual.mean_ms - expected.mean_ms).abs() < 1e-10);
        assert!((actual.std_ms - expected.std_ms).abs() < 1e-10);
    }

    #[test]
    fn truncated_spool_is_an_error() {
        let mut samples = Samples::default();
        for i in 0..5000 {
            samples.push(i as f64).unwrap();
        }
        let file = samples.disk.as_mut().unwrap();
        file.flush().unwrap();
        file.get_mut().set_len(8).unwrap();
        assert!(
            samples
                .finish()
                .unwrap_err()
                .to_string()
                .contains("read exact report samples")
        );
    }

    #[test]
    fn sample_limit_refuses_before_writing() {
        let mut samples = Samples {
            count: MAX_SAMPLES,
            ..Default::default()
        };
        assert!(
            samples
                .push(1.0)
                .unwrap_err()
                .to_string()
                .contains("resource_limited")
        );
        assert!(samples.memory.is_empty());
    }
}
