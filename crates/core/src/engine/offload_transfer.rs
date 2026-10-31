// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Max-min fair byte service shared by offload tiers and the P/D handoff.
//!
//! Each owner keeps its own instance and budgets. Callers resolve bandwidth,
//! first-byte readiness and job IDs; this module only advances bytes through
//! virtual time. A job draws from up to two links, each a budget shared by the
//! moving jobs that name it in the same slot and direction: an offload tier
//! uses its client's link and the pool's link, the P/D handoff its source and
//! destination ranks' links. Rates are max-min fair: the most constrained
//! link's jobs get equal shares, and capacity they leave unused on their other
//! link goes to that link's remaining jobs. Jobs waiting for their first byte
//! consume no bandwidth, and admission, completion or cancellation reflows
//! service.

use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Direction {
    Read,
    Write,
}

/// Decimal GB/s to bytes per millisecond. Zero removes the limit.
pub(crate) fn bytes_per_ms(gbps: f64) -> f64 {
    if gbps == 0.0 {
        f64::INFINITY
    } else {
        gbps * 1e6
    }
}

/// One bandwidth budget in bytes per millisecond. An infinite rate is no limit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Link {
    pub(crate) id: u64,
    pub(crate) rate: f64,
}

impl Link {
    pub(crate) const UNLIMITED: Self = Self {
        id: 0,
        rate: f64::INFINITY,
    };
}

pub(crate) struct Job {
    pub(crate) id: u64,
    pub(crate) direction: Direction,
    /// First-byte time; the job moves no bytes before it.
    pub(crate) ready: f64,
    pub(crate) bytes: f64,
    /// Budgets by slot. Jobs naming one link ID in one slot and direction
    /// share it, and all of them must give it the same rate.
    pub(crate) links: [Link; 2],
}

/// Above this many links, progressive filling keeps a heap of shares.
const SCAN_BUDGETS: usize = 32;

struct Budget {
    remaining: f64,
    active: usize,
}

impl Budget {
    fn share(&self) -> f64 {
        self.remaining.max(0.0) / self.active as f64
    }
}

/// A budget's equal share when it was queued. The heap pops the smallest
/// share first and, among equal shares, the first-seen budget.
#[derive(PartialEq)]
struct Bottleneck {
    share: f64,
    budget: usize,
}

impl Eq for Bottleneck {}

impl Ord for Bottleneck {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .share
            .total_cmp(&self.share)
            .then_with(|| other.budget.cmp(&self.budget))
    }
}

impl PartialOrd for Bottleneck {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Buffers `fair_rates_into` reuses across events; no state survives a call.
#[derive(Default)]
struct RateScratch {
    index: FxHashMap<(usize, u64, Direction), usize>,
    budgets: Vec<Budget>,
    job_budgets: Vec<[Option<usize>; 2]>,
    /// Jobs of each budget, contiguous from `member_starts[budget]`.
    member_starts: Vec<usize>,
    members: Vec<usize>,
    cursor: Vec<usize>,
    fixed: Vec<bool>,
    bottlenecks: BinaryHeap<Bottleneck>,
}

/// Max-min fair rate of every job at `now`, in job order; waiters get zero.
#[cfg(test)]
fn fair_rates(jobs: &[Job], now: f64) -> Vec<f64> {
    let mut rates = Vec::new();
    fair_rates_into(jobs, now, &mut RateScratch::default(), &mut rates);
    rates
}

/// Write the max-min fair rate of every job at `now` into `rates` and return
/// the earliest first byte still to come, or infinity.
///
/// Progressive filling: repeatedly fix the jobs of the link with the smallest
/// equal share, then charge their rate to their other link. Rates depend only
/// on the jobs and which are past their first byte, so they stay exact until
/// the job set changes or the clock reaches the returned first byte.
fn fair_rates_into(jobs: &[Job], now: f64, scratch: &mut RateScratch, rates: &mut Vec<f64>) -> f64 {
    let RateScratch {
        index,
        budgets,
        job_budgets,
        member_starts,
        members,
        cursor,
        fixed,
        bottlenecks,
    } = scratch;
    index.clear();
    budgets.clear();
    job_budgets.clear();
    rates.clear();
    rates.resize(jobs.len(), 0.0);
    let mut coupled = false;
    let mut first_byte = f64::INFINITY;
    for (job_index, job) in jobs.iter().enumerate() {
        let mut own = [None; 2];
        if job.ready <= now {
            rates[job_index] = f64::INFINITY;
            for (slot, link) in job.links.iter().enumerate() {
                if link.rate.is_infinite() {
                    continue;
                }
                let budget = *index
                    .entry((slot, link.id, job.direction))
                    .or_insert_with(|| {
                        budgets.push(Budget {
                            remaining: link.rate,
                            active: 0,
                        });
                        budgets.len() - 1
                    });
                debug_assert_eq!(budgets[budget].remaining, link.rate);
                budgets[budget].active += 1;
                own[slot] = Some(budget);
            }
            coupled |= own.iter().all(Option::is_some);
        } else {
            first_byte = first_byte.min(job.ready);
        }
        job_budgets.push(own);
    }
    if !coupled {
        // No job spans two finite links, so every link is its own bottleneck.
        for (rate, own) in rates.iter_mut().zip(job_budgets.iter()) {
            if let Some(budget) = own.iter().flatten().next() {
                let budget = &budgets[*budget];
                *rate = budget.remaining / budget.active as f64;
            }
        }
        return first_byte;
    }
    member_starts.clear();
    let mut start = 0;
    for budget in budgets.iter() {
        member_starts.push(start);
        start += budget.active;
    }
    members.clear();
    members.resize(start, 0);
    cursor.clone_from(member_starts);
    for (job_index, own) in job_budgets.iter().enumerate() {
        for &budget in own.iter().flatten() {
            members[cursor[budget]] = job_index;
            cursor[budget] += 1;
        }
    }
    fixed.clear();
    fixed.resize(jobs.len(), false);
    if budgets.len() <= SCAN_BUDGETS {
        // Few links: a full scan per round is cheaper than heap upkeep.
        // Ties keep first-seen order, so equal shares resolve deterministically.
        while let Some((bottleneck, share)) = budgets
            .iter()
            .enumerate()
            .filter(|(_, budget)| budget.active > 0)
            .map(|(budget_index, budget)| (budget_index, budget.share()))
            .min_by(|left, right| left.1.total_cmp(&right.1))
        {
            fix_bottleneck(
                bottleneck,
                share,
                member_starts,
                members,
                fixed,
                job_budgets,
                budgets,
                rates,
                |_, _| {},
            );
        }
        return first_byte;
    }
    // A change to a budget queues its new share, and a popped share that no
    // longer matches its budget is stale. Each round thus fixes the jobs of
    // the smallest current share, first-seen budget on ties, as the scan does.
    bottlenecks.clear();
    bottlenecks.extend(
        budgets
            .iter()
            .enumerate()
            .map(|(budget, state)| Bottleneck {
                share: state.share(),
                budget,
            }),
    );
    while let Some(Bottleneck {
        share,
        budget: bottleneck,
    }) = bottlenecks.pop()
    {
        let state = &budgets[bottleneck];
        if state.active == 0 || state.share().total_cmp(&share).is_ne() {
            continue;
        }
        fix_bottleneck(
            bottleneck,
            share,
            member_starts,
            members,
            fixed,
            job_budgets,
            budgets,
            rates,
            |budget, state| {
                bottlenecks.push(Bottleneck {
                    share: state.share(),
                    budget,
                })
            },
        );
    }
    first_byte
}

/// Fix the unfixed jobs of `bottleneck` at `share` and charge their other
/// link, reporting each charged budget that still has jobs.
#[allow(clippy::too_many_arguments)]
fn fix_bottleneck(
    bottleneck: usize,
    share: f64,
    member_starts: &[usize],
    members: &[usize],
    fixed: &mut [bool],
    job_budgets: &[[Option<usize>; 2]],
    budgets: &mut [Budget],
    rates: &mut [f64],
    mut charged: impl FnMut(usize, &Budget),
) {
    let end = member_starts
        .get(bottleneck + 1)
        .copied()
        .unwrap_or(members.len());
    for &job_index in &members[member_starts[bottleneck]..end] {
        if std::mem::replace(&mut fixed[job_index], true) {
            continue;
        }
        rates[job_index] = share;
        for budget in job_budgets[job_index].into_iter().flatten() {
            budgets[budget].active -= 1;
            if budget != bottleneck {
                budgets[budget].remaining -= share;
                if budgets[budget].active > 0 {
                    charged(budget, &budgets[budget]);
                }
            }
        }
    }
}

/// One client's next event, from [`FairTransfers::next_client_events`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct ClientEvent {
    /// Jobs past their first byte, by direction.
    ready: [usize; 2],
    pub(crate) at: f64,
}

#[derive(Default)]
pub(crate) struct FairTransfers {
    now: f64,
    /// Submission order; completions in one step are reported in this order.
    jobs: Vec<Job>,
    /// Boxed so an idle tier stays small.
    scratch: Box<RefCell<RateScratch>>,
    rates: RefCell<Vec<f64>>,
    /// `rates` hold the current jobs' rates while the clock is before this
    /// first-byte time. A new waiter appends a zero rate and can lower it; any
    /// other job set change resets it.
    rates_valid_until: Cell<f64>,
    /// `next_event`, kept until the clock or the job set changes; every client
    /// of a shared pool asks between changes.
    cached_event: Cell<Option<Option<f64>>>,
    events: Vec<f64>,
    #[cfg(test)]
    pub(crate) rate_refreshes: Cell<usize>,
    /// Iterations of the `advance` loop.
    #[cfg(test)]
    steps: usize,
}

impl FairTransfers {
    pub(crate) fn now(&self) -> f64 {
        self.now
    }

    /// Whether any job names `link` in its first slot.
    pub(crate) fn has_link_jobs(&self, link: u64) -> bool {
        self.jobs.iter().any(|job| job.links[0].id == link)
    }

    /// Caller must first advance to the submission time.
    pub(crate) fn submit(&mut self, job: Job) {
        debug_assert!(job.ready >= self.now);
        if job.ready > self.now && self.now < self.rates_valid_until.get() {
            // A waiter holds no share before its first byte, so the other
            // jobs keep their rates.
            self.rates.get_mut().push(0.0);
            let valid_until = self.rates_valid_until.get_mut();
            *valid_until = valid_until.min(job.ready);
            *self.cached_event.get_mut() = None;
        } else {
            self.invalidate();
        }
        self.jobs.push(job);
    }

    pub(crate) fn cancel(&mut self, id: u64) -> bool {
        let before = self.jobs.len();
        self.jobs.retain(|job| job.id != id);
        let canceled = self.jobs.len() != before;
        if canceled {
            self.invalidate();
        }
        canceled
    }

    /// Forget the rates and the next event after a job set change.
    fn invalidate(&mut self) {
        *self.rates_valid_until.get_mut() = f64::NEG_INFINITY;
        *self.cached_event.get_mut() = None;
    }

    fn event_time(&self, job: &Job, rate: f64) -> f64 {
        if job.ready > self.now {
            job.ready
        } else {
            self.now + job.bytes / rate
        }
    }

    fn event_times<'a>(&'a self, rates: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
        self.jobs
            .iter()
            .zip(rates)
            .map(|(job, &rate)| self.event_time(job, rate))
    }

    /// Recompute `rates` unless they still hold at the current instant.
    fn refresh_rates(&self) {
        if self.now < self.rates_valid_until.get() {
            debug_assert_eq!(self.rates.borrow().len(), self.jobs.len());
            return;
        }
        #[cfg(test)]
        self.rate_refreshes.set(self.rate_refreshes.get() + 1);
        let first_byte = fair_rates_into(
            &self.jobs,
            self.now,
            &mut self.scratch.borrow_mut(),
            &mut self.rates.borrow_mut(),
        );
        self.rates_valid_until.set(first_byte);
    }

    /// Earliest first-byte or completion event under the current job set.
    pub(crate) fn next_event(&self) -> Option<f64> {
        if let Some(event) = self.cached_event.get() {
            return event;
        }
        self.refresh_rates();
        let event = self
            .event_times(&self.rates.borrow())
            .min_by(f64::total_cmp);
        self.cached_event.set(Some(event));
        event
    }

    /// Whether an already computed next event is due by `target`. Computes
    /// nothing, so callers can skip a step without pricing the job set.
    pub(crate) fn known_event_due(&self, target: f64) -> bool {
        matches!(self.cached_event.get(), Some(Some(event)) if event <= target)
    }

    /// Earliest first-byte or completion event of each client with jobs, as
    /// `(client, event)` into `events`, in two passes over all jobs. `events` is
    /// cleared first and keeps its capacity. A client is a job's first link;
    /// with an unlimited second link, only a client's own jobs set their rates,
    /// so no other client's event can move its event earlier.
    pub(crate) fn next_client_events(&self, events: &mut FxHashMap<u64, ClientEvent>) {
        events.clear();
        for job in &self.jobs {
            debug_assert!(job.links[1].rate.is_infinite());
            let event = events.entry(job.links[0].id).or_insert(ClientEvent {
                ready: [0; 2],
                at: f64::INFINITY,
            });
            if job.ready <= self.now {
                event.ready[job.direction as usize] += 1;
            }
        }
        for job in &self.jobs {
            let event = events
                .get_mut(&job.links[0].id)
                .expect("the first pass counts every client");
            let at = if job.ready > self.now {
                job.ready
            } else {
                // Same arithmetic as uncoupled `fair_rates`, so the event
                // matches the boundary `advance` selects.
                let rate = job.links[0].rate / event.ready[job.direction as usize].max(1) as f64;
                self.now + job.bytes / rate
            };
            if at.total_cmp(&event.at).is_lt() {
                event.at = at;
            }
        }
    }

    /// Move service to `target`, returning `(job id, completion time)` in
    /// event order and, within one event, submission order.
    pub(crate) fn advance(&mut self, target: f64) -> Vec<(u64, f64)> {
        let target = target.max(self.now);
        let mut completed = Vec::new();
        if target > self.now {
            // Service ends at `target`, so every remaining event moves.
            *self.cached_event.get_mut() = None;
        } else if self
            .cached_event
            .get()
            .is_some_and(|event| event.is_none_or(|event| event > target))
        {
            // A known event after the current instant leaves nothing to do.
            return completed;
        }
        let mut events = std::mem::take(&mut self.events);
        loop {
            #[cfg(test)]
            {
                self.steps += 1;
            }
            self.refresh_rates();
            events.clear();
            events.extend(self.event_times(&self.rates.borrow()));
            let event = events.iter().copied().min_by(f64::total_cmp);
            if self.now == target && event.is_none_or(|time| time > target) {
                // Nothing moves or becomes due at `target`, including waiters
                // a step to it made eligible; keep the event for the next poll.
                *self.cached_event.get_mut() = Some(event);
                break;
            }
            let next = event.unwrap_or(target).min(target);
            let elapsed = next - self.now;
            let now = self.now;
            let rates = self.rates.get_mut();
            for ((job, &rate), &event) in self.jobs.iter_mut().zip(rates.iter()).zip(&events) {
                if job.ready > now {
                    continue;
                }
                // The selected completion boundary is authoritative. Repeated
                // byte subtraction at a large timestamp can leave a residual
                // whose duration is below one clock ULP and would never drain.
                job.bytes = if event <= next {
                    0.0
                } else {
                    (job.bytes - elapsed * rate).max(0.0)
                };
            }
            self.now = next;
            let before = completed.len();
            let now = self.now;
            self.jobs.retain(|job| {
                let done = job.ready <= now && job.bytes == 0.0;
                if done {
                    completed.push((job.id, now));
                }
                !done
            });
            if completed.len() != before {
                self.invalidate();
            }
            if event.is_none_or(|time| time > target) {
                break;
            }
        }
        self.events = events;
        completed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: f64 = 1_000_000.0;

    fn job(id: u64, client: u64, direction: Direction, ready: f64, rates: (f64, f64)) -> Job {
        Job {
            id,
            direction,
            ready,
            bytes: MB,
            links: [
                Link {
                    id: client,
                    rate: bytes_per_ms(rates.0),
                },
                Link {
                    id: 0,
                    rate: bytes_per_ms(rates.1),
                },
            ],
        }
    }

    /// One-MB job over a 1 GB/s source and destination link.
    fn pd_job(id: u64, source: u64, destination: u64) -> Job {
        let link = |id| Link {
            id,
            rate: bytes_per_ms(1.0),
        };
        Job {
            id,
            direction: Direction::Write,
            ready: 0.0,
            bytes: MB,
            links: [link(source), link(destination)],
        }
    }

    fn run(jobs: Vec<Job>) -> Vec<(u64, f64)> {
        let mut transfers = FairTransfers::default();
        jobs.into_iter().for_each(|job| transfers.submit(job));
        transfers.advance(f64::MAX)
    }

    #[test]
    fn directions_have_independent_client_and_shared_budgets() {
        // G3 tier tests cover client and shared caps within one direction.
        let rates = (1.0, 1.0);
        assert_eq!(
            run(vec![
                job(0, 0, Direction::Write, 0.0, rates),
                job(1, 0, Direction::Read, 0.0, rates),
            ]),
            [(0, 1.0), (1, 1.0)]
        );
    }

    #[test]
    fn same_time_completions_follow_event_steps() {
        // Equal timestamps follow event steps: the moving job finishes in the
        // step that makes the instantaneous waiter ready. Tiers own any other
        // same-time order.
        assert_eq!(
            run(vec![
                job(0, 0, Direction::Write, 1.0, (0.0, 0.0)),
                job(1, 1, Direction::Write, 0.0, (1.0, 0.0)),
            ]),
            [(1, 1.0), (0, 1.0)]
        );
    }

    #[test]
    fn cancellation_reflows_the_remaining_bytes() {
        let mut transfers = FairTransfers::default();
        for id in 0..2 {
            transfers.submit(job(id, 0, Direction::Read, 0.0, (1.0, 0.0)));
        }
        assert_eq!(transfers.next_event(), Some(2.0));
        assert!(transfers.advance(1.0).is_empty());
        assert!(transfers.cancel(0) && !transfers.cancel(0));
        assert!(!transfers.has_link_jobs(1) && transfers.has_link_jobs(0));
        assert_eq!(transfers.next_event(), Some(1.5));
        assert_eq!(transfers.advance(10.0), [(1, 1.5)]);
        assert!(transfers.next_event().is_none() && transfers.now() == 10.0);
    }

    #[test]
    fn single_link_shares_are_equal() {
        let jobs = (0..4)
            .map(|id| job(id, id % 2, Direction::Read, 0.0, (1.0, 0.0)))
            .collect::<Vec<_>>();
        assert_eq!(fair_rates(&jobs, 0.0), [MB / 2.0; 4]);
    }

    #[test]
    fn waiters_hold_no_share() {
        let jobs = vec![
            job(0, 0, Direction::Read, 0.0, (1.0, 0.0)),
            job(1, 0, Direction::Read, 1.0, (1.0, 0.0)),
        ];
        assert_eq!(fair_rates(&jobs, 0.0), [MB, 0.0]);
        assert_eq!(fair_rates(&jobs, 1.0), [MB / 2.0; 2]);
    }

    #[test]
    fn a_completion_reflows_its_link() {
        // Two reads share 1 GB/s until the 1 MB one lands at 2; the other's
        // last MB then moves alone.
        let mut long = job(1, 0, Direction::Read, 0.0, (1.0, 0.0));
        long.bytes = 2.0 * MB;
        assert_eq!(
            run(vec![job(0, 0, Direction::Read, 0.0, (1.0, 0.0)), long]),
            [(0, 2.0), (1, 3.0)]
        );
    }

    #[test]
    fn rates_are_reused_until_a_first_byte() {
        let mut transfers = FairTransfers::default();
        let mut moving = job(0, 0, Direction::Read, 0.0, (1.0, 0.0));
        moving.bytes = 2.0 * MB;
        transfers.submit(moving);
        assert_eq!(transfers.next_event(), Some(2.0));
        // A waiter holds no share, so submitting it keeps the moving rate.
        transfers.submit(job(1, 0, Direction::Read, 1.0, (1.0, 0.0)));
        assert_eq!(transfers.next_event(), Some(1.0));
        assert!(transfers.advance(0.5).is_empty());
        assert_eq!(transfers.next_event(), Some(1.0));
        assert_eq!(transfers.rate_refreshes.get(), 1);
        // The waiter's first byte halves the moving job's share.
        assert!(transfers.advance(1.0).is_empty());
        assert_eq!(transfers.next_event(), Some(3.0));
        assert_eq!(transfers.rate_refreshes.get(), 2);
        assert_eq!(transfers.advance(f64::MAX), [(0, 3.0), (1, 3.0)]);
    }

    #[test]
    fn next_event_follows_same_instant_changes() {
        let mut transfers = FairTransfers::default();
        transfers.submit(job(0, 0, Direction::Read, 0.0, (1.0, 0.0)));
        // Nothing is due now: the first advance scans once and keeps the
        // event, so polls and later advances at this instant do no work.
        assert!(transfers.advance(0.0).is_empty());
        assert_eq!(transfers.cached_event.get(), Some(Some(1.0)));
        assert!(transfers.advance(0.0).is_empty());
        assert_eq!((transfers.steps, transfers.rate_refreshes.get()), (1, 1));
        // A peer on the same link halves the share; its removal restores it.
        transfers.submit(job(1, 0, Direction::Read, 0.0, (1.0, 0.0)));
        assert_eq!(transfers.next_event(), Some(2.0));
        assert!(transfers.cancel(1));
        assert_eq!(transfers.next_event(), Some(1.0));
        assert_eq!(transfers.advance(1.0), [(0, 1.0)]);
    }

    #[test]
    fn pool_capacity_a_capped_client_leaves_goes_to_other_clients() {
        // Client 1's nine jobs saturate its 3 GB/s link, so the 10 GB/s pool
        // still has 7 GB/s for client 0, which then hits its own 3 GB/s cap.
        let mut jobs = vec![job(0, 0, Direction::Read, 0.0, (3.0, 10.0))];
        jobs.extend((1..10).map(|id| job(id, 1, Direction::Read, 0.0, (3.0, 10.0))));
        let rates = fair_rates(&jobs, 0.0);
        assert_eq!(rates[0], 3.0 * MB);
        assert!(rates[1..].iter().all(|&rate| rate == 3.0 * MB / 9.0));
    }

    #[test]
    fn a_saturated_pool_is_shared_per_transfer() {
        // The pool binds before either client link, so all ten transfers get
        // 2 GB/s: client 1 moves 18 GB/s and client 0 moves 2 GB/s, as under
        // the previous min(client / n_client, pool / n_total) rule.
        let mut jobs = vec![job(0, 0, Direction::Read, 0.0, (32.0, 20.0))];
        jobs.extend((1..10).map(|id| job(id, 1, Direction::Read, 0.0, (32.0, 20.0))));
        assert_eq!(fair_rates(&jobs, 0.0), [2.0 * MB; 10]);
    }

    #[test]
    fn a_saturated_pool_is_shared_equally() {
        let jobs = (0..4)
            .map(|id| job(id, id, Direction::Read, 0.0, (32.0, 80.0)))
            .collect::<Vec<_>>();
        assert_eq!(fair_rates(&jobs, 0.0), [20.0 * MB; 4]);
    }

    #[test]
    fn source_and_destination_links_are_both_respected() {
        // Destination 10 receives from sources 1, 2 and 3, so each gets a
        // third. Source 1 also sends to destination 11 and gives that job the
        // two thirds its first job cannot use.
        let jobs = vec![
            pd_job(0, 1, 10),
            pd_job(1, 2, 10),
            pd_job(2, 3, 10),
            pd_job(3, 1, 11),
        ];
        let rates = fair_rates(&jobs, 0.0);
        assert!(rates[..3].iter().all(|&rate| rate == MB / 3.0));
        assert!((rates[3] - 2.0 * MB / 3.0).abs() < 1e-6);
    }

    #[test]
    fn transfers_into_one_destination_serialize_its_link() {
        // Two 1 MB transfers from different sources into one destination at
        // 1 GB/s take 2 ms together, not 1 ms each.
        assert_eq!(
            run(vec![pd_job(0, 1, 10), pd_job(1, 2, 10)]),
            [(0, 2.0), (1, 2.0)]
        );
        assert_eq!(
            run(vec![pd_job(0, 1, 10), pd_job(1, 2, 11)]),
            [(0, 1.0), (1, 1.0)]
        );
    }

    #[test]
    fn a_client_event_ignores_other_clients_under_an_unlimited_shared_rate() {
        let mut transfers = FairTransfers::default();
        let mut events = FxHashMap::default();
        let mut event = |transfers: &FairTransfers, client| {
            transfers.next_client_events(&mut events);
            events.get(&client).map(|event| event.at)
        };
        // Client 0: two 1 MB reads share 1 GB/s; client 1: one read ready at 0.5.
        for id in 0..2 {
            transfers.submit(job(id, 0, Direction::Read, 0.0, (1.0, 0.0)));
        }
        transfers.submit(job(2, 1, Direction::Read, 0.5, (1.0, 0.0)));
        assert_eq!(transfers.next_event(), Some(0.5));
        assert_eq!(event(&transfers, 0), Some(2.0));
        assert_eq!(event(&transfers, 1), Some(0.5));
        assert_eq!(event(&transfers, 7), None);
        // Each prediction is the boundary `advance` selects.
        assert_eq!(transfers.advance(1.0), []);
        assert_eq!(event(&transfers, 1), Some(1.5));
        assert_eq!(transfers.advance(2.0), [(2, 1.5), (0, 2.0), (1, 2.0)]);
        assert_eq!(event(&transfers, 0), None);
    }

    /// Progressive filling with a full scan per round, the definition the
    /// heap must reproduce exactly.
    fn reference_rates(jobs: &[Job]) -> Vec<f64> {
        let mut keys = Vec::<(usize, u64)>::new();
        let mut remaining = Vec::<f64>::new();
        let mut active = Vec::<usize>::new();
        let own = jobs
            .iter()
            .map(|job| {
                std::array::from_fn::<_, 2, _>(|slot| {
                    let link = job.links[slot];
                    let key = (slot, link.id);
                    let budget = keys.iter().position(|k| *k == key).unwrap_or_else(|| {
                        keys.push(key);
                        remaining.push(link.rate);
                        active.push(0);
                        keys.len() - 1
                    });
                    active[budget] += 1;
                    budget
                })
            })
            .collect::<Vec<_>>();
        let mut rates = vec![f64::NAN; jobs.len()];
        while let Some((bottleneck, share)) = (0..keys.len())
            .filter(|&budget| active[budget] > 0)
            .map(|budget| (budget, remaining[budget].max(0.0) / active[budget] as f64))
            .min_by(|left, right| left.1.total_cmp(&right.1))
        {
            for job in 0..jobs.len() {
                if !rates[job].is_nan() || !own[job].contains(&bottleneck) {
                    continue;
                }
                rates[job] = share;
                for budget in own[job] {
                    active[budget] -= 1;
                    if budget != bottleneck {
                        remaining[budget] -= share;
                    }
                }
            }
        }
        rates
    }

    #[test]
    fn heap_filling_matches_full_scan_filling() {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = |bound: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % bound
        };
        let mut heap_cases = 0;
        for case in 0..600 {
            // Alternate small instances (scan) with large ones (heap).
            let span = if case % 2 == 0 { 6 } else { 48 };
            let sources = 1 + next(span);
            let destinations = 1 + next(span);
            let jobs = (0..1 + next(4 * span))
                .map(|id| {
                    let link = |id: u64, gbps: u64| Link {
                        id,
                        rate: bytes_per_ms(1.0 + gbps as f64),
                    };
                    let source = next(sources);
                    let destination = next(destinations);
                    Job {
                        id,
                        direction: Direction::Write,
                        ready: 0.0,
                        bytes: MB,
                        links: [link(source, source % 3), link(destination, destination % 4)],
                    }
                })
                .collect::<Vec<_>>();
            let mut scratch = RateScratch::default();
            let mut rates = Vec::new();
            fair_rates_into(&jobs, 0.0, &mut scratch, &mut rates);
            heap_cases += usize::from(scratch.budgets.len() > SCAN_BUDGETS);
            let expected = reference_rates(&jobs);
            assert_eq!(
                rates.iter().map(|rate| rate.to_bits()).collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|rate| rate.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        assert!(heap_cases > 50, "only {heap_cases} instances used the heap");
    }
}
