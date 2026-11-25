//! Pipelined terminal input and resize requests.
//!
//! Rex guarantees that `client.key` and `block.event` execute in FIFO arrival order on one control
//! connection, so key presses, releases, pastes, pointer events, wheel turns and resizes can be
//! transmitted immediately without blocking the host loop for a full round trip per event.
//!
//! Responses are still tracked on the same connection: API errors surface as notices without
//! dropping the host, while transport errors or an 8-second stall on the oldest unanswered request
//! disconnect so failed input pauses and is never replayed.
use super::{Client, Shared};
use anyhow::{Result, anyhow};
use serde_json::Value;
use std::{
    collections::VecDeque,
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    task::Poll,
    time::{Duration, Instant},
};
use tokio::{
    sync::oneshot,
    time::{Instant as TokioInstant, Sleep},
};

/// Maximum unanswered input/resize requests in flight at once before applying backpressure.
pub(super) const MAX_IN_FLIGHT: usize = 64;

/// Maximum time the oldest unanswered input request may wait without progress before failing.
pub(super) const TIMEOUT: Duration = Duration::from_secs(8);

/// Upper clamp for a single RTT sample (in ms) so one pause cannot poison the EWMA for long.
const MAX_RTT_SAMPLE_MS: u32 = 5_000;

/// Record one measured round-trip duration into a lock-free host RTT slot.
///
/// Encoding: `0` means unmeasured; any positive value `v` encodes `v - 1` milliseconds.
pub(super) fn record_rtt(slot: &AtomicU32, elapsed: Duration) {
    let sample = elapsed.as_millis().min(u128::from(MAX_RTT_SAMPLE_MS)) as u32;
    let prev = slot.load(Ordering::Relaxed);
    let next = if prev == 0 {
        sample
    } else {
        let srtt = prev - 1;
        (srtt * 3 + sample) / 4
    };
    slot.store(next.saturating_add(1), Ordering::Relaxed);
}

struct InFlight {
    reply: oneshot::Receiver<String>,
    sent_at: Instant,
}

pub(super) struct Queue {
    in_flight: VecDeque<InFlight>,
    last_head_progress: Option<Instant>,
    timer: Pin<Box<Sleep>>,
    timeout: Duration,
    rtt_ms: Arc<AtomicU32>,
}

impl Default for Queue {
    fn default() -> Self {
        Self::with_rtt(Arc::new(AtomicU32::new(0)))
    }
}

impl Queue {
    pub(super) fn with_rtt(rtt_ms: Arc<AtomicU32>) -> Self {
        Self::with_timeout_and_rtt(TIMEOUT, rtt_ms)
    }

    fn with_timeout_and_rtt(timeout: Duration, rtt_ms: Arc<AtomicU32>) -> Self {
        Self {
            in_flight: VecDeque::new(),
            last_head_progress: None,
            timer: Box::pin(tokio::time::sleep(Duration::ZERO)),
            timeout,
            rtt_ms,
        }
    }

    #[cfg(test)]
    fn with_timeout(timeout: Duration) -> Self {
        Self::with_timeout_and_rtt(timeout, Arc::new(AtomicU32::new(0)))
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }

    pub fn is_full(&self) -> bool {
        self.in_flight.len() >= MAX_IN_FLIGHT
    }

    fn reset_head_timer(&mut self) {
        if let Some(head) = self.in_flight.front() {
            let base = self
                .last_head_progress
                .map_or(head.sent_at, |progress| progress.max(head.sent_at));
            self.timer
                .as_mut()
                .reset(TokioInstant::from_std(base + self.timeout));
        }
    }

    /// Transmit one input/resize request without waiting for its reply.
    /// Local serialization/size errors become a host notice; transport errors disconnect.
    pub async fn send(
        &mut self,
        control: &mut Client,
        shared: &Shared,
        method: &str,
        payload: Value,
    ) -> Result<()> {
        match control.transport().detach(method, payload).await {
            Ok(reply) => {
                let now = Instant::now();
                let was_empty = self.in_flight.is_empty();
                self.in_flight.push_back(InFlight {
                    reply,
                    sent_at: now,
                });
                if was_empty {
                    self.last_head_progress = None;
                    self.reset_head_timer();
                }
                Ok(())
            }
            Err(error @ (rexrs::Error::Json(_) | rexrs::Error::RequestTooLarge { .. })) => {
                shared.update(|m| m.notice = Some(format!("{error:#}")));
                Ok(())
            }
            Err(error) => Err(error.into()),
        }
    }

    /// Wait for any in-flight input reply or for the oldest unanswered request to time out.
    /// Cancel-safe: an entry is removed only in the poll that returns it.
    pub async fn finished(&mut self, shared: &Shared) -> Result<()> {
        if self.in_flight.is_empty() {
            return std::future::pending().await;
        }
        let outcome = std::future::poll_fn(|cx| {
            // Check replies first so an arrived answer is never lost to a coincident deadline.
            for (index, entry) in self.in_flight.iter_mut().enumerate() {
                if let Poll::Ready(reply) = Pin::new(&mut entry.reply).poll(cx) {
                    return Poll::Ready(Ok((index, reply)));
                }
            }
            if self.timer.as_mut().poll(cx).is_ready() {
                return Poll::Ready(Err(()));
            }
            Poll::Pending
        })
        .await;
        let Ok((index, reply)) = outcome else {
            return Err(anyhow!("Rex request timed out; action was not retried"));
        };
        let removed = self.in_flight.remove(index);
        if self.in_flight.is_empty() {
            self.last_head_progress = None;
        } else if index == 0 {
            self.last_head_progress = Some(Instant::now());
            self.reset_head_timer();
        }
        let text = reply
            .map_err(|_| anyhow!("Rex control connection closed while awaiting input reply"))?;
        match crate::transport::detached_result::<Value>(&text) {
            Ok(_) => {
                if let Some(entry) = removed {
                    record_rtt(&self.rtt_ms, entry.sent_at.elapsed());
                }
                Ok(())
            }
            Err(error @ rexrs::Error::Api(_)) => {
                if let Some(entry) = removed {
                    record_rtt(&self.rtt_ms, entry.sent_at.elapsed());
                }
                shared.update(|m| m.notice = Some(format!("{error:#}")));
                Ok(())
            }
            Err(error) => Err(error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::HostModel;
    use std::sync::{Arc, Mutex};

    fn shared() -> Shared {
        let (wake, _rx) = flume::bounded(1);
        Shared {
            model: Arc::new(Mutex::new(HostModel::default())),
            wake,
        }
    }

    fn push(queue: &mut Queue, sent_at: Instant) -> oneshot::Sender<String> {
        let (tx, rx) = oneshot::channel();
        let was_empty = queue.in_flight.is_empty();
        queue.in_flight.push_back(InFlight { reply: rx, sent_at });
        if was_empty {
            queue.last_head_progress = None;
            queue.reset_head_timer();
        }
        tx
    }

    #[tokio::test]
    async fn api_errors_notify_without_disconnecting_while_success_is_quiet() {
        let shared = shared();
        let mut queue = Queue::with_timeout(Duration::from_secs(5));
        let now = Instant::now();
        let ok_tx = push(&mut queue, now);
        let err_tx = push(&mut queue, now);
        assert_eq!(queue.in_flight(), 2);

        ok_tx
            .send(
                r#"{"type":"response","id":"detached:1","seq":1,"payload":{"handled":false}}"#
                    .into(),
            )
            .unwrap();
        queue.finished(&shared).await.unwrap();
        assert_eq!(queue.in_flight(), 1);
        assert_eq!(shared.model.lock().unwrap().notice, None);

        err_tx
            .send(r#"{"type":"response","id":"detached:2","seq":2,"error":{"code":"not_found","status":404,"message":"block gone"}}"#.into())
            .unwrap();
        queue.finished(&shared).await.unwrap();
        assert_eq!(queue.in_flight(), 0);
        assert!(
            shared
                .model
                .lock()
                .unwrap()
                .notice
                .as_deref()
                .is_some_and(|n| n.contains("block gone"))
        );
    }

    #[tokio::test]
    async fn head_progress_resets_next_deadline_but_tail_reply_does_not() {
        let shared = shared();
        let mut queue = Queue::with_timeout(Duration::from_millis(80));
        let now = Instant::now();
        let first = push(&mut queue, now);
        let second = push(&mut queue, now);

        // Second finishing out of order (e.g. an unordered resize) must not extend a hung head.
        tokio::time::sleep(Duration::from_millis(45)).await;
        second
            .send(r#"{"type":"response","id":"detached:2","seq":2,"payload":{}}"#.into())
            .unwrap();
        queue.finished(&shared).await.unwrap();
        assert_eq!(queue.in_flight(), 1);

        // First still times out near its original 80ms budget.
        let started = Instant::now();
        let err = queue.finished(&shared).await.unwrap_err();
        assert!(err.to_string().contains("timed out"), "{err:#}");
        assert!(
            started.elapsed() < Duration::from_millis(75),
            "tail completion must not reset head deadline: {:?}",
            started.elapsed()
        );
        drop(first);

        // Head finishing after a delay grants the next queued item a fresh timeout window.
        let mut queue = Queue::with_timeout(Duration::from_millis(80));
        let now = Instant::now();
        let first = push(&mut queue, now);
        let second = push(&mut queue, now);
        tokio::time::sleep(Duration::from_millis(55)).await;
        first
            .send(r#"{"type":"response","id":"detached:3","seq":3,"payload":{}}"#.into())
            .unwrap();
        queue.finished(&shared).await.unwrap();

        tokio::time::sleep(Duration::from_millis(45)).await;
        second
            .send(r#"{"type":"response","id":"detached:4","seq":4,"payload":{}}"#.into())
            .unwrap();
        queue
            .finished(&shared)
            .await
            .expect("second received fresh timeout after head completed");
    }

    #[tokio::test]
    async fn dropped_sender_returns_transport_error() {
        let shared = shared();
        let mut queue = Queue::with_timeout(Duration::from_secs(5));
        let tx = push(&mut queue, Instant::now());
        drop(tx);
        let err = queue.finished(&shared).await.unwrap_err();
        assert!(err.to_string().contains("closed"), "{err:#}");
    }

    #[test]
    fn ewma_rtt_seeds_smooths_and_caps_extremes() {
        let slot = AtomicU32::new(0);
        record_rtt(&slot, Duration::from_millis(200));
        assert_eq!(slot.load(Ordering::Relaxed), 201);
        record_rtt(&slot, Duration::from_millis(120));
        // (200 * 3 + 120) / 4 = 180 -> stored as 181
        assert_eq!(slot.load(Ordering::Relaxed), 181);
        record_rtt(&slot, Duration::from_secs(30));
        // Clamped to 5000ms: (180 * 3 + 5000) / 4 = 1385 -> stored as 1386
        assert_eq!(slot.load(Ordering::Relaxed), 1386);
    }
}
