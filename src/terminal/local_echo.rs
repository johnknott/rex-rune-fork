//! Mosh-style speculative local echo engine.
//!
//! Pure, render-only prediction state: it never mutates Ghostty's terminal state and never sends
//! network requests. It tracks pending printable ASCII insertions and plain Backspace edits on
//! the active cursor row, confirms them against authoritative server frames using Mosh's epoch
//! model (so password prompts and non-echoing modes such as `vim` normal mode never show ghost
//! characters), and produces a compact per-row visual overlay for the terminal renderer.
use super::Frame;
use crate::settings::terminal::LocalEcho;
use serde_json::Value;
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

/// RTT at or above which `LocalEcho::Adaptive` engages prediction.
pub const ADAPTIVE_ON_MS: u32 = 30;
/// RTT below which `LocalEcho::Adaptive` disengages prediction (hysteresis).
pub const ADAPTIVE_OFF_MS: u32 = 20;
/// RTT at or above which `LocalEcho::Adaptive` underlines pending predictions immediately.
pub const UNDERLINE_RTT_MS: u32 = 80;
/// Wait time after which an unconfirmed prediction is underlined in `LocalEcho::Adaptive`.
pub const UNDERLINE_AGE: Duration = Duration::from_millis(100);
/// Minimum expiry window before an unanswered prediction is silently cleared.
const MIN_EXPIRE: Duration = Duration::from_millis(1000);
/// Maximum expiry window before an unanswered prediction is silently cleared.
const MAX_EXPIRE: Duration = Duration::from_millis(4000);
/// Hard cap on queued predictions so a held key cannot grow unboundedly.
const MAX_PENDING: usize = 128;

/// One cell of a local-echo overlay on the edited row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PredictedCell {
    /// `Some(ch)` draws `ch`; `None` blanks the underlying cell (predicted Backspace).
    pub ch: Option<char>,
    /// Whether this cell should carry an underline cue as an unconfirmed prediction.
    pub underline: bool,
}

/// Visual overlay computed from pending confirmed-epoch predictions against a server [`Frame`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Overlay {
    /// Row index in `Frame::cells`.
    pub row: u16,
    /// First column covered by `cells`.
    pub start_col: u16,
    /// Contiguous cells starting at `start_col`.
    pub cells: Vec<PredictedCell>,
    /// Predicted cursor cell `(col, row)`, replacing `Frame::cursor` visually.
    pub cursor: (u16, u16),
}

/// A local input event classified for the prediction engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputAction {
    /// Ignore completely (key release, pointer motion, wheel, local selection).
    Ignore,
    /// Printable single-width ASCII character press/repeat.
    Insert(char),
    /// Plain Backspace press/repeat.
    Backspace,
    /// Any other terminal input (Enter, Tab, Escape, arrows, Ctrl/Alt chords, paste, click):
    /// clear pending predictions and start a fresh unconfirmed epoch.
    Reset,
}

/// Classify a `client.key` JSON payload into a local-echo [`InputAction`].
pub fn classify_key(payload: &Value) -> InputAction {
    let action = payload["action"].as_str().unwrap_or("");
    if action == "release" {
        return InputAction::Ignore;
    }
    if payload["composing"].as_bool().unwrap_or(false) {
        return InputAction::Reset;
    }
    let mods = payload["mods"].as_str().unwrap_or("");
    let consumed = payload["consumed_mods"].as_str().unwrap_or("");
    let has = |list: &str, name: &str| list.split('+').any(|m| m == name);
    if has(mods, "ctrl") || has(mods, "super") {
        return InputAction::Reset;
    }
    if has(mods, "alt") && !has(consumed, "alt") {
        return InputAction::Reset;
    }
    if payload["key"].as_str() == Some("backspace") && !has(mods, "shift") && !has(mods, "alt") {
        return InputAction::Backspace;
    }
    let text = payload["text"].as_str().unwrap_or("");
    let mut chars = text.chars();
    if let Some(ch) = chars.next()
        && chars.next().is_none()
        && ch.is_ascii()
        && !ch.is_ascii_control()
    {
        return InputAction::Insert(ch);
    }
    InputAction::Reset
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PendingOp {
    Insert {
        ch: char,
        at: (u16, u16),
        after: (u16, u16),
        epoch: u64,
        sent_at: Instant,
    },
    Backspace {
        from: (u16, u16),
        after: (u16, u16),
        sent_at: Instant,
    },
}

impl PendingOp {
    fn start(&self) -> (u16, u16) {
        match *self {
            Self::Insert { at, .. } => at,
            Self::Backspace { from, .. } => from,
        }
    }
    fn after(&self) -> (u16, u16) {
        match *self {
            Self::Insert { after, .. } | Self::Backspace { after, .. } => after,
        }
    }
    fn sent_at(&self) -> Instant {
        match *self {
            Self::Insert { sent_at, .. } | Self::Backspace { sent_at, .. } => sent_at,
        }
    }
}

/// Per-pane Mosh-style speculative local echo state machine.
#[derive(Debug)]
pub struct Predictor {
    pending: VecDeque<PendingOp>,
    /// Highest epoch confirmed by an authoritative server echo of an `Insert`.
    confirmed_epoch: u64,
    /// Current typing epoch; predictions display only once `confirmed_epoch == tentative_epoch`.
    tentative_epoch: u64,
    /// Hysteresis latch for `LocalEcho::Adaptive`.
    adaptive_active: bool,
    /// Last observed server frame dimensions and screen kind, to detect transitions.
    last_size: Option<(u16, u16)>,
    last_alt_screen: Option<bool>,
    last_server_cursor: Option<(u16, u16)>,
    /// Oldest `sent_at` for which crossing `UNDERLINE_AGE` has already triggered one UI redraw.
    underline_notified: Option<Instant>,
}

impl Default for Predictor {
    fn default() -> Self {
        Self {
            pending: VecDeque::new(),
            confirmed_epoch: 0,
            tentative_epoch: 1,
            adaptive_active: false,
            last_size: None,
            last_alt_screen: None,
            last_server_cursor: None,
            underline_notified: None,
        }
    }
}

impl Predictor {
    /// Reset all predictions and start a fresh unconfirmed epoch (e.g. reconnect or mode change).
    pub fn reset(&mut self) {
        self.pending.clear();
        self.underline_notified = None;
        self.bump_epoch();
        self.last_size = None;
        self.last_alt_screen = None;
        self.last_server_cursor = None;
    }

    /// Clear pending predictions and require fresh echo confirmation before displaying again.
    pub fn reset_epoch(&mut self) {
        self.pending.clear();
        self.underline_notified = None;
        self.bump_epoch();
    }

    fn bump_epoch(&mut self) {
        if self.tentative_epoch <= self.confirmed_epoch {
            self.tentative_epoch = self.confirmed_epoch.saturating_add(1);
        }
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Update the `LocalEcho::Adaptive` RTT hysteresis latch and return whether prediction is
    /// currently enabled for `mode` and `rtt_ms` (`None` when unmeasured).
    fn enabled(&mut self, mode: LocalEcho, rtt_ms: Option<u32>) -> bool {
        match mode {
            LocalEcho::Off => {
                self.adaptive_active = false;
                false
            }
            LocalEcho::Always => true,
            LocalEcho::Adaptive => {
                if let Some(rtt) = rtt_ms {
                    if rtt >= ADAPTIVE_ON_MS {
                        self.adaptive_active = true;
                    } else if rtt < ADAPTIVE_OFF_MS {
                        self.adaptive_active = false;
                    }
                }
                self.adaptive_active
            }
        }
    }

    /// Record a classified local input event sent to the pane.
    pub fn on_input(
        &mut self,
        action: InputAction,
        frame: Option<&Frame>,
        mode: LocalEcho,
        rtt_ms: Option<u32>,
        now: Instant,
    ) {
        if !self.enabled(mode, rtt_ms) {
            if !self.pending.is_empty() {
                self.reset_epoch();
            }
            return;
        }
        match action {
            InputAction::Ignore => {}
            InputAction::Reset => {
                self.reset_epoch();
            }
            InputAction::Insert(ch) => {
                let Some(frame) = frame.filter(|f| Self::frame_predictable(f)) else {
                    self.reset_epoch();
                    return;
                };
                if self.pending.len() >= MAX_PENDING {
                    return;
                }
                let Some((col, row)) = self.predicted_cursor(frame) else {
                    self.reset_epoch();
                    return;
                };
                // Never predict past or onto the right margin where wrap behavior varies by app.
                if usize::from(col) + 1 >= usize::from(frame.cols) {
                    self.reset_epoch();
                    return;
                }
                self.pending.push_back(PendingOp::Insert {
                    ch,
                    at: (col, row),
                    after: (col + 1, row),
                    epoch: self.tentative_epoch,
                    sent_at: now,
                });
            }
            InputAction::Backspace => {
                let Some(frame) = frame.filter(|f| Self::frame_predictable(f)) else {
                    self.reset_epoch();
                    return;
                };
                if self.pending.len() >= MAX_PENDING {
                    return;
                }
                let Some((col, row)) = self.predicted_cursor(frame) else {
                    self.reset_epoch();
                    return;
                };
                if col == 0 {
                    self.reset_epoch();
                    return;
                }
                self.pending.push_back(PendingOp::Backspace {
                    from: (col, row),
                    after: (col - 1, row),
                    sent_at: now,
                });
            }
        }
    }

    fn frame_predictable(frame: &Frame) -> bool {
        frame.viewport_active
            && frame.copy.is_none()
            && frame.cursor.is_some()
            && frame.cols > 1
            && frame.rows > 0
    }

    fn predicted_cursor(&self, frame: &Frame) -> Option<(u16, u16)> {
        self.pending
            .back()
            .map(PendingOp::after)
            .or(frame.cursor)
            .filter(|&(x, y)| x < frame.cols && y < frame.rows)
    }

    /// Reconcile pending predictions against a freshly arrived authoritative server [`Frame`].
    pub fn reconcile(&mut self, frame: &Frame, mode: LocalEcho, rtt_ms: Option<u32>, now: Instant) {
        let size = (frame.cols, frame.rows);
        let structural_change = self.last_size.is_some_and(|prev| prev != size)
            || self
                .last_alt_screen
                .is_some_and(|prev| prev != frame.alt_screen);
        self.last_size = Some(size);
        self.last_alt_screen = Some(frame.alt_screen);

        if !self.enabled(mode, rtt_ms) || structural_change || !Self::frame_predictable(frame) {
            self.reset_epoch();
            self.last_server_cursor = frame.cursor;
            return;
        }

        let Some((cx, cy)) = frame.cursor else {
            self.reset_epoch();
            self.last_server_cursor = None;
            return;
        };

        if self.pending.is_empty() {
            // A server-driven jump to another row (e.g. command output followed by a new prompt,
            // possibly a password prompt) starts a fresh unconfirmed epoch.
            if self.last_server_cursor.is_some_and(|(_, py)| py != cy) {
                self.bump_epoch();
            }
            self.last_server_cursor = Some((cx, cy));
            return;
        }

        self.last_server_cursor = Some((cx, cy));

        // Find the longest prefix `1..=pending.len()` whose net result matches `frame`.
        let matched = self.longest_matching_prefix(frame, (cx, cy));
        if matched > 0 {
            let mut confirmed_insert_epoch = None;
            for _ in 0..matched {
                if let Some(op) = self.pending.pop_front()
                    && let PendingOp::Insert { epoch, .. } = op
                {
                    confirmed_insert_epoch = Some(epoch);
                }
            }
            if let Some(epoch) = confirmed_insert_epoch {
                self.confirmed_epoch = self.confirmed_epoch.max(epoch);
            }
            self.expire_stale(rtt_ms, now);
            return;
        }

        // No prefix matched. If the server cursor hasn't moved from the oldest pending op's origin
        // (e.g. an unrelated clock/status redraw elsewhere on screen before our key arrived), keep
        // waiting until either the echo arrives or the prediction expires.
        let head = self.pending.front().expect("non-empty pending");
        let head_start = head.start();
        if (cx, cy) == head_start {
            self.expire_stale(rtt_ms, now);
            return;
        }

        // Cursor moved somewhere that doesn't match any prefix (vim normal mode motion, tab
        // completion, prompt jump, or program output): immediately roll back and unconfirm.
        self.reset_epoch();
    }

    /// Check whether a prefix `pending[0..k]` was applied by the server in `frame`.
    fn longest_matching_prefix(&self, frame: &Frame, server_cursor: (u16, u16)) -> usize {
        let Some(first) = self.pending.front() else {
            return 0;
        };
        let row = first.start().1;
        if server_cursor.1 != row {
            return 0;
        }
        let Some(row_cells) = frame.cells.get(usize::from(row)) else {
            return 0;
        };

        // Track surviving characters written by Insert ops in `0..k`, keyed by column relative to
        // a small simulated buffer of `Option<char>` (Some(ch) = must match verbatim).
        let cols = usize::from(frame.cols);
        let mut expected: Vec<Option<char>> = vec![None; cols];
        let mut best = 0;

        for (index, op) in self.pending.iter().enumerate() {
            match *op {
                PendingOp::Insert { ch, at, .. } => {
                    if at.1 != row || usize::from(at.0) >= cols {
                        break;
                    }
                    let col = usize::from(at.0);
                    // Shift previously expected tail right by 1 for mid-line insert consistency.
                    expected.copy_within(col..cols - 1, col + 1);
                    expected[col] = Some(ch);
                }
                PendingOp::Backspace { from, .. } => {
                    if from.1 != row || from.0 == 0 || usize::from(from.0) > cols {
                        break;
                    }
                    let col = usize::from(from.0 - 1);
                    expected.copy_within(col + 1..cols, col);
                    expected[cols - 1] = None;
                }
            }
            if op.after() == server_cursor {
                let all_match = expected.iter().enumerate().all(|(col, want)| {
                    want.is_none_or(|ch| row_cells.get(col).is_some_and(|c| cell_matches(c, ch)))
                });
                if all_match {
                    best = index + 1;
                }
            }
        }
        best
    }

    fn expire_after(rtt_ms: Option<u32>) -> Duration {
        let scaled = rtt_ms.map_or(MIN_EXPIRE, |ms| Duration::from_millis(u64::from(ms) * 3));
        scaled.clamp(MIN_EXPIRE, MAX_EXPIRE)
    }

    /// Expire stale unanswered predictions; returns true if anything was cleared.
    pub fn expire_stale(&mut self, rtt_ms: Option<u32>, now: Instant) -> bool {
        let Some(oldest) = self.pending.front().map(PendingOp::sent_at) else {
            return false;
        };
        if now.saturating_duration_since(oldest) >= Self::expire_after(rtt_ms) {
            self.reset_epoch();
            true
        } else {
            false
        }
    }

    /// Advance time-driven state (stale prediction expiry and adaptive underline threshold).
    /// Returns `(needs_redraw, next_wake_instant)`.
    pub fn poll(
        &mut self,
        mode: LocalEcho,
        rtt_ms: Option<u32>,
        now: Instant,
    ) -> (bool, Option<Instant>) {
        if self.expire_stale(rtt_ms, now) {
            return (true, None);
        }
        let Some(oldest) = self.pending.front().map(PendingOp::sent_at) else {
            self.underline_notified = None;
            return (false, None);
        };
        let expire_at = oldest + Self::expire_after(rtt_ms);
        if mode == LocalEcho::Adaptive
            && self.confirmed_epoch >= self.tentative_epoch
            && rtt_ms.is_none_or(|ms| ms < UNDERLINE_RTT_MS)
        {
            let underline_at = oldest + UNDERLINE_AGE;
            if now >= underline_at {
                let changed = self.underline_notified != Some(oldest);
                self.underline_notified = Some(oldest);
                return (changed, Some(expire_at));
            }
            return (false, Some(underline_at.min(expire_at)));
        }
        (false, Some(expire_at))
    }

    /// Next instant when either underline state or stale expiry changes, so the UI can wake once.
    pub fn next_wake(&self, mode: LocalEcho, rtt_ms: Option<u32>, now: Instant) -> Option<Instant> {
        let oldest = self.pending.front()?.sent_at();
        let expire_at = oldest + Self::expire_after(rtt_ms);
        if mode == LocalEcho::Adaptive
            && self.confirmed_epoch >= self.tentative_epoch
            && rtt_ms.is_none_or(|ms| ms < UNDERLINE_RTT_MS)
        {
            let underline_at = oldest + UNDERLINE_AGE;
            if underline_at > now {
                return Some(underline_at.min(expire_at));
            }
        }
        Some(expire_at)
    }

    /// Build the render-time [`Overlay`] for `frame`, or `None` if nothing should be drawn.
    pub fn overlay(
        &self,
        frame: &Frame,
        mode: LocalEcho,
        rtt_ms: Option<u32>,
        now: Instant,
    ) -> Option<Overlay> {
        if self.pending.is_empty() || !Self::frame_predictable(frame) {
            return None;
        }
        let active = match mode {
            LocalEcho::Off => false,
            LocalEcho::Always => true,
            LocalEcho::Adaptive => self.adaptive_active,
        };
        if !active {
            return None;
        }
        // Only display once this epoch has been confirmed by a real server echo (password safety).
        if self.confirmed_epoch < self.tentative_epoch {
            return None;
        }

        let (cx, cy) = frame.cursor?;
        let base_row = frame.cells.get(usize::from(cy))?;
        let cols = usize::from(frame.cols);
        if base_row.len() < cols {
            return None;
        }
        // Pending ops must target the same row as the current server cursor.
        if self.pending.front()?.start().1 != cy {
            return None;
        }

        let real_char = |col: usize| -> Option<char> {
            let cell = &base_row[col];
            if cell.style.faint || cell.spacer_tail || cell.text.is_empty() || cell.text == " " {
                None
            } else {
                let mut chars = cell.text.chars();
                chars.next().filter(|_| chars.next().is_none())
            }
        };

        let mut chars: Vec<Option<char>> = (0..cols).map(real_char).collect();
        let mut predicted: Vec<bool> = vec![false; cols];
        let mut cur_x = usize::from(cx);

        let underline_all = match mode {
            LocalEcho::Off => false,
            LocalEcho::Always => true,
            LocalEcho::Adaptive => {
                rtt_ms.is_some_and(|ms| ms >= UNDERLINE_RTT_MS)
                    || self.pending.front().is_some_and(|op| {
                        now.saturating_duration_since(op.sent_at()) >= UNDERLINE_AGE
                    })
            }
        };

        for op in &self.pending {
            match *op {
                PendingOp::Insert { ch, .. } => {
                    if cur_x + 1 >= cols {
                        break;
                    }
                    // Shift non-faint tail text right by 1 for mid-line edits, stopping before
                    // a double-space gap (e.g. RPROMPT) or wide cells; faint autosuggestions
                    // are overwritten in place.
                    if predicted[cur_x] || chars[cur_x].is_some() {
                        let end = tail_end(base_row, &chars, &predicted, cur_x).min(cols - 1);
                        if end > cur_x {
                            chars.copy_within(cur_x..end, cur_x + 1);
                            predicted.copy_within(cur_x..end, cur_x + 1);
                        }
                    }
                    chars[cur_x] = Some(ch);
                    predicted[cur_x] = true;
                    cur_x += 1;
                }
                PendingOp::Backspace { .. } => {
                    if cur_x == 0 {
                        break;
                    }
                    cur_x -= 1;
                    let end = tail_end(base_row, &chars, &predicted, cur_x + 1);
                    if end > cur_x + 1 {
                        chars.copy_within(cur_x + 1..end, cur_x);
                        predicted.copy_within(cur_x + 1..end, cur_x);
                        chars[end - 1] = None;
                        predicted[end - 1] = false;
                    } else {
                        chars[cur_x] = None;
                        predicted[cur_x] = false;
                    }
                }
            }
        }

        let mut first_diff = None;
        let mut last_diff = None;
        for col in 0..cols {
            let changed = chars[col] != real_char(col) || predicted[col];
            if changed {
                first_diff.get_or_insert(col);
                last_diff = Some(col);
            }
        }

        let (start_col, cells) = match (first_diff, last_diff) {
            (Some(a), Some(b)) => (
                a as u16,
                (a..=b)
                    .map(|col| PredictedCell {
                        ch: chars[col],
                        underline: underline_all && predicted[col],
                    })
                    .collect(),
            ),
            _ if cur_x != usize::from(cx) => (cur_x as u16, Vec::new()),
            _ => return None,
        };

        Some(Overlay {
            row: cy,
            start_col,
            cells,
            cursor: (cur_x as u16, cy),
        })
    }
}

fn cell_matches(cell: &super::Cell, ch: char) -> bool {
    if ch == ' ' {
        cell.text.is_empty() || cell.text == " "
    } else {
        cell.text.len() == 1 && cell.text.starts_with(ch)
    }
}

/// End column (exclusive) of the contiguous editable tail starting at `from`, stopping before a
/// run of two blank unpredicted cells (so right prompts like Starship `RPROMPT` never shift) or
/// a wide character cell.
fn tail_end(
    base_row: &[super::Cell],
    chars: &[Option<char>],
    predicted: &[bool],
    from: usize,
) -> usize {
    let cols = chars.len();
    let mut col = from;
    while col < cols {
        let cell = &base_row[col];
        if cell.spacer_tail || base_row.get(col + 1).is_some_and(|next| next.spacer_tail) {
            break;
        }
        let empty = chars[col].is_none() && !predicted[col];
        if empty {
            let next_empty = col + 1 >= cols || (chars[col + 1].is_none() && !predicted[col + 1]);
            if next_empty {
                break;
            }
        }
        col += 1;
    }
    col
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminal::Renderer;
    use ghostty_vt::{Terminal, TerminalOptions};
    use serde_json::json;

    fn make_terminal(cols: u16, rows: u16, initial: &[u8]) -> (Terminal, Renderer, Frame) {
        let mut term = Terminal::new(TerminalOptions {
            cols,
            rows,
            ..Default::default()
        })
        .unwrap();
        term.vt_write(initial);
        let mut renderer = Renderer::new().unwrap();
        let frame = renderer.frame(&term).unwrap();
        (term, renderer, frame)
    }

    fn key(name: &str, text: &str, mods: &str, consumed: &str, action: &str) -> Value {
        json!({
            "key": name,
            "mods": mods,
            "consumed_mods": consumed,
            "text": text,
            "unshifted_codepoint": text.chars().next().map_or(0, |c| c as u32),
            "composing": false,
            "action": action,
        })
    }

    #[test]
    fn key_classification_accepts_printable_ascii_and_plain_backspace_only() {
        assert_eq!(
            classify_key(&key("key_a", "a", "", "", "press")),
            InputAction::Insert('a')
        );
        assert_eq!(
            classify_key(&key("key_a", "A", "shift", "shift", "repeat")),
            InputAction::Insert('A')
        );
        assert_eq!(
            classify_key(&key("key_a", "", "", "", "release")),
            InputAction::Ignore
        );
        assert_eq!(
            classify_key(&key("backspace", "", "", "", "press")),
            InputAction::Backspace
        );
        assert_eq!(
            classify_key(&key("backspace", "", "ctrl", "", "press")),
            InputAction::Reset
        );
        assert_eq!(
            classify_key(&key("key_c", "", "ctrl", "", "press")),
            InputAction::Reset
        );
        assert_eq!(
            classify_key(&key("key_b", "b", "alt", "", "press")),
            InputAction::Reset
        );
        // Option-composed ASCII on macOS (e.g. `@` via Option+L on German layout).
        assert_eq!(
            classify_key(&key("key_l", "@", "alt", "alt", "press")),
            InputAction::Insert('@')
        );
        assert_eq!(
            classify_key(&key("enter", "\n", "", "", "press")),
            InputAction::Reset
        );
        assert_eq!(
            classify_key(&key("tab", "\t", "", "", "press")),
            InputAction::Reset
        );
        assert_eq!(
            classify_key(&key("escape", "", "", "", "press")),
            InputAction::Reset
        );
    }

    #[test]
    fn password_prompt_never_shows_predictions_while_normal_echo_confirms_and_predicts() {
        let (mut term, mut renderer, frame0) = make_terminal(40, 4, b"$ ");
        let mut engine = Predictor::default();
        let t0 = Instant::now();
        engine.reconcile(&frame0, LocalEcho::Adaptive, Some(220), t0);

        // First char is tentative (unconfirmed epoch) -> not displayed yet.
        engine.on_input(
            InputAction::Insert('g'),
            Some(&frame0),
            LocalEcho::Adaptive,
            Some(220),
            t0,
        );
        engine.on_input(
            InputAction::Insert('i'),
            Some(&frame0),
            LocalEcho::Adaptive,
            Some(220),
            t0 + Duration::from_millis(50),
        );
        assert_eq!(
            engine.overlay(&frame0, LocalEcho::Adaptive, Some(220), t0),
            None,
            "unconfirmed epoch stays invisible (password safety)"
        );

        // Server echoes 'g' -> confirms epoch! 'i' immediately becomes visible as a prediction.
        term.vt_write(b"g");
        let frame1 = renderer.frame(&term).unwrap();
        let t1 = t0 + Duration::from_millis(220);
        engine.reconcile(&frame1, LocalEcho::Adaptive, Some(220), t1);
        let ov = engine
            .overlay(&frame1, LocalEcho::Adaptive, Some(220), t1)
            .expect("'i' should appear immediately once epoch confirms");
        assert_eq!(ov.row, 0);
        assert_eq!(ov.start_col, 3);
        assert_eq!(ov.cursor, (4, 0));
        assert_eq!(
            ov.cells,
            vec![PredictedCell {
                ch: Some('i'),
                underline: true,
            }]
        );

        // Next character 't' appears instantaneously at 0 ms!
        engine.on_input(
            InputAction::Insert('t'),
            Some(&frame1),
            LocalEcho::Adaptive,
            Some(220),
            t1 + Duration::from_millis(10),
        );
        let ov = engine
            .overlay(
                &frame1,
                LocalEcho::Adaptive,
                Some(220),
                t1 + Duration::from_millis(10),
            )
            .unwrap();
        assert_eq!(ov.cursor, (5, 0));
        assert_eq!(
            ov.cells.iter().map(|c| c.ch).collect::<Vec<_>>(),
            vec![Some('i'), Some('t')]
        );

        // Server catches up with both 'i' and 't' -> overlay disappears cleanly.
        term.vt_write(b"it");
        let frame2 = renderer.frame(&term).unwrap();
        engine.reconcile(
            &frame2,
            LocalEcho::Adaptive,
            Some(220),
            t1 + Duration::from_millis(220),
        );
        assert!(!engine.has_pending());
        assert_eq!(
            engine.overlay(
                &frame2,
                LocalEcho::Adaptive,
                Some(220),
                t1 + Duration::from_millis(220)
            ),
            None
        );

        // Pressing Enter resets and starts a fresh unconfirmed epoch (e.g. `sudo` password prompt).
        engine.on_input(
            InputAction::Reset,
            Some(&frame2),
            LocalEcho::Adaptive,
            Some(220),
            t1 + Duration::from_millis(300),
        );
        term.vt_write(b"\r\nPassword: ");
        let pwd_frame = renderer.frame(&term).unwrap();
        let t2 = t1 + Duration::from_millis(500);
        engine.reconcile(&pwd_frame, LocalEcho::Adaptive, Some(220), t2);
        for ch in ['s', 'e', 'c', 'r', 'e', 't'] {
            engine.on_input(
                InputAction::Insert(ch),
                Some(&pwd_frame),
                LocalEcho::Adaptive,
                Some(220),
                t2,
            );
            assert_eq!(
                engine.overlay(&pwd_frame, LocalEcho::Adaptive, Some(220), t2),
                None,
                "password characters must never appear locally"
            );
        }
    }

    #[test]
    fn backspace_on_predicted_and_coalesced_frames_reconciles_cleanly() {
        let (mut term, mut renderer, frame0) = make_terminal(40, 4, b"$ ");
        let mut engine = Predictor::default();
        let t0 = Instant::now();
        engine.reconcile(&frame0, LocalEcho::Always, Some(150), t0);

        // Confirm epoch with 'a'.
        engine.on_input(
            InputAction::Insert('a'),
            Some(&frame0),
            LocalEcho::Always,
            Some(150),
            t0,
        );
        term.vt_write(b"a");
        let frame1 = renderer.frame(&term).unwrap();
        engine.reconcile(&frame1, LocalEcho::Always, Some(150), t0);

        // Type 'b', 'c', then Backspace before server replies.
        engine.on_input(
            InputAction::Insert('b'),
            Some(&frame1),
            LocalEcho::Always,
            Some(150),
            t0,
        );
        engine.on_input(
            InputAction::Insert('c'),
            Some(&frame1),
            LocalEcho::Always,
            Some(150),
            t0,
        );
        engine.on_input(
            InputAction::Backspace,
            Some(&frame1),
            LocalEcho::Always,
            Some(150),
            t0,
        );

        let ov = engine
            .overlay(&frame1, LocalEcho::Always, Some(150), t0)
            .unwrap();
        assert_eq!(ov.start_col, 3);
        assert_eq!(ov.cursor, (4, 0));
        assert_eq!(
            ov.cells,
            vec![PredictedCell {
                ch: Some('b'),
                underline: true,
            }]
        );

        // Server coalesces 'b', 'c', Backspace into one frame showing "$ ab" at col 4.
        term.vt_write(b"bc\x08 \x08");
        let frame2 = renderer.frame(&term).unwrap();
        engine.reconcile(
            &frame2,
            LocalEcho::Always,
            Some(150),
            t0 + Duration::from_millis(150),
        );
        assert!(
            !engine.has_pending(),
            "coalesced insert+backspace prefix must settle"
        );
        assert_eq!(
            engine.overlay(&frame2, LocalEcho::Always, Some(150), t0),
            None
        );
    }

    #[test]
    fn vim_normal_mode_and_autosuggestions_behave_safely() {
        let (mut term, mut renderer, frame0) = make_terminal(40, 4, b"$ ");
        let mut engine = Predictor::default();
        let t0 = Instant::now();
        engine.reconcile(&frame0, LocalEcho::Always, Some(200), t0);

        // Confirm epoch with 'g' while fish/zsh draws faint autosuggestion "it status" to the right.
        engine.on_input(
            InputAction::Insert('g'),
            Some(&frame0),
            LocalEcho::Always,
            Some(200),
            t0,
        );
        term.vt_write(b"g\x1b[2mit status\x1b[0m\x1b[1;4H");
        let frame1 = renderer.frame(&term).unwrap();
        assert_eq!(frame1.cursor, Some((3, 0)));
        engine.reconcile(&frame1, LocalEcho::Always, Some(200), t0);
        assert!(
            !engine.has_pending(),
            "autosuggestion to right of cursor still confirms 'g'"
        );

        // Typing 'i' overwrites faint suggestion in place without shifting it.
        engine.on_input(
            InputAction::Insert('i'),
            Some(&frame1),
            LocalEcho::Always,
            Some(200),
            t0,
        );
        let ov = engine
            .overlay(&frame1, LocalEcho::Always, Some(200), t0)
            .unwrap();
        assert_eq!(ov.start_col, 3);
        assert_eq!(ov.cells.len(), 1);
        assert_eq!(ov.cells[0].ch, Some('i'));

        // Switch to alternate screen (vim) -> resets and unconfirms.
        term.vt_write(b"\x1b[?1049hhello\r\nworld\x1b[1;1H");
        let alt = renderer.frame(&term).unwrap();
        assert!(alt.alt_screen);
        engine.reconcile(&alt, LocalEcho::Always, Some(200), t0);
        assert!(!engine.has_pending());

        // Press 'j' in vim normal mode -> cursor moves down (0,1), not right -> rejected & stays unconfirmed.
        engine.on_input(
            InputAction::Insert('j'),
            Some(&alt),
            LocalEcho::Always,
            Some(200),
            t0,
        );
        assert_eq!(engine.overlay(&alt, LocalEcho::Always, Some(200), t0), None);
        term.vt_write(b"\x1b[2;1H");
        let alt2 = renderer.frame(&term).unwrap();
        engine.reconcile(&alt2, LocalEcho::Always, Some(200), t0);
        assert!(!engine.has_pending());
        // Still unconfirmed: another normal-mode key stays invisible.
        engine.on_input(
            InputAction::Insert('l'),
            Some(&alt2),
            LocalEcho::Always,
            Some(200),
            t0,
        );
        assert_eq!(
            engine.overlay(&alt2, LocalEcho::Always, Some(200), t0),
            None
        );
    }

    #[test]
    fn adaptive_mode_uses_rtt_hysteresis_and_expires_swallowed_keys() {
        let (mut term, mut renderer, frame0) = make_terminal(40, 4, b"$ ");
        let mut engine = Predictor::default();
        let t0 = Instant::now();

        // Low RTT (< 30ms): Adaptive stays inactive.
        engine.reconcile(&frame0, LocalEcho::Adaptive, Some(5), t0);
        engine.on_input(
            InputAction::Insert('a'),
            Some(&frame0),
            LocalEcho::Adaptive,
            Some(5),
            t0,
        );
        assert!(!engine.has_pending());

        // High RTT (>= 30ms): Adaptive activates and stays active at 25ms (hysteresis).
        engine.reconcile(&frame0, LocalEcho::Adaptive, Some(40), t0);
        engine.on_input(
            InputAction::Insert('a'),
            Some(&frame0),
            LocalEcho::Adaptive,
            Some(25),
            t0,
        );
        assert!(engine.has_pending());
        term.vt_write(b"a");
        let frame1 = renderer.frame(&term).unwrap();
        engine.reconcile(&frame1, LocalEcho::Adaptive, Some(25), t0);

        // At 40ms RTT (< 80ms), fresh prediction is NOT underlined until UNDERLINE_AGE passes.
        engine.on_input(
            InputAction::Insert('b'),
            Some(&frame1),
            LocalEcho::Adaptive,
            Some(40),
            t0,
        );
        let fresh = engine
            .overlay(&frame1, LocalEcho::Adaptive, Some(40), t0)
            .unwrap();
        assert!(!fresh.cells[0].underline);
        let aged = engine
            .overlay(&frame1, LocalEcho::Adaptive, Some(40), t0 + UNDERLINE_AGE)
            .unwrap();
        assert!(aged.cells[0].underline);

        // Unanswered prediction expires cleanly after timeout.
        assert!(!engine.expire_stale(Some(40), t0 + Duration::from_millis(500)));
        assert!(engine.expire_stale(Some(40), t0 + Duration::from_secs(2)));
        assert!(!engine.has_pending());
    }

    #[test]
    fn space_reconciles_rprompt_stays_fixed_and_poll_wakes_once_for_underline() {
        // Prompt with a right-aligned RPROMPT separated by spaces at col 30.
        let (mut term, mut renderer, frame0) = make_terminal(40, 4, b"$ \x1b[1;31H12:34\x1b[1;3H");
        let mut engine = Predictor::default();
        let t0 = Instant::now();
        engine.reconcile(&frame0, LocalEcho::Adaptive, Some(40), t0);

        // Confirm epoch with 'l', then type 's', ' ', then Backspace.
        engine.on_input(
            InputAction::Insert('l'),
            Some(&frame0),
            LocalEcho::Adaptive,
            Some(40),
            t0,
        );
        term.vt_write(b"l");
        let frame1 = renderer.frame(&term).unwrap();
        engine.reconcile(&frame1, LocalEcho::Adaptive, Some(40), t0);

        engine.on_input(
            InputAction::Insert('s'),
            Some(&frame1),
            LocalEcho::Adaptive,
            Some(40),
            t0,
        );
        engine.on_input(
            InputAction::Insert(' '),
            Some(&frame1),
            LocalEcho::Adaptive,
            Some(40),
            t0,
        );
        let ov = engine
            .overlay(&frame1, LocalEcho::Adaptive, Some(40), t0)
            .unwrap();
        assert_eq!(ov.start_col, 3);
        assert_eq!(ov.cells.len(), 2);
        assert_eq!(ov.cursor, (5, 0));

        // Poll before UNDERLINE_AGE schedules underline wake; at UNDERLINE_AGE triggers redraw once.
        let (changed, wake) = engine.poll(
            LocalEcho::Adaptive,
            Some(40),
            t0 + Duration::from_millis(10),
        );
        assert!(!changed);
        assert_eq!(wake, Some(t0 + UNDERLINE_AGE));
        let (changed, _) = engine.poll(LocalEcho::Adaptive, Some(40), t0 + UNDERLINE_AGE);
        assert!(changed, "crossing UNDERLINE_AGE must request one redraw");
        let (changed, _) = engine.poll(
            LocalEcho::Adaptive,
            Some(40),
            t0 + UNDERLINE_AGE + Duration::from_millis(5),
        );
        assert!(
            !changed,
            "subsequent polls do not redraw again before expiry"
        );

        // Server echoes "s " -> both 's' and space reconcile cleanly.
        term.vt_write(b"s ");
        let frame2 = renderer.frame(&term).unwrap();
        engine.reconcile(
            &frame2,
            LocalEcho::Adaptive,
            Some(40),
            t0 + Duration::from_millis(40),
        );
        assert!(
            !engine.has_pending(),
            "space must reconcile cleanly against Ghostty cells"
        );

        // Backspace at end of command must NOT shift the RPROMPT at col 30.
        engine.on_input(
            InputAction::Backspace,
            Some(&frame2),
            LocalEcho::Adaptive,
            Some(40),
            t0,
        );
        let ov = engine
            .overlay(&frame2, LocalEcho::Adaptive, Some(40), t0)
            .unwrap();
        assert_eq!(ov.cursor, (4, 0));
        assert!(
            usize::from(ov.start_col) + ov.cells.len() < 30,
            "RPROMPT must stay untouched: {ov:?}"
        );
    }
}
