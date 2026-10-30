// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay-owned prefill/decode handoff ordering and virtual-transfer state.

use crate::engine::Backend;
pub use crate::engine::{HandoffId, HandoffTransferTiming};
use anyhow::{Result, bail};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum HandoffOrder {
    SourceFirst,
    DestinationFirst,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HandoffFact {
    SourceChunkReady {
        handoff_id: HandoffId,
        ready_tokens: usize,
        transfer_timing: HandoffTransferTiming,
    },
    SourceHeld {
        handoff_id: HandoffId,
        transfer_timing: HandoffTransferTiming,
    },
    DestinationReserved {
        handoff_id: HandoffId,
        transferable_prompt_tokens: usize,
        #[serde(default)]
        cached_prompt_tokens: Option<usize>,
    },
    TransferCompleted {
        handoff_id: HandoffId,
        #[serde(default)]
        chunk_index: usize,
    },
    Failed {
        handoff_id: HandoffId,
    },
    TimedOut {
        handoff_id: HandoffId,
    },
    Canceled {
        handoff_id: HandoffId,
    },
}

impl HandoffFact {
    fn handoff_id(&self) -> HandoffId {
        match *self {
            Self::SourceChunkReady { handoff_id, .. }
            | Self::SourceHeld { handoff_id, .. }
            | Self::DestinationReserved { handoff_id, .. }
            | Self::TransferCompleted { handoff_id, .. }
            | Self::Failed { handoff_id }
            | Self::TimedOut { handoff_id }
            | Self::Canceled { handoff_id } => handoff_id,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum HandoffAction {
    SubmitPrefill {
        handoff_id: HandoffId,
    },
    ReserveDestination {
        handoff_id: HandoffId,
    },
    StartTransfer {
        handoff_id: HandoffId,
        delay_ms: f64,
        #[serde(default)]
        chunk_index: usize,
    },
    ActivateDestination {
        handoff_id: HandoffId,
    },
    ReleaseSource {
        handoff_id: HandoffId,
    },
    CancelSource {
        handoff_id: HandoffId,
    },
    CancelDestination {
        handoff_id: HandoffId,
    },
    Complete {
        handoff_id: HandoffId,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct HandoffActionId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct IssuedHandoffAction {
    pub id: HandoffActionId,
    pub action: HandoffAction,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum HandoffActionOutcome {
    Submitted,
    Accepted,
    Scheduled,
    Applied,
    Noop,
    Failed(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CoordinatorMode {
    Active,
    CleaningUp,
    Complete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandoffCompletion {
    Success,
    Canceled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NormalizedHandoffEvent {
    SourceHeld,
    DestinationAccepted,
    DestinationReserved,
    DestinationActivated,
    SourceReleased,
    Completed,
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NormalizedHandoffConformance {
    pub engine_type: Backend,
    pub order: HandoffOrder,
    pub lifecycle: Vec<NormalizedHandoffEvent>,
    pub source_output_tokens: usize,
    pub destination_output_tokens: usize,
    pub completed_requests: usize,
    pub destination_stored: NormalizedStoredTiming,
    pub source_drained: bool,
    pub destination_drained: bool,
    pub driver_drained: bool,
}

#[doc(hidden)]
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct NormalizedStoredTiming {
    pub before_activation: usize,
    pub on_activation: usize,
    pub repeated_activation_hashes_after_activation: usize,
}

impl NormalizedHandoffConformance {
    #[doc(hidden)]
    pub fn validate(&self) -> Result<()> {
        let expected_order = match self.engine_type {
            Backend::Vllm | Backend::Trtllm => HandoffOrder::SourceFirst,
            Backend::Sglang => HandoffOrder::DestinationFirst,
        };
        if self.order != expected_order {
            bail!(
                "normalized handoff order mismatch: expected {expected_order:?}, got {:?}",
                self.order
            );
        }
        if self.lifecycle != expected_normalized_handoff(self.order) {
            bail!(
                "normalized handoff lifecycle mismatch: got {:?}",
                self.lifecycle
            );
        }
        if self.source_output_tokens != 1
            || self.destination_output_tokens != 2
            || self.completed_requests != 1
        {
            bail!("normalized handoff output/completion counts do not match the fixture");
        }
        if self.destination_stored.before_activation != 0
            || self.destination_stored.on_activation == 0
            || self
                .destination_stored
                .repeated_activation_hashes_after_activation
                != 0
        {
            bail!("normalized handoff destination KV visibility is invalid");
        }
        if !self.source_drained || !self.destination_drained || !self.driver_drained {
            bail!("normalized handoff did not drain");
        }
        Ok(())
    }
}

pub fn expected_normalized_handoff(order: HandoffOrder) -> &'static [NormalizedHandoffEvent] {
    use NormalizedHandoffEvent::*;
    match order {
        HandoffOrder::SourceFirst => &[
            SourceHeld,
            DestinationAccepted,
            DestinationReserved,
            DestinationActivated,
            SourceReleased,
            Completed,
        ],
        HandoffOrder::DestinationFirst => &[
            DestinationAccepted,
            DestinationReserved,
            SourceHeld,
            DestinationActivated,
            SourceReleased,
            Completed,
        ],
    }
}

#[derive(Default)]
struct ActionJournal {
    started: bool,
    next_id: u64,
    issued: FxHashMap<HandoffActionId, HandoffAction>,
    outcomes: FxHashMap<HandoffActionId, HandoffActionOutcome>,
}

#[derive(Default)]
struct SourceProgress {
    submit_issued: bool,
    submitted: bool,
    held: bool,
    transfer_timing: Option<HandoffTransferTiming>,
    ready_tokens: usize,
    chunks: VecDeque<usize>,
    release_issued: bool,
    cancel_issued: bool,
    cleanup_done: bool,
}

#[derive(Default)]
struct DestinationProgress {
    reserve_issued: bool,
    accepted: bool,
    reserved: bool,
    transferable_prompt_tokens: Option<usize>,
    cached_prompt_tokens: Option<usize>,
    activation_issued: bool,
    activation_applied: bool,
    cancel_issued: bool,
    cleanup_done: bool,
}

#[derive(Default)]
struct TransferProgress {
    active: Option<ActiveTransfer>,
    next_chunk_index: usize,
    sent_tokens: usize,
    completed: bool,
}

struct ActiveTransfer {
    chunk_index: usize,
    end_tokens: usize,
    last: bool,
    scheduled: bool,
}

/// Pure state machine for one prefill-to-decode ownership handoff.
pub struct HandoffCoordinatorCore {
    handoff_id: HandoffId,
    order: HandoffOrder,
    mode: CoordinatorMode,
    actions: ActionJournal,
    source: SourceProgress,
    destination: DestinationProgress,
    transfer: TransferProgress,
    completion: Option<HandoffCompletion>,
    fallback_transfer_delay_ms: f64,
    transfer_curve: Option<std::sync::Arc<crate::KvTransferCurve>>,
}

impl HandoffCoordinatorCore {
    pub fn new(handoff_id: HandoffId, order: HandoffOrder) -> Self {
        Self::new_with_fallback(handoff_id, order, 0.0)
    }

    pub fn new_with_fallback(
        handoff_id: HandoffId,
        order: HandoffOrder,
        fallback_transfer_delay_ms: f64,
    ) -> Self {
        debug_assert!(fallback_transfer_delay_ms.is_finite());
        debug_assert!(fallback_transfer_delay_ms >= 0.0);
        Self {
            handoff_id,
            order,
            mode: CoordinatorMode::Active,
            actions: ActionJournal::default(),
            source: SourceProgress::default(),
            destination: DestinationProgress::default(),
            transfer: TransferProgress::default(),
            completion: None,
            fallback_transfer_delay_ms,
            transfer_curve: None,
        }
    }

    pub(crate) fn set_transfer_curve(
        &mut self,
        curve: Option<std::sync::Arc<crate::KvTransferCurve>>,
    ) {
        assert!(!self.actions.started);
        self.transfer_curve = curve;
    }

    pub fn start(&mut self) -> Result<Vec<IssuedHandoffAction>> {
        if self.actions.started {
            return Ok(Vec::new());
        }
        self.actions.started = true;
        let action = match self.order {
            HandoffOrder::SourceFirst => self.issue_submit_prefill(),
            HandoffOrder::DestinationFirst => self.issue_reserve_destination(),
        };
        Ok(vec![action])
    }

    pub fn on_fact(&mut self, fact: HandoffFact) -> Result<Vec<IssuedHandoffAction>> {
        self.validate_handoff(fact.handoff_id())?;
        if self.mode != CoordinatorMode::Active {
            return Ok(Vec::new());
        }
        match fact {
            HandoffFact::SourceChunkReady {
                ready_tokens,
                transfer_timing,
                ..
            } => {
                if self.order != HandoffOrder::DestinationFirst {
                    bail!("source chunks require destination-first handoff ordering");
                }
                self.accept_transfer_timing(transfer_timing)?;
                if ready_tokens >= transfer_timing.full_prompt_tokens {
                    bail!("non-final source chunk must end before the full prompt");
                }
                if self.source.held || ready_tokens <= self.source.ready_tokens {
                    return Ok(Vec::new());
                }
                self.source.ready_tokens = ready_tokens;
                self.source.chunks.push_back(ready_tokens);
                self.advance_active()
            }
            HandoffFact::SourceHeld {
                transfer_timing, ..
            } => {
                if self.source.held {
                    return Ok(Vec::new());
                }
                self.accept_transfer_timing(transfer_timing)?;
                self.source.held = true;
                self.source.ready_tokens = transfer_timing.full_prompt_tokens;
                self.source
                    .chunks
                    .push_back(transfer_timing.full_prompt_tokens);
                self.advance_active()
            }
            HandoffFact::DestinationReserved {
                transferable_prompt_tokens,
                cached_prompt_tokens,
                ..
            } => {
                if self.destination.reserved {
                    return Ok(Vec::new());
                }
                if !self.destination.accepted {
                    bail!("destination reserved before ownership was accepted");
                }
                self.destination.reserved = true;
                self.destination.transferable_prompt_tokens = Some(transferable_prompt_tokens);
                self.destination.cached_prompt_tokens = cached_prompt_tokens;
                self.advance_active()
            }
            HandoffFact::TransferCompleted { chunk_index, .. } => {
                let Some(active) = self.transfer.active.as_ref() else {
                    if chunk_index < self.transfer.next_chunk_index {
                        return Ok(Vec::new());
                    }
                    bail!("transfer completed before it was issued");
                };
                if chunk_index < active.chunk_index {
                    return Ok(Vec::new());
                }
                if chunk_index != active.chunk_index || !active.scheduled {
                    bail!("transfer completed before it was scheduled");
                }
                self.transfer.sent_tokens = active.end_tokens;
                self.transfer.completed = active.last;
                self.transfer.active = None;
                self.advance_active()
            }
            HandoffFact::Failed { .. }
            | HandoffFact::TimedOut { .. }
            | HandoffFact::Canceled { .. } => self.begin_cleanup(),
        }
    }

    pub fn on_action_outcome(
        &mut self,
        action_id: HandoffActionId,
        outcome: HandoffActionOutcome,
    ) -> Result<Vec<IssuedHandoffAction>> {
        if self.mode == CoordinatorMode::Complete {
            return Ok(Vec::new());
        }
        let Some(action) = self.actions.issued.get(&action_id).copied() else {
            bail!("unknown handoff action {action_id:?}");
        };
        if let Some(previous) = self.actions.outcomes.get(&action_id) {
            if previous != &outcome {
                bail!("conflicting outcome for handoff action {action_id:?}");
            }
            return Ok(Vec::new());
        }
        self.actions.outcomes.insert(action_id, outcome.clone());

        if let HandoffActionOutcome::Failed(_) = outcome {
            if matches!(
                action,
                HandoffAction::CancelSource { .. } | HandoffAction::CancelDestination { .. }
            ) {
                bail!("handoff cleanup action {action_id:?} failed");
            }
            return self.begin_cleanup();
        }

        match action {
            HandoffAction::SubmitPrefill { .. } => {
                require_outcome(&outcome, &[HandoffActionOutcome::Submitted])?;
                self.source.submitted = true;
            }
            HandoffAction::ReserveDestination { .. } => {
                require_outcome(&outcome, &[HandoffActionOutcome::Accepted])?;
                self.destination.accepted = true;
            }
            HandoffAction::StartTransfer { chunk_index, .. } => {
                require_outcome(&outcome, &[HandoffActionOutcome::Scheduled])?;
                let active = self
                    .transfer
                    .active
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("scheduled transfer is not active"))?;
                if active.chunk_index != chunk_index {
                    bail!("scheduled transfer does not match the active chunk");
                }
                active.scheduled = true;
            }
            HandoffAction::ActivateDestination { .. } => {
                require_outcome(&outcome, &[HandoffActionOutcome::Applied])?;
                self.destination.activation_applied = true;
            }
            HandoffAction::ReleaseSource { .. } => {
                require_outcome(
                    &outcome,
                    &[HandoffActionOutcome::Applied, HandoffActionOutcome::Noop],
                )?;
                self.source.cleanup_done = true;
            }
            HandoffAction::CancelSource { .. } => {
                require_outcome(
                    &outcome,
                    &[HandoffActionOutcome::Applied, HandoffActionOutcome::Noop],
                )?;
                self.source.cleanup_done = true;
            }
            HandoffAction::CancelDestination { .. } => {
                require_outcome(
                    &outcome,
                    &[HandoffActionOutcome::Applied, HandoffActionOutcome::Noop],
                )?;
                self.destination.cleanup_done = true;
            }
            HandoffAction::Complete { .. } => return Ok(Vec::new()),
        }

        match self.mode {
            CoordinatorMode::Active => self.advance_active(),
            CoordinatorMode::CleaningUp => self.advance_cleanup(),
            CoordinatorMode::Complete => Ok(Vec::new()),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.mode == CoordinatorMode::Complete
    }

    pub fn completion(&self) -> Option<HandoffCompletion> {
        self.completion
    }

    fn accept_transfer_timing(&mut self, timing: HandoffTransferTiming) -> Result<()> {
        if !self.source.submitted {
            bail!("source KV became ready before prefill submission was acknowledged");
        }
        validate_transfer_timing(timing)?;
        if self
            .source
            .transfer_timing
            .is_some_and(|previous| previous != timing)
        {
            bail!("source transfer timing changed within a handoff");
        }
        self.source.transfer_timing = Some(timing);
        Ok(())
    }

    fn advance_active(&mut self) -> Result<Vec<IssuedHandoffAction>> {
        if self.order == HandoffOrder::SourceFirst
            && self.source.held
            && !self.destination.reserve_issued
        {
            return Ok(vec![self.issue_reserve_destination()]);
        }
        if self.order == HandoffOrder::DestinationFirst
            && self.destination.reserved
            && !self.source.submit_issued
        {
            return Ok(vec![self.issue_submit_prefill()]);
        }
        while self.destination.reserved
            && self.transfer.active.is_none()
            && let Some(end_tokens) = self.source.chunks.pop_front()
        {
            let transfer_timing = self
                .source
                .transfer_timing
                .expect("ready source must retain transfer timing");
            let transferable_prompt_tokens = self
                .destination
                .transferable_prompt_tokens
                .expect("reserved destination must report its transferable footprint");
            let last = self.source.held && end_tokens == transfer_timing.full_prompt_tokens;
            let tokens = match transfer_timing.mode {
                crate::engine::TransferTimingMode::FullPrompt => {
                    end_tokens.saturating_sub(self.transfer.sent_tokens)
                }
                crate::engine::TransferTimingMode::DestinationMissing => {
                    let prefix_tokens =
                        self.destination.cached_prompt_tokens.unwrap_or_else(|| {
                            transfer_timing
                                .full_prompt_tokens
                                .saturating_sub(transferable_prompt_tokens)
                        });
                    let sent_tokens = self.transfer.sent_tokens.saturating_sub(prefix_tokens);
                    let remaining_tokens = transferable_prompt_tokens.saturating_sub(sent_tokens);
                    if last {
                        remaining_tokens
                    } else {
                        end_tokens
                            .saturating_sub(self.transfer.sent_tokens.max(prefix_tokens))
                            .min(remaining_tokens)
                    }
                }
            };
            if tokens == 0 && !last {
                self.transfer.sent_tokens = end_tokens;
                continue;
            }
            let delay_ms = if let Some(curve) = &self.transfer_curve {
                curve.latency_ms(tokens)?
            } else {
                HandoffTransferTiming {
                    mode: crate::engine::TransferTimingMode::FullPrompt,
                    full_prompt_tokens: tokens,
                    state_bytes: if last { transfer_timing.state_bytes } else { 0 },
                    ..transfer_timing
                }
                .full_prompt_delay_ms()
                .unwrap_or(if last {
                    self.fallback_transfer_delay_ms
                } else {
                    0.0
                })
            };
            let chunk_index = self.transfer.next_chunk_index;
            self.transfer.next_chunk_index += 1;
            self.transfer.active = Some(ActiveTransfer {
                chunk_index,
                end_tokens,
                last,
                scheduled: false,
            });
            return Ok(vec![self.issue(HandoffAction::StartTransfer {
                handoff_id: self.handoff_id,
                delay_ms,
                chunk_index,
            })]);
        }
        let mut ready = Vec::new();
        if self.transfer.completed && !self.destination.activation_issued {
            self.destination.activation_issued = true;
            ready.push(self.issue(HandoffAction::ActivateDestination {
                handoff_id: self.handoff_id,
            }));
        }
        let release_ready = self.destination.activation_applied
            || (self.order == HandoffOrder::DestinationFirst && self.transfer.completed);
        if release_ready && !self.source.release_issued {
            self.source.release_issued = true;
            ready.push(self.issue(HandoffAction::ReleaseSource {
                handoff_id: self.handoff_id,
            }));
        }
        if !ready.is_empty() {
            return Ok(ready);
        }
        if self.source.cleanup_done && self.destination.activation_applied {
            return Ok(vec![self.complete()]);
        }
        Ok(Vec::new())
    }

    fn begin_cleanup(&mut self) -> Result<Vec<IssuedHandoffAction>> {
        if self.mode == CoordinatorMode::Complete {
            return Ok(Vec::new());
        }
        self.mode = CoordinatorMode::CleaningUp;
        self.advance_cleanup()
    }

    fn advance_cleanup(&mut self) -> Result<Vec<IssuedHandoffAction>> {
        let mut actions = Vec::new();
        if self.source.submit_issued && !self.source.cancel_issued && !self.source.cleanup_done {
            self.source.cancel_issued = true;
            actions.push(self.issue(HandoffAction::CancelSource {
                handoff_id: self.handoff_id,
            }));
        }
        if self.destination.reserve_issued
            && !self.destination.cancel_issued
            && !self.destination.cleanup_done
        {
            self.destination.cancel_issued = true;
            actions.push(self.issue(HandoffAction::CancelDestination {
                handoff_id: self.handoff_id,
            }));
        }
        if actions.is_empty()
            && (!self.source.submit_issued || self.source.cleanup_done)
            && (!self.destination.reserve_issued || self.destination.cleanup_done)
        {
            actions.push(self.complete());
        }
        Ok(actions)
    }

    fn issue_submit_prefill(&mut self) -> IssuedHandoffAction {
        self.source.submit_issued = true;
        self.issue(HandoffAction::SubmitPrefill {
            handoff_id: self.handoff_id,
        })
    }

    fn issue_reserve_destination(&mut self) -> IssuedHandoffAction {
        self.destination.reserve_issued = true;
        self.issue(HandoffAction::ReserveDestination {
            handoff_id: self.handoff_id,
        })
    }

    fn complete(&mut self) -> IssuedHandoffAction {
        self.completion = Some(if self.mode == CoordinatorMode::CleaningUp {
            HandoffCompletion::Canceled
        } else {
            HandoffCompletion::Success
        });
        self.mode = CoordinatorMode::Complete;
        let action = self.issue(HandoffAction::Complete {
            handoff_id: self.handoff_id,
        });
        self.actions.issued.clear();
        self.actions.outcomes.clear();
        action
    }

    fn issue(&mut self, action: HandoffAction) -> IssuedHandoffAction {
        let id = HandoffActionId(self.actions.next_id);
        self.actions.next_id = self
            .actions
            .next_id
            .checked_add(1)
            .expect("handoff action ID overflow");
        let previous = self.actions.issued.insert(id, action);
        debug_assert!(previous.is_none());
        IssuedHandoffAction { id, action }
    }

    fn validate_handoff(&self, handoff_id: HandoffId) -> Result<()> {
        if handoff_id != self.handoff_id {
            bail!("fact belongs to a different handoff");
        }
        Ok(())
    }
}

pub fn validate_transfer_delay_ms(transfer_delay_ms: Option<f64>) -> Result<()> {
    let Some(delay_ms) = transfer_delay_ms else {
        return Ok(());
    };
    if !delay_ms.is_finite() || delay_ms < 0.0 {
        bail!("invalid handoff transfer delay {delay_ms}");
    }
    Ok(())
}

pub fn validate_transfer_timing(transfer_timing: HandoffTransferTiming) -> Result<()> {
    if let Some(bandwidth_gb_s) = transfer_timing.bandwidth_gb_s
        && (!bandwidth_gb_s.is_finite() || bandwidth_gb_s < 0.0)
    {
        bail!("invalid handoff transfer bandwidth {bandwidth_gb_s}");
    }
    validate_transfer_delay_ms(transfer_timing.full_prompt_delay_ms())
}

fn require_outcome(outcome: &HandoffActionOutcome, allowed: &[HandoffActionOutcome]) -> Result<()> {
    if allowed.contains(outcome) {
        return Ok(());
    }
    bail!("invalid handoff action outcome {outcome:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::TransferTimingMode;
    use uuid::Uuid;

    fn start_transfer_delay(timing: HandoffTransferTiming, fallback_ms: f64) -> f64 {
        start_transfer_with_curve(timing, fallback_ms, None, 3).unwrap()
    }

    fn start_transfer_with_curve(
        timing: HandoffTransferTiming,
        fallback_ms: f64,
        curve: Option<std::sync::Arc<crate::KvTransferCurve>>,
        missing_tokens: usize,
    ) -> Result<f64> {
        let handoff_id = HandoffId::new(Uuid::from_u128(1));
        let mut coordinator = HandoffCoordinatorCore::new_with_fallback(
            handoff_id,
            HandoffOrder::SourceFirst,
            fallback_ms,
        );
        coordinator.set_transfer_curve(curve);
        let submit = coordinator.start().unwrap().remove(0);
        assert!(matches!(submit.action, HandoffAction::SubmitPrefill { .. }));
        assert!(
            coordinator
                .on_action_outcome(submit.id, HandoffActionOutcome::Submitted)
                .unwrap()
                .is_empty()
        );
        let reserve = coordinator
            .on_fact(HandoffFact::SourceHeld {
                handoff_id,
                transfer_timing: timing,
            })
            .unwrap()
            .remove(0);
        assert!(matches!(
            reserve.action,
            HandoffAction::ReserveDestination { .. }
        ));
        assert!(
            coordinator
                .on_action_outcome(reserve.id, HandoffActionOutcome::Accepted)
                .unwrap()
                .is_empty()
        );
        let transfer = coordinator
            .on_fact(HandoffFact::DestinationReserved {
                handoff_id,
                transferable_prompt_tokens: missing_tokens,
                cached_prompt_tokens: None,
            })?
            .remove(0);
        match transfer.action {
            HandoffAction::StartTransfer { delay_ms, .. } => Ok(delay_ms),
            action => panic!("expected transfer action, got {action:?}"),
        }
    }

    fn transferred_sglang_handoff() -> (HandoffCoordinatorCore, Vec<IssuedHandoffAction>) {
        let handoff_id = HandoffId::new(Uuid::from_u128(2));
        let mut coordinator =
            HandoffCoordinatorCore::new(handoff_id, HandoffOrder::DestinationFirst);
        let reserve = coordinator.start().unwrap().remove(0);
        coordinator
            .on_action_outcome(reserve.id, HandoffActionOutcome::Accepted)
            .unwrap();
        let submit = coordinator
            .on_fact(HandoffFact::DestinationReserved {
                handoff_id,
                transferable_prompt_tokens: 128,
                cached_prompt_tokens: None,
            })
            .unwrap()
            .remove(0);
        coordinator
            .on_action_outcome(submit.id, HandoffActionOutcome::Submitted)
            .unwrap();
        let transfer = coordinator
            .on_fact(HandoffFact::SourceHeld {
                handoff_id,
                transfer_timing: HandoffTransferTiming {
                    mode: TransferTimingMode::FullPrompt,
                    full_prompt_tokens: 128,
                    kv_bytes_per_token: None,
                    state_bytes: 0,
                    bandwidth_gb_s: None,
                },
            })
            .unwrap()
            .remove(0);
        coordinator
            .on_action_outcome(transfer.id, HandoffActionOutcome::Scheduled)
            .unwrap();
        let ready = coordinator
            .on_fact(HandoffFact::TransferCompleted {
                handoff_id,
                chunk_index: 0,
            })
            .unwrap();
        (coordinator, ready)
    }

    fn streaming_handoff(missing_tokens: usize, measured: bool) -> HandoffCoordinatorCore {
        streaming_handoff_with_prefix(missing_tokens, None, measured)
    }

    fn streaming_handoff_with_prefix(
        missing_tokens: usize,
        cached_prompt_tokens: Option<usize>,
        measured: bool,
    ) -> HandoffCoordinatorCore {
        let handoff_id = HandoffId::new(Uuid::from_u128(3));
        let mut coordinator = HandoffCoordinatorCore::new_with_fallback(
            handoff_id,
            HandoffOrder::DestinationFirst,
            7.0,
        );
        if measured {
            coordinator.set_transfer_curve(Some(crate::KvTransferCurve::test_curve()));
        }
        let reserve = coordinator.start().unwrap().remove(0);
        coordinator
            .on_action_outcome(reserve.id, HandoffActionOutcome::Accepted)
            .unwrap();
        let submit = coordinator
            .on_fact(HandoffFact::DestinationReserved {
                handoff_id,
                transferable_prompt_tokens: missing_tokens,
                cached_prompt_tokens,
            })
            .unwrap()
            .remove(0);
        coordinator
            .on_action_outcome(submit.id, HandoffActionOutcome::Submitted)
            .unwrap();
        coordinator
    }

    fn chunk_timing() -> HandoffTransferTiming {
        HandoffTransferTiming {
            mode: TransferTimingMode::FullPrompt,
            full_prompt_tokens: 384,
            kv_bytes_per_token: Some(1_000_000),
            state_bytes: 0,
            bandwidth_gb_s: Some(128.0),
        }
    }

    #[test]
    fn sglang_chunks_keep_fifo_boundaries_and_ignore_old_completions() {
        for measured in [false, true] {
            let mut coordinator = streaming_handoff(384, measured);
            let handoff_id = coordinator.handoff_id;
            let timing = chunk_timing();
            let mut actions = coordinator
                .on_fact(HandoffFact::SourceChunkReady {
                    handoff_id,
                    ready_tokens: 128,
                    transfer_timing: timing,
                })
                .unwrap();
            assert!(
                coordinator
                    .on_fact(HandoffFact::SourceChunkReady {
                        handoff_id,
                        ready_tokens: 256,
                        transfer_timing: timing,
                    })
                    .unwrap()
                    .is_empty()
            );
            assert!(
                coordinator
                    .on_fact(HandoffFact::SourceHeld {
                        handoff_id,
                        transfer_timing: timing,
                    })
                    .unwrap()
                    .is_empty()
            );
            for chunk_index in 0..3 {
                assert_eq!(actions.len(), 1);
                let transfer = actions.remove(0);
                assert_eq!(
                    transfer.action,
                    HandoffAction::StartTransfer {
                        handoff_id,
                        delay_ms: 1.0,
                        chunk_index,
                    }
                );
                coordinator
                    .on_action_outcome(transfer.id, HandoffActionOutcome::Scheduled)
                    .unwrap();
                if chunk_index > 0 {
                    assert!(
                        coordinator
                            .on_fact(HandoffFact::TransferCompleted {
                                handoff_id,
                                chunk_index: chunk_index - 1,
                            })
                            .unwrap()
                            .is_empty()
                    );
                }
                actions = coordinator
                    .on_fact(HandoffFact::TransferCompleted {
                        handoff_id,
                        chunk_index,
                    })
                    .unwrap();
            }
            assert_eq!(actions.len(), 2);
            assert!(matches!(
                actions[0].action,
                HandoffAction::ActivateDestination { .. }
            ));
            assert!(matches!(
                actions[1].action,
                HandoffAction::ReleaseSource { .. }
            ));
            assert!(
                coordinator
                    .on_fact(HandoffFact::TransferCompleted {
                        handoff_id,
                        chunk_index: 2,
                    })
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[test]
    fn sglang_chunks_skip_destination_prefix_and_send_state_once() {
        for missing_tokens in [0, 128] {
            let mut coordinator = streaming_handoff(missing_tokens, false);
            let handoff_id = coordinator.handoff_id;
            let timing = HandoffTransferTiming {
                mode: TransferTimingMode::DestinationMissing,
                state_bytes: 128_000_000,
                ..chunk_timing()
            };
            for ready_tokens in [128, 256] {
                assert!(
                    coordinator
                        .on_fact(HandoffFact::SourceChunkReady {
                            handoff_id,
                            ready_tokens,
                            transfer_timing: timing,
                        })
                        .unwrap()
                        .is_empty()
                );
            }
            let actions = coordinator
                .on_fact(HandoffFact::SourceHeld {
                    handoff_id,
                    transfer_timing: timing,
                })
                .unwrap();
            assert_eq!(
                actions[0].action,
                HandoffAction::StartTransfer {
                    handoff_id,
                    delay_ms: if missing_tokens == 0 { 1.0 } else { 2.0 },
                    chunk_index: 0,
                }
            );
        }
    }

    #[test]
    fn sglang_chunks_separate_cached_prefix_from_partial_page_padding() {
        let mut coordinator = streaming_handoff_with_prefix(256, Some(128), false);
        let handoff_id = coordinator.handoff_id;
        let timing = HandoffTransferTiming {
            full_prompt_tokens: 257,
            mode: TransferTimingMode::DestinationMissing,
            ..chunk_timing()
        };
        assert!(
            coordinator
                .on_fact(HandoffFact::SourceChunkReady {
                    handoff_id,
                    ready_tokens: 128,
                    transfer_timing: timing,
                })
                .unwrap()
                .is_empty()
        );
        let first = coordinator
            .on_fact(HandoffFact::SourceChunkReady {
                handoff_id,
                ready_tokens: 256,
                transfer_timing: timing,
            })
            .unwrap()
            .remove(0);
        assert_eq!(
            first.action,
            HandoffAction::StartTransfer {
                handoff_id,
                delay_ms: 1.0,
                chunk_index: 0,
            }
        );
        coordinator
            .on_action_outcome(first.id, HandoffActionOutcome::Scheduled)
            .unwrap();
        assert!(
            coordinator
                .on_fact(HandoffFact::TransferCompleted {
                    handoff_id,
                    chunk_index: 0,
                })
                .unwrap()
                .is_empty()
        );
        let last = coordinator
            .on_fact(HandoffFact::SourceHeld {
                handoff_id,
                transfer_timing: timing,
            })
            .unwrap()
            .remove(0);
        assert_eq!(
            last.action,
            HandoffAction::StartTransfer {
                handoff_id,
                delay_ms: 1.0,
                chunk_index: 1,
            }
        );
    }

    #[test]
    fn sglang_chunks_apply_fixed_fallback_only_to_final_send() {
        let mut coordinator = streaming_handoff(384, false);
        let handoff_id = coordinator.handoff_id;
        let timing = HandoffTransferTiming {
            bandwidth_gb_s: None,
            ..chunk_timing()
        };
        let transfer = coordinator
            .on_fact(HandoffFact::SourceChunkReady {
                handoff_id,
                ready_tokens: 128,
                transfer_timing: timing,
            })
            .unwrap()
            .remove(0);
        assert_eq!(
            transfer.action,
            HandoffAction::StartTransfer {
                handoff_id,
                delay_ms: 0.0,
                chunk_index: 0,
            }
        );
        coordinator
            .on_action_outcome(transfer.id, HandoffActionOutcome::Scheduled)
            .unwrap();
        assert!(
            coordinator
                .on_fact(HandoffFact::TransferCompleted {
                    handoff_id,
                    chunk_index: 0,
                })
                .unwrap()
                .is_empty()
        );
        let final_transfer = coordinator
            .on_fact(HandoffFact::SourceHeld {
                handoff_id,
                transfer_timing: timing,
            })
            .unwrap()
            .remove(0);
        assert_eq!(
            final_transfer.action,
            HandoffAction::StartTransfer {
                handoff_id,
                delay_ms: 7.0,
                chunk_index: 1,
            }
        );
    }

    #[test]
    fn sglang_chunks_reject_changed_identity_and_premature_completion() {
        let mut coordinator = streaming_handoff(384, false);
        let handoff_id = coordinator.handoff_id;
        let timing = chunk_timing();
        assert!(
            coordinator
                .on_fact(HandoffFact::SourceChunkReady {
                    handoff_id,
                    ready_tokens: 384,
                    transfer_timing: timing,
                })
                .is_err()
        );
        let transfer = coordinator
            .on_fact(HandoffFact::SourceChunkReady {
                handoff_id,
                ready_tokens: 128,
                transfer_timing: timing,
            })
            .unwrap()
            .remove(0);
        assert!(
            coordinator
                .on_fact(HandoffFact::TransferCompleted {
                    handoff_id,
                    chunk_index: 0,
                })
                .is_err()
        );
        assert!(
            coordinator
                .on_fact(HandoffFact::SourceChunkReady {
                    handoff_id,
                    ready_tokens: 256,
                    transfer_timing: HandoffTransferTiming {
                        full_prompt_tokens: 512,
                        ..timing
                    },
                })
                .is_err()
        );
        coordinator
            .on_action_outcome(transfer.id, HandoffActionOutcome::Scheduled)
            .unwrap();
        assert!(
            coordinator
                .on_fact(HandoffFact::TransferCompleted {
                    handoff_id,
                    chunk_index: 1,
                })
                .is_err()
        );
    }

    #[test]
    fn sglang_chunks_cancel_without_activating_partial_kv() {
        let mut coordinator = streaming_handoff(384, false);
        let handoff_id = coordinator.handoff_id;
        let transfer = coordinator
            .on_fact(HandoffFact::SourceChunkReady {
                handoff_id,
                ready_tokens: 128,
                transfer_timing: chunk_timing(),
            })
            .unwrap()
            .remove(0);
        coordinator
            .on_action_outcome(transfer.id, HandoffActionOutcome::Scheduled)
            .unwrap();
        let cleanup = coordinator
            .on_fact(HandoffFact::Canceled { handoff_id })
            .unwrap();
        assert_eq!(cleanup.len(), 2);
        assert!(matches!(
            cleanup[0].action,
            HandoffAction::CancelSource { .. }
        ));
        assert!(matches!(
            cleanup[1].action,
            HandoffAction::CancelDestination { .. }
        ));
        assert!(
            coordinator
                .on_fact(HandoffFact::TransferCompleted {
                    handoff_id,
                    chunk_index: 0,
                })
                .unwrap()
                .is_empty()
        );
        for action in cleanup {
            coordinator
                .on_action_outcome(action.id, HandoffActionOutcome::Applied)
                .unwrap();
        }
        assert_eq!(coordinator.completion(), Some(HandoffCompletion::Canceled));
    }

    #[test]
    fn sglang_transfer_completion_allows_independent_release_and_activation() {
        for release_first in [false, true] {
            let (mut coordinator, ready) = transferred_sglang_handoff();
            assert_eq!(ready.len(), 2);
            assert!(matches!(
                ready[0].action,
                HandoffAction::ActivateDestination { .. }
            ));
            assert!(matches!(
                ready[1].action,
                HandoffAction::ReleaseSource { .. }
            ));
            let (first, second) = if release_first { (1, 0) } else { (0, 1) };
            assert!(
                coordinator
                    .on_action_outcome(ready[first].id, HandoffActionOutcome::Applied)
                    .unwrap()
                    .is_empty()
            );
            assert!(!coordinator.is_complete());
            let completed = coordinator
                .on_action_outcome(ready[second].id, HandoffActionOutcome::Applied)
                .unwrap();
            assert_eq!(completed.len(), 1);
            assert!(matches!(
                completed[0].action,
                HandoffAction::Complete { .. }
            ));
            assert!(coordinator.is_complete());
        }
    }

    #[test]
    fn sglang_cancel_after_source_release_still_cleans_destination() {
        let (mut coordinator, ready) = transferred_sglang_handoff();
        coordinator
            .on_action_outcome(ready[1].id, HandoffActionOutcome::Applied)
            .unwrap();
        let cleanup = coordinator
            .on_fact(HandoffFact::Canceled {
                handoff_id: HandoffId::new(Uuid::from_u128(2)),
            })
            .unwrap();
        assert_eq!(cleanup.len(), 1);
        assert!(matches!(
            cleanup[0].action,
            HandoffAction::CancelDestination { .. }
        ));
        assert!(!coordinator.is_complete());
        let completed = coordinator
            .on_action_outcome(cleanup[0].id, HandoffActionOutcome::Applied)
            .unwrap();
        assert_eq!(completed.len(), 1);
        assert!(coordinator.is_complete());
        assert!(
            coordinator
                .on_action_outcome(ready[0].id, HandoffActionOutcome::Applied)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn measured_curve_uses_destination_footprint_and_never_bandwidth_fallback() {
        let curve = crate::KvTransferCurve::test_curve();
        let timing = HandoffTransferTiming {
            mode: TransferTimingMode::DestinationMissing,
            full_prompt_tokens: 256,
            kv_bytes_per_token: None,
            state_bytes: 0,
            bandwidth_gb_s: None,
        };
        assert_eq!(
            start_transfer_with_curve(timing, 99.0, Some(curve.clone()), 0).unwrap(),
            0.0
        );
        assert_eq!(
            start_transfer_with_curve(timing, 99.0, Some(curve.clone()), 3).unwrap(),
            1.0
        );
        let full = HandoffTransferTiming {
            mode: TransferTimingMode::FullPrompt,
            ..timing
        };
        assert_eq!(
            start_transfer_with_curve(full, 99.0, Some(curve.clone()), 3).unwrap(),
            3.0
        );
        assert!(start_transfer_with_curve(timing, 99.0, Some(curve), 300).is_err());
    }

    #[test]
    fn missing_or_zero_bandwidth_uses_configured_fallback() {
        let missing = HandoffTransferTiming {
            mode: TransferTimingMode::DestinationMissing,
            full_prompt_tokens: 10,
            kv_bytes_per_token: None,
            state_bytes: 0,
            bandwidth_gb_s: None,
        };
        assert_eq!(start_transfer_delay(missing, 7.5), 7.5);

        let zero_bandwidth = HandoffTransferTiming {
            kv_bytes_per_token: Some(1024),
            bandwidth_gb_s: Some(0.0),
            ..missing
        };
        validate_transfer_timing(zero_bandwidth).unwrap();
        assert_eq!(start_transfer_delay(zero_bandwidth, 7.5), 7.5);
    }

    #[test]
    fn negative_bandwidth_is_rejected() {
        let timing = HandoffTransferTiming {
            mode: TransferTimingMode::FullPrompt,
            full_prompt_tokens: 10,
            kv_bytes_per_token: Some(1024),
            state_bytes: 0,
            bandwidth_gb_s: Some(-1.0),
        };
        assert!(validate_transfer_timing(timing).is_err());
    }
}
