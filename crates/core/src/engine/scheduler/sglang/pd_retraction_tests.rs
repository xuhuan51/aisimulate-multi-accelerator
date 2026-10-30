use std::collections::BTreeMap;

use uuid::Uuid;

use super::core::SglangCore;
use crate::engine::common::protocols::{DirectRequest, EngineType, MockEngineArgs, SglangArgs};
use crate::engine::scheduler::{SchedulerCommand, SchedulerLifecycleEvent};
use crate::engine::{HandoffId, PressureKind};

fn decode_core(reserved: usize, prefix_caching: bool) -> SglangCore {
    decode_core_with_blocks(reserved, prefix_caching, 8)
}

fn decode_core_with_blocks(reserved: usize, prefix_caching: bool, blocks: usize) -> SglangCore {
    let args = MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .num_gpu_blocks(blocks)
        .block_size(4)
        .enable_prefix_caching(prefix_caching)
        .max_num_seqs(Some(4))
        .sglang(Some(SglangArgs {
            page_size: Some(4),
            chunked_prefill_size: Some(32),
            decode_reserved_tokens: Some(reserved),
            ..Default::default()
        }))
        .build()
        .unwrap();
    SglangCore::new(args)
}

fn activate(core: &mut SglangCore, identifier: u128, output_tokens: usize) -> Uuid {
    let request_id = Uuid::from_u128(identifier);
    let handoff_id = HandoffId::from(request_id);
    let effects = core
        .apply_command_effects(
            SchedulerCommand::ReserveDestination {
                handoff_id,
                request: DirectRequest {
                    uuid: Some(request_id),
                    tokens: vec![identifier as u32; 8],
                    max_output_tokens: output_tokens,
                    ..Default::default()
                },
            },
            true,
        )
        .unwrap();
    assert!(effects.lifecycle_events.iter().any(|event| matches!(
        event,
        SchedulerLifecycleEvent::DestinationReserved { request_id: actual, .. }
            if *actual == request_id
    )));
    core.apply_command(SchedulerCommand::ActivateDestination { handoff_id })
        .unwrap();
    request_id
}

#[rstest::rstest]
fn pd_retraction_restores_kv_without_prefill_or_duplicate_output(
    #[values(0, 4)] reserved: usize,
    #[values(false, true)] prefix_caching: bool,
) {
    let mut core = decode_core(reserved, prefix_caching);
    let identifiers = [activate(&mut core, 100, 12), activate(&mut core, 200, 12)];
    let mut outputs = BTreeMap::<Uuid, Vec<u32>>::new();
    let mut admissions = Vec::new();
    let mut completed = Vec::new();
    let mut retractions = 0;
    let mut now = 0.0;
    for _ in 0..100 {
        let pass = core.try_execute_hidden_pass(now).unwrap();
        now = pass.end_ms;
        let snapshot = pass.fpm.unwrap();
        assert_eq!(
            snapshot.sum_prefill_tokens, 0,
            "a PD decode worker restores KV"
        );
        admissions.extend(pass.admissions.iter().map(|event| event.uuid));
        retractions += pass
            .pressure_events
            .iter()
            .filter(|event| event.kind == PressureKind::SglangRetraction)
            .count();
        for signal in pass
            .start_output_signals
            .into_iter()
            .chain(pass.output_signals)
        {
            assert!(!signal.rejected);
            if let Some(token) = signal.token_id {
                outputs.entry(signal.uuid).or_default().push(token);
            }
            if signal.completed {
                completed.push(signal.uuid);
            }
        }
        if core.is_drained() {
            break;
        }
    }
    assert!(core.is_drained());
    assert!(retractions > 0);
    assert_eq!(admissions, identifiers);
    completed.sort();
    assert_eq!(completed, identifiers);
    for identifier in identifiers {
        let expected: Vec<_> = (0..12)
            .map(|index| crate::engine::common::sequence::synthetic_output_token(identifier, index))
            .collect();
        assert_eq!(outputs[&identifier], expected);
    }
    assert_eq!(
        core.kv_manager.cache().available_tokens() + core.kv_manager.cache().evictable_size,
        32
    );
}

#[test]
fn pd_restore_excludes_pending_sample_from_page_footprint() {
    let mut core = decode_core_with_blocks(4, false, 4);
    core.config.clip_max_new_tokens = 1;
    let identifier = activate(&mut core, 100, 12);
    let mut now = 0.0;
    for _ in 0..4 {
        now = core.try_execute_hidden_pass(now).unwrap().end_ms;
    }
    let mut request = core.running.pop().unwrap();
    assert_eq!(request.current_sequence_len(), 13);
    assert_eq!(request.materialized_tokens, 12);
    core.kv_manager.retract_in_place(&mut request.kv_lease);
    request.reset_for_retract();
    core.waiting.push_back(request);
    let pass = core.try_execute_hidden_pass(now).unwrap();
    assert_eq!(pass.fpm.unwrap().sum_prefill_tokens, 0);
    assert!(pass.admissions.is_empty());
    assert_eq!(pass.output_signals.len(), 1);
    assert_eq!(pass.output_signals[0].uuid, identifier);
    assert_eq!(core.running[0].output_len(), 6);
    assert!(core.waiting.is_empty());
}

#[rstest::rstest]
fn pd_retraction_blocks_fresh_allocation_until_restore_or_cancel(
    #[values(false, true)] cancel: bool,
) {
    let mut core = decode_core(4, false);
    activate(&mut core, 100, 12);
    let retracted = activate(&mut core, 200, 12);
    let mut now = 0.0;
    for _ in 0..20 {
        let pass = core.try_execute_hidden_pass(now).unwrap();
        now = pass.end_ms;
        if !pass.pressure_events.is_empty() {
            break;
        }
    }
    assert_eq!(core.waiting.len(), 1);
    assert_eq!(core.waiting[0].uuid, retracted);
    let fresh = Uuid::from_u128(300);
    let handoff_id = HandoffId::from(fresh);
    let effects = core
        .apply_command_effects(
            SchedulerCommand::ReserveDestination {
                handoff_id,
                request: DirectRequest {
                    uuid: Some(fresh),
                    tokens: vec![300],
                    max_output_tokens: 1,
                    ..Default::default()
                },
            },
            true,
        )
        .unwrap();
    assert!(effects.lifecycle_events.is_empty());
    assert!(core.destination_pages(handoff_id).is_empty());
    if cancel {
        let effects = core
            .apply_command_effects(
                SchedulerCommand::CancelRequest {
                    request_id: retracted,
                },
                true,
            )
            .unwrap();
        assert_eq!(effects.lifecycle_events.len(), 1);
        assert!(core.waiting.is_empty());
        assert!(!core.destination_pages(handoff_id).is_empty());
    } else {
        for _ in 0..20 {
            let pass = core.try_execute_hidden_pass(now).unwrap();
            now = pass.end_ms;
            assert_eq!(pass.fpm.unwrap().sum_prefill_tokens, 0);
            core.retry_pending_destinations();
            if !core.destination_pages(handoff_id).is_empty() {
                break;
            }
        }
        assert!(!core.destination_pages(handoff_id).is_empty());
        assert!(!core.waiting.iter().any(|request| request.uuid == retracted));
    }
    core.apply_command(SchedulerCommand::CancelDestination { handoff_id })
        .unwrap();
    core.apply_command(SchedulerCommand::CancelRequest {
        request_id: Uuid::from_u128(100),
    })
    .unwrap();
    core.apply_command(SchedulerCommand::CancelRequest {
        request_id: retracted,
    })
    .unwrap();
    assert!(core.is_drained());
    assert_eq!(core.kv_manager.cache().available_tokens(), 32);
    activate(&mut core, 200, 12);
}
