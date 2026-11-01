use std::collections::BTreeMap;

use serde::Serialize;

use crate::replay::WorkerStage;
use crate::replay::protocol::ForwardPassSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ForwardBatchPhase {
    Prefill,
    Decode,
    Mixed,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayForwardBatchSummary {
    pub stage: WorkerStage,
    pub worker_id: usize,
    pub dp_rank: Option<u32>,
    pub phase: ForwardBatchPhase,
    pub passes: u64,
    pub request_instances: u64,
    pub max_batch_size: u64,
    pub duration_ms: f64,
    pub request_duration_ms: f64,
    pub batch_histogram: BTreeMap<u64, u64>,
}

#[derive(Debug, Default)]
pub(super) struct ForwardBatchCollector {
    rows: BTreeMap<(WorkerStage, usize, Option<u32>, ForwardBatchPhase), ReplayForwardBatchSummary>,
}

impl ForwardBatchCollector {
    pub(super) fn observe<'a>(
        &mut self,
        stage: WorkerStage,
        worker_id: usize,
        snapshots: impl Iterator<Item = &'a ForwardPassSnapshot>,
    ) {
        let mut prefill_requests = 0;
        let mut decode_requests = 0;
        let mut duration_ms = 0.0_f64;
        for snapshot in snapshots {
            prefill_requests += u64::from(snapshot.num_prefill_requests);
            decode_requests += u64::from(snapshot.num_decode_requests);
            duration_ms = duration_ms.max(snapshot.wall_time_secs * 1_000.0);
            self.record(
                stage,
                worker_id,
                Some(snapshot.dp_rank),
                u64::from(snapshot.num_prefill_requests),
                u64::from(snapshot.num_decode_requests),
                snapshot.wall_time_secs * 1_000.0,
            );
        }
        self.record(
            stage,
            worker_id,
            None,
            prefill_requests,
            decode_requests,
            duration_ms,
        );
    }

    fn record(
        &mut self,
        stage: WorkerStage,
        worker_id: usize,
        dp_rank: Option<u32>,
        prefill_requests: u64,
        decode_requests: u64,
        duration_ms: f64,
    ) {
        let phase = match (prefill_requests > 0, decode_requests > 0) {
            (true, true) => ForwardBatchPhase::Mixed,
            (true, false) => ForwardBatchPhase::Prefill,
            (false, true) => ForwardBatchPhase::Decode,
            (false, false) => return,
        };
        let batch_size = prefill_requests + decode_requests;
        let row = self
            .rows
            .entry((stage, worker_id, dp_rank, phase))
            .or_insert_with(|| ReplayForwardBatchSummary {
                stage,
                worker_id,
                dp_rank,
                phase,
                passes: 0,
                request_instances: 0,
                max_batch_size: 0,
                duration_ms: 0.0,
                request_duration_ms: 0.0,
                batch_histogram: BTreeMap::new(),
            });
        row.passes += 1;
        row.request_instances += batch_size;
        row.max_batch_size = row.max_batch_size.max(batch_size);
        row.duration_ms += duration_ms;
        row.request_duration_ms += duration_ms * batch_size as f64;
        *row.batch_histogram.entry(batch_size).or_default() += 1;
    }

    pub(super) fn finish(self) -> Vec<ReplayForwardBatchSummary> {
        self.rows.into_values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(rank: u32, prefill: u32, decode: u32, duration_ms: f64) -> ForwardPassSnapshot {
        ForwardPassSnapshot {
            dp_rank: rank,
            num_prefill_requests: prefill,
            num_decode_requests: decode,
            num_queued_decode: 99,
            wall_time_secs: duration_ms / 1_000.0,
            ..Default::default()
        }
    }

    #[test]
    fn grouped_batch_counts_real_ranks_once_and_excludes_waiting_and_idle() {
        let mut collector = ForwardBatchCollector::default();
        collector.observe(
            WorkerStage::Decode,
            3,
            [
                snapshot(0, 0, 1, 10.0),
                snapshot(1, 0, 3, 10.0),
                snapshot(2, 0, 0, 10.0),
            ]
            .iter(),
        );
        collector.observe(WorkerStage::Decode, 3, [snapshot(0, 0, 2, 30.0)].iter());
        let rows = collector.finish();
        assert_eq!(rows.len(), 3);
        let group = &rows[0];
        assert_eq!(group.dp_rank, None);
        assert_eq!(group.passes, 2);
        assert_eq!(group.request_instances, 6);
        assert_eq!(group.max_batch_size, 4);
        assert_eq!(group.duration_ms, 40.0);
        assert_eq!(group.request_duration_ms, 100.0);
        assert_eq!(group.batch_histogram, BTreeMap::from([(2, 1), (4, 1)]));
        assert_eq!(rows[1].request_instances, 3);
        assert_eq!(rows[1].request_duration_ms, 70.0);
        assert_eq!(rows[2].request_instances, 3);
    }

    #[test]
    fn worker_role_and_mixed_passes_have_distinct_rows() {
        let mut collector = ForwardBatchCollector::default();
        collector.observe(WorkerStage::Prefill, 0, [snapshot(0, 2, 0, 5.0)].iter());
        collector.observe(WorkerStage::Decode, 0, [snapshot(0, 0, 2, 5.0)].iter());
        collector.observe(WorkerStage::Decode, 1, [snapshot(0, 0, 3, 5.0)].iter());
        collector.observe(WorkerStage::Aggregated, 0, [snapshot(0, 1, 2, 5.0)].iter());
        let rows = collector.finish();
        assert_eq!(rows.len(), 8);
        let groups: Vec<_> = rows.iter().filter(|row| row.dp_rank.is_none()).collect();
        assert_eq!(groups[0].phase, ForwardBatchPhase::Mixed);
        assert_eq!(groups[0].request_instances, 3);
        assert_eq!(groups[1].phase, ForwardBatchPhase::Prefill);
        assert_eq!(groups[2].worker_id, 0);
        assert_eq!(groups[3].worker_id, 1);
    }

    #[test]
    fn reporting_epoch_keeps_capture_enabled_without_reusing_preparation_batches() {
        let mut collector = crate::replay::report::TraceCollector::default();
        collector.set_capture_forward_passes(true);
        collector.on_completed_forward(WorkerStage::Decode, 0, [snapshot(0, 0, 4, 10.0)].iter());
        let preparation = collector.take_report(10.0).forward_pass_batches.unwrap();
        assert_eq!(preparation[0].request_instances, 4);
        collector.on_completed_forward(WorkerStage::Decode, 0, [snapshot(0, 0, 1, 20.0)].iter());
        let measured = collector.take_report(30.0).forward_pass_batches.unwrap();
        assert_eq!(measured[0].request_instances, 1);
        assert_eq!(measured[0].duration_ms, 20.0);
        assert_eq!(collector.finish().forward_pass_batches.unwrap(), Vec::new());
        assert!(
            crate::replay::report::TraceCollector::default()
                .finish()
                .forward_pass_batches
                .is_none()
        );
    }
}
