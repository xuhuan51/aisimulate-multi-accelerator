use serde::{Deserialize, Serialize};

use crate::AicError;
use crate::operators::{PerformanceResult, Source};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendEagerOverheadOp {
    pub name: String,
    pub scale_factor: f64,
    pub latency_ms: f64,
    pub calibration_id: String,
    #[serde(default)]
    pub per_extra_active_rank_ms: f64,
    #[serde(default)]
    pub active_ranks: Option<u32>,
}

impl AscendEagerOverheadOp {
    pub fn validate(&self) -> Result<(), AicError> {
        if self.calibration_id.trim().is_empty()
            || !self.latency_ms.is_finite()
            || self.latency_ms < 0.0
            || !self.per_extra_active_rank_ms.is_finite()
            || self.per_extra_active_rank_ms < 0.0
            || !self.scale_factor.is_finite()
            || self.scale_factor < 0.0
        {
            return Err(AicError::InvalidEngineConfig(
                "Invalid Ascend eager runtime overhead calibration".into(),
            ));
        }
        Ok(())
    }

    pub fn query(&self, requests: u32) -> Result<PerformanceResult, AicError> {
        self.validate()?;
        let latency = if requests > 0 {
            let active_ranks = self.active_ranks.unwrap_or(1);
            if active_ranks == 0 {
                return Err(AicError::InvalidForwardPassMetrics(
                    "Nonempty eager forward requires an active attention rank".into(),
                ));
            }
            (self.latency_ms + f64::from(active_ranks - 1) * self.per_extra_active_rank_ms)
                * self.scale_factor
        } else {
            0.0
        };
        Ok(PerformanceResult::new(latency, Source::Estimated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrated_elapsed_time_is_per_forward_and_empty_batches_cost_nothing() {
        let mut overhead = AscendEagerOverheadOp {
            name: "decode_eager_overhead".into(),
            scale_factor: 1.0,
            latency_ms: 2.5,
            calibration_id: "synthetic-two-and-a-half-millisecond-forward".into(),
            per_extra_active_rank_ms: 0.0,
            active_ranks: None,
        };
        assert_eq!(overhead.query(0).unwrap().latency_ms, 0.0);
        assert_eq!(overhead.query(1).unwrap().latency_ms, 2.5);
        assert_eq!(overhead.query(32).unwrap().latency_ms, 2.5);
        assert_eq!(overhead.query(1).unwrap().source, Source::Estimated);
        overhead.scale_factor = 2.0;
        assert_eq!(overhead.query(32).unwrap().latency_ms, 5.0);
        overhead.latency_ms = -1.0;
        assert!(overhead.query(1).is_err());
    }

    #[test]
    fn two_and_a_half_base_plus_three_quarters_per_extra_active_rank() {
        let mut overhead = AscendEagerOverheadOp {
            name: "synthetic-active-rank-forward".into(),
            scale_factor: 1.0,
            latency_ms: 2.5,
            calibration_id: "hand-derived-base-2.5-extra-rank-0.75".into(),
            per_extra_active_rank_ms: 0.75,
            active_ranks: Some(2),
        };
        assert_eq!(overhead.query(64).unwrap().latency_ms, 3.25);
        overhead.active_ranks = Some(4);
        assert_eq!(overhead.query(1).unwrap().latency_ms, 4.75);
        assert_eq!(overhead.query(0).unwrap().latency_ms, 0.0);
        overhead.active_ranks = Some(0);
        assert!(overhead.query(1).is_err());
        assert_eq!(overhead.query(0).unwrap().latency_ms, 0.0);
        overhead.per_extra_active_rank_ms = f64::NAN;
        assert!(overhead.validate().is_err());
    }
}
