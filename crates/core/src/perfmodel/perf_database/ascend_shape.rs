use serde::{Deserialize, Serialize};

use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::operators::base::{PerformanceResult, Source};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AscendShapePolicy {
    #[default]
    Measured,
    Anchored,
}

impl AscendShapePolicy {
    pub fn is_measured(&self) -> bool {
        *self == Self::Measured
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ShapeEstimate {
    pub latency_ms: f64,
    pub extrapolated: bool,
}

impl ShapeEstimate {
    pub fn measured(latency_ms: f64) -> Self {
        Self {
            latency_ms,
            extrapolated: false,
        }
    }

    pub fn result(self) -> PerformanceResult {
        PerformanceResult::new(
            self.latency_ms,
            if self.extrapolated {
                Source::Estimated
            } else {
                Source::Silicon
            },
        )
    }

    pub fn blend(self, other: Self, weight: f64) -> Self {
        Self {
            latency_ms: self.latency_ms + (other.latency_ms - self.latency_ms) * weight,
            extrapolated: self.extrapolated || other.extrapolated,
        }
    }

    pub fn scale_work(self, query: f64, anchor: f64, floor: f64) -> Result<Self, AicError> {
        if !query.is_finite() || query <= 0.0 || !anchor.is_finite() || anchor <= 0.0 {
            return Err(AicError::InvalidEngineConfig(
                "invalid Ascend analytical work estimate".into(),
            ));
        }
        let fixed = floor.min(self.latency_ms);
        let latency_ms = (fixed + (self.latency_ms - fixed) * (query / anchor)).max(floor);
        if !latency_ms.is_finite() || latency_ms <= 0.0 {
            return Err(AicError::InvalidEngineConfig(
                "Ascend anchored latency overflow".into(),
            ));
        }
        Ok(Self {
            latency_ms,
            extrapolated: true,
        })
    }
}

pub fn roofline_ms(spec: &SystemSpec, flops: f64, bytes: f64, int8: bool) -> Result<f64, AicError> {
    let rate = if int8 {
        spec.gpu.int8_tc_flops
    } else {
        spec.gpu.bfloat16_tc_flops
    };
    let rate = rate
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| {
            AicError::MissingSystemFlops(
                "Ascend anchored prediction requires calibrated compute throughput".into(),
            )
        })?;
    if !spec.gpu.mem_bw.is_finite() || spec.gpu.mem_bw <= 0.0 {
        return Err(AicError::InvalidEngineConfig(
            "Ascend anchored prediction requires positive HBM bandwidth".into(),
        ));
    }
    Ok((flops / rate).max(bytes / spec.gpu.mem_bw) * 1000.0)
}

pub fn linear_work(
    spec: &SystemSpec,
    tokens: f64,
    width: f64,
    inner: f64,
    int8: bool,
) -> Result<f64, AicError> {
    let weight_bytes = if int8 { 1.0 } else { 2.0 };
    roofline_ms(
        spec,
        2.0 * tokens * width * inner,
        weight_bytes * width * inner + 2.0 * tokens * (width + inner),
        int8,
    )
}

pub fn anchored_axis(
    coordinate: u32,
    minimum: u32,
    maximum: u32,
    lookup: &dyn Fn(u32) -> Result<f64, AicError>,
    work: &dyn Fn(f64) -> Result<f64, AicError>,
) -> Result<ShapeEstimate, AicError> {
    if coordinate == 0 {
        return Err(AicError::InvalidEngineConfig(
            "Ascend extrapolation requires a positive coordinate".into(),
        ));
    }
    let anchor = coordinate.clamp(minimum, maximum);
    let measured = lookup(anchor)?;
    ShapeEstimate::measured(measured).scale_work(
        work(f64::from(coordinate))?,
        work(f64::from(anchor))?,
        lookup(minimum)?.min(measured),
    )
}

pub(crate) fn sparse_prefill_pairs(queries: u32, prefix: u32) -> f64 {
    let queries = f64::from(queries);
    let prefix = f64::from(prefix);
    let unsaturated = (2048.0 - prefix).max(0.0).min(queries);
    unsaturated * (prefix + (unsaturated + 1.0) / 2.0) + (queries - unsaturated) * 2048.0
}

#[allow(clippy::too_many_arguments)]
pub fn attention_work(
    spec: &SystemSpec,
    indexer: bool,
    prefill: bool,
    heads: u32,
    batch: u32,
    sequence: u32,
    prefix: u32,
    dcp: u32,
) -> Result<f64, AicError> {
    let queries = if prefill { f64::from(sequence) } else { 1.0 };
    let context = if prefill {
        f64::from(prefix) + queries
    } else {
        f64::from(sequence)
    };
    let degree = f64::from(dcp);
    let causal_pairs = if prefill {
        queries * (f64::from(prefix) + (queries + 1.0) / 2.0)
    } else {
        context
    };
    let pairs = if indexer {
        causal_pairs
    } else if prefill {
        sparse_prefill_pairs(sequence, prefix)
    } else {
        context.min(2048.0)
    };
    let width = if indexer { 128.0 } else { 576.0 };
    let per_batch = roofline_ms(
        spec,
        if indexer {
            2.0 * f64::from(heads) * width * pairs
        } else {
            2.0 * f64::from(heads) * degree * (576.0 + 512.0) * pairs / degree
        },
        2.0 * (width
            * if indexer {
                context
            } else {
                context.min(2048.0) / degree
            }
            + queries * f64::from(heads) * width),
        false,
    )?;
    let storage = if dcp > 1 {
        2.0 * queries * (576.0 + 128.0) / spec.gpu.mem_bw * 1000.0
    } else {
        0.0
    };
    let communication = if dcp > 1 {
        if !spec.node.intra_node_bw.is_finite() || spec.node.intra_node_bw <= 0.0 {
            return Err(AicError::InvalidEngineConfig(
                "DCP anchored prediction requires measured intra-node bandwidth".into(),
            ));
        }
        queries * f64::from(heads) * (degree - 1.0) * (2.0 * 576.0 + 4.0 * 513.0)
            / spec.node.intra_node_bw
            * 1000.0
    } else {
        0.0
    };
    Ok(f64::from(batch) * (per_batch + storage + communication))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles_variable_work_and_retains_the_measured_fixed_component() {
        let anchor = ShapeEstimate::measured(3.0);
        assert_eq!(anchor.scale_work(8.0, 4.0, 1.0).unwrap().latency_ms, 5.0);
        assert_eq!(anchor.scale_work(0.1, 4.0, 1.0).unwrap().latency_ms, 1.05);
        assert_eq!(
            anchor.scale_work(8.0, 4.0, 1.0).unwrap().result().source,
            Source::Estimated
        );
        assert!(anchor.scale_work(f64::INFINITY, 4.0, 1.0).is_err());
        assert!(anchor.scale_work(4.0, 0.0, 1.0).is_err());
    }
}
