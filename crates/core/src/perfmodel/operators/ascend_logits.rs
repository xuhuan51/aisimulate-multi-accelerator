use serde::{Deserialize, Serialize};

use crate::AicError;
use crate::operators::ascend_linear::{AscendLinearBias, AscendLinearTokens};
use crate::operators::{AscendLinearOp, AscendVectorOp, HcclOp, PerformanceResult, Source};
use crate::perf_database::PerfDatabase;
use crate::perf_database::ascend_linear::AscendLinearKey;
use crate::perf_database::ascend_vector::AscendVectorKey;
use crate::perf_database::hccl::HcclKey;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendLogitsOp {
    pub name: String,
    pub scale_factor: f64,
    pub global_tp: u32,
    pub attention_dp: u32,
    pub vocabulary: u32,
    pub hidden_size: u32,
    pub execution_mode: String,
    pub enable_all_to_all: bool,
    pub request_counts: Option<Vec<u32>>,
    pub rank: usize,
}

impl AscendLogitsOp {
    pub fn shape(&self, requests: u32) -> Result<(u32, u32, bool), AicError> {
        if !matches!(self.global_tp, 16 | 32)
            || self.attention_dp == 0
            || self.global_tp % self.attention_dp != 0
            || self.vocabulary != 154880
            || self.hidden_size != 6144
            || !matches!(self.execution_mode.as_str(), "eager" | "graph")
            || !self.scale_factor.is_finite()
            || self.scale_factor < 0.0
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid Ascend global logits geometry".into(),
            ));
        }
        let uniform;
        let counts = if let Some(counts) = &self.request_counts {
            counts.as_slice()
        } else {
            uniform = vec![requests; self.attention_dp as usize];
            &uniform
        };
        if counts.len() != self.attention_dp as usize || self.rank >= counts.len() {
            return Err(AicError::InvalidEngineConfig(
                "invalid Ascend logits request vector".into(),
            ));
        }
        let total = counts
            .iter()
            .try_fold(0u32, |sum, count| sum.checked_add(*count))
            .ok_or_else(|| {
                AicError::InvalidEngineConfig("Ascend logits row count overflow".into())
            })?;
        let local = counts[self.rank];
        let all_to_all = self.enable_all_to_all
            && self.attention_dp == self.global_tp
            && local > 0
            && counts.iter().all(|count| *count == local);
        Ok((total, local, all_to_all))
    }

    pub fn weight_bytes(&self) -> f64 {
        2.0 * f64::from(self.vocabulary / self.global_tp.max(1))
            * f64::from(self.hidden_size)
            * self.scale_factor
    }

    pub fn ensure_ready(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.shape(1)?;
        for operation in ["all_reduce", "all_gather"]
            .into_iter()
            .chain(self.enable_all_to_all.then_some("all_to_all"))
        {
            db.hccl.ensure_key(&HcclKey {
                dtype: "bfloat16".into(),
                operation: operation.into(),
                num_ranks: self.global_tp,
                ranks_per_node: self.global_tp.min(16),
                rank_layout: "contiguous".into(),
                execution_mode: self.execution_mode.clone(),
            })?;
        }
        db.ascend_linear.ensure_key(
            &AscendLinearKey {
                quant_mode: "bfloat16".into(),
                n: self.vocabulary / self.global_tp,
                k: self.hidden_size,
                weight_format: 2,
                execution_mode: self.execution_mode.clone(),
            },
            false,
        )?;
        db.ascend_vector.ensure_key(&AscendVectorKey {
            role: "logits_finalize".into(),
            dimensions: vec![1, self.vocabulary],
            execution_mode: self.execution_mode.clone(),
        })
    }

    pub fn query(&self, db: &PerfDatabase, requests: u32) -> Result<PerformanceResult, AicError> {
        let (total, local, all_to_all) = self.shape(requests)?;
        if total == 0 || self.scale_factor == 0.0 {
            return Ok(PerformanceResult::new(0.0, Source::Silicon));
        }
        let collective = |operation: &str, elements: u32| -> Result<PerformanceResult, AicError> {
            HcclOp {
                name: format!("{}_{}", self.name, operation),
                scale_factor: 1.0,
                key: HcclKey {
                    dtype: "bfloat16".into(),
                    operation: operation.into(),
                    num_ranks: self.global_tp,
                    ranks_per_node: self.global_tp.min(16),
                    rank_layout: "contiguous".into(),
                    execution_mode: self.execution_mode.clone(),
                },
                elements_per_token: elements,
                token_split: 1,
                token_axis: AscendLinearTokens::Tokens,
                dp_padding_mode: None,
            }
            .query(db, total)
        };
        let mut result = PerformanceResult::new(0.0, Source::Estimated);
        let mut add = |part: PerformanceResult| {
            result.latency_ms += part.latency_ms;
            result.source = result.source.combine(part.source);
        };
        if self.attention_dp > 1 {
            add(collective("all_reduce", self.hidden_size)?);
        }
        add(AscendLinearOp {
            name: format!("{}_gemm", self.name),
            scale_factor: 1.0,
            key: AscendLinearKey {
                quant_mode: "bfloat16".into(),
                n: self.vocabulary / self.global_tp,
                k: self.hidden_size,
                weight_format: 2,
                execution_mode: self.execution_mode.clone(),
            },
            bias: AscendLinearBias::Never,
            token_axis: AscendLinearTokens::Requests,
            token_split: 1,
            token_multiple: 1,
        }
        .query(db, total, total)?);
        add(collective(
            if all_to_all {
                "all_to_all"
            } else {
                "all_gather"
            },
            self.vocabulary / self.global_tp,
        )?);
        if local > 0 {
            add(AscendVectorOp {
                name: format!("{}_local_cast", self.name),
                scale_factor: 1.0,
                key: AscendVectorKey {
                    role: "logits_finalize".into(),
                    dimensions: vec![1, self.vocabulary],
                    execution_mode: self.execution_mode.clone(),
                },
                token_axis: AscendLinearTokens::Requests,
                token_split: 1,
                token_multiple: 1,
                resident_copies: None,
            }
            .query(db, local, local)?);
        }
        Ok(result.scaled(self.scale_factor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(counts: Vec<u32>) -> AscendLogitsOp {
        AscendLogitsOp {
            name: "logits".into(),
            scale_factor: 1.0,
            global_tp: 32,
            attention_dp: 32,
            vocabulary: 154880,
            hidden_size: 6144,
            execution_mode: "eager".into(),
            enable_all_to_all: true,
            request_counts: Some(counts),
            rank: 0,
        }
    }

    #[test]
    fn global_head_uses_unpadded_rows_and_equal_count_exchange_only() {
        let mut uneven = vec![0; 32];
        uneven[..4].fill(1);
        assert_eq!(logits(uneven).shape(99).unwrap(), (4, 1, false));
        assert_eq!(logits(vec![2; 32]).shape(99).unwrap(), (64, 2, true));
        assert_eq!(logits(vec![0; 32]).shape(99).unwrap(), (0, 0, false));
        assert_eq!(logits(vec![1; 32]).weight_bytes(), 59_473_920.0);
        assert!(logits(vec![1; 31]).shape(1).is_err());
        let operation = crate::operators::Op::AscendLogits(logits(vec![2; 32]));
        let encoded = serde_json::to_string(&operation).unwrap();
        let restored: crate::operators::Op = serde_json::from_str(&encoded).unwrap();
        assert_eq!(operation, restored);
    }
}
