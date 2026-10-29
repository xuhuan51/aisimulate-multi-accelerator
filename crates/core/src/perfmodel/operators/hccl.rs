use serde::{Deserialize, Serialize};

use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::operators::ascend_linear::AscendLinearTokens;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;
use crate::perf_database::hccl::HcclKey;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DpPaddingMode {
    MaxLen,
    SumLen,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HcclOp {
    pub name: String,
    pub scale_factor: f64,
    pub key: HcclKey,
    pub elements_per_token: u32,
    pub token_split: u32,
    #[serde(default)]
    pub token_axis: AscendLinearTokens,
    #[serde(default)]
    pub dp_padding_mode: Option<DpPaddingMode>,
}

impl HcclOp {
    pub fn payload_bytes(&self, num_tokens: u32) -> Result<u64, AicError> {
        self.key.validate()?;
        if self.token_split == 0
            || self.elements_per_token == 0
            || !self.scale_factor.is_finite()
            || self.scale_factor < 0.0
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid HCCL token geometry or scale".into(),
            ));
        }
        let element_bytes = if matches!(self.key.dtype.as_str(), "bfloat16" | "float16") {
            2
        } else {
            4
        };
        u64::from(num_tokens.div_ceil(self.token_split))
            .checked_mul(u64::from(self.elements_per_token))
            .and_then(|elements| elements.checked_mul(element_bytes))
            .ok_or_else(|| AicError::InvalidEngineConfig("HCCL payload overflow".into()))
    }

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        if self.dp_padding_mode == Some(DpPaddingMode::SumLen) {
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        let bytes = self.payload_bytes(num_tokens)?;
        if bytes == 0 || self.scale_factor == 0.0 {
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        match db.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => {
                let ranks = f64::from(self.key.num_ranks);
                let bandwidth = if self.key.num_ranks == self.key.ranks_per_node {
                    db.system_spec.node.intra_node_bw
                } else {
                    db.system_spec.node.inter_node_bw
                };
                let factor = match self.key.operation.as_str() {
                    "all_reduce" => 2.0 * (ranks - 1.0) / ranks,
                    "all_gather" => ranks - 1.0,
                    _ => (ranks - 1.0) / ranks,
                };
                Ok(
                    PerformanceResult::new(bytes as f64 * factor / bandwidth * 1000.0, Source::Sol)
                        .scaled(self.scale_factor),
                )
            }
            DatabaseMode::Empirical => Err(AicError::EmpiricalNotImplemented(
                "HCCL requires its measured tuple".into(),
            )),
            _ => {
                if !db.ascend_shape_policy.is_measured() {
                    return Ok(db
                        .hccl
                        .estimate(&self.key, bytes, db.ascend_shape_policy, &db.system_spec)?
                        .result()
                        .scaled(self.scale_factor));
                }
                let value = db.hccl.query(&self.key, bytes)?;
                Ok(
                    PerformanceResult::with_energy(value.latency, value.energy, Source::Silicon)
                        .scaled(self.scale_factor),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_is_input_bytes_and_rounds_token_shards_up() {
        let mut op = HcclOp {
            name: "attention_gather".into(),
            scale_factor: 78.0,
            key: HcclKey {
                dtype: "bfloat16".into(),
                operation: "all_gather".into(),
                num_ranks: 8,
                ranks_per_node: 8,
                rank_layout: "contiguous".into(),
                execution_mode: "graph".into(),
            },
            elements_per_token: 6144,
            token_split: 8,
            token_axis: AscendLinearTokens::Tokens,
            dp_padding_mode: None,
        };
        assert_eq!(op.payload_bytes(9).unwrap(), 24576);
        assert_eq!(op.payload_bytes(0).unwrap(), 0);
        op.key.dtype = "int32".into();
        assert_eq!(op.payload_bytes(9).unwrap(), 49152);
        op.token_split = 0;
        assert!(op.payload_bytes(9).is_err());
    }
}
