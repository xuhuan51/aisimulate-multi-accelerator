use serde::{Deserialize, Serialize};

use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;
use crate::perf_database::ascend_attention::AscendAttentionKey;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendAttentionOp {
    pub name: String,
    pub scale_factor: f64,
    pub key: AscendAttentionKey,
    pub graph_buckets: Vec<u32>,
    #[serde(default = "default_eager_token_multiple")]
    pub eager_token_multiple: u32,
}

fn default_eager_token_multiple() -> u32 {
    1
}

impl AscendAttentionOp {
    pub fn validate(&self) -> Result<(), AicError> {
        self.key.validate()?;
        let graph_decode = self.key.phase == "decode" && self.key.execution_mode == "graph";
        if !self.scale_factor.is_finite()
            || self.scale_factor < 0.0
            || self.eager_token_multiple == 0
            || graph_decode == self.graph_buckets.is_empty()
            || self.graph_buckets.iter().any(|value| *value == 0)
            || self.graph_buckets.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid Ascend attention graph/scale policy".into(),
            ));
        }
        Ok(())
    }

    pub fn padded_batch(&self, batch: u32) -> Result<u32, AicError> {
        self.validate()?;
        if self.key.phase == "decode" && self.key.execution_mode == "graph" {
            self.graph_buckets
                .iter()
                .copied()
                .find(|bucket| *bucket >= batch)
                .ok_or_else(|| {
                    AicError::PerfDatabase(format!(
                        "Ascend attention batch={batch} exceeds graph buckets"
                    ))
                })
        } else if self.key.phase == "decode" {
            batch
                .div_ceil(self.eager_token_multiple)
                .checked_mul(self.eager_token_multiple)
                .ok_or_else(|| {
                    AicError::InvalidEngineConfig("Ascend attention padding overflow".into())
                })
        } else {
            Ok(batch)
        }
    }

    fn validate_database(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate()?;
        if db.backend != "sglang" {
            return Err(AicError::InvalidEngineConfig(
                "Ascend attention requires SGLang".into(),
            ));
        }
        if !matches!(
            db.database_mode,
            DatabaseMode::Silicon | DatabaseMode::Hybrid
        ) {
            return Err(AicError::EmpiricalNotImplemented(
                "Ascend attention requires measured data; no analytic fallback".into(),
            ));
        }
        Ok(())
    }

    pub fn ensure_ready(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate_database(db)?;
        if self.scale_factor == 0.0 {
            return Ok(());
        }
        db.ascend_attention.ensure_key(&self.key)
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch: u32,
        sequence: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        self.validate_database(db)?;
        if batch == 0 || self.scale_factor == 0.0 {
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        let value = db.ascend_attention.estimate(
            &self.key,
            batch,
            self.padded_batch(batch)?,
            sequence,
            if self.key.phase == "prefill" {
                prefix
            } else {
                0
            },
            db.ascend_shape_policy,
            &db.system_spec,
        )?;
        Ok(value.result().scaled(self.scale_factor))
    }

    pub fn query_eager(
        &self,
        db: &PerfDatabase,
        batch: u32,
        sequence: u32,
    ) -> Result<PerformanceResult, AicError> {
        let mut eager = self.clone();
        eager.key.execution_mode = "eager".into();
        eager.graph_buckets.clear();
        eager.query(db, batch, sequence, 0)
    }
}
