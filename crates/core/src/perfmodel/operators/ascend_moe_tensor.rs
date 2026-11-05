use serde::{Deserialize, Serialize};

use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;
use crate::perf_database::ascend_moe_tensor::AscendMoeTensorKey;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendMoeTensorOp {
    pub name: String,
    pub scale_factor: f64,
    pub key: AscendMoeTensorKey,
    pub input_token_multiple: u32,
    #[serde(default)]
    pub routing_profile: Option<super::ascend_moe_routing::TensorMoeRoutingProfile>,
}

impl AscendMoeTensorOp {
    pub fn validate(&self) -> Result<(), AicError> {
        self.key.validate()?;
        if let Some(profile) = &self.routing_profile {
            profile.validate()?;
            if self.key.execution_mode != "graph"
                || self.key.intermediate_size != profile.intermediate_size
                || self.scale_factor != profile.layers.len() as f64
            {
                return Err(AicError::InvalidEngineConfig(
                    "tensor-MoE routing profile requires matching decode geometry and layer count"
                        .into(),
                ));
            }
        }
        if self.input_token_multiple == 0
            || !self.scale_factor.is_finite()
            || self.scale_factor < 0.0
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid tensor-MoE padding/scale policy".into(),
            ));
        }
        Ok(())
    }

    pub fn measured_tokens(&self, tokens: u32) -> Result<u32, AicError> {
        self.validate()?;
        tokens
            .div_ceil(self.input_token_multiple)
            .checked_mul(self.input_token_multiple)
            .ok_or_else(|| {
                AicError::InvalidEngineConfig("tensor-MoE token rounding overflow".into())
            })
    }

    fn validate_database(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate()?;
        if db.backend != "sglang" || db.system != "ascend910c" {
            return Err(AicError::InvalidEngineConfig(
                "Ascend tensor-MoE requires SGLang on910C".into(),
            ));
        }
        if self
            .routing_profile
            .as_ref()
            .is_some_and(|profile| profile.backend_version != db.version)
        {
            return Err(AicError::InvalidEngineConfig(
                "tensor-MoE routing profile backend version differs from the selected database"
                    .into(),
            ));
        }
        if !matches!(
            db.database_mode,
            DatabaseMode::Silicon | DatabaseMode::Hybrid
        ) {
            return Err(AicError::EmpiricalNotImplemented(
                "Ascend tensor-MoE requires measured data".into(),
            ));
        }
        Ok(())
    }

    pub fn ensure_ready(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate_database(db)?;
        if self.scale_factor == 0.0 || self.routing_profile.is_some() {
            return Ok(());
        }
        db.ascend_moe_tensor.ensure_key(&self.key)
    }

    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        self.validate_database(db)?;
        let count = self.measured_tokens(tokens)?;
        if count == 0 || self.scale_factor == 0.0 {
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        if let Some(profile) = &self.routing_profile {
            return Ok(
                PerformanceResult::new(profile.estimate(tokens, count)?, Source::Empirical)
                    .scaled(self.scale_factor),
            );
        }
        Ok(db
            .ascend_moe_tensor
            .estimate(&self.key, count, db.ascend_shape_policy, &db.system_spec)?
            .result()
            .scaled(self.scale_factor))
    }

    pub fn weight_bytes(&self) -> f64 {
        let intermediate = f64::from(self.key.intermediate_size);
        256.0
            * (6144.0 * intermediate * 3.0 + (2.0 * intermediate + 6144.0) * 6.0)
            * self.scale_factor
    }
}
