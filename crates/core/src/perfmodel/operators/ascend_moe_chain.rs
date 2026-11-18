use serde::{Deserialize, Serialize};

use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;
use crate::perf_database::ascend_moe_chain::AscendMoeChainKey;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeExpertRank {
    pub received_tokens: u32,
    pub active_experts: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeExpertSample {
    pub request_counts: Vec<u32>,
    pub sequence_length: u32,
    pub layers: Vec<Vec<DecodeExpertRank>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeExpertWorkload {
    pub observation_id: String,
    pub quantization_sha256: String,
    pub framework_commit: String,
    pub samples_sha256: String,
    pub input_tokens: u32,
    pub graph_bucket: u32,
    pub history_policy: String,
    pub samples: Vec<DecodeExpertSample>,
}

impl DecodeExpertWorkload {
    pub fn expert_geometry(&self) -> (u32, usize, u64) {
        // The bucket identifies the two audited capture contracts. Assignment
        // counts include the Graph padding actually sent through EP16.
        if self.graph_bucket == 4 {
            (32, 60, 160)
        } else {
            (16, 75, 256)
        }
    }

    pub fn validate(&self) -> Result<(), AicError> {
        use sha2::{Digest, Sha256};
        let invalid =
            || AicError::InvalidEngineConfig("invalid Graph Decode expert workload".into());
        let (local_experts, layers, assignments) = self.expert_geometry();
        if self.observation_id.trim().is_empty()
            || self.quantization_sha256.len() != 64
            || !self
                .quantization_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
            || self.framework_commit.len() != 40
            || !self.framework_commit.bytes().all(|b| b.is_ascii_hexdigit())
            || self.input_tokens == 0
            || !matches!(self.graph_bucket, 4 | 8)
            || !matches!(self.history_policy.as_str(), "deny" | "hold_boundary")
            || self.samples.is_empty()
        {
            return Err(invalid());
        }
        for sample in &self.samples {
            if sample.request_counts.len() != 4
                || sample.request_counts.iter().any(|n| *n > self.graph_bucket)
                || sample.request_counts.iter().all(|n| *n == 0)
                || sample.sequence_length <= self.input_tokens
                || sample.layers.len() != layers
                || sample.layers.iter().any(|ranks| {
                    ranks.len() != 16
                        || ranks
                            .iter()
                            .map(|r| u64::from(r.received_tokens))
                            .sum::<u64>()
                            != assignments
                        || ranks.iter().any(|r| {
                            r.active_experts > r.received_tokens.min(local_experts)
                                || (r.received_tokens == 0) != (r.active_experts == 0)
                        })
                })
            {
                return Err(invalid());
            }
        }
        let bytes = serde_json::to_vec(&self.samples).map_err(|_| invalid())?;
        if format!("{:x}", Sha256::digest(bytes)) != self.samples_sha256 {
            return Err(AicError::InvalidEngineConfig(
                "Decode expert workload sample SHA256 mismatch".into(),
            ));
        }
        Ok(())
    }

    pub fn cache_key(&self) -> String {
        format!(
            "{:?}:{}:{}:{}:{}:{}:{}",
            self.observation_id,
            self.quantization_sha256,
            self.framework_commit,
            self.samples_sha256,
            self.input_tokens,
            self.graph_bucket,
            self.history_policy
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendMoeChainOp {
    pub name: String,
    pub scale_factor: f64,
    pub key: AscendMoeChainKey,
    pub token_numerator: u32,
    pub token_denominator: u32,
    pub input_token_multiple: u32,
    #[serde(default)]
    pub decode_workload: Option<DecodeExpertWorkload>,
    #[serde(default)]
    pub profile_request_counts: Option<Vec<u32>>,
}

impl AscendMoeChainOp {
    pub fn validate(&self) -> Result<(), AicError> {
        self.key.validate()?;
        if self.profile_request_counts.as_ref().is_some_and(|counts| {
            self.decode_workload.is_none() || counts.len() != 4 || counts.iter().any(|n| *n > 8)
        }) {
            return Err(AicError::InvalidEngineConfig(
                "invalid Decode workload request vector".into(),
            ));
        }
        if self.token_numerator == 0
            || self.token_denominator == 0
            || self.input_token_multiple == 0
            || !self.scale_factor.is_finite()
            || self.scale_factor < 0.0
            || self.decode_workload.as_ref().is_some_and(|workload| {
                self.key.phase != "decode"
                    || self.key.execution_mode != "graph"
                    || self.key.local_experts != workload.expert_geometry().0
                    || self.key.distribution != "balanced"
                    || self.key.active_experts.is_some()
            })
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid Ascend MoE token/scale policy".into(),
            ));
        }
        Ok(())
    }

    pub fn received_tokens(&self, tokens: u32) -> Result<u32, AicError> {
        self.validate()?;
        let padded = tokens
            .div_ceil(self.input_token_multiple)
            .checked_mul(self.input_token_multiple)
            .ok_or_else(|| {
                AicError::InvalidEngineConfig("Ascend MoE token rounding overflow".into())
            })?;
        let routed = u64::from(padded) * u64::from(self.token_numerator);
        u32::try_from(routed.div_ceil(u64::from(self.token_denominator)))
            .map_err(|_| AicError::InvalidEngineConfig("Ascend MoE receive-count overflow".into()))
    }

    fn validate_database(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate()?;
        if db.backend != "sglang" {
            return Err(AicError::InvalidEngineConfig(
                "Ascend MoE chain requires SGLang".into(),
            ));
        }
        if !matches!(
            db.database_mode,
            DatabaseMode::Silicon | DatabaseMode::Hybrid
        ) {
            return Err(AicError::EmpiricalNotImplemented(
                "Ascend MoE chain requires measured data; no analytic fallback".into(),
            ));
        }
        Ok(())
    }

    pub fn ensure_ready(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate_database(db)?;
        if self.scale_factor == 0.0 {
            return Ok(());
        }
        db.ascend_moe_chain.ensure_key(&self.key)?;
        if let Some(workload) = &self.decode_workload {
            db.ascend_moe_chain.prepare_workload(
                &self.key,
                workload,
                db.ascend_shape_policy,
                &db.system_spec,
            )?;
        }
        Ok(())
    }

    pub fn query_runtime(
        &self,
        db: &PerfDatabase,
        ctx: &crate::operators::RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        self.query_runtime_with_request_counts(db, ctx, self.profile_request_counts.as_deref())
    }

    pub(crate) fn query_runtime_with_request_counts(
        &self,
        db: &PerfDatabase,
        ctx: &crate::operators::RuntimeContext,
        request_counts: Option<&[u32]>,
    ) -> Result<PerformanceResult, AicError> {
        self.validate_database(db)?;
        if self.scale_factor == 0.0 || ctx.num_tokens == 0 {
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        if let Some(workload) = &self.decode_workload {
            let uniform;
            let requests = match request_counts {
                Some(counts) => counts,
                None => {
                    uniform = vec![ctx.batch_size; 4];
                    &uniform
                }
            };
            if self.received_tokens(ctx.num_tokens)? == (workload.expert_geometry().2 / 16) as u32 {
                if let Some(value) = db.ascend_moe_chain.estimate_workload(
                    &self.key,
                    workload,
                    requests,
                    ctx.s,
                    db.ascend_shape_policy,
                    &db.system_spec,
                )? {
                    return Ok(value.result().scaled(self.scale_factor));
                }
            }
        }
        self.query(db, ctx.num_tokens)
    }

    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        self.validate_database(db)?;
        let received = self.received_tokens(tokens)?;
        if received == 0 || self.scale_factor == 0.0 {
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        Ok(db
            .ascend_moe_chain
            .estimate(&self.key, received, db.ascend_shape_policy, &db.system_spec)?
            .result()
            .scaled(self.scale_factor))
    }

    pub fn weight_bytes(&self) -> f64 {
        self.key.parameter_bytes() * self.scale_factor
    }
}
