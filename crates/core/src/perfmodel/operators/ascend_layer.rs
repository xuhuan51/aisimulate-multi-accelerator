use serde::{Deserialize, Serialize};

use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::operators::{Op, PerformanceResult, Source};
use crate::perf_database::PerfDatabase;
use crate::perf_database::ascend_layer::{AscendLayerKey, AscendLayerTable};
use crate::perf_database::ascend_shape::AscendShapePolicy;

const LAYER_COUNTS: [(u32, u32); 8] = [
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 37),
    (5, 18),
    (6, 18),
    (77, 1),
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendDecoderLayersOp {
    pub name: String,
    pub scale_factor: f64,
    pub attention_tp: u32,
    pub attention_dp: u32,
    #[serde(default)]
    pub active_attention_dp: Option<u32>,
    #[serde(default)]
    pub population_transfer: bool,
    pub moe_ep: u32,
    pub quantization_sha256: String,
    pub covered_ops: Vec<Op>,
}

impl AscendDecoderLayersOp {
    pub fn validate(&self) -> Result<(), AicError> {
        self.key(0).validate()?;
        if !self.scale_factor.is_finite()
            || self.scale_factor < 0.0
            || self
                .covered_ops
                .iter()
                .any(|op| matches!(op, Op::AscendDecoderLayers(_)))
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid Ascend runtime-layer replacement".into(),
            ));
        }
        Ok(())
    }

    fn key(&self, layer: u32) -> AscendLayerKey {
        AscendLayerKey {
            attention_tp: self.attention_tp,
            attention_dp: self.attention_dp,
            active_attention_dp: self.active_attention_dp.unwrap_or(self.attention_dp),
            moe_ep: self.moe_ep,
            layer,
            quantization_sha256: self.quantization_sha256.clone(),
        }
    }

    pub fn ensure_ready(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate()?;
        if db.system != "ascend910c"
            || db.backend != "sglang"
            || !matches!(
                db.database_mode,
                DatabaseMode::Silicon | DatabaseMode::Hybrid
            )
        {
            return Err(AicError::EmpiricalNotImplemented(
                "Ascend runtime layers require measured 910C SGLang data".into(),
            ));
        }
        for layer in [0, 1, 2, 3, 4, 5, 6, 77] {
            db.ascend_layer.ensure_key(&self.key(layer))?;
        }
        Ok(())
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        requests: u32,
        kv_tokens: u32,
    ) -> Result<PerformanceResult, AicError> {
        self.ensure_ready(db)?;
        let mut source = self.timing_source(db.ascend_shape_policy)?;
        if requests == 0 || self.scale_factor == 0.0 {
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        let (latency, extrapolated) = self.estimate_layers(
            &db.ascend_layer,
            db.ascend_shape_policy,
            requests,
            kv_tokens,
            |sequence| self.context_attention_ms(db, requests, sequence),
        )?;
        if extrapolated {
            source = Source::Estimated;
        }
        Ok(PerformanceResult::new(latency, source).scaled(self.scale_factor))
    }

    fn context_attention_ms(
        &self,
        db: &PerfDatabase,
        requests: u32,
        kv_tokens: u32,
    ) -> Result<f64, AicError> {
        let mut attention_count = 0;
        let mut indexer_count = 0;
        let mut latency = 0.0;
        for operation in &self.covered_ops {
            let Op::AscendAttention(attention) = operation else {
                continue;
            };
            if attention.key.phase != "decode"
                || attention.key.execution_mode != "eager"
                || attention.eager_token_multiple != self.attention_tp
            {
                return Err(AicError::InvalidEngineConfig(
                    "Ascend runtime-layer context transfer requires the original eager attention identity".into(),
                ));
            }
            match attention.key.operation.as_str() {
                "sparse_attention"
                    if attention.key.local_heads == 64 / self.attention_tp
                        && attention.scale_factor == 78.0 =>
                {
                    attention_count += 1;
                }
                "lightning_indexer"
                    if attention.key.local_heads == 32 && attention.scale_factor == 21.0 =>
                {
                    indexer_count += 1;
                }
                _ => {
                    return Err(AicError::InvalidEngineConfig(
                        "Ascend runtime-layer context transfer requires 78 attention and 21 indexer calls".into(),
                    ));
                }
            }
            latency += attention.query(db, requests, kv_tokens, 0)?.latency_ms;
        }
        if (attention_count, indexer_count) != (1, 1) {
            return Err(AicError::InvalidEngineConfig(
                "Ascend runtime-layer context transfer requires both covered attention operators"
                    .into(),
            ));
        }
        Ok(latency)
    }

    fn estimate_layers(
        &self,
        table: &AscendLayerTable,
        policy: AscendShapePolicy,
        requests: u32,
        kv_tokens: u32,
        attention_ms: impl Fn(u32) -> Result<f64, AicError>,
    ) -> Result<(f64, bool), AicError> {
        if policy.is_measured() {
            return Ok((self.query_layers(table, requests, kv_tokens)?, false));
        }
        let mut minimum = 0;
        let mut maximum = u32::MAX;
        for (layer, _) in LAYER_COUNTS {
            let (lower, upper) = table.kv_bounds(&self.key(layer), requests)?;
            minimum = minimum.max(lower);
            maximum = maximum.min(upper);
        }
        if minimum > maximum || kv_tokens < minimum {
            return Err(AicError::PerfDatabase(format!(
                "Ascend runtime-layer KV={kv_tokens} lacks a common measured anchor [{minimum}, {maximum}]"
            )));
        }
        if kv_tokens <= maximum {
            return Ok((self.query_layers(table, requests, kv_tokens)?, false));
        }
        if maximum < 2048 {
            return Err(AicError::PerfDatabase(
                "Ascend context transfer cannot cross the DSA top-k=2048 execution boundary".into(),
            ));
        }
        let anchor = self.query_layers(table, requests, maximum)?;
        let latency = anchor + attention_ms(kv_tokens)? - attention_ms(maximum)?;
        if !latency.is_finite() || latency <= 0.0 {
            return Err(AicError::InvalidPerfData(
                "Ascend attention context transfer produced an invalid layer latency".into(),
            ));
        }
        Ok((latency, true))
    }

    fn timing_source(
        &self,
        policy: crate::perf_database::ascend_shape::AscendShapePolicy,
    ) -> Result<Source, AicError> {
        if self.population_transfer {
            if policy.is_measured() {
                return Err(AicError::InvalidForwardPassMetrics(
                    "Ascend skewed active-rank profiles require shape_policy=anchored".into(),
                ));
            }
            Ok(Source::Estimated)
        } else {
            Ok(Source::Silicon)
        }
    }

    fn query_layers(
        &self,
        table: &AscendLayerTable,
        requests: u32,
        kv_tokens: u32,
    ) -> Result<f64, AicError> {
        let mut latency = 0.0;
        for (layer, count) in LAYER_COUNTS {
            latency += table.query(&self.key(layer), requests, kv_tokens)? * f64::from(count);
        }
        Ok(latency)
    }

    pub fn weight_bytes(&self) -> f64 {
        self.covered_ops.iter().map(Op::weight_bytes).sum::<f64>() * self.scale_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::ascend_layer::{BASENAME, CONTRACT, SOURCE};
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};

    #[test]
    fn seventy_eight_measured_layers_replace_instead_of_adding_covered_timings() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        write_parquet(
            &path,
            &[
                Col::I64("layer", vec![0, 1, 2, 3, 4, 5, 6, 77]),
                Col::I64("attention_tp", vec![4; 8]),
                Col::I64("attention_dp", vec![4; 8]),
                Col::I64("moe_ep", vec![16; 8]),
                Col::I64("requests_per_dp", vec![1; 8]),
                Col::I64("kv_tokens", vec![8192; 8]),
                Col::F64("latency_ms", vec![2.0; 8]),
                Col::F64("cv", vec![0.01; 8]),
                Col::I64("sample_count", vec![9; 8]),
                Col::I64("numerical_checks", vec![144; 8]),
                Col::Str("boundary_contract", vec![CONTRACT; 8]),
                Col::Str("kernel_source", vec![SOURCE; 8]),
                Col::Str(
                    "runtime_identity_sha256",
                    vec!["2222222222222222222222222222222222222222222222222222222222222222"; 8],
                ),
                Col::Str(
                    "quantization_sha256",
                    vec!["1111111111111111111111111111111111111111111111111111111111111111"; 8],
                ),
            ],
        );
        let covered = Op::AscendEagerOverhead(
            crate::operators::ascend_eager_overhead::AscendEagerOverheadOp {
                name: "deliberately_unqueried_covered_timing".into(),
                scale_factor: 1.0,
                latency_ms: 1_000_000.0,
                calibration_id: "synthetic_no_double_count_oracle".into(),
                per_extra_active_rank_ms: 0.0,
                active_ranks: None,
            },
        );
        let layers = AscendDecoderLayersOp {
            name: "layers".into(),
            scale_factor: 1.0,
            attention_tp: 4,
            attention_dp: 4,
            active_attention_dp: None,
            population_transfer: false,
            moe_ep: 16,
            covered_ops: vec![covered],
            quantization_sha256: "1".repeat(64),
        };
        let table = AscendLayerTable::new(Some(path));
        assert_eq!(layers.query_layers(&table, 1, 8192).unwrap(), 156.0);
        let attention_oracle = |sequence| match sequence {
            8192 => Ok(8.22),
            12288 => Ok(12.54),
            _ => Err(AicError::PerfDatabase("unmeasured test attention".into())),
        };
        let (extended, estimated) = layers
            .estimate_layers(
                &table,
                AscendShapePolicy::Anchored,
                1,
                12288,
                attention_oracle,
            )
            .unwrap();
        assert!((extended - 160.32).abs() < 1e-10);
        assert!(estimated);
        assert_eq!(
            layers
                .estimate_layers(&table, AscendShapePolicy::Anchored, 1, 8192, |_| {
                    panic!("covered queries must not query covered kernels")
                })
                .unwrap(),
            (156.0, false)
        );
        for (policy, requests, context) in [
            (AscendShapePolicy::Measured, 1, 12288),
            (AscendShapePolicy::Anchored, 2, 12288),
            (AscendShapePolicy::Anchored, 1, 4096),
            (AscendShapePolicy::Anchored, 1, 16384),
        ] {
            assert!(
                layers
                    .estimate_layers(&table, policy, requests, context, attention_oracle)
                    .is_err()
            );
        }
        assert!(
            layers
                .query_layers(&table, 2, 8192)
                .unwrap_err()
                .is_missing_perf_data()
        );
        let mut partial = layers;
        use crate::perf_database::ascend_shape::AscendShapePolicy;
        assert_eq!(
            partial.timing_source(AscendShapePolicy::Measured).unwrap(),
            Source::Silicon
        );
        partial.population_transfer = true;
        assert!(partial.timing_source(AscendShapePolicy::Measured).is_err());
        assert_eq!(
            partial.timing_source(AscendShapePolicy::Anchored).unwrap(),
            Source::Estimated
        );
        partial.active_attention_dp = Some(1);
        assert!(
            partial
                .query_layers(&table, 1, 8192)
                .unwrap_err()
                .is_missing_perf_data()
        );
    }
}
