use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::SourceResolver;
use super::axis_curve::AxisCurve;
use super::parquet_loader::PerfReader;
use crate::common::error::AicError;

pub const BASENAME: &str = "ascend_vector_perf.parquet";
pub const CONTRACT: &str = "glm52_modelslim_vectors/v1";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendVectorKey {
    pub role: String,
    pub dimensions: Vec<u32>,
    pub execution_mode: String,
}

impl AscendVectorKey {
    pub fn validate(&self) -> Result<(), AicError> {
        let valid = match (self.role.as_str(), self.dimensions.as_slice()) {
            (
                "rms_bias" | "add_rms_bias" | "cast_hidden_fp32" | "routed_scale" | "shared_add",
                [6144],
            )
            | ("split_qk_norm", [2048, 512, 64])
            | ("indexer_norm", [128])
            | ("rope_indexer", [32, 64, 128, 128])
            | ("concat_indexer_q", [32, 64, 128])
            | ("concat_indexer_k", [1, 64, 128])
            | ("router_topk", [256, 8])
            | ("cast_weights_bf16", [32])
            | ("rope_cache_gather", [1048576, 64]) => true,
            ("seq_lens_h2d", [4]) => self.execution_mode == "eager",
            ("rope_attention", [heads, 64, 256, 64]) => matches!(heads, 4 | 8 | 16 | 32 | 64),
            ("swiglu", [width]) => {
                matches!(width, 64 | 128 | 256 | 384 | 512 | 768 | 1536 | 2048 | 3072)
            }
            ("logits_finalize", [parallel, 154880]) => matches!(parallel, 1 | 2 | 4 | 8 | 16),
            ("cache_scatter", [53376, width]) => matches!(width, 64 | 128 | 512),
            ("embedding", [vocabulary, 6144]) => {
                matches!(vocabulary, 9680 | 19360 | 38720 | 77440 | 154880)
            }
            _ => false,
        };
        if !valid || !matches!(self.execution_mode.as_str(), "eager" | "graph") {
            return Err(AicError::InvalidEngineConfig(format!(
                "invalid Ascend vector identity: {self:?}"
            )));
        }
        Ok(())
    }

    pub fn kernel_source(&self) -> &'static str {
        match self.role.as_str() {
            "rms_bias" | "add_rms_bias" => {
                "sglang.srt.layers.quantization.modelslim.modelslim.npu_wrapper_rmsnorm_forward"
            }
            "split_qk_norm" => "sgl_kernel_npu.norm.fused_split_qk_norm.fused_split_qk_norm",
            "indexer_norm" => "sglang.srt.layers.layernorm.LayerNorm.forward_npu",
            "rope_attention" | "rope_indexer" => {
                "sgl_kernel_npu.norm.fused_rope_qk_mqa.fused_rope_qk_mqa"
            }
            "concat_indexer_q" | "concat_indexer_k" => "torch.cat",
            "cache_scatter" => "torch_npu.npu_scatter_nd_update_",
            "swiglu" => "torch_npu.npu_swiglu",
            "router_topk" => "torch.ops.npu.npu_moe_gating_top_k",
            "logits_finalize" => {
                "sglang.srt.layers.logits_processor.LogitsProcessor._copy_logits_to_buffer"
            }
            "cast_hidden_fp32" | "cast_weights_bf16" | "seq_lens_h2d" => "torch.Tensor.to",
            "embedding" => "torch.nn.functional.embedding",
            "rope_cache_gather" => "torch.Tensor.index_select",
            "routed_scale" => "torch.Tensor.mul_",
            "shared_add" => "torch.Tensor.add_",
            _ => "unsupported",
        }
    }

    pub fn parameter_bytes(&self) -> f64 {
        match (self.role.as_str(), self.dimensions.as_slice()) {
            ("rms_bias" | "add_rms_bias", [width]) => 4.0 * f64::from(*width),
            ("split_qk_norm", [query, key, _]) => 4.0 * (f64::from(*query) + f64::from(*key)),
            ("indexer_norm", [width]) => 8.0 * f64::from(*width),
            ("router_topk", [experts, _]) => 4.0 * f64::from(*experts),
            ("embedding" | "rope_cache_gather", [rows, width]) => {
                2.0 * f64::from(*rows) * f64::from(*width)
            }
            _ => 0.0,
        }
    }
}

struct Curve {
    values: AxisCurve,
    minimum: u32,
    maximum: u32,
}

type Grids = BTreeMap<AscendVectorKey, Curve>;

pub struct AscendVectorTable {
    path: Option<PathBuf>,
    grids: OnceLock<Result<Grids, (bool, String)>>,
}

impl AscendVectorTable {
    pub fn estimate(
        &self,
        key: &AscendVectorKey,
        tokens: u32,
        policy: super::ascend_shape::AscendShapePolicy,
    ) -> Result<super::ascend_shape::ShapeEstimate, AicError> {
        match self.query(key, tokens) {
            Ok(value) => return Ok(super::ascend_shape::ShapeEstimate::measured(value)),
            Err(error)
                if policy == super::ascend_shape::AscendShapePolicy::Anchored
                    && error.is_missing_perf_data() => {}
            Err(error) => return Err(error),
        }
        let curve = self.curve(key)?;
        super::ascend_shape::anchored_axis(
            tokens,
            curve.minimum,
            curve.maximum,
            &|value| self.query(key, value),
            &|count| Ok(count),
        )
    }
    pub fn new(path: Option<PathBuf>) -> Self {
        Self {
            path,
            grids: OnceLock::new(),
        }
    }

    pub fn with_sources(data_root: &Path, resolver: &SourceResolver) -> Result<Self, AicError> {
        let primary = resolver
            .prioritized_sources_for(BASENAME, data_root)?
            .into_iter()
            .find(|source| source.channel == "primary");
        if primary
            .as_ref()
            .is_some_and(|source| source.source.kernel_sources().is_some())
        {
            return Err(AicError::InvalidPerfData(
                "Ascend vector rejects filtered primary sources".into(),
            ));
        }
        let path = primary.map(|source| source.source.0);
        if let Some(path) = &path {
            let family = data_root
                .parent()
                .and_then(Path::parent)
                .and_then(|system| {
                    Some(
                        system
                            .join("vector")
                            .join(data_root.parent()?.file_name()?)
                            .join(data_root.file_name()?)
                            .join(BASENAME),
                    )
                });
            if path != &data_root.join(BASENAME) && family.as_ref() != Some(path) {
                return Err(AicError::InvalidPerfData(
                    "Ascend vector requires the exact requested system/backend/version".into(),
                ));
            }
        }
        Ok(Self::new(path))
    }

    fn curve(&self, key: &AscendVectorKey) -> Result<&Curve, AicError> {
        key.validate()?;
        let grids = self
            .grids
            .get_or_init(|| {
                let Some(path) = self.path.as_ref().filter(|path| path.is_file()) else {
                    return Err((false, "Ascend vector primary data is missing".into()));
                };
                load(path).map_err(|error| (true, error.to_string()))
            })
            .as_ref()
            .map_err(|(invalid, message)| {
                if *invalid {
                    AicError::InvalidPerfData(message.clone())
                } else {
                    AicError::PerfDatabase(message.clone())
                }
            })?;
        grids.get(key).ok_or_else(|| {
            AicError::PerfDatabase(format!("unmeasured Ascend vector tuple: {key:?}"))
        })
    }

    pub fn ensure_key(&self, key: &AscendVectorKey) -> Result<(), AicError> {
        self.curve(key).map(|_| ())
    }

    pub fn query(&self, key: &AscendVectorKey, tokens: u32) -> Result<f64, AicError> {
        let curve = self.curve(key)?;
        if tokens < curve.minimum || tokens > curve.maximum {
            return Err(AicError::PerfDatabase(format!(
                "Ascend vector tokens={tokens} outside measured [{}, {}]",
                curve.minimum, curve.maximum
            )));
        }
        curve.values.query(f64::from(tokens), &|_| 1.0)
    }
}

fn load(path: &Path) -> Result<Grids, AicError> {
    let reader = PerfReader::open(path)?;
    let role = reader.col("role")?;
    let dimensions = reader.col("dimensions")?;
    let mode = reader.col("execution_mode")?;
    let tokens = reader.col("tokens")?;
    let kernel = reader.col("kernel_source")?;
    let contract = reader.col("boundary_contract")?;
    let signature = reader.col("signature_sha256")?;
    let latency = reader.col("latency_ms")?;
    let cv = reader.col("cv")?;
    let checks = reader.col("numerical_checks")?;
    let samples = reader.col("sample_count")?;
    let verification = reader.col("execution_verification")?;
    let provenance = reader.col("runtime_identity_sha256")?;
    let mut raw: BTreeMap<AscendVectorKey, BTreeMap<u32, f64>> = BTreeMap::new();
    let mut identities: BTreeMap<AscendVectorKey, String> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = AscendVectorKey {
            role: row.str_owned(role)?,
            dimensions: serde_json::from_str(row.str(dimensions)?).map_err(|error| {
                AicError::InvalidPerfData(format!("invalid vector dimensions: {error}"))
            })?,
            execution_mode: row.str_owned(mode)?,
        };
        key.validate()?;
        let count = row.u32(tokens)?;
        let measured = row.f64(latency)?;
        let variation = row.f64(cv)?;
        let runtime = row.str(provenance)?;
        let signature_hash = row.str(signature)?;
        let expected = if key.execution_mode == "graph" {
            "captured_graph_replay"
        } else {
            "eager_invocation"
        };
        if count == 0
            || row.str(contract)? != CONTRACT
            || row.str(kernel)? != key.kernel_source()
            || !measured.is_finite()
            || measured <= 0.0
            || !variation.is_finite()
            || !(0.0..=0.1).contains(&variation)
            || row.u32(checks)? == 0
            || row.u32(samples)? < 3
            || row.str(verification)? != expected
            || [runtime, signature_hash].iter().any(|value| {
                value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
        {
            return Err(AicError::InvalidPerfData(format!(
                "unqualified Ascend vector row: {key:?}"
            )));
        }
        let expected = identities
            .entry(key.clone())
            .or_insert_with(|| runtime.to_owned());
        if expected != runtime {
            return Err(AicError::InvalidPerfData(
                "Ascend vector curve mixes runtime identities".into(),
            ));
        }
        if raw
            .entry(key)
            .or_default()
            .insert(count, measured)
            .is_some()
        {
            return Err(AicError::InvalidPerfData(
                "duplicate Ascend vector physical key".into(),
            ));
        }
    }
    if raw.is_empty() {
        return Err(AicError::InvalidPerfData(
            "empty Ascend vector table".into(),
        ));
    }
    Ok(raw
        .into_iter()
        .map(|(key, points)| {
            (
                key,
                Curve {
                    minimum: *points.first_key_value().unwrap().0,
                    maximum: *points.last_key_value().unwrap().0,
                    values: AxisCurve::from_map("tokens", points),
                },
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::ascend_linear::AscendLinearTokens;
    use crate::operators::ascend_vector::AscendVectorOp;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
    use crate::perf_database::{PerfDbSources, PerfSource};

    fn key() -> AscendVectorKey {
        AscendVectorKey {
            role: "rms_bias".into(),
            dimensions: vec![6144],
            execution_mode: "graph".into(),
        }
    }

    fn columns() -> Vec<Col> {
        vec![
            Col::Str("role", vec!["rms_bias"; 2]),
            Col::Str("dimensions", vec!["[6144]"; 2]),
            Col::Str("execution_mode", vec!["graph"; 2]),
            Col::I64("tokens", vec![1, 9]),
            Col::Str(
                "kernel_source",
                vec![
                    "sglang.srt.layers.quantization.modelslim.modelslim.npu_wrapper_rmsnorm_forward";
                    2
                ],
            ),
            Col::Str("boundary_contract", vec![CONTRACT; 2]),
            Col::Str(
                "signature_sha256",
                vec!["1111111111111111111111111111111111111111111111111111111111111111"; 2],
            ),
            Col::F64("latency_ms", vec![1.0, 5.0]),
            Col::F64("cv", vec![0.01; 2]),
            Col::I64("numerical_checks", vec![96; 2]),
            Col::I64("sample_count", vec![7; 2]),
            Col::Str("execution_verification", vec!["captured_graph_replay"; 2]),
            Col::Str(
                "runtime_identity_sha256",
                vec!["2222222222222222222222222222222222222222222222222222222222222222"; 2],
            ),
        ]
    }

    #[test]
    fn bounds_and_role_identity_survive_prepared_lookup() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        write_parquet(&path, &columns());
        let table = AscendVectorTable::new(Some(path));
        assert_eq!(table.query(&key(), 1).unwrap(), 1.0);
        assert_eq!(table.query(&key(), 5).unwrap(), 3.0);
        assert!(table.query(&key(), 10).unwrap_err().is_missing_perf_data());
        let mut other = key();
        other.role = "add_rms_bias".into();
        assert!(table.query(&other, 5).unwrap_err().is_missing_perf_data());
        other = key();
        other.execution_mode = "eager".into();
        assert!(table.query(&other, 5).unwrap_err().is_missing_perf_data());
        other.dimensions = vec![128];
        assert!(matches!(
            table.query(&other, 5),
            Err(AicError::InvalidEngineConfig(_))
        ));
    }

    #[test]
    fn sequence_length_transfer_is_eager_int32_and_has_no_resident_weights() {
        let mut transfer = AscendVectorKey {
            role: "seq_lens_h2d".into(),
            dimensions: vec![4],
            execution_mode: "eager".into(),
        };
        transfer.validate().unwrap();
        assert_eq!(transfer.kernel_source(), "torch.Tensor.to");
        assert_eq!(transfer.parameter_bytes(), 0.0);
        transfer.execution_mode = "graph".into();
        assert!(transfer.validate().is_err());
        transfer.execution_mode = "eager".into();
        transfer.dimensions = vec![8];
        assert!(transfer.validate().is_err());
    }

    #[test]
    fn separate_execution_curves_keep_separate_collection_identities() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        let first = "1111111111111111111111111111111111111111111111111111111111111111";
        let second = "2222222222222222222222222222222222222222222222222222222222222222";
        let mut fixture = columns();
        fixture[2] = Col::Str("execution_mode", vec!["graph", "eager"]);
        for column in &mut fixture {
            if let Col::Str("execution_verification", _) = column {
                *column = Col::Str(
                    "execution_verification",
                    vec!["captured_graph_replay", "eager_invocation"],
                );
            }
        }
        *fixture.last_mut().unwrap() = Col::Str("runtime_identity_sha256", vec![first, second]);
        write_parquet(&path, &fixture);
        let table = AscendVectorTable::new(Some(path.clone()));
        assert_eq!(table.query(&key(), 1).unwrap(), 1.0);
        let mut eager = key();
        eager.execution_mode = "eager".into();
        assert_eq!(table.query(&eager, 9).unwrap(), 5.0);
        let mut fixture = columns();
        *fixture.last_mut().unwrap() = Col::Str("runtime_identity_sha256", vec![first, second]);
        write_parquet(&path, &fixture);
        assert!(matches!(
            AscendVectorTable::new(Some(path)).ensure_key(&key()),
            Err(AicError::InvalidPerfData(_))
        ));
    }

    #[test]
    fn invalid_geometry_verification_and_duplicate_rows_fail_closed() {
        for (index, replacement) in [
            (1, Col::Str("dimensions", vec!["[6145]"; 2])),
            (3, Col::I64("tokens", vec![1; 2])),
            (4, Col::Str("kernel_source", vec!["other_norm"; 2])),
            (5, Col::Str("boundary_contract", vec!["other"; 2])),
            (6, Col::Str("signature_sha256", vec!["unverified"; 2])),
            (8, Col::F64("cv", vec![0.11; 2])),
            (9, Col::I64("numerical_checks", vec![0; 2])),
            (
                11,
                Col::Str("execution_verification", vec!["eager_invocation"; 2]),
            ),
        ] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join(BASENAME);
            let mut data = columns();
            data[index] = replacement;
            write_parquet(&path, &data);
            assert!(matches!(
                AscendVectorTable::new(Some(path)).ensure_key(&key()),
                Err(AicError::InvalidPerfData(_))
            ));
        }
        assert!(
            AscendVectorTable::new(None)
                .ensure_key(&key())
                .unwrap_err()
                .is_missing_perf_data()
        );
    }

    #[test]
    fn token_axis_padding_and_parameter_storage_are_explicit() {
        let mut operation = AscendVectorOp {
            name: "norm".into(),
            scale_factor: 2.0,
            key: key(),
            token_axis: AscendLinearTokens::Tokens,
            token_split: 2,
            token_multiple: 8,
            resident_copies: None,
        };
        assert_eq!(operation.measured_tokens(18, 3).unwrap(), 16);
        assert_eq!(operation.weight_bytes(), 49_152.0);
        operation.resident_copies = Some(1);
        assert_eq!(operation.weight_bytes(), 24_576.0);
        operation.token_axis = AscendLinearTokens::Requests;
        assert_eq!(operation.measured_tokens(8192, 3).unwrap(), 8);
        operation.token_split = 1;
        assert!(operation.measured_tokens(1, u32::MAX).is_err());
        let embedding = AscendVectorKey {
            role: "embedding".into(),
            dimensions: vec![19360, 6144],
            execution_mode: "eager".into(),
        };
        assert_eq!(embedding.parameter_bytes(), 237_895_680.0);
        let cache = AscendVectorKey {
            role: "cache_scatter".into(),
            dimensions: vec![53376, 128],
            execution_mode: "graph".into(),
        };
        assert_eq!(cache.parameter_bytes(), 0.0);
    }

    #[test]
    fn primary_paths_never_relabel_other_systems_or_runtimes() {
        let root = tempfile::tempdir().unwrap();
        let requested = root.path().join("data/ascend910c/sglang/0.5.22");
        for relative in [
            "data/ascend910c/vector/sglang/0.5.22",
            "data/ascend910c/sglang/0.5.22",
        ] {
            let path = root.path().join(relative).join(BASENAME);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            write_parquet(&path, &columns());
            let resolver = SourceResolver::fixed(PerfDbSources::from([(
                BASENAME.into(),
                vec![PerfSource(path, None)],
            )]));
            assert_eq!(
                AscendVectorTable::with_sources(&requested, &resolver)
                    .unwrap()
                    .query(&key(), 5)
                    .unwrap(),
                3.0
            );
        }
        for relative in [
            "data/other/vector/sglang/0.5.22",
            "data/ascend910c/vector/vllm/0.5.22",
            "data/ascend910c/vector/sglang/0.5.21",
        ] {
            let resolver = SourceResolver::fixed(PerfDbSources::from([(
                BASENAME.into(),
                vec![PerfSource(root.path().join(relative).join(BASENAME), None)],
            )]));
            assert!(matches!(
                AscendVectorTable::with_sources(&requested, &resolver),
                Err(AicError::InvalidPerfData(_))
            ));
        }
        let resolver = SourceResolver::fixed(PerfDbSources::from([(BASENAME.into(), Vec::new())]));
        assert!(
            AscendVectorTable::with_sources(&requested, &resolver)
                .unwrap()
                .ensure_key(&key())
                .unwrap_err()
                .is_missing_perf_data()
        );
    }
}
