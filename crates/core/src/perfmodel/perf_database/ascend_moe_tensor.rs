use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::SourceResolver;
use super::axis_curve::AxisCurve;
use super::parquet_loader::PerfReader;
use crate::common::error::AicError;

pub const BASENAME: &str = "ascend_moe_tensor_perf.parquet";
pub const CONTRACT: &str = "ascend_tp_w8a8/v1";
const KERNELS: [&str; 4] = [
    "sglang.srt.hardware_backend.npu.moe.init_routing.NPUMoEInitRouting_v2",
    "sglang.srt.hardware_backend.npu.quantization.moe_methods.NPUW8A8Int8MoEMethod",
    "sglang.srt.hardware_backend.npu.moe.activation.NPUSwigluQuant",
    "sglang.srt.hardware_backend.npu.moe.finalize_routing.NPUFinalizeRouting",
];

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendMoeTensorKey {
    pub intermediate_size: u32,
    pub distribution: String,
    pub execution_mode: String,
}

impl AscendMoeTensorKey {
    pub fn validate(&self) -> Result<(), AicError> {
        if !matches!(self.intermediate_size, 64 | 128 | 256 | 512)
            || self.distribution != "balanced"
            || !matches!(self.execution_mode.as_str(), "eager" | "graph")
        {
            return Err(AicError::InvalidEngineConfig(format!(
                "invalid Ascend tensor-MoE identity: {self:?}"
            )));
        }
        Ok(())
    }
}

struct Curve {
    values: AxisCurve,
    minimum: u32,
    maximum: u32,
}
type Grids = BTreeMap<AscendMoeTensorKey, Curve>;

pub struct AscendMoeTensorTable {
    path: Option<PathBuf>,
    grids: OnceLock<Result<Grids, (bool, String)>>,
}

impl AscendMoeTensorTable {
    pub fn estimate(
        &self,
        key: &AscendMoeTensorKey,
        tokens: u32,
        policy: super::ascend_shape::AscendShapePolicy,
        spec: &crate::common::system_spec::SystemSpec,
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
            &|count| {
                let active_experts = (count * 8.0).min(256.0);
                super::ascend_shape::roofline_ms(
                    spec,
                    6.0 * count * 8.0 * 6144.0 * f64::from(key.intermediate_size),
                    active_experts * 3.0 * 6144.0 * f64::from(key.intermediate_size)
                        + count * 8.0 * 6144.0 * 4.0,
                    true,
                )
            },
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
                "Ascend tensor-MoE rejects filtered primary sources".into(),
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
                            .join("moe")
                            .join(data_root.parent()?.file_name()?)
                            .join(data_root.file_name()?)
                            .join(BASENAME),
                    )
                });
            if path != &data_root.join(BASENAME) && family.as_ref() != Some(path) {
                return Err(AicError::InvalidPerfData(
                    "Ascend tensor-MoE requires the exact system/backend/version".into(),
                ));
            }
        }
        Ok(Self::new(path))
    }

    fn curve(&self, key: &AscendMoeTensorKey) -> Result<&Curve, AicError> {
        key.validate()?;
        let grids = self
            .grids
            .get_or_init(|| {
                let Some(path) = self.path.as_ref().filter(|path| path.is_file()) else {
                    return Err((false, "Ascend tensor-MoE primary data is missing".into()));
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
            AicError::PerfDatabase(format!("unmeasured Ascend tensor-MoE tuple: {key:?}"))
        })
    }

    pub fn ensure_key(&self, key: &AscendMoeTensorKey) -> Result<(), AicError> {
        self.curve(key).map(|_| ())
    }

    pub fn query(&self, key: &AscendMoeTensorKey, tokens: u32) -> Result<f64, AicError> {
        let curve = self.curve(key)?;
        if tokens < curve.minimum || tokens > curve.maximum {
            return Err(AicError::PerfDatabase(format!(
                "Ascend tensor-MoE M={tokens} outside measured [{}, {}]",
                curve.minimum, curve.maximum
            )));
        }
        curve.values.query(f64::from(tokens), &|_| 1.0)
    }
}

fn load(path: &Path) -> Result<Grids, AicError> {
    let reader = PerfReader::open(path)?;
    let intermediate = reader.col("intermediate_size")?;
    let distribution = reader.col("distribution")?;
    let mode = reader.col("execution_mode")?;
    let tokens = reader.col("tokens")?;
    let hidden = reader.col("hidden_size")?;
    let experts = reader.col("num_experts")?;
    let top_k = reader.col("top_k")?;
    let contract = reader.col("kernel_contract")?;
    let kernels = reader.col("kernel_sources")?;
    let format = reader.col("weight_format")?;
    let input_dtype = reader.col("input_dtype")?;
    let output_dtype = reader.col("output_dtype")?;
    let group = reader.col("group_list_type")?;
    let quant = reader.col("quant_mode")?;
    let drop_pad = reader.col("drop_pad_mode")?;
    let verification = reader.col("execution_verification")?;
    let latency = reader.col("latency_ms")?;
    let variation = reader.col("cv")?;
    let samples = reader.col("sample_count")?;
    let checks = reader.col("numerical_checks")?;
    let provenance = reader.col("runtime_identity_sha256")?;
    let mut raw: BTreeMap<AscendMoeTensorKey, BTreeMap<u32, f64>> = BTreeMap::new();
    let mut identity: Option<String> = None;
    for row in reader.rows()? {
        let row = row?;
        let key = AscendMoeTensorKey {
            intermediate_size: row.u32(intermediate)?,
            distribution: row.str_owned(distribution)?,
            execution_mode: row.str_owned(mode)?,
        };
        key.validate()?;
        let count = row.u32(tokens)?;
        let measured = row.f64(latency)?;
        let cv = row.f64(variation)?;
        let runtime = row.str(provenance)?;
        let kernel_sources: Vec<String> =
            serde_json::from_str(row.str(kernels)?).map_err(|error| {
                AicError::InvalidPerfData(format!("invalid tensor-MoE kernel sources: {error}"))
            })?;
        let expected = if key.execution_mode == "graph" {
            "captured_graph_replay"
        } else {
            "eager_invocation"
        };
        if count == 0
            || row.u32(hidden)? != 6144
            || row.u32(experts)? != 256
            || row.u32(top_k)? != 8
            || row.str(contract)? != CONTRACT
            || kernel_sources != KERNELS
            || row.u32(format)? != 29
            || row.str(input_dtype)? != "bfloat16"
            || row.str(output_dtype)? != "bfloat16"
            || row.u32(group)? != 1
            || row.u32(quant)? != 1
            || row.u32(drop_pad)? != 2
            || row.str(verification)? != expected
            || !measured.is_finite()
            || measured <= 0.0
            || !cv.is_finite()
            || !(0.0..=0.1).contains(&cv)
            || row.u32(samples)? < 3
            || row.u32(checks)? == 0
            || runtime.len() != 64
            || !runtime.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(AicError::InvalidPerfData(format!(
                "unqualified Ascend tensor-MoE row: {key:?}"
            )));
        }
        if identity
            .as_ref()
            .is_some_and(|expected| expected != runtime)
        {
            return Err(AicError::InvalidPerfData(
                "Ascend tensor-MoE mixes collection identities".into(),
            ));
        }
        identity.get_or_insert_with(|| runtime.to_owned());
        if raw
            .entry(key)
            .or_default()
            .insert(count, measured)
            .is_some()
        {
            return Err(AicError::InvalidPerfData(
                "duplicate tensor-MoE physical key".into(),
            ));
        }
    }
    if raw.is_empty() {
        return Err(AicError::InvalidPerfData(
            "empty Ascend tensor-MoE table".into(),
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
    use crate::operators::AscendMoeTensorOp;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};

    fn key() -> AscendMoeTensorKey {
        AscendMoeTensorKey {
            intermediate_size: 128,
            distribution: "balanced".into(),
            execution_mode: "graph".into(),
        }
    }

    fn columns() -> Vec<Col> {
        vec![
            Col::I64("intermediate_size", vec![128; 2]),
            Col::Str("distribution", vec!["balanced"; 2]),
            Col::Str("execution_mode", vec!["graph"; 2]),
            Col::I64("tokens", vec![8, 16]),
            Col::I64("hidden_size", vec![6144; 2]),
            Col::I64("num_experts", vec![256; 2]),
            Col::I64("top_k", vec![8; 2]),
            Col::Str("kernel_contract", vec![CONTRACT; 2]),
            Col::Str(
                "kernel_sources",
                vec![
                    "[\"sglang.srt.hardware_backend.npu.moe.init_routing.NPUMoEInitRouting_v2\",\"sglang.srt.hardware_backend.npu.quantization.moe_methods.NPUW8A8Int8MoEMethod\",\"sglang.srt.hardware_backend.npu.moe.activation.NPUSwigluQuant\",\"sglang.srt.hardware_backend.npu.moe.finalize_routing.NPUFinalizeRouting\"]";
                    2
                ],
            ),
            Col::I64("weight_format", vec![29; 2]),
            Col::Str("input_dtype", vec!["bfloat16"; 2]),
            Col::Str("output_dtype", vec!["bfloat16"; 2]),
            Col::I64("group_list_type", vec![1; 2]),
            Col::I64("quant_mode", vec![1; 2]),
            Col::I64("drop_pad_mode", vec![2; 2]),
            Col::Str("execution_verification", vec!["captured_graph_replay"; 2]),
            Col::F64("latency_ms", vec![1.0, 3.0]),
            Col::F64("cv", vec![0.01; 2]),
            Col::I64("sample_count", vec![7; 2]),
            Col::I64("numerical_checks", vec![1024; 2]),
            Col::Str(
                "runtime_identity_sha256",
                vec!["1111111111111111111111111111111111111111111111111111111111111111"; 2],
            ),
        ]
    }

    #[test]
    fn midpoint_of_one_and_three_ms_is_two_and_does_not_extrapolate() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        write_parquet(&path, &columns());
        let table = AscendMoeTensorTable::new(Some(path));
        assert_eq!(table.query(&key(), 8).unwrap(), 1.0);
        assert_eq!(table.query(&key(), 12).unwrap(), 2.0);
        assert_eq!(table.query(&key(), 16).unwrap(), 3.0);
        for count in [1, 17] {
            assert!(
                table
                    .query(&key(), count)
                    .unwrap_err()
                    .is_missing_perf_data()
            );
        }
        let mut other = key();
        other.intermediate_size = 256;
        assert!(table.query(&other, 12).unwrap_err().is_missing_perf_data());
    }

    #[test]
    fn rejects_wrong_dispatcher_and_mixed_measurement_identity() {
        let root = tempfile::tempdir().unwrap();
        for replacement in [
            Col::I64("quant_mode", vec![-1; 2]),
            Col::I64("drop_pad_mode", vec![0; 2]),
            Col::Str("kernel_sources", vec!["[]"; 2]),
            Col::F64("cv", vec![0.11; 2]),
            Col::Str(
                "runtime_identity_sha256",
                vec![
                    "1111111111111111111111111111111111111111111111111111111111111111",
                    "2222222222222222222222222222222222222222222222222222222222222222",
                ],
            ),
        ] {
            let mut fixture = columns();
            let name = match &replacement {
                Col::I64(name, _) | Col::F64(name, _) | Col::Str(name, _) => *name,
                _ => unreachable!(),
            };
            fixture.retain(|column| match column {
                Col::I64(existing, _) | Col::F64(existing, _) | Col::Str(existing, _) => {
                    *existing != name
                }
                _ => true,
            });
            fixture.push(replacement);
            let path = root.path().join(BASENAME);
            write_parquet(&path, &fixture);
            assert!(matches!(
                AscendMoeTensorTable::new(Some(path)).ensure_key(&key()),
                Err(AicError::InvalidPerfData(_))
            ));
        }
    }

    #[test]
    fn tensor_moe_pads_input_tokens_without_dividing_or_topk_expansion() {
        let operation = AscendMoeTensorOp {
            name: "tensor".into(),
            scale_factor: 75.0,
            key: key(),
            input_token_multiple: 16,
            routing_profile: None,
        };
        assert_eq!(operation.measured_tokens(0).unwrap(), 0);
        assert_eq!(operation.measured_tokens(1).unwrap(), 16);
        assert_eq!(operation.measured_tokens(17).unwrap(), 32);
        assert!(operation.measured_tokens(u32::MAX).is_err());
    }
}
