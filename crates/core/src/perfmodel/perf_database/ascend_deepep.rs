use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::SourceResolver;
use super::axis_curve::AxisCurve;
use super::parquet_loader::PerfReader;
use crate::common::error::AicError;

pub const BASENAME: &str = "ascend_deepep_perf.parquet";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendDeepEpKey {
    pub phase: String,
    pub operation: String,
    pub num_ranks: u32,
    pub ranks_per_node: u32,
    pub distribution: String,
    pub execution_mode: String,
}

impl AscendDeepEpKey {
    pub fn validate(&self) -> Result<(), AicError> {
        if !matches!(self.phase.as_str(), "normal" | "low_latency")
            || !matches!(self.operation.as_str(), "dispatch" | "combine")
            || !matches!(self.num_ranks, 16 | 32)
            || self.ranks_per_node != 16
            || !matches!(
                self.distribution.as_str(),
                "balanced" | "single_expert_group"
            )
            || !matches!(self.execution_mode.as_str(), "eager" | "graph")
            || (self.phase == "normal" && self.execution_mode != "eager")
        {
            return Err(AicError::InvalidEngineConfig(format!(
                "invalid Ascend DeepEP identity: {self:?}"
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
type Grids = BTreeMap<AscendDeepEpKey, Curve>;

pub struct AscendDeepEpTable {
    path: Option<PathBuf>,
    grids: OnceLock<Result<Grids, (bool, String)>>,
}

impl AscendDeepEpTable {
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
                "Ascend DeepEP rejects filtered primary sources".into(),
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
                    "Ascend DeepEP requires the exact requested system/backend/version".into(),
                ));
            }
        }
        Ok(Self::new(path))
    }

    fn curve(&self, key: &AscendDeepEpKey) -> Result<&Curve, AicError> {
        key.validate()?;
        let grids = self
            .grids
            .get_or_init(|| {
                let Some(path) = self.path.as_ref().filter(|path| path.is_file()) else {
                    return Err((false, "Ascend DeepEP primary data is missing".into()));
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
            AicError::PerfDatabase(format!("unmeasured Ascend DeepEP tuple: {key:?}"))
        })
    }

    pub fn ensure_key(&self, key: &AscendDeepEpKey) -> Result<(), AicError> {
        self.curve(key).map(|_| ())
    }

    pub fn query(&self, key: &AscendDeepEpKey, received: u32) -> Result<f64, AicError> {
        let curve = self.curve(key)?;
        if received < curve.minimum || received > curve.maximum {
            return Err(AicError::PerfDatabase(format!(
                "Ascend DeepEP sent tokens={received} outside measured [{}, {}]",
                curve.minimum, curve.maximum
            )));
        }
        curve.values.query(f64::from(received), &|_| 1.0)
    }
}

fn load(path: &Path) -> Result<Grids, AicError> {
    let reader = PerfReader::open(path)?;
    let phase = reader.col("phase")?;
    let operation = reader.col("operation")?;
    let ranks = reader.col("num_ranks")?;
    let local = reader.col("ranks_per_node")?;
    let layout = reader.col("rank_layout")?;
    let distribution = reader.col("distribution")?;
    let mode = reader.col("execution_mode")?;
    let tokens = reader.col("tokens_per_rank")?;
    let hidden = reader.col("hidden_size")?;
    let experts = reader.col("num_experts")?;
    let topk = reader.col("topk")?;
    let capacity = reader.col("max_dispatch_tokens_per_rank")?;
    let dispatch_dtype = reader.col("dispatch_dtype")?;
    let combine_dtype = reader.col("combine_dtype")?;
    let kernel = reader.col("kernel_source")?;
    let latency = reader.col("latency_ms")?;
    let cv = reader.col("cv")?;
    let checks = reader.col("numerical_checks")?;
    let samples = reader.col("sample_count")?;
    let verification = reader.col("execution_verification")?;
    let provenance = reader.col("runtime_identity_sha256")?;
    let mut raw: BTreeMap<AscendDeepEpKey, BTreeMap<u32, f64>> = BTreeMap::new();
    let mut identities: BTreeMap<AscendDeepEpKey, String> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = AscendDeepEpKey {
            phase: row.str_owned(phase)?,
            operation: row.str_owned(operation)?,
            num_ranks: row.u32(ranks)?,
            ranks_per_node: row.u32(local)?,
            distribution: row.str_owned(distribution)?,
            execution_mode: row.str_owned(mode)?,
        };
        key.validate()?;
        let count = row.u32(tokens)?;
        let measured = row.f64(latency)?;
        let variation = row.f64(cv)?;
        let runtime = row.str(provenance)?;
        let expected = if key.execution_mode == "graph" {
            "captured_graph_replay"
        } else {
            "eager_invocation"
        };
        let implementation = if key.phase == "normal" {
            "sglang.srt.layers.moe.token_dispatcher.deepep._DeepEPDispatcherImplNormal"
        } else {
            "sglang.srt.layers.moe.token_dispatcher.deepep._DeepEPDispatcherImplLowLatency"
        };
        if count == 0
            || (key.phase == "low_latency" && count > 32)
            || row.str(layout)? != "contiguous"
            || row.u32(hidden)? != 6144
            || row.u32(experts)? != 256
            || row.u32(topk)? != 8
            || row.u32(capacity)? != 32
            || row.str(dispatch_dtype)? != "int8"
            || row.str(combine_dtype)? != "bfloat16"
            || row.str(kernel)? != implementation
            || !measured.is_finite()
            || measured <= 0.0
            || !variation.is_finite()
            || !(0.0..=0.1).contains(&variation)
            || row.u64(checks)? == 0
            || row.u32(samples)? < 3
            || row.str(verification)? != expected
            || runtime.len() != 64
            || !runtime.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(AicError::InvalidPerfData(format!(
                "unqualified Ascend DeepEP row: {key:?}"
            )));
        }
        let expected = identities
            .entry(key.clone())
            .or_insert_with(|| runtime.to_owned());
        if expected != runtime {
            return Err(AicError::InvalidPerfData(
                "Ascend DeepEP curve mixes runtime identities".into(),
            ));
        }
        if raw
            .entry(key)
            .or_default()
            .insert(count, measured)
            .is_some()
        {
            return Err(AicError::InvalidPerfData(
                "duplicate Ascend DeepEP physical key".into(),
            ));
        }
    }
    if raw.is_empty() {
        return Err(AicError::InvalidPerfData(
            "empty Ascend DeepEP table".into(),
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
                    values: AxisCurve::from_map("tokens_per_rank", points),
                },
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::AscendDeepEpOp;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};

    fn key() -> AscendDeepEpKey {
        AscendDeepEpKey {
            phase: "low_latency".into(),
            operation: "dispatch".into(),
            num_ranks: 16,
            ranks_per_node: 16,
            distribution: "balanced".into(),
            execution_mode: "graph".into(),
        }
    }

    fn columns() -> Vec<Col> {
        vec![
            Col::Str("phase", vec!["low_latency"; 2]),
            Col::Str("operation", vec!["dispatch"; 2]),
            Col::I64("num_ranks", vec![16; 2]),
            Col::I64("ranks_per_node", vec![16; 2]),
            Col::Str("rank_layout", vec!["contiguous"; 2]),
            Col::Str("distribution", vec!["balanced"; 2]),
            Col::Str("execution_mode", vec!["graph"; 2]),
            Col::I64("tokens_per_rank", vec![1, 4]),
            Col::I64("hidden_size", vec![6144; 2]),
            Col::I64("num_experts", vec![256; 2]),
            Col::I64("topk", vec![8; 2]),
            Col::I64("max_dispatch_tokens_per_rank", vec![32; 2]),
            Col::Str("dispatch_dtype", vec!["int8"; 2]),
            Col::Str("combine_dtype", vec!["bfloat16"; 2]),
            Col::Str(
                "kernel_source",
                vec![
                    "sglang.srt.layers.moe.token_dispatcher.deepep._DeepEPDispatcherImplLowLatency";
                    2
                ],
            ),
            Col::F64("latency_ms", vec![1.0, 4.0]),
            Col::F64("cv", vec![0.01; 2]),
            Col::I64("numerical_checks", vec![10_000_000_000; 2]),
            Col::I64("sample_count", vec![7; 2]),
            Col::Str("execution_verification", vec!["captured_graph_replay"; 2]),
            Col::Str(
                "runtime_identity_sha256",
                vec!["1111111111111111111111111111111111111111111111111111111111111111"; 2],
            ),
        ]
    }

    #[test]
    fn transport_lookup_keeps_phase_topology_distribution_and_bounds() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        write_parquet(&path, &columns());
        let table = AscendDeepEpTable::new(Some(path));
        assert_eq!(table.query(&key(), 1).unwrap(), 1.0);
        assert_eq!(table.query(&key(), 2).unwrap(), 2.0);
        assert!(table.query(&key(), 5).unwrap_err().is_missing_perf_data());
        let mut other = key();
        other.num_ranks = 32;
        assert!(table.query(&other, 2).unwrap_err().is_missing_perf_data());
        other = key();
        other.operation = "combine".into();
        assert!(table.query(&other, 2).unwrap_err().is_missing_perf_data());
        other = key();
        other.distribution = "single_expert_group".into();
        assert!(table.query(&other, 2).unwrap_err().is_missing_perf_data());
        other = key();
        other.phase = "normal".into();
        assert!(matches!(
            table.query(&other, 2),
            Err(AicError::InvalidEngineConfig(_))
        ));
    }

    #[test]
    fn transport_rows_require_correct_precision_and_actual_execution() {
        for (index, replacement) in [
            (7, Col::I64("tokens_per_rank", vec![1, 1])),
            (12, Col::Str("dispatch_dtype", vec!["float8"; 2])),
            (16, Col::F64("cv", vec![0.2; 2])),
            (
                19,
                Col::Str("execution_verification", vec!["not_executed"; 2]),
            ),
        ] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join(BASENAME);
            let mut values = columns();
            values[index] = replacement;
            write_parquet(&path, &values);
            assert!(matches!(
                AscendDeepEpTable::new(Some(path)).ensure_key(&key()),
                Err(AicError::InvalidPerfData(_))
            ));
        }
        assert!(
            AscendDeepEpTable::new(None)
                .ensure_key(&key())
                .unwrap_err()
                .is_missing_perf_data()
        );
    }

    #[test]
    fn runtime_identity_is_uniform_within_each_complete_curve() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        let mut values = columns();
        values[20] = Col::Str(
            "runtime_identity_sha256",
            vec![
                "1111111111111111111111111111111111111111111111111111111111111111",
                "2222222222222222222222222222222222222222222222222222222222222222",
            ],
        );
        write_parquet(&path, &values);
        assert!(matches!(
            AscendDeepEpTable::new(Some(path.clone())).ensure_key(&key()),
            Err(AicError::InvalidPerfData(_))
        ));

        values[1] = Col::Str("operation", vec!["dispatch", "combine"]);
        write_parquet(&path, &values);
        let table = AscendDeepEpTable::new(Some(path));
        assert_eq!(table.query(&key(), 1).unwrap(), 1.0);
        let mut combine = key();
        combine.operation = "combine".into();
        assert_eq!(table.query(&combine, 4).unwrap(), 4.0);
    }

    #[test]
    fn attention_tokens_are_padded_before_local_sharding() {
        let mut operation = AscendDeepEpOp {
            name: "dispatch".into(),
            scale_factor: 75.0,
            key: key(),
            token_split: 8,
            input_token_multiple: 8,
        };
        assert_eq!(operation.sent_tokens(1).unwrap(), 1);
        assert_eq!(operation.sent_tokens(9).unwrap(), 2);
        assert_eq!(operation.sent_tokens(32).unwrap(), 4);
        assert!(operation.sent_tokens(u32::MAX).is_err());
        operation.token_split = 0;
        assert!(operation.sent_tokens(1).is_err());
    }
}
