use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::axis_curve::LeafAxisCurve;
use super::parquet_loader::PerfReader;
use super::perf_interp::LeafValue;
use crate::common::error::AicError;

pub const BASENAME: &str = "hccl_perf.parquet";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HcclKey {
    pub dtype: String,
    pub operation: String,
    pub num_ranks: u32,
    pub ranks_per_node: u32,
    pub rank_layout: String,
    pub execution_mode: String,
}

impl HcclKey {
    pub fn validate(&self) -> Result<(), AicError> {
        if !matches!(
            self.dtype.as_str(),
            "bfloat16" | "float16" | "float32" | "int32"
        ) || !matches!(
            self.operation.as_str(),
            "all_reduce" | "all_gather" | "reduce_scatter" | "all_to_all"
        ) || !matches!(self.execution_mode.as_str(), "eager" | "graph")
            || self.rank_layout != "contiguous"
            || self.num_ranks < 2
            || self.ranks_per_node == 0
            || self.num_ranks % self.ranks_per_node != 0
        {
            return Err(AicError::InvalidEngineConfig(format!(
                "invalid HCCL identity: {self:?}"
            )));
        }
        Ok(())
    }
}

struct HcclCurve {
    values: LeafAxisCurve<u64>,
    minimum: u64,
    maximum: u64,
}

pub struct HcclTable {
    path: Option<PathBuf>,
    grids: OnceLock<Result<BTreeMap<HcclKey, HcclCurve>, (bool, String)>>,
}

impl HcclTable {
    pub fn estimate(
        &self,
        key: &HcclKey,
        bytes: u64,
        policy: super::ascend_shape::AscendShapePolicy,
        spec: &crate::common::system_spec::SystemSpec,
    ) -> Result<super::ascend_shape::ShapeEstimate, AicError> {
        match self.query(key, bytes) {
            Ok(value) => return Ok(super::ascend_shape::ShapeEstimate::measured(value.latency)),
            Err(error)
                if policy == super::ascend_shape::AscendShapePolicy::Anchored
                    && error.is_missing_perf_data() => {}
            Err(error) => return Err(error),
        }
        if bytes == 0 || bytes > (1_u64 << 53) {
            return Err(AicError::InvalidEngineConfig(
                "invalid HCCL extrapolation payload".into(),
            ));
        }
        let grids = self.loaded()?;
        let curve = grids
            .get(key)
            .ok_or_else(|| AicError::PerfDatabase(format!("HCCL tuple is unmeasured: {key:?}")))?;
        let bandwidth = if key.num_ranks > key.ranks_per_node {
            spec.node.inter_node_bw
        } else {
            spec.node.intra_node_bw
        };
        if !bandwidth.is_finite()
            || bandwidth <= 0.0
            || !spec.node.p2p_latency.is_finite()
            || spec.node.p2p_latency < 0.0
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid HCCL topology calibration".into(),
            ));
        }
        let work = |payload| spec.node.p2p_latency + payload as f64 / bandwidth;
        let anchor = bytes.clamp(curve.minimum, curve.maximum);
        let measured = self.query(key, anchor)?.latency;
        super::ascend_shape::ShapeEstimate::measured(measured).scale_work(
            work(bytes),
            work(anchor),
            self.query(key, curve.minimum)?.latency.min(measured),
        )
    }
    pub fn new(root: Option<PathBuf>) -> Self {
        Self {
            path: root.map(|root| root.join(BASENAME)),
            grids: OnceLock::new(),
        }
    }

    pub fn query(&self, key: &HcclKey, input_bytes_per_rank: u64) -> Result<LeafValue, AicError> {
        key.validate()?;
        let grids = self.loaded()?;
        let curve = grids
            .get(key)
            .ok_or_else(|| AicError::PerfDatabase(format!("HCCL tuple is unmeasured: {key:?}")))?;
        if input_bytes_per_rank < curve.minimum || input_bytes_per_rank > curve.maximum {
            return Err(AicError::PerfDatabase(format!(
                "HCCL payload {input_bytes_per_rank} is outside measured [{}, {}] for {key:?}",
                curve.minimum, curve.maximum,
            )));
        }
        curve.values.query(input_bytes_per_rank as f64, &|_| 1.0)
    }

    pub fn ensure_key(&self, key: &HcclKey) -> Result<(), AicError> {
        key.validate()?;
        if !self.loaded()?.contains_key(key) {
            return Err(AicError::PerfDatabase(format!(
                "HCCL tuple is unmeasured: {key:?}"
            )));
        }
        Ok(())
    }

    fn loaded(&self) -> Result<&BTreeMap<HcclKey, HcclCurve>, AicError> {
        self.grids
            .get_or_init(|| {
                let Some(path) = &self.path else {
                    return Err((
                        false,
                        "HCCL is not configured: misc.hccl_version is missing".into(),
                    ));
                };
                if !path.is_file() {
                    return Err((false, format!("HCCL data is missing: {}", path.display())));
                }
                load(path).map_err(|error| (true, error.to_string()))
            })
            .as_ref()
            .map_err(|(invalid, message)| {
                if *invalid {
                    AicError::InvalidPerfData(message.clone())
                } else {
                    AicError::PerfDatabase(message.clone())
                }
            })
    }
}

fn load(path: &std::path::Path) -> Result<BTreeMap<HcclKey, HcclCurve>, AicError> {
    let reader = PerfReader::open(path)?;
    let dtype = reader.col("dtype")?;
    let operation = reader.col("operation")?;
    let num_ranks = reader.col("num_ranks")?;
    let ranks_per_node = reader.col("ranks_per_node")?;
    let rank_layout = reader.col("rank_layout")?;
    let execution_mode = reader.col("execution_mode")?;
    let execution_verification = reader.col("execution_verification")?;
    let input_bytes = reader.col("input_bytes_per_rank")?;
    let latency = reader.col("latency_ms")?;
    let cv = reader.col("cv")?;
    let samples = reader.col("sample_count")?;
    let checks = reader.col("correctness_checks")?;
    let provenance = reader.col("runtime_identity_sha256")?;
    let kernel = reader.col("kernel_source")?;
    let mut runtime_identity = None;
    let mut rows: BTreeMap<HcclKey, BTreeMap<u64, LeafValue>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = HcclKey {
            dtype: row.str_owned(dtype)?,
            operation: row.str_owned(operation)?,
            num_ranks: row.u32(num_ranks)?,
            ranks_per_node: row.u32(ranks_per_node)?,
            rank_layout: row.str_owned(rank_layout)?,
            execution_mode: row.str_owned(execution_mode)?,
        };
        key.validate()?;
        let expected_verification = if key.execution_mode == "graph" {
            "captured_graph_replay"
        } else {
            "eager_invocation"
        };
        if row.str_owned(execution_verification)? != expected_verification {
            return Err(AicError::InvalidPerfData(
                "HCCL row does not verify its actual execution path".into(),
            ));
        }
        let bytes = row.u64(input_bytes)?;
        let value = row.f64(latency)?;
        let variability = row.f64(cv)?;
        let identity = row.str_owned(provenance)?;
        if bytes == 0
            || bytes > (1_u64 << 53)
            || !value.is_finite()
            || value <= 0.0
            || !variability.is_finite()
            || !(0.0..=0.1).contains(&variability)
            || row.u32(samples)? < 3
            || u64::from(row.u32(checks)?) < 2 * u64::from(key.num_ranks)
            || identity.len() != 64
            || !identity.bytes().all(|byte| byte.is_ascii_hexdigit())
            || row.str_owned(kernel)? != "torch.distributed.hccl"
        {
            return Err(AicError::InvalidPerfData(format!(
                "unqualified HCCL row for {key:?}"
            )));
        }
        if let Some(expected) = &runtime_identity {
            if expected != &identity {
                return Err(AicError::InvalidPerfData(
                    "HCCL file mixes runtime/hardware identities".into(),
                ));
            }
        } else {
            runtime_identity = Some(identity);
        }
        if rows
            .entry(key)
            .or_default()
            .insert(bytes, LeafValue::latency_only(value))
            .is_some()
        {
            return Err(AicError::InvalidPerfData(
                "duplicate HCCL physical key".into(),
            ));
        }
    }
    if rows.is_empty() {
        return Err(AicError::InvalidPerfData("empty HCCL table".into()));
    }
    Ok(rows
        .into_iter()
        .map(|(key, points)| {
            let minimum = *points.first_key_value().unwrap().0;
            let maximum = *points.last_key_value().unwrap().0;
            (
                key,
                HcclCurve {
                    values: LeafAxisCurve::from_map("input_bytes_per_rank", points),
                    minimum,
                    maximum,
                },
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};

    fn columns() -> Vec<Col> {
        vec![
            Col::Str("dtype", vec!["bfloat16"; 2]),
            Col::Str("operation", vec!["all_gather"; 2]),
            Col::I64("num_ranks", vec![8; 2]),
            Col::I64("ranks_per_node", vec![8; 2]),
            Col::Str("rank_layout", vec!["contiguous"; 2]),
            Col::Str("execution_mode", vec!["graph"; 2]),
            Col::I64("input_bytes_per_rank", vec![1024, 3072]),
            Col::F64("latency_ms", vec![0.010, 0.030]),
            Col::F64("cv", vec![0.01; 2]),
            Col::I64("sample_count", vec![9; 2]),
            Col::I64("correctness_checks", vec![16; 2]),
            Col::Str(
                "runtime_identity_sha256",
                vec!["1111111111111111111111111111111111111111111111111111111111111111"; 2],
            ),
            Col::Str("kernel_source", vec!["torch.distributed.hccl"; 2]),
            Col::Str("execution_verification", vec!["captured_graph_replay"; 2]),
        ]
    }

    #[test]
    fn system_selector_reaches_the_measured_native_operator() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("data/npu/comm/hccl/qualified-runtime");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::create_dir_all(root.path().join("data/npu/sglang/0.5.22")).unwrap();
        std::fs::write(root.path().join("npu.yaml"),
            "data_dir: data/npu\ngpu:\n  mem_bw: 1000000\nnode:\n  num_gpus_per_node: 16\n  intra_node_bw: 1000000\n  inter_node_bw: 100000\nmisc:\n  hccl_version: qualified-runtime\n").unwrap();
        write_parquet(&directory.join(BASENAME), &columns());
        let database =
            crate::perf_database::PerfDatabase::load(root.path(), "npu", "sglang", "0.5.22")
                .unwrap();
        let op = crate::operators::HcclOp {
            name: "gather".into(),
            scale_factor: 2.0,
            key: key(),
            elements_per_token: 512,
            token_split: 1,
            token_axis: crate::operators::ascend_linear::AscendLinearTokens::Requests,
            dp_padding_mode: None,
        };
        assert!((op.query(&database, 2).unwrap().latency_ms - 0.040).abs() < 1e-12);
        assert!(op.query(&database, 4).unwrap_err().is_missing_perf_data());
        let context = crate::operators::RuntimeContext {
            batch_size: 2,
            num_tokens: 8192,
            ..Default::default()
        };
        let measured = crate::operators::Op::Hccl(op)
            .query(&database, &context)
            .unwrap();
        assert!((measured.latency_ms - 0.040).abs() < 1e-12);
    }

    #[test]
    fn rejects_duplicate_keys_unqualified_points_and_mixed_runtime() {
        let replacements = [
            (6, Col::I64("input_bytes_per_rank", vec![1024; 2])),
            (8, Col::F64("cv", vec![0.01, 0.3])),
            (10, Col::I64("correctness_checks", vec![16, 1])),
            (
                13,
                Col::Str("execution_verification", vec!["eager_invocation"; 2]),
            ),
            (
                11,
                Col::Str(
                    "runtime_identity_sha256",
                    vec![
                        "1111111111111111111111111111111111111111111111111111111111111111",
                        "2222222222222222222222222222222222222222222222222222222222222222",
                    ],
                ),
            ),
        ];
        for (index, replacement) in replacements {
            let root = tempfile::tempdir().unwrap();
            let mut values = columns();
            values[index] = replacement;
            write_parquet(&root.path().join(BASENAME), &values);
            let table = HcclTable::new(Some(root.path().to_owned()));
            assert!(matches!(
                table.query(&key(), 1024),
                Err(AicError::InvalidPerfData(_))
            ));
        }
    }

    #[test]
    fn graph_rows_without_replay_verification_are_not_publishable() {
        let root = tempfile::tempdir().unwrap();
        let mut values = columns();
        values.pop();
        write_parquet(&root.path().join(BASENAME), &values);
        let table = HcclTable::new(Some(root.path().to_owned()));
        assert!(matches!(
            table.query(&key(), 1024),
            Err(AicError::InvalidPerfData(_))
        ));
    }

    fn key() -> HcclKey {
        HcclKey {
            dtype: "bfloat16".into(),
            operation: "all_gather".into(),
            num_ranks: 8,
            ranks_per_node: 8,
            rank_layout: "contiguous".into(),
            execution_mode: "graph".into(),
        }
    }

    #[test]
    fn interpolation_preserves_exact_identity_and_refuses_extrapolation() {
        let table = HcclTable::new(None);
        let points = BTreeMap::from([
            (1024, LeafValue::latency_only(0.010)),
            (3072, LeafValue::latency_only(0.030)),
        ]);
        assert!(
            table
                .grids
                .set(Ok(BTreeMap::from([(
                    key(),
                    HcclCurve {
                        values: LeafAxisCurve::from_map("input_bytes_per_rank", points),
                        minimum: 1024,
                        maximum: 3072,
                    }
                )])))
                .is_ok()
        );
        assert_eq!(table.query(&key(), 1024).unwrap().latency, 0.010);
        assert!((table.query(&key(), 2048).unwrap().latency - 0.020).abs() < 1e-12);
        assert!(table.query(&key(), 512).is_err());
        assert!(table.query(&key(), 4096).is_err());
        let mut other = key();
        other.ranks_per_node = 4;
        assert!(table.query(&other, 2048).is_err());
        other = key();
        other.execution_mode = "eager".into();
        assert!(table.query(&other, 2048).is_err());
        other = key();
        other.dtype = "float16".into();
        assert!(table.query(&other, 2048).is_err());
    }

    #[test]
    fn missing_configuration_is_distinct_from_corrupt_data() {
        let table = HcclTable::new(None);
        assert!(matches!(
            table.query(&key(), 1024),
            Err(AicError::PerfDatabase(_))
        ));
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join(BASENAME), b"invalid parquet").unwrap();
        let table = HcclTable::new(Some(root.path().to_owned()));
        for _ in 0..2 {
            assert!(matches!(
                table.query(&key(), 1024),
                Err(AicError::InvalidPerfData(_))
            ));
        }
    }
}
