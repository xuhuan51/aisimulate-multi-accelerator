use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use super::SourceResolver;
use super::axis_curve::AxisCurve;
use super::parquet_loader::PerfReader;
use crate::common::error::AicError;

pub const BASENAME: &str = "ascend_moe_chain_perf.parquet";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendMoeChainKey {
    pub phase: String,
    pub local_experts: u32,
    pub distribution: String,
    pub execution_mode: String,
    #[serde(default)]
    pub active_experts: Option<u32>,
}

impl AscendMoeChainKey {
    pub fn validate(&self) -> Result<(), AicError> {
        if !matches!(self.phase.as_str(), "prefill" | "decode")
            || !matches!(self.local_experts, 8 | 16 | 32)
            || !matches!(self.distribution.as_str(), "balanced" | "single_expert")
            || !matches!(self.execution_mode.as_str(), "eager" | "graph")
            || (self.local_experts == 32
                && (self.phase != "decode" || self.execution_mode != "graph"))
            || self.active_experts.is_some_and(|active| {
                active == 0 || active > self.local_experts || self.distribution != "balanced"
            })
        {
            return Err(AicError::InvalidEngineConfig(format!(
                "invalid Ascend MoE chain identity: {self:?}"
            )));
        }
        Ok(())
    }

    pub fn physical_rows(&self, received: u32) -> u32 {
        if self.phase == "prefill" {
            received
        } else if self.local_experts == 32 {
            // Qwen512/EP16, cap16, top10. The buffer retains padded rows.
            16 * 16 * 10
        } else {
            (256 / self.local_experts) * 32 * 8
        }
    }

    fn dimensions(&self) -> (u32, u32) {
        if self.local_experts == 32 {
            (4096, 1024)
        } else {
            (6144, 2048)
        }
    }

    pub fn parameter_bytes(&self) -> f64 {
        let (hidden, intermediate) = self.dimensions();
        let scale_bytes = if self.local_experts == 32 { 4 } else { 6 };
        f64::from(self.local_experts)
            * (f64::from(hidden) * f64::from(intermediate) * 3.0
                + f64::from((2 * intermediate + hidden) * scale_bytes))
    }
}

struct Curve {
    values: AxisCurve,
    minimum: u32,
    maximum: u32,
}
type Grids = BTreeMap<AscendMoeChainKey, Curve>;
type WorkloadCurves = BTreeMap<Vec<u32>, BTreeMap<u32, f64>>;
type PreparedWorkload = (
    Vec<crate::operators::ascend_moe_chain::DecodeExpertSample>,
    WorkloadCurves,
);

pub struct AscendMoeChainTable {
    path: Option<PathBuf>,
    grids: OnceLock<Result<Grids, (bool, String)>>,
    workloads: Mutex<BTreeMap<String, Arc<PreparedWorkload>>>,
}

impl AscendMoeChainTable {
    pub fn estimate(
        &self,
        key: &AscendMoeChainKey,
        received: u32,
        policy: super::ascend_shape::AscendShapePolicy,
        spec: &crate::common::system_spec::SystemSpec,
    ) -> Result<super::ascend_shape::ShapeEstimate, AicError> {
        if let Some(active) = key.active_experts {
            key.validate()?;
            if active > received {
                return Err(AicError::InvalidEngineConfig(
                    "active experts cannot exceed received tokens".into(),
                ));
            }
            let mut endpoint = key.clone();
            endpoint.active_experts = None;
            let maximum_active = received.min(key.local_experts);
            if active == maximum_active {
                return self.estimate(&endpoint, received, policy, spec);
            }
            endpoint.distribution = "single_expert".into();
            let single = self.estimate(&endpoint, received, policy, spec)?;
            if active == 1 {
                return Ok(single);
            }
            endpoint.distribution = "balanced".into();
            let balanced = self.estimate(&endpoint, received, policy, spec)?;
            // Interpolate the active weight working set at fixed receive volume.
            // Both endpoint curves are independent operator measurements.
            let mut estimate = single.blend(
                balanced,
                f64::from(active - 1) / f64::from(maximum_active - 1),
            );
            // The active-count axis is modeled, not a measured distribution.
            estimate.extrapolated = true;
            return Ok(estimate);
        }
        match self.query(key, received) {
            Ok(value) => return Ok(super::ascend_shape::ShapeEstimate::measured(value)),
            Err(error)
                if policy == super::ascend_shape::AscendShapePolicy::Anchored
                    && error.is_missing_perf_data() => {}
            Err(error) => return Err(error),
        }
        let curve = self.curve(key)?;
        let (hidden, intermediate) = key.dimensions();
        let hidden = f64::from(hidden);
        let intermediate = f64::from(intermediate);
        super::ascend_shape::anchored_axis(
            received,
            curve.minimum,
            curve.maximum,
            &|value| self.query(key, value),
            &|count| {
                let experts = if key.distribution == "balanced" {
                    count.min(f64::from(key.local_experts))
                } else {
                    1.0
                };
                super::ascend_shape::roofline_ms(
                    spec,
                    6.0 * count * hidden * intermediate,
                    experts * 3.0 * hidden * intermediate + count * hidden * 4.0,
                    true,
                )
            },
        )
    }
    pub fn new(path: Option<PathBuf>) -> Self {
        Self {
            path,
            grids: OnceLock::new(),
            workloads: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn prepare_workload(
        &self,
        key: &AscendMoeChainKey,
        workload: &crate::operators::ascend_moe_chain::DecodeExpertWorkload,
        policy: super::ascend_shape::AscendShapePolicy,
        spec: &crate::common::system_spec::SystemSpec,
    ) -> Result<(), AicError> {
        workload.validate()?;
        let (local_experts, layers, _) = workload.expert_geometry();
        if key.phase != "decode"
            || key.execution_mode != "graph"
            || key.local_experts != local_experts
            || key.distribution != "balanced"
            || key.active_experts.is_some()
        {
            return Err(AicError::InvalidEngineConfig(
                "Decode routing workload requires its matching EP16 Graph expert chain".into(),
            ));
        }
        let id = format!("{}:{policy:?}", workload.cache_key());
        let mut cache = self
            .workloads
            .lock()
            .map_err(|_| AicError::InvalidPerfData("expert workload cache lock poisoned".into()))?;
        if cache.contains_key(&id) {
            return Ok(());
        }
        let mut sums: BTreeMap<Vec<u32>, BTreeMap<u32, (f64, u32)>> = BTreeMap::new();
        for sample in &workload.samples {
            let mut total = 0.0;
            for layer in &sample.layers {
                let mut critical: f64 = 0.0;
                for rank in layer {
                    if rank.received_tokens == 0 {
                        continue;
                    }
                    let mut rank_key = key.clone();
                    rank_key.active_experts = Some(rank.active_experts);
                    critical = critical.max(
                        self.estimate(&rank_key, rank.received_tokens, policy, spec)?
                            .latency_ms,
                    );
                }
                total += critical;
            }
            let entry = sums
                .entry(sample.request_counts.clone())
                .or_default()
                .entry(sample.sequence_length)
                .or_default();
            entry.0 += total / layers as f64;
            entry.1 += 1;
        }
        let curves = sums
            .into_iter()
            .map(|(shape, points)| {
                (
                    shape,
                    points
                        .into_iter()
                        .map(|(s, (sum, n))| (s, sum / f64::from(n)))
                        .collect(),
                )
            })
            .collect();
        cache.insert(id, Arc::new((workload.samples.clone(), curves)));
        Ok(())
    }

    pub fn estimate_workload(
        &self,
        key: &AscendMoeChainKey,
        workload: &crate::operators::ascend_moe_chain::DecodeExpertWorkload,
        requests: &[u32],
        sequence: u32,
        policy: super::ascend_shape::AscendShapePolicy,
        spec: &crate::common::system_spec::SystemSpec,
    ) -> Result<Option<super::ascend_shape::ShapeEstimate>, AicError> {
        let id = format!("{}:{policy:?}", workload.cache_key());
        let cached = self
            .workloads
            .lock()
            .map_err(|_| AicError::InvalidPerfData("expert workload cache lock poisoned".into()))?
            .get(&id)
            .cloned();
        let curves = match cached {
            Some(curves) => curves,
            None => {
                self.prepare_workload(key, workload, policy, spec)?;
                self.workloads.lock().map_err(|_| {
                    AicError::InvalidPerfData("expert workload cache lock poisoned".into())
                })?[&id]
                    .clone()
            }
        };
        if curves.0 != workload.samples {
            return Err(AicError::InvalidPerfData(
                "Decode routing samples conflict with their cached SHA256".into(),
            ));
        }
        let Some(points) = curves.1.get(requests) else {
            return Ok(None);
        };
        let lower = points.range(..=sequence).next_back();
        let upper = points.range(sequence..).next();
        let latency = match (lower, upper) {
            (Some((s0, v0)), Some((s1, v1))) if s0 != s1 => {
                v0 + (v1 - v0) * f64::from(sequence - s0) / f64::from(s1 - s0)
            }
            (Some((_, value)), Some(_)) => *value,
            (Some((_, value)), None) | (None, Some((_, value)))
                if workload.history_policy == "hold_boundary" =>
            {
                *value
            }
            _ => {
                return Err(AicError::PerfDatabase(
                    "Decode expert sequence length outside observed workload range".into(),
                ));
            }
        };
        Ok(Some(super::ascend_shape::ShapeEstimate {
            latency_ms: latency,
            extrapolated: true,
        }))
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
                "Ascend MoE chain rejects filtered primary sources".into(),
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
                    "Ascend MoE chain requires the exact requested system/backend/version".into(),
                ));
            }
        }
        Ok(Self::new(path))
    }

    fn curve(&self, key: &AscendMoeChainKey) -> Result<&Curve, AicError> {
        key.validate()?;
        let grids = self
            .grids
            .get_or_init(|| {
                let Some(path) = self.path.as_ref().filter(|path| path.is_file()) else {
                    return Err((false, "Ascend MoE chain primary data is missing".into()));
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
            AicError::PerfDatabase(format!("unmeasured Ascend MoE chain tuple: {key:?}"))
        })
    }

    pub fn ensure_key(&self, key: &AscendMoeChainKey) -> Result<(), AicError> {
        if key.active_experts.is_some() {
            key.validate()?;
            let mut endpoint = key.clone();
            endpoint.active_experts = None;
            self.curve(&endpoint)?;
            if key.active_experts != Some(key.local_experts) {
                endpoint.distribution = "single_expert".into();
                self.curve(&endpoint)?;
            }
            return Ok(());
        }
        self.curve(key).map(|_| ())
    }

    pub fn query(&self, key: &AscendMoeChainKey, received: u32) -> Result<f64, AicError> {
        let curve = self.curve(key)?;
        if received < curve.minimum || received > curve.maximum {
            return Err(AicError::PerfDatabase(format!(
                "Ascend MoE received tokens={received} outside measured [{}, {}]",
                curve.minimum, curve.maximum
            )));
        }
        curve.values.query(f64::from(received), &|_| 1.0)
    }
}

fn load(path: &Path) -> Result<Grids, AicError> {
    let reader = PerfReader::open(path)?;
    let phase = reader.col("phase")?;
    let experts = reader.col("local_experts")?;
    let distribution = reader.col("distribution")?;
    let mode = reader.col("execution_mode")?;
    let received = reader.col("received_tokens")?;
    let physical = reader.col("physical_rows")?;
    let counts = reader.col("expert_counts")?;
    let hidden = reader.col("hidden_size")?;
    let intermediate = reader.col("intermediate_size")?;
    let group_type = reader.col("group_list_type")?;
    let format = reader.col("weight_format")?;
    let input_dtype = reader.col("input_dtype")?;
    let output_dtype = reader.col("output_dtype")?;
    let gmm = reader.col("gmm_source")?;
    let activation = reader.col("activation_source")?;
    let latency = reader.col("latency_ms")?;
    let cv = reader.col("cv")?;
    let checks = reader.col("numerical_checks")?;
    let samples = reader.col("sample_count")?;
    let verification = reader.col("execution_verification")?;
    let provenance = reader.col("runtime_identity_sha256")?;
    let mut raw: BTreeMap<AscendMoeChainKey, BTreeMap<u32, f64>> = BTreeMap::new();
    let mut identity: Option<String> = None;
    for row in reader.rows()? {
        let row = row?;
        let key = AscendMoeChainKey {
            phase: row.str_owned(phase)?,
            local_experts: row.u32(experts)?,
            distribution: row.str_owned(distribution)?,
            execution_mode: row.str_owned(mode)?,
            active_experts: None,
        };
        key.validate()?;
        let count = row.u32(received)?;
        let group: Vec<u32> = serde_json::from_str(row.str(counts)?)
            .map_err(|error| AicError::InvalidPerfData(format!("invalid MoE counts: {error}")))?;
        let counts_valid = group.len() == key.local_experts as usize
            && group.iter().enumerate().all(|(index, value)| {
                *value
                    == if key.distribution == "balanced" {
                        count / key.local_experts
                            + u32::from((index as u32) < count % key.local_experts)
                    } else if index == 0 {
                        count
                    } else {
                        0
                    }
            });
        let measured = row.f64(latency)?;
        let (expected_hidden, expected_intermediate) = key.dimensions();
        let variation = row.f64(cv)?;
        let runtime = row.str(provenance)?;
        let expected = if key.execution_mode == "graph" {
            "captured_graph_replay"
        } else {
            "eager_invocation"
        };
        if !counts_valid
            || count == 0
            || row.u32(physical)? != key.physical_rows(count)
            || count > row.u32(physical)?
            || row.u32(hidden)? != expected_hidden
            || row.u32(intermediate)? != expected_intermediate
            || row.u32(group_type)? != 1
            || row.u32(format)? != 29
            || row.str(input_dtype)? != "int8"
            || row.str(output_dtype)? != "bfloat16"
            || row.str(gmm)? != "sglang.srt.hardware_backend.npu.moe.matmul.GroupedMatmul"
            || row.str(activation)? != "sgl_kernel_npu.activation.swiglu_quant.swiglu_quant"
            || !measured.is_finite()
            || measured <= 0.0
            || !variation.is_finite()
            || !(0.0..=0.1).contains(&variation)
            || row.u32(checks)? == 0
            || row.u32(samples)? < 3
            || row.str(verification)? != expected
            || runtime.len() != 64
            || !runtime.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(AicError::InvalidPerfData(format!(
                "unqualified Ascend MoE chain row: {key:?}"
            )));
        }
        if identity
            .as_ref()
            .is_some_and(|expected| expected != runtime)
        {
            return Err(AicError::InvalidPerfData(
                "Ascend MoE chain mixes runtime identities".into(),
            ));
        }
        if identity.is_none() {
            identity = Some(runtime.to_owned());
        }
        if raw
            .entry(key)
            .or_default()
            .insert(count, measured)
            .is_some()
        {
            return Err(AicError::InvalidPerfData(
                "duplicate Ascend MoE chain physical key".into(),
            ));
        }
    }
    if raw.is_empty() {
        return Err(AicError::InvalidPerfData(
            "empty Ascend MoE chain table".into(),
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
                    values: AxisCurve::from_map("received_tokens", points),
                },
            )
        })
        .collect())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::operators::ascend_moe_chain::AscendMoeChainOp;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
    use crate::perf_database::{PerfDbSources, PerfSource};

    fn key() -> AscendMoeChainKey {
        AscendMoeChainKey {
            phase: "decode".into(),
            local_experts: 16,
            distribution: "balanced".into(),
            execution_mode: "graph".into(),
            active_experts: None,
        }
    }

    fn columns() -> Vec<Col> {
        vec![
            Col::Str("phase", vec!["decode"; 2]),
            Col::I64("local_experts", vec![16; 2]),
            Col::Str("distribution", vec!["balanced"; 2]),
            Col::Str("execution_mode", vec!["graph"; 2]),
            Col::I64("received_tokens", vec![16, 32]),
            Col::I64("physical_rows", vec![4096; 2]),
            Col::Str(
                "expert_counts",
                vec![
                    "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]",
                    "[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]",
                ],
            ),
            Col::I64("hidden_size", vec![6144; 2]),
            Col::I64("intermediate_size", vec![2048; 2]),
            Col::I64("group_list_type", vec![1; 2]),
            Col::I64("weight_format", vec![29; 2]),
            Col::Str("input_dtype", vec!["int8"; 2]),
            Col::Str("output_dtype", vec!["bfloat16"; 2]),
            Col::Str(
                "gmm_source",
                vec!["sglang.srt.hardware_backend.npu.moe.matmul.GroupedMatmul"; 2],
            ),
            Col::Str(
                "activation_source",
                vec!["sgl_kernel_npu.activation.swiglu_quant.swiglu_quant"; 2],
            ),
            Col::F64("latency_ms", vec![1.0, 5.0]),
            Col::F64("cv", vec![0.01; 2]),
            Col::I64("numerical_checks", vec![1024; 2]),
            Col::I64("sample_count", vec![7; 2]),
            Col::Str("execution_verification", vec!["captured_graph_replay"; 2]),
            Col::Str(
                "runtime_identity_sha256",
                vec!["1111111111111111111111111111111111111111111111111111111111111111"; 2],
            ),
        ]
    }

    pub(crate) fn qwen_columns() -> Vec<Col> {
        let mut columns = columns();
        for column in &mut columns {
            match column {
                Col::I64("local_experts", values) => *values = vec![32; 2],
                Col::I64("physical_rows", values) => *values = vec![2560; 2],
                Col::I64("hidden_size", values) => *values = vec![4096; 2],
                Col::I64("intermediate_size", values) => *values = vec![1024; 2],
                Col::Str("expert_counts", values) => {
                    *values = vec![
                        "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]",
                        "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]",
                    ];
                }
                _ => {}
            }
        }
        columns
    }

    #[test]
    fn qwen_receive_volume_uses_its_own_matrix_and_buffer_geometry() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        write_parquet(&path, &qwen_columns());
        let table = AscendMoeChainTable::new(Some(path));
        let mut qwen = key();
        qwen.local_experts = 32;
        // Named fixture endpoints are1ms at receive16 and5ms at receive32.
        assert_eq!(table.query(&qwen, 24).unwrap(), 3.0);
        assert_eq!(qwen.physical_rows(24), 2560);
        assert_eq!(qwen.parameter_bytes(), 403439616.0);
        assert!(table.ensure_key(&key()).is_err());
        qwen.execution_mode = "eager".into();
        assert!(qwen.validate().is_err());
        qwen.execution_mode = "graph".into();
        qwen.phase = "prefill".into();
        assert!(qwen.validate().is_err());
    }

    #[test]
    fn qwen_receive_table_rejects_glm_shapes_and_buffers() {
        for field in ["physical_rows", "hidden_size", "intermediate_size"] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join(BASENAME);
            let mut columns = qwen_columns();
            for column in &mut columns {
                if let Col::I64(name, values) = column {
                    if *name == field {
                        *values = vec![6144; 2];
                    }
                }
            }
            write_parquet(&path, &columns);
            let table = AscendMoeChainTable::new(Some(path));
            let mut qwen = key();
            qwen.local_experts = 32;
            assert!(matches!(
                table.ensure_key(&qwen),
                Err(AicError::InvalidPerfData(_))
            ));
        }
    }

    #[test]
    fn qwen_observed_counts_keep_sixty_layers_and_padding_assignments() {
        use crate::operators::ascend_moe_chain::{
            DecodeExpertRank, DecodeExpertSample, DecodeExpertWorkload,
        };
        use sha2::{Digest, Sha256};
        let ranks = vec![
            DecodeExpertRank {
                received_tokens: 10,
                active_experts: 10,
            };
            16
        ];
        let samples = vec![DecodeExpertSample {
            request_counts: vec![3; 4],
            sequence_length: 8194,
            layers: vec![ranks; 60],
        }];
        let mut workload = DecodeExpertWorkload {
            observation_id: "qwen-synthetic-receive-contract".into(),
            quantization_sha256: "1".repeat(64),
            framework_commit: "2".repeat(40),
            samples_sha256: format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&samples).unwrap())
            ),
            input_tokens: 8192,
            graph_bucket: 4,
            history_policy: "hold_boundary".into(),
            samples,
        };
        workload.validate().unwrap();
        let mut qwen = key();
        qwen.local_experts = 32;
        let table = AscendMoeChainTable::new(None);
        let mut grids = Grids::new();
        // All sixty layers have a specified0.2ms ten-active-expert endpoint.
        grids.insert(
            qwen.clone(),
            Curve {
                values: AxisCurve::from_map("received_tokens", BTreeMap::from([(10, 0.2)])),
                minimum: 10,
                maximum: 10,
            },
        );
        table.grids.set(Ok(grids)).ok().unwrap();
        let spec = crate::common::system_spec::SystemSpec::load(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../python/aisimulate/src/aisimulate_core/systems/ascend910c.yaml"),
        )
        .unwrap();
        let policy = super::super::ascend_shape::AscendShapePolicy::Measured;
        let value = table
            .estimate_workload(&qwen, &workload, &[3; 4], 8194, policy, &spec)
            .unwrap()
            .unwrap();
        assert!((value.latency_ms - 0.2).abs() < 1e-12);
        assert!(
            table
                .prepare_workload(&key(), &workload, policy, &spec)
                .is_err()
        );
        workload.graph_bucket = 8;
        assert!(workload.validate().is_err());
        workload.graph_bucket = 4;
        workload.samples[0].layers[0][0].received_tokens += 1;
        assert!(workload.validate().is_err());
    }

    #[test]
    fn received_counts_interpolate_without_treating_capacity_as_work() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        write_parquet(&path, &columns());
        let table = AscendMoeChainTable::new(Some(path));
        assert_eq!(table.query(&key(), 16).unwrap(), 1.0);
        assert_eq!(table.query(&key(), 24).unwrap(), 3.0);
        assert!(
            table
                .query(&key(), 4096)
                .unwrap_err()
                .is_missing_perf_data()
        );
        let mut other = key();
        other.distribution = "single_expert".into();
        assert!(table.query(&other, 24).unwrap_err().is_missing_perf_data());
        other = key();
        other.local_experts = 8;
        assert!(table.query(&other, 24).unwrap_err().is_missing_perf_data());
        other = key();
        other.phase = "prefill".into();
        assert!(table.query(&other, 24).unwrap_err().is_missing_perf_data());
    }

    #[test]
    fn active_expert_axis_preserves_endpoints_and_the_fixed_cost() {
        // Independent specification: a 32-token single-expert chain costs
        // 0.08 ms and the 16-active-expert chain costs 0.44 ms. Five of the
        // fifteen additional active experts therefore cost 0.12 ms.
        let mut grids = Grids::new();
        for (distribution, latency) in [("single_expert", 0.08), ("balanced", 0.44)] {
            let mut endpoint = key();
            endpoint.distribution = distribution.into();
            grids.insert(
                endpoint,
                Curve {
                    values: AxisCurve::from_map("received_tokens", BTreeMap::from([(32, latency)])),
                    minimum: 32,
                    maximum: 32,
                },
            );
        }
        let table = AscendMoeChainTable::new(None);
        assert!(table.grids.set(Ok(grids)).is_ok());
        let spec = crate::common::system_spec::SystemSpec::load(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../python/aisimulate/src/aisimulate_core/systems/ascend910c.yaml"),
        )
        .unwrap();
        let policy = super::super::ascend_shape::AscendShapePolicy::Measured;
        let mut sparse = key();
        sparse.active_experts = Some(6);
        let value = table.estimate(&sparse, 32, policy, &spec).unwrap();
        assert!((value.latency_ms - 0.20).abs() < 1e-12);
        assert_eq!(
            value.result().source,
            crate::operators::base::Source::Estimated
        );
        for (active, expected) in [(1, 0.08), (16, 0.44)] {
            sparse.active_experts = Some(active);
            let value = table.estimate(&sparse, 32, policy, &spec).unwrap();
            assert_eq!(value.latency_ms, expected);
            assert!(!value.extrapolated);
        }
        sparse.active_experts = Some(9);
        assert!(table.estimate(&sparse, 8, policy, &spec).is_err());
        for active in [0, 17] {
            sparse.active_experts = Some(active);
            assert!(sparse.validate().is_err());
        }
        sparse.active_experts = Some(2);
        sparse.distribution = "single_expert".into();
        assert!(sparse.validate().is_err());
    }

    #[test]
    fn observed_workload_uses_each_layers_slowest_cost_and_preserves_shape_fallback() {
        use crate::operators::ascend_moe_chain::{
            DecodeExpertRank, DecodeExpertSample, DecodeExpertWorkload,
        };
        use sha2::{Digest, Sha256};
        let mut grids = Grids::new();
        for (distribution, latency) in [("single_expert", 0.08), ("balanced", 0.44)] {
            let mut endpoint = key();
            endpoint.distribution = distribution.into();
            grids.insert(
                endpoint,
                Curve {
                    values: AxisCurve::from_map(
                        "received_tokens",
                        BTreeMap::from([(16, latency), (32, latency)]),
                    ),
                    minimum: 16,
                    maximum: 32,
                },
            );
        }
        let table = AscendMoeChainTable::new(None);
        assert!(table.grids.set(Ok(grids)).is_ok());
        let mut early = vec![
            DecodeExpertRank {
                received_tokens: 16,
                active_experts: 1
            };
            16
        ];
        // The largest receive volume belongs to rank0, but rank1 is slower
        // because its six active experts cost0.20 ms at the specified endpoints.
        early[0].received_tokens = 32;
        early[1].active_experts = 6;
        early[15] = DecodeExpertRank {
            received_tokens: 0,
            active_experts: 0,
        };
        let late = vec![
            DecodeExpertRank {
                received_tokens: 16,
                active_experts: 16
            };
            16
        ];
        let samples = vec![
            DecodeExpertSample {
                request_counts: vec![3; 4],
                sequence_length: 8194,
                layers: vec![early; 75],
            },
            DecodeExpertSample {
                request_counts: vec![3; 4],
                sequence_length: 9218,
                layers: vec![late; 75],
            },
        ];
        let mut workload = DecodeExpertWorkload {
            observation_id: "synthetic-routing-counts".into(),
            quantization_sha256: "1".repeat(64),
            framework_commit: "2".repeat(40),
            samples_sha256: format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&samples).unwrap())
            ),
            input_tokens: 8192,
            graph_bucket: 8,
            history_policy: "hold_boundary".into(),
            samples,
        };
        workload.validate().unwrap();
        let spec = crate::common::system_spec::SystemSpec::load(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../python/aisimulate/src/aisimulate_core/systems/ascend910c.yaml"),
        )
        .unwrap();
        let policy = super::super::ascend_shape::AscendShapePolicy::Measured;
        for (sequence, expected) in [
            (8193, 0.20),
            (8194, 0.20),
            (8706, 0.32),
            (9218, 0.44),
            (10000, 0.44),
        ] {
            let result = table
                .estimate_workload(&key(), &workload, &[3; 4], sequence, policy, &spec)
                .unwrap()
                .unwrap();
            assert!((result.latency_ms - expected).abs() < 1e-12);
            assert!(result.extrapolated);
        }
        assert!(
            table
                .estimate_workload(&key(), &workload, &[2; 4], 8706, policy, &spec)
                .unwrap()
                .is_none()
        );
        workload.history_policy = "deny".into();
        assert!(
            table
                .estimate_workload(&key(), &workload, &[3; 4], 10000, policy, &spec)
                .is_err()
        );
        workload.samples[0].layers[0][0].received_tokens += 1;
        assert!(workload.validate().is_err());
        assert!(
            table
                .estimate_workload(&key(), &workload, &[3; 4], 8194, policy, &spec)
                .is_err()
        );
    }

    #[test]
    fn count_vectors_precision_and_execution_must_match_the_contract() {
        for (index, replacement) in [
            (5, Col::I64("physical_rows", vec![16, 32])),
            (
                6,
                Col::Str(
                    "expert_counts",
                    vec!["[16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]"; 2],
                ),
            ),
            (9, Col::I64("group_list_type", vec![0; 2])),
            (10, Col::I64("weight_format", vec![2; 2])),
            (11, Col::Str("input_dtype", vec!["bfloat16"; 2])),
            (16, Col::F64("cv", vec![0.2; 2])),
            (
                19,
                Col::Str("execution_verification", vec!["eager_invocation"; 2]),
            ),
        ] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join(BASENAME);
            let mut data = columns();
            data[index] = replacement;
            write_parquet(&path, &data);
            assert!(matches!(
                AscendMoeChainTable::new(Some(path)).ensure_key(&key()),
                Err(AicError::InvalidPerfData(_))
            ));
        }
        assert!(
            AscendMoeChainTable::new(None)
                .ensure_key(&key())
                .unwrap_err()
                .is_missing_perf_data()
        );
    }

    #[test]
    fn input_padding_precedes_routing_ratio_and_scales_are_resident() {
        let mut operation = AscendMoeChainOp {
            name: "experts".into(),
            scale_factor: 1.0,
            key: key(),
            token_numerator: 8,
            token_denominator: 16,
            input_token_multiple: 8,
            decode_workload: None,
            profile_request_counts: None,
        };
        assert_eq!(operation.received_tokens(9).unwrap(), 8);
        assert_eq!(operation.received_tokens(0).unwrap(), 0);
        assert_eq!(operation.weight_bytes(), 604_962_816.0);
        operation.token_numerator = u32::MAX;
        operation.token_denominator = 1;
        assert!(operation.received_tokens(9).is_err());
    }

    #[test]
    fn primary_source_is_exact_and_does_not_override_veto() {
        let root = tempfile::tempdir().unwrap();
        let requested = root.path().join("data/ascend910c/sglang/0.5.22");
        for relative in [
            "data/ascend910c/moe/sglang/0.5.22",
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
                AscendMoeChainTable::with_sources(&requested, &resolver)
                    .unwrap()
                    .query(&key(), 24)
                    .unwrap(),
                3.0
            );
        }
        let resolver = SourceResolver::fixed(PerfDbSources::from([(
            BASENAME.into(),
            vec![PerfSource(
                root.path()
                    .join("data/other/moe/sglang/0.5.22")
                    .join(BASENAME),
                None,
            )],
        )]));
        assert!(matches!(
            AscendMoeChainTable::with_sources(&requested, &resolver),
            Err(AicError::InvalidPerfData(_))
        ));
        let resolver = SourceResolver::fixed(PerfDbSources::from([(BASENAME.into(), Vec::new())]));
        assert!(
            AscendMoeChainTable::with_sources(&requested, &resolver)
                .unwrap()
                .ensure_key(&key())
                .unwrap_err()
                .is_missing_perf_data()
        );
    }
}
