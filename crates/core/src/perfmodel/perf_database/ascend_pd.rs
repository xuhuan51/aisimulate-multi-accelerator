use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};

use super::SourceResolver;
use super::axis_curve::AxisCurve;
use super::parquet_loader::PerfReader;
use crate::common::error::AicError;

mod dense_gqa;

pub const BASENAME: &str = "ascend_pd_perf.parquet";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendPdKey {
    pub source_tp: u32,
    pub destination_tp: u32,
    pub page_layout: String,
}

impl AscendPdKey {
    pub fn validate(&self) -> Result<(), AicError> {
        if !matches!(self.source_tp, 1 | 2 | 4 | 8 | 16)
            || !matches!(self.destination_tp, 1 | 2 | 4 | 8 | 16)
            || !matches!(self.page_layout.as_str(), "contiguous" | "fragmented")
        {
            return Err(AicError::InvalidEngineConfig(format!(
                "invalid Ascend PD mapping: {self:?}"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct KvTransferCurve {
    key: AscendPdKey,
    values: AxisCurve,
    minimum: u32,
    maximum: u32,
    identity: String,
    policy: super::ascend_shape::AscendShapePolicy,
    dense_placement: Option<dense_gqa::DensePlacement>,
}

impl KvTransferCurve {
    pub(crate) fn serializes_same_destination(&self) -> bool {
        self.dense_placement
            .as_ref()
            .is_some_and(dense_gqa::DensePlacement::serializes_same_destination)
    }

    pub(crate) fn for_dense_gqa(
        source: &crate::perfmodel::ForwardPassPerfModelConfig,
        destination: &crate::perfmodel::ForwardPassPerfModelConfig,
    ) -> Result<Arc<Self>, AicError> {
        dense_gqa::load(source, destination)
    }

    pub(crate) fn validate_placement_counts(
        &self,
        prefill_workers: usize,
        decode_workers: usize,
    ) -> Result<(), AicError> {
        if self.dense_placement.as_ref().is_some_and(|placement| {
            !placement.matches_counts(prefill_workers, decode_workers)
        }) {
            return Err(AicError::InvalidEngineConfig(
                "Qwen transfer placement does not match the replay replica counts".into(),
            ));
        }
        Ok(())
    }

    pub fn with_shape_policy(&self, policy: super::ascend_shape::AscendShapePolicy) -> Arc<Self> {
        let mut curve = self.clone();
        curve.policy = policy;
        Arc::new(curve)
    }
    #[cfg(test)]
    pub(crate) fn test_curve() -> Arc<Self> {
        let points = BTreeMap::from([(128, 1.0), (256, 3.0)]);
        Arc::new(Self {
            key: AscendPdKey {
                source_tp: 8,
                destination_tp: 8,
                page_layout: "contiguous".into(),
            },
            values: AxisCurve::from_map("tokens", points),
            minimum: 128,
            maximum: 256,
            identity: "1".repeat(64),
            policy: Default::default(),
            dense_placement: None,
        })
    }
    pub fn latency_ms(&self, tokens: usize) -> Result<f64, AicError> {
        self.latency_ms_for_workers(tokens, None)
    }

    pub(crate) fn latency_ms_for_workers(
        &self,
        tokens: usize,
        workers: Option<(usize, usize)>,
    ) -> Result<f64, AicError> {
        if tokens == 0 {
            return Ok(0.0);
        }
        let values = match &self.dense_placement {
            Some(placement) => placement.select(&self.values, workers)?,
            None => &self.values,
        };
        if self.dense_placement.is_some() && tokens > 32768 {
            return Err(AicError::InvalidEngineConfig(
                "Qwen transfer exceeds the qualified 32K context".into(),
            ));
        }
        let rounded = tokens
            .div_ceil(128)
            .checked_mul(128)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| AicError::InvalidEngineConfig("PD page rounding overflow".into()))?;
        if rounded < self.minimum || rounded > self.maximum {
            if self.policy == super::ascend_shape::AscendShapePolicy::Anchored {
                return super::ascend_shape::anchored_axis(
                    rounded,
                    self.minimum,
                    self.maximum,
                    &|value| values.query(f64::from(value), &|_| 1.0),
                    &|tokens| Ok(1.0 + tokens / 128.0),
                )
                .map(|value| value.latency_ms);
            }
            return Err(AicError::PerfDatabase(format!(
                "Ascend PD tokens={rounded} outside measured [{}, {}] for {:?}",
                self.minimum, self.maximum, self.key
            )));
        }
        values.query(f64::from(rounded), &|_| 1.0)
    }

    pub fn metadata(&self) -> serde_json::Value {
        if let Some(placement) = &self.dense_placement {
            let mut metadata = placement.description();
            metadata["key"] = serde_json::json!(self.key);
            metadata["minimum_tokens"] = serde_json::json!(self.minimum);
            metadata["maximum_measured_tokens"] = serde_json::json!(self.maximum);
            metadata["extrapolation"] = serde_json::json!(
                "boundary-anchored page work from the largest measured payload up to 32768"
            );
            return metadata;
        }
        serde_json::json!({
            "source": "measured_ascend_memfabric",
            "key": self.key,
            "minimum_tokens": self.minimum,
            "maximum_tokens": self.maximum,
            "page_size": 128,
            "bytes_per_token_per_replica": 109824,
            "endpoint_scope": "one contiguous attention-DP group per role on distinct nodes",
            "runtime_identity_sha256": self.identity,
            "shape_policy": self.policy,
            "extrapolation": if self.policy == super::ascend_shape::AscendShapePolicy::Anchored { Some("boundary-anchored page work with a measured minimum latency floor; estimates outside stated token bounds") } else { None },
            "scope": "isolated request with source-selected sender fanout; other attention-DP groups excluded; no bootstrap or request contention"
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};

    fn key() -> AscendPdKey {
        AscendPdKey {
            source_tp: 8,
            destination_tp: 8,
            page_layout: "contiguous".into(),
        }
    }

    fn columns() -> Vec<Col> {
        vec![
            Col::I64("source_tp", vec![8; 2]),
            Col::I64("destination_tp", vec![8; 2]),
            Col::Str("page_layout", vec!["contiguous"; 2]),
            Col::I64("tokens", vec![128, 512]),
            Col::Str("protocol", vec!["sdma"; 2]),
            Col::I64("active_requests", vec![1; 2]),
            Col::I64("ranks_per_node", vec![16; 2]),
            Col::I64("num_layers", vec![78; 2]),
            Col::I64("page_size", vec![128; 2]),
            Col::I64("reference_pool_tokens", vec![53248; 2]),
            Col::Str("component_widths", vec!["[512,64,128]"; 2]),
            Col::Str("dtype", vec!["bfloat16"; 2]),
            Col::I64("payload_bytes_per_replica", vec![14_057_472, 56_229_888]),
            Col::I64("total_wire_bytes", vec![112_459_776, 449_839_104]),
            Col::I64("segments_per_buffer", vec![1; 2]),
            Col::I64("verified_receiver_copies", vec![16; 2]),
            Col::Str(
                "execution_verification",
                vec!["full_payload_sha256_before_after"; 2],
            ),
            Col::Str(
                "kernel_source",
                vec!["sglang.srt.disaggregation.ascend.conn.AscendKVManager.send_kvcache"; 2],
            ),
            Col::F64("latency_ms", vec![1.0, 4.0]),
            Col::F64("cv", vec![0.01; 2]),
            Col::I64("sample_count", vec![7; 2]),
            Col::Str(
                "runtime_identity_sha256",
                vec!["1111111111111111111111111111111111111111111111111111111111111111"; 2],
            ),
        ]
    }

    #[test]
    fn ascend_pd_rounds_actual_pages_and_keeps_measured_bounds() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        write_parquet(&path, &columns());
        let table = AscendPdTable::new(Some(path));
        let curve = table.curve(&key()).unwrap();
        assert_eq!(curve.latency_ms(0).unwrap(), 0.0);
        assert_eq!(curve.latency_ms(1).unwrap(), 1.0);
        assert_eq!(curve.latency_ms(129).unwrap(), 2.0);
        assert_eq!(curve.latency_ms(512).unwrap(), 4.0);
        assert!(curve.latency_ms(513).unwrap_err().is_missing_perf_data());
        assert!(curve.latency_ms(usize::MAX).is_err());
        let mut missing = key();
        missing.source_tp = 4;
        assert!(table.curve(&missing).unwrap_err().is_missing_perf_data());
        missing = key();
        missing.page_layout = "fragmented".into();
        assert!(table.curve(&missing).unwrap_err().is_missing_perf_data());
    }

    #[test]
    fn ascend_pd_rejects_wrong_payload_and_unverified_replica_copies() {
        for (index, replacement) in [
            (11, Col::Str("dtype", vec!["int8"; 2])),
            (
                13,
                Col::I64("total_wire_bytes", vec![14_057_472, 56_229_888]),
            ),
            (15, Col::I64("verified_receiver_copies", vec![2; 2])),
            (19, Col::F64("cv", vec![0.2; 2])),
        ] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join(BASENAME);
            let mut values = columns();
            values[index] = replacement;
            write_parquet(&path, &values);
            assert!(matches!(
                AscendPdTable::new(Some(path)).curve(&key()),
                Err(AicError::InvalidPerfData(_))
            ));
        }
    }

    #[test]
    fn independent_curves_keep_separate_collection_identities() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        let mut values = columns();
        values[21] = Col::Str(
            "runtime_identity_sha256",
            vec![
                "1111111111111111111111111111111111111111111111111111111111111111",
                "2222222222222222222222222222222222222222222222222222222222222222",
            ],
        );
        write_parquet(&path, &values);
        assert!(matches!(
            AscendPdTable::new(Some(path.clone())).curve(&key()),
            Err(AicError::InvalidPerfData(_))
        ));
        values[0] = Col::I64("source_tp", vec![8, 4]);
        write_parquet(&path, &values);
        let table = AscendPdTable::new(Some(path));
        assert_eq!(
            table.curve(&key()).unwrap().metadata()["runtime_identity_sha256"],
            "1".repeat(64)
        );
        let mut second = key();
        second.source_tp = 4;
        assert_eq!(
            table.curve(&second).unwrap().metadata()["runtime_identity_sha256"],
            "2".repeat(64)
        );
    }
}

type Grids = BTreeMap<AscendPdKey, Arc<KvTransferCurve>>;

pub struct AscendPdTable {
    path: Option<PathBuf>,
    grids: OnceLock<Result<Grids, (bool, String)>>,
}

impl AscendPdTable {
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
                "Ascend PD rejects filtered primary sources".into(),
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
                            .join("kv_transfer")
                            .join(data_root.parent()?.file_name()?)
                            .join(data_root.file_name()?)
                            .join(BASENAME),
                    )
                });
            if path != &data_root.join(BASENAME) && family.as_ref() != Some(path) {
                return Err(AicError::InvalidPerfData(
                    "Ascend PD requires the exact requested system/backend/version".into(),
                ));
            }
        }
        Ok(Self::new(path))
    }

    pub fn curve(&self, key: &AscendPdKey) -> Result<Arc<KvTransferCurve>, AicError> {
        key.validate()?;
        let grids = self
            .grids
            .get_or_init(|| {
                let Some(path) = self.path.as_ref().filter(|path| path.is_file()) else {
                    return Err((false, "Ascend PD primary data is missing".into()));
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
        grids
            .get(key)
            .cloned()
            .ok_or_else(|| AicError::PerfDatabase(format!("unmeasured Ascend PD tuple: {key:?}")))
    }
}

fn load(path: &Path) -> Result<Grids, AicError> {
    let reader = PerfReader::open(path)?;
    let source_tp = reader.col("source_tp")?;
    let destination_tp = reader.col("destination_tp")?;
    let layout = reader.col("page_layout")?;
    let tokens = reader.col("tokens")?;
    let protocol = reader.col("protocol")?;
    let active = reader.col("active_requests")?;
    let node = reader.col("ranks_per_node")?;
    let layers = reader.col("num_layers")?;
    let page = reader.col("page_size")?;
    let pool = reader.col("reference_pool_tokens")?;
    let widths = reader.col("component_widths")?;
    let dtype = reader.col("dtype")?;
    let payload = reader.col("payload_bytes_per_replica")?;
    let wire = reader.col("total_wire_bytes")?;
    let segments = reader.col("segments_per_buffer")?;
    let copies = reader.col("verified_receiver_copies")?;
    let verification = reader.col("execution_verification")?;
    let kernel = reader.col("kernel_source")?;
    let latency = reader.col("latency_ms")?;
    let cv = reader.col("cv")?;
    let samples = reader.col("sample_count")?;
    let provenance = reader.col("runtime_identity_sha256")?;
    let mut raw: BTreeMap<AscendPdKey, BTreeMap<u32, f64>> = BTreeMap::new();
    let mut identities: BTreeMap<AscendPdKey, String> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = AscendPdKey {
            source_tp: row.u32(source_tp)?,
            destination_tp: row.u32(destination_tp)?,
            page_layout: row.str_owned(layout)?,
        };
        key.validate()?;
        let count = row.u32(tokens)?;
        let value = row.f64(latency)?;
        let variation = row.f64(cv)?;
        let runtime = row.str(provenance)?;
        let expected_segments = if key.page_layout == "contiguous" {
            1
        } else {
            count / 128
        };
        if count == 0
            || count % 128 != 0
            || count > 16384
            || row.str(protocol)? != "sdma"
            || row.u32(active)? != 1
            || row.u32(node)? != 16
            || row.u32(layers)? != 78
            || row.u32(page)? != 128
            || row.u32(pool)? != 53248
            || row.str(widths)? != "[512,64,128]"
            || row.str(dtype)? != "bfloat16"
            || row.u64(payload)? != u64::from(count) * 109824
            || row.u64(wire)? != u64::from(count) * 109824 * u64::from(key.destination_tp)
            || row.u32(segments)? != expected_segments
            || row.u32(copies)? != 2 * key.destination_tp
            || row.str(verification)? != "full_payload_sha256_before_after"
            || row.str(kernel)?
                != "sglang.srt.disaggregation.ascend.conn.AscendKVManager.send_kvcache"
            || !value.is_finite()
            || value <= 0.0
            || !variation.is_finite()
            || !(0.0..=0.1).contains(&variation)
            || row.u32(samples)? < 3
            || runtime.len() != 64
            || !runtime.bytes().all(|value| value.is_ascii_hexdigit())
        {
            return Err(AicError::InvalidPerfData(format!(
                "unqualified Ascend PD row: {key:?}"
            )));
        }
        if identities
            .get(&key)
            .is_some_and(|expected| expected != runtime)
        {
            return Err(AicError::InvalidPerfData(
                "Ascend PD curve mixes collection identities".into(),
            ));
        }
        identities
            .entry(key.clone())
            .or_insert_with(|| runtime.to_owned());
        if raw.entry(key).or_default().insert(count, value).is_some() {
            return Err(AicError::InvalidPerfData(
                "duplicate Ascend PD physical key".into(),
            ));
        }
    }
    if raw.is_empty() {
        return Err(AicError::InvalidPerfData("empty Ascend PD table".into()));
    }
    Ok(raw
        .into_iter()
        .map(|(key, points)| {
            let curve = Arc::new(KvTransferCurve {
                key: key.clone(),
                minimum: *points.first_key_value().unwrap().0,
                maximum: *points.last_key_value().unwrap().0,
                values: AxisCurve::from_map("tokens", points),
                identity: identities.remove(&key).unwrap(),
                policy: Default::default(),
                dense_placement: None,
            });
            (key, curve)
        })
        .collect())
}
#[test]
fn pd_extrapolation_keeps_page_rounding_and_isolates_policy_views() {
    let measured = KvTransferCurve::test_curve();
    let anchored =
        measured.with_shape_policy(crate::perf_database::ascend_shape::AscendShapePolicy::Anchored);
    assert_eq!(anchored.latency_ms(256).unwrap(), 3.0);
    assert_eq!(anchored.latency_ms(257).unwrap(), 1.0 + 8.0 / 3.0);
    assert_eq!(anchored.latency_ms(0).unwrap(), 0.0);
    assert!(measured.latency_ms(257).is_err());
    assert_eq!(anchored.metadata()["shape_policy"], "anchored");
    assert_eq!(measured.metadata()["shape_policy"], "measured");
}
