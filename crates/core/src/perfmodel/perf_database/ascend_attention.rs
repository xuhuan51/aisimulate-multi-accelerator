use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::SourceResolver;
use super::ascend_shape::{AscendShapePolicy, ShapeEstimate, attention_work};
use super::axis_curve::AxisCurve;
use super::parquet_loader::PerfReader;
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;

pub const BASENAME: &str = "ascend_attention_perf.parquet";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendAttentionKey {
    pub operation: String,
    pub phase: String,
    pub local_heads: u32,
    pub execution_mode: String,
}

impl AscendAttentionKey {
    pub fn validate(&self) -> Result<(), AicError> {
        let heads_valid = match self.operation.as_str() {
            "lightning_indexer" => self.local_heads == 32,
            "sparse_attention" => matches!(self.local_heads, 4 | 8 | 16 | 32 | 64),
            _ => false,
        };
        if !heads_valid
            || !matches!(self.phase.as_str(), "prefill" | "decode")
            || !matches!(self.execution_mode.as_str(), "eager" | "graph")
        {
            return Err(AicError::InvalidEngineConfig(format!(
                "invalid Ascend attention identity: {self:?}"
            )));
        }
        Ok(())
    }

    pub fn kernel_source(&self) -> &'static str {
        if self.operation == "lightning_indexer" {
            "torch_npu.npu_lightning_indexer"
        } else {
            "torch_npu.npu_sparse_flash_attention"
        }
    }

    fn partition(&self, active_batch: u32, padded_batch: u32) -> u32 {
        if self.phase == "prefill" || self.execution_mode == "graph" || padded_batch != active_batch
        {
            padded_batch
        } else {
            0
        }
    }
}

struct BoundedCurve {
    points: AxisCurve,
    minimum: u32,
    maximum: u32,
}

impl BoundedCurve {
    fn new(points: BTreeMap<u32, f64>) -> Self {
        Self {
            minimum: *points.first_key_value().unwrap().0,
            maximum: *points.last_key_value().unwrap().0,
            points: AxisCurve::from_map("sequence_or_tokens", points),
        }
    }

    fn query(&self, coordinate: u32) -> Result<f64, AicError> {
        if coordinate < self.minimum || coordinate > self.maximum {
            return Err(AicError::PerfDatabase(format!(
                "Ascend attention axis={coordinate} outside measured [{}, {}]",
                self.minimum, self.maximum
            )));
        }
        self.points.query(f64::from(coordinate), &|_| 1.0)
    }

    fn estimate(
        &self,
        coordinate: u32,
        work: &dyn Fn(u32) -> Result<f64, AicError>,
    ) -> Result<ShapeEstimate, AicError> {
        let anchor = coordinate.clamp(self.minimum, self.maximum);
        let value = ShapeEstimate::measured(self.query(anchor)?);
        if anchor == coordinate {
            return Ok(value);
        }
        let floor = self
            .points
            .iter()
            .map(|(_, value)| value)
            .fold(f64::INFINITY, f64::min);
        value.scale_work(work(coordinate)?, work(anchor)?, floor)
    }
}

pub(crate) struct Plane(BTreeMap<u32, BoundedCurve>);

impl Plane {
    pub(crate) fn from_raw(curves: BTreeMap<u32, BTreeMap<u32, f64>>) -> Self {
        Self(
            curves
                .into_iter()
                .map(|(axis, points)| (axis, BoundedCurve::new(points)))
                .collect(),
        )
    }

    pub(crate) fn query(&self, outer: u32, inner: u32) -> Result<f64, AicError> {
        if let Some(curve) = self.0.get(&outer) {
            return curve.query(inner);
        }
        let lower = self.0.range(..outer).next_back();
        let upper = self.0.range(outer..).next();
        match (lower, upper) {
            (Some((lower_axis, lower_curve)), Some((upper_axis, upper_curve))) => {
                let lower_value = lower_curve.query(inner)?;
                let upper_value = upper_curve.query(inner)?;
                let weight = f64::from(outer - lower_axis) / f64::from(upper_axis - lower_axis);
                Ok(lower_value + (upper_value - lower_value) * weight)
            }
            _ => Err(AicError::PerfDatabase(format!(
                "Ascend attention active-batch/prefix={outer} lacks measured brackets"
            ))),
        }
    }

    fn estimate(
        &self,
        outer: u32,
        inner: u32,
        work: &dyn Fn(u32, u32) -> Result<f64, AicError>,
    ) -> Result<ShapeEstimate, AicError> {
        if let Some(curve) = self.0.get(&outer) {
            return curve.estimate(inner, &|value| work(outer, value));
        }
        let lower = self.0.range(..outer).next_back();
        let upper = self.0.range(outer..).next();
        if let (Some((lower_axis, lower_curve)), Some((upper_axis, upper_curve))) = (lower, upper) {
            return Ok(lower_curve
                .estimate(inner, &|value| work(*lower_axis, value))?
                .blend(
                    upper_curve.estimate(inner, &|value| work(*upper_axis, value))?,
                    f64::from(outer - lower_axis) / f64::from(upper_axis - lower_axis),
                ));
        }
        let (anchor, curve) = lower
            .or(upper)
            .ok_or_else(|| AicError::PerfDatabase("empty Ascend shape plane".into()))?;
        let value = curve.estimate(inner, &|value| work(*anchor, value))?;
        let floor = curve
            .points
            .iter()
            .map(|(_, value)| value)
            .fold(f64::INFINITY, f64::min);
        value.scale_work(work(outer, inner)?, work(*anchor, inner)?, floor)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_attention_grid(
    partitions: &BTreeMap<u32, Plane>,
    prefill: bool,
    active: u32,
    padded: u32,
    sequence: u32,
    prefix: u32,
    work: &dyn Fn(u32, u32, u32) -> Result<f64, AicError>,
) -> Result<ShapeEstimate, AicError> {
    if !prefill {
        let plane = partitions.get(&padded).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "unmeasured Ascend attention padding bucket={padded}"
            ))
        })?;
        return plane.estimate(active, sequence, &|batch, length| work(batch, length, 0));
    }
    if let Some(plane) = partitions.get(&active) {
        return plane.estimate(prefix, sequence, &|past, tokens| work(active, tokens, past));
    }
    let lower = partitions.range(..active).next_back();
    let upper = partitions.range(active..).next();
    if let (Some((lower_batch, lower_plane)), Some((upper_batch, upper_plane))) = (lower, upper) {
        return Ok(lower_plane
            .estimate(prefix, sequence, &|past, tokens| {
                work(*lower_batch, tokens, past)
            })?
            .blend(
                upper_plane.estimate(prefix, sequence, &|past, tokens| {
                    work(*upper_batch, tokens, past)
                })?,
                f64::from(active - lower_batch) / f64::from(upper_batch - lower_batch),
            ));
    }
    let (anchor, plane) = lower
        .or(upper)
        .ok_or_else(|| AicError::PerfDatabase("empty Ascend batch grid".into()))?;
    let value = plane.estimate(prefix, sequence, &|past, tokens| {
        work(*anchor, tokens, past)
    })?;
    value.scale_work(
        work(active, sequence, prefix)?,
        work(*anchor, sequence, prefix)?,
        plane
            .0
            .values()
            .flat_map(|curve| curve.points.iter().map(|(_, value)| value))
            .fold(f64::INFINITY, f64::min),
    )
}

pub(crate) fn query_prefill_batch(
    partitions: &BTreeMap<u32, Plane>,
    batch: u32,
    prefix: u32,
    sequence: u32,
) -> Result<f64, AicError> {
    if let Some(plane) = partitions.get(&batch) {
        return plane.query(prefix, sequence);
    }
    let lower = partitions.range(..batch).next_back();
    let upper = partitions.range(batch..).next();
    match (lower, upper) {
        (Some((lower_batch, lower_plane)), Some((upper_batch, upper_plane))) => {
            let lower_value = lower_plane.query(prefix, sequence)?;
            let upper_value = upper_plane.query(prefix, sequence)?;
            let weight = f64::from(batch - lower_batch) / f64::from(upper_batch - lower_batch);
            Ok(lower_value + (upper_value - lower_value) * weight)
        }
        _ => Err(AicError::PerfDatabase(format!(
            "Ascend prefill batch={batch} lacks measured brackets"
        ))),
    }
}

type Grids = BTreeMap<AscendAttentionKey, BTreeMap<u32, Plane>>;

pub struct AscendAttentionTable {
    path: Option<PathBuf>,
    grids: OnceLock<Result<Grids, (bool, String)>>,
}

impl AscendAttentionTable {
    #[allow(clippy::too_many_arguments)]
    pub fn estimate(
        &self,
        key: &AscendAttentionKey,
        active: u32,
        padded: u32,
        sequence: u32,
        prefix: u32,
        policy: AscendShapePolicy,
        spec: &SystemSpec,
    ) -> Result<ShapeEstimate, AicError> {
        match self.query(key, active, padded, sequence, prefix) {
            Ok(value) => return Ok(ShapeEstimate::measured(value)),
            Err(error) if policy == AscendShapePolicy::Anchored && error.is_missing_perf_data() => {
            }
            Err(error) => return Err(error),
        }
        let partitions = self.grids()?.get(key).ok_or_else(|| {
            AicError::PerfDatabase(format!("unmeasured Ascend attention identity: {key:?}"))
        })?;
        if key.phase == "prefill" && key.execution_mode != "eager" {
            return Err(AicError::PerfDatabase(
                "Ascend extrapolation cannot substitute a prefill capture bucket".into(),
            ));
        }
        estimate_attention_grid(
            partitions,
            key.phase == "prefill",
            active,
            key.partition(active, padded),
            sequence,
            prefix,
            &|batch, tokens, past| {
                attention_work(
                    spec,
                    key.operation == "lightning_indexer",
                    key.phase == "prefill",
                    key.local_heads,
                    batch,
                    tokens,
                    past,
                    1,
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
                "Ascend attention rejects filtered primary sources".into(),
            ));
        }
        let path = primary.map(|source| source.source.0);
        if let Some(path) = &path {
            let family_path = data_root
                .parent()
                .and_then(Path::parent)
                .and_then(|system| {
                    Some(
                        system
                            .join("attention")
                            .join(data_root.parent()?.file_name()?)
                            .join(data_root.file_name()?)
                            .join(BASENAME),
                    )
                });
            if path != &data_root.join(BASENAME) && family_path.as_ref() != Some(path) {
                return Err(AicError::InvalidPerfData(
                    "Ascend attention requires the exact requested system/backend/version".into(),
                ));
            }
        }
        Ok(Self::new(path))
    }

    fn grids(&self) -> Result<&Grids, AicError> {
        self.grids
            .get_or_init(|| {
                let Some(path) = self.path.as_ref().filter(|path| path.is_file()) else {
                    return Err((false, "Ascend attention primary data is missing".into()));
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
            })
    }

    pub fn ensure_key(&self, key: &AscendAttentionKey) -> Result<(), AicError> {
        key.validate()?;
        if !self.grids()?.contains_key(key) {
            return Err(AicError::PerfDatabase(format!(
                "unmeasured Ascend attention identity: {key:?}"
            )));
        }
        Ok(())
    }

    pub fn query(
        &self,
        key: &AscendAttentionKey,
        active: u32,
        padded: u32,
        sequence: u32,
        prefix: u32,
    ) -> Result<f64, AicError> {
        key.validate()?;
        if active == 0
            || sequence == 0
            || padded < active
            || (key.phase == "prefill" && padded != active)
            || (key.phase == "decode" && prefix != 0)
        {
            return Err(AicError::InvalidEngineConfig(
                "unmeasured Ascend attention batch/sequence layout".into(),
            ));
        }
        let partitions = self.grids()?.get(key).ok_or_else(|| {
            AicError::PerfDatabase(format!("unmeasured Ascend attention identity: {key:?}"))
        })?;
        if key.phase == "prefill" && key.execution_mode == "eager" {
            return query_prefill_batch(partitions, active, prefix, sequence);
        }
        let plane = partitions
            .get(&key.partition(active, padded))
            .ok_or_else(|| {
                AicError::PerfDatabase(format!(
                    "unmeasured Ascend attention tuple: {key:?}, padded_batch={padded}"
                ))
            })?;
        if key.phase == "prefill" {
            plane.query(prefix, sequence)
        } else {
            plane.query(active, sequence)
        }
    }
}

fn load(path: &Path) -> Result<Grids, AicError> {
    let reader = PerfReader::open(path)?;
    let operation = reader.col("operation")?;
    let phase = reader.col("phase")?;
    let heads = reader.col("local_heads")?;
    let mode = reader.col("execution_mode")?;
    let tokens = reader.col("tokens")?;
    let active = reader.col("active_batch")?;
    let padded = reader.col("padded_batch")?;
    let sequence = reader.col("sequence_length")?;
    let prefix = reader.col("prefix")?;
    let pool = reader.col("pool_pages")?;
    let table_width = reader.col("block_table_width")?;
    let latency = reader.col("latency_ms")?;
    let variation = reader.col("cv")?;
    let kernel = reader.col("kernel_source")?;
    let verification = reader.col("execution_verification")?;
    let checks = reader.col("numerical_checks")?;
    let samples = reader.col("sample_count")?;
    let dtype = reader.col("input_dtype")?;
    let provenance = reader.col("runtime_identity_sha256")?;
    let fixed = [
        ("page_size", 128),
        ("sparse_topk", 2048),
        ("kv_lora_rank", 512),
        ("rope_dim", 64),
        ("index_heads", 32),
        ("index_dim", 128),
        ("cache_format", 30),
    ]
    .into_iter()
    .map(|(name, value)| Ok((reader.col(name)?, value)))
    .collect::<Result<Vec<_>, AicError>>()?;
    type RawGrids = BTreeMap<AscendAttentionKey, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>>;
    let mut raw: RawGrids = BTreeMap::new();
    let mut identities: BTreeMap<AscendAttentionKey, String> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = AscendAttentionKey {
            operation: row.str_owned(operation)?,
            phase: row.str_owned(phase)?,
            local_heads: row.u32(heads)?,
            execution_mode: row.str_owned(mode)?,
        };
        key.validate()?;
        let count = row.u32(tokens)?;
        let active_batch = row.u32(active)?;
        let padded_batch = row.u32(padded)?;
        let length = row.u32(sequence)?;
        let prefix_tokens = row.u32(prefix)?;
        let measured = row.f64(latency)?;
        let cv = row.f64(variation)?;
        let runtime = row.str(provenance)?;
        let pages = length.div_ceil(128);
        let pool_pages = active_batch
            .checked_mul(pages)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| AicError::InvalidPerfData("Ascend attention pool overflow".into()))?
            .max(417);
        let (outer, inner, expected_width, shape_ok) = if key.phase == "prefill" {
            (
                prefix_tokens,
                count.checked_div(active_batch).unwrap_or(0),
                pages,
                active_batch > 0
                    && padded_batch == active_batch
                    && count.is_multiple_of(active_batch)
                    && (count / active_batch).checked_add(prefix_tokens) == Some(length),
            )
        } else {
            (
                active_batch,
                length,
                1407,
                count == padded_batch && prefix_tokens == 0,
            )
        };
        let verification_mode = if key.execution_mode == "graph" {
            "captured_graph_replay"
        } else {
            "eager_invocation"
        };
        let fixed_valid = fixed.iter().try_fold(true, |valid, (column, expected)| {
            Ok::<bool, AicError>(valid && row.u32(*column)? == *expected)
        })?;
        if !shape_ok
            || !fixed_valid
            || count == 0
            || active_batch == 0
            || length == 0
            || padded_batch < active_batch
            || !measured.is_finite()
            || measured <= 0.0
            || !cv.is_finite()
            || !(0.0..=0.1).contains(&cv)
            || row.u32(pool)? != pool_pages
            || row.u32(table_width)? != expected_width
            || row.u32(checks)? == 0
            || row.u32(samples)? < 3
            || row.str(dtype)? != "bfloat16"
            || row.str(kernel)? != key.kernel_source()
            || row.str(verification)? != verification_mode
            || runtime.len() != 64
            || !runtime.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(AicError::InvalidPerfData(format!(
                "unqualified Ascend attention row: {key:?}"
            )));
        }
        let identity = identities
            .entry(key.clone())
            .or_insert_with(|| runtime.to_owned());
        if identity != runtime {
            return Err(AicError::InvalidPerfData(
                "Ascend attention grid mixes runtime identities".into(),
            ));
        }
        let partition = key.partition(active_batch, padded_batch);
        if raw
            .entry(key)
            .or_default()
            .entry(partition)
            .or_default()
            .entry(outer)
            .or_default()
            .insert(inner, measured)
            .is_some()
        {
            return Err(AicError::InvalidPerfData(
                "duplicate Ascend attention physical key".into(),
            ));
        }
    }
    if raw.is_empty() {
        return Err(AicError::InvalidPerfData(
            "empty Ascend attention table".into(),
        ));
    }
    Ok(raw
        .into_iter()
        .map(|(key, partitions)| {
            (
                key,
                partitions
                    .into_iter()
                    .map(|(partition, planes)| {
                        (
                            partition,
                            Plane(
                                planes
                                    .into_iter()
                                    .map(|(outer, points)| (outer, BoundedCurve::new(points)))
                                    .collect(),
                            ),
                        )
                    })
                    .collect(),
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::ascend_attention::AscendAttentionOp;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
    use crate::perf_database::{PerfDbSources, PerfSource};

    fn key() -> AscendAttentionKey {
        AscendAttentionKey {
            operation: "sparse_attention".into(),
            phase: "decode".into(),
            local_heads: 8,
            execution_mode: "graph".into(),
        }
    }

    fn columns() -> Vec<Col> {
        vec![
            Col::Str("operation", vec!["sparse_attention"; 4]),
            Col::Str("phase", vec!["decode"; 4]),
            Col::I64("local_heads", vec![8; 4]),
            Col::Str("execution_mode", vec!["graph"; 4]),
            Col::I64("tokens", vec![8; 4]),
            Col::I64("active_batch", vec![1, 1, 5, 5]),
            Col::I64("padded_batch", vec![8; 4]),
            Col::I64("sequence_length", vec![128, 1024, 128, 1024]),
            Col::I64("prefix", vec![0; 4]),
            Col::I64("pool_pages", vec![417; 4]),
            Col::I64("block_table_width", vec![1407; 4]),
            Col::F64("latency_ms", vec![1.0, 3.0, 5.0, 7.0]),
            Col::F64("cv", vec![0.01; 4]),
            Col::Str(
                "kernel_source",
                vec!["torch_npu.npu_sparse_flash_attention"; 4],
            ),
            Col::Str("execution_verification", vec!["captured_graph_replay"; 4]),
            Col::I64("numerical_checks", vec![24; 4]),
            Col::I64("sample_count", vec![7; 4]),
            Col::Str("input_dtype", vec!["bfloat16"; 4]),
            Col::Str(
                "runtime_identity_sha256",
                vec!["1111111111111111111111111111111111111111111111111111111111111111"; 4],
            ),
            Col::I64("page_size", vec![128; 4]),
            Col::I64("sparse_topk", vec![2048; 4]),
            Col::I64("kv_lora_rank", vec![512; 4]),
            Col::I64("rope_dim", vec![64; 4]),
            Col::I64("index_heads", vec![32; 4]),
            Col::I64("index_dim", vec![128; 4]),
            Col::I64("cache_format", vec![30; 4]),
        ]
    }

    #[test]
    fn interpolate_only_measured_active_and_kv_brackets_within_capture_bucket() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        write_parquet(&path, &columns());
        let table = AscendAttentionTable::new(Some(path));
        assert_eq!(table.query(&key(), 1, 8, 128, 0).unwrap(), 1.0);
        assert_eq!(table.query(&key(), 3, 8, 576, 0).unwrap(), 4.0);
        for (active, padded, sequence) in [(6, 8, 576), (3, 16, 576), (3, 8, 1025), (3, 8, 127)] {
            assert!(
                table
                    .query(&key(), active, padded, sequence, 0)
                    .unwrap_err()
                    .is_missing_perf_data()
            );
        }
        let mut other = key();
        other.execution_mode = "eager".into();
        assert!(
            table
                .query(&other, 3, 3, 576, 0)
                .unwrap_err()
                .is_missing_perf_data()
        );
        other = key();
        other.local_heads = 16;
        assert!(
            table
                .query(&other, 3, 8, 576, 0)
                .unwrap_err()
                .is_missing_perf_data()
        );
    }

    #[test]
    fn eager_padding_uses_its_own_measured_plane() {
        let root = tempfile::tempdir().unwrap();
        let mut eager = key();
        eager.execution_mode = "eager".into();
        for padded in [true, false] {
            let path = root.path().join(if padded {
                "padded.parquet"
            } else {
                "unpadded.parquet"
            });
            let mut data = columns();
            data[3] = Col::Str("execution_mode", vec!["eager"; 4]);
            data[14] = Col::Str("execution_verification", vec!["eager_invocation"; 4]);
            if !padded {
                data[4] = Col::I64("tokens", vec![1, 1, 5, 5]);
                data[6] = Col::I64("padded_batch", vec![1, 1, 5, 5]);
            }
            write_parquet(&path, &data);
            let table = AscendAttentionTable::new(Some(path));
            let selected = if padded { 8 } else { 3 };
            let other = if padded { 3 } else { 8 };
            assert_eq!(table.query(&eager, 3, selected, 576, 0).unwrap(), 4.0);
            assert!(
                table
                    .query(&eager, 3, other, 576, 0)
                    .unwrap_err()
                    .is_missing_perf_data()
            );
            assert!(
                table
                    .query(&eager, 3, 16, 576, 0)
                    .unwrap_err()
                    .is_missing_perf_data()
            );
        }
    }

    #[test]
    fn prefill_uses_tokens_and_prefix_without_decode_padding() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        let mut data = columns();
        data[1] = Col::Str("phase", vec!["prefill"; 4]);
        data[4] = Col::I64("tokens", vec![128, 1024, 128, 1024]);
        data[5] = Col::I64("active_batch", vec![1; 4]);
        data[6] = Col::I64("padded_batch", vec![1; 4]);
        data[7] = Col::I64("sequence_length", vec![128, 1024, 8320, 9216]);
        data[8] = Col::I64("prefix", vec![0, 0, 8192, 8192]);
        data[10] = Col::I64("block_table_width", vec![1, 8, 65, 72]);
        write_parquet(&path, &data);
        let table = AscendAttentionTable::new(Some(path));
        let mut prefill = key();
        prefill.phase = "prefill".into();
        assert_eq!(table.query(&prefill, 1, 1, 576, 4096).unwrap(), 4.0);
        assert!(
            table
                .query(&prefill, 2, 2, 576, 0)
                .unwrap_err()
                .is_missing_perf_data()
        );
        assert!(matches!(
            table.query(&prefill, 2, 3, 576, 0),
            Err(AicError::InvalidEngineConfig(_))
        ));
        data[4] = Col::I64("tokens", vec![256, 2048, 256, 2048]);
        data[5] = Col::I64("active_batch", vec![2; 4]);
        data[6] = Col::I64("padded_batch", vec![2; 4]);
        let path = root.path().join("two-requests.parquet");
        write_parquet(&path, &data);
        let table = AscendAttentionTable::new(Some(path));
        assert_eq!(table.query(&prefill, 2, 2, 576, 4096).unwrap(), 4.0);
        assert!(
            table
                .query(&prefill, 1, 1, 576, 4096)
                .unwrap_err()
                .is_missing_perf_data()
        );
    }

    #[test]
    fn eager_prefill_batch_three_is_the_midpoint_of_batch_two_and_four() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        let mut data = columns();
        data[1] = Col::Str("phase", vec!["prefill"; 4]);
        data[3] = Col::Str("execution_mode", vec!["eager"; 4]);
        data[4] = Col::I64("tokens", vec![256, 2048, 512, 4096]);
        data[5] = Col::I64("active_batch", vec![2, 2, 4, 4]);
        data[6] = Col::I64("padded_batch", vec![2, 2, 4, 4]);
        data[7] = Col::I64("sequence_length", vec![128, 1024, 128, 1024]);
        data[10] = Col::I64("block_table_width", vec![1, 8, 1, 8]);
        data[14] = Col::Str("execution_verification", vec!["eager_invocation"; 4]);
        write_parquet(&path, &data);
        let table = AscendAttentionTable::new(Some(path));
        let mut prefill = key();
        prefill.phase = "prefill".into();
        prefill.execution_mode = "eager".into();
        assert_eq!(table.query(&prefill, 2, 2, 128, 0).unwrap(), 1.0);
        assert_eq!(table.query(&prefill, 3, 3, 128, 0).unwrap(), 3.0);
        assert_eq!(table.query(&prefill, 3, 3, 576, 0).unwrap(), 4.0);
        for (batch, sequence, prefix) in [(1, 128, 0), (5, 128, 0), (3, 1025, 0), (3, 128, 1)] {
            assert!(
                table
                    .query(&prefill, batch, batch, sequence, prefix)
                    .unwrap_err()
                    .is_missing_perf_data()
            );
        }
        prefill.execution_mode = "graph".into();
        assert!(
            table
                .query(&prefill, 3, 3, 128, 0)
                .unwrap_err()
                .is_missing_perf_data()
        );
    }

    #[test]
    fn malformed_geometry_checks_and_duplicate_rows_fail_closed() {
        for (index, replacement) in [
            (5, Col::I64("active_batch", vec![9; 4])),
            (7, Col::I64("sequence_length", vec![128; 4])),
            (9, Col::I64("pool_pages", vec![416; 4])),
            (
                14,
                Col::Str("execution_verification", vec!["eager_invocation"; 4]),
            ),
            (15, Col::I64("numerical_checks", vec![0; 4])),
            (25, Col::I64("cache_format", vec![2; 4])),
        ] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join(BASENAME);
            let mut data = columns();
            data[index] = replacement;
            write_parquet(&path, &data);
            assert!(matches!(
                AscendAttentionTable::new(Some(path)).ensure_key(&key()),
                Err(AicError::InvalidPerfData(_))
            ));
        }
        assert!(
            AscendAttentionTable::new(None)
                .ensure_key(&key())
                .unwrap_err()
                .is_missing_perf_data()
        );
    }

    #[test]
    fn graph_padding_is_explicit_and_validated() {
        let mut op = AscendAttentionOp {
            name: "attention".into(),
            scale_factor: 1.0,
            key: key(),
            graph_buckets: vec![8, 16, 24, 32],
            eager_token_multiple: 1,
        };
        assert_eq!(op.padded_batch(5).unwrap(), 8);
        assert_eq!(op.padded_batch(9).unwrap(), 16);
        assert!(op.padded_batch(33).unwrap_err().is_missing_perf_data());
        op.graph_buckets = vec![16, 8];
        assert!(op.validate().is_err());
        op.key.execution_mode = "eager".into();
        op.graph_buckets.clear();
        assert_eq!(op.padded_batch(5).unwrap(), 5);
        op.eager_token_multiple = 4;
        assert_eq!(op.padded_batch(5).unwrap(), 8);
        assert_eq!(op.padded_batch(9).unwrap(), 12);
        assert_eq!(op.padded_batch(0).unwrap(), 0);
        assert!(op.padded_batch(u32::MAX).is_err());
        op.eager_token_multiple = 0;
        assert!(op.validate().is_err());
    }

    #[test]
    fn primary_identity_cannot_borrow_another_runtime() {
        let root = tempfile::tempdir().unwrap();
        let requested = root.path().join("data/ascend910c/sglang/0.5.22");
        for relative in [
            "data/ascend910c/attention/sglang/0.5.22",
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
                AscendAttentionTable::with_sources(&requested, &resolver)
                    .unwrap()
                    .query(&key(), 3, 8, 576, 0)
                    .unwrap(),
                4.0
            );
        }
        for relative in [
            "data/other/attention/sglang/0.5.22",
            "data/ascend910c/attention/vllm/0.5.22",
            "data/ascend910c/attention/sglang/0.5.21",
        ] {
            let resolver = SourceResolver::fixed(PerfDbSources::from([(
                BASENAME.into(),
                vec![PerfSource(root.path().join(relative).join(BASENAME), None)],
            )]));
            assert!(matches!(
                AscendAttentionTable::with_sources(&requested, &resolver),
                Err(AicError::InvalidPerfData(_))
            ));
        }
        let resolver = SourceResolver::fixed(PerfDbSources::from([(BASENAME.into(), Vec::new())]));
        assert!(
            AscendAttentionTable::with_sources(&requested, &resolver)
                .unwrap()
                .ensure_key(&key())
                .unwrap_err()
                .is_missing_perf_data()
        );
    }
}
#[test]
fn anchored_ragged_grid_scales_only_the_missing_boundary() {
    let partitions = BTreeMap::from([
        (
            2,
            Plane::from_raw(BTreeMap::from([(
                0,
                BTreeMap::from([(1, 1.0), (4096, 8.0)]),
            )])),
        ),
        (
            4,
            Plane::from_raw(BTreeMap::from([(
                0,
                BTreeMap::from([(1, 2.0), (2048, 8.0)]),
            )])),
        ),
    ]);
    let work = |batch, tokens, _| Ok(f64::from(batch) * f64::from(tokens));
    let estimate = estimate_attention_grid(&partitions, true, 3, 3, 4096, 0, &work).unwrap();
    assert_eq!(estimate.latency_ms, 11.0);
    assert!(estimate.extrapolated);
    assert!(query_prefill_batch(&partitions, 3, 0, 4096).is_err());
    assert!(estimate_attention_grid(&partitions, false, 1, 8, 4096, 0, &work).is_err());
    let measured = estimate_attention_grid(&partitions, true, 2, 2, 4096, 0, &work).unwrap();
    assert_eq!(measured.latency_ms, 8.0);
    assert!(!measured.extrapolated);
}
