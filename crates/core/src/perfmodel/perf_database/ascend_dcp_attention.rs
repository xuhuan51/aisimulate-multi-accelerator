use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::SourceResolver;
use super::ascend_attention::{Plane, estimate_attention_grid, query_prefill_batch};
use super::ascend_shape::{AscendShapePolicy, ShapeEstimate, attention_work};
use super::parquet_loader::PerfReader;
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;

pub const BASENAME: &str = "ascend_dcp_attention_perf.parquet";
pub const CONTRACT: &str = "sglang_npu_dcp_storage_attention_chain/v1";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendDcpAttentionKey {
    pub phase: String,
    pub local_heads: u32,
    pub dcp_size: u32,
    pub remap_topk: bool,
    pub execution_mode: String,
}

impl AscendDcpAttentionKey {
    pub fn validate(&self) -> Result<(), AicError> {
        if !matches!(self.local_heads, 4 | 8 | 16 | 32)
            || !matches!(self.dcp_size, 2 | 4 | 8 | 16)
            || self.local_heads * self.dcp_size > 64
            || !matches!(
                (self.phase.as_str(), self.execution_mode.as_str()),
                ("prefill", "eager") | ("decode", "graph")
            )
        {
            return Err(AicError::InvalidEngineConfig(format!(
                "invalid Ascend DCP identity: {self:?}"
            )));
        }
        Ok(())
    }
}

type Grids = BTreeMap<AscendDcpAttentionKey, BTreeMap<u32, Plane>>;

pub struct AscendDcpAttentionTable {
    path: Option<PathBuf>,
    grids: OnceLock<Result<Grids, (bool, String)>>,
}

impl AscendDcpAttentionTable {
    #[allow(clippy::too_many_arguments)]
    pub fn estimate(
        &self,
        key: &AscendDcpAttentionKey,
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
            AicError::PerfDatabase(format!("unmeasured Ascend DCP identity: {key:?}"))
        })?;
        estimate_attention_grid(
            partitions,
            key.phase == "prefill",
            active,
            padded,
            sequence,
            prefix,
            false,
            &|batch, tokens, past| {
                attention_work(
                    spec,
                    false,
                    key.phase == "prefill",
                    key.local_heads,
                    batch,
                    tokens,
                    past,
                    key.dcp_size,
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
                "Ascend DCP rejects filtered primary sources".into(),
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
                            .join("attention")
                            .join(data_root.parent()?.file_name()?)
                            .join(data_root.file_name()?)
                            .join(BASENAME),
                    )
                });
            if path != &data_root.join(BASENAME) && family.as_ref() != Some(path) {
                return Err(AicError::InvalidPerfData(
                    "Ascend DCP requires the exact system/backend/version".into(),
                ));
            }
        }
        Ok(Self::new(path))
    }

    fn grids(&self) -> Result<&Grids, AicError> {
        self.grids
            .get_or_init(|| {
                let Some(path) = self.path.as_ref().filter(|path| path.is_file()) else {
                    return Err((false, "Ascend DCP primary data is missing".into()));
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

    pub fn ensure_key(&self, key: &AscendDcpAttentionKey) -> Result<(), AicError> {
        key.validate()?;
        if self.grids()?.contains_key(key) {
            return Ok(());
        }
        Err(AicError::PerfDatabase(format!(
            "unmeasured Ascend DCP tuple: {key:?}"
        )))
    }

    pub fn query(
        &self,
        key: &AscendDcpAttentionKey,
        active: u32,
        padded: u32,
        sequence: u32,
        prefix: u32,
    ) -> Result<f64, AicError> {
        key.validate()?;
        if active == 0
            || sequence == 0
            || padded < active
            || (key.phase == "prefill" && active != padded)
            || (key.phase == "decode" && prefix != 0)
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid DCP batch or sequence geometry".into(),
            ));
        }
        let partitions = self.grids()?.get(key).ok_or_else(|| {
            AicError::PerfDatabase(format!("unmeasured Ascend DCP tuple: {key:?}"))
        })?;
        if key.phase == "prefill" {
            return query_prefill_batch(partitions, active, prefix, sequence, false);
        }
        let plane = partitions.get(&padded).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "unmeasured Ascend DCP partition: {key:?}, padded={padded}"
            ))
        })?;
        plane.query(active, sequence)
    }
}

fn load(path: &Path) -> Result<Grids, AicError> {
    let reader = PerfReader::open(path)?;
    let phase = reader.col("phase")?;
    let heads = reader.col("local_heads")?;
    let degree = reader.col("dcp_size")?;
    let remap = reader.col("remap_topk")?;
    let mode = reader.col("execution_mode")?;
    let tokens = reader.col("tokens")?;
    let active = reader.col("active_batch")?;
    let padded = reader.col("padded_batch")?;
    let query = reader.col("query_tokens_per_request")?;
    let sequence = reader.col("sequence_length")?;
    let prefix = reader.col("prefix")?;
    let pool = reader.col("pool_pages")?;
    let table = reader.col("block_table_width")?;
    let contract = reader.col("kernel_contract")?;
    let query_dtype = reader.col("query_dtype")?;
    let output_dtype = reader.col("output_dtype")?;
    let merge_dtype = reader.col("merge_dtype")?;
    let local_index = reader.col("local_cache_index_dtype")?;
    let global_index = reader.col("indexer_cache_index_dtype")?;
    let ranks = reader.col("ranks_per_node")?;
    let layout = reader.col("rank_layout")?;
    let verification = reader.col("execution_verification")?;
    let checks = reader.col("numerical_checks")?;
    let samples = reader.col("sample_count")?;
    let latency = reader.col("latency_ms")?;
    let variation = reader.col("cv")?;
    let provenance = reader.col("runtime_identity_sha256")?;
    let fixed = [
        ("page_size", 128),
        ("kv_interleave_size", 128),
        ("sparse_topk", 2048),
        ("kv_lora_rank", 512),
        ("rope_dim", 64),
        ("cache_format", 30),
    ]
    .into_iter()
    .map(|(name, expected)| Ok((reader.col(name)?, expected)))
    .collect::<Result<Vec<_>, AicError>>()?;
    let mut raw: BTreeMap<AscendDcpAttentionKey, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>> =
        BTreeMap::new();
    let mut identities: BTreeMap<AscendDcpAttentionKey, String> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = AscendDcpAttentionKey {
            phase: row.str_owned(phase)?,
            local_heads: row.u32(heads)?,
            dcp_size: row.u32(degree)?,
            remap_topk: row.bool_strict(remap)?,
            execution_mode: row.str_owned(mode)?,
        };
        key.validate()?;
        let active_batch = row.u32(active)?;
        let padded_batch = row.u32(padded)?;
        let query_tokens = row.u32(query)?;
        let length = row.u32(sequence)?;
        let past = row.u32(prefix)?;
        let measured = row.f64(latency)?;
        let cv = row.f64(variation)?;
        let runtime = row.str(provenance)?;
        let context = key.phase == "prefill";
        let pages = length.div_ceil(128 * key.dcp_size);
        let expected_pool = active_batch
            .checked_mul(pages)
            .and_then(|count| count.checked_add(1))
            .map(|count| count.max(417u32.div_ceil(key.dcp_size)))
            .ok_or_else(|| AicError::InvalidPerfData("DCP pool geometry overflows".into()))?;
        let shape_valid = if context {
            active_batch == padded_batch
                && query_tokens > 0
                && length.checked_sub(past) == Some(query_tokens)
                && active_batch.checked_mul(query_tokens) == Some(row.u32(tokens)?)
        } else {
            query_tokens == 1
                && past == 0
                && row.u32(tokens)? == padded_batch
                && matches!(padded_batch, 8 | 16 | 24 | 32)
                && padded_batch % (64 / key.local_heads) == 0
        };
        if active_batch == 0
            || active_batch > padded_batch
            || length == 0
            || !shape_valid
            || row.u32(pool)? != expected_pool
            || row.u32(table)?
                != if context {
                    pages
                } else {
                    180000u32.div_ceil(128 * key.dcp_size)
                }
            || fixed
                .iter()
                .any(|(column, expected)| row.u32(*column).ok() != Some(*expected))
            || row.str(contract)? != CONTRACT
            || row.str(query_dtype)? != "bfloat16"
            || row.str(output_dtype)? != "bfloat16"
            || row.str(merge_dtype)? != "float32"
            || row.u32(ranks)? != key.dcp_size
            || row.str(layout)? != "contiguous"
            || row.str(local_index)? != if context { "int64" } else { "int32" }
            || row.str(global_index)? != if context { "int32" } else { "int64" }
            || row.str(verification)?
                != if context {
                    "eager_invocation"
                } else {
                    "captured_graph_replay"
                }
            || row.u32(checks)? == 0
            || row.u32(samples)? < 3
            || !measured.is_finite()
            || measured <= 0.0
            || !cv.is_finite()
            || !(0.0..=0.1).contains(&cv)
            || runtime.len() != 64
            || !runtime.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(AicError::InvalidPerfData(format!(
                "unqualified Ascend DCP row: {key:?}"
            )));
        }
        let identity = identities
            .entry(key.clone())
            .or_insert_with(|| runtime.to_owned());
        if identity != runtime {
            return Err(AicError::InvalidPerfData(
                "DCP query grid mixes collection identities".into(),
            ));
        }
        let outer = if context { past } else { active_batch };
        let inner = if context { query_tokens } else { length };
        if raw
            .entry(key)
            .or_default()
            .entry(padded_batch)
            .or_default()
            .entry(outer)
            .or_default()
            .insert(inner, measured)
            .is_some()
        {
            return Err(AicError::InvalidPerfData(
                "duplicate Ascend DCP physical point".into(),
            ));
        }
    }
    if raw.is_empty() {
        return Err(AicError::InvalidPerfData("empty Ascend DCP table".into()));
    }
    Ok(raw
        .into_iter()
        .map(|(key, partitions)| {
            (
                key,
                partitions
                    .into_iter()
                    .map(|(partition, curves)| (partition, Plane::from_raw(curves)))
                    .collect(),
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};

    fn key() -> AscendDcpAttentionKey {
        AscendDcpAttentionKey {
            phase: "decode".into(),
            local_heads: 8,
            dcp_size: 2,
            remap_topk: true,
            execution_mode: "graph".into(),
        }
    }

    fn columns() -> Vec<Col> {
        vec![
            Col::Str("phase", vec!["decode"; 4]),
            Col::I64("local_heads", vec![8; 4]),
            Col::I64("dcp_size", vec![2; 4]),
            Col::Bool("remap_topk", vec![true; 4]),
            Col::Str("execution_mode", vec!["graph"; 4]),
            Col::I64("tokens", vec![8; 4]),
            Col::I64("active_batch", vec![1, 1, 4, 4]),
            Col::I64("padded_batch", vec![8; 4]),
            Col::I64("query_tokens_per_request", vec![1; 4]),
            Col::I64("sequence_length", vec![8192, 9216, 8192, 9216]),
            Col::I64("prefix", vec![0; 4]),
            Col::I64("pool_pages", vec![209; 4]),
            Col::I64("block_table_width", vec![704; 4]),
            Col::Str("kernel_contract", vec![CONTRACT; 4]),
            Col::Str("query_dtype", vec!["bfloat16"; 4]),
            Col::Str("output_dtype", vec!["bfloat16"; 4]),
            Col::Str("merge_dtype", vec!["float32"; 4]),
            Col::Str("local_cache_index_dtype", vec!["int32"; 4]),
            Col::Str("indexer_cache_index_dtype", vec!["int64"; 4]),
            Col::I64("ranks_per_node", vec![2; 4]),
            Col::Str("rank_layout", vec!["contiguous"; 4]),
            Col::Str("execution_verification", vec!["captured_graph_replay"; 4]),
            Col::I64("numerical_checks", vec![1024; 4]),
            Col::I64("sample_count", vec![7; 4]),
            Col::F64("latency_ms", vec![1.0, 3.0, 4.0, 6.0]),
            Col::F64("cv", vec![0.01; 4]),
            Col::Str(
                "runtime_identity_sha256",
                vec!["1111111111111111111111111111111111111111111111111111111111111111"; 4],
            ),
            Col::I64("page_size", vec![128; 4]),
            Col::I64("kv_interleave_size", vec![128; 4]),
            Col::I64("sparse_topk", vec![2048; 4]),
            Col::I64("kv_lora_rank", vec![512; 4]),
            Col::I64("rope_dim", vec![64; 4]),
            Col::I64("cache_format", vec![30; 4]),
        ]
    }

    #[test]
    fn interpolates_sequence_midpoint_then_one_third_of_active_batch_interval() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        write_parquet(&path, &columns());
        let table = AscendDcpAttentionTable::new(Some(path));
        assert_eq!(table.query(&key(), 1, 8, 8192, 0).unwrap(), 1.0);
        assert_eq!(table.query(&key(), 2, 8, 8704, 0).unwrap(), 3.0);
        assert!(
            table
                .query(&key(), 2, 8, 10000, 0)
                .unwrap_err()
                .is_missing_perf_data()
        );
        let mut other = key();
        other.remap_topk = false;
        assert!(
            table
                .query(&other, 2, 8, 8704, 0)
                .unwrap_err()
                .is_missing_perf_data()
        );
        other = key();
        other.dcp_size = 4;
        assert!(
            table
                .query(&other, 2, 8, 8704, 0)
                .unwrap_err()
                .is_missing_perf_data()
        );
    }

    #[test]
    fn eager_prefill_interpolates_between_two_and_four_requests_without_extrapolation() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        let mut data = columns();
        data[0] = Col::Str("phase", vec!["prefill"; 4]);
        data[4] = Col::Str("execution_mode", vec!["eager"; 4]);
        data[5] = Col::I64("tokens", vec![256, 2048, 512, 4096]);
        data[6] = Col::I64("active_batch", vec![2, 2, 4, 4]);
        data[7] = Col::I64("padded_batch", vec![2, 2, 4, 4]);
        data[8] = Col::I64("query_tokens_per_request", vec![128, 1024, 128, 1024]);
        data[9] = Col::I64("sequence_length", vec![128, 1024, 128, 1024]);
        data[12] = Col::I64("block_table_width", vec![1, 4, 1, 4]);
        data[17] = Col::Str("local_cache_index_dtype", vec!["int64"; 4]);
        data[18] = Col::Str("indexer_cache_index_dtype", vec!["int32"; 4]);
        data[21] = Col::Str("execution_verification", vec!["eager_invocation"; 4]);
        write_parquet(&path, &data);
        let table = AscendDcpAttentionTable::new(Some(path));
        let mut prefill = key();
        prefill.phase = "prefill".into();
        prefill.execution_mode = "eager".into();
        assert_eq!(table.query(&prefill, 2, 2, 128, 0).unwrap(), 1.0);
        assert_eq!(table.query(&prefill, 3, 3, 128, 0).unwrap(), 2.5);
        assert_eq!(table.query(&prefill, 3, 3, 576, 0).unwrap(), 3.5);
        for (batch, sequence, prefix) in [(1, 128, 0), (5, 128, 0), (3, 1025, 0), (3, 128, 1)] {
            assert!(
                table
                    .query(&prefill, batch, batch, sequence, prefix)
                    .unwrap_err()
                    .is_missing_perf_data()
            );
        }
    }

    #[test]
    fn rejects_token_interleaving_wrong_cache_dtype_and_incomplete_boundary() {
        let root = tempfile::tempdir().unwrap();
        for replacement in [
            Col::I64("kv_interleave_size", vec![1; 4]),
            Col::Str("indexer_cache_index_dtype", vec!["int32"; 4]),
            Col::Str("kernel_contract", vec!["sglang_npu_dcp_sparse_chain/v1"; 4]),
            Col::F64("cv", vec![0.11; 4]),
        ] {
            let name = match &replacement {
                Col::I64(name, _) | Col::Str(name, _) | Col::F64(name, _) => *name,
                _ => unreachable!(),
            };
            let mut fixture = columns();
            fixture.retain(|column| match column {
                Col::I64(field, _) | Col::Str(field, _) | Col::F64(field, _) => *field != name,
                _ => true,
            });
            fixture.push(replacement);
            let path = root.path().join(BASENAME);
            write_parquet(&path, &fixture);
            assert!(matches!(
                AscendDcpAttentionTable::new(Some(path)).ensure_key(&key()),
                Err(AicError::InvalidPerfData(_))
            ));
        }
    }
}
