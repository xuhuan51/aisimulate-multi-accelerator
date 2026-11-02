use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::SourceResolver;
use super::ascend_attention::Plane;
use super::parquet_loader::PerfReader;
use crate::common::error::AicError;

pub const BASENAME: &str = "ascend_layer_perf.parquet";
pub const CONTRACT: &str = "glm52_eager_decoder_layer/v1";
pub const SOURCE: &str = "sglang.srt.models.deepseek_v2.DeepseekV2DecoderLayer.forward";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendLayerKey {
    pub attention_tp: u32,
    pub attention_dp: u32,
    pub moe_ep: u32,
    pub layer: u32,
    pub quantization_sha256: String,
}

impl AscendLayerKey {
    pub fn validate(&self) -> Result<(), AicError> {
        if self.attention_tp != 4
            || self.attention_dp != 4
            || self.moe_ep != 16
            || !matches!(self.layer, 0..=6 | 77)
            || self.quantization_sha256.len() != 64
            || !self
                .quantization_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(AicError::InvalidEngineConfig(format!(
                "unqualified Ascend eager decoder-layer identity: {self:?}"
            )));
        }
        Ok(())
    }
}

type LayerPlanes = BTreeMap<AscendLayerKey, Plane>;

pub struct AscendLayerTable {
    path: Option<PathBuf>,
    planes: OnceLock<Result<LayerPlanes, (bool, String)>>,
}

impl AscendLayerTable {
    pub fn new(path: Option<PathBuf>) -> Self {
        Self {
            path,
            planes: OnceLock::new(),
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
                "Ascend layer rejects filtered sources".into(),
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
                            .join("runtime_layer")
                            .join(data_root.parent()?.file_name()?)
                            .join(data_root.file_name()?)
                            .join(BASENAME),
                    )
                });
            if path != &data_root.join(BASENAME) && family.as_ref() != Some(path) {
                return Err(AicError::InvalidPerfData(
                    "Ascend layer requires the exact system/backend/version".into(),
                ));
            }
        }
        Ok(Self::new(path))
    }

    fn plane(&self, key: &AscendLayerKey) -> Result<&Plane, AicError> {
        key.validate()?;
        let planes = self
            .planes
            .get_or_init(|| {
                let Some(path) = self.path.as_ref().filter(|path| path.is_file()) else {
                    return Err((
                        false,
                        "Ascend runtime-layer measurements are missing".into(),
                    ));
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
        planes.get(key).ok_or_else(|| {
            AicError::PerfDatabase(format!("unmeasured Ascend runtime layer: {key:?}"))
        })
    }

    pub fn ensure_key(&self, key: &AscendLayerKey) -> Result<(), AicError> {
        self.plane(key).map(|_| ())
    }

    pub fn query(
        &self,
        key: &AscendLayerKey,
        requests: u32,
        kv_tokens: u32,
    ) -> Result<f64, AicError> {
        self.plane(key)?
            .query(requests, kv_tokens)
            .map_err(|error| {
                AicError::PerfDatabase(format!(
                    "Ascend runtime layer {key:?} requires measured request/KV brackets: {error}"
                ))
            })
    }
}

fn load(path: &Path) -> Result<LayerPlanes, AicError> {
    let reader = PerfReader::open(path)?;
    let layer = reader.col("layer")?;
    let attention_tp = reader.col("attention_tp")?;
    let attention_dp = reader.col("attention_dp")?;
    let moe_ep = reader.col("moe_ep")?;
    let requests = reader.col("requests_per_dp")?;
    let kv_tokens = reader.col("kv_tokens")?;
    let latency = reader.col("latency_ms")?;
    let variation = reader.col("cv")?;
    let samples = reader.col("sample_count")?;
    let checks = reader.col("numerical_checks")?;
    let contract = reader.col("boundary_contract")?;
    let source = reader.col("kernel_source")?;
    let identity = reader.col("runtime_identity_sha256")?;
    let quantization = reader.col("quantization_sha256")?;
    let mut raw: BTreeMap<AscendLayerKey, BTreeMap<u32, BTreeMap<u32, f64>>> = BTreeMap::new();
    let mut runtime = None;
    for row in reader.rows()? {
        let row = row?;
        let key = AscendLayerKey {
            attention_tp: row.u32(attention_tp)?,
            attention_dp: row.u32(attention_dp)?,
            moe_ep: row.u32(moe_ep)?,
            layer: row.u32(layer)?,
            quantization_sha256: row.str_owned(quantization)?,
        };
        key.validate()?;
        let requests = row.u32(requests)?;
        let kv_tokens = row.u32(kv_tokens)?;
        let measured = row.f64(latency)?;
        let variation = row.f64(variation)?;
        let signature = row.str(identity)?;
        if requests == 0
            || kv_tokens == 0
            || !measured.is_finite()
            || measured <= 0.0
            || !variation.is_finite()
            || !(0.0..=0.1).contains(&variation)
            || row.u32(samples)? < 9
            || row.u32(checks)? < 16 * 9
            || row.str(contract)? != CONTRACT
            || row.str(source)? != SOURCE
            || signature.len() != 64
            || !signature.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(AicError::InvalidPerfData(format!(
                "unqualified Ascend runtime layer row: {key:?}"
            )));
        }
        if runtime.get_or_insert_with(|| signature.to_owned()).as_str() != signature {
            return Err(AicError::InvalidPerfData(
                "Ascend runtime layers mix collection identities".into(),
            ));
        }
        if raw
            .entry(key)
            .or_default()
            .entry(requests)
            .or_default()
            .insert(kv_tokens, measured)
            .is_some()
        {
            return Err(AicError::InvalidPerfData(
                "duplicate Ascend runtime layer coordinate".into(),
            ));
        }
    }
    if raw.is_empty() {
        return Err(AicError::InvalidPerfData(
            "empty Ascend runtime layer table".into(),
        ));
    }
    Ok(raw
        .into_iter()
        .map(|(key, points)| (key, Plane::from_raw(points)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_grid_interpolates_only_inside_measured_brackets() {
        let plane = Plane::from_raw(BTreeMap::from([
            (1, BTreeMap::from([(512, 4.0), (1536, 6.0)])),
            (3, BTreeMap::from([(512, 6.0), (1536, 8.0)])),
        ]));
        assert_eq!(plane.query(2, 1024).unwrap(), 6.0);
        assert!(plane.query(4, 1024).is_err());
        assert!(plane.query(2, 2048).is_err());
    }
}
