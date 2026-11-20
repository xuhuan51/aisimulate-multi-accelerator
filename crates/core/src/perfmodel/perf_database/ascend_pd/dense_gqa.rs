//! Measured Qwen2.5-72B BF16 sender envelopes for the qualified 9362 runtime.

use super::{AscendPdKey, KvTransferCurve};
use crate::common::error::AicError;
use crate::perfmodel::ForwardPassPerfModelConfig;
use crate::perfmodel::perf_database::axis_curve::AxisCurve;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

const BASENAME: &str = "ascend9362_qwen72_pd.json";
const REVISION: &str = "495f39366efef23836d0cfae4fbe635880d2be31";
const CONFIG_SHA256: &str = "14ca217334fe0fd10148413592d68c99eeb33431ed89c1afa130fee560be2a29";
const BACKEND_VERSION: &str = "0.5.20.dev388+ga6b542813";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    schema_version: u32,
    evidence_sha256: String,
    source_tp: u32,
    destination_tp: u32,
    prefill_nodes: Vec<u32>,
    decode_nodes: Vec<u32>,
    same_host: BTreeMap<u32, f64>,
    cross_host: BTreeMap<u32, f64>,
    estimate: String,
    #[serde(default)]
    serialize_same_destination: bool,
}

#[derive(Clone, Debug)]
pub(super) struct DensePlacement {
    prefill_nodes: Vec<u32>,
    decode_nodes: Vec<u32>,
    same_host: AxisCurve,
    profile_sha256: String,
    evidence_sha256: String,
    estimate: String,
    serialize_same_destination: bool,
}

impl DensePlacement {
    pub(super) fn description(&self) -> serde_json::Value {
        serde_json::json!({
            "source": "measured_ascend9362_qwen72_memfabric",
            "backend_version": BACKEND_VERSION,
            "model_revision": REVISION,
            "page_size": 128,
            "bytes_per_token_per_replica": 327680,
            "prefill_nodes": self.prefill_nodes,
            "decode_nodes": self.decode_nodes,
            "profile_sha256": self.profile_sha256,
            "evidence_sha256": self.evidence_sha256,
            "estimate": self.estimate,
            "serialize_same_destination": self.serialize_same_destination,
            "scope": "one P/D pair, steady single-request payload SDK envelope; contention and setup omitted"
        })
    }

    pub(super) fn serializes_same_destination(&self) -> bool {
        self.serialize_same_destination
    }

    pub(super) fn matches_counts(&self, prefill_count: usize, decode_count: usize) -> bool {
        self.prefill_nodes.len() == prefill_count && self.decode_nodes.len() == decode_count
    }

    pub(super) fn select<'a>(
        &'a self,
        cross_host: &'a AxisCurve,
        workers: Option<(usize, usize)>,
    ) -> Result<&'a AxisCurve, AicError> {
        let Some((source, destination)) = workers else {
            return Err(invalid("Qwen transfer requires assigned source and destination replicas"));
        };
        let source_node = self.prefill_nodes.get(source);
        let destination_node = self.decode_nodes.get(destination);
        match (source_node, destination_node) {
            (Some(a), Some(b)) if a == b => Ok(&self.same_host),
            (Some(_), Some(_)) => Ok(cross_host),
            _ => Err(invalid("Qwen transfer references a replica outside the measured placement")),
        }
    }
}

pub(super) fn load(
    source: &ForwardPassPerfModelConfig,
    target: &ForwardPassPerfModelConfig,
) -> Result<Arc<KvTransferCurve>, AicError> {
    if source.model != target.model
        || source.systems_paths != target.systems_paths
        || [source, target].iter().any(|role| {
            role.system != "ascend910_9362"
                || role.backend.as_str() != "sglang"
                || role.backend_version.as_deref() != Some(BACKEND_VERSION)
                || !matches!(role.tp, 4 | 8)
                || role.attention_dp != 1
                || role.pp != 1
                || role.cp_size.unwrap_or(1) != 1
                || role.dcp.unwrap_or(1) != 1
                || role.kv_block_size != Some(128)
                || role.kvcache_quant_mode.as_deref() != Some("bfloat16")
                || role.fpm_profile.as_ref()
                    .and_then(|profile| profile.get("model_revision"))
                    .and_then(serde_json::Value::as_str) != Some(REVISION)
        })
    {
        return Err(invalid("unsupported Qwen dense P/D hardware, model, revision or layout"));
    }
    let config_path = Path::new(&source.model).join("config.json");
    let config_bytes = std::fs::read(&config_path)
        .map_err(|error| invalid(format!("reading {}: {error}", config_path.display())))?;
    if digest(&config_bytes) != CONFIG_SHA256 {
        return Err(invalid("Qwen dense P/D model config SHA-256 does not match the measurements"));
    }
    let path = source.systems_paths.first().ok_or_else(|| invalid("missing explicit systems root"))?
        .join(BASENAME);
    let encoded = std::fs::read(&path)
        .map_err(|error| invalid(format!("reading {}: {error}", path.display())))?;
    let profile: Profile = serde_json::from_slice(&encoded)
        .map_err(|error| invalid(format!("parsing {}: {error}", path.display())))?;
    if profile.schema_version != 1
        || profile.source_tp != source.tp
        || profile.destination_tp != target.tp
        || (profile.serialize_same_destination && source.tp == target.tp)
        || !matches!(profile.estimate.as_str(), "median" | "observed_high")
        || profile.evidence_sha256.len() != 64
        || !profile.evidence_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        || !valid_layout(&profile)
        || !valid_axis(&profile.same_host)
        || !valid_axis(&profile.cross_host)
        || profile.same_host.keys().ne(profile.cross_host.keys())
    {
        return Err(invalid("invalid Qwen dense P/D evidence or worker placement"));
    }
    let maximum = *profile.cross_host.last_key_value().expect("validated curve").0;
    let placement = DensePlacement {
        prefill_nodes: profile.prefill_nodes,
        decode_nodes: profile.decode_nodes,
        same_host: AxisCurve::from_map("tokens", profile.same_host),
        profile_sha256: digest(&encoded),
        evidence_sha256: profile.evidence_sha256.clone(),
        estimate: profile.estimate,
        serialize_same_destination: profile.serialize_same_destination,
    };
    Ok(Arc::new(KvTransferCurve {
        key: AscendPdKey {
            source_tp: source.tp,
            destination_tp: target.tp,
            page_layout: "contiguous".into(),
        },
        values: AxisCurve::from_map("tokens", profile.cross_host),
        minimum: 128,
        maximum,
        identity: profile.evidence_sha256,
        policy: super::super::ascend_shape::AscendShapePolicy::Anchored,
        dense_placement: Some(placement),
    }))
}

fn valid_axis(axis: &BTreeMap<u32, f64>) -> bool {
    axis.len() >= 3
        && axis.contains_key(&128)
        && axis.keys().all(|tokens| *tokens > 0 && *tokens <= 32768 && tokens % 128 == 0)
        && axis.values().all(|ms| ms.is_finite() && *ms > 0.0)
        && axis.values().zip(axis.values().skip(1)).all(|(a, b)| a <= b)
}

fn valid_layout(profile: &Profile) -> bool {
    let mut loads = BTreeMap::new();
    for (nodes, tp) in [
        (&profile.prefill_nodes, profile.source_tp),
        (&profile.decode_nodes, profile.destination_tp),
    ] {
        if nodes.is_empty() || nodes.len() > 32 || nodes.iter().any(|&node| node >= 8) {
            return false;
        }
        for &node in nodes {
            *loads.entry(node).or_insert(0u32) += tp;
        }
    }
    loads.values().all(|load| *load <= 16)
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn invalid(detail: impl Into<String>) -> AicError {
    AicError::InvalidEngineConfig(detail.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_the_assigned_pair_and_rounds_to_a_page() {
        let local = AxisCurve::from_map("tokens", BTreeMap::from([(128, 1.0), (256, 2.0)]));
        let remote = AxisCurve::from_map("tokens", BTreeMap::from([(128, 3.0), (256, 4.0)]));
        let curve = KvTransferCurve {
            key: AscendPdKey {
                source_tp: 8,
                destination_tp: 4,
                page_layout: "contiguous".into(),
            },
            values: remote,
            minimum: 128,
            maximum: 256,
            identity: "1".repeat(64),
            policy: Default::default(),
            dense_placement: Some(DensePlacement {
                prefill_nodes: vec![0, 1],
                decode_nodes: vec![1],
                same_host: local,
                profile_sha256: "2".repeat(64),
                evidence_sha256: "1".repeat(64),
                estimate: "median".into(),
                serialize_same_destination: true,
            }),
        };
        assert!(curve.validate_placement_counts(2, 1).is_ok());
        assert!(curve.validate_placement_counts(1, 2).is_err());
        assert!(curve.serializes_same_destination());
        assert_eq!(curve.latency_ms_for_workers(129, Some((1, 0))).unwrap(), 2.0);
        assert_eq!(curve.latency_ms_for_workers(129, Some((0, 0))).unwrap(), 4.0);
        assert!(curve.latency_ms_for_workers(129, Some((2, 0))).is_err());
    }
}
