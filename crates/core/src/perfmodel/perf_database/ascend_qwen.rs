//! Qwen3.5 ModelSlim operator measurements. No cross-model or SOL fallback.
use super::parquet_loader::PerfReader;
use crate::common::error::AicError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub const BASENAME: &str = "ascend_qwen_perf.parquet";
pub const CONTRACT: &str = "qwen35_397b_modelslim_bf16_state_eager/v1";
pub const GRAPH_CONTRACT: &str = "qwen35_397b_modelslim_fp32_state_graph_tp4_dp4_ep16/v1";
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendQwenKey {
    pub role: String,
    pub tp: u32,
    pub phase: String,
}
impl AscendQwenKey {
    pub fn validate(&self) -> Result<(), AicError> {
        let expert_role = matches!(
            self.role.as_str(),
            "routed_experts" | "expert_dispatch" | "expert_combine"
        );
        let valid_width = if expert_role {
            self.tp.is_power_of_two() && (2..=512).contains(&self.tp)
        } else {
            matches!(self.tp, 1 | 2 | 4 | 8 | 16)
        };
        if !valid_width
            || !matches!(self.phase.as_str(), "prefill" | "decode")
            || !matches!(
                self.role.as_str(),
                "gdn_conv"
                    | "gdn_scan"
                    | "gdn_state_io"
                    | "gdn_norm_gate"
                    | "gqa_prepare"
                    | "gqa_cache_write"
                    | "gqa_attention"
                    | "gqa_gate"
                    | "rms_norm"
                    | "add_rms_norm"
                    | "router"
                    | "routed_experts"
                    | "expert_dispatch"
                    | "expert_combine"
                    | "shared_swiglu"
                    | "shared_gate_merge"
                    | "embedding"
                    | "logits_finalize"
            )
        {
            return Err(AicError::InvalidEngineConfig(format!(
                "unsupported Qwen Ascend physical identity: {self:?}"
            )));
        }
        Ok(())
    }
    pub fn parameter_bytes(&self) -> f64 {
        let tp = self.tp as f64;
        match self.role.as_str() {
            "gdn_conv" => 12288.0 / tp * 4.0 * 2.0,
            "gdn_scan" => 64.0 / tp * 6.0, // FP32 A_log + BF16 dt_bias
            "gdn_norm_gate" => 128.0 * 2.0,
            "rms_norm" | "add_rms_norm" => 4096.0 * 2.0,
            "gqa_prepare" => 2.0 * 256.0 * 2.0,
            "embedding" => 248320.0 / tp * 4096.0 * 2.0,
            // W8 expert matrix bytes plus BF16 per-output scales and offsets; no shared experts here.
            "routed_experts" => (512.0 / tp) * (3.0 * 4096.0 * 1024.0 + 4.0 * (2048.0 + 4096.0)),
            _ => 0.0,
        }
    }
}
type Plane = BTreeMap<(u32, u32), f64>;
struct QualifiedPlane {
    points: Plane,
    contract: String,
    runtime: String,
}
type Grids = BTreeMap<AscendQwenKey, QualifiedPlane>;
pub struct AscendQwenTable {
    path: Option<PathBuf>,
    grids: OnceLock<Result<Grids, String>>,
}
impl AscendQwenTable {
    pub fn new(path: Option<PathBuf>) -> Self {
        Self {
            path,
            grids: OnceLock::new(),
        }
    }
    pub fn with_sources(root: &Path, resolver: &super::SourceResolver) -> Result<Self, AicError> {
        let sources = resolver.prioritized_sources_for(BASENAME, root)?;
        let source = sources.into_iter().find(|s| s.channel == "primary");
        if source.as_ref().is_some_and(|s| {
            s.source.kernel_sources().is_some() || s.source.0 != root.join(BASENAME)
        }) {
            return Err(AicError::InvalidPerfData(
                "Qwen requires an unfiltered table from the exact requested runtime directory"
                    .into(),
            ));
        }
        Ok(Self::new(source.map(|s| s.source.0)))
    }
    fn plane(&self, key: &AscendQwenKey) -> Result<&QualifiedPlane, AicError> {
        key.validate()?;
        if !self.path.as_ref().is_some_and(|p| p.is_file()) {
            return Err(AicError::PerfDatabase(
                "Qwen Ascend operator table missing".into(),
            ));
        }
        self.grids
            .get_or_init(|| load(self.path.as_ref().unwrap()).map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| AicError::InvalidPerfData(e.clone()))?
            .get(key)
            .ok_or_else(|| AicError::PerfDatabase(format!("unmeasured Qwen operator: {key:?}")))
    }
    pub fn ensure_key(&self, key: &AscendQwenKey) -> Result<(), AicError> {
        self.plane(key).map(|_| ())
    }
    pub fn ensure_contract(&self, key: &AscendQwenKey, expected: &str) -> Result<(), AicError> {
        if self.plane(key)?.contract != expected {
            return Err(AicError::InvalidPerfData(format!(
                "Qwen measurement contract does not match requested execution: {key:?}"
            )));
        }
        Ok(())
    }
    pub fn query(&self, key: &AscendQwenKey, tokens: u32, history: u32) -> Result<f64, AicError> {
        interpolate(&self.plane(key)?.points, tokens, history)
    }
}
fn interpolate(plane: &Plane, x: u32, y: u32) -> Result<f64, AicError> {
    if let Some(v) = plane.get(&(x, y)) {
        return Ok(*v);
    }
    let miss = || {
        AicError::PerfDatabase(format!(
            "Qwen unmeasured or incomplete rectangle at tokens={x}, history={y}; extrapolation disabled"
        ))
    };
    let x0 = plane
        .keys()
        .map(|p| p.0)
        .filter(|v| *v <= x)
        .max()
        .ok_or_else(miss)?;
    let x1 = plane
        .keys()
        .map(|p| p.0)
        .filter(|v| *v >= x)
        .min()
        .ok_or_else(miss)?;
    let y0 = plane
        .keys()
        .map(|p| p.1)
        .filter(|v| *v <= y)
        .max()
        .ok_or_else(miss)?;
    let y1 = plane
        .keys()
        .map(|p| p.1)
        .filter(|v| *v >= y)
        .min()
        .ok_or_else(miss)?;
    let lerp = |a: f64, b: f64, v: u32, lo: u32, hi: u32| {
        if lo == hi {
            a
        } else {
            a + (b - a) * f64::from(v - lo) / f64::from(hi - lo)
        }
    };
    let a = lerp(
        *plane.get(&(x0, y0)).ok_or_else(miss)?,
        *plane.get(&(x1, y0)).ok_or_else(miss)?,
        x,
        x0,
        x1,
    );
    let b = lerp(
        *plane.get(&(x0, y1)).ok_or_else(miss)?,
        *plane.get(&(x1, y1)).ok_or_else(miss)?,
        x,
        x0,
        x1,
    );
    Ok(lerp(a, b, y, y0, y1))
}
fn load(path: &Path) -> Result<Grids, AicError> {
    let reader = PerfReader::open(path)?;
    let role = reader.col("role")?;
    let tp = reader.col("tp")?;
    let phase = reader.col("phase")?;
    let tokens = reader.col("tokens")?;
    let history = reader.col("history")?;
    let latency = reader.col("latency_ms")?;
    let cv = reader.col("cv")?;
    let samples = reader.col("sample_count")?;
    let checks = reader.col("numerical_checks")?;
    let contract = reader.col("boundary_contract")?;
    let identity = reader.col("runtime_identity_sha256")?;
    let signature = reader.col("signature_sha256")?;
    let verify = reader.col("execution_verification")?;
    let mut grids: Grids = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = AscendQwenKey {
            role: row.str_owned(role)?,
            tp: row.u32(tp)?,
            phase: row.str_owned(phase)?,
        };
        key.validate()?;
        let x = row.u32(tokens)?;
        let y = row.u32(history)?;
        let value = row.f64(latency)?;
        let variation = row.f64(cv)?;
        let id = row.str(identity)?;
        let boundary = row.str(contract)?;
        let expected_verification = match boundary {
            CONTRACT => "eager_invocation",
            GRAPH_CONTRACT => "captured_graph_replay",
            _ => {
                return Err(AicError::InvalidPerfData(
                    "unknown Qwen boundary contract".into(),
                ));
            }
        };
        if x == 0
            || !value.is_finite()
            || value <= 0.0
            || !variation.is_finite()
            || !(0.0..=0.1).contains(&variation)
            || row.u32(samples)? < 10
            || row.u32(checks)? == 0
            || row.str(verify)? != expected_verification
            || [id, row.str(signature)?]
                .iter()
                .any(|s| s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()))
            || (key.role != "gqa_attention" && y != 0)
        {
            return Err(AicError::InvalidPerfData(format!(
                "unqualified Qwen measurement {key:?}"
            )));
        }
        let grid = grids.entry(key.clone()).or_insert_with(|| QualifiedPlane {
            points: Plane::new(),
            contract: boundary.into(),
            runtime: id.into(),
        });
        if grid.runtime != id || grid.contract != boundary {
            return Err(AicError::InvalidPerfData(
                "mixed Qwen execution identities within one lookup curve".into(),
            ));
        }
        if grid.points.insert((x, y), value).is_some() {
            return Err(AicError::InvalidPerfData(
                "duplicate Qwen physical point".into(),
            ));
        }
    }
    if grids.is_empty() {
        return Err(AicError::InvalidPerfData("empty Qwen table".into()));
    }
    Ok(grids)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn measured_rectangle_and_missing_corners() {
        // Four independently specified corners: midpoint is their arithmetic mean, 5 ms.
        let mut plane = Plane::from([((1, 0), 2.0), ((3, 0), 4.0), ((1, 10), 6.0), ((3, 10), 8.0)]);
        assert_eq!(interpolate(&plane, 2, 5).unwrap(), 5.0);
        assert!(interpolate(&plane, 4, 5).is_err());
        plane.remove(&(3, 10));
        assert!(interpolate(&plane, 2, 5).is_err());
    }
    #[test]
    fn identity_and_missing_data_fail_closed() {
        let key = AscendQwenKey {
            role: "gdn_scan".into(),
            tp: 8,
            phase: "decode".into(),
        };
        assert!(key.validate().is_ok());
        assert!(AscendQwenTable::new(None).ensure_key(&key).is_err());
        assert!(
            AscendQwenKey {
                tp: 32,
                ..key.clone()
            }
            .validate()
            .is_err()
        );
        assert!(
            AscendQwenKey {
                role: "glm_scan".into(),
                ..key
            }
            .validate()
            .is_err()
        );
    }
    #[test]
    fn parquet_contract_rejects_unqualified_measurements() {
        use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
        let directory = tempfile::tempdir().unwrap();
        let key = AscendQwenKey {
            role: "gqa_attention".into(),
            tp: 8,
            phase: "decode".into(),
        };
        for (name, variation, samples, identity, verification, succeeds) in [
            (
                "valid",
                0.02,
                30,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "eager_invocation",
                true,
            ),
            (
                "unstable",
                0.11,
                30,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "eager_invocation",
                false,
            ),
            (
                "few_samples",
                0.02,
                2,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "eager_invocation",
                false,
            ),
            ("identity", 0.02, 30, "unknown", "eager_invocation", false),
            (
                "unchecked_graph",
                0.02,
                30,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "captured_graph_replay",
                false,
            ),
        ] {
            let path = directory.path().join(format!("{name}.parquet"));
            write_parquet(
                &path,
                &[
                    Col::Str("role", vec!["gqa_attention"]),
                    Col::I64("tp", vec![8]),
                    Col::Str("phase", vec!["decode"]),
                    Col::I64("tokens", vec![8]),
                    Col::I64("history", vec![1024]),
                    Col::F64("latency_ms", vec![2.5]),
                    Col::F64("cv", vec![variation]),
                    Col::I64("sample_count", vec![samples]),
                    Col::I64("numerical_checks", vec![2]),
                    Col::Str("boundary_contract", vec![CONTRACT]),
                    Col::Str("runtime_identity_sha256", vec![identity]),
                    Col::Str(
                        "signature_sha256",
                        vec!["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],
                    ),
                    Col::Str("execution_verification", vec![verification]),
                ],
            );
            let table = AscendQwenTable::new(Some(path));
            let result = table.query(&key, 8, 1024);
            assert_eq!(result.is_ok(), succeeds, "{name}");
            if succeeds {
                assert_eq!(result.unwrap(), 2.5);
                assert!(table.query(&key, 9, 1024).is_err());
                assert!(table.query(&key, 8, 2048).is_err());
            }
        }
    }
}

#[cfg(test)]
mod topology_width_tests {
    use super::*;
    #[test]
    fn attention_and_expert_partition_widths_are_distinct() {
        let mut key = AscendQwenKey {
            role: "routed_experts".into(),
            tp: 32,
            phase: "decode".into(),
        };
        assert!(key.validate().is_ok());
        let ep32 = key.parameter_bytes();
        key.tp = 16;
        assert_eq!(key.parameter_bytes(), 2.0 * ep32);
        key.role = "gdn_scan".into();
        assert!(key.validate().is_ok());
        key.tp = 32;
        assert!(key.validate().is_err());
    }
}
