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
pub const GRAPH_TP2_CONTRACT: &str = "qwen35_397b_modelslim_fp32_state_graph_tp2_dp8_ep16/v1";
pub const GRAPH_TP8_CONTRACT: &str = "qwen35_397b_modelslim_fp32_state_graph_tp8_dp2_ep16/v1";
pub const FP32_EAGER_TP2_CONTRACT: &str = "qwen35_397b_modelslim_fp32_state_eager_tp2_dp8_ep16/v1";
pub const FP32_EAGER_TP4_CONTRACT: &str = "qwen35_397b_modelslim_fp32_state_eager_tp4_dp4_ep16/v1";
pub const FP32_EAGER_TP8_CONTRACT: &str = "qwen35_397b_modelslim_fp32_state_eager_tp8_dp2_ep16/v1";
pub const PREFILL_INT8_EXPERT_CONTRACT: &str = "qwen35_397b_modelslim_prefill_int8_expert_chain/v1";
pub const NORMAL_TRANSPORT_CONTRACT: &str =
    "sglang_a6b5428_910c_bf16_h4096_ep16_top10_aiv48_normal_ready/v1";
pub const HOST_CPU_CONTRACT: &str =
    "qwen35_a6b5428_node20_fp32_eager_tp4_dp4_ep16_exclusive_cpu_mean/v1";

pub fn fp32_eager_attention_tp(contract: &str) -> Option<u32> {
    match contract {
        FP32_EAGER_TP2_CONTRACT => Some(2),
        FP32_EAGER_TP4_CONTRACT => Some(4),
        FP32_EAGER_TP8_CONTRACT => Some(8),
        _ => None,
    }
}

pub fn graph_attention_tp(contract: &str) -> Option<u32> {
    match contract {
        GRAPH_TP2_CONTRACT => Some(2),
        GRAPH_CONTRACT => Some(4),
        GRAPH_TP8_CONTRACT => Some(8),
        _ => None,
    }
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendQwenKey {
    pub role: String,
    pub tp: u32,
    pub phase: String,
}
impl AscendQwenKey {
    pub fn is_host_cpu(&self) -> bool {
        let Some((component, rank)) = self.role.rsplit_once("_rank") else {
            return false;
        };
        let Ok(rank_number) = rank.parse::<u32>() else {
            return false;
        };
        self.tp == 4
            && self.phase == "decode"
            && rank_number < 16
            && rank == rank_number.to_string()
            && matches!(
                component,
                "host_cpu_linear_attention"
                    | "host_cpu_linear_layer_host"
                    | "host_cpu_full_attention_layer_host"
                    | "host_cpu_moe"
                    | "host_cpu_layer_norms"
                    | "host_cpu_outer_forward"
            )
    }
    pub fn is_normal_transport(&self) -> bool {
        self.tp == 16
            && self.phase == "prefill"
            && matches!(
                self.role.as_str(),
                "normal_dispatch_ready" | "normal_combine"
            )
    }
    pub fn validate(&self) -> Result<(), AicError> {
        let expert_role = matches!(
            self.role.as_str(),
            "routed_experts"
                | "expert_dispatch"
                | "expert_combine"
                | "expert_transport_pair"
                | "normal_dispatch_ready"
                | "normal_combine"
        );
        let valid_width = if expert_role {
            self.tp.is_power_of_two() && (2..=512).contains(&self.tp)
        } else {
            matches!(self.tp, 1 | 2 | 4 | 8 | 16)
        };
        if !valid_width
            || (self.role == "expert_transport_pair" && (self.phase != "decode" || self.tp != 16))
            || (matches!(
                self.role.as_str(),
                "normal_dispatch_ready" | "normal_combine"
            ) && !self.is_normal_transport())
            || !matches!(self.phase.as_str(), "prefill" | "decode")
            || !(self.is_host_cpu()
                || matches!(
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
                        | "expert_transport_pair"
                        | "normal_dispatch_ready"
                        | "normal_combine"
                        | "shared_swiglu"
                        | "shared_gate_merge"
                        | "embedding"
                        | "logits_finalize"
                ))
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
pub fn is_prefill_int8_expert_boundary(key: &AscendQwenKey, contract: &str) -> bool {
    contract == PREFILL_INT8_EXPERT_CONTRACT
        && key.role == "routed_experts"
        && key.phase == "prefill"
        && key.tp == 16
}

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
        let measured = &self.plane(key)?.contract;
        if measured != expected
            && !same_decode_expert_boundary(key, measured, expected)
            && !same_fp32_prefill_stateless_boundary(key, measured, expected)
            && !same_fp32_prefill_dispatch_boundary(key, measured, expected)
        {
            return Err(AicError::InvalidPerfData(format!(
                "Qwen measurement contract does not match requested execution: {key:?}"
            )));
        }
        Ok(())
    }
    pub fn query(&self, key: &AscendQwenKey, tokens: u32, history: u32) -> Result<f64, AicError> {
        interpolate(&self.plane(key)?.points, tokens, history)
            .map_err(|error| AicError::PerfDatabase(format!("{key:?}: {error}")))
    }
    pub fn query_normal_transport(
        &self,
        key: &AscendQwenKey,
        tokens: u32,
        core_work_per_input: f64,
    ) -> Result<f64, AicError> {
        if !key.is_normal_transport() {
            return Err(AicError::InvalidEngineConfig(
                "Normal transport physical key required".into(),
            ));
        }
        self.ensure_contract(key, NORMAL_TRANSPORT_CONTRACT)?;
        interpolate_normal_transport(&self.plane(key)?.points, tokens, core_work_per_input)
    }
}

/// Each measured send-size slice has its own exact concentration coordinates.
/// Interpolate within each bounding slice at the same physical work/input ratio,
/// then across send sizes. All four contributing coordinates must be measured.
fn interpolate_normal_transport(
    plane: &Plane,
    tokens: u32,
    core_work_per_input: f64,
) -> Result<f64, AicError> {
    let missing = || {
        AicError::PerfDatabase(format!(
            "Normal transport outside measured send/core-work domain: tokens={tokens}, core_work_per_input={core_work_per_input}; extrapolation disabled"
        ))
    };
    if tokens == 0 || !core_work_per_input.is_finite() || core_work_per_input <= 0.0 {
        return Err(missing());
    }
    let lower = plane
        .keys()
        .map(|point| point.0)
        .filter(|&value| value <= tokens)
        .max()
        .ok_or_else(missing)?;
    let upper = plane
        .keys()
        .map(|point| point.0)
        .filter(|&value| value >= tokens)
        .min()
        .ok_or_else(missing)?;
    let query_slice = |send_tokens: u32| -> Result<f64, AicError> {
        let work = core_work_per_input * f64::from(send_tokens);
        let points: Vec<_> = plane
            .range((send_tokens, 0)..=(send_tokens, u32::MAX))
            .collect();
        let (a, av) = points
            .iter()
            .rev()
            .find(|(point, _)| f64::from(point.1) <= work + 1e-8)
            .ok_or_else(missing)?;
        let (b, bv) = points
            .iter()
            .find(|(point, _)| f64::from(point.1) >= work - 1e-8)
            .ok_or_else(missing)?;
        if a.1 == b.1 {
            return Ok(**av);
        }
        Ok(**av + (**bv - **av) * (work - f64::from(a.1)) / f64::from(b.1 - a.1))
    };
    let a = query_slice(lower)?;
    if lower == upper {
        return Ok(a);
    }
    let b = query_slice(upper)?;
    Ok(a + (b - a) * f64::from(tokens - lower) / f64::from(upper - lower))
}
fn same_decode_expert_boundary(key: &AscendQwenKey, measured: &str, requested: &str) -> bool {
    // EP16 expert input/weights and transport capacity are independent of
    // attention TP. Their table token axis is sent tokens per expert rank.
    // The caller's contract still controls request-to-sent-token conversion.
    key.tp == 16
        && key.phase == "decode"
        && matches!(
            key.role.as_str(),
            "routed_experts" | "expert_dispatch" | "expert_combine" | "expert_transport_pair"
        )
        && ((graph_attention_tp(measured).is_some() && graph_attention_tp(requested).is_some())
            || (fp32_eager_attention_tp(measured).is_some()
                && fp32_eager_attention_tp(requested).is_some()))
}

fn same_fp32_prefill_stateless_boundary(
    key: &AscendQwenKey,
    measured: &str,
    requested: &str,
) -> bool {
    let Some(attention_tp) = fp32_eager_attention_tp(requested) else {
        return false;
    };
    // The audited Prefill path retains BF16 embedding, H4096 Gemma norms and
    // shared SwiGLU when the recurrent SSM state changes to FP32. The SwiGLU
    // producer already maps the pre-scatter token key to ceil(tokens / TP)
    // rows of its contiguous width2048 input. None consumes recurrent state.
    // DP attention uses a replicated embedding, represented by key.tp == 1.
    // Source and numerical-oracle receipts are in the NPU20 Prefill reuse audit.
    measured == CONTRACT
        && key.phase == "prefill"
        && match key.role.as_str() {
            "embedding" => key.tp == 1,
            "rms_norm" | "add_rms_norm" | "shared_swiglu" => key.tp == attention_tp,
            _ => false,
        }
}
fn same_fp32_prefill_dispatch_boundary(
    key: &AscendQwenKey,
    measured: &str,
    requested: &str,
) -> bool {
    // Normal dispatch consumes the same BF16 H4096, top10 payload across
    // attention partitions. Its axis is global input tokens for EP16.
    key.tp == 16
        && key.phase == "prefill"
        && key.role == "expert_dispatch"
        && fp32_eager_attention_tp(measured).is_some()
        && fp32_eager_attention_tp(requested).is_some()
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
    let statistic = reader.col("latency_statistic").ok();
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
        let statistic = match statistic {
            Some(column) => row.str(column)?,
            None if key.is_host_cpu() => "mean",
            None => "median",
        };
        let sampling_error = match statistic {
            "mean" => variation / f64::from(row.u32(samples)?).sqrt(),
            "median" if !key.is_host_cpu() => variation,
            _ => {
                return Err(AicError::InvalidPerfData(
                    "invalid Qwen latency statistic".into(),
                ));
            }
        };
        let id = row.str(identity)?;
        let boundary = row.str(contract)?;
        let expected_verification = match boundary {
            HOST_CPU_CONTRACT if key.is_host_cpu() => "exclusive_host_api_mean",
            CONTRACT => "eager_invocation",
            boundary if is_prefill_int8_expert_boundary(&key, boundary) => "eager_invocation",
            NORMAL_TRANSPORT_CONTRACT if key.is_normal_transport() => "eager_invocation",
            boundary if fp32_eager_attention_tp(boundary).is_some() => "eager_invocation",
            boundary if graph_attention_tp(boundary).is_some() => "captured_graph_replay",
            _ => {
                return Err(AicError::InvalidPerfData(
                    "unknown Qwen boundary contract".into(),
                ));
            }
        };
        if x == 0
            || (key.role == "expert_transport_pair" && fp32_eager_attention_tp(boundary).is_none())
            || !value.is_finite()
            || value <= 0.0
            || !variation.is_finite()
            // Means are assessed by their sampling uncertainty. Legacy median
            // observations retain the raw-CV guard. Never rewrite the raw CV.
            || !(0.0..=0.1).contains(&sampling_error)
            || row.u32(samples)? < 10
            || row.u32(checks)? == 0
            || row.str(verify)? != expected_verification
            || [id, row.str(signature)?]
                .iter()
                .any(|s| s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()))
            || (key.is_normal_transport() && (boundary != NORMAL_TRANSPORT_CONTRACT || y == 0))
            || (key.is_host_cpu() && boundary != HOST_CPU_CONTRACT)
            || (key.role != "gqa_attention"
                && !key.is_normal_transport()
                && !(key.role == "gdn_conv"
                    && key.phase == "prefill"
                    && fp32_eager_attention_tp(boundary).is_some()
                    && y <= 1)
                && y != 0)
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
    fn explicit_mean_uses_standard_error_and_preserves_legacy_cv_rejection() {
        use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
        let directory = tempfile::tempdir().unwrap();
        let key = AscendQwenKey {
            role: "gdn_conv".into(),
            tp: 4,
            phase: "prefill".into(),
        };
        // CV=.2 over25 observations gives relative SE=.04 for a stored mean.
        // The old median contract still rejects the same raw CV.
        for (label, statistic, accepted) in [
            ("mean", Some("mean"), true),
            ("median", Some("median"), false),
            ("unknown", Some("maximum"), false),
            ("legacy", None, false),
        ] {
            let path = directory.path().join(format!("{label}.parquet"));
            let mut columns = vec![
                Col::Str("role", vec!["gdn_conv"]),
                Col::I64("tp", vec![4]),
                Col::Str("phase", vec!["prefill"]),
                Col::I64("tokens", vec![1024]),
                Col::I64("history", vec![0]),
                Col::F64("latency_ms", vec![2.5]),
                Col::F64("cv", vec![0.2]),
                Col::I64("sample_count", vec![25]),
                Col::I64("numerical_checks", vec![2]),
                Col::Str("boundary_contract", vec![FP32_EAGER_TP4_CONTRACT]),
                Col::Str(
                    "runtime_identity_sha256",
                    vec!["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
                ),
                Col::Str(
                    "signature_sha256",
                    vec!["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],
                ),
                Col::Str("execution_verification", vec!["eager_invocation"]),
            ];
            if let Some(statistic) = statistic {
                columns.push(Col::Str("latency_statistic", vec![statistic]));
            }
            write_parquet(&path, &columns);
            let table = AscendQwenTable::new(Some(path));
            let result = table.query(&key, 1024, 0);
            assert_eq!(result.is_ok(), accepted, "{label}");
            if accepted {
                assert_eq!(result.unwrap(), 2.5);
            }
        }
    }

    #[test]
    fn host_cpu_means_have_strict_identity_and_sampling_uncertainty() {
        use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
        let directory = tempfile::tempdir().unwrap();
        let key = AscendQwenKey {
            role: "host_cpu_moe_rank15".into(),
            tp: 4,
            phase: "decode".into(),
        };
        assert!(key.is_host_cpu());
        for role in [
            "host_cpu_moe_rank16",
            "host_cpu_moe_rank01",
            "host_cpu_unknown_rank0",
        ] {
            assert!(
                AscendQwenKey {
                    role: role.into(),
                    ..key.clone()
                }
                .validate()
                .is_err()
            );
        }
        // Twelve independent observations with CV=.17 give mean relative SE=.0491.
        // Endpoints1/3 requests cost2/6ms, so the held-out midpoint must be4ms.
        for (variation, boundary, accepted) in [
            (0.17, HOST_CPU_CONTRACT, true),
            (0.40, HOST_CPU_CONTRACT, false),
            (0.01, FP32_EAGER_TP4_CONTRACT, false),
        ] {
            let path = directory
                .path()
                .join(format!("{variation}-{accepted}.parquet"));
            write_parquet(
                &path,
                &[
                    Col::Str("role", vec!["host_cpu_moe_rank15"; 2]),
                    Col::I64("tp", vec![4; 2]),
                    Col::Str("phase", vec!["decode"; 2]),
                    Col::I64("tokens", vec![1, 3]),
                    Col::I64("history", vec![0; 2]),
                    Col::F64("latency_ms", vec![2.0, 6.0]),
                    Col::F64("cv", vec![variation; 2]),
                    Col::I64("sample_count", vec![12; 2]),
                    Col::I64("numerical_checks", vec![12; 2]),
                    Col::Str("boundary_contract", vec![boundary; 2]),
                    Col::Str(
                        "runtime_identity_sha256",
                        vec!["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"; 2],
                    ),
                    Col::Str(
                        "signature_sha256",
                        vec!["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"; 2],
                    ),
                    Col::Str("execution_verification", vec!["exclusive_host_api_mean"; 2]),
                ],
            );
            let table = AscendQwenTable::new(Some(path));
            let result = table.query(&key, 2, 0);
            assert_eq!(result.is_ok(), accepted);
            if accepted {
                assert_eq!(result.unwrap(), 4.0);
                assert!(table.query(&key, 4, 0).is_err());
            }
        }
    }

    #[test]
    fn normal_transport_interpolates_each_measured_physical_slice_without_extrapolation() {
        // At1024 sent tokens, core-work numerator256..1024 costs2..5ms.
        // At2048, the same concentration ratios numerator512..2048 cost4..10ms.
        // Mid-concentration.625 gives3.5 and7ms. Send-size midpoint1536 gives5.25ms.
        let plane = Plane::from([
            ((1024, 256), 2.0),
            ((1024, 1024), 5.0),
            ((2048, 512), 4.0),
            ((2048, 2048), 10.0),
        ]);
        assert_eq!(
            interpolate_normal_transport(&plane, 1536, 0.625).unwrap(),
            5.25
        );
        assert_eq!(
            interpolate_normal_transport(&plane, 1024, 0.25).unwrap(),
            2.0
        );
        for (tokens, ratio) in [
            (512, 0.625),
            (4096, 0.625),
            (1536, 0.2),
            (1536, 1.1),
            (1536, f64::NAN),
        ] {
            assert!(interpolate_normal_transport(&plane, tokens, ratio).is_err());
        }
    }

    #[test]
    fn current_eager_expert_reuse_preserves_mode_and_ep_geometry() {
        let mut key = AscendQwenKey {
            role: "routed_experts".into(),
            tp: 16,
            phase: "decode".into(),
        };
        for role in ["routed_experts", "expert_dispatch", "expert_combine"] {
            key.role = role.into();
            assert!(same_decode_expert_boundary(
                &key,
                FP32_EAGER_TP4_CONTRACT,
                FP32_EAGER_TP2_CONTRACT
            ));
            assert!(same_decode_expert_boundary(
                &key,
                FP32_EAGER_TP4_CONTRACT,
                FP32_EAGER_TP8_CONTRACT
            ));
            assert!(!same_decode_expert_boundary(
                &key,
                GRAPH_CONTRACT,
                FP32_EAGER_TP2_CONTRACT
            ));
            assert!(!same_decode_expert_boundary(
                &key,
                FP32_EAGER_TP4_CONTRACT,
                GRAPH_TP8_CONTRACT
            ));
        }
        key.phase = "prefill".into();
        assert!(!same_decode_expert_boundary(
            &key,
            FP32_EAGER_TP4_CONTRACT,
            FP32_EAGER_TP2_CONTRACT
        ));
        key.phase = "decode".into();
        key.tp = 8;
        assert!(!same_decode_expert_boundary(
            &key,
            FP32_EAGER_TP4_CONTRACT,
            FP32_EAGER_TP2_CONTRACT
        ));
    }

    #[test]
    fn prefill_int8_expert_reference_is_a_distinct_physical_boundary() {
        let mut key = AscendQwenKey {
            role: "routed_experts".into(),
            tp: 16,
            phase: "prefill".into(),
        };
        assert!(is_prefill_int8_expert_boundary(
            &key,
            PREFILL_INT8_EXPERT_CONTRACT
        ));
        assert!(!is_prefill_int8_expert_boundary(
            &key,
            FP32_EAGER_TP4_CONTRACT
        ));
        key.phase = "decode".into();
        assert!(!is_prefill_int8_expert_boundary(
            &key,
            PREFILL_INT8_EXPERT_CONTRACT
        ));
        key.phase = "prefill".into();
        key.role = "expert_dispatch".into();
        assert!(!is_prefill_int8_expert_boundary(
            &key,
            PREFILL_INT8_EXPERT_CONTRACT
        ));
    }

    #[test]
    fn normal_prefill_dispatch_reuses_only_matching_ep16_current_boundaries() {
        let key = AscendQwenKey {
            role: "expert_dispatch".into(),
            tp: 16,
            phase: "prefill".into(),
        };
        for target in [
            FP32_EAGER_TP2_CONTRACT,
            FP32_EAGER_TP4_CONTRACT,
            FP32_EAGER_TP8_CONTRACT,
        ] {
            assert!(same_fp32_prefill_dispatch_boundary(
                &key,
                FP32_EAGER_TP4_CONTRACT,
                target
            ));
            assert!(!same_fp32_prefill_dispatch_boundary(&key, CONTRACT, target));
            assert!(!same_fp32_prefill_dispatch_boundary(
                &key,
                GRAPH_CONTRACT,
                target
            ));
        }
        for other in [
            AscendQwenKey {
                role: "routed_experts".into(),
                ..key.clone()
            },
            AscendQwenKey {
                role: "expert_combine".into(),
                ..key.clone()
            },
            AscendQwenKey {
                tp: 32,
                ..key.clone()
            },
            AscendQwenKey {
                phase: "decode".into(),
                ..key
            },
        ] {
            assert!(!same_fp32_prefill_dispatch_boundary(
                &other,
                FP32_EAGER_TP4_CONTRACT,
                FP32_EAGER_TP2_CONTRACT
            ));
        }
    }

    #[test]
    fn fp32_prefill_reuses_only_identical_stateless_signatures() {
        for (tp, contract) in [
            (2, FP32_EAGER_TP2_CONTRACT),
            (4, FP32_EAGER_TP4_CONTRACT),
            (8, FP32_EAGER_TP8_CONTRACT),
        ] {
            for role in ["embedding", "rms_norm", "add_rms_norm", "shared_swiglu"] {
                let mut key = AscendQwenKey {
                    role: role.into(),
                    tp: if role == "embedding" { 1 } else { tp },
                    phase: "prefill".into(),
                };
                assert!(same_fp32_prefill_stateless_boundary(
                    &key, CONTRACT, contract
                ));
                key.phase = "decode".into();
                assert!(!same_fp32_prefill_stateless_boundary(
                    &key, CONTRACT, contract
                ));
                key.phase = "prefill".into();
                key.tp = if role == "embedding" { tp } else { 16 };
                assert!(!same_fp32_prefill_stateless_boundary(
                    &key, CONTRACT, contract
                ));
            }
        }
    }

    #[test]
    fn fp32_prefill_stateless_reuse_keeps_stateful_and_graph_contracts_distinct() {
        for role in [
            "gdn_conv",
            "gdn_scan",
            "gdn_state_io",
            "gdn_norm_gate",
            "router",
            "routed_experts",
            "gqa_gate",
            "shared_gate_merge",
        ] {
            let key = AscendQwenKey {
                role: role.into(),
                tp: 4,
                phase: "prefill".into(),
            };
            assert!(!same_fp32_prefill_stateless_boundary(
                &key,
                CONTRACT,
                FP32_EAGER_TP4_CONTRACT
            ));
        }
        let key = AscendQwenKey {
            role: "rms_norm".into(),
            tp: 4,
            phase: "prefill".into(),
        };
        for (measured, requested) in [
            (CONTRACT, GRAPH_CONTRACT),
            (GRAPH_CONTRACT, FP32_EAGER_TP4_CONTRACT),
            (FP32_EAGER_TP4_CONTRACT, CONTRACT),
            (CONTRACT, "unknown"),
            (CONTRACT, FP32_EAGER_TP2_CONTRACT),
        ] {
            assert!(!same_fp32_prefill_stateless_boundary(
                &key, measured, requested
            ));
        }
    }

    #[test]
    fn fp32_prefill_stateless_reuse_preserves_measured_values_and_bounds() {
        use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("stateless-prefill.parquet");
        write_parquet(
            &path,
            &[
                Col::Str(
                    "role",
                    vec!["embedding", "rms_norm", "add_rms_norm", "shared_swiglu"],
                ),
                Col::I64("tp", vec![1, 4, 4, 4]),
                Col::Str("phase", vec!["prefill"; 4]),
                Col::I64("tokens", vec![1024; 4]),
                Col::I64("history", vec![0; 4]),
                // Independent synthetic points exercise identity reuse, not interpolation.
                Col::F64("latency_ms", vec![0.25, 0.5, 0.75, 0.125]),
                Col::F64("cv", vec![0.01; 4]),
                Col::I64("sample_count", vec![30; 4]),
                Col::I64("numerical_checks", vec![2; 4]),
                Col::Str("boundary_contract", vec![CONTRACT; 4]),
                Col::Str("execution_verification", vec!["eager_invocation"; 4]),
                Col::Str(
                    "runtime_identity_sha256",
                    vec!["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"; 4],
                ),
                Col::Str(
                    "signature_sha256",
                    vec!["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"; 4],
                ),
            ],
        );
        let table = AscendQwenTable::new(Some(path));
        for (role, tp, expected) in [
            ("embedding", 1, 0.25),
            ("rms_norm", 4, 0.5),
            ("add_rms_norm", 4, 0.75),
            ("shared_swiglu", 4, 0.125),
        ] {
            let key = AscendQwenKey {
                role: role.into(),
                tp,
                phase: "prefill".into(),
            };
            table.ensure_contract(&key, CONTRACT).unwrap();
            table
                .ensure_contract(&key, FP32_EAGER_TP4_CONTRACT)
                .unwrap();
            assert_eq!(table.query(&key, 1024, 0).unwrap(), expected);
            assert!(table.query(&key, 2048, 0).is_err());
        }
    }
    #[test]
    fn ep16_graph_plane_reuse_keeps_the_measured_token_domain() {
        use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("expert.parquet");
        write_parquet(
            &path,
            &[
                Col::Str("role", vec!["routed_experts"]),
                Col::I64("tp", vec![16]),
                Col::Str("phase", vec!["decode"]),
                Col::I64("tokens", vec![1]),
                Col::I64("history", vec![0]),
                Col::F64("latency_ms", vec![7.0]),
                Col::F64("cv", vec![0.01]),
                Col::I64("sample_count", vec![10]),
                Col::I64("numerical_checks", vec![2]),
                Col::Str("boundary_contract", vec![GRAPH_CONTRACT]),
                Col::Str("execution_verification", vec!["captured_graph_replay"]),
                Col::Str(
                    "runtime_identity_sha256",
                    vec!["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
                ),
                Col::Str(
                    "signature_sha256",
                    vec!["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],
                ),
            ],
        );
        let table = AscendQwenTable::new(Some(path));
        let key = AscendQwenKey {
            role: "routed_experts".into(),
            tp: 16,
            phase: "decode".into(),
        };
        for requested in [GRAPH_TP2_CONTRACT, GRAPH_CONTRACT, GRAPH_TP8_CONTRACT] {
            table.ensure_contract(&key, requested).unwrap();
            assert_eq!(table.query(&key, 1, 0).unwrap(), 7.0);
            assert!(table.query(&key, 2, 0).is_err());
        }
        assert!(
            table
                .ensure_contract(&key, FP32_EAGER_TP4_CONTRACT)
                .is_err()
        );
    }

    #[test]
    fn ep16_graph_reuse_keeps_attention_phase_and_execution_boundaries() {
        for role in ["routed_experts", "expert_dispatch", "expert_combine"] {
            let key = AscendQwenKey {
                role: role.into(),
                tp: 16,
                phase: "decode".into(),
            };
            for measured in [GRAPH_TP2_CONTRACT, GRAPH_CONTRACT, GRAPH_TP8_CONTRACT] {
                for requested in [GRAPH_TP2_CONTRACT, GRAPH_CONTRACT, GRAPH_TP8_CONTRACT] {
                    assert!(same_decode_expert_boundary(&key, measured, requested));
                }
                assert!(!same_decode_expert_boundary(&key, measured, CONTRACT));
                assert!(!same_decode_expert_boundary(
                    &key,
                    measured,
                    FP32_EAGER_TP4_CONTRACT
                ));
                assert!(!same_decode_expert_boundary(
                    &AscendQwenKey {
                        phase: "prefill".into(),
                        ..key.clone()
                    },
                    measured,
                    GRAPH_CONTRACT,
                ));
                assert!(!same_decode_expert_boundary(
                    &AscendQwenKey {
                        tp: 8,
                        ..key.clone()
                    },
                    measured,
                    GRAPH_CONTRACT,
                ));
            }
        }
        for role in ["gdn_scan", "gqa_attention", "router", "add_rms_norm"] {
            let key = AscendQwenKey {
                role: role.into(),
                tp: 4,
                phase: "decode".into(),
            };
            assert!(!same_decode_expert_boundary(
                &key,
                GRAPH_CONTRACT,
                GRAPH_TP2_CONTRACT
            ));
        }
    }

    #[test]
    fn fp32_eager_rows_require_eager_execution_and_exact_contract() {
        use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
        let directory = tempfile::tempdir().unwrap();
        for (verification, accepted) in
            [("eager_invocation", true), ("captured_graph_replay", false)]
        {
            let path = directory.path().join(format!("{verification}.parquet"));
            write_parquet(
                &path,
                &[
                    Col::Str("role", vec!["gdn_scan"]),
                    Col::I64("tp", vec![4]),
                    Col::Str("phase", vec!["decode"]),
                    Col::I64("tokens", vec![3]),
                    Col::I64("history", vec![0]),
                    // A synthetic measured point of 2.5ms is independent of interpolation.
                    Col::F64("latency_ms", vec![2.5]),
                    Col::F64("cv", vec![0.01]),
                    Col::I64("sample_count", vec![10]),
                    Col::I64("numerical_checks", vec![2]),
                    Col::Str("boundary_contract", vec![FP32_EAGER_TP4_CONTRACT]),
                    Col::Str("execution_verification", vec![verification]),
                    Col::Str(
                        "runtime_identity_sha256",
                        vec!["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
                    ),
                    Col::Str(
                        "signature_sha256",
                        vec!["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],
                    ),
                ],
            );
            let table = AscendQwenTable::new(Some(path));
            let key = AscendQwenKey {
                role: "gdn_scan".into(),
                tp: 4,
                phase: "decode".into(),
            };
            assert_eq!(table.query(&key, 3, 0).is_ok(), accepted);
            if accepted {
                assert_eq!(table.query(&key, 3, 0).unwrap(), 2.5);
                table
                    .ensure_contract(&key, FP32_EAGER_TP4_CONTRACT)
                    .unwrap();
                assert!(table.ensure_contract(&key, CONTRACT).is_err());
                assert!(table.ensure_contract(&key, GRAPH_CONTRACT).is_err());
            }
        }
    }
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
                let missing = table.query(&key, 9, 1024).unwrap_err().to_string();
                assert!(missing.contains("gqa_attention"));
                assert!(missing.contains("tokens=9"));
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
