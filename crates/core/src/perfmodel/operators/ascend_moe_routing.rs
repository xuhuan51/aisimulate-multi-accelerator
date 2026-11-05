//! Explicit model routing priors over independently measured tensor-MoE costs.

use serde::{Deserialize, Serialize};

use crate::common::error::AicError;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingLayer {
    pub expert_probabilities: Vec<f64>,
    pub padding_experts: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingCostPoint {
    pub physical_tokens: u32,
    pub active_experts: u32,
    pub latency_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorMoeRoutingProfile {
    pub model: String,
    pub backend_version: String,
    pub evidence_sha256: String,
    pub intermediate_size: u32,
    pub layers: Vec<RoutingLayer>,
    pub costs: Vec<RoutingCostPoint>,
}

fn invalid(message: &str) -> AicError {
    AicError::InvalidEngineConfig(format!("tensor-MoE routing profile: {message}"))
}

impl TensorMoeRoutingProfile {
    pub fn validate(&self) -> Result<(), AicError> {
        if self.model.trim().is_empty()
            || self.backend_version.trim().is_empty()
            || self.evidence_sha256.len() != 64
            || !self
                .evidence_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || !matches!(self.intermediate_size, 64 | 128 | 256 | 512)
            || self.layers.len() != 75
        {
            return Err(invalid(
                "missing identity or incompatible GLM tensor-MoE geometry",
            ));
        }
        for layer in &self.layers {
            let mut padding = layer.padding_experts.clone();
            padding.sort_unstable();
            padding.dedup();
            if layer.expert_probabilities.len() != 256
                || layer
                    .expert_probabilities
                    .iter()
                    .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
                || (layer.expert_probabilities.iter().sum::<f64>() - 8.0).abs() > 1e-6
                || padding.len() != 8
                || layer.padding_experts.len() != 8
                || padding.iter().any(|expert| *expert >= 256)
            {
                return Err(invalid(
                    "each layer requires top-8 probabilities and eight distinct padding experts",
                ));
            }
        }
        if self.costs.is_empty() {
            return Err(invalid("missing measured cost grid"));
        }
        let mut previous = None;
        for point in &self.costs {
            let key = (point.physical_tokens, point.active_experts);
            if point.physical_tokens == 0
                || point.active_experts < 8
                || point.active_experts > 256
                || u64::from(point.active_experts) > u64::from(point.physical_tokens) * 8
                || !point.latency_ms.is_finite()
                || point.latency_ms <= 0.0
                || previous.is_some_and(|last| last >= key)
            {
                return Err(invalid(
                    "cost grid must have ordered unique valid measurements",
                ));
            }
            previous = Some(key);
        }
        Ok(())
    }

    fn cost_at_tokens(&self, tokens: u32, active: f64) -> Result<f64, AicError> {
        let lower = self.costs.iter().rev().find(|point| {
            point.physical_tokens == tokens && f64::from(point.active_experts) <= active
        });
        let upper = self.costs.iter().find(|point| {
            point.physical_tokens == tokens && f64::from(point.active_experts) >= active
        });
        let (Some(lower), Some(upper)) = (lower, upper) else {
            return Err(AicError::PerfDatabase(format!(
                "tensor-MoE active experts {active} outside measured grid at {tokens} physical tokens"
            )));
        };
        if lower.active_experts == upper.active_experts {
            return Ok(lower.latency_ms);
        }
        let fraction = (active - f64::from(lower.active_experts))
            / f64::from(upper.active_experts - lower.active_experts);
        Ok(lower.latency_ms + fraction * (upper.latency_ms - lower.latency_ms))
    }

    /// Mean layer time. Real tokens are IID draws from the recorded per-layer
    /// marginal prior. Graph padding repeatedly selects the recorded eight IDs.
    /// The model does not use the target requests' expert counts or latencies.
    pub fn estimate(&self, real_tokens: u32, physical_tokens: u32) -> Result<f64, AicError> {
        self.validate()?;
        if real_tokens > physical_tokens {
            return Err(invalid("real tokens exceed physical tokens"));
        }
        if real_tokens == 0 {
            return Ok(0.0);
        }
        let lower = self
            .costs
            .iter()
            .rev()
            .find(|point| point.physical_tokens <= physical_tokens);
        let upper = self
            .costs
            .iter()
            .find(|point| point.physical_tokens >= physical_tokens);
        let (Some(lower), Some(upper)) = (lower, upper) else {
            return Err(AicError::PerfDatabase(
                "tensor-MoE physical tokens outside measured grid".into(),
            ));
        };
        let mut total = 0.0;
        for layer in &self.layers {
            let active: f64 = layer
                .expert_probabilities
                .iter()
                .enumerate()
                .map(|(expert, probability)| {
                    if physical_tokens > real_tokens
                        && layer.padding_experts.contains(&(expert as u32))
                    {
                        1.0
                    } else {
                        1.0 - (1.0 - probability).powf(f64::from(real_tokens))
                    }
                })
                .sum();
            let low = self.cost_at_tokens(lower.physical_tokens, active)?;
            let value = if lower.physical_tokens == upper.physical_tokens {
                low
            } else {
                let high = self.cost_at_tokens(upper.physical_tokens, active)?;
                let fraction = f64::from(physical_tokens - lower.physical_tokens)
                    / f64::from(upper.physical_tokens - lower.physical_tokens);
                low + fraction * (high - low)
            };
            total += value;
        }
        Ok(total / self.layers.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> TensorMoeRoutingProfile {
        // Every real token chooses experts 0..7, padding chooses 8..15.
        // Hand-written synthetic costs: 8 experts -> 1 ms, 16 -> 3 ms.
        let mut probabilities = vec![0.0; 256];
        probabilities[..8].fill(1.0);
        TensorMoeRoutingProfile {
            model: "test/glm".into(),
            backend_version: "test".into(),
            evidence_sha256: "a".repeat(64),
            intermediate_size: 128,
            layers: vec![
                RoutingLayer {
                    expert_probabilities: probabilities,
                    padding_experts: (8..16).collect()
                };
                75
            ],
            costs: vec![
                RoutingCostPoint {
                    physical_tokens: 16,
                    active_experts: 8,
                    latency_ms: 1.0,
                },
                RoutingCostPoint {
                    physical_tokens: 16,
                    active_experts: 16,
                    latency_ms: 3.0,
                },
                RoutingCostPoint {
                    physical_tokens: 32,
                    active_experts: 8,
                    latency_ms: 2.0,
                },
                RoutingCostPoint {
                    physical_tokens: 32,
                    active_experts: 16,
                    latency_ms: 4.0,
                },
            ],
        }
    }

    #[test]
    fn real_tokens_and_graph_padding_select_distinct_experts() {
        let p = profile();
        assert_eq!(p.estimate(16, 16).unwrap(), 1.0);
        assert_eq!(p.estimate(6, 16).unwrap(), 3.0);
        assert_eq!(p.estimate(6, 24).unwrap(), 3.5);
        assert_eq!(p.estimate(0, 16).unwrap(), 0.0);
        assert!(p.estimate(17, 16).is_err());
        assert!(p.estimate(6, 64).is_err());
    }

    #[test]
    fn invalid_probabilities_duplicate_grid_and_padding_fail() {
        let mut p = profile();
        p.layers[0].expert_probabilities[0] = f64::NAN;
        assert!(p.validate().is_err());
        let mut p = profile();
        p.costs.push(p.costs[0].clone());
        assert!(p.validate().is_err());
        let mut p = profile();
        p.layers[0].padding_experts[1] = p.layers[0].padding_experts[0];
        assert!(p.validate().is_err());
    }
}
