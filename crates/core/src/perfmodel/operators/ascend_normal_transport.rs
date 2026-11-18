// SPDX-FileCopyrightText: Copyright (c) 2025 sgl-project
// SPDX-License-Identifier: MIT
// Modified 2026-11-17. Independently reformulated workload partition from
// sgl-project/sgl-kernel-npu at 0234a271453bb55eb395aa471ace1e815513e92a,
// csrc/deepep/ops/op_kernel/{cam_moe_dispatch_normal,cam_moe_combine_normal_multi_round}.h.
// See THIRD_PARTY_NOTICES.md for the upstream license and attribution.

//! Ready-input EP16 Normal transport indexed by physical receive-core workload.

use crate::common::{enums::DatabaseMode, error::AicError};
use crate::operators::{PerformanceResult, Source, op::RuntimeContext};
use crate::perf_database::{PerfDatabase, ascend_qwen::AscendQwenKey};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendNormalTransportOp {
    pub name: String,
    pub scale_factor: f64,
    pub key: AscendQwenKey,
    pub reference_tokens: u32,
    pub expert_counts: Vec<u32>,
}

impl AscendNormalTransportOp {
    pub fn validate(&self) -> Result<(), AicError> {
        self.key.validate()?;
        if !self.key.is_normal_transport()
            || !self.scale_factor.is_finite()
            || self.scale_factor < 0.0
            || self.reference_tokens == 0
            || self.expert_counts.len() != 512
            || self
                .expert_counts
                .iter()
                .any(|&count| count > self.reference_tokens)
            || self
                .expert_counts
                .iter()
                .map(|&count| u64::from(count))
                .sum::<u64>()
                != u64::from(self.reference_tokens) * 10
        {
            return Err(AicError::InvalidEngineConfig(
                "Normal transport requires BF16 H4096 top10 EP16 expert counts and a finite multiplicity".into(),
            ));
        }
        Ok(())
    }

    /// Work is ordered by local expert then source rank. The 48 AIV cores
    /// receive contiguous groups of 11 or 10 of those 512 blocks. A reference
    /// contains destination-expert totals, so each source's expected share is
    /// count / 16. Retain that denominator until the native table query.
    pub fn peak_core_work_numerator(&self) -> Result<u64, AicError> {
        self.validate()?;
        let mut maximum = 0;
        for rank in 0..16 {
            for core in 0..48 {
                let begin = core * 10 + core.min(32);
                let end = (core + 1) * 10 + (core + 1).min(32);
                let work: u64 = (begin..end)
                    .map(|block| u64::from(self.expert_counts[rank * 32 + block / 16]))
                    .sum();
                maximum = maximum.max(work);
            }
        }
        Ok(maximum)
    }

    pub fn ensure_ready(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate()?;
        if db.backend != "sglang"
            || !matches!(
                db.database_mode,
                DatabaseMode::Silicon | DatabaseMode::Hybrid
            )
        {
            return Err(AicError::InvalidEngineConfig(
                "Normal transport requires measured SGLang Ascend910C costs".into(),
            ));
        }
        db.ascend_qwen.ensure_contract(
            &self.key,
            crate::perf_database::ascend_qwen::NORMAL_TRANSPORT_CONTRACT,
        )
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        self.ensure_ready(db)?;
        if ctx.num_tokens == 0 || self.scale_factor == 0.0 {
            return Ok(PerformanceResult::new(0.0, Source::Silicon));
        }
        let ratio = self.peak_core_work_numerator()? as f64 / f64::from(self.reference_tokens);
        let latency = db
            .ascend_qwen
            .query_normal_transport(&self.key, ctx.num_tokens, ratio)?;
        Ok(PerformanceResult::new(latency, Source::Silicon).scaled(self.scale_factor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn operation(counts: Vec<u32>) -> AscendNormalTransportOp {
        AscendNormalTransportOp {
            name: "normal_combine".into(),
            scale_factor: 1.0,
            key: AscendQwenKey {
                role: "normal_combine".into(),
                tp: 16,
                phase: "prefill".into(),
            },
            reference_tokens: 8192,
            expert_counts: counts,
        }
    }

    #[test]
    fn uniform_work_and_local_order_have_hand_counted_core_loads() {
        // 512 experts each receive160 assignments. Eleven expert/source blocks
        // on a core contribute11*160 before division by16 sources.
        assert_eq!(
            operation(vec![160; 512])
                .peak_core_work_numerator()
                .unwrap(),
            1760
        );
        // Ten fully selected experts keep top10 and total81920 valid. Nine
        // are in different receiver ranks. The remaining hot expert is at
        // local15 (11 source blocks/core) or16 (8+8 blocks across two cores).
        let mut counts = vec![0; 512];
        for rank in 1..10 {
            counts[rank * 32 + 16] = 8192;
        }
        counts[15] = 8192;
        assert_eq!(
            operation(counts.clone())
                .peak_core_work_numerator()
                .unwrap(),
            11 * 8192
        );
        counts.swap(15, 16);
        assert_eq!(
            operation(counts).peak_core_work_numerator().unwrap(),
            8 * 8192
        );
    }

    #[test]
    fn unsupported_or_inconsistent_workload_is_rejected() {
        let mut op = operation(vec![160; 512]);
        op.key.tp = 8;
        assert!(op.validate().is_err());
        op.key.tp = 16;
        op.expert_counts[0] -= 1;
        assert!(op.validate().is_err());
        op.expert_counts = vec![160; 511];
        assert!(op.validate().is_err());
    }
}
