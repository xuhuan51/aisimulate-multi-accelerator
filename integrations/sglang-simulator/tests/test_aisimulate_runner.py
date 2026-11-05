# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
import math
import unittest
import importlib.util
from unittest.mock import patch
from sglang_simulator.aisimulate_runner import summarize_requests


def row(rid, created, latencies):
    return dict(rid=rid, created_time=created,
                last_event_time=created+sum(latencies),
                gen_token_latencies=latencies, input_length=8192,
                output_length=len(latencies))


class MetricContractTests(unittest.TestCase):
    def test_parallel_workers_use_makespan_not_sum_of_durations(self):
        rows = [row('a', 0, [1, .1, .1]), row('b', 0, [2, .2, .2])]
        m = summarize_requests(rows, devices=32, expected=2)
        self.assertAlmostEqual(m['duration_ms'], 2400)
        self.assertAlmostEqual(m['output_throughput_tok_s'], 2.5)
        self.assertAlmostEqual(m['mean_ttft_ms'], 1500)
        self.assertAlmostEqual(m['mean_tpot_ms'], 150)
        self.assertAlmostEqual(m['gpu_hours'], 32*2.4/3600)

    def test_single_output_has_no_tpot_sample(self):
        m = summarize_requests([row('a', 10, [2])], devices=16, expected=1)
        self.assertEqual(m['num_tpot_samples'], 0)
        self.assertEqual(m['mean_tpot_ms'], 0)
        self.assertEqual(m['duration_ms'], 2000)

    def test_missing_requests_fail_instead_of_inflating_throughput(self):
        with self.assertRaisesRegex(ValueError, 'every distinct request'):
            summarize_requests([row('a', 0, [1])], devices=16, expected=2)

    def test_duplicate_ids_fail(self):
        with self.assertRaisesRegex(ValueError, 'every distinct request'):
            summarize_requests([row('a', 0, [1])]*2, devices=32, expected=2)

    def test_invalid_latencies_fail(self):
        for value in (-1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Invalid native'):
                summarize_requests([row('a', 0, [value])], devices=16, expected=1)

    def test_incomplete_token_stream_fails(self):
        r = row('a', 0, [1])
        r['output_length'] = 1024
        with self.assertRaisesRegex(ValueError, 'Incomplete native'):
            summarize_requests([r], devices=16, expected=1)


@unittest.skipUnless(importlib.util.find_spec('aisimulate'), 'Requires AISimulate contract package')
class NativeContractTests(unittest.TestCase):
    def spec(self, replicas=1, concurrency=32, count=64, ep=1):
        from aisimulate.sweeper.replay import BackendDeploymentSpec, ForwardPassEstimatorSpec, ReplaySpec
        config = dict(model='/checkpoint',system='ascend910c',backend='sglang',backend_version='0.5.22',
                      tp=4,attention_dp=4,moe_tp_size=16//ep,moe_ep_size=ep,pp=1,
                      worker_type='aggregated',nextn=0,gemm_quant_mode='w8a8',
                      moe_quant_mode='w8a8_dynamic',kvcache_quant_mode='bfloat16',
                      fmha_quant_mode='bfloat16',comm_quant_mode='half',
                      estimation_mode='op_level',fallback_policy='deny',
                      estimator_config={'op_level':{'ascend':{'shape_policy':'anchored'}}})
        args = dict(worker_type='aggregated',engine_type='sglang',tensor_parallel_size=4,dp_size=4,
                    max_num_batched_tokens=4096,max_num_seqs=32,block_size=128,num_gpu_blocks=256,
                    enable_prefix_caching=False,max_model_len=9218,
                    timing_model={'type':'external','provider':'aic','config':config})
        deployment = BackendDeploymentSpec(deployment_mode='agg',backend='sglang',backend_version='0.5.22',
            num_workers=replicas,agg_engine_args=args,
            forward_pass_estimators={'agg':ForwardPassEstimatorSpec(config=config)})
        return ReplaySpec(backend_deployment=deployment,workload={'isl':8192,'osl':1024,'request_count':count},goal={},concurrency=concurrency)

    def compile(self, spec):
        from sglang_simulator.aisimulate_runner import compile_case
        with patch('aisimulate.capacity.materialize_aic_num_gpu_blocks', side_effect=lambda x:x), patch('aisimulate_core.sdk.perf_database.load_system_spec', return_value={'gpu':{'mem_capacity':65787658240},'node':{'num_gpus_per_node':16}}):
            return compile_case(spec)

    def test_canonical_identity_and_rank_local_capacity_preserved(self):
        spec = self.spec()
        case = self.compile(spec)
        self.assertEqual(case['predictor_config'], spec.backend_deployment.forward_pass_estimators['agg'].config)
        self.assertEqual(case['world_size'], 16)
        self.assertEqual(case['server_args']['max_total_tokens'], 32768)
        self.assertEqual(case['server_args']['dp_size'], 4)
        self.assertEqual(len(case['requests']), 64)

    def test_symmetric_two_replica_workload(self):
        case = self.compile(self.spec(replicas=2))
        self.assertEqual(case['routing'], 'symmetric_identical_replicas')
        self.assertEqual(case['concurrency'], 32)
        self.assertEqual(case['replicas'], 2)

    def test_uneven_closed_loop_is_not_silently_partitioned(self):
        with self.assertRaisesRegex(ValueError, 'divide evenly'):
            self.compile(self.spec(replicas=2, concurrency=31))

    def test_full_ep_selects_native_deepep_metadata_path(self):
        case = self.compile(self.spec(ep=16))
        self.assertEqual(case['server_args']['moe_a2a_backend'], 'deepep')
        self.assertEqual(case['server_args']['ep_size'], 16)
        self.assertEqual(case['predictor_config']['moe_tp_size'], 1)

    def test_unqualified_hybrid_ep_is_not_silently_replaced(self):
        with self.assertRaisesRegex(ValueError, 'hybrid expert parallelism'):
            self.compile(self.spec(ep=2))

    def test_pd_not_advertised_until_handoff_exists(self):
        from sglang_simulator.aisimulate_runner import NativeSGLangRunnerFactory
        self.assertFalse(NativeSGLangRunnerFactory().capabilities().supports_backend_topology('sglang','disagg'))

    def test_native_reserved_slots_fail_before_expensive_launch(self):
        spec = self.spec()
        spec.backend_deployment.agg_engine_args['max_model_len'] = 9216
        with self.assertRaisesRegex(ValueError, '9218 context/KV slots'):
            self.compile(spec)


if __name__ == '__main__':
    unittest.main()
