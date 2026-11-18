import math
import unittest
import importlib.util
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch
from sglang_simulator.aisimulate_runner import summarize_predictor_provenance, summarize_requests


class PredictorProvenanceTests(unittest.TestCase):
    def predictor(self):
        return dict(native_sha256='native-sha', python_executable='/predictor/python',
            module_path='/predictor/aisimulate_core/__init__.py', query_count=12,
            query_wall_seconds=.5, diagnostics=dict(source='aic', provenance=dict(
                selected_estimation_mode='op_level', selected_systems_root='/costs',
                config=dict(backend_version='measured-version', estimator_config=dict(
                    op_level=dict(ascend=dict(qwen_runtime_contract='historical')))))))

    def test_executed_selection_and_full_configuration_are_preserved(self):
        predictor = self.predictor()
        entry = summarize_predictor_provenance([predictor])['agg']
        self.assertEqual(entry['estimator'], 'op_level')
        self.assertEqual(entry['timing_source'], 'aic')
        self.assertEqual(entry['data_version'], 'measured-version')
        self.assertEqual(entry['configuration'], predictor['diagnostics']['provenance']['config'])
        self.assertEqual(entry['native_sha256'], 'native-sha')
        self.assertEqual(entry['replicas'], [dict(query_count=12, query_wall_seconds=.5)])

    def test_identical_replicas_keep_separate_query_counts(self):
        first, second = self.predictor(), self.predictor()
        second['query_count'] = 15
        entry = summarize_predictor_provenance([first, second])['agg']
        self.assertEqual([replica['query_count'] for replica in entry['replicas']], [12, 15])

    def test_missing_metrics_report_unavailable(self):
        for predictors in ([], [{}], [self.predictor(), {}]):
            with self.subTest(predictors=predictors):
                self.assertIn('unavailable_reason', summarize_predictor_provenance(predictors)['agg'])

    def test_mismatched_native_builds_fail(self):
        first, second = self.predictor(), self.predictor()
        second['native_sha256'] = 'other-build'
        with self.assertRaisesRegex(ValueError, 'different effective predictor identities'):
            summarize_predictor_provenance([first, second])

    def test_pd_fleet_report_restores_the_global_time_origin(self):
        from sglang_simulator.aisimulate_runner import summarize_pd_run

        cohorts = [dict(worker_id=name, role=role, replica=replica, request_indices=indices)
                   for name, role, replica, indices in (
                       ('prefill', 'prefill', 0, [0, 1]),
                       ('decode-0', 'decode', 0, [0]),
                       ('decode-1', 'decode', 1, [1]),
                   )]
        state = dict(max_requests=2, bytes_per_request=10)
        case = dict(requests=[dict(id='a'), dict(id='b')], cohorts=cohorts,
                    world_size=1, total_devices=3, pd_bandwidth_bytes_per_second=50e9,
                    workers={role: dict(world_size=1, recurrent_state=state)
                             for role in ('prefill', 'decode')})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for cohort in cohorts:
                path = root / cohort['worker_id']
                path.mkdir()
                predictor = self.predictor()
                predictor['diagnostics']['provenance']['config']['worker_type'] = cohort['role']
                origin = 10 if cohort['worker_id'] == 'decode-1' else 0
                metrics = dict(completed=len(cohort['request_indices']), aisimulate=predictor,
                               request_time_origin_seconds=origin,
                               recurrent_state_pools={'0': dict(slots=2, available_slots=2, bytes_per_request=10)})
                (path / 'metrics.json').write_text(json.dumps(metrics))
                (path / 'iteration.jsonl').write_text(json.dumps(dict(
                    timing_model='aisimulate_whole_forward', forward_estimate=dict(latency_ms=1))) + '\n')
                if cohort['role'] == 'decode':
                    rid = case['requests'][cohort['request_indices'][0]]['id']
                    (path / 'request.jsonl').write_text(json.dumps(row(rid, 0, [1, 1])) + '\n')
            (root / 'admission.json').write_text(json.dumps(dict(active=[], completed=['a', 'b'], peak_active=1)))
            report = summarize_pd_run(root, case)
            self.assertEqual(report['metrics']['duration_ms'], 12000)
            self.assertAlmostEqual(report['metrics']['output_throughput_tok_s'], 4 / 12)
            self.assertEqual(report['metrics']['num_total_gpus'], 3)
            self.assertEqual(len(report['metadata']['prediction_provenance']['decode']['replicas']), 2)
            wrong = root / 'decode-1/request.jsonl'
            wrong.write_text(json.dumps(row('a', 0, [1, 1])) + '\n')
            with self.assertRaisesRegex(ValueError, 'owned by another cohort'):
                summarize_pd_run(root, case)


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

    def test_serial_scheduler_api_cost_uses_existing_host_stage(self):
        spec = self.spec()
        spec.backend_deployment.agg_engine_args['sglang'] = {'scheduler_select_time_ms': 2.0}
        case = self.compile(spec)
        self.assertEqual(case['request_timing'], {'scheduler_pre_ms': 2.0})
        self.assertTrue(case['server_args']['disable_overlap_schedule'])
        self.assertEqual(case['predictor_config'], spec.backend_deployment.forward_pass_estimators['agg'].config)

    def test_invalid_scheduler_api_cost_fails_before_launch(self):
        for value in [-1.0, float('nan'), True, '2.0']:
            spec = self.spec()
            spec.backend_deployment.agg_engine_args['sglang'] = {'scheduler_select_time_ms': value}
            with self.assertRaises(ValueError):
                self.compile(spec)

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

    def test_pd_handoff_and_attention_dp_are_advertised(self):
        from sglang_simulator.aisimulate_runner import NativeSGLangRunnerFactory
        capabilities = NativeSGLangRunnerFactory().capabilities()
        self.assertTrue(capabilities.supports_backend_topology('sglang','disagg'))
        self.assertTrue(capabilities.supports_attention_dp('disagg', 4, 4))

    def pd_spec(self):
        from aisimulate.sweeper.replay import ForwardPassEstimatorSpec

        spec = self.spec(ep=16, count=8, concurrency=4)
        roles, estimators = {}, {}
        for role in ('prefill', 'decode'):
            args = deepcopy(spec.backend_deployment.agg_engine_args)
            config = args['timing_model']['config']
            config['worker_type'] = args['worker_type'] = role
            args['sglang'] = {'recurrent_state': {'bytes_per_request': 48015360, 'max_requests': 32}}
            args.update(kv_transfer_bandwidth=50, kv_transfer_bytes_per_token=15360)
            roles[role], estimators[role] = args, ForwardPassEstimatorSpec(config=config)
        deployment = replace(spec.backend_deployment, deployment_mode='disagg', num_workers=0,
                             agg_engine_args=None, num_prefill_workers=1, num_decode_workers=1,
                             prefill_engine_args=roles['prefill'], decode_engine_args=roles['decode'],
                             forward_pass_estimators=estimators)
        return replace(spec, backend_deployment=deployment)

    def compile_pd(self, spec):
        geometry = {'bytes_per_request': 48015360, 'ssm_dtype': 'float32', 'kv_bytes_per_token': 15360}
        with patch('aisimulate.capacity.estimate_state_cache', return_value=geometry):
            return self.compile(spec)

    def test_pd_preserves_independent_role_cost_and_capacity_inputs(self):
        spec = self.pd_spec()
        spec.backend_deployment.decode_engine_args['sglang']['recurrent_state']['max_requests'] = 64
        case = self.compile_pd(spec)
        self.assertEqual(case['mode'], 'disaggregated')
        self.assertEqual(case['pd_bandwidth_bytes_per_second'], 50e9)
        self.assertEqual(case['total_devices'], 32)
        for role in ('prefill', 'decode'):
            worker = case['workers'][role]
            self.assertEqual(worker['predictor_config'], spec.backend_deployment.forward_pass_estimators[role].config)
            self.assertEqual(worker['predictor_config']['worker_type'], role)
            self.assertEqual(worker['requests'], case['requests'])
            self.assertEqual(worker['concurrency'], 4)
        self.assertEqual(case['workers']['prefill']['server_args']['max_mamba_cache_size'], 32)
        self.assertEqual(case['workers']['decode']['server_args']['max_mamba_cache_size'], 64)

    def test_pd_requires_matching_role_identity_and_explicit_transport(self):
        spec = self.pd_spec()
        spec.backend_deployment.forward_pass_estimators['decode'].config['worker_type'] = 'aggregated'
        with self.assertRaisesRegex(ValueError, 'worker_type=decode'):
            self.compile_pd(spec)
        spec = self.pd_spec()
        spec.backend_deployment.prefill_engine_args.pop('kv_transfer_bandwidth')
        with self.assertRaisesRegex(ValueError, 'positive kv_transfer_bandwidth'):
            self.compile_pd(spec)
        spec = self.pd_spec()
        spec.backend_deployment.decode_engine_args['kv_transfer_bytes_per_token'] = 1
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            self.compile_pd(spec)

    def test_pd_replica_routes_rotate_dp_inside_each_cohort(self):
        spec = self.pd_spec()
        spec = replace(spec, backend_deployment=replace(spec.backend_deployment,
                       num_prefill_workers=2, num_decode_workers=2, pd_routing='replica_round_robin'))
        case = self.compile_pd(spec)
        self.assertEqual(case['total_devices'], 64)
        self.assertEqual([cohort['worker_id'] for cohort in case['cohorts']],
                         ['prefill-0', 'prefill-1', 'decode-0', 'decode-1'])
        for index, request in enumerate(case['requests']):
            rid = str(request['id'])
            for role in ('prefill', 'decode'):
                self.assertEqual(case['request_routes'][rid][role], f'{role}-{index % 2}')
                self.assertEqual(case['request_dp_ranks'][rid][role], (index // 2) % 4)

    def test_pd_accepts_uneven_request_counts_across_native_cohorts(self):
        spec = self.pd_spec()
        spec = replace(spec, backend_deployment=replace(spec.backend_deployment, num_decode_workers=3))
        case = self.compile_pd(spec)
        self.assertEqual(case['total_devices'], 64)
        self.assertEqual([len(cohort['request_indices']) for cohort in case['cohorts']], [8, 3, 3, 2])
        self.assertEqual(case['concurrency'], 4)

    def test_pd_rejects_state_repartition(self):
        spec = self.pd_spec()
        spec.backend_deployment.forward_pass_estimators['decode'].config.update(tp=8, moe_ep_size=32)
        with self.assertRaisesRegex(ValueError, 'matching P/D tp'):
            self.compile_pd(spec)

    def test_recurrent_state_pool_is_not_silently_omitted(self):
        spec = self.spec()
        spec.backend_deployment.agg_engine_args['sglang'] = {
            'recurrent_state': {'bytes_per_request': 48015360, 'max_requests': 32}
        }
        geometry = {'bytes_per_request': 48015360, 'ssm_dtype': 'float32'}
        with patch('aisimulate.capacity.estimate_state_cache', return_value=geometry):
            case = self.compile(spec)
        self.assertEqual(case['server_args']['max_mamba_cache_size'], 32)
        self.assertEqual(case['server_args']['mamba_ssm_dtype'], 'float32')
        self.assertEqual(case['recurrent_state']['max_requests'], 32)
        geometry['bytes_per_request'] = 24007680
        with patch('aisimulate.capacity.estimate_state_cache', return_value=geometry):
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                self.compile(spec)

    def test_native_reserved_slots_fail_before_expensive_launch(self):
        spec = self.spec()
        spec.backend_deployment.agg_engine_args['max_model_len'] = 9216
        with self.assertRaisesRegex(ValueError, '9218 context/KV slots'):
            self.compile(spec)


class NativePDCleanupTests(unittest.TestCase):
    def test_runner_interrupts_paired_parent_and_reaps_its_descendants(self):
        from sglang_simulator.aisimulate_runner import NativeSGLangRunner
        import signal

        runner = NativeSGLangRunner(0)
        runner.pd_run = True
        process = runner.process = Mock(pid=12345)
        process.poll.return_value = None
        child = Mock()
        with patch('psutil.Process') as parent, patch('psutil.wait_procs', return_value=([], [])), \
                patch('os.kill') as kill, patch('os.killpg') as killpg:
            parent.return_value.children.return_value = [child]
            runner.close()
            runner.close()
        kill.assert_called_once_with(12345, signal.SIGINT)
        killpg.assert_not_called()
        process.wait.assert_called_once_with(timeout=75)
        child.terminate.assert_called_once()
        self.assertIsNone(runner.process)

    def test_pd_roles_and_manager_close_even_if_clock_rpc_fails(self):
        from sglang_simulator.native_pd_replay import close_native_pd
        import signal

        clock, manager, process = Mock(), Mock(), Mock(pid=12345)
        clock.fail.side_effect = EOFError('closed manager pipe')
        with patch('os.killpg') as kill:
            with self.assertRaises(EOFError):
                close_native_pd([SimpleNamespace(pid=None), process], manager, clock)
        self.assertEqual(kill.call_args_list, [unittest.mock.call(12345, signal.SIGTERM),
                                               unittest.mock.call(12345, signal.SIGKILL)])
        self.assertEqual(process.join.call_count, 2)
        manager.shutdown.assert_called_once()


if __name__ == '__main__':
    unittest.main()
