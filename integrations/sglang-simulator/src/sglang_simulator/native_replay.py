# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
"""Isolated native SGLang process entry point for the AISimulate runner."""
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys


def reserve_port_span():
    # Native DP launch uses nearby ports; avoid the ephemeral client range.
    for port in random.sample(range(12000, 24000, 128), 90):
        probes = []
        try:
            for offset in range(96):
                probe = socket.socket()
                probes.append(probe)
                probe.bind(('127.0.0.1', port+offset))
            return port
        except OSError:
            pass
        finally:
            for probe in probes:
                probe.close()
    raise RuntimeError('No free port span for native simulator')


def run_replica(case_path, replica):
    case = json.loads(case_path.read_text())
    out = case_path.parent / f'replica-{replica}'
    out.mkdir()
    identity = case['predictor_config']
    config = {
        'platform': {'accelerator': case['hardware'], 'num_device_per_node': case['devices_per_node']},
        'predictor': {'name': 'aisimulate', 'python_executable': os.environ['SGLANG_SIMULATOR_PREDICTOR_PYTHON'], 'timeout_seconds': 120, 'config': identity},
        'scheduler': {'tp_size': case['world_size'], 'dp_size': identity['attention_dp'], 'ep_size': identity['moe_ep_size'], 'backend_name': identity['backend'], 'backend_version': identity['backend_version']},
    }
    config_path = out / 'sim-config.json'
    config_path.write_text(json.dumps(config, indent=2))
    os.environ.update(
        SGLANG_SIMULATOR_CONFIG_PATH=str(config_path),
        SGLANG_SIMULATOR_OUTPUT_DIR=str(out), SGLANG_SIMULATOR_OUTPUT_MODE='OFFLINE',
        SGLANG_USE_CPU_ENGINE='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
        OMP_NUM_THREADS='1', SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE='1',
    )
    os.environ.setdefault('SGLANG_SIMULATOR_RUNTIME', 'upstream')
    os.environ.setdefault('SGLANG_SIMULATOR_LEGACY_ATTENTION_DP', '1')
    from sglang_simulator.simulation.sglang.engine import SGLangSimulationEngine
    from sglang_simulator.simulation.sglang.runtime_adapter import SGLangRuntimeAdapter, server_arg_values
    from sglang.srt.server_args import ServerArgs

    template = os.environ.get('SGLANG_SIMULATOR_NATIVE_ARGS')
    kwargs = json.loads(Path(template).read_text()) if template else {}
    fields = set(getattr(ServerArgs, '__struct_fields__', ()))
    if not fields:
        fields = {f.name for f in dataclasses.fields(ServerArgs)}
    kwargs = {k: v for k, v in kwargs.items() if k in fields}
    kwargs.update({k: v for k, v in case['server_args'].items() if v is not None})
    kwargs.update(
        load_format='dummy', device='cpu', dtype='bfloat16', quantization=None,
        sampling_backend='pytorch', skip_tokenizer_init=True,
        disable_custom_all_reduce=True, log_level='warning',
        port=reserve_port_span(), nccl_port=None, dist_init_addr=None,
    )
    if 'cuda_graph_backend_decode' in fields:
        from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
        kwargs.update(cuda_graph_backend_decode='disabled', cuda_graph_backend_prefill='disabled', cuda_graph_config=CudaGraphConfig())
    if template:
        # The template explicitly records an already resolved native launch.
        # Recompute this candidate's DP chunk from the public global input once.
        kwargs['chunked_prefill_size'] //= identity['attention_dp']
        original_resolve = ServerArgs.resolve_once
        def resolved_template(self):
            if self.device == 'cpu':
                self._resolution_finished = True
                return
            return original_resolve(self)
        ServerArgs.resolve_once = resolved_template
    unknown = set(kwargs) - fields
    if unknown:
        raise ValueError(f'Native runtime does not expose requested server arguments: {sorted(unknown)}')
    (out / 'launch-args.json').write_text(json.dumps(kwargs, indent=2, default=lambda x: dataclasses.asdict(x)))
    requests = case['requests'][replica::case['replicas']]
    concurrency = case['concurrency']
    if concurrency is not None:
        concurrency //= case['replicas']
    corpus, params, ids = [], [], []
    frozen_corpus = None
    corpus_path = os.environ.get('SGLANG_SIMULATOR_CORPUS_FILE')
    if corpus_path:
        payload = Path(corpus_path).read_bytes()
        expected_hash = os.environ.get('SGLANG_SIMULATOR_CORPUS_SHA256')
        if not expected_hash or hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError('Frozen native corpus SHA256 mismatch or missing expected hash')
        all_prompts = [json.loads(line) for line in payload.splitlines()]
        if len(all_prompts) != len(case['requests']):
            raise ValueError('Frozen native corpus request count mismatch')
        frozen_corpus = all_prompts[replica::case['replicas']]
        if any(len(prompt) != request['input_tokens'] for prompt, request in zip(frozen_corpus, requests)):
            raise ValueError('Frozen native corpus prompt length mismatch')
        (out / 'corpus-provenance.json').write_text(json.dumps({'path': corpus_path, 'sha256': expected_hash}))
    for index, request in enumerate(requests):
        rng = random.Random(910000 + replica*100000 + index)
        corpus.append(frozen_corpus[index] if frozen_corpus is not None else [rng.randrange(100, 10000) for _ in range(request['input_tokens'])])
        simulation = {'created_time_ms': request['arrival_time_ms'], 'total_request': len(requests), 'request_index': index}
        if concurrency is not None:
            simulation['max_concurrency'] = concurrency
        params.append({'max_new_tokens': request['output_tokens'], 'temperature': 0, 'ignore_eos': True, 'custom_params': {'simulation': simulation}})
        ids.append(str(request['id']))
    (out / 'workload.json').write_text(json.dumps(requests, indent=2))
    engine = SGLangSimulationEngine(**kwargs)
    try:
        effective = server_arg_values(engine.server_args)
        for name in ('page_size', 'max_total_tokens'):
            if effective[name] != case['server_args'][name]:
                raise ValueError(f'Native resolution changed {name}: {effective[name]}')
        (out / 'resolved-args.json').write_text(json.dumps(effective, indent=2))
        runtime = SGLangRuntimeAdapter.validate()
        source_root = Path(inspect.getfile(ServerArgs)).parent
        runtime['source_hashes'] = {name: hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in ('server_args.py', 'managers/scheduler.py', 'managers/schedule_policy.py')}
        runtime['predictor_backend_version'] = identity['backend_version']
        runtime['version_match'] = runtime['sglang_version'] == identity['backend_version']
        (out / 'runtime.json').write_text(json.dumps(runtime, indent=2))
        engine.start_profile()
        responses = engine.generate(input_ids=corpus, sampling_params=params, rid=ids)
        engine.stop_profile()
        if len(responses) != len(requests):
            raise RuntimeError('Native Engine returned an incomplete response list')
        steps = 0
        with (out / 'iteration.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('timing_model') != 'aisimulate_whole_forward' or row.get('forward_estimate', {}).get('latency_ms') is None:
                    raise ValueError('Native iteration did not use the canonical AISimulate predictor')
                steps += 1
        if not steps:
            raise ValueError('Native replay produced no forward iterations')
        (out / 'verification.json').write_text(json.dumps({'completed': len(responses), 'forward_iterations': steps, 'status': 'passed'}))
    finally:
        engine.shutdown()


def main():
    case_path = Path(sys.argv[1]).resolve()
    if len(sys.argv) == 3:
        run_replica(case_path, int(sys.argv[2]))
        return
    case = json.loads(case_path.read_text())
    rows, runtimes = [], []
    for replica in range(case['replicas']):
        subprocess.run([sys.executable, '-u', __file__, str(case_path), str(replica)], check=True)
        out = case_path.parent / f'replica-{replica}'
        rows.extend(json.loads(line) for line in (out/'request.jsonl').read_text().splitlines())
        runtimes.append(json.loads((out/'runtime.json').read_text()))
    from sglang_simulator.aisimulate_runner import summarize_requests
    metrics = summarize_requests(rows, devices=case['world_size']*case['replicas'], expected=len(case['requests']))
    report = {'metrics': metrics, 'metadata': {'executor': 'native_sglang_simulator', 'routing': case['routing'], 'capacity_source': case['capacity_source'], 'runtime': runtimes, 'hardware_accuracy': 'not_validated'}}
    (case_path.parent/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
