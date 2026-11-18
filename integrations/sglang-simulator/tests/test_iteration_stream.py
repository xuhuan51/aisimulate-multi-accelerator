from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from sglang_simulator.simulation.parallel.coordinator import Coordinator


@dataclass
class Request:
    extend_length: int = 1


@dataclass
class Batch:
    reqs: list


class IterationStreamTests(TestCase):
    def test_iteration_is_readable_before_profile_finishes(self):
        with TemporaryDirectory() as path:
            coordinator = Coordinator(2, 2, path)
            coordinator._record_iteration(dict(step=0, batch=Batch([Request()])))
            row = json.loads(Path(path, 'iteration.jsonl').read_text())
            self.assertEqual(row, dict(step=0, batch=dict(reqs=[dict(extend_length=1)])))
            self.assertEqual(coordinator.iteration_count, 1)
            self.assertFalse(hasattr(coordinator, 'iterations'))
            coordinator.close()

    def test_forward_timing_and_step_order_are_preserved(self):
        with TemporaryDirectory() as path:
            coordinator = Coordinator(2, 2, path)
            coordinator.config = object()
            inputs = [dict(batch=Batch([Request()])) for _ in range(2)]
            estimate = dict(durations=[.001, .002], timing_model='test')
            with patch('sglang_simulator.time_predictor.parallel.predict_parallel_step', return_value=estimate):
                first = coordinator._step(inputs)
                second = coordinator._step(inputs)
            self.assertEqual([entry['clock'] for entry in first], [.001, .002])
            self.assertEqual([entry['clock'] for entry in second], [.003, .004])
            rows = [json.loads(line) for line in Path(path, 'iteration.jsonl').read_text().splitlines()]
            self.assertEqual([row['step'] for row in rows], [0, 1])
            self.assertEqual([row['end'] for row in rows], [.002, .004])
            coordinator.close()

    def test_profile_reset_starts_a_fresh_step_sequence_and_file(self):
        with TemporaryDirectory() as path:
            coordinator = Coordinator(2, 2, path)
            coordinator._record_iteration(dict(step=0, run='first'))
            coordinator._profile([dict(stats=[]), dict(stats=[])])
            self.assertIsNone(coordinator.iteration_stream)
            self.assertEqual(coordinator.iteration_count, 0)
            coordinator._record_iteration(dict(step=0, run='second'))
            coordinator.close()
            rows = [json.loads(line) for line in Path(path, 'iteration.jsonl').read_text().splitlines()]
            self.assertEqual(rows, [dict(step=0, run='second')])

    def test_shutdown_keeps_partial_evidence_without_completion_metrics(self):
        with TemporaryDirectory() as path:
            coordinator = Coordinator(2, 2, path)
            for step in range(200):
                coordinator._record_iteration(dict(step=step, latency_ms=.25))
            coordinator.close()
            rows = Path(path, 'iteration.jsonl').read_text().splitlines()
            self.assertEqual(len(rows), 200)
            self.assertEqual(json.loads(rows[-1])['step'], 199)
            self.assertFalse(Path(path, 'metrics.json').exists())
