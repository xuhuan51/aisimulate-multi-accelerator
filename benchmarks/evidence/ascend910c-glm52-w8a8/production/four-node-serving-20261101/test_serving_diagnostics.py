import copy
import unittest

from analyze_serving_diagnostics import analyze_records


def metric(name, value, rank=0, **labels):
    return {"name": "sglang:" + name, "value": value, "labels": {"dp_rank": "0", "tp_rank": str(rank), **labels}}


def record(timestamp, rows, measured=True):
    return {"time": timestamp, "measured_approx": measured, "endpoints": [{"replica": "D0", "metrics": rows}]}


class DiagnosticsTests(unittest.TestCase):
    def test_tp_duplicate_queue_and_ratio_gauges(self):
        rows = [metric("num_decode_prealloc_queue_reqs", 3, rank=rank) for rank in range(4)]
        rows += [metric("token_usage", 0.75, rank=rank) for rank in range(4)]
        result = analyze_records([record(1, rows)])
        gauges = {row["metric"]: row for row in result["gauges"]}
        self.assertEqual(gauges["sglang:num_decode_prealloc_queue_reqs"]["peak"], 3)
        self.assertEqual(gauges["sglang:token_usage"]["rank_peak"], 0.75)
        self.assertEqual(gauges["sglang:token_usage"]["dp_rank_counts"], [1])

    def test_warmup_and_failed_endpoint_samples_excluded(self):
        bad = record(3, [metric("num_decode_prealloc_queue_reqs", 40)])
        bad["endpoints"].append({"replica": "P0", "error": "timeout"})
        result = analyze_records([
            record(1, [metric("num_decode_prealloc_queue_reqs", 80)], measured=False),
            record(2, [metric("num_decode_prealloc_queue_reqs", 2)]), bad,
            record(4, [metric("num_decode_prealloc_queue_reqs", 4)]),
        ])
        self.assertEqual(result["gauges"][0]["peak"], 4)
        self.assertEqual(result["gauges"][0]["mean"], 3)
        self.assertEqual(result["window"]["excluded_endpoint_error_samples"], 1)
        self.assertEqual(result["window"]["start"], 2)

    def test_stage_delta_and_histogram_weighted_rank_observations(self):
        before = [metric("scheduler_stage_seconds_total", 100, category="run_batch"), metric("scheduler_stage_seconds_total", 20, category="other")]
        after = [metric("scheduler_stage_seconds_total", 109, category="run_batch"), metric("scheduler_stage_seconds_total", 21, category="other")]
        before += [metric("kv_transfer_latency_ms_sum", 50), metric("kv_transfer_latency_ms_count", 2)]
        after += [metric("kv_transfer_latency_ms_sum", 110), metric("kv_transfer_latency_ms_count", 5)]
        result = analyze_records([record(1, before), record(11, after)])
        self.assertEqual(result["scheduler"]["decode"]["run_batch_percent"], 90)
        self.assertEqual(result["histograms"][0]["mean"], 20)
        self.assertEqual(result["histograms"][0]["rank_observation_count"], 3)

    def test_reset_even_when_final_counter_exceeds_start(self):
        records = [record(timestamp, [metric("scheduler_stage_seconds_total", value, category="run_batch")]) for timestamp, value in [(1, 100), (2, 1), (3, 200)]]
        result = analyze_records(records)
        self.assertEqual(result["counter_quality"]["reset_series"], 1)
        self.assertEqual(result["scheduler"], {})

    def test_new_histogram_and_missing_stage_are_not_assumed_zero(self):
        before = record(1, [metric("scheduler_stage_seconds_total", 10, category="run_batch")])
        after = copy.deepcopy(before)
        after["time"] = 10
        after["endpoints"][0]["metrics"].extend([
            metric("scheduler_stage_seconds_total", 5, category="other"),
            metric("kv_transfer_latency_ms_sum", 80), metric("kv_transfer_latency_ms_count", 2),
        ])
        result = analyze_records([before, after])
        self.assertEqual(result["scheduler"], {})
        self.assertEqual(result["histograms"], [])
        self.assertEqual(result["counter_quality"]["missing_boundary_series"], 3)
        self.assertEqual(analyze_records([record(1, [])])["scheduler"], {})


if __name__ == "__main__":
    unittest.main()
