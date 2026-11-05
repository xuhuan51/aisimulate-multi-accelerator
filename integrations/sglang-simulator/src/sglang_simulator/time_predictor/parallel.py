"""Price a distributed iteration once, with one batch per attention DP group."""

import math


def predict_parallel_step(
    predictor, batches, config, padded_tokens=None, padding_mode=None
):
    if hasattr(predictor, "predict_parallel"):
        result = predictor.predict_parallel(
            batches, padded_tokens=padded_tokens, padding_mode=padding_mode
        )
    else:
        durations = [
            float(predictor.predict_infer_time(b)) if b.reqs else 0.0 for b in batches
        ]
        # Replay/ML describe complete forwards, including their collectives.
        # Their synchronous group completion is the slowest supplied forward.
        result = {
            "durations": [max(durations)] * len(batches),
            "timing_model": "whole_forward_max",
            "local_estimates": durations,
        }
    if len(result["durations"]) != len(batches):
        raise ValueError("Predictor must return one duration per attention DP group")
    if any(not math.isfinite(t) or t < 0 for t in result["durations"]):
        raise ValueError("Predictor returned a negative or non-finite duration")
    result["global_tokens"] = sum(r.extend_length for b in batches for r in b.reqs)
    result["local_tokens"] = [sum(r.extend_length for r in b.reqs) for b in batches]
    result["padded_tokens"] = padded_tokens
    result["padding_mode"] = padding_mode
    return result
