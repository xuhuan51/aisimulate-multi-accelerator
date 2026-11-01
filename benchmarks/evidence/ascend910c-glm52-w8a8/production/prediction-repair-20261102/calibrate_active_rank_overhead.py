import argparse
import gzip
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_record(name):
    path = ROOT / name
    raw = path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    summaries = []
    for topology, source_name, base_name, fit_case in (
        ("tp16-adp4", "paired-forward-analysis.json", "eager-overhead-tp16-adp4.json", "validation-c4"),
        ("tp32-adp32", "tp32-paired-forward-analysis.json", "eager-overhead-tp32-adp32.json", "validation-c8"),
    ):
        data, source_hash = read_record(source_name)
        base, base_hash = read_record(base_name)
        cases = {case["case"]: case["decode_rows"] for case in data["cases"]} if "cases" in data else data["forwards"]
        fitting = cases[fit_case]
        if len(fitting) < 100:
            raise ValueError("At least100 paired steady forwards required for the new rank term")
        coefficients = []
        fit_rows = []
        for row in fitting:
            active = sum(rank["scheduled_requests"]["num_decode_requests"] > 0 for rank in row["metrics"])
            if active <= 1:
                raise ValueError("The extra-rank calibration requires several active ranks")
            coefficients.append((row["residual_ms"] - base["decode_ms"]) / (active - 1))
            fit_rows.append({"metrics": row["metrics"], "measured_ms": row["measured_ms"], "residual_ms": row["residual_ms"], "active_ranks": active})
        slope = statistics.median(coefficients)
        if slope < 0:
            raise ValueError("Nonnegative extra-rank model does not explain these observations")
        identity = hashlib.sha256(json.dumps({"base_sha256": base_hash, "fit_rows": fit_rows}, sort_keys=True).encode()).hexdigest()
        profile = {**base, "calibration_id": f"paired-active-{topology}-{identity[:16]}", "decode_per_extra_active_rank_ms": slope}
        path = ROOT / f"eager-overhead-active-{topology}.json"
        encoded = json.dumps(profile, indent=2) + "\n"
        if path.exists() and path.read_text() != encoded:
            raise ValueError("Preserve the existing calibration identity")
        path.write_text(encoded)
        report = {
            "topology": topology,
            "base_profile": base,
            "profile": profile,
            "source_analysis_sha256": source_hash,
            "source_analysis": source_name,
            "fit_case": fit_case,
            "fit_forwards": len(fitting),
            "fit_rows_sha256": identity,
            "rule": "Retain the prior C1 base. Fit median((measured minus raw operator estimate minus base)/(real active ranks-1)) on only the named paired-forward case. This previously held-out case is now training, not independent validation. Original23 serving points and all new PD campaigns remain held out.",
            "cases": [],
        }
        if not args.prepare_only:
            from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel
            from calibrate_eager_overhead import configuration
            from calibrate_tp32_overhead import model_config

            configure = configuration if topology == "tp16-adp4" else model_config
            before = RustForwardPassPerfModel.best_available(configure(base))
            after = RustForwardPassPerfModel.best_available(configure(profile))
            for name, rows in cases.items():
                comparisons = []
                for row in rows:
                    old = before.estimate_forward_pass_time_ms(row["metrics"])
                    new = after.estimate_forward_pass_time_ms(row["metrics"])
                    comparisons.append({"measured_ms": row["measured_ms"], "old_ms": old, "new_ms": new, "metrics": row["metrics"]})
                summary = {
                    "case": name,
                    "used_for_rank_fit": name == fit_case,
                    "used_for_base_fit": topology == "tp32-adp32" and name == "calibration-c1",
                    "forwards": len(rows),
                    "before_mape": statistics.mean(abs(row["old_ms"] / row["measured_ms"] - 1) * 100 for row in comparisons),
                    "after_mape": statistics.mean(abs(row["new_ms"] / row["measured_ms"] - 1) * 100 for row in comparisons),
                    "rows": comparisons,
                }
                report["cases"].append(summary)
                print(json.dumps({"topology": topology, **{key: value for key, value in summary.items() if key != "rows"}}), flush=True)
        summaries.append(report)
        print(json.dumps({"topology": topology, "base_ms": base["decode_ms"], "extra_active_rank_ms": slope}), flush=True)
    if not args.prepare_only:
        (ROOT / "active-rank-forward-validation.json").write_text(json.dumps({"topologies": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
