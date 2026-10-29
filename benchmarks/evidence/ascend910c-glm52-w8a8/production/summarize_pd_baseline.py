import argparse
import csv
import hashlib
import json
import statistics
import tarfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics = ("mean_ttft_s", "mean_tpot_ms", "output_tokens_per_s")
    records, audits, datasets = [], {}, {}
    archive = args.output_dir / "replay-evidence.tar.gz"
    members = []
    with tarfile.open(archive, "w:gz") as bundle:
        for specification in args.replay:
            label, directory = specification.split("=", 1)
            folder = Path(directory)
            summary = json.loads((folder / "summary.json").read_text())
            datasets[label] = sorted(summary["records"], key=lambda row: row["name"])
            sources, readmissions = Counter(), {}
            paths = [folder / "summary.json", *sorted((folder / "configs").glob("*.yaml"))]
            for row in datasets[label]:
                for phase in (row.get("performance_diagnostics") or {}).get("phases", []):
                    sources.update(operation["source"] for operation in phase["operations"])
                requests = folder / row["name"] / "requests.jsonl"
                if requests.is_file():
                    entries = [json.loads(line) for line in requests.read_text().splitlines()]
                    readmissions[row["name"]] = sum(entry["readmission_count"] for entry in entries)
                    paths.append(requests)
                prediction = folder / row["name"] / "prediction.json"
                if prediction.is_file():
                    paths.append(prediction)
                for metric in metrics:
                    records.append(
                        {
                            "series": label,
                            "case": row["name"],
                            "concurrency": row["measured"]["concurrency"],
                            "requests": row["measured"]["completed_requests"],
                            "metric": metric,
                            "measured": row["measured"][metric],
                            "predicted": row.get("predicted", {}).get(metric),
                            "relative_error": row.get("relative_error", {}).get(metric),
                        }
                    )
            for path in paths:
                relative = str(Path(label) / path.relative_to(folder))
                bundle.add(path, arcname=relative, recursive=False)
                members.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            complete = [row for row in datasets[label] if row.get("returncode") == 0]
            audits[label] = {
                "completed": len(complete),
                "attempted": len(datasets[label]),
                "source_categories": dict(sources),
                "readmissions": readmissions,
                "native_sha256": summary["native_sha256"],
                "tables": summary["tables"],
                "decode_reserved_tokens": summary.get("decode_reserved_tokens", 0),
                "median_absolute_relative_error": {
                    metric: statistics.median(abs(row["relative_error"][metric]) for row in complete)
                    for metric in metrics
                },
            }
    with (args.output_dir / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    labels = ("Mean TTFT (s)", "Mean TPOT (ms)", "Output tokens / s")
    for row_index, workload in enumerate(("w1", "w2")):
        for column, metric in enumerate(metrics):
            panel = axes[row_index, column]
            first = True
            for name, dataset in datasets.items():
                selected = sorted(
                    (row for row in dataset if row["name"].startswith(f"1p1d-{workload}-")),
                    key=lambda row: row["measured"]["concurrency"],
                )
                if first:
                    panel.plot(
                        [row["measured"]["concurrency"] for row in selected],
                        [row["measured"][metric] for row in selected],
                        "o-",
                        label="DOCX measured",
                    )
                    first = False
                panel.plot(
                    [row["measured"]["concurrency"] for row in selected if row.get("returncode") == 0],
                    [row["predicted"][metric] for row in selected if row.get("returncode") == 0],
                    ".--",
                    label=name,
                )
            panel.set_title(f"1P1D / {'8K' if workload == 'w1' else '32K'} input / 1K output")
            panel.set_xlabel("Client concurrency")
            panel.set_ylabel(labels[column])
            panel.grid(alpha=0.25)
            panel.legend(fontsize=8)
    figure.savefig(args.output_dir / "comparison.png", dpi=160)
    plt.close(figure)
    report = {
        "scope": "Finite-request DOCX comparison; fixed label lengths and declared-config alignment only.",
        "series": audits,
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "members": members,
        "source_interpretation": "Silicon includes interpolation within measured ranges, "
        "not only exact measured shapes. "
        "Composite operation categories do not expose every internal query coordinate.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(audits))


if __name__ == "__main__":
    main()
