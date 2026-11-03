import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--ssh-socket", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-set", type=Path)
    args = parser.parse_args()
    repo = Path.cwd()
    source = Path(__file__).resolve().parent.relative_to(repo.resolve())
    manifests = ["layer-campaign-formal.json"]
    campaigns = json.loads(args.campaign_set.read_text()) if args.campaign_set else []
    if any(Path(item["campaign"]).resolve() == args.campaign.resolve() for item in campaigns):
        raise ValueError("Earlier campaigns must not duplicate the active campaign")
    campaigns.append({"campaign": str(args.campaign), "manifests": manifests})
    progress = args.campaign / "qualification-progress.json"
    if progress.exists() or args.output.exists():
        raise FileExistsError("Use a fresh qualification output and progress record")
    status = {"status": "waiting", "manifests": manifests, "publication": str(args.output),
              "production_data_modified": False, "serving_accuracy_accepted": False}

    def save():
        progress.write_text(json.dumps(status, indent=2) + "\n")

    def run_cpu(arguments, log):
        command = ["docker", "run", "--rm", "--network", "none", "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=1g,mode=1777",
                   "--cpuset-cpus", "308-309", "--cpus", "1", "-v", f"{repo}:/workspace",
                   "-v", "/data_lib/data/models/glm-q:/data_lib/data/models/glm-q:ro",
                   "-w", "/workspace", "-e", "PYTHONPATH=/workspace/python/aisimulate", "--entrypoint", "python3",
                   "aisimulate-910c:client-dispatch-20261104", *arguments]
        with log.open("w") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=180)

    save()
    try:
        while True:
            current = json.loads((args.campaign / manifests[-1]).read_text())
            if current["status"] == "completed":
                break
            if current["status"] == "failed":
                raise RuntimeError("Collection failed: " + str(current.get("error")))
            time.sleep(10)
        for attempt in range(3):
            snapshot = args.campaign / f"quality-{attempt}.json"
            inputs = args.campaign / f"quality-inputs-{attempt}.json"
            inputs.write_text(json.dumps(campaigns, indent=2) + "\n")
            arguments = [str(source / "inspect_tp32_layer_quality.py"), "--campaign-set", str(inputs), "--output", str(snapshot)]
            status.update(status="qualifying", attempt=attempt)
            save()
            run_cpu(arguments, args.campaign / f"quality-{attempt}.log")
            quality = json.loads(snapshot.read_text())
            if quality["qualified"]:
                break
            if attempt == 2:
                raise ValueError("Timing quality remains insufficient after two recorded repeats")
            plan = [[case["id"] + f"-quality{attempt + 1}", case["input_length"], case["global_requests"], True]
                    for case in quality["retry_cases"]]
            plan_path = args.campaign / f"quality-plan-{attempt + 1}.json"
            plan_path.write_text(json.dumps(plan, indent=2) + "\n")
            manifest_name = f"layer-campaign-quality{attempt + 1}.json"
            status.update(status="repeating_variance_failures", retry_cases=len(plan))
            save()
            with (args.campaign / f"quality-campaign-{attempt + 1}.log").open("w") as stream:
                subprocess.run(["python3", str(source / "run_tp32_layer_campaign.py"), "--campaign", str(args.campaign),
                                "--ssh-socket", args.ssh_socket, "--plan", str(plan_path), "--manifest", manifest_name],
                               check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=7200)
            manifests.append(manifest_name)
        selected = {(item["campaign"], item["case"]) for item in quality["selected_cases"]}
        qualified_campaigns = []
        for item in campaigns:
            campaign = Path(item["campaign"])
            qualified_manifests = []
            for name in item["manifests"]:
                original = campaign / name
                manifest = json.loads(original.read_text())
                cases = [case for case in manifest["cases"]
                         if not case["training"] or (str(campaign), case["id"]) in selected]
                if not cases:
                    continue
                destination = original.with_name(original.stem + "-qualified-populations.json")
                manifest.update(cases=cases, selection={"policy": quality["scope"], "parent": str(original),
                                "parent_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                                "quality_report": str(snapshot),
                                "quality_report_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()})
                destination.write_text(json.dumps(manifest, indent=2) + "\n")
                qualified_manifests.append(destination.name)
            if qualified_manifests:
                qualified_campaigns.append({"campaign": str(campaign), "manifests": qualified_manifests})
        campaign_set = args.campaign / "qualified-campaign-set.json"
        campaign_set.write_text(json.dumps(qualified_campaigns, indent=2) + "\n")
        status["status"] = "exporting"
        save()
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        run_cpu(["-m", "collector.sglang.ascend_layer_population_export", "--campaign-set", str(campaign_set),
                 "--model-config", "/data_lib/data/models/glm-q/config.json", "--output", str(args.output),
                 "--collector-ref", revision], args.campaign / "qualified-export.log")
        provenance_path = args.output / "metadata/provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["population_quality_selection"] = {"policy": quality["scope"], "report": str(snapshot),
            "report_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(), "rejected": quality["rejected"],
            "all_campaigns": campaigns, "selected_cases": quality["selected_cases"]}
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        (args.campaign / "node0/layer-probe-control.json").unlink(missing_ok=True)
        remote = json.loads((args.campaign / "node1/launch.json").read_text())["remote_root"]
        subprocess.run(["ssh", "-S", args.ssh_socket, "-o", "BatchMode=yes", "youxingye@npu1-25",
                        shlex.join(["rm", "-f", remote + "/layer-probe-control.json"])], check=True, timeout=15)
        status.update(status="qualified", accepted_cells=quality["accepted_cells"], probes_disarmed=True)
    except Exception as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
