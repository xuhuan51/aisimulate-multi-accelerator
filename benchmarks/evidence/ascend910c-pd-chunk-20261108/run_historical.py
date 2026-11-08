"""Run frozen historical configurations through the ordinary prediction CLI."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import time


parser = argparse.ArgumentParser()
parser.add_argument('--campaign', required=True)
parser.add_argument('--all', action='store_true')
parser.add_argument('--prepared', default='prepared.json')
args = parser.parse_args()
root = Path('/opt/pd-chunk-20261108')
prepared = json.loads((root/args.prepared).read_text())
selected = [job for job in prepared['jobs'] if args.all or
            (job['group'] in ('B','C','D') and job['concurrency']==16)]
directory = root/args.campaign
directory.mkdir(exist_ok=False)
(directory/'runner.py').write_bytes(Path(__file__).read_bytes())
results = []


def run(job):
    output = directory/job['job']
    command = ['/opt/aisimulate-venv/bin/aisimulate', 'predict', '-c', job['config'],
               '--output-dir', str(output), '--format', 'json']
    started = time.time()
    with (directory/(job['job']+'.log')).open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=os.environ.copy())
        try:
            code = process.wait(timeout=360)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 'timeout'
    path = output/'prediction.json'
    metrics = json.loads(path.read_text()) if path.exists() else None
    return dict(**job, command=command, exit_code=code, metrics=metrics,
                status='passed' if code==0 and metrics and metrics['completed_requests']==job['requests'] else 'failed',
                wall_seconds=time.time()-started)


with ThreadPoolExecutor(max_workers=2) as pool:
    futures = {pool.submit(run,job):job for job in selected}
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        temporary = directory/'status.tmp'
        temporary.write_text(json.dumps(dict(total=len(selected),completed=len(results),results=results),indent=2))
        temporary.replace(directory/'status.json')
        print(json.dumps({key:result[key] for key in ('job','group','mode','concurrency','status','exit_code','wall_seconds')}),flush=True)
