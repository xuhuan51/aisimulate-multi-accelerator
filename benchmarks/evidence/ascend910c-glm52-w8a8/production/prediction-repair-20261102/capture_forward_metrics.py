import argparse
import json
import signal
import time
from pathlib import Path

import msgspec
import zmq
from sglang.srt.observability.forward_pass_metrics import decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=3600)
    args = parser.parse_args()
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')
    for rank in range(args.ranks):
        subscriber.connect(f'ipc:///campaign/forward-metrics.{rank}')
    active = True

    def stop(signum, frame):
        nonlocal active
        active = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    deadline = time.monotonic() + args.seconds
    count = 0
    with args.output.open('x', buffering=1) as output:
        while active and time.monotonic() < deadline:
            if not subscriber.poll(1000):
                continue
            parts = subscriber.recv_multipart()
            metrics = decode(parts[-1])
            record = {'received_monotonic': time.monotonic(), 'received_unix': time.time(), 'metrics': msgspec.to_builtins(metrics)}
            output.write(json.dumps(record, separators=(',', ':')) + '\n')
            count += 1
    subscriber.close(linger=0)
    context.term()
    print(json.dumps({'messages': count, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
