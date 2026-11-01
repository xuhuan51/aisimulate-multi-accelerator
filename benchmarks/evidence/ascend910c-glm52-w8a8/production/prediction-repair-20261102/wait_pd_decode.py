import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from start_pd_telemetry import IMAGE


root = Path(__file__).resolve().parent
deadline = time.monotonic() + 1800
while time.monotonic() < deadline:
    image_ready = subprocess.run(["docker", "image", "inspect", IMAGE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    store_ready = False
    try:
        socket.create_connection(("10.120.72.20", 19707), timeout=2).close()
        store_ready = True
    except OSError:
        pass
    (root / "decode-launch-state.json").write_text(json.dumps({"time": time.time(), "image_ready": image_ready, "store_ready": store_ready}) + "\n")
    if image_ready and store_ready:
        subprocess.run([sys.executable, str(root / "start_pd_telemetry.py"), "--role", "decode", "--output", str(root / "decode")], check=True)
        break
    time.sleep(10)
else:
    raise TimeoutError("Image or MemFabric Store did not become ready")
