# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
"""Private JSON-lines worker; run by filename in the AISimulate interpreter.

Keep this module stdlib-only until initialization. It must not import SGLang,
the simulator package, or legacy AIConfigurator.
"""

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def main():
    # Reserve the original stdout for the protocol. Also redirect native-library
    # stdout, so diagnostics cannot corrupt JSON responses.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    predictor = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request["op"] == "init":
                    if predictor is not None:
                        raise ValueError("AISimulate worker is already initialized")
                    # This file runs in a separately chosen interpreter.
                    if sys.version_info < (3, 11):  # noqa: UP036
                        raise RuntimeError(
                            "AISimulate requires Python >=3.11; set predictor.python_executable "
                            "to an interpreter with AISimulate installed"
                        )
                    import aisimulate._runtime as native
                    import aisimulate_core
                    from aisimulate_core.sdk import RustForwardPassPerfModel

                    predictor = RustForwardPassPerfModel.best_available(
                        request["config"]
                    )
                    try:
                        version = importlib.metadata.version("aisimulate")
                    except importlib.metadata.PackageNotFoundError:
                        version = "unknown"
                    with Path(native.__file__).open("rb") as library:
                        native_sha256 = hashlib.file_digest(
                            library, "sha256"
                        ).hexdigest()
                    result = {
                        "version": version,
                        "python_executable": sys.executable,
                        "python_version": sys.version.split()[0],
                        "module_path": aisimulate_core.__file__,
                        "native_path": native.__file__,
                        "native_sha256": native_sha256,
                        "diagnostics": predictor.diagnostics(),
                    }
                elif request["op"] == "estimate" and predictor is not None:
                    result = predictor.estimate_forward_pass_detailed(
                        request["metrics"]
                    )
                else:
                    raise ValueError(
                        "Unknown operation or uninitialized AISimulate worker"
                    )
                response = {"result": result}
                encoded = json.dumps(response, allow_nan=False)
            # Serialize dependency errors across the private RPC boundary.
            except Exception as error:  # noqa: BLE001
                encoded = json.dumps({"error": f"{type(error).__name__}: {error}"})
            protocol.write(encoded + "\n")
    finally:
        if predictor is not None:
            predictor.close()
        protocol.close()


if __name__ == "__main__":
    main()
