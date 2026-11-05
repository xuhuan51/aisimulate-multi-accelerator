# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
def get_obj_from_args(type_name: str, *args, **kwargs):
    for obj in args:
        if type_name == f"{type(obj).__module__}.{type(obj).__name__}":
            return obj
    for obj in kwargs.values():
        if type_name == f"{type(obj).__module__}.{type(obj).__name__}":
            return obj
    return None
