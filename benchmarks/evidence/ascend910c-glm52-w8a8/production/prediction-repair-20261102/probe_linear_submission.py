import hashlib
import importlib
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    import torch
    import torch_npu

    importlib.import_module("sglang.srt.layers.quantization")
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.layers.quantization.modelslim.schemes.modelslim_w8a8_int8 import ModelSlimW8A8Int8

    world = int(os.environ.get("WORLD_SIZE", "1"))
    device_index = int(os.environ.get("LOCAL_RANK", "15"))
    destination = Path("/campaign") / (f"linear-submission-rank{device_index}.json" if world > 1 else "linear-submission-probe.json")
    if destination.exists():
        raise FileExistsError(destination)
    first_cpu = device_index * 20
    os.sched_setaffinity(0, set(range(first_cpu, first_cpu + 20)) | set(range(first_cpu + 320, first_cpu + 340)))
    init_npu_backend()
    torch.npu.set_device(device_index)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(910)
    if world > 1:
        torch.distributed.init_process_group("gloo")
    result = {"scope": "Diagnostic of benchmark retention, module dispatch, and pipelining; not published performance data",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "device": device_index,
              "concurrent_workers": world, "cases": []}
    for tokens, width, inner in [(4, 2112, 6144), (4, 18432, 1536), (4, 6144, 4096)]:
        layers = []
        scheme = ModelSlimW8A8Int8({"linear.weight": "W8A8"}, "linear")
        for _ in range(4):
            layer = RowParallelLinear.__new__(RowParallelLinear)
            torch.nn.Module.__init__(layer)
            layer.tp_group = SimpleNamespace(rank_in_group=0)
            with torch.device("npu"):
                scheme.create_weights(layer, inner, [width], inner, width, torch.bfloat16)
            layer.weight.data.random_(-3, 4)
            layer.weight_scale.data.fill_(0.125)
            layer.weight_offset.data.zero_()
            layer.input_scale.data.fill_(1 / 32)
            layer.input_offset.data.zero_()
            layer.deq_scale.data.fill_(1 / 256)
            layer.quant_bias.data.zero_()
            scheme.process_weights_after_loading(layer)
            layers.append(layer)
        inputs = torch.randn(tokens, inner, device="npu", dtype=torch.bfloat16)

        class ModuleBoundary(torch.nn.Module):
            def forward(self, index):
                return scheme.apply_weights(layers[index % 4], inputs)

        module = ModuleBoundary()
        reference = scheme.apply_weights(layers[0], inputs).clone()
        for name in ("retained", "released", "module_released", "per_call_sync"):
            samples = []
            for repeat in range(12):
                retained = []
                torch.npu.synchronize()
                if world > 1:
                    torch.distributed.barrier()
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                cpu_start = time.thread_time_ns()
                started = time.perf_counter_ns()
                begin.record()
                for index in range(64):
                    output = module(index) if name == "module_released" else scheme.apply_weights(layers[index % 4], inputs)
                    if name == "retained":
                        retained.append(output)
                    del output
                    if name == "per_call_sync":
                        torch.npu.synchronize()
                end.record()
                submitted = time.perf_counter_ns()
                cpu_ns = time.thread_time_ns() - cpu_start
                end.synchronize()
                if repeat >= 3:
                    samples.append({"device_ms": begin.elapsed_time(end) / 64,
                                    "host_submit_ms": (submitted - started) / 64e6,
                                    "host_cpu_ms": cpu_ns / 64e6})
                del retained
            observed = scheme.apply_weights(layers[0], inputs)
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
            row = {"mode": name, "m": tokens, "n": width, "k": inner, "samples": samples,
                   "medians": {key: statistics.median(sample[key] for sample in samples) for key in samples[0]}}
            result["cases"].append(row)
            destination.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({key: value for key, value in row.items() if key != "samples"}), flush=True)
        del layers, layer, reference, observed, inputs, module
        torch.npu.empty_cache()
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
