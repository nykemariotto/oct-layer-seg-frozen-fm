"""Peak GPU memory and energy per B-scan of the five models (Table VI).

Same forward pass and setting as efficiency.py: 1x3x224x224, batch 1, FP32 weights with FP16 autocast.
  memory   peak memory allocated by PyTorch on the GPU (max_memory_allocated) over 15 warm-up + 60 passes,
           minus what was allocated before the model was loaded (at most the cuBLAS workspace, < 9 MiB;
           a larger leftover stops the run). Also measured with the model cast to FP16 (.half()).
  latency  median of the 60 passes; checked against efficiency.json (within 15%) when that file exists.
  energy   GPU board power (nvidia-smi power.draw.instant) sampled every 100 ms: 10 s idle with the model
           loaded, then 20 s of continuous inference; 3 repetitions, median.
           gross energy per B-scan = P_load * T / N; net = (P_load - P_idle) * T / N.
           Board power only, not the whole computer.
Close other programs that use the GPU before running. Values in MiB (2^20 bytes) and mJ.
Writes: <RESULTS_DIR>/memory_energy.json
"""
import gc
import json
import os
import subprocess
import threading
import time

import numpy as np
import torch

from config import RESULTS_DIR
from efficiency import make, params_M, MODELS

N_WARM, N_RUN = 15, 60
T_IDLE, T_LOAD, N_REP = 10.0, 20.0, 3
LATENCY_TOL = 0.15
MAX_BEFORE = 9   # MiB
MiB = 1024 ** 2


class Power:
    """Reads power.draw.instant from nvidia-smi every 100 ms, time-stamped locally."""

    def __init__(self):
        self.samples = []
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw.instant", "--format=csv,noheader,nounits",
                                   "-lms", "100"], stdout=subprocess.PIPE, text=True)
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        for line in self.p.stdout:
            try:
                self.samples.append((time.perf_counter(), float(line.strip())))
            except ValueError:
                pass

    def mean(self, t0, t1):
        v = [w for t, w in self.samples if t0 <= t <= t1]
        return float(np.mean(v)), len(v)

    def close(self):
        self.p.terminate()


def step(model, x, autocast):
    with torch.no_grad():
        if autocast:
            with torch.cuda.amp.autocast():
                model(x)
        else:
            model(x)


def release():
    """The UNETR forward hooks form a reference cycle: without gc.collect() the previous model stays on the GPU."""
    gc.collect()
    torch.cuda.empty_cache()


def memory_and_latency(name, mode):
    release()
    before = torch.cuda.memory_allocated()
    assert before <= MAX_BEFORE * MiB, f"{before / MiB:.1f} MiB still allocated before loading {name} ({mode})"
    torch.cuda.reset_peak_memory_stats()
    model = make(name).cuda().eval()
    x = torch.randn(1, 3, 224, 224, device="cuda")
    if mode == "fp16":
        model, x = model.half(), x.half()
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    autocast = True   # also in fp16 mode: the UNETR decoder receives float tensors from the encoder
    for _ in range(N_WARM):
        step(model, x, autocast)
    torch.cuda.synchronize()
    ts = []
    for _ in range(N_RUN):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        step(model, x, autocast)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    peak = torch.cuda.max_memory_allocated()
    return model, x, {"memory_before_MiB": before / MiB, "memory_weights_MiB": (weights - before) / MiB,
                      "memory_peak_MiB": (peak - before) / MiB, "latency_ms_median": float(np.median(ts))}


def energy(model, x, power):
    reps = []
    for _ in range(N_REP):
        torch.cuda.synchronize()
        t0 = time.perf_counter(); time.sleep(T_IDLE); t1 = time.perf_counter()
        n = 0
        while time.perf_counter() - t1 < T_LOAD:
            step(model, x, True)
            n += 1
            if n % 20 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        time.sleep(0.3)                                       # the last reading of the interval arrives late
        p_idle, n_idle = power.mean(t0 + 1.0, t1)             # first second of each phase discarded
        p_load, n_load = power.mean(t1 + 1.0, t2)
        dur = t2 - t1
        reps.append({"p_idle_W": p_idle, "p_load_W": p_load, "n_bscans": n, "duration_s": dur,
                     "samples_idle": n_idle, "samples_load": n_load,
                     "energy_gross_mJ": p_load * dur / n * 1000,
                     "energy_net_mJ": (p_load - p_idle) * dur / n * 1000})
    med = {k: float(np.median([r[k] for r in reps])) for k in ("p_idle_W", "p_load_W", "energy_gross_mJ", "energy_net_mJ")}
    return {"median": med, "repetitions": reps}


def main():
    eff_path = os.path.join(RESULTS_DIR, "efficiency.json")
    eff = json.load(open(eff_path, encoding="utf-8"))["models"] if os.path.exists(eff_path) else {}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    out = {"gpu": gpu, "torch": torch.__version__, "input": "1x3x224x224, batch 1", "models": {}}
    power = Power()
    time.sleep(2)
    try:
        for name in MODELS:
            reg = {}
            model, x, reg["autocast"] = memory_and_latency(name, "autocast")
            reg["params_total_M"], reg["params_trainable_M"] = params_M(model)
            if name in eff:
                ref = eff[name]["latency_ms_median"]
                now = reg["autocast"]["latency_ms_median"]
                reg["latency_check"] = {"efficiency_json_ms": ref, "now_ms": now, "ok": abs(now - ref) / ref <= LATENCY_TOL}
            reg["energy_autocast"] = energy(model, x, power)
            del model, x
            model, x, reg["fp16"] = memory_and_latency(name, "fp16")
            del model, x
            out["models"][name] = reg
            print(name, json.dumps({k: v for k, v in reg.items() if k != "energy_autocast"}),
                  "| energy:", reg["energy_autocast"]["median"], flush=True)
    finally:
        power.close()
    with open(os.path.join(RESULTS_DIR, "memory_energy.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
