"""Inference latency/throughput benchmark for all configurations (GPU).

Usage: python bench_latency.py --runs_dir runs_seeds --out latency_results.json
Measures forward-pass latency for batch sizes 1/8/32 on the RTX GPU for
baseline, +Wavelet, +EDL, +Wavelet+EDL, and SE-only checkpoints (seed 42),
using random 1x224x224 inputs. Reports ms per batch, ms per slice, and
throughput (slices/s). Mean +- std over timed iterations.
"""

import argparse
import json
import time

import torch
import torch.nn as nn

from model import ICHModel

CONFIGS = {
    "baseline": dict(use_wavelet=False, use_edl=False, se_only=False),
    "wavelet_only": dict(use_wavelet=True, use_edl=False, se_only=False),
    "edl_only": dict(use_wavelet=False, use_edl=True, se_only=False),
    "wavelet_edl": dict(use_wavelet=True, use_edl=True, se_only=False),
    "se_only": dict(use_wavelet=True, use_edl=False, se_only=True),
}


def build_and_load(cfg_name, runs_dir, seed=42, se_reduction=16):
    flags = CONFIGS[cfg_name]
    model = ICHModel(
        "resnet50",
        flags["use_wavelet"],
        flags["use_edl"],
        pretrained=False,
        se_reduction=se_reduction,
        se_only=flags["se_only"],
    )
    ckpt = torch.load(
        f"{runs_dir}/{cfg_name}_s{seed}/best_model.pt", map_location="cpu"
    )
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] {cfg_name}: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing:", missing[:6])
        if unexpected:
            print("  unexpected:", unexpected[:6])
    model.eval().cuda()
    return model


def bench(model, batch, iters, warmup):
    x = torch.randn(batch, 1, 224, 224, device="cuda")
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    mean = sum(ts) / len(ts)
    var = sum((t - mean) ** 2 for t in ts) / (len(ts) - 1)
    return {
        "batch": batch,
        "ms_per_batch_mean": mean * 1e3,
        "ms_per_batch_std": (var**0.5) * 1e3,
        "ms_per_slice": mean * 1e3 / batch,
        "slices_per_s": batch / mean,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs_seeds")
    ap.add_argument("--out", default="latency_results.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    name = torch.cuda.get_device_name(0)
    results = {
        "gpu": name,
        "iters": args.iters,
        "warmup": args.warmup,
        "note": "random 1x224x224 inputs, torch.inference_mode, seed-42 checkpoints, exact IDWT (default)",
    }
    for cfg_name in CONFIGS:
        model = build_and_load(cfg_name, args.runs_dir, args.seed)
        entry = {}
        for b in args.batches:
            entry[str(b)] = bench(model, b, args.iters, args.warmup)
            print(
                cfg_name,
                b,
                entry[str(b)]["ms_per_slice"],
                "ms/slice",
                entry[str(b)]["slices_per_s"],
                "slices/s",
            )
        results[cfg_name] = entry
        del model
        torch.cuda.empty_cache()

    # derived overheads at batch=1 and batch=32
    r = results
    if "baseline" in r:
        r["derived"] = {}
        for b in args.batches:
            base = r["baseline"][str(b)]["ms_per_slice"]
            for c in ("wavelet_only", "edl_only", "wavelet_edl", "se_only"):
                if c in r:
                    ms = r[c][str(b)]["ms_per_slice"]
                    r["derived"][f"{c}_vs_baseline_b{b}"] = {
                        "abs_ms_per_slice": ms - base,
                        "rel_pct": 100.0 * (ms - base) / base,
                    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
