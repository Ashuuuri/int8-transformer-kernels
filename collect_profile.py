#!/usr/bin/env python3
"""Collect per-kernel CUDA-time breakdown (torch.profiler, no sudo) for the
INT8 MLP and INT8 attention forwards on the graded sweep shape, and write it
to results/kernel_time_breakdown.csv for the dashboard.

This is the "nsys-style" view: where does each forward's GPU time actually go,
kernel by kernel (the fused GEMMs vs the quantize/transpose helpers).
"""
import csv, os, torch
from torch.utils.cpp_extension import load
from torch.profiler import profile, ProfilerActivity

ROOT = os.path.dirname(os.path.abspath(__file__))
# Graded shape (matches profile_kernel.py / CLAUDE.md §3): b=8, s=512.
B, S, DM, DFF = 8, 512, 1024, 4096
HEADS, HEAD_DIM = 8, 64


def load_int8_ext():
    kdir = os.path.join(ROOT, "kernels")
    return load(
        name="int8_ext",
        sources=[os.path.join(kdir, f) for f in
                 ("int8_attention.cu", "int8_decode_attention.cu", "int8_mlp.cu",
                  "quant_utils.cu", "int8_ext.cu")],
        extra_cuda_cflags=["-O2", "--std=c++17", "-arch=sm_80"],
        verbose=False,
    )


def q_tensor(t):
    s = t.abs().max() / 127.0
    return (t / s).round().clamp(-128, 127).to(torch.int8), float(s)


def q_per_token(t):
    am = t.abs().amax(-1, keepdim=True) / 127.0
    i8 = (t / am).round().clamp(-128, 127).to(torch.int8)
    return i8, am.squeeze(-1).float().contiguous()


def profile_forward(fn, n=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    rows = {}
    for e in prof.key_averages():
        if e.device_type.name != "CUDA":
            continue
        t = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        if t <= 0:
            continue
        rows[e.key] = rows.get(e.key, 0.0) + t
    total = sum(rows.values())
    return total, rows


def classify(name):
    n = name.lower()
    if "gemm_int8_wmma_f16" in n or "gemm" in n:
        return "GEMM1/GEMM2 (INT8 WMMA)"
    if "attention" in n:
        return "attention (INT8 WMMA)"
    if "quantize_rows" in n:
        return "quantize_rows (per-token)"
    if "quantize_tensor" in n:
        return "quantize_tensor"
    if "quantize" in n:
        return "quantize (other)"
    if "transpose" in n:
        return "transpose W (B k-contig)"
    return "other / elementwise"


def main():
    ext = load_int8_ext()
    torch.manual_seed(42)
    out_rows = []

    # ---- INT8 MLP forward (per-tensor graded path) ----
    x = torch.randn(B, S, DM, device="cuda", dtype=torch.float16)
    W1 = torch.randn(DM, DFF, device="cuda", dtype=torch.float16) * 0.02
    W2 = torch.randn(DFF, DM, device="cuda", dtype=torch.float16) * 0.02
    x_i8, sx = q_tensor(x); W1_i8, sW1 = q_tensor(W1); W2_i8, sW2 = q_tensor(W2)

    # (a) all-in-one int8_mlp_forward — re-transposes the weights every call.
    total, rows = profile_forward(
        lambda: ext.int8_mlp_forward(x_i8, W1_i8, W2_i8, sx, sW1, sW2))
    agg = {}
    for k, v in rows.items():
        agg[classify(k)] = agg.get(classify(k), 0.0) + v
    for cat, us in sorted(agg.items(), key=lambda kv: -kv[1]):
        out_rows.append(("int8_mlp", cat, us, 100.0 * us / total))
        print(f"  MLP        {cat:34s} {us:10.1f} us  {100*us/total:5.1f}%")

    # (b) prepacked path — weights transposed ONCE at load (static-weight
    # inference, what sweep.py grades): the per-forward transpose is gone.
    W1T, W2T = ext.transpose_int8_weights(W1_i8, W2_i8)
    total, rows = profile_forward(
        lambda: ext.int8_mlp_forward_prepacked(x_i8, W1T, W2T, sx, sW1, sW2))
    agg = {}
    for k, v in rows.items():
        agg[classify(k)] = agg.get(classify(k), 0.0) + v
    for cat, us in sorted(agg.items(), key=lambda kv: -kv[1]):
        out_rows.append(("int8_mlp_prepacked", cat, us, 100.0 * us / total))
        print(f"  MLP(prepack) {cat:32s} {us:10.1f} us  {100*us/total:5.1f}%")

    # ---- INT8 attention forward ----
    Q = torch.randn(B, HEADS, S, HEAD_DIM, device="cuda", dtype=torch.float16)
    K = torch.randn(B, HEADS, S, HEAD_DIM, device="cuda", dtype=torch.float16)
    V = torch.randn(B, HEADS, S, HEAD_DIM, device="cuda", dtype=torch.float16)
    Q_i8, sQ = q_per_token(Q); K_i8, sK = q_per_token(K); V_i8, sV = q_per_token(V)
    total, rows = profile_forward(
        lambda: ext.int8_attention_forward(Q_i8, K_i8, V_i8, sQ, sK, sV))
    agg = {}
    for k, v in rows.items():
        agg[classify(k)] = agg.get(classify(k), 0.0) + v
    for cat, us in sorted(agg.items(), key=lambda kv: -kv[1]):
        out_rows.append(("int8_attn", cat, us, 100.0 * us / total))
        print(f"  ATTN {cat:34s} {us:10.1f} us  {100*us/total:5.1f}%")

    outp = os.path.join(ROOT, "results", "kernel_time_breakdown.csv")
    with open(outp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["forward", "category", "self_us_total", "pct"])
        w.writerows(out_rows)
    print(f"[ok] wrote {outp}")


if __name__ == "__main__":
    main()
