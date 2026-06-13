"""Decode attention vs a real QUANTIZED-KV SOTA peer — not just FP16.

Why this exists
---------------
`bench_attn_decode.py` compares our INT8 decode kernel only against FP16 SDPA.
That answers "should I quantize the KV cache?" but NOT "is my kernel competitive
with a production quantized-KV decode kernel?" — the honest peer for a quantized
decode kernel is another *quantized* decode kernel. On A100 (sm_80) that is
**FlashInfer's FP8 (e4m3) paged-KV decode**: same memory-bound idea (KV stored at
1 byte/elem, half the FP16 bytes, dequant-on-load + FP16 math), production-tuned.

FlashInfer does NOT support an INT8 KV cache on sm_80 (only FP8), so FP8 is the
apples-to-apples peer: **both stream 1 byte/elem of KV**, i.e. identical KV
bandwidth — so this isolates kernel quality at equal bytes, removing the
"you only beat FP16" objection. (FP8 e4m3 is also what production serving
actually deploys for quantized KV, so it is arguably the stronger peer.)

Peers, per (B,H,D,S):
  ours_int8 : int8_decode_attention_forward     (INT8 Q + INT8 KV, per-token scales)
  fi_fp8    : FlashInfer batch decode, FP8 e4m3 paged KV  (the quantized SOTA)
  fi_fp16   : FlashInfer batch decode, FP16 paged KV      (unquantized reference)
cos is vs an fp32 SDPA reference (computed only where the FP16 KV still fits).

Usage: python3 bench_decode_sota.py
"""
import os
import sys
import csv
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (one up from bench/)
sys.path.insert(0, os.path.join(ROOT, "common"))
from benchmark import benchmark

import flashinfer

CSV_OUT = os.path.join(ROOT, "results", "decode_sota.csv")

# serving-scale configs (skip grid-starved small b*h); D ∈ {64,128}.
CONFIGS = [(32, 8, 64), (64, 16, 64), (128, 16, 64), (64, 16, 128)]
SEQS = [2048, 4096, 8192, 16384, 32768]
PAGE = 16                     # FlashInfer paged-KV page size (S % PAGE == 0)
FP8_MAX = 448.0               # e4m3 max magnitude
OOM = torch.cuda.OutOfMemoryError
# fp32 accuracy reference rebuilds FP16 K,V — only attempt where it fits.
COS_MAX_ELEMS = 64 * 16 * 4096 * 64


def _load():
    return load(name="int8_ext",
        sources=["kernels/int8_attention.cu", "kernels/int8_decode_attention.cu",
                 "kernels/int8_mlp.cu", "kernels/quant_utils.cu", "kernels/int8_ext.cu"],
        extra_cuda_cflags=["-arch=sm_80", "--std=c++17", "-O3"], verbose=False)


def _qpt(t):
    """Per-token int8 (our kernel's quantization): scale per (b,h,s) over D."""
    am = t.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    sc = am / 127.0
    return (t / sc).round().clamp(-128, 127).to(torch.int8), sc.squeeze(-1).contiguous().float()


def _to_fp8_paged(t, scale):
    """(B,H,S,D) fp16 -> FlashInfer NHD paged fp8 cache (num_pages, PAGE, H, D)."""
    B, H, S, D = t.shape
    q = (t / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.permute(0, 2, 1, 3).contiguous().reshape(B * S // PAGE, PAGE, H, D)


def _to_fp16_paged(t):
    B, H, S, D = t.shape
    return t.permute(0, 2, 1, 3).contiguous().reshape(B * S // PAGE, PAGE, H, D)


def _plan_meta(B, H, D, S, kv_dtype, ws):
    """Build a BatchDecode wrapper planned for B requests of length S."""
    npp = S // PAGE                       # pages per request
    indptr = torch.arange(0, (B + 1) * npp, npp, device="cuda", dtype=torch.int32)
    indices = torch.arange(0, B * npp, device="cuda", dtype=torch.int32)
    last = torch.full((B,), PAGE, device="cuda", dtype=torch.int32)
    # use_tensor_cores=True is REQUIRED for a fair peer: it >2× speeds FlashInfer's
    # FP8 decode at these head dims (3.29→1.52 ms @ B64 S8K) — leaving it off would
    # cripple the SOTA. Give the peer its best path.
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="NHD",
                                                      use_tensor_cores=True)
    w.plan(indptr, indices, last, H, H, D, PAGE,
           q_data_type=torch.float16, kv_data_type=kv_dtype)
    return w


def run(ext, B, H, D, S, ws):
    row = dict(B=B, H=H, D=D, S=S, ours_ms="", fi_fp8_ms="", fi_fp16_ms="",
               ours_cos="", fi_fp8_cos="", spd_vs_fp8="", spd_vs_fp16="")
    torch.manual_seed(0)
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    Qi, sQ = _qpt(q)

    # Build + quantize K, then free its fp16 copy (KV caches dominate memory).
    K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    kscale = (K.abs().amax() / FP8_MAX).item()
    Ki, sK = _qpt(K)
    Kf8 = _to_fp8_paged(K, kscale)
    keep_fp16 = (B * H * S * D) <= COS_MAX_ELEMS
    Kf16 = _to_fp16_paged(K) if keep_fp16 else None
    Kref = K if keep_fp16 else None
    if not keep_fp16:
        del K
    torch.cuda.empty_cache()

    V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    vscale = (V.abs().amax() / FP8_MAX).item()
    Vi, sV = _qpt(V)
    Vf8 = _to_fp8_paged(V, vscale)
    Vf16 = _to_fp16_paged(V) if keep_fp16 else None
    Vref = V if keep_fp16 else None
    if not keep_fp16:
        del V
    torch.cuda.empty_cache()

    # ours (INT8)
    row["ours_ms"] = benchmark(ext.int8_decode_attention_forward, Qi, Ki, Vi, sQ, sK, sV)
    ours = ext.int8_decode_attention_forward(Qi, Ki, Vi, sQ, sK, sV)

    # FlashInfer FP8 (the quantized-KV SOTA peer) — same 1 byte/elem KV as ours
    w8 = _plan_meta(B, H, D, S, torch.float8_e4m3fn, ws)
    fp8_run = lambda: w8.run(q, (Kf8, Vf8), k_scale=kscale, v_scale=vscale)
    o_fp8 = fp8_run()
    row["fi_fp8_ms"] = benchmark(fp8_run)
    row["spd_vs_fp8"] = row["fi_fp8_ms"] / row["ours_ms"]

    # FlashInfer FP16 (unquantized reference, 2 bytes/elem) + fp32 accuracy ref:
    # both rebuild the FP16 KV cache, so only run where it still fits on 40 GB.
    if keep_fp16:
        w16 = _plan_meta(B, H, D, S, torch.float16, ws)
        fp16_run = lambda: w16.run(q, (Kf16, Vf16))
        _ = fp16_run()
        row["fi_fp16_ms"] = benchmark(fp16_run)
        row["spd_vs_fp16"] = row["fi_fp16_ms"] / row["ours_ms"]
        ref = F.scaled_dot_product_attention(
            q.float().unsqueeze(2), Kref.float(), Vref.float()).squeeze(2)
        row["ours_cos"] = F.cosine_similarity(ours.float().flatten(), ref.flatten(), dim=0).item()
        row["fi_fp8_cos"] = F.cosine_similarity(o_fp8.float().flatten(), ref.flatten(), dim=0).item()
        del Kref, Vref, Kf16, Vf16
    torch.cuda.empty_cache()
    return row


def main():
    ext = _load()
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")  # FI workspace
    print("\nDecode vs quantized-KV SOTA (FlashInfer FP8).  ms = mean / decode step (50 it).")
    print("ours=INT8 KV;  fi_fp8=FlashInfer FP8 KV (SOTA peer, same 1 B/elem);  "
          "fi_fp16=FlashInfer FP16 KV.\n")
    hdr = (f"{'B':>4} {'H':>3} {'D':>4} {'S':>6} | {'ours':>7} {'fp8':>7} {'fp16':>7} | "
           f"{'ours/fp8':>8} {'ours/fp16':>9} | {'ours cos':>8} {'fp8 cos':>8}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for (B, H, D) in CONFIGS:
        for S in SEQS:
            try:
                r = run(ext, B, H, D, S, ws)
            except OOM:
                torch.cuda.empty_cache()
                print(f"{B:>4} {H:>3} {D:>4} {S:>6} | OOM"); continue

            def f(x, w=7, p=4):
                return f"{x:{w}.{p}f}" if isinstance(x, float) else f"{'--':>{w}}"
            print(f"{B:>4} {H:>3} {D:>4} {S:>6} | {f(r['ours_ms'])} {f(r['fi_fp8_ms'])} "
                  f"{f(r['fi_fp16_ms'])} | {f(r['spd_vs_fp8'],8,2)} {f(r['spd_vs_fp16'],9,2)} | "
                  f"{f(r['ours_cos'],8,5)} {f(r['fi_fp8_cos'],8,5)}")
            rows.append(r)
    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[CSV] {len(rows)} rows -> {CSV_OUT}")
    print("ours/fp8 > 1.0 = our INT8 decode beats the FP8 SOTA at EQUAL KV bytes "
          "(kernel quality, not just precision).\n")


if __name__ == "__main__":
    main()
