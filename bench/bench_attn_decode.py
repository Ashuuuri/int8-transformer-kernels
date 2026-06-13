"""Decode-regime benchmark: INT8 KV-cache attention vs the FP16 decode path.

The decode regime (seq_q == 1, long KV) is where INT8 attention's real value
lives: each generated token re-reads the entire KV cache, so decode is
**bandwidth-bound on the KV cache**, and an INT8 cache is half the bytes of FP16.
The square WMMA kernel cannot run this shape (seq_q==1 wastes its Q-tiling);
int8_decode_attention_forward (kernels/int8_decode_attention.cu) is the
flash-decoding entry point built for it.

Peers:
  - ours : int8_decode_attention_forward  (INT8 Q + INT8 KV cache)
  - sdpa : F.scaled_dot_product_attention, seq_q==1  (FP16, PyTorch's decode path)
SageAttention is a *prefill* (block) kernel and is not a decode peer, so it is
not included here (see bench_attn_sota.py for the prefill SOTA comparison).

Also reports the KV-cache footprint: INT8 stores K and V at 1 byte/elem vs FP16's
2, the structural memory win a serving system cares about (more context / bigger
batch per card, half the bytes streamed per token).

Usage: python3 bench_attn_decode.py
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

CSV_OUT = os.path.join(ROOT, "results", "attn_decode.csv")

# (batch, heads, head_dim) — small to serving-scale; one warp per (b,h), so
# batch*heads is the parallelism (small b*h is grid-starved on 108 SMs).
CONFIGS = [(8, 8, 64), (32, 8, 64), (64, 16, 64), (128, 16, 64), (64, 16, 128)]
# Long-context grid: decode re-reads the WHOLE KV cache per token, so this is
# the regime that matters today. The FP16 SDPA reference OOMs at the largest
# (config, seq) on a 40 GB card — that is itself the point (INT8 KV = half the
# bytes), and those rows are skipped gracefully.
SEQS = [1024, 2048, 4096, 8192, 16384, 32768]


def _load():
    return load(name="int8_ext",
        sources=["kernels/int8_attention.cu", "kernels/int8_decode_attention.cu",
                 "kernels/int8_mlp.cu", "kernels/quant_utils.cu", "kernels/int8_ext.cu"],
        extra_cuda_cflags=["-arch=sm_80", "--std=c++17", "-O3"], verbose=False)


def _qpt(t):
    am = t.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    sc = am / 127.0
    return (t / sc).round().clamp(-128, 127).to(torch.int8), sc.squeeze(-1).contiguous().float()


def run(ext, B, H, D, S):
    """OOM-safe. Footprint is analytical (always); the FP16 SDPA reference and
    the fp32 accuracy ref are each guarded — at the largest (config, seq) the
    FP16 path OOMs on 40 GB while the INT8 path still runs (that contrast is the
    point), so sdpa_ms/cos come back empty and the row is INT8-only."""
    OOM = torch.cuda.OutOfMemoryError
    kv_int8_mb = B * H * S * D * 2 / 1e6        # K+V, 1 byte each
    kv_fp16_mb = kv_int8_mb * 2
    row = dict(B=B, H=H, D=D, S=S, ours_ms="", sdpa_ms="", cos="",
               kv_int8_mb=kv_int8_mb, kv_fp16_mb=kv_fp16_mb, ours_gbps="")
    torch.manual_seed(0)
    # Build + quantize each tensor, then free its fp16 copy immediately. A real
    # serving system stores the KV cache as int8 and never materializes fp16, so
    # the INT8 path must run where holding fp16 K,V would not even fit. Only Q's
    # fp16 is kept (tiny: seq_q==1) for the SDPA reference / accuracy check.
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    Qi, sQ = _qpt(Q)
    K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    Ki, sK = _qpt(K); del K; torch.cuda.empty_cache()
    V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    Vi, sV = _qpt(V); del V; torch.cuda.empty_cache()

    ours_ms = benchmark(ext.int8_decode_attention_forward, Qi, Ki, Vi, sQ, sK, sV)
    row["ours_ms"] = ours_ms
    row["ours_gbps"] = (B * H * S * D * 2) / (ours_ms * 1e-3) / 1e9

    out = ext.int8_decode_attention_forward(Qi, Ki, Vi, sQ, sK, sV)
    # FP16 reference: rebuild K,V in fp16. At the largest (config, seq) this is
    # where the FP16 KV cache does not fit on 40 GB — that OOM IS the result.
    try:
        torch.manual_seed(0)
        _ = torch.randn(B, H, D, device="cuda", dtype=torch.float16)  # match RNG stream
        K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        Q4 = Q.unsqueeze(2)  # (B,H,1,D) for SDPA decode
        row["sdpa_ms"] = benchmark(F.scaled_dot_product_attention, Q4, K, V)
        try:
            ref = F.scaled_dot_product_attention(
                Q.float().unsqueeze(2), K.float(), V.float()).squeeze(2)
        except OOM:
            torch.cuda.empty_cache()
            ref = F.scaled_dot_product_attention(Q.unsqueeze(2), K, V).squeeze(2).float()
        row["cos"] = F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
        del K, V
    except OOM:
        torch.cuda.empty_cache()  # FP16 reference does not fit — INT8-only row
    return row


def main():
    ext = _load()
    print("\nDecode regime (seq_q == 1).  Latency = mean ms / decode step over 50 iters.")
    print("ours = INT8 Q + INT8 KV cache;  sdpa = FP16 SDPA decode path.\n")
    hdr = (f"{'B':>4} {'H':>3} {'D':>4} {'S':>5} | {'ours ms':>8} {'sdpa ms':>8} "
           f"{'ours/sdpa':>9} | {'cos':>8} | {'KV int8':>9} {'KV fp16':>9} "
           f"{'ours GB/s':>9}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for (B, H, D) in CONFIGS:
        for S in SEQS:
            r = run(ext, B, H, D, S)
            if r["sdpa_ms"] == "":   # FP16 reference OOM'd — INT8-only row
                spd = ""
                print(f"{B:>4} {H:>3} {D:>4} {S:>5} | {r['ours_ms']:8.4f} {'OOM':>8} "
                      f"{'--':>8}  | {'--':>8} | {r['kv_int8_mb']:8.1f}M {r['kv_fp16_mb']:8.1f}M "
                      f"{r['ours_gbps']:9.1f}   <- FP16 KV does not fit on 40 GB")
            else:
                spd = r["sdpa_ms"] / r["ours_ms"]
                print(f"{B:>4} {H:>3} {D:>4} {S:>5} | {r['ours_ms']:8.4f} {r['sdpa_ms']:8.4f} "
                      f"{spd:8.2f}x | {r['cos']:8.5f} | {r['kv_int8_mb']:8.1f}M {r['kv_fp16_mb']:8.1f}M "
                      f"{r['ours_gbps']:9.1f}")
            r["speedup_vs_sdpa"] = spd
            rows.append(r)
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[CSV] {len(rows)} rows -> {CSV_OUT}")
    print("ours/sdpa > 1.0 = INT8 decode kernel faster than the FP16 path.")
    print("INT8 KV cache is half the FP16 bytes (more context / bigger batch per card).")
    print("A100-SXM4 HBM peak ~1555 GB/s — ours GB/s shows how bandwidth-bound we are.\n")


if __name__ == "__main__":
    main()
