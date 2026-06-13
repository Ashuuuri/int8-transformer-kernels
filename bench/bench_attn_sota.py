"""INT8 attention vs the real INT8 SOTA (SageAttention), not just FP16 peers.

Why this exists
---------------
`sweep.py`'s INT8-attention row compares our kernel against naive PyTorch and
FlashAttention-2 — both *FP16* references. Neither of them
is an INT8 attention kernel, so "we lose to FA2" conflated two things: the INT8
algorithm and the engineering level. The honest peer for an INT8 attention
kernel is another *INT8* attention kernel. On A100 (sm_80) that is
**SageAttention v1** (Triton; INT8 QK^T with smoothing, FP16 PV) — the same idea
this project implements, but production-tuned. FlashAttention-3 FP8 is Hopper-only
(sm_90) and cannot run here, so it is correctly excluded.

This reports, per shape:
  - ours      : our int8_attention_forward (pre-quantized int8 in)
  - sage      : SageAttention sageattn (fp16 in, INT8 internally) — the SOTA peer
  - fa2       : F.scaled_dot_product_attention (fp16) — FP16 reference
plus cosine similarity of each against an fp32 math reference.

Shapes default to sweep.py's GRADED prefill grid (b=8, h=8, head_dim=64,
seq 512..4096). This is the *prefill / square* regime — the one the current
kernel supports (it is square-only, seq_q == seq_kv). The decode regime
(seq_q=1, long KV), where INT8's KV-cache byte advantage actually pays off,
needs a separate kernel entry point and lives in bench_attn_decode.py once that
exists.

Usage:  python3 bench_attn_sota.py            # graded prefill grid
        python3 bench_attn_sota.py 512 1024   # specific seq lens
"""
import sys
import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (one up from bench/)
sys.path.insert(0, os.path.join(ROOT, "common"))
from benchmark import benchmark

CSV_OUT = os.path.join(ROOT, "results", "attn_sota.csv")

BATCH, HEADS, HEAD_DIM = 8, 8, 64
SEQS = [512, 1024, 2048, 4096]

try:
    from sageattention import sageattn
    HAVE_SAGE = True
except Exception as e:  # pragma: no cover
    HAVE_SAGE = False
    _SAGE_ERR = repr(e)


def _load_int8():
    return load(
        name="int8_ext",
        sources=["kernels/int8_attention.cu", "kernels/int8_decode_attention.cu",
                 "kernels/int8_mlp.cu",
                 "kernels/quant_utils.cu", "kernels/int8_ext.cu"],
        extra_cuda_cflags=["-arch=sm_80", "--std=c++17", "-O3"],
        verbose=False,
    )


def _q_per_token(t):
    am = t.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    sc = am / 127.0
    i8 = (t / sc).round().clamp(-128, 127).to(torch.int8)
    return i8, sc.squeeze(-1).contiguous().float()


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def run_shape(int8_ext, b, h, s, d):
    torch.manual_seed(42)
    Q = torch.randn(b, h, s, d, device="cuda", dtype=torch.float16)
    K = torch.randn(b, h, s, d, device="cuda", dtype=torch.float16)
    V = torch.randn(b, h, s, d, device="cuda", dtype=torch.float16)

    Qi, sQ = _q_per_token(Q)
    Ki, sK = _q_per_token(K)
    Vi, sV = _q_per_token(V)
    # Ground-truth reference = fp32 attention on the ORIGINAL inputs. Both INT8
    # kernels are scored against this, so each pays for its own quantization
    # error (ours: per-token input quant; sage: its internal quant). Scoring
    # against the dequantized inputs instead would hand ours a trivial cos=1.0.
    ref = F.scaled_dot_product_attention(Q.float(), K.float(), V.float())

    out = {"b": b, "h": h, "s": s, "d": d}

    # ours (int8 in, fp16 out)
    out["ours_ms"] = benchmark(int8_ext.int8_attention_forward,
                               Qi, Ki, Vi, sQ, sK, sV)
    ours = int8_ext.int8_attention_forward(Qi, Ki, Vi, sQ, sK, sV)
    out["ours_cos"] = _cos(ours, ref)

    # SOTA peer: SageAttention (fp16 in, INT8 internally)
    if HAVE_SAGE:
        try:
            _sage = lambda: sageattn(Q, K, V, tensor_layout="HND", is_causal=False)
            o_sage = _sage()
            out["sage_ms"] = benchmark(_sage)
            out["sage_cos"] = _cos(o_sage, ref)
        except Exception as e:
            out["sage_ms"], out["sage_cos"] = None, None
            out["sage_err"] = repr(e)[:80]
    else:
        out["sage_ms"], out["sage_cos"] = None, None

    # FP16 reference (fa2_cos = fp16 SDPA vs fp32 ref, i.e. FA2's own fp16 error
    # — the precision floor an INT8 kernel is trying to approach)
    out["fa2_ms"] = benchmark(F.scaled_dot_product_attention, Q, K, V)
    out["fa2_cos"] = _cos(F.scaled_dot_product_attention(Q, K, V), ref)
    return out


def main():
    seqs = [int(x) for x in sys.argv[1:]] or SEQS
    if not HAVE_SAGE:
        print(f"[warn] SageAttention not importable ({_SAGE_ERR}); "
              f"SOTA column will be empty. pip install --user sageattention")
    int8_ext = _load_int8()
    print(f"\nPrefill / square regime  (b={BATCH} h={HEADS} d={HEAD_DIM}, "
          f"seq_q == seq_kv).  Latency = mean ms over 50 iters.\n")
    hdr = (f"{'seq':>6} | {'ours ms':>9} {'sage ms':>9} {'fa2 ms':>9} | "
           f"{'ours/sage':>9} {'ours/fa2':>9} | "
           f"{'ours cos':>9} {'sage cos':>9}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for s in seqs:
        r = run_shape(int8_ext, BATCH, HEADS, s, HEAD_DIM)
        sage_ms = r["sage_ms"]
        spd_sage = f"{sage_ms / r['ours_ms']:.2f}x" if sage_ms else "  n/a"
        spd_fa2 = f"{r['fa2_ms'] / r['ours_ms']:.2f}x"
        sage_str = f"{sage_ms:9.3f}" if sage_ms else f"{'n/a':>9}"
        sage_cos = f"{r['sage_cos']:9.5f}" if r["sage_cos"] is not None else f"{'n/a':>9}"
        print(f"{s:>6} | {r['ours_ms']:9.3f} {sage_str} {r['fa2_ms']:9.3f} | "
              f"{spd_sage:>9} {spd_fa2:>9} | "
              f"{r['ours_cos']:9.5f} {sage_cos}")
        if r.get("sage_err"):
            print(f"        sage error: {r['sage_err']}")
        rows.append({
            "seq_len": s, "ours_ms": r["ours_ms"], "sage_ms": sage_ms or "",
            "fa2_ms": r["fa2_ms"], "ours_cos": r["ours_cos"],
            "sage_cos": r["sage_cos"] if r["sage_cos"] is not None else "",
            "fa2_cos": r["fa2_cos"],
            "speedup_vs_sage": (sage_ms / r["ours_ms"]) if sage_ms else "",
            "speedup_vs_fa2": r["fa2_ms"] / r["ours_ms"],
        })
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[CSV] {len(rows)} rows -> {CSV_OUT}")
    print("speedup > 1.0x means ours is faster.  cos vs fp32 SDPA reference.\n")


if __name__ == "__main__":
    main()
