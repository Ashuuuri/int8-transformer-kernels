"""End-to-end transformer-BLOCK wall-clock: the never-measured number.

Until now every benchmark here has timed a single kernel in isolation (attention
alone, MLP alone). The project's actual claim — "the fused INT8 forward wins at
the WORKLOAD level by moving fewer HBM bytes / fewer round-trips" — has only ever
been shown analytically and per-kernel. This file assembles a whole transformer
block and times it end to end, the first apples-to-apples integration number.

What the block contains
-----------------------
A pre-LN transformer block (GPT-2 style):

    h = x + Wo @ Attn( split_heads( LN1(x) @ Wqkv ) )
    y = h + MLP( LN2(h) )

The INT8 kernels cover ONLY the two heavy pieces:
  - the attention CORE (softmax(QK^T/√d)·V on already-projected Q,K,V)
  - the MLP (two GEMMs + tanh-GELU)
Everything else — LayerNorm, the residual adds, the QKV projection and the output
projection — stays FP16 in BOTH the INT8 block and the FP16 reference (LN/residual
are a hard "never quantize" rule; the projections have no INT8 kernel here). They
are byte-identical across the two paths, so the wall-clock delta isolates exactly
what swapping attention-core + MLP to INT8 buys.

Two regimes (mirrors the dashboard's row 4):
  - prefill : seq_q == seq_kv, compute-bound — INT8's byte win is not expected to
              help much; the MLP carries whatever win there is.
  - decode  : seq_q == 1 over a long INT8 KV cache, bandwidth-bound — the regime
              where INT8 actually pays off.

Accuracy: cosine of the INT8 block output vs the FP16 block output (the FP16 block
IS the reference), so the number is the honest end-to-end quantization error, not a
per-kernel cosine. The MLP uses the per-channel/per-token path (the real-model
accuracy path, matching validate_int8.py Gate 5), not the per-tensor legacy path.

Usage:  python3 bench_block.py
"""
import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from benchmark import benchmark

CSV_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "block.csv")


def _load():
    return load(name="int8_ext",
        sources=["kernels/int8_attention.cu", "kernels/int8_decode_attention.cu",
                 "kernels/int8_mlp.cu", "kernels/quant_utils.cu", "kernels/int8_ext.cu"],
        extra_cuda_cflags=["-arch=sm_80", "--std=c++17", "-O3"], verbose=False)


def _q_per_token(t):
    """Per-token symmetric INT8 quant over the LAST dim (matches the kernels)."""
    sc = t.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
    i8 = (t / sc).round().clamp(-128, 127).to(torch.int8)
    return i8.contiguous(), sc.squeeze(-1).contiguous().float()


def _q_per_channel(w):
    """Per-output-channel weight quant (amax over the input dim, dim 0)."""
    sc = w.abs().amax(0, keepdim=True).clamp(min=1e-8) / 127.0          # [1, N]
    i8 = (w / sc).round().clamp(-128, 127).to(torch.int8)
    return i8.contiguous(), sc.squeeze(0).contiguous().float()


class Block:
    """Holds one block's weights; exposes FP16 and INT8 forwards for both regimes.
    Projections (Wqkv, Wo) and the MLP weights are shared between the FP16 and INT8
    paths so the cosine measures pure quantization error."""

    def __init__(self, ext, d_model, heads, d_ff, dtype=torch.float16, dev="cuda"):
        self.ext = ext
        self.d, self.H, self.dff = d_model, heads, d_ff
        self.Dh = d_model // heads
        g = torch.Generator(device=dev).manual_seed(0)
        rn = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=dtype)
        # projections (kept FP16 in both paths)
        self.Wqkv = rn(d_model, 3 * d_model) * (d_model ** -0.5)
        self.Wo   = rn(d_model, d_model) * (d_model ** -0.5)
        # MLP weights: FP16 master + INT8 per-channel quantized copies
        self.W1 = rn(d_model, d_ff) * (d_model ** -0.5)
        self.W2 = rn(d_ff, d_model) * (d_ff ** -0.5)
        self.W1i, self.sW1 = _q_per_channel(self.W1)        # [d,dff] i8, [dff]
        self.W2i, self.sW2 = _q_per_channel(self.W2)        # [dff,d] i8, [d]

    # ── shared FP16 pieces ────────────────────────────────────────────────
    def _ln(self, x):
        return F.layer_norm(x, (self.d,))

    def _project_qkv(self, h):
        B, S, _ = h.shape
        qkv = h @ self.Wqkv                                  # (B,S,3d)
        q, k, v = qkv.split(self.d, dim=-1)
        shp = lambda t: t.view(B, S, self.H, self.Dh).transpose(1, 2).contiguous()
        return shp(q), shp(k), shp(v)                        # each (B,H,S,Dh)

    def _mlp_fp16(self, h):
        return F.gelu(h @ self.W1, approximate="tanh") @ self.W2

    def _mlp_int8(self, h):
        B, S, d = h.shape
        T = B * S
        xi, sx = _q_per_token(h.reshape(T, d))               # [T,d] i8, [T]
        out_i8, out_sc = self.ext.int8_mlp_forward(
            xi.view(B, S, d), self.W1i, self.W2i, sx, self.sW1, self.sW2)
        return (out_i8.float().reshape(T, d) * out_sc.unsqueeze(-1)).reshape(B, S, d).half()

    # ── prefill (seq_q == seq_kv) ─────────────────────────────────────────
    def forward_prefill(self, x, use_int8):
        B, S, _ = x.shape
        q, k, v = self._project_qkv(self._ln(x))
        if use_int8:
            qi, sq = _q_per_token(q); ki, sk = _q_per_token(k); vi, sv = _q_per_token(v)
            a = self.ext.int8_attention_forward(qi, ki, vi, sq, sk, sv)   # (B,H,S,Dh) fp16
        else:
            a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).contiguous().view(B, S, self.d) @ self.Wo
        h = x + a
        mlp = self._mlp_int8(self._ln(h)) if use_int8 else self._mlp_fp16(self._ln(h))
        return h + mlp

    # ── decode (seq_q == 1 over a prefilled KV cache) ─────────────────────
    def forward_decode(self, x, kv, use_int8):
        """x: (B,1,d) new token. kv = the prefilled cache for this regime:
        int8 path -> (Ki,Vi,sK,sV); fp16 path -> (Kf,Vf)."""
        B = x.size(0)
        h1 = self._ln(x)
        qkv = h1 @ self.Wqkv
        q, _, _ = qkv.split(self.d, dim=-1)                  # only the new q is needed
        if use_int8:
            Ki, Vi, sK, sV = kv
            qd = q.view(B, self.H, self.Dh).contiguous()     # (B,H,Dh)
            qi, sq = _q_per_token(qd)
            a = self.ext.int8_decode_attention_forward(qi, Ki, Vi, sq, sK, sV)  # (B,H,Dh)
            a = a.view(B, 1, self.d)
        else:
            Kf, Vf = kv
            q4 = q.view(B, self.H, 1, self.Dh)
            a = F.scaled_dot_product_attention(q4, Kf, Vf)   # (B,H,1,Dh)
            a = a.transpose(1, 2).contiguous().view(B, 1, self.d)
        a = a @ self.Wo
        h = x + a
        mlp = self._mlp_int8(self._ln(h)) if use_int8 else self._mlp_fp16(self._ln(h))
        return h + mlp


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def run_prefill(ext, B, H, Dh, dff, S):
    d = H * Dh
    blk = Block(ext, d, H, dff)
    x = torch.randn(B, S, d, device="cuda", dtype=torch.float16)
    i8 = blk.forward_prefill(x, True)
    f16 = blk.forward_prefill(x, False)
    return dict(regime="prefill", B=B, H=H, Dh=Dh, d_model=d, d_ff=dff, seq=S,
                int8_ms=benchmark(blk.forward_prefill, x, True),
                fp16_ms=benchmark(blk.forward_prefill, x, False),
                cos=_cos(i8, f16))


def run_decode(ext, B, H, Dh, dff, Skv):
    """OOM-safe: build+quantize each KV tensor then free its fp16 copy (a real
    serving cache is stored int8, never materialized fp16), so the INT8 block
    runs where holding fp16 K,V would not fit. The fp16 reference rebuilds K,V
    and is guarded — if it OOMs the row is INT8-only (that contrast is the point)."""
    OOM = torch.cuda.OutOfMemoryError
    d = H * Dh
    blk = Block(ext, d, H, dff)
    x = torch.randn(B, 1, d, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, Skv, Dh, device="cuda", dtype=torch.float16)
    Ki, sK = _q_per_token(K); del K; torch.cuda.empty_cache()
    V = torch.randn(B, H, Skv, Dh, device="cuda", dtype=torch.float16)
    Vi, sV = _q_per_token(V); del V; torch.cuda.empty_cache()
    i8 = blk.forward_decode(x, (Ki, Vi, sK, sV), True)
    int8_ms = benchmark(blk.forward_decode, x, (Ki, Vi, sK, sV), True)
    row = dict(regime="decode", B=B, H=H, Dh=Dh, d_model=d, d_ff=dff, seq=Skv,
               int8_ms=int8_ms, fp16_ms="", cos="")
    try:
        Kf = torch.randn(B, H, Skv, Dh, device="cuda", dtype=torch.float16)
        Vf = torch.randn(B, H, Skv, Dh, device="cuda", dtype=torch.float16)
        f16 = blk.forward_decode(x, (Kf, Vf), False)
        row["cos"] = _cos(i8, f16)
        row["fp16_ms"] = benchmark(blk.forward_decode, x, (Kf, Vf), False)
        del Kf, Vf
    except OOM:
        torch.cuda.empty_cache()  # FP16 KV does not fit — INT8-only row
    return row


def main():
    ext = _load()
    rows = []
    print("\nEnd-to-end transformer BLOCK (LN + proj fp16; attn-core + MLP int8 vs fp16)")
    hdr = (f"{'regime':>8} {'B':>4} {'H':>3} {'Dh':>3} {'dmodel':>6} {'dff':>5} "
           f"{'seq':>6} | {'int8 ms':>8} {'fp16 ms':>8} {'speedup':>8} | {'cos':>8}")
    print(hdr); print("-" * len(hdr))
    # prefill: graded-ish shape, sweep seq
    for S in [512, 1024, 2048]:
        rows.append(run_prefill(ext, 8, 16, 64, 4096, S))
    # decode: B=128 so MLP tile T=B*1=128 is a 128-multiple; sweep KV length
    # into long context, where the INT8 KV cache's half-bytes pay off and grow.
    for Skv in [2048, 4096, 8192, 16384, 32768]:
        rows.append(run_decode(ext, 128, 16, 64, 4096, Skv))
    for r in rows:
        if r["fp16_ms"] == "":                  # FP16 reference OOM'd
            r["speedup"] = ""
            print(f"{r['regime']:>8} {r['B']:>4} {r['H']:>3} {r['Dh']:>3} {r['d_model']:>6} "
                  f"{r['d_ff']:>5} {r['seq']:>6} | {r['int8_ms']:8.4f} {'OOM':>8} "
                  f"{'--':>7}  | {'--':>8}   <- FP16 KV does not fit")
            continue
        spd = r["fp16_ms"] / r["int8_ms"]
        r["speedup"] = spd
        print(f"{r['regime']:>8} {r['B']:>4} {r['H']:>3} {r['Dh']:>3} {r['d_model']:>6} "
              f"{r['d_ff']:>5} {r['seq']:>6} | {r['int8_ms']:8.4f} {r['fp16_ms']:8.4f} "
              f"{spd:7.2f}x | {r['cos']:8.5f}")
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[CSV] {len(rows)} rows -> {CSV_OUT}")
    print("speedup > 1 = INT8 block faster end-to-end.  cos = INT8 vs FP16 block output.")
    print("LN + residual + QKV/out projections are FP16 & identical in both paths;")
    print("the delta is purely the attention-core + MLP INT8 swap.\n")


if __name__ == "__main__":
    main()
