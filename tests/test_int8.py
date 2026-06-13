# -*- coding: utf-8 -*-
"""Test harness for INT8 attention + MLP kernels.

Usage:
    python tests/test_int8.py              # full test (correctness + benchmark)
    python tests/test_int8.py --quick      # correctness only, all head_dims
    python tests/test_int8.py --head-dim 256  # test specific head_dim only

Attention uses per-token symmetric quantization (each token row has its own scale).
MLP still uses per-tensor quantization.

Correctness:
  Reference = dequantized FP16 attention (per-token dequant -> FP16 matmuls).
  INT8 kernel output is FP16 (no dequant needed on the Python side).
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from baseline import attention_baseline, mlp_baseline, check_cuda
from correctness import check_correctness
from benchmark import benchmark

# ── A100 peak performance ───────────────────────────────────────────────
A100_FP16_TFLOPS = 312.0
A100_INT8_TOPS = 624.0
# INT8 attention is mixed-precision: QK^T runs INT8 (624 TOPS) but attn×V runs
# FP16 (312 TFLOPS). The two halves have equal FLOP counts, so the blended
# achievable peak is 1 / (0.5/624 + 0.5/312) = 416.
A100_INT8_ATTN_TOPS = 416.0


# ── Per-tensor quantization (MLP still uses this) ──────────────────────
def quantize_to_int8(t):
    """Per-tensor symmetric quantization to INT8."""
    t_f = t.float()
    scale = t_f.abs().max() / 127.0
    t_int8 = (t_f / scale).round().clamp(-128, 127).to(torch.int8)
    return t_int8, scale


# ── Per-token quantization (attention uses this) ──────────────────────
def quantize_per_token(t):
    """Per-token symmetric quantization to INT8.

    Each token row (last dim = head_dim) gets its own scale factor.
    Input:  (batch, heads, seq_len, head_dim)
    Returns:
        t_int8: (batch, heads, seq_len, head_dim) torch.int8
        scales: (batch, heads, seq_len) torch.float32  (one scale per row)
    """
    t_f = t.float()
    # abs max along last dim -> [batch, heads, seq_len, 1]
    abs_max = t_f.abs().amax(dim=-1, keepdim=True)
    abs_max = abs_max.clamp(min=1e-8)
    scales = abs_max / 127.0                        # [batch, heads, seq_len, 1]
    t_int8 = (t_f / scales).round().clamp(-128, 127).to(torch.int8)
    scales = scales.squeeze(-1)                      # [batch, heads, seq_len]
    return t_int8, scales


def compute_attention_flops(batch, heads, seq_len, head_dim):
    qk_flops = 2 * batch * heads * seq_len * head_dim * seq_len
    av_flops = 2 * batch * heads * seq_len * seq_len * head_dim
    return qk_flops + av_flops


def compute_mlp_flops(batch, seq_len, d_model, d_ff):
    tokens = batch * seq_len
    return 2 * tokens * d_model * d_ff + 2 * tokens * d_ff * d_model


# ── Multi-config correctness ──────────────────────────────────────────
ATTN_CONFIGS = [
    # (batch, heads, seq_len, head_dim, description)
    (2, 8, 512,  64,  "head_dim=64  (d_model=512)"),
    (2, 8, 512,  128, "head_dim=128 (d_model=1024)"),
    (2, 8, 512,  256, "head_dim=256 (d_model=2048) [Opt#3 target]"),
    (2, 8, 1024, 64,  "head_dim=64  seq=1024"),
    (2, 8, 1024, 128, "head_dim=128 seq=1024"),
    (2, 8, 1024, 256, "head_dim=256 seq=1024 [Opt#3 target]"),
]


def test_attention_correctness(int8_ext, configs=None):
    """Run attention correctness across multiple configs. Returns (n_pass, n_total)."""
    if configs is None:
        configs = ATTN_CONFIGS

    device = "cuda"
    dtype = torch.float16
    n_pass = 0

    for batch, heads, seq_len, head_dim, desc in configs:
        torch.manual_seed(42)
        Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

        # Per-token quantization
        Q_i8, sQ = quantize_per_token(Q)   # sQ: [batch, heads, seq_len]
        K_i8, sK = quantize_per_token(K)
        V_i8, sV = quantize_per_token(V)

        # Reference: dequant per-token, then FP16 attention
        Q_deq = (Q_i8.float() * sQ.unsqueeze(-1)).half()   # broadcast [b,h,s,1] * [b,h,s,d]
        K_deq = (K_i8.float() * sK.unsqueeze(-1)).half()
        V_deq = (V_i8.float() * sV.unsqueeze(-1)).half()
        ref = attention_baseline(Q_deq, K_deq, V_deq)

        # INT8 kernel — scales passed as CUDA float tensors
        # The kernel expects flat [batch*heads*seq_len] scales
        out = int8_ext.int8_attention_forward(
            Q_i8, K_i8, V_i8,
            sQ.contiguous().cuda(),
            sK.contiguous().cuda(),
            sV.contiguous().cuda())

        passed = check_correctness(ref, out, label=f"int8_attn_pertoken [{desc}]", mode="int8")
        if passed:
            n_pass += 1

    return n_pass, len(configs)


def test_mlp_correctness(int8_ext):
    """Run MLP correctness check (still per-tensor). Returns True if passed."""
    device = "cuda"
    dtype = torch.float16
    batch, seq_len, d_model, d_ff = 2, 512, 512, 2048

    torch.manual_seed(42)
    x  = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    W1 = torch.randn(d_model, d_ff, device=device, dtype=dtype) * 0.02
    W2 = torch.randn(d_ff, d_model, device=device, dtype=dtype) * 0.02

    x_i8, sx   = quantize_to_int8(x)
    W1_i8, sW1 = quantize_to_int8(W1)
    W2_i8, sW2 = quantize_to_int8(W2)

    x_deq  = (x_i8.float() * sx).half()
    W1_deq = (W1_i8.float() * sW1).half()
    W2_deq = (W2_i8.float() * sW2).half()
    ref = mlp_baseline(x_deq, W1_deq, W2_deq)

    out_i8, out_scale = int8_ext.int8_mlp_forward(
        x_i8, W1_i8, W2_i8, float(sx), float(sW1), float(sW2))
    out = (out_i8.float() * out_scale).half()

    return check_correctness(ref, out, label="int8_mlp", mode="int8")


def main():
    parser = argparse.ArgumentParser(description="INT8 kernel correctness & benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Correctness only (skip benchmark)")
    parser.add_argument("--head-dim", type=int, default=None,
                        help="Test only this head_dim (64, 128, or 256)")
    args = parser.parse_args()

    check_cuda()

    # ── Load INT8 CUDA kernels ───────────────────────────────────────────
    from torch.utils.cpp_extension import load
    print("Compiling INT8 kernels ...")
    int8_ext = load(
        name="int8_ext",
        sources=[
            "kernels/int8_attention.cu",
            "kernels/int8_decode_attention.cu",
            "kernels/int8_mlp.cu",
            "kernels/quant_utils.cu",
            "kernels/int8_ext.cu",
        ],
        extra_cuda_cflags=["-arch=sm_80", "--std=c++17", "-O3"],
        verbose=False,
    )
    print("Done.\n")

    # ── Correctness ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  CORRECTNESS: INT8 Attention (per-token quantization)")
    print("=" * 60)

    if args.head_dim:
        configs = [(b, h, s, hd, desc) for b, h, s, hd, desc in ATTN_CONFIGS
                   if hd == args.head_dim]
        if not configs:
            print(f"No configs with head_dim={args.head_dim}")
            sys.exit(1)
    else:
        configs = ATTN_CONFIGS

    attn_pass, attn_total = test_attention_correctness(int8_ext, configs)
    print(f"\nAttention: {attn_pass}/{attn_total} configs passed")

    print("\n" + "=" * 60)
    print("  CORRECTNESS: INT8 MLP")
    print("=" * 60)
    mlp_passed = test_mlp_correctness(int8_ext)

    # ── Summary ─────────────────────────────────────────────────────────
    all_pass = (attn_pass == attn_total) and mlp_passed
    print("\n" + "=" * 60)
    if all_pass:
        print("  ALL CORRECTNESS CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED")
        if attn_pass < attn_total:
            print(f"    Attention: {attn_total - attn_pass} config(s) failed")
        if not mlp_passed:
            print(f"    MLP: FAILED")
    print("=" * 60)

    if args.quick:
        sys.exit(0 if all_pass else 1)

    if not all_pass:
        print("\nSkipping benchmark due to correctness failures.")
        sys.exit(1)

    # ── Benchmark: INT8 Attention ───────────────────────────────────────
    device = "cuda"
    dtype = torch.float16

    for batch, heads, seq_len, head_dim, tag in [
        (2, 8, 512, 64,  "head_dim=64"),
        (2, 8, 512, 256, "head_dim=256 [Opt #3 target]"),
    ]:
        torch.manual_seed(42)
        Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        Q_i8, sQ = quantize_per_token(Q)
        K_i8, sK = quantize_per_token(K)
        V_i8, sV = quantize_per_token(V)
        # All FP16 references run on the dequantized values — the same numbers
        # the INT8 kernel represents — so latencies compare the same workload.
        Q_deq = (Q_i8.float() * sQ.unsqueeze(-1)).half()
        K_deq = (K_i8.float() * sK.unsqueeze(-1)).half()
        V_deq = (V_i8.float() * sV.unsqueeze(-1)).half()

        print(f"\n=== Benchmark: INT8 Attention ({tag}, per-token, "
              f"b={batch} h={heads} s={seq_len}) ===")
        naive_ms = benchmark(attention_baseline, Q_deq, K_deq, V_deq)
        flash_ms = benchmark(F.scaled_dot_product_attention, Q_deq, K_deq, V_deq)
        int8_ms  = benchmark(int8_ext.int8_attention_forward,
                             Q_i8, K_i8, V_i8,
                             sQ.contiguous(), sK.contiguous(), sV.contiguous())

        flops = compute_attention_flops(batch, heads, seq_len, head_dim)
        tops = flops / (int8_ms * 1e-3) / 1e12
        print(f"  Naive PyTorch FP16:  {naive_ms:.3f} ms")
        print(f"  FlashAttention-2:    {flash_ms:.3f} ms")
        print(f"  Your INT8 kernel:    {int8_ms:.3f} ms")
        print(f"  Speedup vs naive:    {naive_ms / int8_ms:.2f}x")
        print(f"  Speedup vs FA-2:     {flash_ms / int8_ms:.2f}x")
        print(f"  Your TOPS:           {tops:.1f}  |  "
              f"Utilization: {tops / A100_INT8_ATTN_TOPS * 100:.1f}% "
              f"(blended INT8/FP16 peak {A100_INT8_ATTN_TOPS:.0f})")

    # ── Benchmark: INT8 MLP ─────────────────────────────────────────────
    batch, seq_len, d_model, d_ff = 8, 512, 1024, 4096

    torch.manual_seed(42)
    x  = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    W1 = torch.randn(d_model, d_ff, device=device, dtype=dtype) * 0.02
    W2 = torch.randn(d_ff, d_model, device=device, dtype=dtype) * 0.02
    x_i8, sx   = quantize_to_int8(x)
    W1_i8, sW1 = quantize_to_int8(W1)
    W2_i8, sW2 = quantize_to_int8(W2)
    x_deq  = (x_i8.float() * sx).half()
    W1_deq = (W1_i8.float() * sW1).half()
    W2_deq = (W2_i8.float() * sW2).half()

    print(f"\n=== Benchmark: INT8 MLP (b={batch} s={seq_len} "
          f"d_model={d_model} d_ff={d_ff}) ===")
    # FP16 reference on dequantized values (the practical FP16 path).
    fp16_mlp_ms = benchmark(mlp_baseline, x_deq, W1_deq, W2_deq)
    int8_mlp_ms = benchmark(lambda: int8_ext.int8_mlp_forward(
        x_i8, W1_i8, W2_i8, float(sx), float(sW1), float(sW2)))

    # Fair cuBLAS INT8 reference: the full pipeline our kernel performs
    # (GEMM1 → dequant → GELU → requant → GEMM2 → quantized output),
    # built from torch._int_mm + elementwise ops. Same static hidden scale
    # as the kernel: sx*sW1*d_model.
    x_2d_i8 = x_i8.view(-1, d_model)
    sxw1 = float(sx) * float(sW1)
    scale_h = sxw1 * d_model
    scale_o = scale_h * float(sW2) * d_ff
    def cublas_int8_pipeline():
        h32 = torch._int_mm(x_2d_i8, W1_i8)
        h = F.gelu(h32.float() * sxw1, approximate="tanh")
        h_i8 = (h / scale_h).round_().clamp_(-128, 127).to(torch.int8)
        o32 = torch._int_mm(h_i8, W2_i8)
        (o32.float() * (scale_h * float(sW2)) / scale_o) \
            .round_().clamp_(-128, 127).to(torch.int8)
    cublas_pipe_ms = benchmark(cublas_int8_pipeline)

    # GEMM-only lower bound: the two bare INT8 GEMMs with no epilogue work.
    h_i8_pre = torch.randint(-128, 128, (batch * seq_len, d_ff),
                             device=device, dtype=torch.int8)
    def cublas_int8_gemms():
        torch._int_mm(x_2d_i8, W1_i8)
        torch._int_mm(h_i8_pre, W2_i8)
    cublas_gemm_ms = benchmark(cublas_int8_gemms)

    mlp_flops = compute_mlp_flops(batch, seq_len, d_model, d_ff)
    mlp_tops = mlp_flops / (int8_mlp_ms * 1e-3) / 1e12

    print(f"  FP16 cuBLAS MLP:        {fp16_mlp_ms:.3f} ms")
    print(f"  cuBLAS INT8 pipeline:   {cublas_pipe_ms:.3f} ms  (fair: full quantized MLP)")
    print(f"  cuBLAS INT8 2x GEMM:    {cublas_gemm_ms:.3f} ms  (GEMM-only lower bound)")
    print(f"  Your INT8 kernel:       {int8_mlp_ms:.3f} ms")
    print(f"  Speedup vs FP16:        {fp16_mlp_ms / int8_mlp_ms:.2f}x")
    print(f"  Speedup vs INT8 pipe:   {cublas_pipe_ms / int8_mlp_ms:.2f}x")
    print(f"  Your TOPS:              {mlp_tops:.1f}  |  "
          f"Utilization: {mlp_tops / A100_INT8_TOPS * 100:.1f}%")


if __name__ == "__main__":
    main()
