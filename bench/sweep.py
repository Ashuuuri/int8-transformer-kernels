"""sweep.py — Parameter sweep for the INT8 CUDA kernels.

Sweeps:
  seq_len  : 512, 1024, 2048, 4096
  d_model  : 512, 1024, 2048   (d_ff = 4 × d_model)
  head_dim : fixed at 64       (standard; heads fixed at 8)
  batch    : fixed at 8

Usage:
    python sweep.py --kernel int8_mlp   # INT8 MLP   (alias: int8)
    python sweep.py --kernel int8_attn  # INT8 attention

Each INT8 kernel is graded against FP16/cuBLAS references (PyTorch eager,
FlashAttention-2 SDPA, cuBLAS INT8 _int_mm pipeline) — there are no
hand-written FP16 baseline kernels in this repo.

Each run produces:
    results/<kernel>_sweep.csv

Figures: build the consolidated figure set with `python make_figures.py`
(or the legacy single dashboard via `python make_dashboard.py`) after the
sweeps + `python collect_profile.py`. Pass `--legacy-figs` to also emit the
old per-kernel PNGs (results/figures/<kernel>_*.png).
"""

import argparse
import collections
import csv
import os
import sys
import time

import torch
from torch.utils.cpp_extension import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (one up from bench/)
sys.path.insert(0, os.path.join(ROOT, "common"))

from baseline import attention_baseline, mlp_baseline, check_cuda
from benchmark import benchmark

# ══════════════════════════════════════════════════════════════════════════
#  A100 hardware constants
# ══════════════════════════════════════════════════════════════════════════
A100_FP16_TFLOPS = 312.0   # Tensor Core FP16 peak (TFLOPS)
A100_INT8_TOPS   = 624.0   # Tensor Core INT8 peak (TOPS)
# INT8 attention is mixed-precision: QK^T runs INT8 (624) but attn×V runs FP16
# (312). Equal FLOPs in each half → blended peak = 1/(0.5/624 + 0.5/312).
A100_INT8_ATTN_TOPS = 416.0
A100_HBM_BW_TBps = 2.0    # HBM bandwidth peak (TB/s)

# ══════════════════════════════════════════════════════════════════════════
#  Sweep grid  (§4 of proposal)
# ══════════════════════════════════════════════════════════════════════════
SEQ_LENS  = [512, 1024, 2048, 4096]
D_MODELS  = [512, 1024, 2048]
BATCH     = 8
HEADS     = 8
HEAD_DIM  = 64   # d_model // heads for attention

# ══════════════════════════════════════════════════════════════════════════
#  Shared plot colours/markers  (uniform across all three kernels)
# ══════════════════════════════════════════════════════════════════════════
COLORS = {
    "Fused kernel":      "#2563EB",
    "Naive PyTorch":     "#DC2626",
    "cuBLAS / Flash":    "#16A34A",
}
MARKERS = {
    "Fused kernel":      "o",
    "Naive PyTorch":     "s",
    "cuBLAS / Flash":    "^",
}

# ══════════════════════════════════════════════════════════════════════════
#  Per-kernel loaders
# ══════════════════════════════════════════════════════════════════════════
def load_int8_ext():
    kdir = os.path.join(ROOT, "kernels")
    return load(
        name="int8_ext",
        sources=[
            os.path.join(kdir, "int8_attention.cu"),
            os.path.join(kdir, "int8_decode_attention.cu"),
            os.path.join(kdir, "int8_mlp.cu"),
            os.path.join(kdir, "quant_utils.cu"),
            os.path.join(kdir, "int8_ext.cu"),
        ],
        extra_cuda_cflags=["-O2", "--std=c++17", "-arch=sm_80"],
        verbose=False,
    )


# ══════════════════════════════════════════════════════════════════════════
#  FLOPs counters
# ══════════════════════════════════════════════════════════════════════════
def mlp_flops(batch, seq_len, d_model, d_ff):
    T = batch * seq_len
    return 2 * T * d_model * d_ff + 2 * T * d_ff * d_model


def attention_flops(batch, heads, seq_len, head_dim):
    # QK^T + softmax(·)V  — two batched matmuls dominate
    return 2 * batch * heads * seq_len * head_dim * seq_len * 2


def int8_mlp_flops(batch, seq_len, d_model, d_ff):
    return mlp_flops(batch, seq_len, d_model, d_ff)  # same op count


# ══════════════════════════════════════════════════════════════════════════
#  Analytical HBM bytes
# ══════════════════════════════════════════════════════════════════════════
def mlp_hbm_fused(batch, seq_len, d_model, d_ff):
    """Fused: hidden never written to HBM."""
    T = batch * seq_len
    reads  = (T * d_model + d_model * d_ff + d_ff * d_model) * 2   # FP16 = 2 B
    writes = T * d_model * 2
    return reads + writes


def mlp_hbm_unfused(batch, seq_len, d_model, d_ff):
    """Unfused: hidden written after GEMM1, read before GEMM2."""
    T     = batch * seq_len
    extra = 2 * T * d_ff * 2
    return mlp_hbm_fused(batch, seq_len, d_model, d_ff) + extra


def attn_hbm_fused(batch, heads, seq_len, head_dim):
    """Fused attention: attn matrix (S×S) stays in SRAM."""
    T  = batch * heads
    qkv = T * seq_len * head_dim * 2 * 3
    out = T * seq_len * head_dim * 2
    return qkv + out


def attn_hbm_unfused(batch, heads, seq_len, head_dim):
    """Unfused: full S×S attention matrix written + read."""
    T     = batch * heads
    extra = 2 * T * seq_len * seq_len * 2   # write + read attn matrix
    return attn_hbm_fused(batch, heads, seq_len, head_dim) + extra


def int8_mlp_hbm_fused(batch, seq_len, d_model, d_ff):
    """INT8 fused MLP: INT8 in/out/weights (1 B), hidden in HBM as INT8."""
    T = batch * seq_len
    reads  = T * d_model + d_model * d_ff + d_ff * d_model + T * d_ff
    writes = T * d_ff + T * d_model
    return reads + writes


def int8_attn_hbm_fused(batch, heads, seq_len, head_dim):
    """INT8 fused attention: Q/K/V INT8 (1 B), output FP16 (2 B)."""
    T   = batch * heads
    qkv = T * seq_len * head_dim * 3            # INT8 = 1 B
    out = T * seq_len * head_dim * 2            # FP16 = 2 B
    scales = T * seq_len * 4 * 3                # per-token float scales
    return qkv + out + scales


# ══════════════════════════════════════════════════════════════════════════
#  INT8 quantisation helpers
# ══════════════════════════════════════════════════════════════════════════
def quantize_to_int8(t):
    t_f   = t.float()
    scale = t_f.abs().max() / 127.0
    t_i8  = (t_f / scale).round().clamp(-128, 127).to(torch.int8)
    return t_i8, scale


def quantize_per_token(t):
    """Per-token symmetric INT8 quantization over the last dim.

    Returns (int8 tensor, float32 scales with the last dim squeezed).
    """
    t_f = t.float()
    abs_max = t_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scales = abs_max / 127.0
    t_i8 = (t_f / scales).round().clamp(-128, 127).to(torch.int8)
    return t_i8, scales.squeeze(-1).contiguous()


def _err_stats(reference, actual):
    """Max/mean absolute error between kernel output and FP16 reference."""
    diff = (reference.float() - actual.float()).abs()
    return diff.max().item(), diff.mean().item()


# ══════════════════════════════════════════════════════════════════════════
#  Per-kernel benchmark functions
#  Each returns a dict with the SAME keys so the CSV/plot code is shared.
# ══════════════════════════════════════════════════════════════════════════

# ── MLP ──────────────────────────────────────────────────────────────────
# ── INT8 MLP ──────────────────────────────────────────────────────────────
def bench_int8(ext, batch, seq_len, d_model):
    d_ff   = d_model * 4
    device = "cuda"
    dtype  = torch.float16
    torch.manual_seed(42)

    x  = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    W1 = torch.randn(d_model, d_ff,           device=device, dtype=dtype) * 0.02
    W2 = torch.randn(d_ff,    d_model,         device=device, dtype=dtype) * 0.02

    x_i8,  sx  = quantize_to_int8(x)
    W1_i8, sW1 = quantize_to_int8(W1)
    W2_i8, sW2 = quantize_to_int8(W2)
    x_deq  = x_i8.float().mul(sx).half()
    W1_deq = W1_i8.float().mul(sW1).half()
    W2_deq = W2_i8.float().mul(sW2).half()

    sx_f, sW1_f, sW2_f = float(sx), float(sW1), float(sW2)

    # Static-weight inference: transpose the weights ONCE at "load" (outside the
    # timed loop), then call the prepacked entry point per forward. Weights are
    # constant across forwards, so re-transposing them every call (as the
    # all-in-one int8_mlp_forward does) is pure overhead a real deployment
    # amortizes. Bit-identical output to int8_mlp_forward (verified).
    W1T_i8, W2T_i8 = ext.transpose_int8_weights(W1_i8, W2_i8)

    def run_int8():
        ext.int8_mlp_forward_prepacked(x_i8, W1T_i8, W2T_i8, sx_f, sW1_f, sW2_f)

    kernel_ms = benchmark(run_int8)
    # Fair FP16 reference for INT8: use the quantized/dequantized values that
    # the INT8 path actually represents, not the original unquantized tensors.
    naive_ms  = benchmark(mlp_baseline, x_deq, W1_deq, W2_deq)

    # Fair cuBLAS INT8 reference: the full pipeline our kernel performs
    # (GEMM1 → dequant → GELU → requant → GEMM2 → quantized output), using
    # the same static hidden scale as the kernel (sx*sW1*d_model).
    x_2d = x_i8.view(-1, d_model)
    sxw1    = sx_f * sW1_f
    scale_h = sxw1 * d_model
    scale_o = scale_h * sW2_f * d_ff
    import torch.nn.functional as F
    def cublas_int8_pipeline():
        h32  = torch._int_mm(x_2d, W1_i8)
        h    = F.gelu(h32.float() * sxw1, approximate="tanh")
        h_i8 = (h / scale_h).round_().clamp_(-128, 127).to(torch.int8)
        o32  = torch._int_mm(h_i8, W2_i8)
        (o32.float() * (scale_h * sW2_f) / scale_o) \
            .round_().clamp_(-128, 127).to(torch.int8)
    ref2_ms = benchmark(cublas_int8_pipeline)

    # GEMM-only lower bound: two bare INT8 GEMMs, no epilogue work.
    h_i8_pre = torch.randint(-128, 128, (batch * seq_len, d_ff),
                             device=device, dtype=torch.int8)
    def cublas_int8_gemms():
        torch._int_mm(x_2d, W1_i8)
        torch._int_mm(h_i8_pre, W2_i8)
    ref3_ms = benchmark(cublas_int8_gemms)

    # Accuracy vs the FP16 reference on dequantized values.
    out_i8, out_scale = ext.int8_mlp_forward_prepacked(x_i8, W1T_i8, W2T_i8,
                                                       sx_f, sW1_f, sW2_f)
    out_deq = out_i8.float().mul(out_scale).half()
    max_err, mean_err = _err_stats(mlp_baseline(x_deq, W1_deq, W2_deq), out_deq)

    flops     = int8_mlp_flops(batch, seq_len, d_model, d_ff)
    peak_tops = A100_INT8_TOPS

    return _make_row(
        batch, seq_len, d_model,
        kernel_ms, naive_ms, ref2_ms,
        flops, peak_tops,
        int8_mlp_hbm_fused(batch, seq_len, d_model, d_ff),
        mlp_hbm_unfused(batch, seq_len, d_model, d_ff),
        ref2_label="cuBLAS INT8 pipeline",
        kernel_max_err=max_err, kernel_mean_err=mean_err,
        ref3_ms=ref3_ms, ref3_label="cuBLAS INT8 2xGEMM",
    )


# ── INT8 Attention ────────────────────────────────────────────────────────
def bench_int8_attn(ext, batch, seq_len, d_model):
    """INT8 attention vs FP16 references (PyTorch eager + FlashAttention-2).
    All rows time the SAME workload — the FP16 baselines run on the
    dequantized tensors the INT8 kernel actually represents."""
    import torch.nn.functional as F
    int8_ext = ext
    heads    = HEADS
    head_dim = d_model // heads
    device   = "cuda"
    dtype    = torch.float16
    torch.manual_seed(42)

    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    Q_i8, sQ = quantize_per_token(Q)
    K_i8, sK = quantize_per_token(K)
    V_i8, sV = quantize_per_token(V)
    # FP16 references run on the dequantized values — the same numbers the
    # INT8 kernel represents — so all rows time the same workload.
    Q_deq = (Q_i8.float() * sQ.unsqueeze(-1)).half()
    K_deq = (K_i8.float() * sK.unsqueeze(-1)).half()
    V_deq = (V_i8.float() * sV.unsqueeze(-1)).half()

    kernel_ms = benchmark(int8_ext.int8_attention_forward,
                          Q_i8, K_i8, V_i8, sQ, sK, sV)
    naive_ms  = benchmark(attention_baseline, Q_deq, K_deq, V_deq)
    flash_ms  = benchmark(F.scaled_dot_product_attention, Q_deq, K_deq, V_deq)

    out = int8_ext.int8_attention_forward(Q_i8, K_i8, V_i8, sQ, sK, sV)
    max_err, mean_err = _err_stats(attention_baseline(Q_deq, K_deq, V_deq), out)

    flops     = attention_flops(batch, heads, seq_len, head_dim)
    peak_tops = A100_INT8_ATTN_TOPS

    return _make_row(
        batch, seq_len, d_model,
        kernel_ms, naive_ms, flash_ms,
        flops, peak_tops,
        int8_attn_hbm_fused(batch, heads, seq_len, head_dim),
        attn_hbm_unfused(batch, heads, seq_len, head_dim),
        ref2_label="FlashAttn-2",
        kernel_max_err=max_err, kernel_mean_err=mean_err,
    )


# ══════════════════════════════════════════════════════════════════════════
#  Shared row builder  (all bench_* functions return this dict schema)
# ══════════════════════════════════════════════════════════════════════════
def _make_row(
    batch, seq_len, d_model,
    kernel_ms, naive_ms, ref2_ms,
    flops, peak_tops,
    hbm_fused, hbm_unfused,
    ref2_label,
    kernel_max_err=None, kernel_mean_err=None,
    ref3_ms=None, ref3_label="",
):
    def to_tops(ms):
        return flops / (ms * 1e-3) / 1e12

    def bw_util(ms, hbm_bytes):
        achieved = hbm_bytes / (ms * 1e-3) / 1e12
        return achieved / A100_HBM_BW_TBps * 100

    kernel_tops = to_tops(kernel_ms)
    naive_tops  = to_tops(naive_ms)

    return {
        "batch":               batch,
        "seq_len":             seq_len,
        "d_model":             d_model,
        # latencies
        "kernel_ms":           kernel_ms,
        "naive_ms":            naive_ms,
        "ref2_ms":             ref2_ms,
        "ref2_label":          ref2_label,
        "ref3_ms":             ref3_ms if ref3_ms is not None else "",
        "ref3_label":          ref3_label,
        # throughput
        "kernel_tops":         kernel_tops,
        "naive_tops":          naive_tops,
        "ref2_tops":           to_tops(ref2_ms),
        "ref3_tops":           to_tops(ref3_ms) if ref3_ms is not None else "",
        # A100 compute utilisation
        "kernel_util_pct":     kernel_tops / peak_tops * 100,
        "naive_util_pct":      naive_tops  / peak_tops * 100,
        # HBM bandwidth utilisation (analytical)
        "kernel_bw_util_pct":  bw_util(kernel_ms, hbm_fused),
        "naive_bw_util_pct":   bw_util(naive_ms,  hbm_unfused),
        # speedups
        "speedup_vs_naive":    naive_ms  / kernel_ms,
        "speedup_vs_ref2":     ref2_ms   / kernel_ms,
        "speedup_vs_ref3":     ref3_ms / kernel_ms if ref3_ms is not None else "",
        # numerical accuracy vs FP16 reference
        "kernel_max_err":      kernel_max_err if kernel_max_err is not None else "",
        "kernel_mean_err":     kernel_mean_err if kernel_mean_err is not None else "",
    }


# ══════════════════════════════════════════════════════════════════════════
#  CSV
# ══════════════════════════════════════════════════════════════════════════
FIELDNAMES = [
    "batch", "seq_len", "d_model",
    "kernel_ms", "naive_ms", "ref2_ms", "ref2_label",
    "ref3_ms", "ref3_label",
    "kernel_tops", "naive_tops", "ref2_tops", "ref3_tops",
    "kernel_util_pct", "naive_util_pct",
    "kernel_bw_util_pct", "naive_bw_util_pct",
    "speedup_vs_naive", "speedup_vs_ref2", "speedup_vs_ref3",
    "kernel_max_err", "kernel_mean_err",
]

def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[CSV] {len(rows)} rows → {path}")


# ══════════════════════════════════════════════════════════════════════════
#  Plots  (uniform style for all kernels)
# ══════════════════════════════════════════════════════════════════════════
def make_plots(rows, kernel_name, figures_dir, peak_tops, dtype_label="FP16"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARN] matplotlib not found — skipping plots.")
        return

    ref2_label = rows[0]["ref2_label"]
    ref3_label = rows[0].get("ref3_label", "")
    has_ref3   = ref3_label != "" and rows[0].get("ref3_ms", "") != ""
    kname      = kernel_name.upper()

    # Map our internal keys to display names for the legend
    method_cols = [
        ("Fused kernel",   "kernel_ms"),
        ("Naive PyTorch",  "naive_ms"),
        (ref2_label,       "ref2_ms"),
    ]
    # Reuse shared colours; map ref2_label to the third slot
    col_map = {
        "Fused kernel":  COLORS["Fused kernel"],
        "Naive PyTorch": COLORS["Naive PyTorch"],
        ref2_label:      COLORS["cuBLAS / Flash"],
    }
    mrk_map = {
        "Fused kernel":  MARKERS["Fused kernel"],
        "Naive PyTorch": MARKERS["Naive PyTorch"],
        ref2_label:      MARKERS["cuBLAS / Flash"],
    }
    if has_ref3:
        method_cols.append((ref3_label, "ref3_ms"))
        col_map[ref3_label] = "#9333EA"
        mrk_map[ref3_label] = "D"

    def group_by(rows, key):
        d = collections.defaultdict(list)
        for r in rows:
            d[r[key]].append(r)
        return d

    by_dm   = group_by(rows, "d_model")
    max_seq = max(SEQ_LENS)

    # ── Fig 1: Latency vs seq_len (one subplot per d_model) ───────────
    fig, axes = plt.subplots(1, len(D_MODELS),
                             figsize=(5 * len(D_MODELS), 4), sharey=False)
    if len(D_MODELS) == 1:
        axes = [axes]

    for ax, dm in zip(axes, D_MODELS):
        sub  = sorted(by_dm[dm], key=lambda r: r["seq_len"])
        seqs = [r["seq_len"] for r in sub]
        for lbl, col in method_cols:
            ax.plot(seqs, [r[col] for r in sub],
                    marker=mrk_map[lbl], color=col_map[lbl],
                    label=lbl, linewidth=2, markersize=6)
        ax.set_title(f"d_model={dm}  d_ff={dm*4}", fontsize=11)
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Latency (ms)")
        ax.set_xticks(seqs)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{kname} Kernel: Latency vs Sequence Length  "
                 f"(batch={BATCH}, {dtype_label}, A100)", fontsize=13, y=1.02)
    fig.tight_layout()
    p = os.path.join(figures_dir, f"{kernel_name}_latency_vs_seqlen.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {p}")

    # ── Fig 2: Bar chart at largest seq_len ───────────────────────────
    maxseq_rows = {
        dm: next(r for r in rows
                 if r["d_model"] == dm and r["seq_len"] == max_seq)
        for dm in D_MODELS
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    bw = 0.8 / len(method_cols)
    for i, (lbl, col) in enumerate(method_cols):
        vals = [maxseq_rows[dm][col] for dm in D_MODELS]
        offs = [x + (i - (len(method_cols) - 1) / 2) * bw
                for x in range(len(D_MODELS))]
        bars = ax.bar(offs, vals, width=bw, label=lbl,
                      color=col_map[lbl], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(range(len(D_MODELS))))
    ax.set_xticklabels([f"d_model={dm}\nd_ff={dm*4}" for dm in D_MODELS])
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"{kname} Latency by Hidden Size  "
                 f"(seq_len={max_seq}, batch={BATCH}, {dtype_label}, A100)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(figures_dir, f"{kernel_name}_latency_bar_by_dmodel.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {p}")

    # ── Fig 3: TFLOPS vs seq_len ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for dm in D_MODELS:
        sub = sorted(by_dm[dm], key=lambda r: r["seq_len"])
        ax.plot([r["seq_len"] for r in sub],
                [r["kernel_tops"] for r in sub],
                marker="o", label=f"d_model={dm}", linewidth=2, markersize=6)
    ax.axhline(peak_tops, color="gray", linestyle="--", linewidth=1.2,
               label=f"A100 peak ({peak_tops:.0f} TOPS)")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("TFLOPS / TOPS")
    ax.set_xticks(SEQ_LENS)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_title(f"{kname} Kernel Throughput  (batch={BATCH}, {dtype_label})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(figures_dir, f"{kernel_name}_tflops_vs_seqlen.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {p}")

    # ── Fig 4: Speedup vs naive ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for dm in D_MODELS:
        sub = sorted(by_dm[dm], key=lambda r: r["seq_len"])
        ax.plot([r["seq_len"] for r in sub],
                [r["speedup_vs_naive"] for r in sub],
                marker="o", label=f"d_model={dm}", linewidth=2, markersize=6)
    ax.axhline(1.0, color="gray", linestyle="--",
               linewidth=1.0, label="Baseline (1×)")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Speedup over naive PyTorch (×)")
    ax.set_xticks(SEQ_LENS)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_title(f"{kname} Speedup vs Naive  (batch={BATCH}, {dtype_label})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(figures_dir, f"{kernel_name}_speedup_vs_seqlen.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {p}")

    # ── Fig 5: HBM bandwidth utilisation ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    bw_methods = [("Fused kernel",  "kernel_bw_util_pct"),
                  ("Naive PyTorch", "naive_bw_util_pct")]
    for i, (lbl, col) in enumerate(bw_methods):
        vals = [maxseq_rows[dm][col] for dm in D_MODELS]
        offs = [x + (i - 0.5) * 0.35 for x in range(len(D_MODELS))]
        bars = ax.bar(offs, vals, width=0.32, label=lbl,
                      color=col_map[lbl], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(range(len(D_MODELS))))
    ax.set_xticklabels([f"d_model={dm}" for dm in D_MODELS])
    ax.set_ylabel("HBM Bandwidth Utilisation (%)")
    ax.set_title(f"{kname} HBM Bandwidth Utilisation  "
                 f"(seq_len={max_seq}, batch={BATCH}, A100 peak=2TB/s)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(figures_dir,
                     f"{kernel_name}_hbm_bw_utilisation.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {p}")


# ══════════════════════════════════════════════════════════════════════════
#  Kernel registry  — add a new kernel here, nothing else needs to change
# ══════════════════════════════════════════════════════════════════════════
KERNELS = {
    "int8_mlp": {
        "loader":      load_int8_ext,
        "bench_fn":    bench_int8,
        "peak_tops":   A100_INT8_TOPS,
        "dtype_label": "INT8",
    },
    "int8_attn": {
        "loader":      load_int8_ext,
        "bench_fn":    bench_int8_attn,
        "peak_tops":   A100_INT8_ATTN_TOPS,
        "dtype_label": "INT8",
    },
}
# Backwards-compatible alias for the old --kernel int8 (MLP sweep).
KERNELS["int8"] = KERNELS["int8_mlp"]


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep for CUDA Transformer kernels."
    )
    parser.add_argument(
        "--kernel",
        choices=list(KERNELS.keys()),
        required=True,
        help="Which kernel to sweep: mlp | attention | int8",
    )
    parser.add_argument(
        "--legacy-figs",
        action="store_true",
        help="Also emit the 5 old per-kernel PNGs. Off by default — the "
             "consolidated dashboard (python make_dashboard.py) replaced them.",
    )
    args = parser.parse_args()

    check_cuda()
    cfg = KERNELS[args.kernel]

    results_dir = os.path.join(ROOT, "results")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{args.kernel}_sweep.csv")

    print(f"Kernel  : {args.kernel}  ({cfg['dtype_label']})")
    print(f"Grid    : seq_lens={SEQ_LENS}  d_models={D_MODELS}  batch={BATCH}")
    print(f"Output  : {csv_path}\n")

    print("Compiling CUDA kernel …")
    ext = cfg["loader"]()
    print("Done.\n")

    total, done, rows = len(SEQ_LENS) * len(D_MODELS), 0, []

    for d_model in D_MODELS:
        for seq_len in SEQ_LENS:
            done += 1
            print(f"[{done:2d}/{total}] seq={seq_len:4d}  d_model={d_model}",
                  end="  ", flush=True)
            t0  = time.perf_counter()
            row = cfg["bench_fn"](ext, BATCH, seq_len, d_model)
            elapsed = time.perf_counter() - t0
            print(f"kernel={row['kernel_ms']:.2f}ms  "
                  f"naive={row['naive_ms']:.2f}ms  "
                  f"speedup={row['speedup_vs_naive']:.2f}×  "
                  f"({elapsed:.1f}s)")
            rows.append(row)

    save_csv(rows, csv_path)
    if args.legacy_figs:
        make_plots(rows, args.kernel, figures_dir, cfg["peak_tops"],
                   cfg.get("dtype_label", "FP16"))
        print(f"Legacy per-kernel PNGs written to {figures_dir}/")
    else:
        print("CSV written. Build the consolidated figure with: "
              "python make_dashboard.py  (use --legacy-figs for the old PNGs)")
    print("\nSweep complete.")


if __name__ == "__main__":
    main()
