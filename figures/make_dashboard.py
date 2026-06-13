#!/usr/bin/env python3
"""Consolidated INT8 performance + profiling dashboard.

Produces ONE big figure, results/figures/int8_dashboard.png, replacing the
20 scattered per-kernel PNGs with a single readable sheet:

  Row 1  INT8 MLP   — latency / HBM bytes moved / speedup vs 3 labelled refs
  Row 2  INT8 attn  — latency / HBM bytes moved / speedup vs FlashAttn-2 (SDPA)
  Row 3  Profiling  — ncu stall breakdown · kernel-time split (nsys-style) ·
                      MLP optimisation journey (iter 7→8→10) from ncu metrics

The middle column is the MEMORY-BOUND hint: these kernels win by moving fewer
HBM bytes (INT8 = 1 B vs FP16 2 B + fusion removes intermediate round-trips),
not by peak TOPS — so bytes-moved, not throughput, is plotted next to latency.

Data sources (all under results/):
  *_sweep.csv               sweep.py latency/throughput/speedup
  kernel_time_breakdown.csv collect_profile.py torch.profiler per-kernel time
  ncu_{mlp,attn}_raw.csv    raw `ncu --csv` stall metrics (preamble tolerated)
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (one up from figures/)
RES = os.path.join(ROOT, "results")
FIGDIR = os.path.join(RES, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Consistent colour roles across the whole sheet:
C_OURS  = "#2563EB"   # blue  — ALWAYS our INT8 fused kernel
C_SOTA  = "#16A34A"   # green — the INT8 SOTA peer (cuBLAS INT8 pipeline / SageAttention)
C_FP16  = "#DC2626"   # red   — primary FP16 baseline (cuBLAS FP16 / FP16 manual)
C_FA2   = "#6B7280"   # gray  — FlashAttn-2 (FP16, well-tuned reference floor)
C_LOWER = "#F59E0B"   # amber — secondary floor (cuBLAS INT8 2×GEMM)
REP_DM  = 1024        # representative d_model for the line panels
BATCH   = 8           # graded batch (matches sweep.py)
HEADS   = 8           # graded heads (matches sweep.py)


# ────────────────────────────────────────────────────────────────────────
#  Analytical HBM bytes moved  (the memory-bound story — mirrors sweep.py).
#  These kernels are MEMORY-BOUND at the workload level: the win is fewer HBM
#  bytes (INT8 = 1 B vs FP16 2 B) + fewer round-trips (fusion), NOT peak TOPS.
#  Byte formulas match sweep.py's; the INT8 *pipeline* model below adds the
#  INT32/INT8 intermediate round-trips a real `_int_mm` + dequant/GELU/requant
#  deployment pays — that traffic is exactly what the fused kernel removes.
# ────────────────────────────────────────────────────────────────────────
def mlp_bytes_fp16_unfused(s, dm):
    dff = dm * 4; T = BATCH * s
    fused = (T*dm + dm*dff + dff*dm) * 2 + T*dm*2      # FP16 dense, hidden in SRAM
    return fused + 2 * T * dff * 2                      # + hidden write+read (unfused)


def mlp_bytes_int8_pipeline(s, dm):
    """_int_mm (INT32 out) + dequant/GELU/requant + _int_mm + dequant.
    Minimal 2-epilogue model: the INT32/INT8 hidden round-trips the fused
    kernel avoids. Validated ≈5.0x ours at s=4096 vs measured 4.76x speedup."""
    dff = dm * 4; T = BATCH * s
    return 11*T*dm + 2*dm*dff + 10*T*dff


def mlp_bytes_int8_fused(s, dm):                        # ours
    dff = dm * 4; T = BATCH * s
    return T*dm*2 + 2*dm*dff + 2*T*dff


def attn_bytes_fp16_unfused(s, dm):                     # materializes S×S
    hd = dm // HEADS; T = BATCH * HEADS
    return T*s*hd*2*3 + T*s*hd*2 + 2*T*s*s*2


def attn_bytes_fp16_fused(s, dm):                       # FlashAttn-2 (no S×S)
    hd = dm // HEADS; T = BATCH * HEADS
    return T*s*hd*2*3 + T*s*hd*2


def attn_bytes_int8_fused(s, dm):                       # ours (INT8 QKV)
    hd = dm // HEADS; T = BATCH * HEADS
    return T*s*hd*3 + T*s*hd*2 + T*s*4*3


def read_sweep(name):
    rows = []
    with open(os.path.join(RES, name)) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if _isnum(v) else v) for k, v in r.items()})
    return rows


def _isnum(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


def at_dm(rows, dm):
    sub = [r for r in rows if int(r["d_model"]) == dm]
    sub.sort(key=lambda r: r["seq_len"])
    return sub


# ────────────────────────────────────────────────────────────────────────
#  ncu stall parsing
# ────────────────────────────────────────────────────────────────────────
STALL_KEYS = [
    ("wait",            "smsp__warp_issue_stalled_wait_per_warp_active.pct"),
    ("long_scoreboard", "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"),
    ("short_scoreboard","smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"),
    ("mio_throttle",    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct"),
    ("barrier",         "smsp__warp_issue_stalled_barrier_per_warp_active.pct"),
    ("math_throttle",   "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"),
    ("not_selected",    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct"),
    ("lg_throttle",     "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct"),
]
IMMA_KEY = "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active"


def read_ncu(name):
    """Return {kernel_label: {metric_short: value}} from a raw `ncu --csv` dump.

    Tolerates the `==PROF==` / banner preamble ncu prints before the CSV: we
    skip everything up to the real header row (ncu quotes every field, and the
    header contains "Kernel Name").
    """
    out = {}
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return out
    with open(path) as f:
        lines = [ln for ln in f if ln.startswith('"')]
    if not lines:
        return out
    for r in csv.DictReader(lines):
        kn = r.get("Kernel Name", "")
        try:
            val = float(str(r.get("Metric Value", "")).replace(",", ""))
        except ValueError:
            continue
        mn = r.get("Metric Name", "")
        out.setdefault(kn, {})[mn] = val
    return out


def label_kernel(kn):
    if "gemm_int8_wmma_f16_kernel<1, 0, 0>" in kn:
        return "MLP GEMM1\n(x·W1+GELU)"
    if "gemm_int8_wmma_f16_kernel<0, 1, 0>" in kn:
        return "MLP GEMM2\n(h·W2)"
    if "attention" in kn:
        return "Attn QK^T\n(INT8 part)"
    return kn[:18]


# ────────────────────────────────────────────────────────────────────────
def panel_latency(ax, rows, series, title, ylabel="latency (ms)"):
    sub = at_dm(rows, REP_DM)
    x = [r["seq_len"] for r in sub]
    for col, lbl, color in series:
        y = [r[col] for r in sub]
        ax.plot(x, y, marker="o", color=color, label=lbl, linewidth=2, markersize=5)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=10)
    ax.set_xticks(x); ax.set_xticklabels([int(s) for s in x])
    ax.set_xlabel("seq_len"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8, loc="upper left")


def panel_tops(ax, rows, series, peak, title):
    sub = at_dm(rows, REP_DM)
    x = [r["seq_len"] for r in sub]
    for col, lbl, color in series:
        y = [r[col] for r in sub]
        ax.plot(x, y, marker="o", color=color, label=lbl, linewidth=2, markersize=5)
    if peak:
        ax.axhline(peak, color="gray", ls="--", lw=1.2, label=f"A100 peak ≈{peak:.0f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x); ax.set_xticklabels([int(s) for s in x])
    ax.set_xlabel("seq_len"); ax.set_ylabel("throughput (TOPS / TFLOPS)")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="lower right")


def panel_bytes(ax, rows, byte_series, title, note):
    """HBM bytes moved per forward vs seq_len (analytical) — the memory-bound
    hint. Parallels the latency panel's x-axis so 'fewer bytes' reads directly
    against 'lower latency'. byte_series = [(label, color, fn(seq, dm)), ...]."""
    sub = at_dm(rows, REP_DM)
    x = [r["seq_len"] for r in sub]
    for lbl, color, fn in byte_series:
        y = [fn(int(s), REP_DM) / 1e9 for s in x]      # GB
        ax.plot(x, y, marker="s", color=color, label=lbl, linewidth=2, markersize=5)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=10)
    ax.set_xticks(x); ax.set_xticklabels([int(s) for s in x])
    ax.set_xlabel("seq_len"); ax.set_ylabel("HBM bytes moved / forward (GB)")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8, loc="upper left")
    ax.text(0.5, -0.30, note, transform=ax.transAxes, fontsize=8.5,
            ha="center", va="top", color="#1E3A8A", style="italic", wrap=True)


def panel_speedup(ax, rows, series, title):
    sub = at_dm(rows, REP_DM)
    x = np.arange(len(sub)); seqs = [int(r["seq_len"]) for r in sub]
    n = len(series); w = 0.8 / n
    for i, (col, lbl, color) in enumerate(series):
        y = [r[col] for r in sub]
        ax.bar(x + (i - (n - 1) / 2) * w, y, w, label=lbl, color=color, alpha=0.9)
    ax.axhline(1.0, color="gray", ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(seqs)
    ax.set_xlabel("seq_len"); ax.set_ylabel("speedup (× ref, >1 = ours faster)")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=8, loc="upper left")


def panel_stalls(ax, ncu_mlp, ncu_attn):
    kernels = []
    for kn, m in ncu_mlp.items():
        kernels.append((label_kernel(kn), m))
    for kn, m in ncu_attn.items():
        kernels.append((label_kernel(kn), m))
    if not kernels:
        ax.text(0.5, 0.5, "no ncu data", ha="center"); return
    labels = [k for k, _ in kernels]
    x = np.arange(len(labels))
    cmap = plt.get_cmap("tab10")
    bottoms = np.zeros(len(labels))
    for i, (short, full) in enumerate(STALL_KEYS):
        vals = np.array([m.get(full, 0.0) for _, m in kernels])
        ax.bar(x, vals, 0.55, bottom=bottoms, label=short, color=cmap(i % 10))
        bottoms += vals
    # tensor pipe utilisation overlay (markers)
    imma = [m.get(IMMA_KEY, np.nan) for _, m in kernels]
    ax.plot(x, imma, "D", color="black", markersize=8, label="tensor_op_imma %", zorder=5)
    for xi, v in zip(x, imma):
        if not np.isnan(v):
            ax.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points",
                        xytext=(7, 0), fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("% of warp-active cycles (stacked stalls)")
    ax.set_title("ncu stall breakdown — wait = MMA-dependency ceiling",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.25)


def panel_kerneltime(ax):
    path = os.path.join(RES, "kernel_time_breakdown.csv")
    if not os.path.exists(path):
        ax.text(0.5, 0.5, "no breakdown data", ha="center"); return
    from collections import defaultdict
    data = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            data[r["forward"]].append((r["category"], float(r["pct"])))
    # Show MLP all-in-one vs MLP prepacked (transpose hoisted out) vs attn.
    fwds   = [f for f in ["int8_mlp", "int8_mlp_prepacked", "int8_attn"] if f in data]
    labels = {"int8_mlp": "MLP forward\n(transpose/call)",
              "int8_mlp_prepacked": "MLP forward\n(prepacked W)",
              "int8_attn": "attn forward"}
    x = np.arange(len(fwds))
    cats = []
    for fw in fwds:
        for c, _ in data[fw]:
            if c not in cats: cats.append(c)
    cmap = plt.get_cmap("Set2")
    bottoms = np.zeros(len(fwds))
    for i, cat in enumerate(cats):
        vals = np.array([dict(data[fw]).get(cat, 0.0) for fw in fwds])
        ax.bar(x, vals, 0.5, bottom=bottoms, label=cat, color=cmap(i % 8))
        for xi, (b, v) in enumerate(zip(bottoms, vals)):
            if v >= 6:
                ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold")
        bottoms += vals
    ax.set_xticks(x); ax.set_xticklabels([labels.get(f, f) for f in fwds], fontsize=8)
    ax.set_ylabel("% of forward CUDA time")
    ax.set_title("Where the GPU time goes (torch.profiler, nsys-style)\n"
                 "prepacking W removes the per-call transpose",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.25)


def panel_journey(ax):
    # Documented graded-shape GEMM2 ncu metrics across the MLP iterations
    # (CLAUDE.md / README iter 7→8; iter 10 measured this session, reverted).
    iters = ["iter 7\n(bank-conflict\nfix)", "iter 8\n(STAGE_K 64)",
             "iter 10\n(denser-mma\nREVERTED)"]
    imma = [30.9, 38.2, 38.24]
    wait = [24.3, 28.2, 28.18]
    longsb = [27.4, 7.7, 7.65]
    x = np.arange(len(iters)); w = 0.25
    b1 = ax.bar(x - w, imma, w, label="tensor_op_imma", color=C_OURS)
    b2 = ax.bar(x,     wait, w, label="wait (MMA dep)", color=C_FP16)
    b3 = ax.bar(x + w, longsb, w, label="long_scoreboard", color=C_LOWER)
    # mark iter 10 bars as reverted (hatch)
    for b in (b1, b2, b3):
        b[-1].set_hatch("//"); b[-1].set_alpha(0.55)
    ax.axhline(60, color="green", ls="--", lw=1.3)
    ax.text(0.02, 61, "cuBLAS tensor-pipe target ≈60%+", color="green",
            fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels(iters, fontsize=8)
    ax.set_ylabel("% (GEMM2, graded shape)")
    ax.set_title("MLP optimisation journey — iter 10 moved nothing (dead end)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 70)


def panel_sota_prefill(ax, rows):
    """Prefill latency vs the REAL INT8 SOTA peer (SageAttention) — the HONEST
    footnote: prefill is compute-bound, so INT8's byte advantage does NOT help
    here; we trail SOTA as seq grows. The INT8 win is decode/footprint (→)."""
    if not rows:
        ax.text(0.5, 0.5, "no attn_sota.csv\n(run bench_attn_sota.py)",
                ha="center", va="center"); ax.set_axis_off(); return
    rows = sorted(rows, key=lambda r: r["seq_len"])
    x = [int(r["seq_len"]) for r in rows]
    series = [("ours_ms", "INT8 fused (ours)", C_OURS),
              ("sage_ms", "SageAttention (INT8 SOTA)", C_SOTA),
              ("fa2_ms",  "FlashAttn-2 (FP16 ref)", C_FA2)]
    for col, lbl, color in series:
        y = [r[col] for r in rows if isinstance(r.get(col), float)]
        if len(y) == len(x):
            ax.plot(x, y, marker="o", color=color, label=lbl, linewidth=2, markersize=5)
    for r in rows:                                   # annotate ours/sage
        sp = r.get("speedup_vs_sage")
        if isinstance(sp, float):
            ax.annotate(f"{sp:.2f}×", (int(r["seq_len"]), r["ours_ms"]),
                        textcoords="offset points", xytext=(0, -13),
                        fontsize=8, fontweight="bold", color=C_OURS, ha="center")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=10)
    ax.set_xticks(x); ax.set_xticklabels(x)
    ax.set_xlabel("seq_len (prefill, seq_q==seq_kv)"); ax.set_ylabel("latency (ms)")
    ax.set_title("Prefill is COMPUTE-bound — INT8 has no edge here (honest footnote)",
                 fontweight="bold", fontsize=11)
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8, loc="upper left")
    ax.text(0.5, -0.30,
            "Prefill computes a full S×S score matrix → compute-bound, so halving\n"
            "BYTES buys nothing: × (ours/Sage) erodes 1.4→0.9 as seq grows, trailing\n"
            "SOTA@4096. INT8's byte win lives in decode + footprint (panels →), not here.",
            transform=ax.transAxes, fontsize=8.5, ha="center", va="top",
            color="#9A3412", style="italic")


# serving-scale decode configs (skip grid-starved B8); one line per config.
DECODE_CFGS = [(32, 8, 64), (64, 16, 64), (128, 16, 64), (64, 16, 128)]
_DECODE_COLORS = ["#2563EB", "#16A34A", "#7C3AED", "#EA580C"]


def panel_decode_speedup(ax, rows):
    """Decode speedup vs FP16 SDPA across the long-context grid (1K–32K), one
    line per serving config — the win HOLDS (D=64 even grows) as context grows,
    because decode re-streams the whole KV cache every token (bandwidth-bound)."""
    if not rows:
        ax.text(0.5, 0.5, "no attn_decode.csv\n(run bench_attn_decode.py)",
                ha="center", va="center"); ax.set_axis_off(); return
    for (B, H, D), color in zip(DECODE_CFGS, _DECODE_COLORS):
        sub = [r for r in rows if int(r["B"]) == B and int(r["H"]) == H
               and int(r["D"]) == D and isinstance(r.get("speedup_vs_sdpa"), float)]
        sub.sort(key=lambda r: int(r["S"]))
        if not sub:
            continue
        x = [int(r["S"]) for r in sub]
        y = [r["speedup_vs_sdpa"] for r in sub]
        ax.plot(x, y, marker="o", color=color, lw=2, ms=5,
                label=f"B{B} H{H} D{D}")
    ax.axhline(1.0, color="gray", ls="--", lw=1.2)
    ax.set_xscale("log", base=2)
    allx = sorted({int(r["S"]) for r in rows})
    ax.set_xticks(allx); ax.set_xticklabels([f"{s//1024}K" if s >= 1024 else s for s in allx],
                                            fontsize=8)
    ax.set_xlabel("KV context length (decode, seq_q==1)")
    ax.set_ylabel("decode speedup vs FP16 SDPA (>1 = ours faster)")
    ax.set_title("Decode win HOLDS across 1K–32K context — INT8 KV cache",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", title="serving configs", title_fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    ax.text(0.5, -0.30,
            "At serving scale (B≥32) the INT8 decode kernel beats FP16 SDPA 1.2–1.6×\n"
            "at EVERY context length, and D=64 widens with seq — exactly the\n"
            "long-context regime that matters; short-seq prefill (←) does not.",
            transform=ax.transAxes, fontsize=8.5, ha="center", va="top",
            color="#1E3A8A", style="italic")


# KV-footprint config: a 7–13B-class model shard at serving batch. The point is
# structural (½ the bytes/token ⇒ 2× context per card), independent of the kernel.
KV_B, KV_H, KV_D = 32, 32, 128
CARD_GB = 40.0


def panel_kv_footprint(ax):
    """KV-cache footprint vs context length, INT8 vs FP16, against the 40 GB
    card wall. Half the bytes/token ⇒ INT8 fits ~2× the context (or batch) — the
    OOM the FP16 decode path hits at long context turns into a capacity win."""
    seqs = [2 ** k for k in range(12, 19)]            # 4K .. 256K
    per_tok = 2 * KV_B * KV_H * KV_D / 1e9            # K+V, GB per token per byte/elem
    fp16 = [per_tok * 2 * S for S in seqs]
    int8 = [per_tok * 1 * S for S in seqs]
    ax.plot(seqs, fp16, marker="o", color=C_FP16, lw=2, ms=5,
            label="FP16 KV cache (2 B/elem)")
    ax.plot(seqs, int8, marker="o", color=C_OURS, lw=2, ms=5,
            label="INT8 KV cache (1 B/elem, ours)")
    ax.axhline(CARD_GB, color="black", ls="--", lw=1.6)
    ax.text(seqs[0], CARD_GB * 1.08, "A100-40GB card capacity",
            fontsize=8.5, color="black", fontweight="bold")
    max_fp16 = CARD_GB / (per_tok * 2)               # tokens that fit
    max_int8 = CARD_GB / (per_tok * 1)
    for xv, col, tag in [(max_fp16, C_FP16, f"FP16 wall\n≈{max_fp16/1000:.0f}K tok"),
                         (max_int8, C_OURS, f"INT8 wall\n≈{max_int8/1000:.0f}K tok")]:
        ax.axvline(xv, color=col, ls=":", lw=1.5)
        ax.text(xv, CARD_GB * 0.32, tag, color=col, fontsize=8,
                fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=0.8))
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=10)
    ax.set_xticks(seqs); ax.set_xticklabels([f"{s//1024}K" for s in seqs], fontsize=8)
    ax.set_xlabel(f"context length (B={KV_B} H={KV_H} D={KV_D})")
    ax.set_ylabel("KV-cache footprint (GB)")
    ax.set_title("KV footprint vs 40 GB wall — INT8 fits 2× the context",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, which="both", alpha=0.25)
    ax.text(0.5, -0.30,
            "FP16 KV maxes out context where INT8 still has 2× headroom (½ the\n"
            "bytes/token). The FP16 decode path OOMs first; INT8 keeps serving —\n"
            "more context (or 2× the batch) per card is the structural serving win.",
            transform=ax.transAxes, fontsize=8.5, ha="center", va="top",
            color="#1E3A8A", style="italic")


# ────────────────────────────────────────────────────────────────────────
#  Row 5 — END-TO-END BLOCK (the first full-layer integration measurement).
#  LN + residual + QKV/out projections are FP16 & identical in both paths; the
#  delta is purely the attention-core + MLP INT8 swap (bench_block.py).
# ────────────────────────────────────────────────────────────────────────
def _block_rows(rows, regime):
    sub = [r for r in rows if r["regime"] == regime
           and isinstance(r.get("fp16_ms"), float)]
    sub.sort(key=lambda r: int(r["seq"]))
    return sub


def panel_block_prefill(ax, rows):
    sub = _block_rows(rows, "prefill")
    if not sub:
        ax.text(0.5, 0.5, "no block.csv\n(run bench_block.py)",
                ha="center", va="center"); ax.set_axis_off(); return
    x = [int(r["seq"]) for r in sub]
    ax.plot(x, [r["int8_ms"] for r in sub], marker="o", color=C_OURS, lw=2, ms=5,
            label="INT8 block (eager quant glue)")
    ax.plot(x, [r["fp16_ms"] for r in sub], marker="o", color=C_FP16, lw=2, ms=5,
            label="FP16 block (cuBLAS + SDPA)")
    for r in sub:
        ax.annotate(f"{r['fp16_ms']/r['int8_ms']:.2f}×",
                    (int(r["seq"]), r["int8_ms"]), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, fontweight="bold", color=C_OURS, ha="center")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=10)
    ax.set_xticks(x); ax.set_xticklabels(x)
    ax.set_xlabel("seq_len (prefill)"); ax.set_ylabel("block latency (ms)")
    ax.set_title("End-to-end PREFILL block — INT8 loses (compute-bound)",
                 fontweight="bold", fontsize=11)
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8, loc="upper left")
    ax.text(0.5, -0.30,
            "× = FP16/INT8. INT8 loses ~0.52–0.55× — but part is removable eager\n"
            "quant glue, not algorithm. The decompose panel (→) brackets how much.",
            transform=ax.transAxes, fontsize=8.5, ha="center", va="top",
            color="#9A3412", style="italic")


def panel_block_decode(ax, rows):
    sub = _block_rows(rows, "decode")
    if not sub:
        ax.text(0.5, 0.5, "no block.csv", ha="center", va="center")
        ax.set_axis_off(); return
    x = [int(r["seq"]) for r in sub]
    y = [r["fp16_ms"] / r["int8_ms"] for r in sub]
    ax.plot(x, y, marker="o", color=C_OURS, lw=2, ms=6,
            label="INT8 block / FP16 block")
    for xi, v in zip(x, y):
        ax.annotate(f"{v:.2f}×", (xi, v), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, fontweight="bold",
                    color=(C_SOTA if v >= 1 else C_FP16), ha="center")
    ax.axhline(1.0, color="gray", ls="--", lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x); ax.set_xticklabels([f"{s//1024}K" for s in x], fontsize=8)
    ax.set_xlabel("KV context length (decode, seq_q==1, B=128)")
    ax.set_ylabel("end-to-end speedup vs FP16 block")
    ax.set_title("End-to-end DECODE block — INT8 wins past ~8K, grows with context",
                 fontweight="bold", fontsize=11)
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8, loc="upper left")
    ax.text(0.5, -0.30,
            "The ROBUST signal: even carrying the eager quant glue AND hand-kernel\n"
            "vs-cuBLAS handicap, the INT8 block crosses 1.0 at ~8K and reaches\n"
            "1.19× @ 32K — the bandwidth-bound regime where ½ the KV bytes pay off.",
            transform=ax.transAxes, fontsize=8.5, ha="center", va="top",
            color="#1E3A8A", style="italic")


# Measured prefill component split (S=512, bench_block.py diagnostic). The
# "quant glue" slice is the removable integration artifact; "other" folds the
# Wo proj + LN2 + residual not separately timed (closes each bar to its total).
_BLK_DECOMP = {  # (shared, quant_glue, attn, mlp)  -> sums to measured total
    "INT8 block": [0.38, 0.68, 0.13, 0.62],   # = 1.81 ms
    "FP16 block": [0.38, 0.00, 0.10, 0.45],   # = 0.93 ms
}
_DECOMP_CATS = [("shared LN+proj+resid (FP16, both)", "#94A3B8"),
                ("quant/dequant glue (REMOVABLE)",    C_LOWER),
                ("attention core",                    C_SOTA),
                ("MLP",                               C_OURS)]


def panel_block_decompose(ax):
    labels = list(_BLK_DECOMP.keys())
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))
    for i, (cat, color) in enumerate(_DECOMP_CATS):
        vals = np.array([_BLK_DECOMP[l][i] for l in labels])
        hatch = "//" if "REMOVABLE" in cat else None
        ax.bar(x, vals, 0.5, bottom=bottoms, label=cat, color=color,
               hatch=hatch, edgecolor="white")
        for xi, (b, v) in enumerate(zip(bottoms, vals)):
            if v >= 0.08:
                ax.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        bottoms += vals
    # glue-fused optimistic bound for INT8 = total - glue
    fused = _BLK_DECOMP["INT8 block"][0] + _BLK_DECOMP["INT8 block"][2] + _BLK_DECOMP["INT8 block"][3]
    fp16_total = sum(_BLK_DECOMP["FP16 block"])
    ax.axhline(fused, color=C_OURS, ls=":", lw=1.6)
    ax.text(1.02, fused, f"INT8 glue-fused bound ≈{fused:.2f} ms\n(still {fused/fp16_total:.2f}× FP16 → still loses)",
            transform=ax.get_yaxis_transform(), fontsize=8, color=C_OURS,
            va="center", ha="left", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("prefill block latency (ms), S=512")
    ax.set_ylim(0, 2.1)
    ax.set_title("Is it fair? — where the prefill time goes",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    ax.text(0.5, -0.30,
            "Honest bracket: removing the eager quant glue (hatched, an integration\n"
            "artifact) still leaves INT8 > FP16 — the kernels run ~0.7–0.8× cuBLAS at\n"
            "prefill. Glue-fusion helps DECODE, not prefill. (FP16 ref = cuBLAS/SDPA.)",
            transform=ax.transAxes, fontsize=8.5, ha="center", va="top",
            color="#9A3412", style="italic")


def main():
    mlp = read_sweep("int8_mlp_sweep.csv")
    attn = read_sweep("int8_attn_sweep.csv")
    ncu_mlp = read_ncu("ncu_mlp_raw.csv")
    ncu_attn = read_ncu("ncu_attn_raw.csv")
    sota   = read_sweep("attn_sota.csv")   if os.path.exists(os.path.join(RES, "attn_sota.csv"))   else []
    decode = read_sweep("attn_decode.csv") if os.path.exists(os.path.join(RES, "attn_decode.csv")) else []
    block  = read_sweep("block.csv")       if os.path.exists(os.path.join(RES, "block.csv"))       else []

    fig = plt.figure(figsize=(19, 27.5))
    gs = GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.26,
                  top=0.912, bottom=0.03, left=0.10, right=0.98)

    # Row 1 — INT8 MLP
    mlp_lat = [("kernel_ms", "INT8 fused (ours)", C_OURS),
               ("naive_ms",  "FP16 cuBLAS (torch)", C_FP16),
               ("ref2_ms",   "cuBLAS INT8 pipeline", C_SOTA),
               ("ref3_ms",   "cuBLAS INT8 2×GEMM (floor)", C_LOWER)]
    panel_latency(fig.add_subplot(gs[0, 0]), mlp, mlp_lat,
                  "INT8 MLP — latency (d_model=1024, batch=8)")
    mlp_bytes = [("FP16 dense, unfused", C_FP16, mlp_bytes_fp16_unfused),
                 ("INT8 pipeline (_int_mm+dequant), est.", C_SOTA, mlp_bytes_int8_pipeline),
                 ("INT8 fused (ours)", C_OURS, mlp_bytes_int8_fused)]
    panel_bytes(fig.add_subplot(gs[0, 1]), mlp, mlp_bytes,
                "INT8 MLP — HBM bytes moved  (memory-bound: less = faster)",
                "Fusion + INT8 move ~5× fewer bytes than the _int_mm pipeline\n"
                "→ the measured 3.8–4.8× speedup (right). Compute util stays 22–30%.")
    mlp_sp = [("speedup_vs_naive", "vs FP16 cuBLAS", C_FP16),
              ("speedup_vs_ref2",  "vs cuBLAS INT8 pipeline", C_SOTA),
              ("speedup_vs_ref3",  "vs cuBLAS INT8 2×GEMM", C_LOWER)]
    panel_speedup(fig.add_subplot(gs[0, 2]), mlp, mlp_sp,
                  "INT8 MLP — speedup vs each reference")

    # Row 2 — INT8 attention
    at_lat = [("kernel_ms", "INT8 fused (ours)", C_OURS),
              ("naive_ms",  "FP16 manual attn", C_FP16),
              ("ref2_ms",   "FlashAttn-2 (SDPA)", C_FA2)]
    panel_latency(fig.add_subplot(gs[1, 0]), attn, at_lat,
                  "INT8 attention — latency (d_model=1024, batch=8)")
    at_bytes = [("FP16 manual (writes S×S)", C_FP16, attn_bytes_fp16_unfused),
                ("FlashAttn-2 (FP16, fused)", C_FA2, attn_bytes_fp16_fused),
                ("INT8 fused (ours)", C_OURS, attn_bytes_int8_fused)]
    panel_bytes(fig.add_subplot(gs[1, 1]), attn, at_bytes,
                "INT8 attention — HBM bytes moved  (memory-bound)",
                "Ours moves the fewest bytes (INT8 QKV, no S×S), yet loses to FA2 on\n"
                "prefill latency → not bandwidth-bound here; INT8's byte win pays off\n"
                "in the KV-cache (decode) & footprint, not prefill vs FA2.")
    at_sp = [("speedup_vs_naive", "vs FP16 manual", C_FP16),
             ("speedup_vs_ref2",  "vs FlashAttn-2", C_FA2)]
    panel_speedup(fig.add_subplot(gs[1, 2]), attn, at_sp,
                  "INT8 attention — speedup vs each reference")

    # Row 3 — profiling
    panel_stalls(fig.add_subplot(gs[2, 0]), ncu_mlp, ncu_attn)
    panel_kerneltime(fig.add_subplot(gs[2, 1]))
    panel_journey(fig.add_subplot(gs[2, 2]))

    # Row 4 — the long-context story: prefill (compute-bound, honest footnote) →
    # decode win holds across 1K–32K → KV footprint vs the 40 GB card wall.
    panel_sota_prefill(fig.add_subplot(gs[3, 0]), sota)
    panel_decode_speedup(fig.add_subplot(gs[3, 1]), decode)
    panel_kv_footprint(fig.add_subplot(gs[3, 2]))

    # Row 5 — END-TO-END BLOCK integration (the first full-layer wall-clock):
    # prefill loses → decode wins past ~8K → "is it fair?" decomposition.
    panel_block_prefill(fig.add_subplot(gs[4, 0]), block)
    panel_block_decode(fig.add_subplot(gs[4, 1]), block)
    panel_block_decompose(fig.add_subplot(gs[4, 2]))

    fig.suptitle("INT8 Transformer Kernels — Performance & Profiling Dashboard "
                 "(A100-SXM4-40GB, sm_80)\n"
                 "line panels: batch=8, d_model=1024 (head_dim=128);  "
                 "profiling shape b=8 s=512 d_model=1024 d_ff=4096",
                 fontsize=15, fontweight="bold", y=0.985)
    # memory-bound thesis banner — the lens for reading the whole sheet
    fig.text(0.5, 0.948,
             "▸ MEMORY-BOUND, not compute-bound:  the win is FEWER HBM BYTES — "
             "INT8 halves every tensor (1 B vs FP16 2 B) and fusion removes the "
             "INT32/FP16 intermediate round-trips a separate _int_mm + "
             "dequant/GELU/requant pipeline pays.\n"
             "Peak TOPS is NOT the target (tensor-pipe util sits ~28–38%); bytes moved is. "
             "Row ④ = vs the REAL INT8 SOTA (SageAttention) + decode regime; "
             "row ⑤ = first end-to-end BLOCK (INT8 wins only long-context decode; prefill honestly loses).",
             ha="center", va="center", fontsize=10.5, color="#1E3A8A",
             bbox=dict(boxstyle="round,pad=0.5", fc="#EFF6FF", ec="#1E3A8A", lw=1.3))
    # row band labels in the left margin (rotated), centred on each row's actual
    # grid band so they stay aligned no matter how many rows the figure has.
    bottoms, tops, _, _ = gs.get_grid_positions(fig)
    band_txt = ["①  INT8 MLP benchmark",
                "②  INT8 attention benchmark",
                "③  Profiling: ncu stalls · kernel-time · opt journey",
                "④  Long-context: prefill vs SOTA · decode win · KV footprint",
                "⑤  End-to-end block: prefill loss · decode win · fairness bracket"]
    for i, txt in enumerate(band_txt):
        yc = (bottoms[i] + tops[i]) / 2
        fig.text(0.018, yc, txt, fontsize=13, fontweight="bold", color="#374151",
                 rotation=90, va="center", ha="center")

    out = os.path.join(FIGDIR, "int8_dashboard.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
