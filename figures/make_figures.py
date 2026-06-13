#!/usr/bin/env python3
"""Clean, seaborn-styled figures for the repo showcase.

Replaces the single cluttered 15-panel int8_dashboard.png with:

  figures/hero.png          ONE 2x2 headline — the whole value proposition:
                            (1) end-to-end block: INT8 wins decode, loses prefill (honest)
                            (2) decode speedup holds across 1K-32K serving configs
                            (3) KV-cache footprint vs the 40 GB card wall (fit 2x context)
                            (4) accuracy preserved (cosine vs FP32 stays >= 0.998)

  figures/mlp.png           themed detail: INT8 MLP latency / bytes / speedup
  figures/attention.png     themed detail: INT8 attention latency / bytes / speedup
  figures/profiling.png     themed detail: ncu stalls / kernel-time / opt journey
  figures/long_context.png  themed detail: prefill-vs-SOTA / decode / KV footprint
  figures/block.png         themed detail: block prefill / decode / fairness bracket

Design rules (why this is less cluttered than the mega-dashboard):
  - ONE story per figure, not 15 panels on one sheet.
  - seaborn theme = consistent palette, fonts, grid (no hand-rolled chartjunk).
  - the hero carries NO paragraph footnotes — just a one-line caption per panel;
    the long explanations live in the themed figures / README, not the headline.

All data + math helpers are reused from make_dashboard.py (single source of truth
for byte models, CSV parsing, ncu parsing and the detailed themed panels).
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import make_dashboard as D  # reuse read_sweep/read_ncu/byte models/colors/panels

FIGDIR = D.FIGDIR


def _theme():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0,
                  rc={"axes.titleweight": "bold", "axes.titlesize": 11.5,
                      "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
                      "ytick.labelsize": 8.5, "legend.fontsize": 8,
                      "axes.edgecolor": "#334155", "axes.linewidth": 1.0,
                      "grid.color": "#E2E8F0", "legend.frameon": True,
                      "legend.framealpha": 0.9, "savefig.dpi": 150,
                      "figure.facecolor": "white", "axes.facecolor": "white"})


def _kfmt(s):
    return f"{s // 1024}K" if s >= 1024 else str(s)


# ──────────────────────────────────────────────────────────────────────────
#  HERO — the four panels that carry the entire pitch.
# ──────────────────────────────────────────────────────────────────────────
def hero_block_story(ax, block):
    """The honest one-glance summary: end-to-end block speedup vs FP16, prefill
    (compute-bound, loses) and decode (bandwidth-bound, wins and grows)."""
    pre = D._block_rows(block, "prefill")
    dec = D._block_rows(block, "decode")
    ax.axhspan(0, 1.0, color="#FEE2E2", alpha=0.45, zorder=0)
    ax.axhspan(1.0, 1.6, color="#DCFCE7", alpha=0.45, zorder=0)
    ax.axhline(1.0, color="#64748B", ls="--", lw=1.3, zorder=1)
    if pre:
        xp = [int(r["seq"]) for r in pre]
        yp = [r["speedup"] for r in pre]
        ax.plot(xp, yp, marker="s", color=D.C_FP16, lw=2.4, ms=7,
                label="prefill (compute-bound)", zorder=3)
    if dec:
        xd = [int(r["seq"]) for r in dec]
        yd = [r["speedup"] for r in dec]
        ax.plot(xd, yd, marker="o", color=D.C_OURS, lw=2.4, ms=7,
                label="decode (bandwidth-bound)", zorder=3)
        ax.annotate(f"{yd[-1]:.2f}×", (xd[-1], yd[-1]), textcoords="offset points",
                    xytext=(-4, 10), fontsize=9, fontweight="bold", color=D.C_OURS)
    ax.set_xscale("log", base=2)
    allx = sorted({int(r["seq"]) for r in pre + dec})
    ax.set_xticks(allx); ax.set_xticklabels([_kfmt(s) for s in allx], fontsize=8)
    ax.set_xlabel("sequence / KV-context length")
    ax.set_ylabel("end-to-end speedup vs FP16 block")
    ax.set_title("INT8 wins the memory-bound regime (decode), not prefill")
    ax.legend(fontsize=8, loc="upper left")


def hero_decode_configs(ax, decode):
    """The win is not one lucky shape: decode speedup holds across serving
    configs and grows with context."""
    for (B, H, Dd), color in zip(D.DECODE_CFGS, D._DECODE_COLORS):
        sub = [r for r in decode if int(r["B"]) == B and int(r["H"]) == H
               and int(r["D"]) == Dd and isinstance(r.get("speedup_vs_sdpa"), float)]
        sub.sort(key=lambda r: int(r["S"]))
        if not sub:
            continue
        ax.plot([int(r["S"]) for r in sub], [r["speedup_vs_sdpa"] for r in sub],
                marker="o", color=color, lw=2.2, ms=6, label=f"B{B} H{H} D{Dd}")
    ax.axhline(1.0, color="#64748B", ls="--", lw=1.3)
    ax.set_xscale("log", base=2)
    allx = sorted({int(r["S"]) for r in decode})
    ax.set_xticks(allx); ax.set_xticklabels([_kfmt(s) for s in allx], fontsize=8)
    ax.set_xlabel("KV context length (decode, seq_q==1)")
    ax.set_ylabel("decode kernel speedup vs FP16 SDPA")
    ax.set_title("Decode win holds across serving configs (1K–32K)")
    ax.legend(fontsize=7.5, loc="lower right", title="config", title_fontsize=8)


def hero_kv(ax):
    """Structural memory win — independent of kernel quality: half the bytes per
    token ⇒ INT8 fits ~2× the context before the 40 GB wall."""
    seqs = [2 ** k for k in range(12, 19)]
    per_tok = 2 * D.KV_B * D.KV_H * D.KV_D / 1e9
    ax.plot(seqs, [per_tok * 2 * s for s in seqs], marker="o", color=D.C_FP16,
            lw=2.2, ms=6, label="FP16 KV (2 B/elem)")
    ax.plot(seqs, [per_tok * 1 * s for s in seqs], marker="o", color=D.C_OURS,
            lw=2.2, ms=6, label="INT8 KV (1 B/elem)")
    ax.axhline(D.CARD_GB, color="black", ls="--", lw=1.5)
    ax.text(seqs[0], D.CARD_GB * 1.1, "A100-40GB", fontsize=8, fontweight="bold")
    for xv, col in [(D.CARD_GB / (per_tok * 2), D.C_FP16),
                    (D.CARD_GB / (per_tok * 1), D.C_OURS)]:
        ax.axvline(xv, color=col, ls=":", lw=1.4)
        ax.text(xv, D.CARD_GB * 0.30, f"≈{xv/1000:.0f}K\ntok", color=col, fontsize=7.5,
                fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=0.8))
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=10)
    ax.set_xticks(seqs); ax.set_xticklabels([_kfmt(s) for s in seqs], fontsize=8)
    ax.set_xlabel(f"context length (B={D.KV_B} H={D.KV_H} D={D.KV_D})")
    ax.set_ylabel("KV-cache footprint (GB)")
    ax.set_title("INT8 KV cache fits 2× the context per card")
    ax.legend(fontsize=8, loc="upper left")


def hero_accuracy(ax, block):
    """Wins without losing accuracy: cosine vs an FP32 reference stays ≥0.998
    across both regimes, every length."""
    pre = D._block_rows(block, "prefill")
    dec = D._block_rows(block, "decode")
    bars, labels, colors = [], [], []
    for r in pre:
        bars.append(r["cos"]); labels.append(f"pf {_kfmt(int(r['seq']))}"); colors.append(D.C_FP16)
    for r in dec:
        bars.append(r["cos"]); labels.append(f"dc {_kfmt(int(r['seq']))}"); colors.append(D.C_OURS)
    x = np.arange(len(bars))
    ax.bar(x, bars, 0.7, color=colors, edgecolor="white")
    ax.axhline(0.998, color="#EAB308", ls="--", lw=1.6)
    ax.text(len(bars) - 0.5, 0.9982, "0.998 gate", fontsize=8, color="#EAB308",
            ha="right", va="bottom", fontweight="bold")
    lo = min(bars) - 0.0015
    ax.set_ylim(max(0.995, lo), 1.0005)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5, rotation=0)
    ax.set_ylabel("cosine vs FP32 reference")
    ax.set_title("Accuracy preserved end-to-end (cos ≥ 0.9986)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=D.C_FP16, label="prefill"),
                       Patch(color=D.C_OURS, label="decode")],
              fontsize=8, loc="lower right")


def fig_hero(block, decode):
    _theme()
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5))
    hero_block_story(axes[0, 0], block)
    hero_decode_configs(axes[0, 1], decode)
    hero_kv(axes[1, 0])
    hero_accuracy(axes[1, 1], block)
    fig.suptitle("INT8 Transformer Kernels — wins the memory-bound long-context regime",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.6, w_pad=2.5)
    out = os.path.join(FIGDIR, "hero.png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.45); plt.close(fig)
    print(f"[ok] {out}")


# ──────────────────────────────────────────────────────────────────────────
#  THEMED detail figures — reuse the detailed panels from make_dashboard, one
#  story per figure (1x3), seaborn-themed. These carry the full footnotes.
# ──────────────────────────────────────────────────────────────────────────
def _strip_captions(ax):
    """Remove the explanatory footnote paragraphs the reused make_dashboard
    panels draw below their axes (transAxes y≈-0.30). The prose lives in the
    README, not on the image. In-axes texts (data coords, all y>0) are kept."""
    for t in list(ax.texts):
        if t.get_position()[1] < -0.05:
            t.remove()


def _headroom(ax, frac=0.16):
    """Expand the top y-limit so above-point annotation labels (e.g. "0.55×")
    stay inside the axes instead of spilling over the top edge."""
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log" and lo > 0:
        ax.set_ylim(lo, hi * (hi / lo) ** frac)
    else:
        ax.set_ylim(lo, hi + (hi - lo) * frac)


# ──────────────────────────────────────────────────────────────────────────
#  DECODE vs quantized-KV SOTA (FlashInfer FP8) — the honest peer comparison.
# ──────────────────────────────────────────────────────────────────────────
_SOTA_CFGS = [(32, 8, 64), (64, 16, 64), (128, 16, 64), (64, 16, 128)]
_SOTA_COLORS = ["#2563EB", "#16A34A", "#7C3AED", "#DC2626"]  # D=128 = red (the loss)


def _sota_sub(rows, B, H, Dd, col):
    sub = [r for r in rows if int(r["B"]) == B and int(r["H"]) == H
           and int(r["D"]) == Dd and isinstance(r.get(col), float)]
    sub.sort(key=lambda r: int(r["S"]))
    return sub


def panel_sota_speedup(ax, rows, col, title, ylabel):
    ax.axhspan(0, 1.0, color="#FEE2E2", alpha=0.4, zorder=0)
    ax.axhline(1.0, color="#64748B", ls="--", lw=1.3, zorder=1)
    allx = set()
    for (B, H, Dd), color in zip(_SOTA_CFGS, _SOTA_COLORS):
        sub = _sota_sub(rows, B, H, Dd, col)
        if not sub:
            continue
        x = [int(r["S"]) for r in sub]; allx.update(x)
        ax.plot(x, [r[col] for r in sub], marker=("s" if Dd == 128 else "o"),
                color=color, lw=2.2, ms=6,
                label=f"B{B} H{H} D{Dd}" + (" (untuned)" if Dd == 128 else ""))
    ax.set_xscale("log", base=2)
    allx = sorted(allx)
    ax.set_xticks(allx); ax.set_xticklabels([_kfmt(s) for s in allx], fontsize=8)
    ax.set_xlabel("KV context length (decode, seq_q==1)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="lower left", title="config", title_fontsize=8)


def panel_sota_accuracy(ax, rows):
    cfgs, ours, fp8 = [], [], []
    for (B, H, Dd) in _SOTA_CFGS:
        oc = [r["ours_cos"] for r in rows if int(r["B"]) == B and int(r["H"]) == H
              and int(r["D"]) == Dd and isinstance(r.get("ours_cos"), float)]
        fc = [r["fi_fp8_cos"] for r in rows if int(r["B"]) == B and int(r["H"]) == H
              and int(r["D"]) == Dd and isinstance(r.get("fi_fp8_cos"), float)]
        if oc and fc:
            cfgs.append(f"B{B}\nD{Dd}"); ours.append(sum(oc)/len(oc)); fp8.append(sum(fc)/len(fc))
    x = np.arange(len(cfgs)); w = 0.38
    ax.bar(x - w/2, ours, w, color=D.C_OURS, label="ours INT8 (per-token)", edgecolor="white")
    ax.bar(x + w/2, fp8, w, color=D.C_LOWER, label="FlashInfer FP8 (per-tensor)", edgecolor="white")
    lo = min(ours + fp8) - 0.0006
    ax.set_ylim(max(0.998, lo), 1.00005)
    ax.set_xticks(x); ax.set_xticklabels(cfgs, fontsize=8)
    ax.set_ylabel("cosine vs FP32 reference")
    ax.set_title("More accurate at equal bytes (per-token INT8 > per-tensor FP8)")
    ax.legend(fontsize=8, loc="lower right")


def _themed(name, title, draw):
    _theme()
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2))
    draw(axes)
    for ax in axes:
        _strip_captions(ax)
        _headroom(ax)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95), w_pad=2.5)
    out = os.path.join(FIGDIR, name)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.35); plt.close(fig)
    print(f"[ok] {out}")


def main():
    mlp = D.read_sweep("int8_mlp_sweep.csv")
    attn = D.read_sweep("int8_attn_sweep.csv")
    ncu_mlp = D.read_ncu("ncu_mlp_raw.csv")
    ncu_attn = D.read_ncu("ncu_attn_raw.csv")
    sota = D.read_sweep("attn_sota.csv") if os.path.exists(os.path.join(D.RES, "attn_sota.csv")) else []
    decode = D.read_sweep("attn_decode.csv") if os.path.exists(os.path.join(D.RES, "attn_decode.csv")) else []
    block = D.read_sweep("block.csv") if os.path.exists(os.path.join(D.RES, "block.csv")) else []
    dsota = D.read_sweep("decode_sota.csv") if os.path.exists(os.path.join(D.RES, "decode_sota.csv")) else []

    # 1) the headline
    fig_hero(block, decode)

    # 2) themed detail figures
    _themed("mlp.png", "INT8 MLP — latency · HBM bytes · speedup vs each reference",
            lambda ax: (
                D.panel_latency(ax[0], mlp,
                    [("kernel_ms", "INT8 fused (ours)", D.C_OURS),
                     ("naive_ms", "FP16 cuBLAS (torch)", D.C_FP16),
                     ("ref2_ms", "cuBLAS INT8 pipeline", D.C_SOTA),
                     ("ref3_ms", "cuBLAS INT8 2×GEMM (floor)", D.C_LOWER)],
                    "Latency (d_model=1024, batch=8)"),
                D.panel_bytes(ax[1], mlp,
                    [("FP16 dense, unfused", D.C_FP16, D.mlp_bytes_fp16_unfused),
                     ("INT8 pipeline (_int_mm+dequant), est.", D.C_SOTA, D.mlp_bytes_int8_pipeline),
                     ("INT8 fused (ours)", D.C_OURS, D.mlp_bytes_int8_fused)],
                    "HBM bytes moved  (memory-bound: less = faster)",
                    "Fusion + INT8 move ~5× fewer bytes than the _int_mm pipeline\n"
                    "→ the measured 3.8–4.8× speedup (right)."),
                D.panel_speedup(ax[2], mlp,
                    [("speedup_vs_naive", "vs FP16 cuBLAS", D.C_FP16),
                     ("speedup_vs_ref2", "vs cuBLAS INT8 pipeline", D.C_SOTA),
                     ("speedup_vs_ref3", "vs cuBLAS INT8 2×GEMM", D.C_LOWER)],
                    "Speedup vs each reference")))

    # 3) attention
    _themed("attention.png", "INT8 attention — latency · HBM bytes · speedup vs each reference",
            lambda ax: (
                D.panel_latency(ax[0], attn,
                    [("kernel_ms", "INT8 fused (ours)", D.C_OURS),
                     ("naive_ms", "FP16 manual attn", D.C_FP16),
                     ("ref2_ms", "FlashAttn-2 (SDPA)", D.C_FA2)],
                    "Latency (d_model=1024, batch=8)"),
                D.panel_bytes(ax[1], attn,
                    [("FP16 manual (writes S×S)", D.C_FP16, D.attn_bytes_fp16_unfused),
                     ("FlashAttn-2 (FP16, fused)", D.C_FA2, D.attn_bytes_fp16_fused),
                     ("INT8 fused (ours)", D.C_OURS, D.attn_bytes_int8_fused)],
                    "HBM bytes moved  (memory-bound)",
                    "Ours moves the fewest bytes but loses to FA2 on prefill latency →\n"
                    "INT8's byte win pays off in decode + KV footprint, not prefill."),
                D.panel_speedup(ax[2], attn,
                    [("speedup_vs_naive", "vs FP16 manual", D.C_FP16),
                     ("speedup_vs_ref2", "vs FlashAttn-2", D.C_FA2)],
                    "Speedup vs each reference")))

    # 4) profiling
    _themed("profiling.png", "Profiling — ncu stall breakdown · kernel-time split · optimisation journey",
            lambda ax: (D.panel_stalls(ax[0], ncu_mlp, ncu_attn),
                        D.panel_kerneltime(ax[1]),
                        D.panel_journey(ax[2])))

    # 5) long-context
    _themed("long_context.png", "Long-context — prefill vs INT8 SOTA · decode win · KV footprint",
            lambda ax: (D.panel_sota_prefill(ax[0], sota),
                        D.panel_decode_speedup(ax[1], decode),
                        D.panel_kv_footprint(ax[2])))

    # 6) end-to-end block
    _themed("block.png", "End-to-end block — prefill loss · decode win · fairness bracket",
            lambda ax: (D.panel_block_prefill(ax[0], block),
                        D.panel_block_decode(ax[1], block),
                        D.panel_block_decompose(ax[2])))

    # 7) decode vs quantized-KV SOTA (FlashInfer FP8) — the honest peer
    if dsota:
        _themed("decode_sota.png",
                "Decode vs a real quantized-KV SOTA (FlashInfer FP8, equal KV bytes)",
                lambda ax: (
                    panel_sota_speedup(ax[0], dsota, "spd_vs_fp8",
                        "vs FP8 SOTA — D=64 par/win, D=128 trails (untuned)",
                        "speedup vs FlashInfer FP8 (>1 = ours faster)"),
                    panel_sota_speedup(ax[1], dsota, "spd_vs_fp16",
                        "vs FP16 (FlashInfer, 2× the KV bytes)",
                        "speedup vs FlashInfer FP16"),
                    panel_sota_accuracy(ax[2], dsota)))


if __name__ == "__main__":
    main()
