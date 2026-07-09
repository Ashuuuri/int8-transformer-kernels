"""make_roofline.py — A100 roofline chart of the INT8 optimization journey.

One picture for the repo's thesis ("win on bytes, not TOPS") and for the decode
optimization journey (iters 14 → 15 → 21), as two panels:

  LEFT  — the full log-log roofline: prefill attention and the MLP sit on the
          compute-bound side of the ridge, where INT8's byte advantage cannot
          help (cuBLAS/FA2 territory — honestly lost); every decode point lives
          in a tiny memory-bound cluster at AI ≈ 1–2 OPs/byte.
  RIGHT — that cluster zoomed, linear axes: FP16 SDPA is already at ~87% of its
          roof, so the only way past it is MORE INTENSITY (halve the bytes:
          INT8 KV doubles AI → doubles the ceiling), and the iter-15/21 kernel
          work climbs toward the raised roof.

Current-kernel points are read from the published snapshot (results/*.csv).
Historical journey points (pre-optimization kernels that no longer ship) are
constants documented against their OPTIMIZATION.md iteration entries.

Usage:  python figures/make_roofline.py   →  results/figures/roofline.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "results", "figures", "roofline.png")

# ── A100-SXM4-40GB ceilings ────────────────────────────────────────────────
HBM_TBPS        = 1.555   # 40 GB SXM4 part (NOT the 2.0 TB/s 80 GB part)
INT8_TOPS       = 624.0   # tensor-core INT8 peak
ATTN_MIX_TOPS   = 416.0   # blended peak for INT8 QK^T + FP16 PV (see sweep.py)

# ── measured decode shape used for the journey points ─────────────────────
B, H, S = 64, 16, 32768
BH = B * H

INK, MUTED, LEADER = "#0F172A", "#64748B", "#94A3B8"
BLUE, GREEN, RED, PURPLE = "#2563EB", "#16A34A", "#DC2626", "#9333EA"


def _decode_flops(D):          # QK^T + P·V, 2 FLOPs per MAC each
    return 4.0 * BH * S * D

def _decode_bytes_int8(D):     # K+V int8 + per-token f32 scales (Q/out negligible)
    return 2.0 * BH * S * D + 2.0 * BH * S * 4

def _decode_bytes_fp16(D):
    return 4.0 * BH * S * D

def _pt(flops, bytes_, ms):    # → (arithmetic intensity, achieved TOPS)
    return flops / bytes_, flops / (ms * 1e-3) / 1e12


def _csv_rows(path):
    with open(os.path.join(ROOT, "results", path)) as f:
        return list(csv.DictReader(f))


def _decode_ms(rows, D):
    for r in rows:
        if (int(r["B"]), int(r["H"]), int(r["D"]), int(r["S"])) == (B, H, D, S):
            return float(r["ours_ms"]), (float(r["sdpa_ms"]) if r["sdpa_ms"] else None)
    raise KeyError(f"decode row B{B} H{H} D{D} S{S} not in attn_decode.csv")


def main():
    dec = _csv_rows("attn_decode.csv")
    d64_ms,  _        = _decode_ms(dec, 64)
    d128_ms, sdpa_ms  = _decode_ms(dec, 128)

    # Current kernels (published snapshot).
    d64_now   = _pt(_decode_flops(64),  _decode_bytes_int8(64),  d64_ms)
    d128_now  = _pt(_decode_flops(128), _decode_bytes_int8(128), d128_ms)
    # FP16 SDPA decode reference: same FLOPs, 2 B/elem KV → half the intensity.
    sdpa_ref  = _pt(_decode_flops(128), _decode_bytes_fp16(128), sdpa_ms)

    # Historical journey points (documented in OPTIMIZATION.md):
    #  - iter 14: D=64 split-KV v2 ran at ~390 GB/s (lane-per-dim, shuffle-bound).
    #  - iter 21: the D=128 lane-per-dim baseline measured 10.204 ms at this
    #    shape (median of 5) immediately before the batched kernel landed.
    d64_v1_ms = _decode_bytes_int8(64) / 390e9 * 1e3
    d64_v1    = _pt(_decode_flops(64),  _decode_bytes_int8(64),  d64_v1_ms)
    d128_v0   = _pt(_decode_flops(128), _decode_bytes_int8(128), 10.204)

    # Compute-side points from the sweeps (graded head_dim=64 attention shape;
    # the largest MLP shape). AI from the same byte models as sweep.py.
    attn = [r for r in _csv_rows("int8_attn_sweep.csv")
            if r["seq_len"] == "4096" and r["d_model"] == "512"][0]
    T, Sq, Dh, Hh = 8, 4096, 64, 8
    attn_flops = 2.0 * T * Hh * Sq * Dh * Sq * 2
    attn_bytes = T * Hh * Sq * Dh * 3 + T * Hh * Sq * Dh * 2 + T * Hh * Sq * 12
    attn_pt = _pt(attn_flops, attn_bytes, float(attn["kernel_ms"]))

    mlp = [r for r in _csv_rows("int8_mlp_sweep.csv")
           if r["seq_len"] == "4096" and r["d_model"] == "2048"][0]
    Tm, dm, dff = 8 * 4096, 2048, 8192
    mlp_flops = 4.0 * Tm * dm * dff
    mlp_bytes = (Tm * dm + 2.0 * dm * dff + Tm * dff) + (Tm * dff + Tm * dm)
    mlp_pt = _pt(mlp_flops, mlp_bytes, float(mlp["kernel_ms"]))

    pct = lambda p: f"{p[1] / (HBM_TBPS * p[0]) * 100:.0f}% of roof"

    # ── figure: full roofline (log) + decode zoom (linear) ────────────────
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13.6, 5.9), gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- LEFT: full picture -----------------------------------------------
    ai = np.logspace(-1, 12.4, 400, base=2)
    axL.plot(ai, np.minimum(HBM_TBPS * ai, INT8_TOPS), color="#334155", lw=2.0,
             label=f"HBM {HBM_TBPS:.3f} TB/s · INT8 {INT8_TOPS:.0f} TOPS")
    axL.plot(ai, np.minimum(HBM_TBPS * ai, ATTN_MIX_TOPS), color=LEADER,
             lw=1.3, ls=":",
             label=f"mixed INT8/FP16 attention roof ({ATTN_MIX_TOPS:.0f} TOPS)")
    ridge = INT8_TOPS / HBM_TBPS
    axL.axvline(ridge, color="#CBD5E1", lw=1.0, ls="--", zorder=1)
    axL.text(ridge * 0.80, 1.1, f"ridge ≈ {ridge:.0f} OPs/B", color=MUTED,
             fontsize=8.5, rotation=90, va="bottom", ha="right")

    axL.plot(*attn_pt, "s", color=RED, ms=9, zorder=5)
    axL.annotate(f"prefill attention\n{attn_pt[1]:.0f}/{ATTN_MIX_TOPS:.0f} TOPS",
                 attn_pt, xytext=(attn_pt[0] * 0.72, attn_pt[1] * 1.0),
                 fontsize=8.6, color=INK, ha="right", va="center")
    axL.plot(*mlp_pt, "s", color=PURPLE, ms=9, zorder=5)
    axL.annotate(f"fused INT8 MLP\n{mlp_pt[1]:.0f}/{INT8_TOPS:.0f} TOPS",
                 mlp_pt, xytext=(mlp_pt[0] * 0.78, mlp_pt[1] * 1.9),
                 fontsize=8.6, color=INK, ha="right", va="center")

    for p, c in [(sdpa_ref, MUTED), (d64_v1, GREEN), (d64_now, GREEN),
                 (d128_v0, BLUE), (d128_now, BLUE)]:
        axL.plot(*p, "o", color=c, ms=5, zorder=5)
    zoom = mpatches.Rectangle((0.80, 0.52), 1.85, 2.85, fill=False,
                              edgecolor=MUTED, lw=1.1, ls="--", zorder=4)
    axL.add_patch(zoom)
    axL.annotate("decode cluster\n(zoomed →)", (2.65, 3.0), xytext=(6.5, 8.0),
                 fontsize=9, color=MUTED, ha="left",
                 arrowprops=dict(arrowstyle="-", color=LEADER, lw=0.9))

    axL.text(1.15, 200, "memory-bound", color=MUTED, fontsize=9.5)
    axL.text(2 ** 12.2, 2.6, "compute-bound\n(cuBLAS & FA2\nterritory —\nhonestly lost)",
             color=MUTED, fontsize=9, ha="right")

    axL.set_xscale("log", base=2)
    axL.set_yscale("log", base=10)
    axL.set_xlim(0.6, 2 ** 12.4)
    axL.set_ylim(0.4, 1100)
    axL.set_xlabel("arithmetic intensity (OPs / HBM byte)")
    axL.set_ylabel("achieved throughput (TOPS)")
    axL.set_title("Full roofline — where each kernel lives")
    axL.grid(True, which="major", alpha=0.25)
    axL.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # ---- RIGHT: decode zoom (linear axes, room for the journey) ----------
    x = np.linspace(0.55, 2.55, 64)
    axR.plot(x, HBM_TBPS * x, color="#334155", lw=2.0)
    axR.text(1.30, HBM_TBPS * 1.30 + 0.16, "HBM roof (1.555 TB/s × AI)",
             color="#334155", fontsize=8.8, rotation=40,
             rotation_mode="anchor", va="bottom")

    # ceilings at the two intensities: FP16 KV (AI≈1) vs INT8 KV (AI≈2)
    axR.annotate("", xy=(d128_now[0], HBM_TBPS * d128_now[0]),
                 xytext=(sdpa_ref[0], HBM_TBPS * sdpa_ref[0]),
                 arrowprops=dict(arrowstyle="-|>", color=LEADER, lw=1.3, ls="--"))
    axR.text(0.60, 3.32, "INT8 KV: ½ the bytes → 2× the intensity\n"
                         "→ 2× the attainable ceiling",
             fontsize=9, color="#475569", ha="left")

    def dotR(p, color, label, txt_xy, note=None, ha="left", ms=10):
        axR.plot(*p, "o", color=color, ms=ms, zorder=5)
        txt = label if note is None else f"{label}\n{note}"
        axR.annotate(txt, p, xytext=txt_xy, fontsize=9, color=INK,
                     va="center", ha=ha, zorder=6,
                     arrowprops=dict(arrowstyle="-", color=LEADER, lw=0.9,
                                     shrinkA=1, shrinkB=5))

    def climbR(p0, p1, color):
        axR.annotate("", xy=p1, xytext=p0, zorder=4,
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                                     shrinkA=7, shrinkB=7))

    # FP16 reference (left of the wedge)
    dotR(sdpa_ref, MUTED, "FP16 SDPA decode", (0.60, 0.72), note=pct(sdpa_ref))

    # D=64 journey — labels in the lower-right open space
    dotR(d64_v1,  GREEN, "D=64 split-KV v1 (iter 14)", (2.02, 0.42),
         note=pct(d64_v1))
    dotR(d64_now, GREEN, "D=64 dp4a (iter 15)", (2.02, 1.06), note=pct(d64_now))
    climbR(d64_v1, d64_now, GREEN)

    # D=128 journey — labels stacked above the D=64 ones
    dotR(d128_v0,  BLUE, "D=128 lane-per-dim (iters 14–20)", (2.02, 1.62),
         note=pct(d128_v0))
    dotR(d128_now, BLUE, "D=128 batched (iter 21)", (2.02, 2.55),
         note=pct(d128_now))
    climbR(d128_v0, d128_now, BLUE)

    axR.set_xlim(0.55, 2.55)
    axR.set_ylim(0.0, 3.6)
    axR.set_xlabel("arithmetic intensity (OPs / HBM byte)")
    axR.set_ylabel("achieved TOPS")
    axR.set_title(f"Decode zoom — the journey (B={B} H={H} S={S // 1024}K)")
    axR.grid(True, alpha=0.25)

    fig.suptitle("A100-40GB roofline — the INT8 journey: win on bytes, not TOPS",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"[ok] {OUT}")


if __name__ == "__main__":
    main()
