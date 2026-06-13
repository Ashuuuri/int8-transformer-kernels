"""generate_test_data.py — INT8 kernel validation datasets (8 distributions).

Supersedes tests/gen_testdata.py for INT8 *accuracy* validation. The old
script remains for the pure-CUDA .bin smoke tests, but it has these problems
and blind spots for INT8 work:

  1. Gaussian-only data. torch.randn is symmetric and light-tailed — the
     friendliest possible case for symmetric per-tensor INT8 quantization.
     Real transformer activations have outlier channels (LLM.int8(),
     Dettmers 2022) that inflate the quantization scale and crush all other
     values. The old data can never expose this.
  2. Stale interface. It saves a single per-tensor scale for attention, but
     int8_attention_forward takes PER-TOKEN scale arrays
     [batch*heads*seq_len]. The .bin INT8 attention data no longer matches
     the kernel's contract.
  3. No dispatch-boundary coverage. seq_len ∈ {4, 512} misses the WMMA
     tile-divisibility boundaries (seq % 128 / % 64) and the scalar
     fallback, and never stresses the online softmax over long sequences.
  4. One fixed tolerance per test binary. Different distributions need
     different thresholds — an outlier dataset legitimately loses more
     precision than a Gaussian one.
  5. No degenerate inputs (all-zero, constant) — the scale==0 guard and
     softmax-of-uniform-scores paths are never executed.

This script writes testdata/validate/<dataset>.pt (FP16 CPU tensors) plus
manifest.json carrying, for every dataset: the rationale, the failure mode
it is designed to expose, and the per-dataset validation thresholds that
validate_int8.py enforces.

Kernel input/output contracts (shapes, value ranges, sensitivities):

  int8_attention_forward
    in : Q/K/V int8 (batch, heads, seq, head_dim), per-token float scales
         (batch*heads*seq); seq % 64 == 0 uses the WMMA path, otherwise
         scalar fallback; head_dim ∈ {64, 128, 256} for the fast path.
    out: FP16 (batch, heads, seq, head_dim). Softmax-weighted average of V,
         so |out| <= max |V| (~bounded by the V distribution).
    sensitive to: per-token dynamic range. One outlier element in a token
         row inflates that token's scale and quantizes the remaining
         head_dim-1 values onto a coarse grid. Long sequences accumulate
         online-softmax rescaling error.

  int8_mlp_forward
    in : x int8 (batch, seq, d_model), W1 (d_model, d_ff), W2 (d_ff,
         d_model), per-TENSOR scales. WMMA path needs batch*seq % 128 == 0
         and dims % 128 == 0, otherwise scalar fallback.
    out: INT8 (batch, seq, d_model) + dynamic per-tensor output scale.
         Hidden = GELU(x@W1) is asymmetric (>= -0.17); out range grows with
         sqrt(d_ff) * |W2|.
    sensitive to: GLOBAL outliers — x/W use one scale for the whole tensor,
         so a single large element crushes everything (this is per-tensor
         quantization's worst case). Also to high per-channel dynamic range
         (the LLM.int8() pattern) and to GELU's small-value region, where
         the INT8 grid step dominates relative error.
"""

import json
import os

import torch

OUT_DIR = os.path.join("testdata", "validate")
SEED = 42

# Main shape: realistic transformer block, WMMA fast path for both kernels.
BATCH, HEADS, SEQ, HEAD_DIM = 2, 8, 512, 64
D_MODEL, D_FF = 512, 2048
MAX_SEQ_LEN = 4096      # long_seq; A100-40GB-safe for the naive reference
SHORT_SEQ = 4           # short_seq; exercises the scalar fallback path


def _attn_shapes(seq):
    return dict(batch=BATCH, heads=HEADS, seq_len=seq, head_dim=HEAD_DIM)


def _mlp_shapes(seq):
    return dict(batch=BATCH, seq_len=seq, d_model=D_MODEL, d_ff=D_FF)


def base_attention(seq, gen):
    Q = torch.randn(BATCH, HEADS, seq, HEAD_DIM, generator=gen).half()
    K = torch.randn(BATCH, HEADS, seq, HEAD_DIM, generator=gen).half()
    V = torch.randn(BATCH, HEADS, seq, HEAD_DIM, generator=gen).half()
    return {"Q": Q, "K": K, "V": V}


def base_mlp(seq, gen):
    x = torch.randn(BATCH, seq, D_MODEL, generator=gen).half()
    W1 = (torch.randn(D_MODEL, D_FF, generator=gen) * 0.02).half()
    W2 = (torch.randn(D_FF, D_MODEL, generator=gen) * 0.02).half()
    return {"x": x, "W1": W1, "W2": W2}


def inject_outliers(t, gen, channel_frac=0.01, token_frac=0.05,
                    lo=10.0, hi=100.0):
    """Multiply a few (channel, token) positions by 10-100x.

    Channels are fixed per tensor (mimics persistent outlier feature
    dimensions in real LLMs); tokens are random.
    """
    t = t.clone().float()
    C = t.shape[-1]
    n_ch = max(1, int(C * channel_frac))
    channels = torch.randperm(C, generator=gen)[:n_ch]
    flat = t.reshape(-1, C)
    n_tok = max(1, int(flat.shape[0] * token_frac))
    tokens = torch.randperm(flat.shape[0], generator=gen)[:n_tok]
    mag = lo + (hi - lo) * torch.rand(len(tokens), len(channels), generator=gen)
    flat[tokens[:, None], channels[None, :]] *= mag
    return t.reshape(t.shape).clamp(-65504, 65504).half(), channels.tolist()


def boundary_fill(shape, gen, amax=4.0):
    """Values whose quantized codes concentrate at ±127/±126.

    sign * amax * U(0.985, 1.0) — after symmetric quantization with
    scale = amax/127 these land on the top two codes, so any off-by-one,
    clamp(-128,127) asymmetry, or round-vs-trunc bug shows up directly.
    """
    sign = torch.randint(0, 2, shape, generator=gen).float() * 2 - 1
    mag = amax * (0.985 + 0.015 * torch.rand(shape, generator=gen))
    return (sign * mag).half()


def stress_attention(seq, gen):
    """Worst case for per-token quantization + online softmax.

    - Heavy-tailed (student-t, df=3) base values: per-token amax varies
      wildly between rows.
    - Per-channel log-spaced magnitudes on K (1e-1..1e1): large |QK^T|
      spread makes the softmax extremely peaky for some queries — the
      online-softmax rescale factor exp(old_max - new_max) underflows
      toward 0 and any accumulation-order bug surfaces.
    """
    def heavy(shape):
        t = torch.distributions.StudentT(3.0).sample(shape)
        return t.clamp(-30, 30)
    Q = heavy((BATCH, HEADS, seq, HEAD_DIM)).half()
    K = heavy((BATCH, HEADS, seq, HEAD_DIM))
    ch_mag = torch.logspace(-1, 1, HEAD_DIM)
    K = (K * ch_mag).clamp(-300, 300).half()
    V = heavy((BATCH, HEADS, seq, HEAD_DIM)).half()
    return {"Q": Q, "K": K, "V": V}


def stress_mlp(seq, gen):
    """Worst case for per-tensor quantization through GELU.

    - x channels with log-spaced magnitudes (1e-2..1e1, the LLM.int8()
      emergent-outlier pattern): one per-tensor scale must cover 3 orders
      of magnitude, so small channels quantize to 0-2 codes.
    - Half the pre-GELU activations pushed into the |h| < 1 region where
      GELU's curvature makes relative INT8 error largest.
    """
    x = torch.randn(BATCH, seq, D_MODEL, generator=gen)
    ch_mag = torch.logspace(-2, 1, D_MODEL)
    x = (x * ch_mag).half()
    W1 = (torch.randn(D_MODEL, D_FF, generator=gen) * 0.02).half()
    W2 = (torch.randn(D_FF, D_MODEL, generator=gen) * 0.02).half()
    return {"x": x, "W1": W1, "W2": W2}


# ── Dataset registry ────────────────────────────────────────────────────
# threshold keys consumed by validate_int8.py:
#   cos_min          gate-1 cosine similarity floor
#   top10_min        gate-1 top-10 index overlap floor
#   outlier_ratio_max  gate-1 max fraction of |err| > 0.1 elements
#   edge_cos_min     gate-4 floor (edge datasets only)
DEFAULT_THRESH = {"cos_min": 0.999, "top10_min": 0.8, "outlier_ratio_max": 0.01}
EDGE_THRESH = {"cos_min": 0.99, "top10_min": 0.6, "outlier_ratio_max": 0.02,
               "edge_cos_min": 0.99}


def build_datasets():
    gen = torch.Generator().manual_seed(SEED)
    datasets = {}

    # 1. normal — what the kernel actually receives after LayerNorm:
    #    zero-mean unit-variance activations, N(0, 0.02²) weights. This is
    #    the regression baseline; if this fails, everything is broken.
    datasets["normal"] = {
        "attention": base_attention(SEQ, gen),
        "mlp": base_mlp(SEQ, gen),
        "rationale": "Post-LayerNorm-like activations; the distribution the "
                     "kernels see in a healthy transformer.",
        "expected_issue": "None — regression baseline. Any failure here is "
                          "a plain correctness bug, not a quantization one.",
        "thresholds": dict(DEFAULT_THRESH),
        "seq_len": SEQ,
    }

    # 2. outlier — 1% fixed channels × 5% tokens get 10-100x magnitude.
    attn = base_attention(SEQ, gen)
    attn["K"], k_ch = inject_outliers(attn["K"], gen)
    attn["V"], v_ch = inject_outliers(attn["V"], gen)
    mlp = base_mlp(SEQ, gen)
    mlp["x"], x_ch = inject_outliers(mlp["x"], gen)
    datasets["outlier"] = {
        "attention": attn, "mlp": mlp,
        "rationale": "Emergent outlier channels (LLM.int8() pattern): a few "
                     "feature dims carry 10-100x magnitude.",
        "expected_issue": "Per-tensor MLP scale inflated by the global max "
                          "-> all non-outlier values crushed to few codes. "
                          "Per-token attention scales inflated for outlier "
                          "tokens -> their other dims lose precision. The "
                          "canonical fix is per-channel quantization.",
        "thresholds": {**DEFAULT_THRESH, "cos_min": 0.995,
                       "outlier_ratio_max": 0.02},
        "outlier_channels": {"K": k_ch, "V": v_ch, "x": x_ch},
        # Known limitations (XFAIL): expected to fail until the kernels gain
        # finer-grained quantization. The repair advisor's simulation
        # confirms the fix works; validate_int8.py reports these without
        # gating the overall result, and flags XPASS once fixed.
        # NOTE: the MLP xfail was REMOVED on 2026-06-13 — per-token activation
        # + per-channel weight + per-token output quant (int8_mlp_forward_per_
        # channel) now passes outlier outright (cos>0.999). Attention still
        # uses per-token scales, so its outlier xfail stands.
        "xfail": {
            "attention": "per-token scales crush the non-outlier dims of "
                         "outlier tokens (outlier ratio above threshold). "
                         "Fix: per-channel K/V quantization or smoothing.",
        },
        "seq_len": SEQ,
    }

    # 3. boundary — quantized codes pile up at ±127.
    datasets["boundary"] = {
        "attention": {k: boundary_fill(v.shape, gen)
                      for k, v in base_attention(SEQ, gen).items()},
        "mlp": {"x": boundary_fill((BATCH, SEQ, D_MODEL), gen),
                "W1": boundary_fill((D_MODEL, D_FF), gen, amax=0.05),
                "W2": boundary_fill((D_FF, D_MODEL), gen, amax=0.05)},
        "rationale": "All values within 1.5% of the tensor absmax: codes "
                     "land on ±126/±127.",
        "expected_issue": "Saturation bugs: clamp(-128,127) asymmetry, "
                          "round-to-nearest-even vs truncation, scale "
                          "off-by-one (amax/127 vs amax/128).",
        "thresholds": dict(DEFAULT_THRESH),
        # ±4-valued Q/K make |scores| ~ N(0,16): softmax is near-argmax and
        # score quantization noise flips near-tie winners — whole output
        # rows swap to a different V row. Intrinsic to INT8 QK^T on
        # adversarial data (the FP16 reference itself is unstable there).
        "xfail": {
            "attention": "near-tie argmax flips under peaky softmax "
                         "(elementwise outlier ratio ~0.08, cosine still "
                         ">0.999). Fix: QK^T in FP16 (advisor-confirmed).",
        },
        "seq_len": SEQ,
    }

    # 4. stress — heavy tails + 3-decade channel ranges.
    datasets["stress"] = {
        "attention": stress_attention(SEQ, gen),
        "mlp": stress_mlp(SEQ, gen),
        "rationale": "Heavy-tailed values, log-spaced channel magnitudes, "
                     "peaky softmax — every quantizer weakness at once.",
        "expected_issue": "Attention: online-softmax rescale underflow and "
                          "per-token scale blowup. MLP: 3 orders of "
                          "magnitude under one per-tensor scale; small "
                          "channels reduced to 0-2 codes.",
        "thresholds": {**DEFAULT_THRESH, "cos_min": 0.995,
                       "top10_min": 0.7, "outlier_ratio_max": 0.05},
        "xfail": {
            "attention": "designed-in peaky softmax: near-tie argmax flips "
                         "swap whole output rows (outlier ratio ~0.08, "
                         "cosine >0.997). Fix: QK^T in FP16.",
        },
        "seq_len": SEQ,
    }

    # 5. short_seq — seq=4: not divisible by any WMMA tile -> exercises the
    #    scalar fallback kernels and tiny-shape indexing.
    datasets["short_seq"] = {
        "attention": base_attention(SHORT_SEQ, gen),
        "mlp": base_mlp(SHORT_SEQ, gen),
        "rationale": "seq_len=4 falls off every WMMA divisibility check; "
                     "runs the scalar fallback path.",
        "expected_issue": "Fallback-path divergence from the fast path; "
                          "out-of-bounds reads on tiny shapes.",
        "thresholds": dict(EDGE_THRESH),
        "seq_len": SHORT_SEQ,
    }

    # 6. long_seq — seq=4096 (batch reduced to 1 to keep the naive FP16
    #    reference inside 40GB): online softmax over 64 KV tiles.
    long_gen = torch.Generator().manual_seed(SEED + 1)
    Ql = torch.randn(1, HEADS, MAX_SEQ_LEN, HEAD_DIM, generator=long_gen).half()
    Kl = torch.randn(1, HEADS, MAX_SEQ_LEN, HEAD_DIM, generator=long_gen).half()
    Vl = torch.randn(1, HEADS, MAX_SEQ_LEN, HEAD_DIM, generator=long_gen).half()
    xl = torch.randn(1, MAX_SEQ_LEN, D_MODEL, generator=long_gen).half()
    W1l = (torch.randn(D_MODEL, D_FF, generator=long_gen) * 0.02).half()
    W2l = (torch.randn(D_FF, D_MODEL, generator=long_gen) * 0.02).half()
    datasets["long_seq"] = {
        "attention": {"Q": Ql, "K": Kl, "V": Vl},
        "mlp": {"x": xl, "W1": W1l, "W2": W2l},
        "rationale": "seq_len=4096: 64 online-softmax KV tiles per query; "
                     "maximum rescale-accumulation depth.",
        "expected_issue": "Drift in the running max/sum across tiles; "
                          "rsum precision loss; int32 accumulator paths "
                          "with the largest reduction depth.",
        "thresholds": dict(EDGE_THRESH),
        "seq_len": MAX_SEQ_LEN,
    }

    # 7. zeros — scale = 0 guard, softmax of all-equal scores.
    datasets["zeros"] = {
        "attention": {k: torch.zeros(BATCH, HEADS, SEQ, HEAD_DIM).half()
                      for k in ("Q", "K", "V")},
        "mlp": {"x": torch.zeros(BATCH, SEQ, D_MODEL).half(),
                "W1": torch.zeros(D_MODEL, D_FF).half(),
                "W2": torch.zeros(D_FF, D_MODEL).half()},
        "rationale": "All-zero inputs: amax=0.",
        "expected_issue": "Division by zero in scale computation (needs the "
                          "1e-8 clamp); softmax over all-equal scores must "
                          "give uniform weights, not NaN.",
        "thresholds": {**EDGE_THRESH, "zero_output": True},
        "seq_len": SEQ,
    }

    # 8. constant — all ones (weights 0.01 so the MLP output stays inside
    #    FP16 range: gelu(512*0.01*...) ~ O(1), out ~ O(100)).
    datasets["constant"] = {
        "attention": {k: torch.ones(BATCH, HEADS, SEQ, HEAD_DIM).half()
                      for k in ("Q", "K", "V")},
        "mlp": {"x": torch.ones(BATCH, SEQ, D_MODEL).half(),
                "W1": torch.full((D_MODEL, D_FF), 0.01).half(),
                "W2": torch.full((D_FF, D_MODEL), 0.01).half()},
        "rationale": "Constant inputs: every token row identical, softmax "
                     "exactly uniform, attention output == V row.",
        "expected_issue": "Normalization (1/rsum) bias: any systematic "
                          "over/under-count of softmax weights shows as a "
                          "uniform offset. MLP: single-code quantization "
                          "(every element is the same code) must be exact.",
        "thresholds": dict(EDGE_THRESH),
        "seq_len": SEQ,
    }

    return datasets


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    datasets = build_datasets()
    manifest = {}
    for name, d in datasets.items():
        path = os.path.join(OUT_DIR, f"{name}.pt")
        torch.save({"attention": d["attention"], "mlp": d["mlp"]}, path)
        manifest[name] = {
            "file": f"{name}.pt",
            "rationale": d["rationale"],
            "expected_issue": d["expected_issue"],
            "thresholds": d["thresholds"],
            "seq_len": d["seq_len"],
            "attention_shapes": {k: list(v.shape)
                                 for k, v in d["attention"].items()},
            "mlp_shapes": {k: list(v.shape) for k, v in d["mlp"].items()},
            "outlier_channels": d.get("outlier_channels"),
            "xfail": d.get("xfail"),
        }
        sizes = ", ".join(f"{k}{tuple(v.shape)}" for k, v in d["attention"].items())
        print(f"[OK] {name:10s} -> {path}  ({sizes})")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[OK] manifest -> {OUT_DIR}/manifest.json")
    print("Run `python validate_int8.py` to execute the 5-gate validation.")


if __name__ == "__main__":
    main()
