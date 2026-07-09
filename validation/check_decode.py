"""check_decode.py — decode-kernel correctness harness (edge shapes).

The 5-gate validation (`validate_int8.py`) exercises the prefill attention and
MLP kernels; the decode kernel — the repo's strongest result — was previously
only spot-checked by the benchmark scripts. This harness closes that gap: it
runs `int8_decode_attention_forward` against an fp32 reference (on the exact
dequantized values the INT8 path represents) across the shapes that stress the
kernel's edges:

  - S = 1 / 31 / 33 / 127:  batch-group tails (S % NB != 0), sub-tile chunks,
                            the nsplit=1 / empty-chunk paths
  - S = 128 / 1000:         chunk-boundary and non-power-of-two coverage
  - S = 4096 / 8192:        many-split combine (log-sum-exp merge) depth
  - small B*H:              the grid-starved dispatch corner
  - both head dims (64 / 128), i.e. both batched-kernel instantiations

PASS bar per case: cosine > 0.9999, max abs err < 0.05 (fp16 out, per-token
int8 in), no NaN. Run from the repo root after every decode-kernel change,
alongside `python validation/validate_int8.py`:

    python validation/check_decode.py
"""
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_ext():
    kdir = os.path.join(ROOT, "kernels")
    return load(name="int8_ext",
        sources=[os.path.join(kdir, f) for f in (
            "int8_attention.cu", "int8_decode_attention.cu", "int8_mlp.cu",
            "quant_utils.cu", "int8_ext.cu")],
        extra_cuda_cflags=["-arch=sm_80", "--std=c++17", "-O3"], verbose=False)


def _qpt(t):
    """Per-token symmetric INT8 quant over the last dim (matches the kernels)."""
    am = t.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    sc = am / 127.0
    return (t / sc).round().clamp(-128, 127).to(torch.int8), \
           sc.squeeze(-1).contiguous().float()


CASES = [(1, 1, 64, 1), (1, 1, 64, 31), (2, 2, 64, 33), (1, 2, 64, 127),
         (2, 2, 64, 128), (3, 5, 64, 1000), (8, 8, 64, 4096), (64, 16, 64, 8192),
         (1, 1, 128, 1), (1, 1, 128, 31), (2, 2, 128, 33), (1, 2, 128, 127),
         (2, 2, 128, 128), (3, 5, 128, 1000), (8, 8, 128, 4096), (64, 16, 128, 8192)]


def main():
    ext = _load_ext()
    fails = 0
    for B, H, D, S in CASES:
        torch.manual_seed(1234 + S)
        Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        Qi, sQ = _qpt(Q); Ki, sK = _qpt(K); Vi, sV = _qpt(V)
        Qd = Qi.float() * sQ.unsqueeze(-1)
        Kd = Ki.float() * sK.unsqueeze(-1)
        Vd = Vi.float() * sV.unsqueeze(-1)
        ref = F.scaled_dot_product_attention(Qd.unsqueeze(2), Kd, Vd).squeeze(2)
        out = ext.int8_decode_attention_forward(Qi, Ki, Vi, sQ, sK, sV).float()
        cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
        maxe = (out - ref).abs().max().item()
        nan_ok = not torch.isnan(out).any().item()
        ok = cos > 0.9999 and maxe < 0.05 and nan_ok
        fails += (not ok)
        print(f"B{B:>3} H{H:>3} D{D:>4} S{S:>6}  cos={cos:.6f}  maxerr={maxe:.4f}  "
              f"nan_ok={nan_ok}  {'PASS' if ok else 'FAIL'}", flush=True)

    print("DECODE CORRECTNESS:", "ALL PASS" if fails == 0 else f"{fails} FAIL")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
