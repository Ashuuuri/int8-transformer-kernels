"""Profiling harness whose shapes MATCH sweep.py's graded grid.

Rationale (do not "simplify" back to test_int8.py --quick):
  ncu must profile the SAME workload the perf gate grades, or every diagnosis
  comes from the wrong regime. test_int8.py --quick uses batch=2, seq=512 and
  head_dim 64/128/256 — a GRID-STARVED toy (128 blocks / 108 SMs ~ 1.18
  blocks/SM, warps_active ~7.4%). sweep.py grades batch=8, head_dim=64,
  seq 512-4096 — 512-4096 blocks, occupancy-MAXED (3 blocks/SM, warps_active
  ~18%, co-limited by registers AND shared memory). Profiling the toy made the
  loop chase an occupancy/register lever that does not exist in the graded
  config. This script runs ONLY the graded shapes so ncu and the perf gate
  agree.

Usage:
  python3 profile_kernel.py            # attention (default, sweep mid shape)
  python3 profile_kernel.py mlp        # MLP
  python3 profile_kernel.py both       # both, back to back
  python3 profile_kernel.py attn 8 8 4096 64   # override B H S D for attn

Run under ncu, e.g.:
  sudo env "PATH=$PATH" HOME="$HOME" ncu --kernel-name regex:int8_wmma \
      --launch-count 1 --metrics <metrics> python3 profile_kernel.py attn
"""
import sys
import torch
from torch.utils.cpp_extension import load

# sweep.py grid: BATCH=8, HEADS=8, HEAD_DIM=64, SEQ in {512,1024,2048,4096};
# MLP: BATCH=8, SEQ=512, d_model sweep. Defaults below pick a representative
# mid-size shape so one ncu pass reflects the graded regime.
ATTN_DEFAULT = (8, 8, 2048, 64)      # 2048 blocks, the sweep's middle seq
MLP_DEFAULT  = (8, 512, 1024, 4096)  # B, S, d_model, d_ff

WARMUP = 5


def _load():
    return load(
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


def _q_per_token(t):
    am = t.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    sc = am / 127.0
    i8 = (t / sc).round().clamp(-128, 127).to(torch.int8)
    return i8, sc.squeeze(-1).contiguous().float()


def _q_per_tensor(t):
    s = t.float().abs().max() / 127.0
    return (t.float() / s).round().clamp(-128, 127).to(torch.int8), float(s)


def profile_attn(ext, b, h, s, d):
    torch.manual_seed(0)
    Q = torch.randn(b, h, s, d, device="cuda", dtype=torch.float16)
    K = torch.randn(b, h, s, d, device="cuda", dtype=torch.float16)
    V = torch.randn(b, h, s, d, device="cuda", dtype=torch.float16)
    Qi, sQ = _q_per_token(Q)
    Ki, sK = _q_per_token(K)
    Vi, sV = _q_per_token(V)
    for _ in range(WARMUP):
        ext.int8_attention_forward(Qi, Ki, Vi, sQ, sK, sV)
    torch.cuda.synchronize()
    grid_x = (s + 63) // 64
    print(f"[profile_kernel] attn b={b} h={h} s={s} d={d}  "
          f"grid = ({grid_x}, {b*h}) = {grid_x*b*h} blocks")


def profile_mlp(ext, b, s, dm, dff):
    torch.manual_seed(0)
    x = torch.randn(b, s, dm, device="cuda", dtype=torch.float16)
    W1 = torch.randn(dm, dff, device="cuda", dtype=torch.float16) * 0.02
    W2 = torch.randn(dff, dm, device="cuda", dtype=torch.float16) * 0.02
    xi, sx = _q_per_tensor(x)
    w1, s1 = _q_per_tensor(W1)
    w2, s2 = _q_per_tensor(W2)
    for _ in range(WARMUP):
        ext.int8_mlp_forward(xi, w1, w2, sx, s1, s2)
    torch.cuda.synchronize()
    print(f"[profile_kernel] mlp b={b} s={s} d_model={dm} d_ff={dff}  "
          f"tokens = {b*s}")


def main():
    argv = sys.argv[1:]
    which = argv[0] if argv and argv[0] in ("attn", "mlp", "both") else "attn"
    rest = [a for a in argv if a not in ("attn", "mlp", "both")]
    ext = _load()
    if which in ("attn", "both"):
        shape = tuple(int(x) for x in rest[:4]) if len(rest) >= 4 else ATTN_DEFAULT
        profile_attn(ext, *shape)
    if which in ("mlp", "both"):
        shape = tuple(int(x) for x in rest[:4]) if len(rest) >= 4 else MLP_DEFAULT
        profile_mlp(ext, *shape)


if __name__ == "__main__":
    main()
