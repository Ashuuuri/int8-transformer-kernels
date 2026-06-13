"""Generate test data (inputs + reference outputs) for CUDA kernel validation.

Usage:
    python tests/gen_testdata.py

Produces testdata/small/ and testdata/large/ with .bin files that pure CUDA
tests can read directly, without needing Python or PyTorch at test time.
"""

import os
import sys
import numpy as np
import torch

# Add common/ (shared primitives) to path so we can import baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from baseline import attention_baseline, mlp_baseline, check_cuda


# ── Configurations ──────────────────────────────────────────────────────────

CONFIGS = {
    "small": {  # Debug: small enough to inspect by hand
        "batch": 1, "heads": 1, "seq_len": 4, "head_dim": 4,
        "d_model": 4, "d_ff": 8,
    },
    "large": {  # Realistic: matches typical transformer dimensions
        "batch": 2, "heads": 8, "seq_len": 512, "head_dim": 64,
        "d_model": 512, "d_ff": 2048,
    },
}

SEED = 42


def save_tensor(tensor: torch.Tensor, path: str):
    """Save a tensor as raw binary (contiguous, on CPU)."""
    t = tensor.contiguous().cpu()
    np.array(t.numpy()).tofile(path)


def generate(name: str, cfg: dict):
    outdir = os.path.join("testdata", name)
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(SEED)
    device = "cuda"
    dtype = torch.float16

    B, H, S, D = cfg["batch"], cfg["heads"], cfg["seq_len"], cfg["head_dim"]
    d_model, d_ff = cfg["d_model"], cfg["d_ff"]

    # ── Attention test data ─────────────────────────────────────────────
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    attn_ref = attention_baseline(Q, K, V)

    save_tensor(Q, os.path.join(outdir, "attn_Q.bin"))
    save_tensor(K, os.path.join(outdir, "attn_K.bin"))
    save_tensor(V, os.path.join(outdir, "attn_V.bin"))
    save_tensor(attn_ref, os.path.join(outdir, "attn_ref.bin"))

    # ── MLP test data ──────────────────────────────────────────────────
    x = torch.randn(B, S, d_model, device=device, dtype=dtype)
    W1 = torch.randn(d_model, d_ff, device=device, dtype=dtype) * 0.02
    W2 = torch.randn(d_ff, d_model, device=device, dtype=dtype) * 0.02
    mlp_ref = mlp_baseline(x, W1, W2)

    save_tensor(x, os.path.join(outdir, "mlp_x.bin"))
    save_tensor(W1, os.path.join(outdir, "mlp_W1.bin"))
    save_tensor(W2, os.path.join(outdir, "mlp_W2.bin"))
    save_tensor(mlp_ref, os.path.join(outdir, "mlp_ref.bin"))

    # ── INT8 quantized test data ───────────────────────────────────────
    # Quantize FP16 inputs to INT8 with per-tensor scale
    def quantize_to_int8(t):
        t_f = t.float()
        scale = t_f.abs().max() / 127.0
        t_int8 = (t_f / scale).round().clamp(-128, 127).to(torch.int8)
        return t_int8, scale

    Q_int8, scale_Q = quantize_to_int8(Q)
    K_int8, scale_K = quantize_to_int8(K)
    V_int8, scale_V = quantize_to_int8(V)
    x_int8, scale_x = quantize_to_int8(x)
    W1_int8, scale_W1 = quantize_to_int8(W1)
    W2_int8, scale_W2 = quantize_to_int8(W2)

    # INT8 reference: dequantize → run FP16 baseline → quantize back
    Q_deq = Q_int8.float() * scale_Q
    K_deq = K_int8.float() * scale_K
    V_deq = V_int8.float() * scale_V
    int8_attn_ref = attention_baseline(Q_deq.half(), K_deq.half(), V_deq.half())

    x_deq = x_int8.float() * scale_x
    W1_deq = W1_int8.float() * scale_W1
    W2_deq = W2_int8.float() * scale_W2
    int8_mlp_ref = mlp_baseline(x_deq.half(), W1_deq.half(), W2_deq.half())

    save_tensor(Q_int8, os.path.join(outdir, "int8_attn_Q.bin"))
    save_tensor(K_int8, os.path.join(outdir, "int8_attn_K.bin"))
    save_tensor(V_int8, os.path.join(outdir, "int8_attn_V.bin"))
    save_tensor(int8_attn_ref, os.path.join(outdir, "int8_attn_ref.bin"))
    # Save scales as float32
    torch.tensor([scale_Q.item()]).numpy().tofile(os.path.join(outdir, "int8_attn_scale_Q.bin"))
    torch.tensor([scale_K.item()]).numpy().tofile(os.path.join(outdir, "int8_attn_scale_K.bin"))
    torch.tensor([scale_V.item()]).numpy().tofile(os.path.join(outdir, "int8_attn_scale_V.bin"))

    save_tensor(x_int8, os.path.join(outdir, "int8_mlp_x.bin"))
    save_tensor(W1_int8, os.path.join(outdir, "int8_mlp_W1.bin"))
    save_tensor(W2_int8, os.path.join(outdir, "int8_mlp_W2.bin"))
    save_tensor(int8_mlp_ref, os.path.join(outdir, "int8_mlp_ref.bin"))
    torch.tensor([scale_x.item()]).numpy().tofile(os.path.join(outdir, "int8_mlp_scale_x.bin"))
    torch.tensor([scale_W1.item()]).numpy().tofile(os.path.join(outdir, "int8_mlp_scale_W1.bin"))
    torch.tensor([scale_W2.item()]).numpy().tofile(os.path.join(outdir, "int8_mlp_scale_W2.bin"))

    # ── Save metadata ──────────────────────────────────────────────────
    with open(os.path.join(outdir, "config.txt"), "w") as f:
        f.write(f"seed={SEED}\n")
        for k, v in cfg.items():
            f.write(f"{k}={v}\n")

    print(f"[OK] Generated {outdir}/")


if __name__ == "__main__":
    check_cuda()
    for name, cfg in CONFIGS.items():
        generate(name, cfg)
    print("\nDone. Test data saved to testdata/small/ and testdata/large/")
