"""FP16 PyTorch baselines for Attention and MLP."""

import torch
import torch.nn.functional as F


def attention_baseline(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention in FP16.

    Args:
        Q: Query tensor of shape (batch, heads, seq_len, head_dim), dtype float16.
        K: Key tensor of shape (batch, heads, seq_len, head_dim), dtype float16.
        V: Value tensor of shape (batch, heads, seq_len, head_dim), dtype float16.

    Returns:
        Output tensor of same shape as Q.
    """
    head_dim = Q.shape[-1]
    scale = head_dim ** -0.5
    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, V)


def mlp_baseline(x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    """Two-layer MLP: Linear -> GELU -> Linear, in FP16.

    Args:
        x: Input tensor of shape (batch, seq_len, d_model), dtype float16.
        W1: First weight matrix of shape (d_model, d_ff), dtype float16.
        W2: Second weight matrix of shape (d_ff, d_model), dtype float16.

    Returns:
        Output tensor of shape (batch, seq_len, d_model).
    """
    hidden = F.gelu(x @ W1, approximate="tanh")
    return hidden @ W2


def check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Did you forget: module load pytorch?")
    print(f"[CUDA] {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")


if __name__ == "__main__":
    check_cuda()
    device = "cuda"
    dtype = torch.float16
    batch, heads, seq_len, head_dim = 2, 8, 512, 64
    d_model, d_ff = 512, 2048

    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    out_attn = attention_baseline(Q, K, V)
    print(f"Attention output shape: {out_attn.shape}")

    x = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    W1 = torch.randn(d_model, d_ff, device=device, dtype=dtype) * 0.02
    W2 = torch.randn(d_ff, d_model, device=device, dtype=dtype) * 0.02
    out_mlp = mlp_baseline(x, W1, W2)
    print(f"MLP output shape: {out_mlp.shape}")
