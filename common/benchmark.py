"""GPU timing harness using torch.cuda.Event.

Usage:
    from benchmark import benchmark
    latency_ms = benchmark(fn, *args)
"""

import torch
from typing import Callable, Any


def benchmark(fn: Callable[..., Any], *args, n_iter: int = 50, n_warmup: int = 10) -> float:
    """Time a GPU function, returning mean latency in milliseconds.

    Runs n_warmup + n_iter iterations. The first n_warmup are discarded.
    Uses torch.cuda.Event for accurate GPU-side timing.

    Args:
        fn: Callable to benchmark (should operate on CUDA tensors).
        *args: Arguments forwarded to fn.
        n_iter: Number of timed iterations (default 50).
        n_warmup: Number of warmup iterations to discard (default 10).

    Returns:
        Mean latency in milliseconds over the timed iterations.
    """
    # Warmup
    for _ in range(n_warmup):
        fn(*args)

    torch.cuda.synchronize()

    timings = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    return sum(timings) / len(timings)


if __name__ == "__main__":
    from baseline import attention_baseline, mlp_baseline, check_cuda
    check_cuda()

    device = "cuda"
    dtype = torch.float16
    batch, heads, seq_len, head_dim = 2, 8, 512, 64
    d_model, d_ff = 512, 2048

    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    attn_ms = benchmark(attention_baseline, Q, K, V)
    print(f"Attention baseline: {attn_ms:.3f} ms")

    x = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    W1 = torch.randn(d_model, d_ff, device=device, dtype=dtype) * 0.02
    W2 = torch.randn(d_ff, d_model, device=device, dtype=dtype) * 0.02

    mlp_ms = benchmark(mlp_baseline, x, W1, W2)
    print(f"MLP baseline: {mlp_ms:.3f} ms")
