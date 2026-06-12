"""Correctness checker: compare a CUDA kernel output against a reference tensor."""

import torch


def check_correctness(
    reference: torch.Tensor,
    actual: torch.Tensor,
    label: str = "test",
    mode: str = "fp16",
) -> bool:
    """Compare two tensors and report whether they match within tolerance.

    Args:
        reference: Ground-truth tensor (e.g., from PyTorch baseline).
        actual: Tensor produced by the CUDA kernel under test.
        label: Descriptive name printed in the report.
        mode: "fp16" (atol=1e-2) or "int8" (atol=0.1).

    Returns:
        True if tensors match within the specified tolerance.
    """
    atol = 1e-2 if mode == "fp16" else 0.1
    diff = (reference.float() - actual.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    passed = torch.allclose(reference.float(), actual.float(), atol=atol, rtol=0)

    status = "PASSED" if passed else "FAILED"
    print(f"[{status}] {label}  |  max abs error = {max_err:.6f}  |  mean abs error = {mean_err:.6f}  |  atol = {atol}")

    if not passed:
        # Show diagnostics to help debug
        flat_diff = diff.flatten()
        worst_idx = flat_diff.argmax().item()
        print(f"  Worst error at flat index {worst_idx}")
        print(f"  Reference: {reference.flatten()[worst_idx].item():.6f}")
        print(f"  Actual:    {actual.flatten()[worst_idx].item():.6f}")

        # Show first few elements for comparison
        ref_flat = reference.flatten()[:8].tolist()
        act_flat = actual.flatten()[:8].tolist()
        print(f"  First 8 ref: {[f'{v:.4f}' for v in ref_flat]}")
        print(f"  First 8 act: {[f'{v:.4f}' for v in act_flat]}")

        # Error distribution
        pct_over = (flat_diff > atol).float().mean().item() * 100
        print(f"  Elements exceeding atol: {pct_over:.1f}%")

    return passed


if __name__ == "__main__":
    from baseline import check_cuda
    check_cuda()

    device = "cuda"
    dtype = torch.float16

    # Self-check: baseline vs. itself should always pass.
    from baseline import attention_baseline, mlp_baseline

    batch, heads, seq_len, head_dim = 2, 8, 512, 64
    d_model, d_ff = 512, 2048

    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    ref = attention_baseline(Q, K, V)
    check_correctness(ref, ref, label="attention self-check", mode="fp16")

    x = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    W1 = torch.randn(d_model, d_ff, device=device, dtype=dtype) * 0.02
    W2 = torch.randn(d_ff, d_model, device=device, dtype=dtype) * 0.02
    ref = mlp_baseline(x, W1, W2)
    check_correctness(ref, ref, label="mlp self-check", mode="fp16")
