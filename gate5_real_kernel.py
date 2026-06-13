#!/usr/bin/env python3
"""Real GPT-2 perplexity on WikiText-2 with the ACTUAL INT8 MLP CUDA kernel.

This closes the sim-vs-real gap in validate_int8.py's Gate 5. Gate 5 measures
perplexity with `_kernel_faithful_mlp` — a PyTorch *simulation* of the kernel's
quantization. This script instead patches every GPT-2 block's MLP to call the
real kernel `ext.int8_mlp_forward_per_channel_bias`, so the reported perplexity
is produced by the exact CUDA code that ships, not a model of it.

Wiring (matches the kernel + GPT-2 layout):
  - x  : per-token (per-row) INT8 activation quant
  - W1 : GPT-2 c_fc.weight  [d_model, d_ff], per-output-channel INT8 quant
  - b1 : GPT-2 c_fc.bias    [d_ff], added IN-KERNEL before the fused GELU
  - W2 : GPT-2 c_proj.weight [d_ff, d_model], per-output-channel INT8 quant
  - out: per-token INT8 output quant; dequant in FP, then + b2 (c_proj.bias) FP16
  - T (= tokens) padded up to a multiple of 128 so the WMMA path is taken
    (GPT-2's d_model=768 / d_ff=3072 already satisfy the other tile multiples).

LayerNorm, attention, and residuals stay FP16 (only the MLP is INT8) — the same
scope as Gate 5, so the two numbers are directly comparable.

Usage:
    python gate5_real_kernel.py            # gpt2, n_ctx=3072
    python gate5_real_kernel.py --model gpt2-medium --n_ctx 2048
"""
import argparse
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

REAL_CORPUS = "testdata/real_corpus.txt"
BLOCK_M = 128  # kernel tile: T must be a multiple of this for the WMMA path


def build_ext():
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
    )


def q_per_token(t):
    """Per-row (per-token) symmetric INT8 quant of [N, K]. Returns (int8, scale[N])."""
    s = t.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = (t / s).round().clamp(-127, 127).to(torch.int8)
    return q, s.squeeze(-1)


def q_per_channel(w):
    """Per-output-column symmetric INT8 quant of [K, N]. Returns (int8, scale[N])."""
    s = w.abs().amax(0, keepdim=True).clamp(min=1e-8) / 127.0
    q = (w / s).round().clamp(-127, 127).to(torch.int8)
    return q, s.squeeze(0)


def make_kernel_mlp(ext, w1, b1, w2, b2):
    """Return a forward(hidden)->out that runs the real INT8 MLP kernel.

    Weights are quantized ONCE here (static-weight inference); only the
    activation is quantized per call. b1 goes in-kernel (pre-GELU), b2 in FP16.
    """
    d_model, d_ff = w1.shape
    w1i, s1 = q_per_channel(w1)            # [d_model, d_ff] -> col scale [d_ff]
    w2i, s2 = q_per_channel(w2)            # [d_ff, d_model] -> col scale [d_model]
    w1i = w1i.contiguous()
    w2i = w2i.contiguous()
    s1 = s1.float().contiguous()
    s2 = s2.float().contiguous()
    b1 = b1.float().contiguous()
    b2f = b2.float()

    def fwd(hidden):
        shp = hidden.shape
        x = hidden.reshape(-1, shp[-1]).float()       # [N, d_model]
        n = x.shape[0]
        npad = (n + BLOCK_M - 1) // BLOCK_M * BLOCK_M  # round up to tile
        if npad != n:
            x = F.pad(x, (0, 0, 0, npad - n))         # zero-pad the token rows
        xi, sx = q_per_token(x)                        # [npad, d_model]
        out_i8, out_s = ext.int8_mlp_forward_per_channel_bias(
            xi.reshape(1, npad, d_model).contiguous(),
            w1i, w2i,
            sx.float().contiguous(), s1, s2, b1)
        out = out_i8.float().reshape(npad, d_model) * out_s.reshape(npad, 1)
        out = out[:n] + b2f                            # b2 in FP, drop pad rows
        return out.reshape(shp).to(hidden.dtype)

    return fwd


def perplexity(model, ids, stride):
    nll, ntok = 0.0, 0
    for i in range(0, ids.size(1) - 1, stride):
        chunk = ids[:, i:i + stride + 1]
        if chunk.size(1) < 2:
            break
        with torch.no_grad():
            logits = model(chunk[:, :-1]).logits
        tgt = chunk[:, 1:]
        ll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                             tgt.reshape(-1), reduction="sum")
        nll += float(ll)
        ntok += tgt.numel()
    return math.exp(nll / ntok), ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n_ctx", type=int, default=3072)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--chars", type=int, default=120000)
    args = ap.parse_args()

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    if not os.path.exists(REAL_CORPUS):
        raise SystemExit(
            f"corpus missing: {REAL_CORPUS} — run `python prepare_real_corpus.py`")

    ext = build_ext()
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    text = open(REAL_CORPUS).read()[:args.chars]
    ids = tok(text, return_tensors="pt").input_ids[:, :args.n_ctx].cuda()

    model = GPT2LMHeadModel.from_pretrained(args.model).eval().cuda()
    ppl_fp16, ntok = perplexity(model, ids, args.stride)

    for blk in model.transformer.h:
        m = blk.mlp
        blk.mlp.forward = make_kernel_mlp(
            ext,
            m.c_fc.weight.data.float().cuda(),     # [d_model, d_ff]
            m.c_fc.bias.data.float().cuda(),       # [d_ff]
            m.c_proj.weight.data.float().cuda(),   # [d_ff, d_model]
            m.c_proj.bias.data.float().cuda())     # [d_model]
    ppl_i8, _ = perplexity(model, ids, args.stride)

    ppl_inc = ppl_i8 / ppl_fp16 - 1.0
    print(f"\nmodel={args.model}  n_ctx={args.n_ctx}  tokens={ntok}")
    print(f"  FP16 perplexity      : {ppl_fp16:.4f}")
    print(f"  INT8-kernel perplexity: {ppl_i8:.4f}")
    print(f"  perplexity increase  : {ppl_inc*100:+.3f}%   "
          f"({'PASS' if ppl_inc < 0.02 else 'FAIL'} @ <2% bar)")


if __name__ == "__main__":
    main()
