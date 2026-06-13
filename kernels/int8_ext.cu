// int8_ext.cu — PyTorch / pybind11 binding for INT8 attention and MLP kernels.
//
// Usage in Python:
//
//   from torch.utils.cpp_extension import load
//   int8_ext = load(
//       name="int8_ext",
//       sources=["kernels/int8_attention.cu", "kernels/int8_mlp.cu",
//                "kernels/quant_utils.cu",    "kernels/int8_ext.cu"],
//       extra_cuda_cflags=["-arch=sm_80", "--std=c++17", "-O3"],
//       verbose=False,
//   )
//
//   # Per-token quantization: scales are 1-D tensors of shape [batch*heads*seq_len]
//   attn_out = int8_ext.int8_attention_forward(Q_int8, K_int8, V_int8,
//                                               scale_Q, scale_K, scale_V)
//   # attn_out is FP16 (no dequant needed)
//
//   mlp_out  = int8_ext.int8_mlp_forward(x_int8, W1_int8, W2_int8,
//                                         scale_x, scale_W1, scale_W2)

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

// ── Forward declarations (defined in the other .cu files) ──────────────
void int8_attention_forward(
    const int8_t* Q, const int8_t* K, const int8_t* V, half* out,
    const float* scale_Q, const float* scale_K, const float* scale_V,
    int batch, int heads, int seq_len, int head_dim
);

void int8_decode_attention_forward(
    const int8_t* Q, const int8_t* K, const int8_t* V, half* out,
    const float* scale_Q, const float* scale_K, const float* scale_V,
    int BH, int S, int head_dim
);

void int8_mlp_forward(
    const int8_t* x, const int8_t* W1, const int8_t* W2, int8_t* out,
    const float* scale_x, const float* scale_W1, const float* scale_W2,
    int batch, int seq_len, int d_model, int d_ff
);

void int8_mlp_forward_per_channel(
    const int8_t* x, const int8_t* W1, const int8_t* W2, int8_t* out,
    const float* scale_x, const float* scale_W1, const float* scale_W2,
    float* out_row_scales,
    int batch, int seq_len, int d_model, int d_ff
);

void transpose_int8_weights(
    const int8_t* W1, const int8_t* W2, int8_t* W1T, int8_t* W2T,
    int d_model, int d_ff
);

void int8_mlp_forward_prepacked(
    const int8_t* x, const int8_t* W1T, const int8_t* W2T, int8_t* out,
    const float* scale_x, const float* scale_W1, const float* scale_W2,
    int batch, int seq_len, int d_model, int d_ff
);

void int8_mlp_get_output_scale(float* host_out);

// ── Helper: pack Python float scalars into a 1-D CUDA float tensor ─────
static torch::Tensor make_scale_tensor(double val, const torch::Device& dev) {
    return torch::full({1}, (float)val,
                       torch::TensorOptions().dtype(torch::kFloat32).device(dev));
}

// ── INT8 Attention binding (per-token quantization) ───────────────────
// Q, K, V : (batch, heads, seq_len, head_dim)  torch.int8  CUDA
// scale_Q/K/V : (batch*heads*seq_len,) or (batch, heads, seq_len) torch.float32 CUDA
//               — per-token symmetric scale
// returns  : (batch, heads, seq_len, head_dim)  torch.float16  CUDA
torch::Tensor int8_attention_forward_torch(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor scale_Q,
    torch::Tensor scale_K,
    torch::Tensor scale_V
) {
    TORCH_CHECK(Q.device().is_cuda(),           "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kInt8,      "Q must be int8");
    TORCH_CHECK(K.dtype() == torch::kInt8,      "K must be int8");
    TORCH_CHECK(V.dtype() == torch::kInt8,      "V must be int8");
    TORCH_CHECK(Q.is_contiguous(),              "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(),              "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(),              "V must be contiguous");
    TORCH_CHECK(Q.dim() == 4,                   "expected (batch, heads, seq_len, head_dim)");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q/K/V shape mismatch");

    TORCH_CHECK(scale_Q.dtype() == torch::kFloat32, "scale_Q must be float32");
    TORCH_CHECK(scale_K.dtype() == torch::kFloat32, "scale_K must be float32");
    TORCH_CHECK(scale_V.dtype() == torch::kFloat32, "scale_V must be float32");
    TORCH_CHECK(scale_Q.device().is_cuda(),         "scale_Q must be CUDA");

    const int batch    = Q.size(0);
    const int heads    = Q.size(1);
    const int seq_len  = Q.size(2);
    const int head_dim = Q.size(3);

    const int n_tokens = batch * heads * seq_len;
    // Flatten scales to 1-D contiguous
    auto sQ = scale_Q.contiguous().view({n_tokens});
    auto sK = scale_K.contiguous().view({n_tokens});
    auto sV = scale_V.contiguous().view({n_tokens});

    // Output is FP16
    auto out = torch::empty({batch, heads, seq_len, head_dim},
                            torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));

    int8_attention_forward(
        Q.data_ptr<int8_t>(), K.data_ptr<int8_t>(), V.data_ptr<int8_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        sQ.data_ptr<float>(), sK.data_ptr<float>(), sV.data_ptr<float>(),
        batch, heads, seq_len, head_dim
    );

    return out;
}

// ── INT8 decode attention binding (seq_q == 1, flash-decoding) ─────────
// Q       : (batch, heads, head_dim)        torch.int8  CUDA  (the single new token)
// K, V    : (batch, heads, seq_kv, head_dim) torch.int8  CUDA  (the KV cache)
// scale_Q : (batch*heads,)            scale_K/V : (batch*heads*seq_kv,)  float32 CUDA
// returns : (batch, heads, head_dim)  torch.float16  CUDA
torch::Tensor int8_decode_attention_forward_torch(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor scale_Q,
    torch::Tensor scale_K,
    torch::Tensor scale_V
) {
    TORCH_CHECK(Q.device().is_cuda(),      "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kInt8, "Q must be int8");
    TORCH_CHECK(K.dtype() == torch::kInt8 && V.dtype() == torch::kInt8,
                "K, V must be int8");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q/K/V must be contiguous");
    TORCH_CHECK(Q.dim() == 3, "expected Q (batch, heads, head_dim) — seq_q == 1");
    TORCH_CHECK(K.dim() == 4, "expected K (batch, heads, seq_kv, head_dim)");
    TORCH_CHECK(K.sizes() == V.sizes(), "K/V shape mismatch");

    const int batch    = Q.size(0);
    const int heads    = Q.size(1);
    const int head_dim = Q.size(2);
    const int seq_kv   = K.size(2);
    TORCH_CHECK(K.size(0) == batch && K.size(1) == heads && K.size(3) == head_dim,
                "Q/K head/batch/dim mismatch");
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "decode kernel v1 supports head_dim 64 or 128");

    const int BH = batch * heads;
    auto sQ = scale_Q.contiguous().view({BH});
    auto sK = scale_K.contiguous().view({BH * seq_kv});
    auto sV = scale_V.contiguous().view({BH * seq_kv});

    auto out = torch::empty({batch, heads, head_dim},
                            torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));

    int8_decode_attention_forward(
        Q.data_ptr<int8_t>(), K.data_ptr<int8_t>(), V.data_ptr<int8_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        sQ.data_ptr<float>(), sK.data_ptr<float>(), sV.data_ptr<float>(),
        BH, seq_kv, head_dim
    );
    return out;
}

// ── INT8 MLP binding (per-tensor, unchanged) ──────────────────────────
// x  : (batch, seq_len, d_model)  torch.int8  CUDA
// W1 : (d_model, d_ff)             torch.int8  CUDA
// W2 : (d_ff, d_model)              torch.int8  CUDA
// returns : tuple(
//     int8 output tensor  (batch, seq_len, d_model),
//     out_scale float     — dequant scale for correctness checking
// )
std::tuple<torch::Tensor, double> int8_mlp_forward_torch(
    torch::Tensor x,
    torch::Tensor W1,
    torch::Tensor W2,
    double scale_x,
    double scale_W1,
    double scale_W2
) {
    TORCH_CHECK(x.device().is_cuda(),           "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype()  == torch::kInt8,     "x must be int8");
    TORCH_CHECK(W1.dtype() == torch::kInt8,     "W1 must be int8");
    TORCH_CHECK(W2.dtype() == torch::kInt8,     "W2 must be int8");
    TORCH_CHECK(x.is_contiguous(),              "x must be contiguous");
    TORCH_CHECK(W1.is_contiguous(),             "W1 must be contiguous");
    TORCH_CHECK(W2.is_contiguous(),             "W2 must be contiguous");
    TORCH_CHECK(x.dim()  == 3, "x must be 3-D (batch, seq_len, d_model)");
    TORCH_CHECK(W1.dim() == 2, "W1 must be 2-D (d_model, d_ff)");
    TORCH_CHECK(W2.dim() == 2, "W2 must be 2-D (d_ff, d_model)");

    const int batch   = x.size(0);
    const int seq_len = x.size(1);
    const int d_model = x.size(2);
    const int d_ff    = W1.size(1);

    TORCH_CHECK(W1.size(0) == d_model, "W1 shape mismatch");
    TORCH_CHECK(W2.size(0) == d_ff,    "W2 shape mismatch");
    TORCH_CHECK(W2.size(1) == d_model, "W2 shape mismatch");

    auto out = torch::empty_like(x);

    auto d_sx  = make_scale_tensor(scale_x,  x.device());
    auto d_sw1 = make_scale_tensor(scale_W1, x.device());
    auto d_sw2 = make_scale_tensor(scale_W2, x.device());

    int8_mlp_forward(
        x.data_ptr<int8_t>(), W1.data_ptr<int8_t>(), W2.data_ptr<int8_t>(),
        out.data_ptr<int8_t>(),
        d_sx.data_ptr<float>(), d_sw1.data_ptr<float>(), d_sw2.data_ptr<float>(),
        batch, seq_len, d_model, d_ff
    );

    float out_scale;
    int8_mlp_get_output_scale(&out_scale);

    return {out, (double)out_scale};
}

// ── INT8 MLP binding (per-token activation + per-channel weight) ───────
// Overload taking the scales as tensors instead of scalars:
//   scale_x  : (batch*seq_len,)  per-token activation scale
//   scale_W1 : (d_ff,)           per-output-channel weight scale
//   scale_W2 : (d_model,)        per-output-channel weight scale
// This is the LLM.int8()-style outlier fix; the kernel applies scale_x per
// row and scale_W1/scale_W2 per output column in the GEMM epilogues. Length-1
// tensors are accepted too (degenerates to per-tensor). pybind11 selects this
// overload by argument type, so existing scalar callers are unaffected.
// The INT8 output is quantized PER-TOKEN, so this overload returns a per-row
// out_scale TENSOR [T] (not a scalar) — dequant as out_i8.float()*out_scale[t].
std::tuple<torch::Tensor, torch::Tensor> int8_mlp_forward_pc_torch(
    torch::Tensor x,
    torch::Tensor W1,
    torch::Tensor W2,
    torch::Tensor scale_x,
    torch::Tensor scale_W1,
    torch::Tensor scale_W2
) {
    TORCH_CHECK(x.device().is_cuda(),           "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype()  == torch::kInt8,     "x must be int8");
    TORCH_CHECK(W1.dtype() == torch::kInt8,     "W1 must be int8");
    TORCH_CHECK(W2.dtype() == torch::kInt8,     "W2 must be int8");
    TORCH_CHECK(x.is_contiguous(),              "x must be contiguous");
    TORCH_CHECK(W1.is_contiguous(),             "W1 must be contiguous");
    TORCH_CHECK(W2.is_contiguous(),             "W2 must be contiguous");
    TORCH_CHECK(x.dim()  == 3, "x must be 3-D (batch, seq_len, d_model)");
    TORCH_CHECK(W1.dim() == 2, "W1 must be 2-D (d_model, d_ff)");
    TORCH_CHECK(W2.dim() == 2, "W2 must be 2-D (d_ff, d_model)");

    const int batch   = x.size(0);
    const int seq_len = x.size(1);
    const int d_model = x.size(2);
    const int d_ff    = W1.size(1);
    const int T       = batch * seq_len;

    TORCH_CHECK(W1.size(0) == d_model, "W1 shape mismatch");
    TORCH_CHECK(W2.size(0) == d_ff,    "W2 shape mismatch");
    TORCH_CHECK(W2.size(1) == d_model, "W2 shape mismatch");

    auto check_scale = [](const torch::Tensor& s, int64_t n, const char* nm) {
        TORCH_CHECK(s.dtype() == torch::kFloat32, nm, " must be float32");
        TORCH_CHECK(s.device().is_cuda(),         nm, " must be CUDA");
        TORCH_CHECK(s.numel() == n || s.numel() == 1,
                    nm, " must have ", n, " (or 1) elements");
    };
    check_scale(scale_x,  T,       "scale_x");
    check_scale(scale_W1, d_ff,    "scale_W1");
    check_scale(scale_W2, d_model, "scale_W2");

    auto sx  = scale_x.contiguous();
    auto sw1 = scale_W1.contiguous();
    auto sw2 = scale_W2.contiguous();

    auto out = torch::empty_like(x);
    auto out_scale = torch::empty(
        {T}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    int8_mlp_forward_per_channel(
        x.data_ptr<int8_t>(), W1.data_ptr<int8_t>(), W2.data_ptr<int8_t>(),
        out.data_ptr<int8_t>(),
        sx.data_ptr<float>(), sw1.data_ptr<float>(), sw2.data_ptr<float>(),
        out_scale.data_ptr<float>(),
        batch, seq_len, d_model, d_ff);

    return {out, out_scale};
}

// ── Weight pre-transpose (static-weight inference) ────────────────────
// W1 : (d_model, d_ff)  int8  →  W1T : (d_ff, d_model)
// W2 : (d_ff, d_model)  int8  →  W2T : (d_model, d_ff)
// Run ONCE at load; feed the results to int8_mlp_forward_prepacked on every
// forward to skip the per-call transpose. Returns (W1T, W2T).
std::tuple<torch::Tensor, torch::Tensor> transpose_int8_weights_torch(
    torch::Tensor W1,
    torch::Tensor W2
) {
    TORCH_CHECK(W1.device().is_cuda(),       "W1 must be a CUDA tensor");
    TORCH_CHECK(W1.dtype() == torch::kInt8,  "W1 must be int8");
    TORCH_CHECK(W2.dtype() == torch::kInt8,  "W2 must be int8");
    TORCH_CHECK(W1.is_contiguous(),          "W1 must be contiguous");
    TORCH_CHECK(W2.is_contiguous(),          "W2 must be contiguous");
    TORCH_CHECK(W1.dim() == 2, "W1 must be 2-D (d_model, d_ff)");
    TORCH_CHECK(W2.dim() == 2, "W2 must be 2-D (d_ff, d_model)");

    const int d_model = W1.size(0);
    const int d_ff    = W1.size(1);
    TORCH_CHECK(W2.size(0) == d_ff,    "W2 shape mismatch (expected d_ff rows)");
    TORCH_CHECK(W2.size(1) == d_model, "W2 shape mismatch (expected d_model cols)");

    auto opts = torch::TensorOptions().dtype(torch::kInt8).device(W1.device());
    auto W1T = torch::empty({d_ff, d_model}, opts);
    auto W2T = torch::empty({d_model, d_ff}, opts);

    transpose_int8_weights(
        W1.data_ptr<int8_t>(), W2.data_ptr<int8_t>(),
        W1T.data_ptr<int8_t>(), W2T.data_ptr<int8_t>(),
        d_model, d_ff);

    return {W1T, W2T};
}

// ── INT8 MLP binding (per-tensor, PRE-TRANSPOSED weights) ─────────────
// x   : (batch, seq_len, d_model)  int8
// W1T : (d_ff, d_model)             int8   (from transpose_int8_weights)
// W2T : (d_model, d_ff)            int8
// Same result as int8_mlp_forward; skips the per-forward weight transpose.
// Requires WMMA-path tile multiples (T, d_model, d_ff each a multiple of 128).
std::tuple<torch::Tensor, double> int8_mlp_forward_prepacked_torch(
    torch::Tensor x,
    torch::Tensor W1T,
    torch::Tensor W2T,
    double scale_x,
    double scale_W1,
    double scale_W2
) {
    TORCH_CHECK(x.device().is_cuda(),           "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype()   == torch::kInt8,    "x must be int8");
    TORCH_CHECK(W1T.dtype() == torch::kInt8,    "W1T must be int8");
    TORCH_CHECK(W2T.dtype() == torch::kInt8,    "W2T must be int8");
    TORCH_CHECK(x.is_contiguous(),              "x must be contiguous");
    TORCH_CHECK(W1T.is_contiguous(),            "W1T must be contiguous");
    TORCH_CHECK(W2T.is_contiguous(),            "W2T must be contiguous");
    TORCH_CHECK(x.dim()   == 3, "x must be 3-D (batch, seq_len, d_model)");
    TORCH_CHECK(W1T.dim() == 2, "W1T must be 2-D (d_ff, d_model)");
    TORCH_CHECK(W2T.dim() == 2, "W2T must be 2-D (d_model, d_ff)");

    const int batch   = x.size(0);
    const int seq_len = x.size(1);
    const int d_model = x.size(2);
    const int d_ff    = W1T.size(0);
    const int T       = batch * seq_len;

    TORCH_CHECK(W1T.size(1) == d_model, "W1T shape mismatch (expected d_model cols)");
    TORCH_CHECK(W2T.size(0) == d_model, "W2T shape mismatch (expected d_model rows)");
    TORCH_CHECK(W2T.size(1) == d_ff,    "W2T shape mismatch (expected d_ff cols)");
    TORCH_CHECK(T % 128 == 0 && d_model % 128 == 0 && d_ff % 128 == 0,
                "prepacked path requires the WMMA tile multiples "
                "(batch*seq_len, d_model, d_ff each a multiple of 128)");

    auto out = torch::empty_like(x);

    auto d_sx  = make_scale_tensor(scale_x,  x.device());
    auto d_sw1 = make_scale_tensor(scale_W1, x.device());
    auto d_sw2 = make_scale_tensor(scale_W2, x.device());

    int8_mlp_forward_prepacked(
        x.data_ptr<int8_t>(), W1T.data_ptr<int8_t>(), W2T.data_ptr<int8_t>(),
        out.data_ptr<int8_t>(),
        d_sx.data_ptr<float>(), d_sw1.data_ptr<float>(), d_sw2.data_ptr<float>(),
        batch, seq_len, d_model, d_ff);

    float out_scale;
    int8_mlp_get_output_scale(&out_scale);

    return {out, (double)out_scale};
}

// ── Module registration ────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_attention_forward", &int8_attention_forward_torch,
          "INT8 attention with per-token quantization (CUDA) — returns FP16");
    m.def("int8_decode_attention_forward", &int8_decode_attention_forward_torch,
          "INT8 flash-decoding attention (seq_q==1, INT8 KV cache) — returns FP16");
    // Two overloads: scalar scales (per-tensor) and tensor scales (per-token
    // activation + per-channel weight). pybind11 dispatches by argument type.
    m.def("int8_mlp_forward", &int8_mlp_forward_torch,
          "INT8 MLP, per-tensor scales (CUDA) — returns (int8_out, out_scale)");
    m.def("int8_mlp_forward", &int8_mlp_forward_pc_torch,
          "INT8 MLP, per-token/per-channel scales (CUDA) — returns (int8_out, out_scale)");
    // Static-weight inference: transpose weights once, then call prepacked.
    m.def("transpose_int8_weights", &transpose_int8_weights_torch,
          "Pre-transpose INT8 W1/W2 to [N][K] for the prepacked MLP path — "
          "returns (W1T, W2T)");
    m.def("int8_mlp_forward_prepacked", &int8_mlp_forward_prepacked_torch,
          "INT8 MLP, per-tensor scales, PRE-TRANSPOSED weights (CUDA) — "
          "returns (int8_out, out_scale)");
}
