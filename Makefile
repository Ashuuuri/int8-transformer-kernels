# Makefile — pure-CUDA smoke test for the INT8 kernels (A100 / sm_80).
#
# Usage:
#   python tests/gen_testdata.py   # one-time: write the .bin fixtures
#   make test_int8 && ./test_int8  # build + run the correctness smoke test
#   make clean
#
# The Python correctness + benchmark + validation flow (tests/test_int8.py,
# validation/validate_int8.py, bench/sweep.py) JIT-builds the kernels via torch
# cpp_extension and needs no Makefile — see README.md / CLAUDE.md.

NVCC       = nvcc
NVCC_FLAGS = -arch=sm_80 --std=c++17 -O3
INCLUDES   = -I kernels -I tests/cuda
DFLAGS     ?=

.PHONY: all clean

all: test_int8

# ── INT8 kernels (pure-CUDA smoke test, reads tests/.bin fixtures) ──────────
test_int8: kernels/int8_attention.cu kernels/int8_mlp.cu kernels/quant_utils.cu tests/cuda/test_int8.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(DFLAGS) \
		kernels/int8_attention.cu \
		kernels/int8_mlp.cu \
		kernels/quant_utils.cu \
		tests/cuda/test_int8.cu \
		-o test_int8

clean:
	rm -f test_int8
