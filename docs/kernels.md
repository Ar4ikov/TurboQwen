# Kernels: what these checkpoints run on, measured

Every linear layer of the body (48 Gated-DeltaNet layers, 16 full-attention layers) is
int4 asymmetric, group 128, in compressed-tensors `pack-quantized` layout. vLLM serves
that with the Marlin kernel: bf16 activations by default (W4A16), int8 activations when
`VLLM_MARLIN_INPUT_DTYPE=int8` is set (HyperQwen's `INT8_ACT=int8`). This page is the
measurement behind two decisions in this repo.

## 1. The int8 path was compiled for these weights all along

vLLM's `csrc/.../marlin/generate_kernels.py` instantiates the kernel for every
(activation, weight) pairing it supports. Among them:

```
# AWQ-INT4 with INT8 activation
{"a_type": ["kS8"], "b_type": "kU4", ...}      # zero-point weights, int8 activations
# GPTQ-INT4 with INT8 activation
{"a_type": ["kS8"], "b_type": "kU4B8", ...}    # symmetric weights, int8 activations
```

`kU4` is exactly what a `symmetric: false` compressed-tensors export loads as, the
wrapper already permutes zero points for 8-bit activations (`marlin_zero_points(...,
is_a_8bit)`), and `tests/kernels/quantization/test_marlin_gemm.py` covers the pairing.
What refused it were two Python asserts written to keep 8-bit *weights* off the int8 path:

```python
if is_a_8bit:
    assert c.weight_type == scalar_types.uint4b8, "W8A8 is not supported by marlin kernel."
```

`hyperqwen/patches/marlin-int8-asym-zp.patch` admits `uint4` next to `uint4b8` in both
places (the kernel wrapper and `apply_gptq_marlin_linear`). Numerically, on a random
asymmetric int4 g128 weight at this model's shapes
(`hyperqwen/bench/test_marlin_int8_asym.py`, RTX 3090):

| shape (K x N) | M | W4A16 rel. error | W4A8-int8 rel. error |
|---|---|---|---|
| 5120 x 17408 | 1 / 5 / 16 / 256 | 0.0026 | 0.0090 – 0.0093 |
| 17408 x 5120 | 1 / 5 / 16 / 256 | 0.0026 | 0.0099 – 0.0105 |
| 5120 x 12288 | 1 / 5 / 16 / 256 | 0.0026 | 0.0092 – 0.0096 |
| 5120 x 6144 | 1 / 5 / 16 / 256 | 0.0026 | 0.0092 – 0.0104 |
| 5120 x 5120 | 1 / 5 / 16 / 256 | 0.0026 | 0.0086 – 0.0092 |
| 5120 x 1024 | 1 / 5 / 16 / 256 | 0.0026 | 0.0086 – 0.0095 |

The 0.26% of the bf16 path is output rounding; the ~1% of the int8 path is the per-token
activation quantization, the same noise a symmetric checkpoint pays on that path. Nothing
about the zero points shows up in the error.

## 2. Why there is no hand-written GEMV here (yet)

The request was a backend with custom kernels. Before writing one, the question is what
a kernel could win at decode, where the GEMMs see M = 1 (plain decode) to M = 5 or 8
(the MTP / DFlash2 verify block). `boost/bench_marlin_small_m.py` times Marlin on the
four GEMM shapes of this model against the bytes it has to read (RTX 3090, 936 GB/s
peak):

| shape (K x N) | M | W4A16 µs | W4A16 GB/s | W4A8-int8 µs | W4A8 GB/s |
|---|---|---|---|---|---|
| 5120 x 17408 (gate/up) | 1 | 60.7 | 763 | 112.9 | 410 |
| 5120 x 17408 | 5 | 61.0 | 762 | 113.2 | 411 |
| 5120 x 17408 | 8 | 61.3 | 761 | 124.0 | 376 |
| 17408 x 5120 (down) | 1 | 59.7 | 777 | 115.5 | 401 |
| 17408 x 5120 | 5 | 59.7 | 779 | 113.9 | 408 |
| 5120 x 12288 (q, gated) | 1 | 44.9 | 728 | 113.0 | 290 |
| 5120 x 6144 (out/o_proj class) | 1 | 41.8 | 392 | 110.7 | 148 |
| 5120 x 6144 | 5 | 42.5 | 387 | 127.6 | 129 |

Three readings:

- **The big GEMMs already run at 81–83% of the card's bandwidth** at M = 1..8. A
  hand-written dequant-GEMV for the asymmetric layout would top out around 85–90%; that
  is a few percent of GEMM time, and GEMM time is roughly 60% of a decode step. Not
  worth a second weight layout in memory and a second code path under CUDA graphs.
- **The small-N GEMMs are the inefficiency**: `out_proj` / `o_proj` (6144 → 5120) reach
  42% because 5120 outputs do not fill 82 SMs with Marlin's tile shapes. There are 64 of
  them per step, ~25 µs above the floor each: about 1.6 ms of a ~25 ms step, so ≤ 6%
  of single-stream decode is the entire prize. HyperQwen's `marlin-tune-table` patch
  (off by default, needs a locally built extension) is the upstream answer to exactly
  this; a kernel of our own would compete with it for the same 6%.
- **The int8 path is 2x slower than bf16 at small M in isolation** (an extra
  per-token quantization launch, and the int8 MMA tiles start at 16 rows). It pays off
  where the GEMM is compute-bound: prefill and batch mode, which is why HyperQwen ships
  `INT8_ACT=int8` in batch mode and as the prefill knob, and why the asym patch matters
  there and not for single-stream decode.

So the speed levers for these checkpoints are the ones this repo pursues: the int4-GPTQ
`lm_head` (1.27 → 0.65 GB read per verify step), the draft head, DFlash2, and the
int8 path where it is compute-bound. A dedicated small-N kernel is the next thing to
try if those are exhausted; the benchmark above is the yardstick it has to beat.

Run both scripts inside the container or the venv (they bring up a one-process vLLM
context, no server needed):

```bash
CUDA_VISIBLE_DEVICES=0 venv/bin/python bench/test_marlin_int8_asym.py    # in hyperqwen/
CUDA_VISIBLE_DEVICES=0 venv/bin/python boost/bench_marlin_small_m.py
```
