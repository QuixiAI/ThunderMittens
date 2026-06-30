"""Correctness tests for the runtime per-token GPU quantizers (fp8 e4m3, int8).

Validates: (1) per-row scale == absmax/QMAX exactly, and (2) the round-to-nearest
reconstruction error is within half a quantization step everywhere.

Run from kernels/:  python -m pytest quant_rt/correctness/test_quant_rt.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import quantize_per_token_fp8, quantize_per_token_int8
from tk.quant import _e4m3_decode_arr

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
SHAPES = [(8, 256), (4, 64, 128), (3, 513)]  # last is non-multiple-of-32 hidden


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("shape", SHAPES)
def test_quantize_per_token_fp8(dtype, shape):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    D = shape[-1]
    xq = mx.array(x).astype(_MX[dtype])
    codes, scale = quantize_per_token_fp8(xq)
    mx.eval(codes, scale)

    xd = np.array(xq.astype(mx.float32)).reshape(-1, D)
    amax = np.abs(xd).max(axis=1)
    ref_scale = amax / 448.0
    ssafe = np.maximum(ref_scale, 1e-30)[:, None]

    np.testing.assert_allclose(np.array(scale).reshape(-1), ref_scale, rtol=1e-3, atol=1e-8)

    deq = _e4m3_decode_arr(np.array(codes).reshape(-1, D)) * ssafe
    # RNE error <= half ULP: 2^-4 relative for e4m3 normals, + 2 subnormal steps near zero.
    tol = 0.0625 * np.abs(xd) + 2.0 * ssafe
    assert np.all(np.abs(deq - xd) <= tol), \
        f"max excess {(np.abs(deq - xd) - tol).max()}"


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("shape", SHAPES)
def test_quantize_per_token_int8(dtype, shape):
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    D = shape[-1]
    xq = mx.array(x).astype(_MX[dtype])
    codes, scale = quantize_per_token_int8(xq)
    mx.eval(codes, scale)

    xd = np.array(xq.astype(mx.float32)).reshape(-1, D)
    amax = np.abs(xd).max(axis=1)
    ref_scale = amax / 127.0
    ssafe = np.maximum(ref_scale, 1e-30)[:, None]

    np.testing.assert_allclose(np.array(scale).reshape(-1), ref_scale, rtol=1e-3, atol=1e-8)

    c = np.array(codes).astype(np.int32)
    assert c.min() >= -127 and c.max() <= 127
    deq = c.reshape(-1, D).astype(np.float32) * ssafe
    # round-to-nearest int: error <= half a step (= half the scale).
    assert np.all(np.abs(deq - xd) <= 0.5 * ssafe + 1e-6)


if __name__ == "__main__":
    for shp in SHAPES:
        test_quantize_per_token_fp8("float32", shp)
        test_quantize_per_token_int8("float32", shp)
        print("ok", shp)
