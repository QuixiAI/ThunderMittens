"""Correctness tests for the fused residual-add + norm Metal kernels.

The kernels return two arrays: out = norm(x + residual) [* weight (+ bias)], and
res_out = x + residual (the summed residual the next block reads). The kernel
normalizes the fp32 sum and writes the bf16-rounded sum back.

Run from kernels/:  python -m pytest add_norm/correctness/test_add_norm.py -v
"""

import mlx.core as mx
import pytest

from tk import rms_norm_add, layernorm_add


def ref_rms_norm(sum_f32, w, eps):
    ms = (sum_f32 * sum_f32).mean(axis=-1, keepdims=True)
    return (sum_f32 * mx.rsqrt(ms + eps) * w.astype(mx.float32)).astype(mx.bfloat16)


def ref_layernorm(sum_f32, w, b, eps):
    mean = sum_f32.mean(axis=-1, keepdims=True)
    var = ((sum_f32 - mean) ** 2).mean(axis=-1, keepdims=True)
    y = (sum_f32 - mean) * mx.rsqrt(var + eps) * w.astype(mx.float32) + b.astype(mx.float32)
    return y.astype(mx.bfloat16)


SHAPES = [(2, 128, 1024), (4, 64, 512), (1, 256, 768), (8, 256)]


@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_add(shape):
    eps = 1e-5
    D = shape[-1]
    mx.random.seed(0)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    r = mx.random.normal(shape).astype(mx.bfloat16)
    w = mx.random.normal((D,)).astype(mx.bfloat16)

    out, added = rms_norm_add(x, r, w, eps=eps)

    sum_f32 = x.astype(mx.float32) + r.astype(mx.float32)
    added_ref = sum_f32.astype(mx.bfloat16)
    out_ref = ref_rms_norm(sum_f32, w, eps)
    mx.eval(out, added, added_ref, out_ref)

    assert out.shape == x.shape and added.shape == x.shape
    assert out.dtype == mx.bfloat16 and added.dtype == mx.bfloat16
    assert mx.allclose(added, added_ref, atol=2e-2, rtol=2e-2)
    assert mx.allclose(out, out_ref, atol=2e-2, rtol=2e-2), \
        f"max {mx.max(mx.abs(out.astype(mx.float32)-out_ref.astype(mx.float32))).item()}"


@pytest.mark.parametrize("shape", SHAPES)
def test_layernorm_add(shape):
    eps = 1e-5
    D = shape[-1]
    mx.random.seed(1)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    r = mx.random.normal(shape).astype(mx.bfloat16)
    w = mx.random.normal((D,)).astype(mx.bfloat16)
    b = mx.random.normal((D,)).astype(mx.bfloat16)

    out, added = layernorm_add(x, r, w, b, eps=eps)

    sum_f32 = x.astype(mx.float32) + r.astype(mx.float32)
    added_ref = sum_f32.astype(mx.bfloat16)
    out_ref = ref_layernorm(sum_f32, w, b, eps)
    mx.eval(out, added, added_ref, out_ref)

    assert mx.allclose(added, added_ref, atol=2e-2, rtol=2e-2)
    assert mx.allclose(out, out_ref, atol=2e-2, rtol=2e-2), \
        f"max {mx.max(mx.abs(out.astype(mx.float32)-out_ref.astype(mx.float32))).item()}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_rms_norm_add(shp)
        test_layernorm_add(shp)
        print("ok", shp)
