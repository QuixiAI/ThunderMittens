"""Correctness test for the ThunderMittens RMSNorm Metal kernel (oracle mx.fast.rms_norm).

Run from kernels/:  python -m pytest rms_norm/correctness/test_rms_norm.py -v
"""

import mlx.core as mx
import pytest

from tk import rms_norm


def ref_rms_norm(x, w, eps):
    xf = x.astype(mx.float32)
    ms = (xf * xf).mean(axis=-1, keepdims=True)
    return (xf * mx.rsqrt(ms + eps) * w.astype(mx.float32)).astype(mx.bfloat16)


SHAPES = [(2, 128, 1024), (4, 64, 512), (1, 256, 768), (8, 256)]


@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_matches_mlx(shape):
    eps = 1e-5
    D = shape[-1]
    mx.random.seed(0)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    w = mx.random.normal((D,)).astype(mx.bfloat16)
    got = rms_norm(x, w, eps=eps)
    exp_mlx = mx.fast.rms_norm(x, w, eps)
    exp_ref = ref_rms_norm(x, w, eps)
    mx.eval(got, exp_mlx, exp_ref)
    assert got.shape == x.shape and got.dtype == mx.bfloat16
    assert mx.allclose(got, exp_mlx, atol=2e-2, rtol=2e-2), \
        f"vs mlx: {mx.max(mx.abs(got.astype(mx.float32)-exp_mlx.astype(mx.float32))).item()}"
    assert mx.allclose(got, exp_ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    for shp in SHAPES:
        test_rms_norm_matches_mlx(shp)
        print("ok", shp)
