"""KV-cache and paged-attention kernels ported from vLLM/vLLM-Metal references."""

import math

import mlx.core as mx
import numpy as np
import pytest

from tk import (
    kv_cache_copy_blocks,
    kv_cache_gather,
    kv_cache_scales,
    kv_cache_scatter,
    paged_attention,
)


def _mx_dtype(name):
    return {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }[name]


def _np(x):
    return np.array(x.astype(mx.float32))


def _cast_np(x, dtype):
    return _np(mx.array(x).astype(_mx_dtype(dtype))).astype(np.float32)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_kv_cache_scatter(dtype):
    rng = np.random.default_rng(0)
    T, H, D = 7, 2, 64
    block_size, num_blocks = 4, 3
    key = rng.normal(size=(T, H, D)).astype(np.float32)
    value = rng.normal(size=(T, H, D)).astype(np.float32)
    slots = np.array([0, 2, -1, 5, 8, 1, 7], dtype=np.int64)

    km = mx.array(key).astype(_mx_dtype(dtype))
    vm = mx.array(value).astype(_mx_dtype(dtype))
    got_k, got_v = kv_cache_scatter(km, vm, mx.array(slots), num_blocks, block_size)
    mx.eval(got_k, got_v)

    ref_k = np.zeros((num_blocks, block_size, H, D), np.float32)
    ref_v = np.zeros_like(ref_k)
    key_r = _np(km).astype(np.float32)
    value_r = _np(vm).astype(np.float32)
    for t, slot in enumerate(slots):
        if slot < 0:
            continue
        ref_k[slot // block_size, slot % block_size] = key_r[t]
        ref_v[slot // block_size, slot % block_size] = value_r[t]

    np.testing.assert_allclose(_np(got_k).astype(np.float32), ref_k, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(_np(got_v).astype(np.float32), ref_v, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_kv_cache_gather(dtype):
    rng = np.random.default_rng(1)
    num_blocks, block_size, H, D = 3, 4, 2, 64
    key_cache = rng.normal(size=(num_blocks, block_size, H, D)).astype(np.float32)
    value_cache = rng.normal(size=(num_blocks, block_size, H, D)).astype(np.float32)
    block_table = np.array([[0, 1], [2, 0]], dtype=np.int32)
    cu_seq_lens = np.array([0, 5, 9], dtype=np.int32)
    num_tokens = int(cu_seq_lens[-1])

    km = mx.array(key_cache).astype(_mx_dtype(dtype))
    vm = mx.array(value_cache).astype(_mx_dtype(dtype))
    got_k, got_v = kv_cache_gather(km, vm, mx.array(block_table), mx.array(cu_seq_lens), num_tokens)
    mx.eval(got_k, got_v)

    key_r = _np(km).astype(np.float32)
    value_r = _np(vm).astype(np.float32)
    ref_k = np.empty((num_tokens, H, D), np.float32)
    ref_v = np.empty_like(ref_k)
    for b in range(len(cu_seq_lens) - 1):
        for t in range(cu_seq_lens[b], cu_seq_lens[b + 1]):
            local = t - cu_seq_lens[b]
            block = block_table[b, local // block_size]
            slot = local % block_size
            ref_k[t] = key_r[block, slot]
            ref_v[t] = value_r[block, slot]

    np.testing.assert_allclose(_np(got_k).astype(np.float32), ref_k, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(_np(got_v).astype(np.float32), ref_v, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_kv_cache_copy_blocks(dtype):
    rng = np.random.default_rng(2)
    key_cache = rng.normal(size=(4, 3, 2, 64)).astype(np.float32)
    value_cache = rng.normal(size=(4, 3, 2, 64)).astype(np.float32)
    mapping = np.array([[0, 2], [1, 3]], dtype=np.int64)

    km = mx.array(key_cache).astype(_mx_dtype(dtype))
    vm = mx.array(value_cache).astype(_mx_dtype(dtype))
    got_k, got_v = kv_cache_copy_blocks(km, vm, mx.array(mapping))
    mx.eval(got_k, got_v)

    ref_k = _np(km).astype(np.float32).copy()
    ref_v = _np(vm).astype(np.float32).copy()
    for src, dst in mapping:
        ref_k[dst] = ref_k[src]
        ref_v[dst] = ref_v[src]

    np.testing.assert_allclose(_np(got_k).astype(np.float32), ref_k, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(_np(got_v).astype(np.float32), ref_v, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_kv_cache_scales(dtype):
    rng = np.random.default_rng(3)
    key = rng.normal(size=(17, 2, 64)).astype(np.float32)
    value = rng.normal(size=(17, 2, 64)).astype(np.float32)

    km = mx.array(key).astype(_mx_dtype(dtype))
    vm = mx.array(value).astype(_mx_dtype(dtype))
    got_k, got_v = kv_cache_scales(km, vm)
    mx.eval(got_k, got_v)

    ref_k = np.abs(_np(km).astype(np.float32)).max() / 240.0
    ref_v = np.abs(_np(vm).astype(np.float32)).max() / 240.0
    np.testing.assert_allclose(_np(got_k), np.array([ref_k], np.float32), atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(_np(got_v), np.array([ref_v], np.float32), atol=1e-7, rtol=1e-6)


def _paged_ref(q, key_cache, value_cache, block_table, context_lens, scale):
    B, H, D = q.shape
    block_size = key_cache.shape[1]
    out = np.zeros_like(q, dtype=np.float32)
    for b in range(B):
        for h in range(H):
            scores = []
            vals = []
            for t in range(context_lens[b]):
                block = block_table[b, t // block_size]
                slot = t % block_size
                k = key_cache[block, slot, h]
                scores.append(float(np.dot(q[b, h], k) * scale))
                vals.append(value_cache[block, slot, h])
            if not scores:
                continue
            s = np.array(scores, np.float32)
            p = np.exp(s - s.max())
            p /= p.sum()
            out[b, h] = np.sum(p[:, None] * np.stack(vals, axis=0), axis=0)
    return out


@pytest.mark.parametrize("dtype,atol", [("float32", 2e-5), ("float16", 2e-3), ("bfloat16", 2e-2)])
@pytest.mark.parametrize("D", [64, 128])
def test_paged_attention(dtype, atol, D):
    rng = np.random.default_rng(4 + D)
    B, H = 2, 2
    num_blocks, block_size = 4, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    key_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H, D))).astype(np.float32)
    value_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H, D))).astype(np.float32)
    block_table = np.array([[0, 1], [2, 3]], dtype=np.int32)
    context_lens = np.array([6, 7], dtype=np.int32)
    scale = 1.0 / math.sqrt(D)

    qm = mx.array(q).astype(_mx_dtype(dtype))
    km = mx.array(key_cache).astype(_mx_dtype(dtype))
    vm = mx.array(value_cache).astype(_mx_dtype(dtype))
    got = paged_attention(qm, km, vm, mx.array(block_table), mx.array(context_lens), scale=0.0)
    mx.eval(got)

    ref = _paged_ref(
        _np(qm).astype(np.float32),
        _np(km).astype(np.float32),
        _np(vm).astype(np.float32),
        block_table,
        context_lens,
        scale,
    )
    np.testing.assert_allclose(_np(got).astype(np.float32), ref, atol=atol, rtol=2e-3)
