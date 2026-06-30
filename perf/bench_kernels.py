#!/usr/bin/env python3
"""Benchmark ThunderMittens kernels and write JSONL + Markdown summaries.

Run from the repository root, for example:

    .venv/bin/python perf/bench_kernels.py --backend mlx --preset smoke --kernel all

The harness intentionally has no pandas/plotting dependency. It measures target
and framework/decomposed baselines where practical, validates target output
against an oracle, and records skips instead of aborting an entire run.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
KERNELS_DIR = REPO_ROOT / "ThunderMittens" / "kernels"
if str(KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(KERNELS_DIR))

SCHEMA_VERSION = 1
COMMON_QUANT_FORMATS = ["q8_0", "q4_0", "q4_K", "kU4B8", "fp8_e4m3", "bitnet"]


class SkipCase(RuntimeError):
    """Raised by a case factory when a case is unsupported in this environment."""


@dataclass(frozen=True)
class CaseSpec:
    kernel: str
    variant: str
    shape: tuple[int, ...]
    dtype: str = "bf16"
    fmt: str | None = None
    tags: tuple[str, ...] = ()
    factory: Callable[[str], "PreparedCase"] | None = None


@dataclass
class PreparedCase:
    target: Callable[[], Any]
    reference: Callable[[], Any]
    baseline: Callable[[], Any] | None = None
    target_name: str = "tm"
    baseline_name: str = "framework"
    atol: float = 1e-2
    rtol: float = 1e-2
    bytes_moved: int | None = None
    weight_bytes: int | None = None
    flops: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _load_mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import tk
        from tk import quant
        return mx, nn, tk, quant
    except Exception as exc:  # pragma: no cover - depends on local build
        raise SkipCase(f"MLX backend unavailable: {exc}") from exc


def _load_torch():
    try:
        import torch
        import tk
    except Exception as exc:  # pragma: no cover - optional backend
        raise SkipCase(f"PyTorch backend unavailable: {exc}") from exc
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        raise SkipCase("PyTorch MPS backend unavailable")
    return torch, tk


def _git_info() -> tuple[str, bool]:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        rev = "unknown"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except Exception:
        dirty = True
    return rev, dirty


def _version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "__version__", None) or "unknown"
    except Exception:
        return None


def _run_metadata(run_id: str, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    rev, dirty = _git_info()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "user": getpass.getuser(),
        "host": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx": _version("mlx.core"),
        "torch": _version("torch"),
        "git_revision": rev,
        "git_dirty": dirty,
        "command": " ".join(sys.argv),
        "backend": args.backend,
        "preset": args.preset,
        "kernels": args.kernel,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "output_dir": str(output_dir),
    }


def _rng(seed: int, salt: str) -> np.random.Generator:
    salted = (seed + sum((i + 1) * ord(c) for i, c in enumerate(salt))) % (2**32)
    return np.random.default_rng(salted)


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


def _sync_value(value: Any, backend: str) -> None:
    if backend == "mlx":
        mx, _, _, _ = _load_mlx()
        vals = [v for v in _flatten(value) if hasattr(v, "shape")]
        if vals:
            mx.eval(*vals)
    elif backend == "torch":
        torch, _ = _load_torch()
        if hasattr(torch, "mps"):
            torch.mps.synchronize()


def _as_numpy(value: Any) -> list[np.ndarray]:
    arrays = []
    for item in _flatten(value):
        module = type(item).__module__.split(".")[0]
        if module == "mlx":
            arrays.append(np.array(item.astype(_load_mlx()[0].float32)))
        elif module == "torch":
            arrays.append(item.detach().float().cpu().numpy())
        elif isinstance(item, np.ndarray):
            arrays.append(item.astype(np.float32, copy=False))
        else:
            arrays.append(np.asarray(item, dtype=np.float32))
    return arrays


def _error_metrics(got: Any, ref: Any) -> tuple[float, float]:
    got_arrays = _as_numpy(got)
    ref_arrays = _as_numpy(ref)
    if len(got_arrays) != len(ref_arrays):
        return math.inf, math.inf
    max_abs = 0.0
    max_rel = 0.0
    for g, r in zip(got_arrays, ref_arrays):
        diff = np.abs(g.astype(np.float32) - r.astype(np.float32))
        abs_err = float(np.nanmax(diff)) if diff.size else 0.0
        scale = float(np.nanmax(np.abs(r.astype(np.float32)))) + 1e-9 if r.size else 1.0
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, abs_err / scale)
    return max_abs, max_rel


def _time_callable(fn: Callable[[], Any], backend: str, warmup: int, iters: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        _sync_value(fn(), backend)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iters):
            _sync_value(fn(), backend)
        samples.append(1e3 * (time.perf_counter() - start) / max(iters, 1))
    return samples


def _summary_stats(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"median_ms": None, "p20_ms": None, "p80_ms": None, "min_ms": None, "max_ms": None}
    ss = sorted(samples)
    return {
        "median_ms": float(statistics.median(ss)),
        "p20_ms": float(ss[min(len(ss) - 1, int(0.2 * (len(ss) - 1)))]),
        "p80_ms": float(ss[min(len(ss) - 1, int(0.8 * (len(ss) - 1)))]),
        "min_ms": float(ss[0]),
        "max_ms": float(ss[-1]),
    }


def _dtype_mlx(mx: Any, dtype: str) -> Any:
    return {
        "f32": mx.float32,
        "f16": mx.float16,
        "bf16": mx.bfloat16,
    }[dtype]


def _dtype_torch(torch: Any, dtype: str) -> Any:
    return {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype]


def _mlx_array(arr: np.ndarray, dtype: str):
    mx, _, _, _ = _load_mlx()
    return mx.array(arr).astype(_dtype_mlx(mx, dtype))


def _torch_array(arr: np.ndarray, dtype: str):
    torch, _ = _load_torch()
    return torch.tensor(arr, device="mps", dtype=_dtype_torch(torch, dtype))


def _torch_int_array(arr: np.ndarray):
    torch, _ = _load_torch()
    dtype = torch.int8 if arr.dtype == np.int8 else torch.uint8 if arr.dtype == np.uint8 else torch.int32
    return torch.tensor(arr, device="mps", dtype=dtype)


def _gelu_tanh_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def _cos_sin(N: int, D: int, base: float = 10000.0):
    inv_freq = base ** (-(np.arange(0, D, 2).astype(np.float32) / D))
    pos = np.arange(N).astype(np.float32)[:, None]
    ang = pos * inv_freq[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32), inv_freq


def _attn_ref_np(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = False) -> np.ndarray:
    D = q.shape[-1]
    s = (q @ np.swapaxes(k, -1, -2)) / math.sqrt(D)
    if causal:
        N = q.shape[-2]
        s = np.where(np.triu(np.ones((N, N), bool), 1), -1e30, s)
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    p = p / p.sum(axis=-1, keepdims=True)
    return p @ v


def _fft_matrix(N: int, inverse: bool = False) -> np.ndarray:
    n = np.arange(N)
    k = n.reshape(-1, 1)
    sign = 1 if inverse else -1
    return np.exp(sign * 2j * np.pi * n * k / N)


def _twiddle(n: int, m: int, sign: int) -> np.ndarray:
    na = np.arange(n).reshape(-1, 1)
    ma = np.arange(m)
    return np.exp(sign * 2j * np.pi * na * ma / (n * m))


def _stack_complex_mlx(z: np.ndarray):
    mx, _, _, _ = _load_mlx()
    return mx.array(np.stack([z.real, z.imag]).astype(np.float32))


def _case_mlx_pointwise(kernel: str, shape: tuple[int, ...], dtype: str, seed: int) -> PreparedCase:
    mx, nn, tk, _ = _load_mlx()
    rg = _rng(seed, kernel + str(shape) + dtype)
    x_np = rg.standard_normal(shape).astype(np.float32)
    x = _mlx_array(x_np, dtype)
    elem = int(np.prod(shape))
    bytes_per = 4 if dtype == "f32" else 2
    if kernel == "add_rt":
        y = _mlx_array(rg.standard_normal(shape).astype(np.float32), dtype)
        return PreparedCase(lambda: tk.add_rt(x, y), lambda: x + y, lambda: x + y,
                            atol=1e-2, rtol=1e-2, bytes_moved=elem * bytes_per * 3)
    if kernel == "gelu":
        return PreparedCase(lambda: tk.gelu(x),
                            lambda: nn.gelu_approx(x.astype(mx.float32)).astype(mx.bfloat16),
                            lambda: nn.gelu_approx(x.astype(mx.float32)).astype(mx.bfloat16),
                            atol=2e-2, rtol=2e-2, bytes_moved=elem * bytes_per * 2)
    raise SkipCase(f"unknown pointwise kernel {kernel}")


def _case_torch_pointwise(kernel: str, shape: tuple[int, ...], dtype: str, seed: int) -> PreparedCase:
    torch, tk = _load_torch()
    rg = _rng(seed, "torch" + kernel + str(shape) + dtype)
    x_np = rg.standard_normal(shape).astype(np.float32)
    x = _torch_array(x_np, dtype)
    elem = int(np.prod(shape))
    bytes_per = 4 if dtype == "f32" else 2
    if kernel == "add_rt":
        y = _torch_array(rg.standard_normal(shape).astype(np.float32), dtype)
        return PreparedCase(lambda: tk.add_rt(x, y), lambda: x + y, lambda: x + y,
                            atol=1e-2, rtol=1e-2, bytes_moved=elem * bytes_per * 3)
    if kernel == "gelu":
        return PreparedCase(lambda: tk.gelu(x),
                            lambda: torch.nn.functional.gelu(x.float(), approximate="tanh").to(x.dtype),
                            lambda: torch.nn.functional.gelu(x.float(), approximate="tanh").to(x.dtype),
                            atol=2e-2, rtol=2e-2, bytes_moved=elem * bytes_per * 2)
    raise SkipCase(f"unknown pointwise kernel {kernel}")


def _case_mlx_row(kernel: str, shape: tuple[int, ...], seed: int) -> PreparedCase:
    mx, _, tk, _ = _load_mlx()
    rg = _rng(seed, kernel + str(shape))
    x = mx.array(rg.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
    D = shape[-1]
    elem = int(np.prod(shape))
    if kernel == "layernorm":
        w = mx.array(rg.standard_normal((D,)).astype(np.float32)).astype(mx.bfloat16)
        b = mx.array(rg.standard_normal((D,)).astype(np.float32)).astype(mx.bfloat16)
        eps = 1e-5
        return PreparedCase(lambda: tk.layernorm(x, w, b, eps=eps),
                            lambda: mx.fast.layer_norm(x, w, b, eps),
                            lambda: mx.fast.layer_norm(x, w, b, eps),
                            atol=2e-2, rtol=2e-2, bytes_moved=(elem * 2 + 2 * D) * 2)
    if kernel == "rms_norm":
        w = mx.array(rg.standard_normal((D,)).astype(np.float32)).astype(mx.bfloat16)
        eps = 1e-5
        return PreparedCase(lambda: tk.rms_norm(x, w, eps=eps),
                            lambda: mx.fast.rms_norm(x, w, eps),
                            lambda: mx.fast.rms_norm(x, w, eps),
                            atol=2e-2, rtol=2e-2, bytes_moved=(elem * 2 + D) * 2)
    if kernel == "softmax":
        return PreparedCase(lambda: tk.softmax(x),
                            lambda: mx.softmax(x.astype(mx.float32), axis=-1).astype(mx.bfloat16),
                            lambda: mx.softmax(x.astype(mx.float32), axis=-1).astype(mx.bfloat16),
                            atol=2e-2, rtol=2e-2, bytes_moved=elem * 4)
    raise SkipCase(f"unknown row kernel {kernel}")


def _case_torch_row(kernel: str, shape: tuple[int, ...], seed: int) -> PreparedCase:
    torch, tk = _load_torch()
    rg = _rng(seed, "torch" + kernel + str(shape))
    x = _torch_array(rg.standard_normal(shape).astype(np.float32), "bf16")
    D = shape[-1]
    elem = int(np.prod(shape))
    if kernel == "layernorm":
        w = _torch_array(rg.standard_normal((D,)).astype(np.float32), "bf16")
        b = _torch_array(rg.standard_normal((D,)).astype(np.float32), "bf16")
        eps = 1e-5
        ref = lambda: torch.nn.functional.layer_norm(x.float(), (D,), w.float(), b.float(), eps).to(torch.bfloat16)
        return PreparedCase(lambda: tk.layernorm(x, w, b, eps), ref, ref,
                            atol=2e-2, rtol=2e-2, bytes_moved=(elem * 2 + 2 * D) * 2)
    if kernel == "rms_norm":
        w = _torch_array(rg.standard_normal((D,)).astype(np.float32), "bf16")
        eps = 1e-5
        ref = lambda: (x.float() * torch.rsqrt((x.float() * x.float()).mean(dim=-1, keepdim=True) + eps)
                       * w.float()).to(torch.bfloat16)
        return PreparedCase(lambda: tk.rms_norm(x, w, eps), ref, ref,
                            atol=2e-2, rtol=2e-2, bytes_moved=(elem * 2 + D) * 2)
    if kernel == "softmax":
        ref = lambda: torch.softmax(x.float(), dim=-1).to(torch.bfloat16)
        return PreparedCase(lambda: tk.softmax(x), ref, ref,
                            atol=2e-2, rtol=2e-2, bytes_moved=elem * 4)
    raise SkipCase(f"unknown row kernel {kernel}")


def _case_mlx_rotary(shape: tuple[int, ...], seed: int) -> PreparedCase:
    mx, _, tk, _ = _load_mlx()
    B, H, N, D = shape
    rg = _rng(seed, "rotary" + str(shape))
    cos_np, sin_np, inv_freq = _cos_sin(N, D)
    x = mx.array(rg.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
    cos = mx.array(cos_np).astype(mx.bfloat16)
    sin = mx.array(sin_np).astype(mx.bfloat16)
    ref = lambda: mx.fast.rope(x, dims=D, traditional=False, base=None, scale=1.0,
                               offset=0, freqs=mx.array(1.0 / inv_freq))
    return PreparedCase(lambda: tk.rotary(x, cos, sin), ref, ref,
                        atol=3e-2, rtol=3e-2, bytes_moved=int(np.prod(shape)) * 4 + N * D * 2)


def _case_torch_rotary(shape: tuple[int, ...], seed: int) -> PreparedCase:
    torch, tk = _load_torch()
    B, H, N, D = shape
    rg = _rng(seed, "torch_rotary" + str(shape))
    cos_np, sin_np, _ = _cos_sin(N, D)
    x = _torch_array(rg.standard_normal(shape).astype(np.float32), "bf16")
    cos = _torch_array(cos_np, "bf16")
    sin = _torch_array(sin_np, "bf16")

    def ref():
        xf = x.float()
        x1, x2 = xf[..., :D // 2], xf[..., D // 2:]
        c = cos.float()[None, None]
        s = sin.float()[None, None]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(torch.bfloat16)

    return PreparedCase(lambda: tk.rotary(x, cos, sin), ref, ref,
                        atol=3e-2, rtol=3e-2, bytes_moved=int(np.prod(shape)) * 4 + N * D * 2)


def _case_mlx_gemm(kernel: str, shape: tuple[int, int, int], dtype: str, seed: int) -> PreparedCase:
    mx, nn, tk, _ = _load_mlx()
    N, K, M = shape
    rg = _rng(seed, kernel + str(shape) + dtype)
    x = _mlx_array(rg.random((N, K), dtype=np.float32), dtype)
    y = _mlx_array(rg.random((K, M), dtype=np.float32), dtype)
    dtype_obj = _dtype_mlx(mx, dtype)
    ref = lambda: (x.astype(mx.float32) @ y.astype(mx.float32)).astype(dtype_obj)
    flops = 2.0 * N * K * M
    bytes_moved = (N * K + K * M + N * M) * (4 if dtype == "f32" else 2)
    if kernel == "matmul_custom":
        return PreparedCase(lambda: tk.matmul_custom(x, y), ref, ref, atol=0.5 if dtype == "bf16" else 1e-3,
                            rtol=5e-2 if dtype == "bf16" else 1e-4, bytes_moved=bytes_moved, flops=flops)
    if kernel == "gemm_staged":
        return PreparedCase(lambda: tk.gemm_staged(x, y), ref, ref, atol=0.5 if dtype == "bf16" else 1e-3,
                            rtol=0.5 if dtype == "bf16" else 1e-3, bytes_moved=bytes_moved, flops=flops)
    if kernel == "flux_gelu":
        bias = _mlx_array(rg.standard_normal((M,)).astype(np.float32), dtype)
        ref_flux = lambda: nn.gelu_approx(
            x.astype(mx.float32) @ y.astype(mx.float32) + bias.astype(mx.float32)).astype(dtype_obj)
        return PreparedCase(lambda: tk.flux_gelu(x, y, bias), ref_flux, ref_flux,
                            atol=0.5 if dtype == "bf16" else 1e-2, rtol=0.5 if dtype == "bf16" else 1e-2,
                            bytes_moved=bytes_moved + M * 2, flops=flops)
    if kernel == "flux_gate":
        bias = _mlx_array(rg.standard_normal((M,)).astype(np.float32), dtype)
        gate = _mlx_array(rg.standard_normal((M,)).astype(np.float32), dtype)
        residual = _mlx_array(rg.standard_normal((N, M)).astype(np.float32), dtype)
        ref_gate = lambda: ((x.astype(mx.float32) @ y.astype(mx.float32) + bias.astype(mx.float32))
                            * gate.astype(mx.float32) + residual.astype(mx.float32)).astype(dtype_obj)
        return PreparedCase(lambda: tk.flux_gate(x, y, bias, gate, residual), ref_gate, ref_gate,
                            atol=0.5 if dtype == "bf16" else 1e-2, rtol=0.5 if dtype == "bf16" else 1e-2,
                            bytes_moved=bytes_moved + M * 4 + N * M * 2, flops=flops)
    raise SkipCase(f"unknown GEMM kernel {kernel}")


def _case_torch_gemm(kernel: str, shape: tuple[int, int, int], dtype: str, seed: int) -> PreparedCase:
    torch, tk = _load_torch()
    N, K, M = shape
    rg = _rng(seed, "torch" + kernel + str(shape) + dtype)
    x = _torch_array(rg.random((N, K), dtype=np.float32), dtype)
    y = _torch_array(rg.random((K, M), dtype=np.float32), dtype)
    tdtype = _dtype_torch(torch, dtype)
    ref = lambda: (x.float() @ y.float()).to(tdtype)
    flops = 2.0 * N * K * M
    bytes_moved = (N * K + K * M + N * M) * (4 if dtype == "f32" else 2)
    if kernel == "matmul_custom":
        return PreparedCase(lambda: tk.matmul_custom(x, y), ref, ref, atol=0.5 if dtype == "bf16" else 1e-3,
                            rtol=5e-2 if dtype == "bf16" else 1e-4, bytes_moved=bytes_moved, flops=flops)
    if kernel == "gemm_staged":
        return PreparedCase(lambda: tk.gemm_staged(x, y), ref, ref, atol=0.5 if dtype == "bf16" else 1e-3,
                            rtol=0.5 if dtype == "bf16" else 1e-3, bytes_moved=bytes_moved, flops=flops)
    if kernel == "flux_gelu":
        bias = _torch_array(rg.standard_normal((M,)).astype(np.float32), dtype)
        ref_flux = lambda: torch.nn.functional.gelu(x.float() @ y.float() + bias.float(), approximate="tanh").to(tdtype)
        return PreparedCase(lambda: tk.flux_gelu(x, y, bias), ref_flux, ref_flux,
                            atol=0.5 if dtype == "bf16" else 1e-2, rtol=0.5 if dtype == "bf16" else 1e-2,
                            bytes_moved=bytes_moved + M * 2, flops=flops)
    if kernel == "flux_gate":
        bias = _torch_array(rg.standard_normal((M,)).astype(np.float32), dtype)
        gate = _torch_array(rg.standard_normal((M,)).astype(np.float32), dtype)
        residual = _torch_array(rg.standard_normal((N, M)).astype(np.float32), dtype)
        ref_gate = lambda: ((x.float() @ y.float() + bias.float()) * gate.float() + residual.float()).to(tdtype)
        return PreparedCase(lambda: tk.flux_gate(x, y, bias, gate, residual), ref_gate, ref_gate,
                            atol=0.5 if dtype == "bf16" else 1e-2, rtol=0.5 if dtype == "bf16" else 1e-2,
                            bytes_moved=bytes_moved + M * 4 + N * M * 2, flops=flops)
    raise SkipCase(f"unknown GEMM kernel {kernel}")


def _case_mlx_cmplx(shape: tuple[int, int, int], dtype: str, seed: int) -> PreparedCase:
    mx, _, tk, _ = _load_mlx()
    N, K, M = shape
    rg = _rng(seed, "cmplx" + str(shape) + dtype)
    Ar = rg.standard_normal((N, K)).astype(np.float32)
    Ai = rg.standard_normal((N, K)).astype(np.float32)
    Br = rg.standard_normal((K, M)).astype(np.float32)
    Bi = rg.standard_normal((K, M)).astype(np.float32)
    A = _mlx_array(np.stack([Ar, Ai]), dtype)
    B = _mlx_array(np.stack([Br, Bi]), dtype)
    ref = lambda: (
        mx.stack([
            A[0].astype(mx.float32) @ B[0].astype(mx.float32) - A[1].astype(mx.float32) @ B[1].astype(mx.float32),
            A[0].astype(mx.float32) @ B[1].astype(mx.float32) + A[1].astype(mx.float32) @ B[0].astype(mx.float32),
        ]).astype(_dtype_mlx(mx, dtype))
    )
    return PreparedCase(lambda: tk.cmplx_matmul(A, B), ref, ref, atol=6e-2, rtol=6e-2,
                        bytes_moved=2 * (N * K + K * M + N * M) * (4 if dtype == "f32" else 2),
                        flops=8.0 * N * K * M)


def _case_torch_cmplx(shape: tuple[int, int, int], dtype: str, seed: int) -> PreparedCase:
    torch, tk = _load_torch()
    N, K, M = shape
    rg = _rng(seed, "torch_cmplx" + str(shape) + dtype)
    Ar = rg.standard_normal((N, K)).astype(np.float32)
    Ai = rg.standard_normal((N, K)).astype(np.float32)
    Br = rg.standard_normal((K, M)).astype(np.float32)
    Bi = rg.standard_normal((K, M)).astype(np.float32)
    A = _torch_array(np.stack([Ar, Ai]), dtype)
    B = _torch_array(np.stack([Br, Bi]), dtype)
    tdtype = _dtype_torch(torch, dtype)

    def ref():
        return torch.stack([
            A[0].float() @ B[0].float() - A[1].float() @ B[1].float(),
            A[0].float() @ B[1].float() + A[1].float() @ B[0].float(),
        ]).to(tdtype)

    return PreparedCase(lambda: tk.cmplx_matmul(A, B), ref, ref, atol=6e-2, rtol=6e-2,
                        bytes_moved=2 * (N * K + K * M + N * M) * (4 if dtype == "f32" else 2),
                        flops=8.0 * N * K * M)


def _case_mlx_fftconv(shape: tuple[int, int, int], seed: int) -> PreparedCase:
    mx, _, tk, _ = _load_mlx()
    B, H, S = shape
    N = S * S
    rg = _rng(seed, "fftconv" + str(shape))
    u = rg.standard_normal((B, H, N)).astype(np.float32)
    k = rg.standard_normal((H, N)).astype(np.float32)
    F = _fft_matrix(S)
    Finv = _fft_matrix(S, inverse=True)
    TW = _twiddle(S, S, -1)
    TWI = _twiddle(S, S, +1) / N
    kf = np.fft.fft(k, n=N).reshape(H, S, S).transpose(0, 2, 1)
    xr = u.reshape(B, H, S, S).astype(np.float32)
    X = mx.array(np.stack([xr, np.zeros_like(xr)]))
    KF = mx.array(np.stack([kf.real, kf.imag]).astype(np.float32))
    ref_np = np.fft.ifft(np.fft.fft(u, n=N) * np.fft.fft(k, n=N)[None], n=N).real.reshape(B, H, S, S)
    return PreparedCase(lambda: tk.fftconv(X, _stack_complex_mlx(F), _stack_complex_mlx(TW),
                                           _stack_complex_mlx(Finv), _stack_complex_mlx(TWI), KF),
                        lambda: ref_np.astype(np.float32),
                        None,
                        baseline_name="numpy_fft_not_timed",
                        atol=2e-2,
                        rtol=2e-2,
                        bytes_moved=(B * H * S * S + H * S * S) * 4)


def _case_torch_fftconv(shape: tuple[int, int, int], seed: int) -> PreparedCase:
    torch, tk = _load_torch()
    B, H, S = shape
    N = S * S
    rg = _rng(seed, "torch_fftconv" + str(shape))
    u = rg.standard_normal((B, H, N)).astype(np.float32)
    k = rg.standard_normal((H, N)).astype(np.float32)
    F = _fft_matrix(S)
    Finv = _fft_matrix(S, inverse=True)
    TW = _twiddle(S, S, -1)
    TWI = _twiddle(S, S, +1) / N
    kf = np.fft.fft(k, n=N).reshape(H, S, S).transpose(0, 2, 1)
    xr = u.reshape(B, H, S, S).astype(np.float32)
    X = _torch_array(np.stack([xr, np.zeros_like(xr)]), "f32")
    KF = _torch_array(np.stack([kf.real, kf.imag]).astype(np.float32), "f32")

    def stack(z: np.ndarray):
        return _torch_array(np.stack([z.real, z.imag]).astype(np.float32), "f32")

    ref_np = np.fft.ifft(np.fft.fft(u, n=N) * np.fft.fft(k, n=N)[None], n=N).real.reshape(B, H, S, S)
    return PreparedCase(lambda: tk.fftconv(X, stack(F), stack(TW), stack(Finv), stack(TWI), KF),
                        lambda: ref_np.astype(np.float32),
                        None,
                        baseline_name="numpy_fft_not_timed",
                        atol=2e-2,
                        rtol=2e-2,
                        bytes_moved=(B * H * S * S + H * S * S) * 4)


def _case_mlx_attention(kernel: str, shape: tuple[int, int, int, int], seed: int) -> PreparedCase:
    mx, _, tk, _ = _load_mlx()
    B, H, N, D = shape
    rg = _rng(seed, kernel + str(shape))
    q = mx.array((rg.standard_normal(shape) * 0.5).astype(np.float32)).astype(mx.bfloat16)
    k = mx.array((rg.standard_normal(shape) * 0.5).astype(np.float32)).astype(mx.bfloat16)
    v = mx.array((rg.standard_normal(shape) * 0.5).astype(np.float32)).astype(mx.bfloat16)
    scale = 1.0 / math.sqrt(D)
    flops = 4.0 * B * H * N * N * D
    bytes_moved = 4 * B * H * N * D * 2
    if kernel == "attn_fwd":
        ref = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)
        return PreparedCase(lambda: tk.attn_fwd(q, k, v), ref, ref,
                            atol=4e-2, rtol=4e-2, bytes_moved=bytes_moved, flops=flops)
    if kernel == "attn_multiwarp":
        ref = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)
        return PreparedCase(lambda: tk.attn_multiwarp(q, k, v), ref, ref,
                            atol=4e-2, rtol=4e-2, bytes_moved=bytes_moved, flops=flops)
    if kernel == "attn_causal":
        rows = mx.arange(N)[:, None]
        cols = mx.arange(N)[None, :]
        mask = mx.where(cols > rows, -mx.inf, 0.0).astype(mx.float32)
        ref = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        return PreparedCase(lambda: tk.attn_causal(q, k, v), ref, ref,
                            atol=4e-2, rtol=4e-2, bytes_moved=bytes_moved, flops=flops)
    raise SkipCase(f"unknown attention kernel {kernel}")


def _case_torch_attention(kernel: str, shape: tuple[int, int, int, int], seed: int) -> PreparedCase:
    torch, tk = _load_torch()
    B, H, N, D = shape
    rg = _rng(seed, "torch" + kernel + str(shape))
    q = _torch_array((rg.standard_normal(shape) * 0.5).astype(np.float32), "bf16")
    k = _torch_array((rg.standard_normal(shape) * 0.5).astype(np.float32), "bf16")
    v = _torch_array((rg.standard_normal(shape) * 0.5).astype(np.float32), "bf16")
    scale = 1.0 / math.sqrt(D)
    flops = 4.0 * B * H * N * N * D
    bytes_moved = 4 * B * H * N * D * 2

    def ref(causal: bool = False):
        s = (q.float() @ k.float().transpose(-1, -2)) * scale
        if causal:
            mask = torch.triu(torch.ones(N, N, device="mps", dtype=torch.bool), 1)
            s = s.masked_fill(mask, float("-inf"))
        return (torch.softmax(s, dim=-1) @ v.float()).to(torch.bfloat16)

    if kernel == "attn_fwd":
        return PreparedCase(lambda: tk.attn_fwd(q, k, v), lambda: ref(False), lambda: ref(False),
                            atol=4e-2, rtol=4e-2, bytes_moved=bytes_moved, flops=flops)
    if kernel == "attn_multiwarp":
        return PreparedCase(lambda: tk.attn_multiwarp(q, k, v), lambda: ref(False), lambda: ref(False),
                            atol=4e-2, rtol=4e-2, bytes_moved=bytes_moved, flops=flops)
    if kernel == "attn_causal":
        return PreparedCase(lambda: tk.attn_causal(q, k, v), lambda: ref(True), lambda: ref(True),
                            atol=4e-2, rtol=4e-2, bytes_moved=bytes_moved, flops=flops)
    raise SkipCase(f"unknown attention kernel {kernel}")


def _case_mlx_attn_bwd(shape: tuple[int, int, int, int], causal: bool, seed: int) -> PreparedCase:
    mx, _, tk, _ = _load_mlx()
    B, H, N, D = shape
    rg = _rng(seed, "attn_bwd" + str(shape) + str(causal))
    q_np = (rg.standard_normal(shape) * 0.5).astype(np.float32)
    k_np = (rg.standard_normal(shape) * 0.5).astype(np.float32)
    v_np = (rg.standard_normal(shape) * 0.5).astype(np.float32)
    do_np = (rg.standard_normal(shape) * 0.5).astype(np.float32)
    q = mx.array(q_np).astype(mx.bfloat16)
    k = mx.array(k_np).astype(mx.bfloat16)
    v = mx.array(v_np).astype(mx.bfloat16)
    do = mx.array(do_np).astype(mx.bfloat16)

    def target():
        o, L = tk.attn_fwd_l(q, k, v, causal=causal)
        return tk.attn_bwd(q, k, v, o, do, L, causal=causal)

    def reference():
        try:
            import torch
        except Exception as exc:
            raise SkipCase(f"torch CPU autograd unavailable for attn_bwd oracle: {exc}") from exc
        qt = torch.tensor(q_np, dtype=torch.float32, requires_grad=True)
        kt = torch.tensor(k_np, dtype=torch.float32, requires_grad=True)
        vt = torch.tensor(v_np, dtype=torch.float32, requires_grad=True)
        s = (qt @ kt.transpose(-1, -2)) / math.sqrt(D)
        if causal:
            mask = torch.triu(torch.ones(N, N, dtype=torch.bool), 1)
            s = s.masked_fill(mask, float("-inf"))
        o = torch.softmax(s, dim=-1) @ vt
        o.backward(torch.tensor(do_np, dtype=torch.float32))
        return qt.grad.numpy(), kt.grad.numpy(), vt.grad.numpy()

    return PreparedCase(target, reference, None, baseline_name="torch_cpu_autograd_not_timed",
                        atol=6e-2, rtol=6e-2, bytes_moved=5 * B * H * N * D * 2,
                        flops=8.0 * B * H * N * N * D)


def _case_torch_attn_bwd(shape: tuple[int, int, int, int], causal: bool, seed: int) -> PreparedCase:
    torch, tk = _load_torch()
    B, H, N, D = shape
    rg = _rng(seed, "torch_attn_bwd" + str(shape) + str(causal))
    q_np = (rg.standard_normal(shape) * 0.5).astype(np.float32)
    k_np = (rg.standard_normal(shape) * 0.5).astype(np.float32)
    v_np = (rg.standard_normal(shape) * 0.5).astype(np.float32)
    do_np = (rg.standard_normal(shape) * 0.5).astype(np.float32)
    q = _torch_array(q_np, "bf16")
    k = _torch_array(k_np, "bf16")
    v = _torch_array(v_np, "bf16")
    do = _torch_array(do_np, "bf16")

    def target():
        o, L = tk.attn_fwd_l(q, k, v, causal)
        return tk.attn_bwd(q, k, v, o, do, L, causal)

    def reference():
        qt = torch.tensor(q_np, dtype=torch.float32, requires_grad=True)
        kt = torch.tensor(k_np, dtype=torch.float32, requires_grad=True)
        vt = torch.tensor(v_np, dtype=torch.float32, requires_grad=True)
        s = (qt @ kt.transpose(-1, -2)) / math.sqrt(D)
        if causal:
            mask = torch.triu(torch.ones(N, N, dtype=torch.bool), 1)
            s = s.masked_fill(mask, float("-inf"))
        o = torch.softmax(s, dim=-1) @ vt
        o.backward(torch.tensor(do_np, dtype=torch.float32))
        return qt.grad.numpy(), kt.grad.numpy(), vt.grad.numpy()

    return PreparedCase(target, reference, None, baseline_name="torch_cpu_autograd_not_timed",
                        atol=6e-2, rtol=6e-2, bytes_moved=5 * B * H * N * D * 2,
                        flops=8.0 * B * H * N * N * D)


def _case_mlx_linear_family(kernel: str, shape: tuple[int, ...], seed: int) -> PreparedCase:
    mx, _, tk, _ = _load_mlx()
    rg = _rng(seed, kernel + str(shape))
    if kernel == "based":
        B, H, N = shape
        DQK, DVO = 16, 64
        q = mx.array((rg.standard_normal((B, H, N, DQK)) * 0.5).astype(np.float32)).astype(mx.bfloat16)
        k = mx.array((rg.standard_normal((B, H, N, DQK)) * 0.5).astype(np.float32)).astype(mx.bfloat16)
        v = mx.array((rg.standard_normal((B, H, N, DVO)) * 0.5).astype(np.float32)).astype(mx.bfloat16)
        mask = (mx.arange(N)[None, :] <= mx.arange(N)[:, None]).astype(mx.float32)
        ref = lambda: ((1.0 + 0.25 * (q.astype(mx.float32) @ mx.swapaxes(k.astype(mx.float32), -1, -2))
                        + 0.5 * (0.25 * (q.astype(mx.float32) @ mx.swapaxes(k.astype(mx.float32), -1, -2))) ** 2)
                       * mask).astype(mx.float32) @ v.astype(mx.float32)
        return PreparedCase(lambda: tk.based(q, k, v), ref, ref, atol=4e-2, rtol=4e-2,
                            bytes_moved=(B * H * N * (2 * DQK + DVO + DVO)) * 2,
                            flops=4.0 * B * H * N * N * DVO)

    B, H, N, D = shape
    q = mx.array((rg.standard_normal(shape) * 0.5).astype(np.float32)).astype(mx.bfloat16)
    k = mx.array((rg.standard_normal(shape) * 0.5).astype(np.float32)).astype(mx.bfloat16)
    v = mx.array((rg.standard_normal(shape) * 0.5).astype(np.float32)).astype(mx.bfloat16)
    bytes_moved = 4 * B * H * N * D * 2
    flops = 4.0 * B * H * N * N * D
    if kernel == "linear_attn":
        ref = lambda: (q.astype(mx.float32) @ (mx.swapaxes(k.astype(mx.float32), -1, -2) @ v.astype(mx.float32)))
        return PreparedCase(lambda: tk.linear_attn(q, k, v), ref, ref, atol=3e-2, rtol=3e-2,
                            bytes_moved=bytes_moved, flops=flops)
    if kernel == "lin_attn_causal":
        mask = (mx.arange(N)[None, :] <= mx.arange(N)[:, None]).astype(mx.float32)
        ref = lambda: ((q.astype(mx.float32) @ mx.swapaxes(k.astype(mx.float32), -1, -2)) * mask) @ v.astype(mx.float32)
        return PreparedCase(lambda: tk.lin_attn_causal(q, k, v), ref, ref, atol=3e-2, rtol=3e-2,
                            bytes_moved=bytes_moved, flops=flops)
    if kernel == "hedgehog":
        phi = lambda x: mx.exp(x.astype(mx.float32) - mx.max(x.astype(mx.float32), axis=-1, keepdims=True))
        ref = lambda: phi(q) @ (mx.swapaxes(phi(k), -1, -2) @ v.astype(mx.float32))
        return PreparedCase(lambda: tk.hedgehog(q, k, v), ref, ref, atol=3e-2, rtol=3e-2,
                            bytes_moved=bytes_moved, flops=flops)
    if kernel == "lin_attn_decay":
        slopes = np.linspace(0.05, 0.5, H).astype(np.float32)
        pos = mx.arange(N).astype(mx.float32)
        dist = pos[:, None] - pos[None, :]
        causal = (dist >= 0).astype(mx.float32)
        lam_np = np.stack([np.where(np.arange(N)[:, None] - np.arange(N)[None, :] >= 0,
                                    np.exp(-s * (np.arange(N)[:, None] - np.arange(N)[None, :])), 0.0)
                           for s in slopes]).astype(np.float32)
        lam = mx.array(np.broadcast_to(lam_np[None], (B, H, N, N)))
        ref = lambda: ((q.astype(mx.float32) @ mx.swapaxes(k.astype(mx.float32), -1, -2)) * lam
                       * causal[None, None]) @ v.astype(mx.float32)
        return PreparedCase(lambda: tk.lin_attn_decay(q, k, v, slopes), ref, ref, atol=4e-2, rtol=4e-2,
                            bytes_moved=bytes_moved + B * H * N * 4, flops=flops)
    if kernel == "mamba2":
        C = q
        Bm = k
        X = v
        a = mx.sigmoid(mx.array(rg.standard_normal((B, H, N)).astype(np.float32))) * 0.5 + 0.5
        cumlog = mx.cumsum(mx.log(a), axis=-1).astype(mx.float32)
        mask = (mx.arange(N)[None, :] <= mx.arange(N)[:, None]).astype(mx.float32)
        ref = lambda: ((C.astype(mx.float32) @ mx.swapaxes(Bm.astype(mx.float32), -1, -2))
                       * mx.exp(cumlog[..., :, None] - cumlog[..., None, :]) * mask) @ X.astype(mx.float32)
        return PreparedCase(lambda: tk.mamba2(C, Bm, X, cumlog), ref, ref, atol=3e-2, rtol=3e-2,
                            bytes_moved=bytes_moved + B * H * N * 4, flops=flops)
    raise SkipCase(f"unknown linear-family kernel {kernel}")


def _case_torch_linear_family(kernel: str, shape: tuple[int, ...], seed: int) -> PreparedCase:
    torch, tk = _load_torch()
    rg = _rng(seed, "torch_" + kernel + str(shape))
    if kernel == "based":
        B, H, N = shape
        DQK, DVO = 16, 64
        q = _torch_array((rg.standard_normal((B, H, N, DQK)) * 0.5).astype(np.float32), "bf16")
        k = _torch_array((rg.standard_normal((B, H, N, DQK)) * 0.5).astype(np.float32), "bf16")
        v = _torch_array((rg.standard_normal((B, H, N, DVO)) * 0.5).astype(np.float32), "bf16")
        mask = torch.tril(torch.ones(N, N, device="mps", dtype=torch.float32))

        def ref():
            x = 0.25 * (q.float() @ k.float().transpose(-1, -2))
            return ((1.0 + x + 0.5 * x * x) * mask) @ v.float()

        return PreparedCase(lambda: tk.based(q, k, v), ref, ref, atol=4e-2, rtol=4e-2,
                            bytes_moved=(B * H * N * (2 * DQK + DVO + DVO)) * 2,
                            flops=4.0 * B * H * N * N * DVO)

    B, H, N, D = shape
    q = _torch_array((rg.standard_normal(shape) * 0.5).astype(np.float32), "bf16")
    k = _torch_array((rg.standard_normal(shape) * 0.5).astype(np.float32), "bf16")
    v = _torch_array((rg.standard_normal(shape) * 0.5).astype(np.float32), "bf16")
    bytes_moved = 4 * B * H * N * D * 2
    flops = 4.0 * B * H * N * N * D
    if kernel == "linear_attn":
        ref = lambda: q.float() @ (k.float().transpose(-1, -2) @ v.float())
        return PreparedCase(lambda: tk.linear_attn(q, k, v), ref, ref, atol=3e-2, rtol=3e-2,
                            bytes_moved=bytes_moved, flops=flops)
    if kernel == "lin_attn_causal":
        mask = torch.tril(torch.ones(N, N, device="mps", dtype=torch.float32))
        ref = lambda: ((q.float() @ k.float().transpose(-1, -2)) * mask) @ v.float()
        return PreparedCase(lambda: tk.lin_attn_causal(q, k, v), ref, ref, atol=3e-2, rtol=3e-2,
                            bytes_moved=bytes_moved, flops=flops)
    if kernel == "hedgehog":
        phi = lambda x: torch.exp(x.float() - x.float().max(dim=-1, keepdim=True).values)
        ref = lambda: phi(q) @ (phi(k).transpose(-1, -2) @ v.float())
        return PreparedCase(lambda: tk.hedgehog(q, k, v), ref, ref, atol=3e-2, rtol=3e-2,
                            bytes_moved=bytes_moved, flops=flops)
    if kernel == "lin_attn_decay":
        slopes = np.linspace(0.05, 0.5, H).astype(np.float32)
        dist_np = np.arange(N)[:, None] - np.arange(N)[None, :]
        lam_np = np.stack([np.where(dist_np >= 0, np.exp(-s * dist_np), 0.0) for s in slopes]).astype(np.float32)
        lam = torch.tensor(np.broadcast_to(lam_np[None], (B, H, N, N)), device="mps", dtype=torch.float32)
        ref = lambda: ((q.float() @ k.float().transpose(-1, -2)) * lam) @ v.float()
        return PreparedCase(lambda: tk.lin_attn_decay(q, k, v, slopes), ref, ref, atol=4e-2, rtol=4e-2,
                            bytes_moved=bytes_moved + B * H * N * 4, flops=flops)
    if kernel == "mamba2":
        a = torch.sigmoid(torch.tensor(rg.standard_normal((B, H, N)).astype(np.float32), device="mps")) * 0.5 + 0.5
        cumlog = torch.cumsum(torch.log(a), dim=-1)
        mask = torch.tril(torch.ones(N, N, device="mps", dtype=torch.float32))
        ref = lambda: ((q.float() @ k.float().transpose(-1, -2))
                       * torch.exp(cumlog[..., :, None] - cumlog[..., None, :]) * mask) @ v.float()
        return PreparedCase(lambda: tk.mamba2(q, k, v, cumlog), ref, ref, atol=3e-2, rtol=3e-2,
                            bytes_moved=bytes_moved + B * H * N * 4, flops=flops)
    raise SkipCase(f"unknown linear-family kernel {kernel}")


def _case_mlx_quant(kernel: str, shape: tuple[int, ...], fmt: str, seed: int) -> PreparedCase:
    mx, nn, tk, quant = _load_mlx()
    rg = _rng(seed, kernel + str(shape) + fmt)
    if kernel in {"qgemv", "qgemm", "qgemm_direct", "qflux_gelu"}:
        quantize, dequantize = quant.QUANT_FORMATS[fmt]
        if kernel == "qgemv":
            N, K = shape
            M = 1
        else:
            N, K, M = shape
            if M % 32 != 0:
                raise SkipCase("qgemm/qflux kernels require M % 32 == 0; M==1 routes through qgemv")
        W = (rg.standard_normal((N, K)) * 0.3).astype(np.float32)
        X = rg.standard_normal((K, M)).astype(np.float32)
        Wq = quantize(W)
        Wq_mx = mx.array(Wq)
        X_mx = mx.array(X).astype(mx.float16)
        dW = mx.array(dequantize(Wq).astype(np.float32))
        base = lambda: dW @ X_mx.astype(mx.float32)
        flops = 2.0 * N * K * M
        bytes_moved = int(Wq.nbytes + X.nbytes + N * M * 2)
        if kernel == "qgemv":
            return PreparedCase(lambda: tk.qgemv(Wq_mx, X_mx, format=fmt), base, base,
                                atol=2e-2, rtol=2e-2, bytes_moved=Wq.nbytes,
                                weight_bytes=Wq.nbytes, flops=flops)
        if kernel == "qgemm":
            return PreparedCase(lambda: tk.qgemm(Wq_mx, X_mx, format=fmt), base, base,
                                atol=2e-2, rtol=2e-2, bytes_moved=bytes_moved,
                                weight_bytes=Wq.nbytes, flops=flops)
        if kernel == "qgemm_direct":
            return PreparedCase(lambda: tk.qgemm_direct(Wq_mx, X_mx, format=fmt), base, base,
                                atol=2e-2, rtol=2e-2, bytes_moved=bytes_moved,
                                weight_bytes=Wq.nbytes, flops=flops)
        bias = rg.standard_normal((M,)).astype(np.float32)
        bias_mx = mx.array(bias).astype(mx.float16)
        ref = lambda: nn.gelu_approx(base() + bias_mx.astype(mx.float32))
        return PreparedCase(lambda: tk.qflux_gelu(Wq_mx, X_mx, bias_mx, format=fmt), ref, ref,
                            atol=3e-2, rtol=3e-2, bytes_moved=bytes_moved + M * 2,
                            weight_bytes=Wq.nbytes, flops=flops)

    raise SkipCase(f"unknown quant kernel {kernel}")


def _case_torch_quant(kernel: str, shape: tuple[int, ...], fmt: str, seed: int) -> PreparedCase:
    torch, tk = _load_torch()
    _, _, _, quant = _load_mlx()
    rg = _rng(seed, "torch_" + kernel + str(shape) + fmt)
    if kernel in {"qgemv", "qgemm", "qgemm_direct", "qflux_gelu"}:
        if kernel == "qgemm_direct":
            raise SkipCase("qgemm_direct has no PyTorch wrapper")
        quantize, dequantize = quant.QUANT_FORMATS[fmt]
        if kernel == "qgemv":
            N, K = shape
            M = 1
        else:
            N, K, M = shape
            if M % 32 != 0:
                raise SkipCase("qgemm/qflux kernels require M % 32 == 0; M==1 routes through qgemv")
        W = (rg.standard_normal((N, K)) * 0.3).astype(np.float32)
        X = rg.standard_normal((K, M)).astype(np.float32)
        Wq = quantize(W)
        Wq_t = _torch_int_array(Wq)
        X_t = _torch_array(X, "f16")
        dW = torch.tensor(dequantize(Wq).astype(np.float32), device="mps", dtype=torch.float32)
        base = lambda: dW @ X_t.float()
        flops = 2.0 * N * K * M
        bytes_moved = int(Wq.nbytes + X.nbytes + N * M * 2)
        if kernel == "qgemv":
            return PreparedCase(lambda: tk.qgemv(Wq_t, X_t, fmt), base, base,
                                atol=2e-2, rtol=2e-2, bytes_moved=Wq.nbytes,
                                weight_bytes=Wq.nbytes, flops=flops)
        if kernel == "qgemm":
            return PreparedCase(lambda: tk.qgemm(Wq_t, X_t, fmt), base, base,
                                atol=2e-2, rtol=2e-2, bytes_moved=bytes_moved,
                                weight_bytes=Wq.nbytes, flops=flops)
        bias = _torch_array(rg.standard_normal((M,)).astype(np.float32), "f16")
        ref = lambda: torch.nn.functional.gelu(base() + bias.float(), approximate="tanh")
        return PreparedCase(lambda: tk.qflux_gelu(Wq_t, X_t, bias, fmt), ref, ref,
                            atol=3e-2, rtol=3e-2, bytes_moved=bytes_moved + M * 2,
                            weight_bytes=Wq.nbytes, flops=flops)
    raise SkipCase(f"unknown quant kernel {kernel}")


def _case_mlx_quant_special(kernel: str, shape: tuple[int, ...], seed: int) -> PreparedCase:
    mx, _, tk, quant = _load_mlx()
    rg = _rng(seed, kernel + str(shape))
    if kernel in {"qgemv_w8a8", "qgemv_w2a8"}:
        N, K = shape
        W = (rg.standard_normal((N, K)) * 0.3).astype(np.float32)
        X = rg.standard_normal((K, 1)).astype(np.float32)
        _, Xq, xs = quant.quantize_act_int8(X)
        a_scale = float(xs[0, 0])
        if kernel == "qgemv_w8a8":
            Wq, ws = quant.quantize_w8a8(W)
            ref_np = (Wq.astype(np.int32) @ Xq.astype(np.int32)).astype(np.float32) * ws[:, None] * a_scale
            return PreparedCase(lambda: tk.qgemv_w8a8(mx.array(Wq), mx.array(Xq),
                                                      mx.array(ws).astype(mx.float16),
                                                      mx.array(np.array([a_scale], np.float16))),
                                lambda: ref_np, None, baseline_name="numpy_int_oracle_not_timed",
                                atol=2e-2, rtol=2e-2, bytes_moved=Wq.nbytes + Xq.nbytes,
                                weight_bytes=Wq.nbytes, flops=2.0 * N * K)
        Wq = quant.quantize_bitnet(W)
        with np.errstate(all="ignore"):
            ref_np = (quant.dequantize_bitnet(Wq).astype(np.float32) @ Xq.astype(np.float32)) * a_scale
        return PreparedCase(lambda: tk.qgemv_w2a8(mx.array(Wq), mx.array(Xq),
                                                  mx.array(np.array([a_scale], np.float16))),
                            lambda: ref_np, None, baseline_name="numpy_int_oracle_not_timed",
                            atol=2e-2, rtol=2e-2, bytes_moved=Wq.nbytes + Xq.nbytes,
                            weight_bytes=Wq.nbytes, flops=2.0 * N * K)

    if kernel in {"qgemm_w8a8", "qgemm_w2a8"}:
        N, K, M = shape
        W = (rg.standard_normal((N, K)) * 0.3).astype(np.float32)
        X = rg.standard_normal((K, M)).astype(np.float32)
        _, Xq, xs = quant.quantize_act_int8(X)
        Xqt = np.ascontiguousarray(Xq.T)
        asc = xs[0, :].astype(np.float16)
        if kernel == "qgemm_w8a8":
            Wq, ws = quant.quantize_w8a8(W)
            ref_np = (Wq.astype(np.int32) @ Xq.astype(np.int32)).astype(np.float32) * ws[:, None] * xs[0, None, :]
            return PreparedCase(lambda: tk.qgemm_w8a8(mx.array(Wq), mx.array(Xqt),
                                                      mx.array(ws).astype(mx.float16), mx.array(asc)),
                                lambda: ref_np, None, baseline_name="numpy_int_oracle_not_timed",
                                atol=2e-2, rtol=2e-2, bytes_moved=Wq.nbytes + Xq.nbytes + N * M * 2,
                                weight_bytes=Wq.nbytes, flops=2.0 * N * K * M)
        Wq = quant.quantize_bitnet(W)
        with np.errstate(all="ignore"):
            ref_np = (quant.dequantize_bitnet(Wq).astype(np.float32) @ Xq.astype(np.float32)) * xs[0, None, :]
        return PreparedCase(lambda: tk.qgemm_w2a8(mx.array(Wq), mx.array(Xqt), mx.array(asc)),
                            lambda: ref_np, None, baseline_name="numpy_int_oracle_not_timed",
                            atol=2e-2, rtol=2e-2, bytes_moved=Wq.nbytes + Xq.nbytes + N * M * 2,
                            weight_bytes=Wq.nbytes, flops=2.0 * N * K * M)

    if kernel == "qgemm_fp8_block2d":
        N, K, M = shape
        W = (rg.standard_normal((N, K)) * 0.3).astype(np.float32)
        X = rg.standard_normal((K, M)).astype(np.float32)
        codes, scale2d = quant.quantize_fp8_block2d(W)
        X_mx = mx.array(X).astype(mx.float16)
        dW = mx.array(quant.dequantize_fp8_block2d(codes, scale2d).astype(np.float32))
        base = lambda: dW @ X_mx.astype(mx.float32)
        return PreparedCase(lambda: tk.qgemm_fp8_block2d(mx.array(codes), X_mx, mx.array(scale2d)),
                            base, base, atol=2e-2, rtol=2e-2,
                            bytes_moved=codes.nbytes + scale2d.nbytes + X.nbytes + N * M * 2,
                            weight_bytes=codes.nbytes + scale2d.nbytes, flops=2.0 * N * K * M)

    if kernel == "qgemm_fp8_scaled":
        N, K, M = shape
        W = (rg.standard_normal((N, K)) * 0.3).astype(np.float32)
        X = rg.standard_normal((K, M)).astype(np.float32)
        wq, w_scale = quant.quantize_fp8_scaled(W)
        _, xq, s = quant.quantize_act_fp8(X)
        a_scale = s[0, :].astype(np.float16)
        dW = quant._e4m3_decode_arr(wq).astype(np.float32)
        dX = quant._e4m3_decode_arr(xq).astype(np.float32)
        with np.errstate(all="ignore"):
            ref_np = w_scale.astype(np.float32)[:, None] * a_scale.astype(np.float32)[None, :] * (dW @ dX)
        return PreparedCase(lambda: tk.qgemm_fp8_scaled(mx.array(wq), mx.array(xq),
                                                        mx.array(w_scale), mx.array(a_scale)),
                            lambda: ref_np, None, baseline_name="numpy_fp8_oracle_not_timed",
                            atol=2e-2, rtol=2e-2,
                            bytes_moved=wq.nbytes + xq.nbytes + w_scale.nbytes + a_scale.nbytes + N * M * 2,
                            weight_bytes=wq.nbytes + w_scale.nbytes, flops=2.0 * N * K * M)

    if kernel in {"qgemm_actorder", "qgemm_actorder_fused"}:
        N, K, M = shape
        W = (rg.standard_normal((N, K)) * 0.3).astype(np.float32)
        X = rg.standard_normal((K, M)).astype(np.float32)
        perm = rg.permutation(K).astype(np.int32)
        Wq = quant.quantize_kU4B8(W[:, perm])
        X_mx = mx.array(X).astype(mx.float16)
        perm_mx = mx.array(perm)
        ref = lambda: mx.array(quant.dequantize_kU4B8(Wq).astype(np.float32)) @ mx.take(
            X_mx, perm_mx, axis=0).astype(mx.float32)
        fused = kernel == "qgemm_actorder_fused"
        return PreparedCase(lambda: tk.qgemm_actorder(mx.array(Wq), X_mx, perm, w_format="kU4B8", fused=fused),
                            ref, ref, atol=2e-2, rtol=2e-2,
                            bytes_moved=Wq.nbytes + X.nbytes + N * M * 2,
                            weight_bytes=Wq.nbytes, flops=2.0 * N * K * M)

    raise SkipCase(f"unknown quant-special kernel {kernel}")


def _case_torch_quant_special(kernel: str, shape: tuple[int, ...], seed: int) -> PreparedCase:
    torch, tk = _load_torch()
    _, _, _, quant = _load_mlx()
    rg = _rng(seed, "torch_" + kernel + str(shape))
    if kernel in {"qgemv_w8a8", "qgemv_w2a8"}:
        N, K = shape
        W = (rg.standard_normal((N, K)) * 0.3).astype(np.float32)
        X = rg.standard_normal((K, 1)).astype(np.float32)
        _, Xq, xs = quant.quantize_act_int8(X)
        a_scale = float(xs[0, 0])
        if kernel == "qgemv_w8a8":
            Wq, ws = quant.quantize_w8a8(W)
            ref_np = (Wq.astype(np.int32) @ Xq.astype(np.int32)).astype(np.float32) * ws[:, None] * a_scale
            return PreparedCase(lambda: tk.qgemv_w8a8(_torch_int_array(Wq), _torch_int_array(Xq),
                                                      _torch_array(ws, "f16"),
                                                      _torch_array(np.array([a_scale], np.float16), "f16")),
                                lambda: ref_np, None, baseline_name="numpy_int_oracle_not_timed",
                                atol=2e-2, rtol=2e-2, bytes_moved=Wq.nbytes + Xq.nbytes,
                                weight_bytes=Wq.nbytes, flops=2.0 * N * K)
        Wq = quant.quantize_bitnet(W)
        with np.errstate(all="ignore"):
            ref_np = (quant.dequantize_bitnet(Wq).astype(np.float32) @ Xq.astype(np.float32)) * a_scale
        return PreparedCase(lambda: tk.qgemv_w2a8(_torch_int_array(Wq), _torch_int_array(Xq),
                                                  _torch_array(np.array([a_scale], np.float16), "f16")),
                            lambda: ref_np, None, baseline_name="numpy_int_oracle_not_timed",
                            atol=2e-2, rtol=2e-2, bytes_moved=Wq.nbytes + Xq.nbytes,
                            weight_bytes=Wq.nbytes, flops=2.0 * N * K)

    if kernel in {"qgemm_w8a8", "qgemm_w2a8"}:
        N, K, M = shape
        W = (rg.standard_normal((N, K)) * 0.3).astype(np.float32)
        X = rg.standard_normal((K, M)).astype(np.float32)
        _, Xq, xs = quant.quantize_act_int8(X)
        Xqt = np.ascontiguousarray(Xq.T)
        asc = xs[0, :].astype(np.float16)
        if kernel == "qgemm_w8a8":
            Wq, ws = quant.quantize_w8a8(W)
            ref_np = (Wq.astype(np.int32) @ Xq.astype(np.int32)).astype(np.float32) * ws[:, None] * xs[0, None, :]
            return PreparedCase(lambda: tk.qgemm_w8a8(_torch_int_array(Wq), _torch_int_array(Xqt),
                                                      _torch_array(ws, "f16"), _torch_array(asc, "f16")),
                                lambda: ref_np, None, baseline_name="numpy_int_oracle_not_timed",
                                atol=2e-2, rtol=2e-2, bytes_moved=Wq.nbytes + Xq.nbytes + N * M * 2,
                                weight_bytes=Wq.nbytes, flops=2.0 * N * K * M)
        Wq = quant.quantize_bitnet(W)
        with np.errstate(all="ignore"):
            ref_np = (quant.dequantize_bitnet(Wq).astype(np.float32) @ Xq.astype(np.float32)) * xs[0, None, :]
        return PreparedCase(lambda: tk.qgemm_w2a8(_torch_int_array(Wq), _torch_int_array(Xqt), _torch_array(asc, "f16")),
                            lambda: ref_np, None, baseline_name="numpy_int_oracle_not_timed",
                            atol=2e-2, rtol=2e-2, bytes_moved=Wq.nbytes + Xq.nbytes + N * M * 2,
                            weight_bytes=Wq.nbytes, flops=2.0 * N * K * M)

    if kernel == "qgemm_fp8_block2d":
        N, K, M = shape
        W = (rg.standard_normal((N, K)) * 0.3).astype(np.float32)
        X = rg.standard_normal((K, M)).astype(np.float32)
        codes, scale2d = quant.quantize_fp8_block2d(W)
        X_t = _torch_array(X, "f16")
        dW = torch.tensor(quant.dequantize_fp8_block2d(codes, scale2d).astype(np.float32),
                          device="mps", dtype=torch.float32)
        base = lambda: dW @ X_t.float()
        return PreparedCase(lambda: tk.qgemm_fp8_block2d(_torch_int_array(codes), X_t, _torch_array(scale2d, "f16")),
                            base, base, atol=2e-2, rtol=2e-2,
                            bytes_moved=codes.nbytes + scale2d.nbytes + X.nbytes + N * M * 2,
                            weight_bytes=codes.nbytes + scale2d.nbytes, flops=2.0 * N * K * M)

    if kernel == "qgemm_fp8_scaled":
        N, K, M = shape
        W = (rg.standard_normal((N, K)) * 0.3).astype(np.float32)
        X = rg.standard_normal((K, M)).astype(np.float32)
        wq, w_scale = quant.quantize_fp8_scaled(W)
        _, xq, s = quant.quantize_act_fp8(X)
        a_scale = s[0, :].astype(np.float16)
        dW = quant._e4m3_decode_arr(wq).astype(np.float32)
        dX = quant._e4m3_decode_arr(xq).astype(np.float32)
        with np.errstate(all="ignore"):
            ref_np = w_scale.astype(np.float32)[:, None] * a_scale.astype(np.float32)[None, :] * (dW @ dX)
        return PreparedCase(lambda: tk.qgemm_fp8_scaled(_torch_int_array(wq), _torch_int_array(xq),
                                                        _torch_array(w_scale, "f16"), _torch_array(a_scale, "f16")),
                            lambda: ref_np, None, baseline_name="numpy_fp8_oracle_not_timed",
                            atol=2e-2, rtol=2e-2,
                            bytes_moved=wq.nbytes + xq.nbytes + w_scale.nbytes + a_scale.nbytes + N * M * 2,
                            weight_bytes=wq.nbytes + w_scale.nbytes, flops=2.0 * N * K * M)

    if kernel in {"qgemm_actorder", "qgemm_actorder_fused"}:
        N, K, M = shape
        W = (rg.standard_normal((N, K)) * 0.3).astype(np.float32)
        X = rg.standard_normal((K, M)).astype(np.float32)
        perm = rg.permutation(K).astype(np.int32)
        Wq = quant.quantize_kU4B8(W[:, perm])
        X_t = _torch_array(X, "f16")
        dW = torch.tensor(quant.dequantize_kU4B8(Wq).astype(np.float32), device="mps", dtype=torch.float32)
        perm_t = torch.tensor(perm, device="mps", dtype=torch.long)
        ref = lambda: dW @ X_t.index_select(0, perm_t).float()
        fused = kernel == "qgemm_actorder_fused"
        return PreparedCase(lambda: tk.qgemm_actorder(_torch_int_array(Wq), X_t, perm, w_format="kU4B8", fused=fused),
                            ref, ref, atol=2e-2, rtol=2e-2,
                            bytes_moved=Wq.nbytes + X.nbytes + N * M * 2,
                            weight_bytes=Wq.nbytes, flops=2.0 * N * K * M)

    raise SkipCase(f"unknown quant-special kernel {kernel}")


def _case_mlx_attn_q(shape: tuple[int, int, int, int], fmt: str, causal: bool, multiwarp: bool, seed: int) -> PreparedCase:
    mx, _, tk, quant = _load_mlx()
    B, H, N, D = shape
    rg = _rng(seed, "attn_q" + str(shape) + fmt + str(causal) + str(multiwarp))
    q_np = (rg.standard_normal(shape) * 0.5).astype(np.float32)
    k_np = (rg.standard_normal(shape) * 0.5).astype(np.float32)
    v_np = (rg.standard_normal(shape) * 0.5).astype(np.float32)
    Kq = quant.quantize_kv(k_np, fmt)
    Vq = quant.quantize_kv(v_np, fmt)
    dk = quant.dequantize_kv(Kq, fmt).astype(np.float32)
    dv = quant.dequantize_kv(Vq, fmt).astype(np.float32)
    q = mx.array(q_np).astype(mx.bfloat16)
    dk_mx = mx.array(dk).astype(mx.bfloat16)
    dv_mx = mx.array(dv).astype(mx.bfloat16)
    if causal:
        rows = mx.arange(N)[:, None]
        cols = mx.arange(N)[None, :]
        mask = mx.where(cols > rows, -mx.inf, 0.0).astype(mx.float32)
    else:
        mask = None
    base = lambda: mx.fast.scaled_dot_product_attention(q, dk_mx, dv_mx, scale=1.0 / math.sqrt(D), mask=mask)
    return PreparedCase(lambda: tk.attn_q(q, mx.array(Kq), mx.array(Vq), format=fmt,
                                          causal=causal, multiwarp=multiwarp),
                        base, base, atol=0.1, rtol=0.1,
                        bytes_moved=q_np.nbytes + Kq.nbytes + Vq.nbytes + q_np.nbytes,
                        weight_bytes=Kq.nbytes + Vq.nbytes,
                        flops=4.0 * B * H * N * N * D)


def _case_torch_attn_q(shape: tuple[int, int, int, int], fmt: str, causal: bool, multiwarp: bool, seed: int) -> PreparedCase:
    torch, tk = _load_torch()
    _, _, _, quant = _load_mlx()
    B, H, N, D = shape
    rg = _rng(seed, "torch_attn_q" + str(shape) + fmt + str(causal) + str(multiwarp))
    q_np = (rg.standard_normal(shape) * 0.5).astype(np.float32)
    k_np = (rg.standard_normal(shape) * 0.5).astype(np.float32)
    v_np = (rg.standard_normal(shape) * 0.5).astype(np.float32)
    Kq = quant.quantize_kv(k_np, fmt)
    Vq = quant.quantize_kv(v_np, fmt)
    dk = quant.dequantize_kv(Kq, fmt).astype(np.float32)
    dv = quant.dequantize_kv(Vq, fmt).astype(np.float32)
    q = _torch_array(q_np, "bf16")
    dk_t = _torch_array(dk, "bf16")
    dv_t = _torch_array(dv, "bf16")
    if causal:
        mask = torch.triu(torch.ones(N, N, device="mps", dtype=torch.bool), 1)
    else:
        mask = None

    def base():
        s = (q.float() @ dk_t.float().transpose(-1, -2)) / math.sqrt(D)
        if mask is not None:
            s = s.masked_fill(mask, float("-inf"))
        return (torch.softmax(s, dim=-1) @ dv_t.float()).to(torch.bfloat16)

    return PreparedCase(lambda: tk.attn_q(q, _torch_int_array(Kq), _torch_int_array(Vq), fmt, causal, multiwarp),
                        base, base, atol=0.1, rtol=0.1,
                        bytes_moved=q_np.nbytes + Kq.nbytes + Vq.nbytes + q_np.nbytes,
                        weight_bytes=Kq.nbytes + Vq.nbytes,
                        flops=4.0 * B * H * N * N * D)


def _factory_for(spec: CaseSpec, backend: str, seed: int) -> PreparedCase:
    if spec.factory is not None:
        return spec.factory(backend)
    if backend == "mlx":
        if spec.kernel in {"add_rt", "gelu"}:
            return _case_mlx_pointwise(spec.kernel, spec.shape, spec.dtype, seed)
        if spec.kernel in {"layernorm", "rms_norm", "softmax"}:
            return _case_mlx_row(spec.kernel, spec.shape, seed)
        if spec.kernel == "rotary":
            return _case_mlx_rotary(spec.shape, seed)
        if spec.kernel in {"matmul_custom", "gemm_staged", "flux_gelu", "flux_gate"}:
            return _case_mlx_gemm(spec.kernel, spec.shape, spec.dtype, seed)
        if spec.kernel == "cmplx_matmul":
            return _case_mlx_cmplx(spec.shape, spec.dtype, seed)
        if spec.kernel == "fftconv":
            return _case_mlx_fftconv(spec.shape, seed)
        if spec.kernel in {"attn_fwd", "attn_causal", "attn_multiwarp"}:
            return _case_mlx_attention(spec.kernel, spec.shape, seed)
        if spec.kernel == "attn_bwd":
            return _case_mlx_attn_bwd(spec.shape, causal=("causal" in spec.variant), seed=seed)
        if spec.kernel == "attn_q":
            return _case_mlx_attn_q(spec.shape, spec.fmt or "q8_0",
                                    causal=("causal" in spec.variant),
                                    multiwarp=("multiwarp" in spec.variant), seed=seed)
        if spec.kernel in {"linear_attn", "lin_attn_causal", "lin_attn_decay", "hedgehog", "based", "mamba2"}:
            return _case_mlx_linear_family(spec.kernel, spec.shape, seed)
        if spec.kernel in {"qgemv", "qgemm", "qgemm_direct", "qflux_gelu"}:
            return _case_mlx_quant(spec.kernel, spec.shape, spec.fmt or "q8_0", seed)
        if spec.kernel in {"qgemv_w8a8", "qgemv_w2a8", "qgemm_w8a8", "qgemm_w2a8",
                           "qgemm_fp8_block2d", "qgemm_fp8_scaled", "qgemm_actorder",
                           "qgemm_actorder_fused"}:
            return _case_mlx_quant_special(spec.kernel, spec.shape, seed)
    if backend == "torch":
        if spec.kernel in {"add_rt", "gelu"}:
            return _case_torch_pointwise(spec.kernel, spec.shape, spec.dtype, seed)
        if spec.kernel in {"layernorm", "rms_norm", "softmax"}:
            return _case_torch_row(spec.kernel, spec.shape, seed)
        if spec.kernel == "rotary":
            return _case_torch_rotary(spec.shape, seed)
        if spec.kernel in {"matmul_custom", "gemm_staged", "flux_gelu", "flux_gate"}:
            return _case_torch_gemm(spec.kernel, spec.shape, spec.dtype, seed)
        if spec.kernel in {"attn_fwd", "attn_causal", "attn_multiwarp"}:
            return _case_torch_attention(spec.kernel, spec.shape, seed)
        if spec.kernel == "cmplx_matmul":
            return _case_torch_cmplx(spec.shape, spec.dtype, seed)
        if spec.kernel == "fftconv":
            return _case_torch_fftconv(spec.shape, seed)
        if spec.kernel == "attn_bwd":
            return _case_torch_attn_bwd(spec.shape, causal=("causal" in spec.variant), seed=seed)
        if spec.kernel == "attn_q":
            return _case_torch_attn_q(spec.shape, spec.fmt or "q8_0",
                                      causal=("causal" in spec.variant),
                                      multiwarp=("multiwarp" in spec.variant), seed=seed)
        if spec.kernel in {"linear_attn", "lin_attn_causal", "lin_attn_decay", "hedgehog", "based", "mamba2"}:
            return _case_torch_linear_family(spec.kernel, spec.shape, seed)
        if spec.kernel in {"qgemv", "qgemm", "qgemm_direct", "qflux_gelu"}:
            return _case_torch_quant(spec.kernel, spec.shape, spec.fmt or "q8_0", seed)
        if spec.kernel in {"qgemv_w8a8", "qgemv_w2a8", "qgemm_w8a8", "qgemm_w2a8",
                           "qgemm_fp8_block2d", "qgemm_fp8_scaled", "qgemm_actorder",
                           "qgemm_actorder_fused"}:
            return _case_torch_quant_special(spec.kernel, spec.shape, seed)
        raise SkipCase(f"{spec.kernel} torch benchmark not implemented in this phase")
    raise SkipCase(f"unsupported backend {backend}")


def _formats_for(preset: str, arg: str | None) -> list[str]:
    _, _, _, quant = _load_mlx()
    if arg:
        requested = [x.strip() for x in arg.split(",") if x.strip()]
    elif preset == "comprehensive":
        requested = sorted(quant.QUANT_FORMATS)
    elif preset == "quick":
        requested = COMMON_QUANT_FORMATS
    else:
        requested = ["q8_0"]
    unknown = sorted(set(requested) - set(quant.QUANT_FORMATS))
    if unknown:
        raise SystemExit(f"unknown quant format(s): {', '.join(unknown)}")
    return requested


def _case_specs(preset: str, formats: list[str]) -> list[CaseSpec]:
    smoke = preset == "smoke"
    quick = preset == "quick"
    specs: list[CaseSpec] = []

    add_shapes = [(8, 8)] if smoke else [(8, 8), (64, 128)] if quick else [(8, 8), (32, 32), (64, 128), (128, 64)]
    add_dtypes = ["f32"] if smoke else ["f32", "bf16"] if quick else ["f32", "f16", "bf16"]
    for shape in add_shapes:
        for dtype in add_dtypes:
            specs.append(CaseSpec("add_rt", dtype, shape, dtype=dtype))

    row_shapes = [(8, 256)] if smoke else [(8, 256), (4, 64, 512)] if quick else [
        (8, 256), (4, 64, 512), (1, 256, 768), (2, 128, 1024)
    ]
    for kernel in ["gelu", "layernorm", "rms_norm", "softmax"]:
        for shape in row_shapes:
            specs.append(CaseSpec(kernel, "bf16", shape))

    rotary_shapes = [(1, 1, 64, 64)] if smoke else [(1, 2, 128, 64), (1, 2, 128, 128)]
    for shape in rotary_shapes:
        specs.append(CaseSpec("rotary", f"D{shape[-1]}", shape))

    gemm_shapes = [(32, 16, 32)] if smoke else [(64, 32, 64), (128, 64, 128)] if quick else [
        (32, 16, 32), (128, 64, 128), (256, 128, 256), (512, 512, 512)
    ]
    for kernel in ["matmul_custom", "gemm_staged", "flux_gelu", "flux_gate", "cmplx_matmul"]:
        for shape in gemm_shapes:
            specs.append(CaseSpec(kernel, "bf16", shape, dtype="bf16"))
    for shape in ([(1, 1, 16)] if smoke else [(1, 1, 16), (1, 1, 32)] if quick else [(1, 1, 16), (2, 2, 32)]):
        specs.append(CaseSpec("fftconv", f"S{shape[-1]}", shape, dtype="f32"))

    attn_shapes = [(1, 1, 64, 64)] if smoke else [(1, 2, 256, 64), (1, 2, 256, 128)] if quick else [
        (1, 2, 256, 64), (2, 4, 512, 64), (1, 2, 256, 128), (1, 4, 1024, 64)
    ]
    for kernel in ["attn_fwd", "attn_causal"]:
        for shape in attn_shapes:
            specs.append(CaseSpec(kernel, f"D{shape[-1]}", shape))
    mw_shapes = [(1, 1, 128, 64)] if smoke else [(1, 2, 256, 64), (1, 2, 256, 128)]
    for shape in mw_shapes:
        specs.append(CaseSpec("attn_multiwarp", f"D{shape[-1]}", shape))
    bwd_shapes = [(1, 1, 32, 64)] if smoke else [(1, 2, 64, 64), (1, 2, 64, 128)]
    for shape in bwd_shapes:
        for causal in ([False] if smoke else [False, True]):
            specs.append(CaseSpec("attn_bwd", "causal" if causal else "noncausal", shape))

    attn_q_formats = ["q8_0"] if smoke else [f for f in ["q8_0", "q4_0", "fp8_e4m3"] if f in formats]
    for fmt in attn_q_formats:
        specs.append(CaseSpec("attn_q", "noncausal", (1, 2, 64, 64), fmt=fmt))
        if not smoke:
            specs.append(CaseSpec("attn_q", "causal", (1, 2, 64, 64), fmt=fmt))
            if fmt in {"q8_0", "fp8_e4m3"}:
                specs.append(CaseSpec("attn_q", "multiwarp", (1, 4, 128, 64), fmt=fmt))

    linear_shapes = [(1, 1, 64, 64)] if smoke else [(1, 2, 128, 64), (1, 1, 256, 64)] if quick else [
        (1, 2, 128, 64), (2, 4, 256, 64), (1, 1, 512, 64)
    ]
    for kernel in ["linear_attn", "lin_attn_causal", "lin_attn_decay", "hedgehog", "mamba2"]:
        for shape in linear_shapes:
            specs.append(CaseSpec(kernel, f"N{shape[2]}", shape))
    based_shapes = [(1, 1, 64)] if smoke else [(1, 2, 64), (1, 1, 256)]
    for shape in based_shapes:
        specs.append(CaseSpec("based", f"N{shape[2]}", shape))

    q_shape_v = (32, 256) if smoke else (128, 256)
    q_shape_m = (32, 256, 32) if smoke else (128, 512, 64)
    for fmt in formats:
        specs.append(CaseSpec("qgemv", fmt, q_shape_v, dtype="f16", fmt=fmt))
        specs.append(CaseSpec("qgemm", fmt, q_shape_m, dtype="f16", fmt=fmt))
        specs.append(CaseSpec("qflux_gelu", fmt, q_shape_m, dtype="f16", fmt=fmt))
        if not smoke and fmt in COMMON_QUANT_FORMATS:
            specs.append(CaseSpec("qgemm_direct", fmt, q_shape_m, dtype="f16", fmt=fmt))
    if not smoke:
        for M in [1, 2, 4, 8, 16, 32, 64, 128]:
            kernel = "qgemv" if M == 1 else "qgemm"
            shape = (128, 512) if M == 1 else (128, 512, M)
            specs.append(CaseSpec(kernel, f"crossover_M{M}", shape, dtype="f16", fmt="q4_0"))

    for kernel, shape in [
        ("qgemv_w8a8", q_shape_v),
        ("qgemv_w2a8", q_shape_v),
        ("qgemm_w8a8", q_shape_m),
        ("qgemm_w2a8", q_shape_m),
        ("qgemm_fp8_scaled", q_shape_m),
        ("qgemm_actorder", q_shape_m),
        ("qgemm_actorder_fused", q_shape_m),
    ]:
        specs.append(CaseSpec(kernel, "default", shape, dtype="f16"))
    block_shape = (128, 256, 64) if smoke else (256, 512, 128)
    specs.append(CaseSpec("qgemm_fp8_block2d", "default", block_shape, dtype="f16"))

    return specs


def _parse_kernel_filter(value: str) -> set[str] | None:
    if value == "all":
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def _record_skip(spec: CaseSpec, backend: str, reason: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "backend": backend,
        "kernel": spec.kernel,
        "variant": spec.variant,
        "shape": list(spec.shape),
        "dtype": spec.dtype,
        "format": spec.fmt,
        "status": "skipped",
        "skip_reason": reason,
    }


def _run_case(spec: CaseSpec, backend: str, args: argparse.Namespace, meta: dict[str, Any]) -> dict[str, Any]:
    record_base = {
        **meta,
        "backend": backend,
        "preset": args.preset,
        "kernel": spec.kernel,
        "variant": spec.variant,
        "shape": list(spec.shape),
        "dtype": spec.dtype,
        "format": spec.fmt,
        "status": "ok",
        "skip_reason": None,
    }
    try:
        prepared = _factory_for(spec, backend, args.seed)
        got = prepared.target()
        ref = prepared.reference()
        _sync_value(got, backend)
        _sync_value(ref, backend)
        max_abs, max_rel = _error_metrics(got, ref)
        if max_abs > prepared.atol and max_rel > prepared.rtol:
            status = "failed"
            skip_reason = f"correctness failure max_abs={max_abs:.4g} max_rel={max_rel:.4g}"
            target_samples: list[float] = []
            baseline_samples: list[float] = []
        else:
            status = "ok"
            skip_reason = None
            target_samples = _time_callable(prepared.target, backend, args.warmup, args.iters, args.repeats)
            baseline_samples = (
                _time_callable(prepared.baseline, backend, args.warmup, args.iters, args.repeats)
                if prepared.baseline is not None else []
            )
        tstats = _summary_stats(target_samples)
        bstats = _summary_stats(baseline_samples)
        median_ms = tstats["median_ms"]
        baseline_ms = bstats["median_ms"]
        gbps = None
        weight_gbps = None
        gflops = None
        speedup = None
        if median_ms and prepared.bytes_moved:
            gbps = prepared.bytes_moved / (median_ms / 1e3) / 1e9
        if median_ms and prepared.weight_bytes:
            weight_gbps = prepared.weight_bytes / (median_ms / 1e3) / 1e9
        if median_ms and prepared.flops:
            gflops = prepared.flops / (median_ms / 1e3) / 1e9
        if median_ms and baseline_ms:
            speedup = baseline_ms / median_ms
        return {
            **record_base,
            "status": status,
            "skip_reason": skip_reason,
            "target": prepared.target_name,
            "baseline": prepared.baseline_name,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            **tstats,
            "baseline_median_ms": baseline_ms,
            "baseline_p20_ms": bstats["p20_ms"],
            "baseline_p80_ms": bstats["p80_ms"],
            "gbps": gbps,
            "weight_gbps": weight_gbps,
            "gflops": gflops,
            "speedup": speedup,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "atol": prepared.atol,
            "rtol": prepared.rtol,
            "bytes_moved": prepared.bytes_moved,
            "weight_bytes": prepared.weight_bytes,
            "flops": prepared.flops,
            "extra": prepared.extra,
        }
    except SkipCase as exc:
        return _record_skip(spec, backend, str(exc), record_base)
    except Exception as exc:
        if args.fail_fast:
            raise
        return {
            **_record_skip(spec, backend, f"{type(exc).__name__}: {exc}", record_base),
            "traceback": traceback.format_exc(limit=8),
        }


def _write_markdown(results: list[dict[str, Any]], meta: dict[str, Any], path: Path) -> None:
    ok = [r for r in results if r.get("status") == "ok"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    failed = [r for r in results if r.get("status") == "failed"]
    lines = [
        f"# ThunderMittens Benchmark Summary: `{meta['run_id']}`",
        "",
        f"- Command: `{meta['command']}`",
        f"- Git: `{meta['git_revision']}` dirty={meta['git_dirty']}",
        f"- Host: `{meta['host']}` `{meta['machine']}`",
        f"- Python: `{meta['python']}`  MLX: `{meta['mlx']}`  Torch: `{meta['torch']}`",
        f"- Cases: {len(results)} total, {len(ok)} ok, {len(skipped)} skipped, {len(failed)} failed",
        "",
        "## Results",
        "",
        "| Backend | Kernel | Variant | Shape | Format | Target ms | Baseline ms | Speedup | GB/s | W-GB/s | GFLOP/s | Rel err |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(ok, key=lambda x: (x["backend"], x["kernel"], x["variant"], str(x.get("format")))):
        lines.append(
            "| {backend} | `{kernel}` | {variant} | `{shape}` | {fmt} | {target} | {base} | {speedup} | {gbps} | {wgbps} | {gflops} | {rel} |".format(
                backend=r["backend"],
                kernel=r["kernel"],
                variant=r["variant"],
                shape=tuple(r["shape"]),
                fmt=r.get("format") or "",
                target=_fmt_num(r.get("median_ms"), 4),
                base=_fmt_num(r.get("baseline_median_ms"), 4),
                speedup=_fmt_num(r.get("speedup"), 3),
                gbps=_fmt_num(r.get("gbps"), 1),
                wgbps=_fmt_num(r.get("weight_gbps"), 1),
                gflops=_fmt_num(r.get("gflops"), 1),
                rel=_fmt_num(r.get("max_rel_error"), 4),
            )
        )
    if skipped or failed:
        lines += ["", "## Skips And Failures", "",
                  "| Backend | Kernel | Variant | Format | Status | Reason |",
                  "|---|---|---|---|---|---|"]
        for r in skipped + failed:
            lines.append(
                f"| {r['backend']} | `{r['kernel']}` | {r['variant']} | {r.get('format') or ''} | "
                f"{r.get('status')} | {str(r.get('skip_reason', '')).replace('|', '/')} |"
            )
    path.write_text("\n".join(lines) + "\n")


def _fmt_num(value: Any, digits: int) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def _default_counts(preset: str) -> tuple[int, int, int]:
    if preset == "smoke":
        return 1, 3, 1
    if preset == "quick":
        return 3, 10, 3
    return 5, 20, 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mlx", "torch", "both"], default="mlx")
    parser.add_argument("--preset", choices=["smoke", "quick", "comprehensive"], default="comprehensive")
    parser.add_argument("--kernel", default="all", help="comma-separated kernel names or 'all'")
    parser.add_argument("--output-dir", default=None, help="default: perf/results/YYYY-MM-DD/<run-id>")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--formats", default=None, help="comma-separated quant formats")
    parser.add_argument("--markdown", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    warmup, iters, repeats = _default_counts(args.preset)
    args.warmup = warmup if args.warmup is None else args.warmup
    args.iters = iters if args.iters is None else args.iters
    args.repeats = repeats if args.repeats is None else args.repeats
    return args


def main() -> int:
    args = parse_args()
    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "perf" / "results" / _dt.date.today().isoformat() / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = _run_metadata(run_id, args, output_dir)
    (output_dir / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    formats = _formats_for(args.preset, args.formats)
    specs = _case_specs(args.preset, formats)
    kernel_filter = _parse_kernel_filter(args.kernel)
    if kernel_filter is not None:
        specs = [s for s in specs if s.kernel in kernel_filter]
        missing = sorted(kernel_filter - {s.kernel for s in specs})
        if missing:
            raise SystemExit(f"no benchmark specs for kernel(s): {', '.join(missing)}")
    backends = ["mlx", "torch"] if args.backend == "both" else [args.backend]
    results: list[dict[str, Any]] = []
    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("w") as fh:
        for backend in backends:
            for idx, spec in enumerate(specs, 1):
                print(f"[{backend}] {idx}/{len(specs)} {spec.kernel}:{spec.variant}", flush=True)
                record = _run_case(spec, backend, args, meta)
                results.append(record)
                fh.write(json.dumps(record, sort_keys=True) + "\n")
                fh.flush()
    if args.markdown:
        _write_markdown(results, meta, output_dir / "summary.md")
    ok = sum(1 for r in results if r.get("status") == "ok")
    failed = sum(1 for r in results if r.get("status") == "failed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    print(f"Wrote {jsonl_path}")
    print(f"Cases: {ok} ok, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
