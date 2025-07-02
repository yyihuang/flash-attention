"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
from typing import Optional, Tuple
import torch
from triton.testing import do_bench

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import utils

from interface import _flash_attn_fwd
from flash_fwd_sm100 import FlashAttentionForwardSm100

_flash_attn_fwd.compile_cache = {}


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

def bench_fmha_blackwell(
    batch_size,
    q_len,
    kv_len,
    num_heads,
    head_dim,
    causal,
    dtype,
    num_kv_heads=None
):
    def maybe_contiguous(x):
        return x.contiguous() if x is not None and x.stride(-1) != 1 else x

    if num_kv_heads is None:
        num_kv_heads = num_heads

    cu_seqlens_q = None
    cu_seqlens_k = None
    seqused_q = None
    seqused_k = None
    max_seqlen_q = None
    softmax_scale = None
    softcap = 0.0
    m_block_size = 128
    n_block_size = 128
    num_threads = 384
    _compute_capability = None

    q = torch.randn(batch_size, q_len, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = max_seqlen_q
        total_q = q.shape[0]
    seqlen_k, num_head_kv, _ = k.shape[-3:]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), "cu_seqlens_k must have shape (batch_size + 1,)"
    if cu_seqlens_q is not None:
        assert max_seqlen_q is not None, "max_seqlen_q must be provided if cu_seqlens_q is provided"
        assert cu_seqlens_q.shape == (batch_size + 1,), "cu_seqlens_q must have shape (batch_size + 1,)"
    assert seqused_q is None or seqused_q.shape == (batch_size,), "seqused_q must have shape (batch_size,)"
    assert seqused_k is None or seqused_k.shape == (batch_size,), "seqused_k must have shape (batch_size,)"
    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            assert t.stride(0) == 1, "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
    assert all(t is None or t.is_cuda for t in (q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv

    out_torch_dtype = q.dtype
    device = q.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    out = torch.empty(*q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device)
    lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad
    lse = torch.empty(lse_shape, dtype=torch.float32, device=device) if requires_grad else None

    dtype = torch2cute_dtype_map[q.dtype]
    q_tensor, k_tensor, v_tensor, o_tensor = [
        utils.convert_from_dlpack(
            t.detach(), leading_dim=t.ndim - 1, divisibility=128 // dtype.width
        ) for t in (q, k, v, out)
    ]
    lse_tensor = utils.convert_from_dlpack(lse, leading_dim=lse.ndim - 1, alignment=4) if lse is not None else None
    cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0) if t is not None else None
        for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]
    max_seqlen_q = cutlass.Int32(max_seqlen_q) if max_seqlen_q is not None else None
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compute_capability = torch.cuda.get_device_capability()[0] if _compute_capability is None else _compute_capability
    assert compute_capability in [9, 10], "Unsupported compute capability. Supported: 9.x, 10.x"
    compile_key = (
        dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, softcap != 0.0,
        lse is None, cu_seqlens_q is None, cu_seqlens_k is None, seqused_q is None, seqused_k is None,
        m_block_size, n_block_size, num_threads,
        compute_capability,
    )

    if compile_key not in _flash_attn_fwd.compile_cache:
        fa_fwd = FlashAttentionForwardSm100(
                head_dim,
                head_dim_v,
                is_causal=causal,
                qhead_per_kvhead=qhead_per_kvhead,
                is_persistent=True,
            )
        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            fa_fwd, q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor,
            cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor,
            max_seqlen_q, softmax_scale, softcap, current_stream
        )

    # synchronize before starting the benchmark
    torch.cuda.synchronize()

    fn = lambda: _flash_attn_fwd.compile_cache[compile_key](
        q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor,
        cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor,
        max_seqlen_q, softmax_scale, softcap, current_stream
    )

    ms = do_bench(
        fn,
        warmup=100,
        rep=1000,
    )

    def flops(ms):
        # The number of flops is 2 * b * s_q * s_k * h * d.
        # This is for the two GEMMs (QK^T and P*V).
        total_ops = 2 * batch_size * num_heads * q_len * kv_len * head_dim
        if causal:
            total_ops /= 2
        return total_ops / ms / 1e9

    print(
        f"bench_fmha_blackwell (batch_size={batch_size}, q_len={q_len}, kv_len={kv_len}, num_heads={num_heads}, head_dim={head_dim}, causal={causal}),"
        f" perf: {flops(ms):.3f} TFLOPs/s, time: {ms:.3f} ms"
    )


if __name__ == "__main__":
    for causal in [False, True]:
        for q_len in [512, 1024, 2048, 4096]:
            bench_fmha_blackwell(32, q_len, q_len, 32, 128, causal, torch.bfloat16)

        bench_fmha_blackwell(128, 512, 512, 32, 128, causal, torch.bfloat16)
        bench_fmha_blackwell(64, 1024, 1024, 32, 128, causal, torch.bfloat16)
        bench_fmha_blackwell(32, 2048, 2048, 32, 128, causal, torch.bfloat16)
        bench_fmha_blackwell(16, 4096, 4096, 32, 128, causal, torch.bfloat16)
        bench_fmha_blackwell(8, 8192, 8192, 32, 128, causal, torch.bfloat16)
        bench_fmha_blackwell(4, 16384, 16384, 32, 128, causal, torch.bfloat16)
        bench_fmha_blackwell(2, 32768, 32768, 32, 128, causal, torch.bfloat16)
        bench_fmha_blackwell(1, 65536, 65536, 32, 128, causal, torch.bfloat16)
