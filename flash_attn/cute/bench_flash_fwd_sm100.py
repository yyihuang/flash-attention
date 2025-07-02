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

import torch
from triton.testing import do_bench

from interface import _flash_attn_fwd


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
    if num_kv_heads is None:
        num_kv_heads = num_heads

    q_pt = torch.randn(batch_size, q_len, num_heads, head_dim, dtype=dtype, device="cuda")
    k_pt = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_pt = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    # It's important to synchronize before starting the benchmark
    torch.cuda.synchronize()

    fn = lambda: _flash_attn_fwd(q_pt, k_pt, v_pt, causal=causal)

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
