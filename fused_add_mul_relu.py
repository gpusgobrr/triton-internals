from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_mul_relu(dense_in_out_ptr, scalar_ptr, dense_ptr, num_weights, xnumel, multiplier,
                       BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    index = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < xnumel
    scalar_index = index % num_weights
    tmp0 = tl.load(dense_in_out_ptr + index, mask)
    tmp1 = tl.load(scalar_ptr + scalar_index, mask, eviction_policy='evict_last')
    tmp3 = tl.load(dense_ptr + index, mask)
    # later found that there is a tl.fma function that can be used to do the
    # fused multiply add
    # https://triton-lang.org/main/python-api/generated/triton.language.fma.html#triton.language.fma
    # Option 1
    ma_result = tl.maximum(0, multiplier * tmp3 + tmp0 + tmp1)
    # Option 2
    # ma_result = tl.maximum(0, tl.math.fma(multiplier, tmp3, tmp0) + tmp1)
    tl.store(dense_in_out_ptr + index, ma_result, mask)
