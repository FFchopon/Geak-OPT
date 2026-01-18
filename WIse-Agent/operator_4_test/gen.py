# Triton kernel for rowwise dequantization
import math
import torch
import triton
import triton.language as tl

@triton.jit
def dequantize_rowwise_kernel(
    x_ptr,          # int8/uint8
    scale_ptr,      # fp16/fp32, shape [M] (每行一个scale/max)
    out_ptr,        # fp16/bf16/fp32
    n_cols: tl.constexpr,
    INV_127: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)  # row id

    # 本行起始地址
    row_start = pid * n_cols

    # 向量化列索引
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    # 让编译器更敢做向量化/合并访存（若满足对齐更好）
    tl.multiple_of(row_start, 16)
    tl.max_contiguous(cols, 16)

    # load
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0).to(tl.float32)
    s = tl.load(scale_ptr + pid).to(tl.float32)

    # compute
    y = x * (s * INV_127)

    # store
    tl.store(out_ptr + row_start + cols, y.to(tl.float16), mask=mask)


def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor, out_dtype=torch.float16):
    """
    x: [M, N], typically int8/uint8
    state_x: [M], scale/max per row, fp16/fp32
    """
    assert x.is_cuda and state_x.is_cuda
    assert x.ndim == 2 and state_x.ndim == 1
    M, N = x.shape
    assert state_x.numel() == M

    out = torch.empty((M, N), device=x.device, dtype=out_dtype)

    # BLOCK 取 next pow2，适配任意 N
    BLOCK_N = 1 << (N - 1).bit_length()

    # 经验：按 BLOCK_N 选 warps（也可以后面用 autotune）
    if BLOCK_N <= 128:
        num_warps = 4
    elif BLOCK_N <= 256:
        num_warps = 8
    else:
        num_warps = 8

    grid = (M,)
    dequantize_rowwise_kernel[grid](
        x, state_x, out,
        n_cols=N,
        INV_127=1.0 / 127.0,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
