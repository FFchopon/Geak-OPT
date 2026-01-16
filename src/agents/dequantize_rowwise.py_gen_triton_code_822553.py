
import torch
import triton
import triton.language as tl

@triton.jit
def _dequantize_rowwise(x_ptr, state_x, output_ptr, inv_127, n_elements, BLOCK_SIZE: tl.constexpr, P2: tl.constexpr):
    # Get program ID for blocks
    pid = tl.program_id(axis=0)

    # Calculate starting index for the current block
    block_start = pid * BLOCK_SIZE

    # Create index range for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid indices
    mask = offsets < n_elements

    # Calculate row and col indices for each offset
    n_cols = P2
    rows = offsets // n_cols
    cols = offsets % n_cols

    # Load input values with mask
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load the max value for each row corresponding to the element
    row_max_vals = tl.load(state_x + rows, mask=mask, other=0.0)

    # Dequantize: multiply each element by its row's max and inv_127
    dequantized_vals = x_vals.to(tl.float32) * row_max_vals * inv_127

    # Store the results back
    tl.store(output_ptr + offsets, dequantized_vals, mask=mask)


def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
    # Prepare the output tensor with the same shape and device as input
    output = torch.empty_like(x, dtype=torch.float32)

    # Calculate total number of elements
    n_elements = x.numel()

    # Calculate P2 - nearest power of two of the number of columns
    n_cols = x.shape[1]
    P2 = 1
    while P2 < n_cols:
        P2 *= 2

    # Define block size - using a standard efficient block size
    BLOCK_SIZE = 1024

    # Calculate inverse of 127
    inv_127 = 1.0 / 127.0

    # Set up the execution grid
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the Triton kernel
    _dequantize_rowwise[grid](
        x,
        state_x,
        output,
        inv_127,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        P2=P2
    )

    return output

def tensor(dtype, device):
    return torch.tensor([], dtype=dtype, device=device)

def rand(dtype, device):
    return torch.rand(1, dtype=dtype, device=device)

def ones(dtype, device):
    return torch.ones(1, dtype=dtype, device=device)

##################################################################################################################################################

# Test function for dequantize_rowwise

def test_dequantize_rowwise():

    results = {}



    # Test case 1: Simple case

    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8, device='cuda')

    state_x = torch.tensor([4.0, 8.0], dtype=torch.float32, device='cuda')

    output = dequantize_rowwise(x, state_x)

    results['test_case_1'] = output



    # Test case 2: Larger input

    x = torch.randint(-128, 127, (10, 16), dtype=torch.int8, device='cuda')

    state_x = torch.rand(10, dtype=torch.float32, device='cuda') * 10

    output = dequantize_rowwise(x, state_x)

    results['test_case_2'] = output



    # Test case 3: Edge case with zeros

    x = torch.zeros((5, 8), dtype=torch.int8, device='cuda')

    state_x = torch.ones(5, dtype=torch.float32, device='cuda')

    output = dequantize_rowwise(x, state_x)

    results['test_case_3'] = output



    # Test case 4: Different block size

    x = torch.randint(-128, 127, (3, 32), dtype=torch.int8, device='cuda')

    state_x = torch.rand(3, dtype=torch.float32, device='cuda') * 10

    output = dequantize_rowwise(x, state_x)

    results['test_case_4'] = output



    return results



# Run the test function

result_gold = test_dequantize_rowwise()


import os
try:
    import torch
except Exception:
    torch = None
if os.environ.get('GEAK_PROFILE_DIAG', '0') in {'1','true','yes','y'}:
    try:
        if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()
            print('GEAK_PROFILE_DIAG: cuda_synchronized')
        else:
            print('GEAK_PROFILE_DIAG: cuda_not_available')
    except Exception as _e:
        print('GEAK_PROFILE_DIAG: error', type(_e).__name__, str(_e))
if os.environ.get('GEAK_NCU_SELFTEST', '0') in {'1','true','yes','y'}:
    try:
        if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
            x = torch.randn((1024,), device='cuda')
            y = x + 1
            _ = y.sum()
            torch.cuda.synchronize()
            print('GEAK_NCU_SELFTEST: launched_cuda_ops')
        else:
            print('GEAK_NCU_SELFTEST: cuda_not_available')
    except Exception as _e:
        print('GEAK_NCU_SELFTEST: error', type(_e).__name__, str(_e))
