import torch
import torch.nn as nn
import triton
import triton.language as tl


class rms_norm_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        rms = torch.rsqrt(torch.mean(x*x, dim=-1, keepdim=True))
        return x*rms*weight
    
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

@triton.jit
def rms_triton(x_ptr : tl.pointer_type, 
               weight_ptr : tl.pointer_type, 
               x_row_stride : tl.uint32, 
               output_ptr :tl.pointer_type, 
               D_MODEL : tl.uint32, 
               BLOCK_SIZE : tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx*x_row_stride
    
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    weight_ptrs = weight_ptr + offsets
    output_ptrs = output_ptr + row_idx*x_row_stride + offsets

    mask = offsets < D_MODEL
    row = tl.load(x_ptrs, mask=mask, other=0)
    weight = tl.load(weight_ptrs, mask=mask, other=0)

    rms = tl.sqrt(tl.sum(row*row)/D_MODEL)
    #rms = tl.sum(row*row)/D_MODEL
    #row_rms = row/rms

    output = (row*weight)/rms

    #if(row_idx == 0):
    #    print("d_model", D_MODEL)
    #    print("rms",rms)
    #    print("row", row)
    #    print("rms_rms", row_rms)
    #    print("output", output)
    tl.store(output_ptrs, output, mask=mask)


class rms_norm_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        d_model = x.shape[-1]
        orig_shape = x.shape
        x = x.reshape(-1, d_model)

        assert len(weight.shape) == 1 and weight.shape[0] == d_model, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected cuda tensors"
        assert x.is_contiguous(), "Our pointer arithmetic assumes contiguous x"

        ctx.BLOCK_SIZE = triton.next_power_of_2(d_model)
        y = torch.empty(x.shape, device = x.device)
        n_rows = x.shape[0]
        rms_triton[(n_rows, )](x, weight, x.stride(0), y, d_model, num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        y = y.reshape(orig_shape)
        return y
    
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError



        
