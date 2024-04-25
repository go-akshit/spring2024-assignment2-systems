import torch
import torch.nn as nn
import triton
import triton.language as tl


def rmsnorm_jvp_x(x, g, grad_output):
    shape_orig = x.shape 
    d_model = g.shape[0]
    x = x.reshape(-1, d_model)
    grad_output = grad_output.reshape(-1, d_model)
    rms = torch.sqrt(torch.mean(x*x, dim=-1, keepdim=True))
    y = torch.empty(x.shape)
    #import pdb; pdb.set_trace()
    # for row in range(y.shape[0]):
    #     for col in range(y.shape[1]):
    #         temp = -x[row][col]/(d_model * pow(rms[row][0], 3)) * g * x[row] * grad_output[row]
    #         y[row][col] = torch.sum(temp, dim=-1)
    #         y[row][col] += grad_output[row][col] * (g[col]/rms[row][0]).squeeze(dim=0) 

    
    temp1 = -(x/(d_model * pow(rms,3))).unsqueeze(-1)
    temp2 = (g * x * grad_output).unsqueeze(-2)
    temp = torch.sum(temp1 * temp2, dim=-1) + grad_output*g/rms
    y = temp.reshape(shape_orig)
    return y

def rmsnorm_jvp_g(x, g, grad_output):
    rms = torch.rsqrt(torch.mean(x*x, dim=-1, keepdim=True))
    temp = x*rms*grad_output
    grad_g = torch.sum(temp, dim=(0,1))
    return grad_g

class rms_norm_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        rms = torch.rsqrt(torch.mean(x*x, dim=-1, keepdim=True))
        ctx.save_for_backward(x, weight)
        return x*rms*weight
    
    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_weight = rmsnorm_jvp_g(x, weight ,grad_out)
        grad_x = rmsnorm_jvp_x(x, weight, grad_out)
        return grad_x, grad_weight


@triton.jit
def rms_triton_fwd(x_ptr : tl.pointer_type, 
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

@triton.jit
def rms_triton_bwd(grad_out_ptr: tl.pointer_type,
                   grad_x_ptr: tl.pointer_type,
                   partial_grad_weight_ptr: tl.pointer_type, 
                   x_ptr: tl.pointer_type,
                   weight_ptr: tl.pointer_type, 
                   x_row_stride: tl.uint32,
                   D_MODEL: tl.uint32, 
                   BLOCK_SIZE: tl.constexpr):
    
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx*x_row_stride
    
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    weight_ptrs = weight_ptr + offsets
    grad_out_ptrs = grad_out_ptr + row_idx*x_row_stride + offsets
    grad_x_ptrs = grad_x_ptr + row_idx*x_row_stride + offsets
    partial_grad_weight_ptrs = partial_grad_weight_ptr +  row_idx*x_row_stride + offsets

    mask = offsets < D_MODEL
    row = tl.load(x_ptrs, mask=mask, other=0)
    weight = tl.load(weight_ptrs, mask=mask, other=0)
    grad_out = tl.load(grad_out_ptrs, mask=mask, other=0)

    rms = tl.sqrt(tl.sum(row*row)/D_MODEL)
    partial_grad_weight = (row*grad_out)/rms
    tl.store(partial_grad_weight_ptrs, partial_grad_weight, mask=mask)

    temp1 = -row/(D_MODEL * rms * rms * rms)
    temp2 = weight * row * grad_out
    temp1.expand_dims(1)
    temp2.expand_dims(0)
    grad_x = tl.sum(temp1 * temp2) + grad_out * weight/rms
    tl.store(grad_x_ptrs, grad_x, mask=mask)




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
        rms_triton_fwd[(n_rows, )](x, weight, x.stride(0), y, d_model, num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        y = y.reshape(orig_shape)
        return y
    
    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        d_model = x.shape[-1]
        orig_shape = x.shape
        x = x.reshape(-1, d_model)
        grad_out = grad_out.reshape(-1, d_model)

        ctx.BLOCK_SIZE = triton.next_power_of_2(d_model)
        grad_weight = torch.empty(x.shape, device = x.device)
        grad_x = torch.empty(x.shape, device = x.device)
        n_rows = x.shape[0]
        rms_triton_bwd[(n_rows, )](grad_out, grad_x, grad_weight, x, weight, x.stride(0), d_model, num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        grad_weight = torch.sum(grad_weight, dim=0)
        x = x.reshape(orig_shape)
        grad_out = grad_out.reshape(orig_shape)
        grad_x = rmsnorm_jvp_x(x, weight, grad_out)
        return grad_x, grad_weight        
        



        
