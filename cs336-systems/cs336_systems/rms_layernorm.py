import argparse
import torch
import torch.nn as nn
import timeit
import numpy as np
from cs336_basics.model import RMSNorm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50000, help="first dimension size, default=50000")
    parser.add_argument('--hidden_dim_list', type=int, nargs='+', default=[1024, 2048, 4096, 8192], help="second dimension size list, default=[1024, 2048, 4096, 8192]")
    parser.add_argument('--warmup_steps', type=int, default=1, help="Warmup steps, default=1")
    parser.add_argument('--measurement_steps', type=int, default=1000, help="measurement steps, default=1000")
    parser.add_argument('--device', type=str, default='cuda', help="default='cuda'")
    args=parser.parse_args()
    return args

def time_rms(x, w, args):
    hidden_size = x.shape[-1]
    x = x.to(args.device)
    w = w.to(args.device)
    rms_layer = RMSNorm(hidden_size=hidden_size).to(args.device)
    rms_layer.weight.data = w
    
    for _ in range(args.warmup_steps):
        out = rms_layer(x)
    
    torch.cuda.synchronize()
    
    times = []
    for _ in range(args.measurement_steps):
        start_time = timeit.default_timer()
        out = rms_layer(x)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time-start_time)
   
    mean_time = np.mean(times)
    return mean_time


def time_layernorm(x, w, b, args):
    hidden_size = x.shape[-1]
    x = x.to(args.device)
    w = w.to(args.device)
    b = b.to(args.device)
    ln = nn.LayerNorm(hidden_size, bias=True).to(args.device)
    ln.weight.data = w
    ln.bias.data = b

    for _ in range(args.warmup_steps):
        out = ln(x)
    torch.cuda.synchronize()

    times = []
    for _ in range(args.measurement_steps):
        start_time = timeit.default_timer()
        out = ln(x)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time-start_time)
    
    mean_time = np.mean(times)
    return mean_time


def main():
    import pdb; pdb.set_trace()
    args = get_args()
    for i in args.hidden_dim_list:
        input = torch.randn(args.batch_size, i).to(args.device)
        weight_rms = torch.randn(i).to(args.device)
        weight_layer_norm = torch.randn(i).to(args.device)
        bias_layer_norm = torch.randn(i).to(args.device)
        rms_time = time_rms(input, weight_rms, args)
        layernorm_time = time_layernorm(input, weight_layer_norm, bias_layer_norm, args)
        print(f'Mean time for {args.measurement_steps} iterations for rms_norm is {rms_time:0.6f}')
        print(f'Mean time for {args.measurement_steps} iterations for layer_norm is {layernorm_time:0.6f}')

if __name__=='__main__':
    main()