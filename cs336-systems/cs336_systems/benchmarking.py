import argparse
import torch
import torch.nn as nn
import numpy as np
import timeit
from torch.profiler import profile, record_function, ProfilerActivity
import cs336_basics.model as model_def
from cs336_basics.data import get_batch
import cs336_basics.nn_utils as utils
from cs336_basics.optimizer import AdamW


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup_steps", type=int, default=1, help="Number of warmup steps, default=1")
    parser.add_argument("--measurement_steps", type=int, default=1, help="Number of measurement steps, default=1")
    parser.add_argument("--pass_type", type=str, choices=['forward', 'backward', 'both'], default='forward', help="specify which pass to time, default='forward'")
    parser.add_argument("--context_length", type=int, default=128, help="context length for the model, default=128")
    parser.add_argument("--vocab_size", type=int, default=10000, help="vocab size for the model, default=10000")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the model, default=16")
    parser.add_argument("--d_model", type=int, default=768, help="d_model for the model, default=small i.e. 768")
    parser.add_argument("--d_ff", type=int, default=3072, help="d_ff for the model, default=small i.e. 3072")
    parser.add_argument("--num_layers", type=int, default=12, help="num_layers for the model, default=small i.e. 12")
    parser.add_argument("--num_heads", type=int, default=12, help="num_heads for the model, default=small i.e. 12")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'], help="device, default=cuda")
    parser.add_argument("--profiler", type=bool, default='false', help="If want to use pytorch profiler, default= false")
    args = parser.parse_args()
    return args

def profiling(model, tokens, optimizer, args):
    #import pdb; pdb.set_trace()
    for _ in range(args.warmup_steps):
        input, target = get_batch(tokens, args.batch_size, args.context_length, device="cuda")
        optimizer.zero_grad()
        logits = model(input)
        loss = utils.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 experimental_config = torch._C._profiler._ExperimentalConfig(verbose=True),
                 record_shapes=True,
                 profile_memory=False,
                 with_stack=True) as prof:
        
        for _ in range(args.measurement_steps):
            input, target = get_batch(tokens, args.batch_size, args.context_length, device="cuda")
            optimizer.zero_grad() 
            with record_function('forward pass'):
                logits = model(input)
            with record_function('loss computation'):
                loss = utils.cross_entropy(logits, target)
            with record_function('backward pass'):
                loss.backward()
            with record_function('optimizer pass'):
                optimizer.step()
            prof.step()
    
    prof.export_stacks('lm_profiler_stacks.txt', 'self_cuda_time_total')
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=500))


def end_to_end_benchmarking(model, tokens, args):
    forward_measurement_times = []
    backward_measurement_times = []
    
    for _ in range(args.warmup_steps):
        input, target = get_batch(tokens, args.batch_size, args.context_length, device="cuda")
        logits = model(input)
        loss = utils.cross_entropy(logits, target)
        loss.backward()
    
    torch.cuda.synchronize()

    for _ in range(args.measurement_steps):
        input, target = get_batch(tokens, args.batch_size, args.context_length, device="cuda")
        torch.cuda.synchronize()

        start_time = timeit.default_timer()
        logits = model(input)
        loss = utils.cross_entropy(logits,target)
        torch.cuda.synchronize()
        end_time_forward = timeit.default_timer()

        loss.backward()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()

        forward_measurement_times.append(end_time_forward-start_time)
        backward_measurement_times.append(end_time-end_time_forward)
    
    if args.pass_type=='forward' or args.pass_type=='both':
        print(f'mean of forward measurement times = {np.mean(forward_measurement_times):.6f}')
        print(f'std. dev of forward measurement times = {np.std(forward_measurement_times):.6f}')

    if args.pass_type=='backward'or args.pass_type=='both':
        print(f'mean of backward measurement times = {np.mean(backward_measurement_times):.6f}')
        print(f'std. dev of backward measurement times = {np.std(backward_measurement_times):.6f}')

def main():
    args = get_args()
    device = args.device
    model = model_def.BasicsTransformerLM(vocab_size=args.vocab_size, 
                                context_length=args.context_length,
                                d_model=args.d_model,
                                num_layers=args.num_layers,
                                num_heads=args.num_heads,
                                d_ff=args.d_ff,
                                attn_pdrop= None,
                                residual_pdrop=None).to(device)
    tokens = np.random.randint(0, args.vocab_size, 5*args.context_length)
    
    end_to_end_benchmarking(model, tokens, args)

    if(args.profiler):
        adam = AdamW(model.parameters())
        profiling(model, tokens, adam, args)

if __name__ == '__main__':
    main()




    


