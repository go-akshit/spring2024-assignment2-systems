import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import timeit
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=['gloo', 'nccl'], default='gloo', help="default:gloo")
    parser.add_argument("--device", type=str, choices=['cpu', 'gpu'], default='cpu', help="default:cpu")
    parser.add_argument("--data_size", type=str, choices=['512K', '1M', '10M', '50M', '100M', '500M', '1G'], default='512K', help="default:512K")
    parser.add_argument("--n_procs", type=int, choices=[2, 4, 6], default=2, help="default:2")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="default:0")
    parser.add_argument("--num_trial_steps", type=int, default=1, help="default:1")
    args = parser.parse_args()
    return args

def setup(rank, world_size, backend, device):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if(backend == 'nccl' and device=='cpu'):
        raise ValueError("NCCL backend is not compatible with CPU devices.")
    if(device=='gpu'):
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def size_to_bytes(size_str):
    units = size_str[-1]
    value = size_str[:-1]
    units_size = {'K': 1024, 'M': 1024**2, 'G':1024**3}
    multiple = units_size[units]
    return int(value)*multiple


def distributed_demo(rank, *args):
    data_size, backend, device, n_procs, warmup, trial = args
    #print("########################################")
    #print(f"data: {data_size}, backend: {backend}, device: {device}")
    setup(rank=rank, world_size=n_procs, backend=backend, device=device)
    num_elements = size_to_bytes(data_size)//4
    if device == 'gpu':
        device = f"cuda:{rank}"
    else:
        device = "cpu"
    orig_tensor_data = torch.randint(0, 10, (num_elements,), device=device)
    tensor_data = orig_tensor_data

    for _ in range(warmup):
        data = tensor_data
        dist.all_reduce(tensor=data, async_op=False)
        if(device == 'gpu'):
            torch.cuda.synchronize()
        dist.barrier()
        #tensor_data.copy_(orig_tensor_data)
    
    orig_tensor_data = torch.randint(0, 10, (num_elements,), device=device)
    tensor_data = orig_tensor_data
    durations = []
    for _ in range(trial):
        data = tensor_data
        #print(f"rank: {rank} data (before all-reduce): {data}")
        start_time = timeit.default_timer()
        
        dist.all_reduce(tensor=data, async_op=False)
        
        if(device == 'gpu'):
            torch.cuda.synchronize()
        dist.barrier()

        end_time = timeit.default_timer()
        duration = end_time - start_time
        durations.append(duration)
        #tensor_data.copy_(orig_tensor_data)
        
        #print(f"rank: {rank} data (after all-reduce): {data}")
        
    
    #print(f"Mean time = {np.mean(durations):0.6f}. std deviation = {np.std(durations):0.6f}")
    print(f"{data_size}, {backend}, {device}, {np.mean(durations):0.6f}")
        

if __name__ == "__main__":
    args = get_args()
    for backend, device in [('gloo', 'cpu'), ('gloo', 'gpu'), ('nccl', 'gpu')]:
        for size in ['512K', '1M', '10M', '50M', '100M', '500M', '1G']:
            mp.spawn(fn=distributed_demo, nprocs=args.n_procs, join=True, args=(size,
                                                                                backend, 
                                                                                device, 
                                                                                args.n_procs, 
                                                                                args.num_warmup_steps, 
                                                                                args.num_trial_steps), 
                                                                                )
            
