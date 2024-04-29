import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import timeit
import numpy as np
from datetime import timedelta

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=['gloo', 'nccl'], default='gloo', help="default:gloo")
    parser.add_argument("--device", type=str, choices=['cpu', 'gpu'], default='cpu', help="default:cpu")
    parser.add_argument("--data_size", type=str, choices=['512K', '1M', '10M', '50M', '100M', '500M', '1G'], default='512K', help="default:512K")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="default:0")
    parser.add_argument("--num_trial_steps", type=int, default=1, help="default:1")
    args = parser.parse_args()
    return args

def setup(backend, device):
    # These variables are set via srun
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    # MASTER_ADDR and MASTER_PORT should have been set in our sbatch script,
    # so we make sure that's the case.
    assert os.environ["MASTER_ADDR"]
    assert os.environ["MASTER_PORT"]
    # Default timeout is 30 minutes. Reducing the timeout here, so the job fails quicker if there's
    # a communication problem between nodes.
    timeout = timedelta(seconds=60)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timeout)

    if(backend == 'nccl' and device=='cpu'):
        raise ValueError("NCCL backend is not compatible with CPU devices.")
    
    return rank, world_size, local_rank, local_world_size

def size_to_bytes(size_str):
    units = size_str[-1]
    value = size_str[:-1]
    units_size = {'K': 1024, 'M': 1024**2, 'G':1024**3}
    multiple = units_size[units]
    return int(value)*multiple


def multinode_distributed_demo(*args):
    data_size, backend, device, warmup, trial = args
    rank, world_size, local_rank, local_world_size = setup(backend=backend, device=device)
    print( f"World size: {world_size}, global rank: {rank}, local rank: {local_rank}, local world size: {local_world_size}")
    
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
    
    orig_tensor_data = torch.randint(0, 10, (num_elements,), device=device)
    tensor_data = orig_tensor_data
    durations = []
    for _ in range(trial):
        data = tensor_data
        print(f"rank: {rank} data (before all-reduce): {data}")
        start_time = timeit.default_timer()
        
        dist.all_reduce(tensor=data, async_op=False)
        
        if(device == 'gpu'):
            torch.cuda.synchronize()
        dist.barrier()

        end_time = timeit.default_timer()
        duration = end_time - start_time
        durations.append(duration)
        
        print(f"rank: {rank} data (after all-reduce): {data}")
    
    mean_duration_per_rank = np.mean(durations)
    #durations_all_ranks = [None] * n_procs
    #dist.all_gather_object(durations_all_ranks, mean_duration_per_rank)
    
    
    #print(f"Mean time = {np.mean(durations):0.6f}. std deviation = {np.std(durations):0.6f}")
    # if rank == 0:
        # print(f"{data_size}, {backend}, {device}, {n_procs}, {mean_duration_per_rank:0.6f}")
    #print(f"duration all ranks {durations_all_ranks}")
        

def main():
    args = get_args()
    arguments = (args.data_size,args.backend,args.device,args.num_warmup_steps,args.num_trial_steps)
    multinode_distributed_demo(arguments)

if __name__ == "__main__":
    main()
    
            
