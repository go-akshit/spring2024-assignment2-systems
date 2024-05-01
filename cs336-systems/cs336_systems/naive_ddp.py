import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import timeit
import numpy as np
from datetime import timedelta

import cs336_basics.model as model_def
from cs336_basics.data import get_batch
import cs336_basics.nn_utils as utils
from cs336_basics.optimizer import AdamW

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=['gloo', 'nccl'], default='gloo', help="default:gloo")
    parser.add_argument("--device", type=str, choices=['cpu', 'gpu'], default='cpu', help="default:cpu")
    parser.add_argument("--data_size", type=str, choices=['512K', '1M', '10M', '50M', '100M', '500M', '1G'], default='512K', help="default:512K")
    parser.add_argument("--num_warmup_steps", type=int, default=5, help="default:5")
    parser.add_argument("--num_trial_steps", type=int, default=1, help="default:1")
    parser.add_argument("--context_length", type=int, default=128, help="context length for the model, default=128")
    parser.add_argument("--vocab_size", type=int, default=10000, help="vocab size for the model, default=10000")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the model, default=16")
    parser.add_argument("--d_model", type=int, default=768, help="d_model for the model, default=small i.e. 768")
    parser.add_argument("--d_ff", type=int, default=3072, help="d_ff for the model, default=small i.e. 3072")
    parser.add_argument("--num_layers", type=int, default=12, help="num_layers for the model, default=small i.e. 12")
    parser.add_argument("--num_heads", type=int, default=12, help="num_heads for the model, default=small i.e. 12")
    parser.add_argument("--profiler", action="store_true", default=False, help="If want to use pytorch profiler, default= false")
    parser.add_argument("--norm_layer", type=str, default='rms', choices=['rms', 'ln'], help="type of norm layer")
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
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timeout)

    if(backend == 'nccl' and device=='cpu'):
        raise ValueError("NCCL backend is not compatible with CPU devices.")
    
    return rank, local_rank, world_size, local_world_size


def run(args, model, optimizer, num_iterations, input, target):
    for _ in range(num_iterations):
        optimizer.zero_grad()
        x = input
        logits = model(x)
        loss = utils.cross_entropy(logits, target)
        loss.backward()
        for param in model.parameters():
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
       
        if(args.device == 'gpu'):
            torch.cuda.synchronize()
        dist.barrier()
        
        optimizer.step()

def run_individual(args, model, optimizer, num_iterations, input, target):
    for _ in range(num_iterations):
        optimizer.zero_grad()
        x = input
        logits = model(x)
        loss = utils.cross_entropy(logits, target)
        loss.backward()        
        optimizer.step()

def get_model_optimizer_input_targer(args, device):
        
    tokens = np.random.randint(0, 100, 5*args.context_length)
    input, target = get_batch(tokens, args.batch_size, args.context_length, device=device)

    model = model_def.BasicsTransformerLM(vocab_size=args.vocab_size, 
                                context_length=args.context_length,
                                d_model=args.d_model,
                                num_layers=args.num_layers,
                                num_heads=args.num_heads,
                                d_ff=args.d_ff,
                                attn_pdrop= None,
                                residual_pdrop=None, 
                                norm_layer=args.norm_layer).to(device)
    
    adam = AdamW(model.parameters())
    return input, target, model, adam

def broadcast_parameters(model, optimizer):
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            dist.broadcast(v, src=0)

def ddp(args, input, target, model, optimizer):
    import pdb; pdb.set_trace()
    rank, local_rank, world_size, local_world_size = setup(args.backend, args.device)
    if args.device == 'gpu':
        device = f'cuda{local_rank}'
    input, target, model, adam = get_model_optimizer_input_targer(args, device)
    
    sharded_batch_size = args.batch_size/world_size
    start_index = rank * sharded_batch_size
    end_index = start_index + sharded_batch_size
    sharded_input = input[start_index: end_index]
    sharded_target = target[start_index: end_index]
    
    if(args.device == 'gpu'):
        sharded_input = sharded_input.to(device)
        model = model.to(device)
        optimizer = optimizer.to(device)
    
    broadcast_parameters(model, optimizer)
    run(model=model, optimizer=optimizer, num_iterations=1, input=sharded_input, target=sharded_target)
    run_individual(model=model, optimizer=optimizer, num_iterations=1, input=input, target=target)
    print("both done")


def main():
    args = get_args()
    #input, target, model, adam = get_model_optimizer_input_targer(args)
    ddp(args, input, target, model, adam)

if __name__ == "__main__":
    main()
    
            
