import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from copy import deepcopy

class My_DDP(nn.Module):
    def __init__(self, module):
        super(My_DDP, self).__init__()
        self.module = module
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.handles = []
        self.all_hooks = []
        # Broadcast module's initial parameters to all workers
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0, async_op=False)
        
        # Register hook to synchronize gradients
        for param in self.module.parameters():
            if param.requires_grad:
                hook = param.register_post_accumulate_grad_hook(self.hook_func)
                self.all_hooks.append(hook)

    def hook_func(self, param):
        param.grad.data /= dist.get_world_size()
        handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

    def _del_(self):
        for hook in self.all_hooks:
            hook.remove()
        dist.destroy_process_group()

class My_DDP_Bucket(nn.Module):
    def __init__(self, module, bucket_size_mb=25):
        super(My_DDP_Bucket, self).__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb * 1024 * 1024
        self.buckets = []
        self.current_bucket_size = 0
        self.current_bucket = []
        self.handles = []
        self.hooks = []
        self.param_to_idx = {}

        # Broadcast module's initial parameters to all workers
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0, async_op=False)

        # Add parameters to buckets
        for i, param in enumerate(self.module.parameters()):
            if param.requires_grad:
                self.add_param_to_bucket(param, i)
        
        # Add any remaining parameters in the current bucket to the buckets list
        if self.current_bucket:
            self.buckets.append(self.current_bucket)
        if dist.get_rank() == 0:
            print(f"{self.param_to_idx = }")
            print(f"{len(self.buckets) = }")
    def start_gradient_synchronization_on_batch_start(self):
        self.handles = []


    def add_param_to_bucket(self, param, i):
        param_size = param.data.numel() * param.data.element_size()
        if self.current_bucket_size > self.bucket_size_mb:
            self.buckets.append(self.current_bucket)
            self.current_bucket = []
            self.current_bucket_size = 0
        self.current_bucket.append(param)
        self.param_to_idx[i] = len(self.buckets)
        self.current_bucket_size += param_size
        hook = param.register_post_accumulate_grad_hook(lambda p, idx=i: self.hook_func(p, idx))
        self.hooks.append(hook)

    def hook_func(self, param, i):
        if param.grad is not None:
            param.grad.data /= dist.get_world_size()

        print(f"rank = {dist.get_rank()}")
        print(f"{self.param_to_idx = }")
        print(f"{len(self.buckets) = }")
        if all(hasattr(p, 'grad') and p.grad is not None for p in self.buckets[self.param_to_idx[i]]):
            self.all_reduce_bucket(self.current_bucket)

    def all_reduce_bucket(self, bucket):
        for param in bucket:
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles = []  

    def __del__(self):
        for hook in self.hooks:
            hook.remove()
        dist.destroy_process_group()

class My_DDP_Opt(optim.Optimizer):

    def __init__(self, params, optimizer_cls, **kwargs):
        self.optimizer = optimizer_cls
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.kwargs = kwargs
        super(My_DDP_Opt, self).__init__(params, kwargs)
        
    def step(self, closure=None, **kwargs):
        self.optimizer_cls.step(closure, **kwargs)
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                dist.broadcast(param.data, src=self._get_rank(i))
        dist.barrier()

    def _get_rank(self, idx):
        return idx % self.world_size

    def add_param_group(self, param_group):
        param_shrd = {}
        for key in param_group.keys():
            if key != 'params':
                param_shrd[key] = param_group[key]
        param_shrd = deepcopy(param_shrd)
        param_shrd['params'] = [param for i, param in enumerate(param_group['params']) if i % self.world_size == self.rank]
        if self.optimizer is None:
            self.optimizer = self.optimizer_cls([param_shrd], **self.kwargs)
        else:
            self.optimizer.add_param_group(param_shrd)
        super().add_param_group(param_group)