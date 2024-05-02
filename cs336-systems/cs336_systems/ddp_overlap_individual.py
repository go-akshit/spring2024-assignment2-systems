import torch
import torch.nn as nn
import torch.distributed as dist

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
    
    def start_gradient_synchronization_on_batch_start(self):
        pass

        # Broadcast module's initial parameters to all workers
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0, async_op=False)

        # Add parameters to buckets
        for param in self.module.parameters():
            if param.requires_grad:
                self.add_param_to_bucket(param)

        # Add any remaining parameters in the current bucket to the buckets list
        if self.current_bucket:
            self.buckets.append(self.current_bucket)

    def add_param_to_bucket(self, param):
        param_size = param.data.numel() * param.data.element_size()
        if self.current_bucket_size + param_size > self.bucket_size_mb:
            self.buckets.append(self.current_bucket)
            self.current_bucket = []
            self.current_bucket_size = 0
        self.current_bucket.append(param)
        self.current_bucket_size += param_size
        hook = param.register_hook(lambda grad, param=param: self.hook_func(param))
        self.hooks.append(hook)

    def hook_func(self, param):
        param.grad.data /= dist.get_world_size()
        if all(hasattr(p, 'grad') and p.grad is not None for p in self.current_bucket):
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