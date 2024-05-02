import torch
import torch.nn as nn
import torch.distributed as dist

class My_DDP(nn.Module):
    def __init__(self, module):
        super(My_DDP, self).__init__()
        self.module = module
        self.handles = []

        # Broadcast module's initial parameters to all workers
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        # Register hook to synchronize gradients
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.hook_func)
                
    
    def hook_func(self, param):
        handle = dist.all_reduce(tensor=param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
        param.grad.data /= dist.get_world_size()
        self.handles.append(handle)

        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

