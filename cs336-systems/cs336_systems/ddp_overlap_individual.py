import torch
import torch.nn as nn
import torch.distributed as dist

class My_DDP(nn.Module):
    def __init__(self, model):
        super(My_DDP, self).__init__()
        self.model = model
        self.handles = []

        # Broadcast model's initial parameters to all workers
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
        
        # Register hook to synchronize gradients
        for param in self.model.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.hook_func)
                
    
    def hook_func(self, param):
        handle = dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=True)
        self.handles.append(handle)

        
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
            
