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
            hook = param.register_post_accumulate_grad_hook(self.hook_func)
            self.all_hooks.append(hook)

                
    
    def hook_func(self, param):
        dist.all_reduce(tensor=param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
        param.grad.data /= dist.get_world_size()

        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        dist.barrier()

    def _del_(self):
        dist.destroy_process_group()