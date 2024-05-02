import torch
import torch.nn as nn
import torch.distributed as dist

class My_DDP(nn.Module):
    def _init_(self, module: torch.nn.Module):
        super(My_DDP, self)._init_()
        self.module = module
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Broadcast model parameters
        self._broadcast_parameters()

        # Register gradient synchronization hook
        self._hooks = []
        for param in self.module.parameters():
            if param.requires_grad:
                hook = param.register_post_accumulate_grad_hook(self._synchronize_gradients)
                self._hooks.append(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        # Synchronize gradients across all processes
        dist.barrier()

    def _broadcast_parameters(self, async_op=False):
        for param in self.module.parameters():
            dist.broadcast(param.data, 0, async_op=async_op)

    def _synchronize_gradients(self, param, async_op=True):
        param.grad.data /= dist.get_world_size()
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=async_op)


    def _del_(self):
        for hook in self._hooks:
            hook.remove()
        dist.destroy_process_group()

