import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from copy import deepcopy

class OptimizerSharded(optim.Optimizer):

    def _init_(self, params, optimizer_cls, **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer = None
        self.kwargs = kwargs
        self.optimizer_cls = optimizer_cls
        self.params_rank = {}
        super(OptimizerSharded, self)._init_(params, kwargs)

    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure, **kwargs)
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
        param_shrd['params'] = param_group['params'][self.rank::self.world_size]
        if self.optimizer is None:
            self.optimizer = self.optimizer_cls([param_shrd], **self.kwargs)
        else:
            self.optimizer.add_param_group(param_shrd)
        super().add_param_group(param_group)