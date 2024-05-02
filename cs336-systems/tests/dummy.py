import torch
import torch.distributed as dist

class DDPBucketedContainer(torch.nn.Module):
    def _init_(self, module: torch.nn.Module, bucket_size_mb: float):
        super(DDPBucketedContainer, self)._init_()
        self.module = module
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Broadcast model parameters
        self._broadcast_parameters()

        # Register gradient synchronization hook
        self._hooks = []
        self._workers_and_params = [[None, []]] # (worker, params) for the buckets
        self._flattened_params = [[]] # This list will hold the flattened parameters for each bucket
        self.bucket_idx = 0
        self.bucket_size_mb = bucket_size_mb * 10**6  # Convert MB to bytes

        for name, param in self.module.named_parameters():
            if param.requires_grad:
                hook = param.register_post_accumulate_grad_hook(
                    lambda p, param_name=name: self._synchronize_gradients(p, param_name))
                self._hooks.append(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def start_gradient_synchronization_on_batch_start(self):
        self._workers_and_params = [[None, []]]  # (worker, params) for the buckets
        self._flattened_params = [[]]  # This list will hold the flattened parameters for each bucket
        self.bucket_idx = 0
        # Synchronize gradients across all processes
        dist.barrier()
    def finish_gradient_synchronization(self):
        # Synchronize gradients across all processes
        self._flattened_params[self.bucket_idx] = (
            torch._utils._flatten_dense_tensors(self._flattened_params[self.bucket_idx]))
        self._workers_and_params[self.bucket_idx][0] = (
            dist.all_reduce(self._flattened_params[self.bucket_idx], op=dist.ReduceOp.SUM, async_op=False))

        dist.barrier()
        self._unflatten_and_get_grads()

    def _broadcast_parameters(self, async_op=False):
        for param in self.module.parameters():
            dist.broadcast(param.data, 0, async_op=async_op)

    def _synchronize_gradients(self, param, param_name, async_op=True):
        # Get size of bucket index bucket_idx
        bucket_size_bytes = 0
        for p in self._flattened_params[self.bucket_idx]:
            bucket_size_bytes += p.numel() * p.element_size()
        if (bucket_size_bytes > self.bucket_size_mb): # Adding current will make bucket exceed the limit
            # Reduce gradients in the current bucket
            self._flattened_params[self.bucket_idx] = (
                torch._utils._flatten_dense_tensors(self._flattened_params[self.bucket_idx]))
            self._workers_and_params[self.bucket_idx][0] = (
                dist.all_reduce(self._flattened_params[self.bucket_idx], op=dist.ReduceOp.SUM, async_op=async_op))

            self.bucket_idx += 1
            self._flattened_params.append([])
            self._workers_and_params.append([None, []])

        self._flattened_params[self.bucket_idx].append(param.grad.data)
        self._workers_and_params[self.bucket_idx][1].append(param)

    def _unflatten_and_get_grads(self):
        for i in range(self.bucket_idx + 1):
            if i != self.bucket_idx:
                self._workers_and_params[i][0].wait()
            print(f"Type of self._flattened_params[i] = {type(self._flattened_params[i])}")
            print(f"Type of self._workers_and_params[i][1] = {type(self._workers_and_params[i][1])}")
            self._flattened_params[i] = torch._utils._unflatten_dense_tensors(self._flattened_params[i], self._workers_and_params[i][1])
            for param, grad in zip(self._workers_and_params[i][1], self._flattened_params[i]):
                param.grad.data = grad / dist.get_world_size()


    def _del_(self):
        for hook in self._hooks:
            hook.remove()
        dist.destroy_process_group()