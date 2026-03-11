import torch
import numpy as np
from typing import Any

class HDTransform():
    available_transforms_ = ['onlinehd', 'rffm', 'linear', 'cos', 'pow2', 'pow2phi', 'tanh', 'sin', 'weier', 'sign']
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seed: int = 0,
        batch_size: int = 4096,
        batch_norm: bool = False,
        device: str = 'cpu',
        transform_type: str = 'onlinehd',
        dtype: torch.dtype = torch.float32,):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.device = device
        self.transform_type = transform_type

        self._rng = np.random.default_rng(seed)
        self._G = torch.tensor(self._rng.normal(size=(in_channels, out_channels)), dtype=dtype, device=self.device)
        self._phi = torch.tensor(self._rng.uniform(0, 2*np.pi, size=out_channels), dtype=dtype, device=self.device)

    def _transform(self, x: torch.Tensor, projected: bool = False, **kwargs):
        buf1 = torch.mm(x, self._G) if not projected else x

        if self.transform_type == "onlinehd":
            buf2 = buf1 + self._phi
            output = buf1.sin_() * buf2.cos_()
        elif self.transform_type == "rffm":
            output = torch.cos(buf1 + self._phi)
        elif self.transform_type == "linear":
            output = buf1
        elif self.transform_type == "cos":
            output = torch.cos(buf1)
        elif self.transform_type == "pow2":
            output = 1 - torch.pow(buf1, 2) / 2
        elif self.transform_type == "pow2phi":
            output = 1 - torch.pow(buf1 + self._phi, 2) / 2
        elif self.transform_type == "tanh":
            output = torch.tanh(buf1)
        elif self.transform_type == "sin":
            output = torch.sin(buf1)
        elif self.transform_type == "weier":
            output = torch.zeros(*buf1.shape, dtype=torch.float32, device=self.device)

            for i in range(kwargs['n']):
                output += kwargs['b']**i * torch.cos(kwargs['a']**i * buf1)
        elif self.transform_type == "sign":
            output = torch.sign(buf1)
        return output
    
    def __call__(self, data: torch.Tensor, **kwargs):
        # Split data into batches and transform
        data_id = torch.arange(data.size(0), dtype=torch.int64)

        dataset = torch.utils.data.TensorDataset(data, data_id)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        x_new = torch.zeros(data.size(0), self.out_channels, dtype=self._G.dtype)

        for batch, batch_id in loader:
            batch = batch.to(self.device)
            eps = 1e-8 * torch.ones(batch.size(1), device=self.device, dtype=batch.dtype)

            # Don't forget to normalize data            
            batch = batch / torch.maximum(eps, torch.norm(batch, dim=1, keepdim=True))

            # This trick can really make difference 
            if self.batch_norm:
                eps = 1e-8 * torch.ones(batch.size(0), 1, device=self.device, dtype=batch.dtype)
                batch = batch / torch.maximum(eps, torch.norm(batch, dim=0))

            x_new[batch_id] = self._transform(batch, **kwargs).detach().cpu()
            batch.detach()

        x_new.detach()

        return x_new

    def to(self, device: str):
        self._G = self._G.to(device)
        self._phi = self._phi.to(device)
        self.device = device
        return self

    def detach(self):
        self._G.detach()
        self._phi.detach()
        return self
