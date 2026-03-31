import torch
import math
from typing import Optional

class HDTransform(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seed: int = 0,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.normalize = normalize
        self.dtype = dtype

        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)
        two_pi = torch.tensor(2.0 * math.pi, dtype=self.dtype)
        self.register_buffer(
            "_eps",
            torch.tensor(1e-8, dtype=self.dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_G",
            torch.randn(in_channels, out_channels, generator=self._rng, dtype=self.dtype).to(device),
            persistent=False,
        )
        self.register_buffer(
            "_phi",
            torch.rand(out_channels, generator=self._rng, dtype=self.dtype).mul(two_pi).to(device),
            persistent=False,
        )

    def _transform(self, x: torch.Tensor):
        projection_matrix = self._G.unsqueeze(0).expand(x.size(0), -1, -1)

        # OnlineHD feature mapping
        buf1 = torch.bmm(x.unsqueeze(1), projection_matrix).squeeze(1)
        buf2 = buf1 + self._phi
        return buf1.sin_() * buf2.cos_()

    def forward(self, data: torch.Tensor):
        with torch.no_grad():
            data = data.to(device=self._G.device, dtype=self.dtype)
            if self.normalize:
                norms = torch.norm(data, dim=1, keepdim=True)
                data = data / torch.maximum(norms, self._eps)

            if self.batch_size is None or self.batch_size >= data.size(0):
                x_new = self._transform(data)
                return x_new.cpu() if x_new.device.type != 'cpu' else x_new

            x_new = torch.empty(data.size(0), self.out_channels, dtype=self.dtype)
            for start in range(0, data.size(0), self.batch_size):
                end = start + self.batch_size
                batch = self._transform(data[start:end])
                x_new[start:end].copy_(batch.cpu() if batch.device.type != 'cpu' else batch)

            return x_new
