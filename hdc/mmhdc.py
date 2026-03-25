import torch
from torch.nn.functional import relu
from . import _mmhdc_cpp

class MultiMMHDC(torch.nn.Module):
    def __init__(self, num_classes: int, 
                 out_channels: int, 
                 lr: float = 1e-2, 
                 C: float = 1.0, 
                 device: str = 'cpu',
                 backend: str = 'cpp',
                 dtype: torch.dtype = torch.float32):
        
        super().__init__()
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.lr = lr
        self.device = device
        self.C = C
        self.backend = backend
        self.prototypes = torch.nn.parameter.Parameter(
            data=torch.zeros(num_classes, out_channels, dtype=dtype, device=device), 
            requires_grad=False
        )
        self.dtype = dtype

    def forward(self, x: torch.Tensor):
        return torch.argmax(x @ self.prototypes.T, dim=1)
    
    # Initialization of prototypes
    def initialize(self, x: torch.Tensor, y: torch.Tensor):
        for i in range(self.num_classes):
            self.prototypes[i] = torch.mean(x[y.squeeze() == i], 0).to(self.device)

        # Normalizing prototypes
        eps = 1e-8 * torch.ones(self.prototypes.size(0), 1, device=self.prototypes.device, dtype=self.dtype)
        self.prototypes /= torch.maximum(eps, torch.norm(self.prototypes, dim=1, keepdim=True))

    def loss(self, X: torch.Tensor, y: torch.Tensor):
        loss = torch.pow(torch.norm(self.prototypes, dim=-1), 2).sum() / (2 * self.C)
        for cls1 in torch.unique(y):
            loss += torch.sum(relu(2 - X[y == cls1] @ (self.prototypes[cls1] - self.prototypes).T))

        return loss
    
    def step(self, x: torch.Tensor, y: torch.Tensor):
        if self.backend == 'cpp':
            self.prototypes.data = _mmhdc_cpp.step(x, y, self.prototypes, self.lr, self.C)
        elif self.backend == 'python':
            return self._py_step(x, y)

    def _py_step(self, x: torch.Tensor, y: torch.Tensor):
        # TODO: Revise implementation to avoid nested for-loops (similar to C++). Can be used for demonstration purposes, 
        # but not suitable for actual training.
        if self.dtype not in [torch.float32, torch.float64]:
            raise TypeError("MultiMMHDC expects floating-point dtype. Use MultiMMHDCInt for integer training.")

        # Computing hinge loss
        prototypes_update = torch.zeros_like(self.prototypes, dtype=self.dtype)
        loss = self.loss(x, y)
        for cls in y.unique():
            rolled_prototypes = torch.roll(self.prototypes, -cls.item(), dims=0)
            x_cls = x[y == cls]

            dot = x_cls @ (rolled_prototypes[0] - rolled_prototypes[1:]).T
            hinge_loss = relu(2 - dot)

            exceeding_margin = hinge_loss > 0
            num_violations = exceeding_margin.sum(dim=1, dtype=x_cls.dtype)
            prototypes_update[cls] += (x_cls * num_violations.unsqueeze(1)).sum(0)

            y_true_all = exceeding_margin.any(0).nonzero().flatten()
            for y_true in y_true_all:
                prototypes_update[(y_true + 1 + cls) % self.num_classes] -= x_cls[exceeding_margin[:, y_true]].sum(0)
        
        self.prototypes.data = (1 - self.lr / self.C) * self.prototypes.data + self.lr * prototypes_update

        return loss


class MultiMMHDCInt(MultiMMHDC):
    def __init__(self,
                 num_classes: int,
                 out_channels: int,
                 lr: float = 1e-2,
                 C: float = 1.0,
                 device: str = 'cpu',
                 backend: str = 'python',
                 dtype: torch.dtype = torch.int32,
                 hv_bitwidth: int = 4,
                 fixed_point_frac_bits: int = 16):

        super().__init__(
            num_classes=num_classes,
            out_channels=out_channels,
            lr=lr,
            C=C,
            device=device,
            backend=backend,
            dtype=dtype,
        )

        if self.backend not in ['python', 'cpp']:
            raise ValueError("MultiMMHDCInt currently supports backend in {'python', 'cpp'} only.")
        if self.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64]:
            raise TypeError("MultiMMHDCInt expects integer dtype.")
        if hv_bitwidth < 2:
            raise ValueError("hv_bitwidth must be >= 2 for signed symmetric quantization")

        self.hv_bitwidth = hv_bitwidth
        self.fixed_point_frac_bits = fixed_point_frac_bits
        self._prototypes_fp = None

    @staticmethod
    def _int_clip_bounds(dtype: torch.dtype):
        info = torch.iinfo(dtype)
        return info.min, info.max

    @staticmethod
    def _round_divide_int(numer: torch.Tensor, denom: int):
        # Deterministic round-to-nearest for signed integer tensors.
        abs_q = (numer.abs() + denom // 2) // denom
        return abs_q * numer.sign()

    def quantize_hypervectors(self, x: torch.Tensor):
        scale = (1 << (self.hv_bitwidth - 1)) - 1
        qmin = -scale
        qmax = scale

        if x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            x_q = torch.clamp(x, qmin, qmax)
        else:
            x_q = torch.round(x * scale)
            x_q = torch.clamp(x_q, qmin, qmax)

        return x_q.to(self.dtype)

    def dequantize_prototypes(self):
        hv_scale = (1 << (self.hv_bitwidth - 1)) - 1
        return self.prototypes.to(torch.float32) / float(hv_scale)

    def loss_dequantized(self, x_float: torch.Tensor, y: torch.Tensor):
        """
        Loss for integer-trained model computed with:
          1) dequantized integer prototypes, and
          2) original floating-point hypervectors.
        """
        x_dev = x_float.to(self.device, dtype=torch.float32)
        y_dev = y.to(self.device)
        p_float = self.dequantize_prototypes().to(self.device)

        loss = torch.pow(torch.norm(p_float, dim=-1), 2).sum() / (2 * self.C)
        for cls1 in torch.unique(y_dev):
            loss += torch.sum(relu(2 - x_dev[y_dev == cls1] @ (p_float[cls1] - p_float).T))

        return loss

    def initialize(self, x: torch.Tensor, y: torch.Tensor):
        if x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            x_q = x.to(self.device, dtype=self.dtype)
        else:
            x_q = self.quantize_hypervectors(x.to(self.device))
        y_dev = y.to(self.device)

        p_min, p_max = self._int_clip_bounds(self.dtype)
        for i in range(self.num_classes):
            x_i = x_q[y_dev.squeeze() == i]
            if x_i.numel() == 0:
                continue

            # Option B: integer sum with rounded integer division.
            sum_i = x_i.to(torch.int64).sum(dim=0)
            count_i = x_i.size(0)
            mean_i = self._round_divide_int(sum_i, count_i)
            mean_i = torch.clamp(mean_i, p_min, p_max)
            self.prototypes[i] = mean_i.to(self.dtype)

        fp_scale = 1 << self.fixed_point_frac_bits
        self._prototypes_fp = self.prototypes.data.to(torch.int64) * fp_scale

    def step(self, x: torch.Tensor, y: torch.Tensor):
        if self.backend == 'python':
            return self._py_step(x, y)
        if self.backend == 'cpp':
            return self._cpp_step(x, y)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def _prepare_int_inputs(self, x: torch.Tensor, y: torch.Tensor):
        x_orig_float = x if x.dtype in [torch.float32, torch.float64] else None
        if x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            x_q = x.to(self.device, dtype=self.dtype)
        else:
            # Fallback for convenience; recommended usage quantizes once at input creation time.
            x_q = self.quantize_hypervectors(x.to(self.device))
        y_dev = y.to(self.device)

        if self._prototypes_fp is None:
            fp_scale = 1 << self.fixed_point_frac_bits
            self._prototypes_fp = self.prototypes.data.to(torch.int64) * fp_scale

        return x_orig_float, x_q, y_dev

    def _py_step(self, x: torch.Tensor, y: torch.Tensor):
        x_orig_float, x_q, y_dev = self._prepare_int_inputs(x, y)

        x_acc = x_q.to(torch.int64)
        fp_scale = 1 << self.fixed_point_frac_bits

        if self._prototypes_fp is None:
            self._prototypes_fp = self.prototypes.data.to(torch.int64) * fp_scale

        p_acc = self._round_divide_int(self._prototypes_fp, fp_scale)

        scores = x_acc @ p_acc.T
        correct_scores = scores.gather(1, y_dev.unsqueeze(1))

        margin = 2
        violated = (correct_scores - scores) < margin
        violated.scatter_(1, y_dev.unsqueeze(1), torch.zeros_like(y_dev.unsqueeze(1), dtype=torch.bool))

        W = -violated.to(torch.int64)
        W.scatter_add_(1, y_dev.unsqueeze(1), violated.sum(dim=1, keepdim=True).to(torch.int64))
        prototypes_update = W.T @ x_acc

        decay = 1.0 - self.lr / self.C
        decay_q = int(round(decay * fp_scale))
        lr_q = int(round(self.lr * fp_scale))

        # Keep state in fixed-point so sub-integer updates accumulate over steps.
        self._prototypes_fp = (
            self._round_divide_int(decay_q * self._prototypes_fp, fp_scale)
            + lr_q * prototypes_update
        )
        updated = self._round_divide_int(self._prototypes_fp, fp_scale)

        p_min, p_max = self._int_clip_bounds(self.dtype)
        self.prototypes.data = torch.clamp(updated, p_min, p_max).to(self.dtype)

        if x_orig_float is not None:
            return self.loss_dequantized(x_orig_float, y_dev)

        hv_scale = (1 << (self.hv_bitwidth - 1)) - 1
        return self.loss_dequantized(x_q.to(torch.float32) / float(hv_scale), y_dev)

    def _cpp_step(self, x: torch.Tensor, y: torch.Tensor):
        x_orig_float, x_q, y_dev = self._prepare_int_inputs(x, y)
        p_min, p_max = self._int_clip_bounds(self.dtype)

        updated, updated_fp = _mmhdc_cpp.step_int(
            x_q,
            y_dev,
            self.prototypes.data,
            self._prototypes_fp,
            self.lr,
            self.C,
            self.fixed_point_frac_bits,
            p_min,
            p_max,
        )

        self.prototypes.data = updated.to(self.dtype)
        self._prototypes_fp = updated_fp.to(torch.int64)

        if x_orig_float is not None:
            return self.loss_dequantized(x_orig_float, y_dev)

        hv_scale = (1 << (self.hv_bitwidth - 1)) - 1
        return self.loss_dequantized(x_q.to(torch.float32) / float(hv_scale), y_dev)