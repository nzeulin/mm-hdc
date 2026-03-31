import torch
from torch.nn.functional import one_hot, relu
from ._cpp_backend import get_mmhdc_cpp

class MultiMMHDC(torch.nn.Module):
    def __init__(self, num_classes: int, 
                 out_channels: int, 
                 lr: float = 1e-2, 
                 C: float = 1.0, 
                 device: str = 'cpu',
                 backend: str = 'python',
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
    
    @torch.no_grad()
    def initialize(self, x: torch.Tensor, y: torch.Tensor):
        y = y.reshape(-1).to(dtype=torch.int64, device=x.device)
        x = x.to(dtype=self.prototypes.dtype)

        prototype_sums = torch.zeros(
            self.num_classes,
            self.out_channels,
            dtype=self.prototypes.dtype,
            device=x.device,
        )
        prototype_sums.index_add_(0, y, x)

        class_counts = torch.bincount(y, minlength=self.num_classes)
        class_counts = class_counts.unsqueeze(1).to(dtype=self.prototypes.dtype, device=x.device)

        prototypes = prototype_sums / class_counts.clamp_min_(1)
        prototype_norms = torch.norm(prototypes, dim=1, keepdim=True)
        prototypes = prototypes / prototype_norms.clamp_min_(1e-8)

        self.prototypes.copy_(prototypes.to(device=self.prototypes.device))

    def loss(self, X: torch.Tensor, y: torch.Tensor):
        y = y.reshape(-1)
        scores = X @ self.prototypes.T
        correct_scores = scores.gather(1, y.unsqueeze(1))
        margins = relu(2 - (correct_scores - scores))
        true_class_mask = one_hot(y, num_classes=self.num_classes).to(dtype=torch.bool)
        margins = margins.masked_fill(true_class_mask, 0.0)

        regularizer = self.prototypes.square().sum() / (2 * self.C)
        return regularizer + margins.sum()
        
    def step(self, x: torch.Tensor, y: torch.Tensor):
        if self.backend == 'cpp':
            with torch.no_grad():
                mmhdc_cpp = get_mmhdc_cpp()
                updated_prototypes = mmhdc_cpp.step(x, y, self.prototypes, self.lr, self.C)
                self.prototypes.copy_(updated_prototypes)
        elif self.backend == 'python':
            self._py_step(x, y)
        else:
            raise ValueError(f"Unsupported backend '{self.backend}'. Expected 'cpp' or 'python'.")

    @torch.no_grad()
    def _py_step(self, x: torch.Tensor, y: torch.Tensor, optimized=True):
        # This implementation is for reference purpose only. Use matrix-based optimized C++ and Python implementations
        def _py_step_reference(x: torch.Tensor, y: torch.Tensor):
            prototypes_update = torch.zeros_like(self.prototypes, dtype=self.dtype)
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
        
        # Optimized code based on the C++ implementation logic
        def _py_step_optimized(x: torch.Tensor, y: torch.Tensor):
            scores = x @ self.prototypes.T
            correct_scores = scores.gather(1, y.unsqueeze(1))

            violated = (correct_scores - scores) < 2
            violated.scatter_(1, y.unsqueeze(1), False)

            W = -violated.to(x.dtype)
            W.scatter_add_(1, y.unsqueeze(1), violated.sum(dim=1, keepdim=True).to(x.dtype))

            prototypes_update = W.T @ x
            self.prototypes.data = (1 - self.lr / self.C) * self.prototypes.data + self.lr * prototypes_update

        _py_step_reference(x, y) if not optimized else _py_step_optimized(x, y)
