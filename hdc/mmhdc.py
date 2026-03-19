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
        def _initialize_float(x: torch.Tensor, y: torch.Tensor):
            for i in range(self.num_classes):
                self.prototypes[i] = torch.mean(x[y.squeeze() == i], 0).to(self.device)

            # Normalizing prototypes
            eps = 1e-8 * torch.ones(self.prototypes.size(0), 1, device=self.prototypes.device, dtype=self.dtype)
            self.prototypes /= torch.maximum(eps, torch.norm(self.prototypes, dim=1, keepdim=True))

        def _initialize_int(x: torch.Tensor, y: torch.Tensor):
            pass

        if self.dtype in [torch.float32, torch.float64]:
            _initialize_float(x, y)
        elif self.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            _initialize_int(x, y)

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
        # Step procedure for floating-point data types
        # TODO: Revise implementation to avoid nested for-loops (similar to C++). Can be used for demonstration purposes, 
        # but not suitable for actual training.
        def _py_step_float(x: torch.Tensor, y: torch.Tensor):
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

        # Step procedure for integer data types
        def _py_step_int(x: torch.Tensor, y: torch.Tensor):
            pass

        if self.dtype in [torch.float32, torch.float64]:
            return _py_step_float(x, y)
        elif self.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            return _py_step_int(x, y)