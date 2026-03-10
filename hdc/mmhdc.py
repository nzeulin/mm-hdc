import torch
from torch.nn.functional import relu
from . import _mmhdc_cpp

class MultiMMHDC(torch.nn.Module):
    def __init__(self, num_classes: int, 
                 out_channels: int, 
                 lr: float = 1e-2, 
                 C: float = 1.0, 
                 device: str = 'cpu',
                 backend: str = 'cpp'):
        
        super().__init__()
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.lr = lr
        self.device = device
        self.C = C
        self.backend = backend
        self.prototypes = torch.nn.parameter.Parameter(
            data=torch.zeros(num_classes, out_channels, dtype=torch.float32, device=device), 
            requires_grad=False
        )

    def forward(self, x: torch.Tensor):
        return torch.argmax(x @ self.prototypes.T, dim=1)
    
    # Initialization of prototypes
    def initialize(self, x: torch.Tensor, y: torch.Tensor):
        for i in range(self.num_classes):
            self.prototypes[i] = torch.mean(x[y.squeeze() == i], 0)

        # Normalizing prototypes
        eps = 1e-8 * torch.ones(self.prototypes.size(0), 1, device=self.prototypes.device)
        self.prototypes /= torch.maximum(eps, torch.norm(self.prototypes, dim=1, keepdim=True))

    def loss(self, X: torch.Tensor, y: torch.Tensor):
        loss = torch.norm(self.prototypes, dim=-1).sum() / (2 * self.C)
        for cls1 in torch.unique(y):
            loss += torch.sum(relu(2 - X[y == cls1] @ (self.prototypes[cls1] - self.prototypes).T))

        return loss
    
    def step(self, x: torch.Tensor, y: torch.Tensor):
        if self.backend == 'cpp':
            return _mmhdc_cpp.step(x, y, self.prototypes, self.lr, self.C)
        elif self.backend == 'python':
            return self._py_step(x, y)

    def _py_step(self, x: torch.Tensor, y: torch.Tensor):
        # Computing hinge loss
        prototypes_update = torch.zeros_like(self.prototypes)

        loss = self.loss(x, y)
        for cls in y.unique():
            rolled_prototypes = torch.roll(self.prototypes, -cls.item(), dims=0)
            x_cls = x[y == cls]

            dot = x_cls @ (rolled_prototypes[0] - rolled_prototypes[1:]).T
            hinge_loss = relu(2 - dot)

            exceeding_margin = hinge_loss > 0
            idx_wrong = exceeding_margin.any(-1)
            prototypes_update[cls] += x_cls[idx_wrong].sum(0)

            y_true_all = exceeding_margin.any(0).nonzero().flatten()
            for y_true in y_true_all:
                prototypes_update[(y_true + 1 + cls) % self.num_classes] -= x_cls[exceeding_margin[:, y_true]].sum(0)
        
        self.prototypes.data = (1 - self.lr / self.C) * self.prototypes.data + self.lr * prototypes_update

        # Normalizing prototypes
        # eps = 1e-8 * torch.ones(self.prototypes.size(0), 1, device=self.prototypes.device)
        # self.prototypes /= torch.maximum(eps, torch.norm(self.prototypes, dim=1, keepdim=True))
        # self.prototypes *= 10

        return loss
