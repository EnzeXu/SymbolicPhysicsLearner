import torch
from torch import Tensor, nn

class Loss:
    def __init__(self, reduction: str = 'mean', **kwargs) -> None:
        self.reduction = reduction
    
    def __call__(self, ode_func: nn.Module, X: Tensor, tspan: Tensor) -> Tensor:
        raise NotImplementedError