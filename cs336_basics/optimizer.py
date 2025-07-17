from typing import Iterable
import torch
import torch.nn as nn


class AdaGrad(torch.optim.Optimizer):
    
    def __init__(self, params: Iterable[nn.Parameter], lr : float=1e-2, ):
        super(AdaGrad,self).__init__(params,dict(lr=lr))
        
    def step(self):
        # todo
        pass
    
    def zero_grad(self):
        super().zero_grad()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-6, weight_decay: float = 0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        
        