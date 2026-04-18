import torch

SMALL = 1e-6

# Denominator stabilization for :class:`OptimizerAdam`; matches PyTorch ``torch.optim.Adam`` default.
ADAM_EPS = 1e-8


class BaseOptimizer(torch.nn.Module):
    """Hand-crafted and learned optimizers share this interface.

    Neural optimizers accept optional ``loss`` and ``step`` so the meta-network can condition on objective value and time. Classical optimizers usually ignore those kwargs.
    """

    def __init__(self, dim):
        """Store parameter dimension for per-coordinate state tensors."""
        super().__init__()
        self.dim = dim

    def reset(self, device="cpu"):
        """Reset optimizer internal state on ``device`` before a new trajectory."""
        pass

    def forward(self, grad, loss=None, step=None, **kwargs):
        """Return additive update ``delta`` for current gradient ``grad``."""
        raise NotImplementedError
        
        
class OptimizerGD(BaseOptimizer):
    """Vanilla gradient descent with fixed scalar learning rate."""

    def __init__(self, dim, lr=0.001):
        """Initialize GD with parameter dimension and learning rate."""
        super().__init__(dim)
        self.lr = lr
        
    def forward(self, grad, loss=None, step=None, **kwargs):
        """Compute ``delta = -lr * grad``."""
        delta = - self.lr * grad
        return delta


class OptimizerAdaGrad(BaseOptimizer):
    """AdaGrad with diagonal second-moment accumulator."""

    def __init__(self, dim, lr=0.1, eps=1e-6):
        """Initialize hyperparameters and allocate accumulators via ``reset``."""
        super().__init__(dim)
        self.lr = lr
        self.eps = eps
        self.reset()
        
    def reset(self, device="cpu"):
        """Reset accumulated squared-gradient buffer."""
        grad_sqr = torch.full((self.dim,), SMALL, device=device)
        self.register_buffer("grad_sqr", grad_sqr)

    def forward_grad_sqr(self, grad):
        """Accumulate elementwise squared gradients."""
        self.grad_sqr = self.grad_sqr + grad**2
        return self.grad_sqr
        
    def forward(self, grad, loss=None, step=None, **kwargs):
        """Apply AdaGrad preconditioning and return additive update."""
        grad_sqr = self.forward_grad_sqr(grad)
        normalizer = torch.sqrt(grad_sqr + self.eps) + self.eps
        delta = - self.lr * grad / normalizer
        return delta


class OptimizerRMSprop(OptimizerAdaGrad):
    """RMSprop using exponential moving average of squared gradients."""

    def __init__(self, dim, lr=0.1, beta=0.99, eps=1e-6):
        """Initialize RMSprop hyperparameters and state."""
        super().__init__(dim)
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.reset()
        
    def reset(self, device="cpu"):
        """Reset EMA buffer for squared gradients."""
        grad_sqr = torch.full((self.dim,), SMALL, device=device)
        self.register_buffer("grad_sqr", grad_sqr)

    def forward_grad_sqr(self, grad):
        """Update EMA of squared gradients."""
        self.grad_sqr = self.beta * self.grad_sqr + (1. - self.beta) * grad**2
        return self.grad_sqr
        
    
    
class OptimizerAdam(BaseOptimizer):
    """Adam with bias-corrected first/second moments."""

    def __init__(self, dim, lr=0.1, beta1=0.9, beta2=0.999, eps=ADAM_EPS):
        """Initialize Adam hyperparameters and allocate optimizer state."""
        super().__init__(dim)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.iter = 1
        self.reset()
        
    def reset(self, device="cpu"):
        """Reset first/second moments and iteration counter."""
        grad = torch.full((self.dim,), SMALL, device=device)
        self.register_buffer("grad", grad)
        grad_sqr = torch.full((self.dim,), SMALL, device=device)
        self.register_buffer("grad_sqr", grad_sqr)
        self.iter = 1

    def forward_grad(self, grad):
        """Update first moment and return bias-corrected estimate."""
        self.grad = self.beta1 * self.grad + (1. - self.beta1) * grad
        return self.grad / (1. - self.beta1 ** self.iter)

    def forward_grad_sqr(self, grad):
        """Update second moment and return bias-corrected estimate."""
        self.grad_sqr = self.beta2 * self.grad_sqr + (1. - self.beta2) * grad**2
        return self.grad_sqr / (1. - self.beta2 ** self.iter)
        
    def forward(self, grad, loss=None, step=None, **kwargs):
        """Compute Adam update and advance internal step counter."""
        # Momentum
        grad = self.forward_grad(grad)
        # RMSprop
        grad_sqr = self.forward_grad_sqr(grad)
        normalizer = torch.sqrt(grad_sqr + self.eps) + self.eps
        delta = - self.lr * grad / normalizer
        self.iter += 1
        return delta