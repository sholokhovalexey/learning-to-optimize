import math
import torch
import torch.nn as nn
from l2o.nn import ScalarLSTMCell
from optimization import BaseOptimizer, OptimizerAdam

from typing import Optional, Tuple

OPTIMIZER_REGISTRY: dict[str, type[nn.Module]] = {}


def register_optimizer(name: str):
    """Class decorator to register checkpoint ``architecture`` names.

    Args:
        name: Stable architecture id stored in checkpoint metadata.
    """

    def _decorator(cls: type[nn.Module]) -> type[nn.Module]:
        if name in OPTIMIZER_REGISTRY and OPTIMIZER_REGISTRY[name] is not cls:
            raise ValueError(f"Optimizer architecture {name!r} already registered")
        OPTIMIZER_REGISTRY[name] = cls
        return cls

    return _decorator


def log_encode(x, p=10.0):
    """Log-scale gradient preprocessing from classic L2O (stable magnitudes + sign).

    Returns shape ``(B, dim, 2)`` with ``[..., 0] = log|.|``, ``[..., 1] = sign`` so it can be
    stacked per coordinate (do **not** use ``cat`` then ``view(B, dim, 2)`` - that pairs wrong axes).
    """
    logabs = torch.log(torch.clamp(torch.abs(x), min=math.exp(-p))) / p
    sign = torch.clamp(x * math.exp(p), -1, 1)
    return torch.stack([logabs, sign], dim=-1)


@register_optimizer("neural_full")
class OptimizerNeural(BaseOptimizer):
    """Learned optimizer: full-vector LSTM with optional global context.
    """

    def __init__(self, dim: int, max_unroll_steps: float = 512.0):
        super().__init__(dim)
        self.max_unroll_steps = float(max_unroll_steps)
        # RNN sees [grad (dim), log(loss), step_norm] -> hidden dim
        self.rnn = nn.LSTMCell(dim + 2, dim, bias=True)
        self._init_neural_output_heads()

    def _init_neural_output_heads(self) -> None:
        """Register ``log_output_scale`` and ``hyper_mlp`` (shared by subclasses that skip this ``__init__``)."""
        # Positive step magnitude; initialized to 0 => exp(0)=1
        self.register_parameter("log_output_scale", nn.Parameter(torch.zeros(1)))
        # 4 stats -> scalar gate in (0.25, 2) after sigmoid in ``_hyper_multiplier``
        self.hyper_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def pre_process(self, x):
        return x

    def post_process(self, x):
        return x

    def reset(self, batch_size=None, device="cpu"):
        self.hx = None

    def _log_loss_step_tensors(
        self,
        loss: Optional[torch.Tensor],
        step: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scalar log-loss and normalized step per batch row (zeros if missing)."""
        if loss is None:
            log_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)
        else:
            if not isinstance(loss, torch.Tensor):
                loss = torch.as_tensor(loss, device=device, dtype=torch.float32)
            if loss.dim() == 0:
                loss = loss.expand(batch_size)
            log_loss = torch.log(loss.clamp(min=1e-12))
        if step is None:
            step_norm = torch.zeros(batch_size, device=device, dtype=torch.float32)
        else:
            if not isinstance(step, torch.Tensor):
                step = torch.full((), float(step), device=device, dtype=torch.float32)
            if step.dim() == 0:
                step = step.expand(batch_size)
            step_norm = step.float() / self.max_unroll_steps
        return log_loss, step_norm

    def _broadcast_global_cols(
        self,
        log_loss: torch.Tensor,
        step_norm: torch.Tensor,
        batch_size: int,
        dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Broadcast global scalars to each coordinate row for flat RNN batches."""
        ll = log_loss.view(batch_size, 1).expand(batch_size, dim).reshape(-1, 1)
        sn = step_norm.view(batch_size, 1).expand(batch_size, dim).reshape(-1, 1)
        return ll, sn

    def _hyper_multiplier(
        self,
        grad: torch.Tensor,
        loss: Optional[torch.Tensor],
        step: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Per-batch positive multiplier from (mean/std grad, log-loss, step)."""
        batch_size, dim = grad.shape
        device = grad.device
        log_loss, step_norm = self._log_loss_step_tensors(loss, step, batch_size, device)
        mean_g = grad.mean(dim=1)
        std_g = grad.std(dim=1, unbiased=False)
        inp = torch.stack([mean_g, std_g, log_loss, step_norm], dim=1)
        h = self.hyper_mlp(inp)
        # Keep multiplier in a reasonable band so training does not collapse
        return 0.25 + 1.75 * torch.sigmoid(h)

    def _finalize_delta(
        self,
        raw: torch.Tensor,
        grad: torch.Tensor,
        loss: Optional[torch.Tensor],
        step: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bounded = torch.tanh(raw)
        mag = torch.exp(self.log_output_scale)
        delta = mag * bounded
        mult = self._hyper_multiplier(grad, loss, step)
        return delta * mult

    def forward(
        self,
        grad: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
        step: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, dim = grad.shape
        assert self.dim == dim
        device = grad.device
        log_loss, step_norm = self._log_loss_step_tensors(loss, step, batch_size, device)
        # Two **scalar** channels appended to the full gradient vector: shape (batch, dim+2)
        g1 = log_loss.view(batch_size, 1)
        g2 = step_norm.view(batch_size, 1)
        inp = torch.cat([self.pre_process(grad), g1, g2], dim=-1)
        hx = self.rnn(inp, self.hx)
        self.hx = hx
        output = hx[0]
        raw = self.post_process(output)
        return self._finalize_delta(raw, grad, loss, step)


@register_optimizer("scalar_lstm")
class OptimizerNeuralScalarLSTM(OptimizerNeural):
    """Per-dimension tied LSTM (ScalarLSTM) with ``dim + 2`` input channels."""

    def __init__(self, dim: int, max_unroll_steps: float = 512.0):
        # Skip OptimizerNeural.__init__ rnn creation - replace cell type
        BaseOptimizer.__init__(self, dim)
        self.max_unroll_steps = float(max_unroll_steps)
        self._init_neural_output_heads()
        self.rnn = ScalarLSTMCell(dim + 2, bias=True)

    # Inherits forward from OptimizerNeural (same concat layout)


@register_optimizer("coord")
class OptimizerNeuralCoordinatewise(OptimizerNeural):
    """Coordinate-wise stacked LSTMs: shared weights, per-coordinate hidden state.

    Each scalar parameter gets the same MLP, with **two extra input channels** per step:
    log-loss and normalized step index (broadcast to all coordinates).
    """

    def __init__(self, dim: int, hidden_size: int, max_unroll_steps: float = 512.0):
        # Do not call OptimizerNeural.__init__ (different RNN stack)
        BaseOptimizer.__init__(self, dim)
        self.hidden_size = hidden_size
        self.max_unroll_steps = float(max_unroll_steps)
        self._init_neural_output_heads()
        # Per-coordinate input: value + log-loss + step_norm  -> 3
        self.rnn1 = nn.LSTMCell(3, hidden_size, bias=True)
        self.rnn2 = nn.LSTMCell(hidden_size, hidden_size, bias=True)
        self.linear_out = nn.Linear(hidden_size, 1)

    def reset(self, batch_size=None, device="cpu"):
        if batch_size is None:
            self.hx1 = None
            self.hx2 = None
        else:
            h_0 = torch.zeros(batch_size * self.dim, self.hidden_size).to(device)
            c_0 = torch.zeros(batch_size * self.dim, self.hidden_size).to(device)
            self.hx1 = (h_0, c_0)
            h_0 = torch.zeros(batch_size * self.dim, self.hidden_size).to(device)
            c_0 = torch.zeros(batch_size * self.dim, self.hidden_size).to(device)
            self.hx2 = (h_0, c_0)

    def pre_process(self, x):
        return x

    def post_process(self, x):
        return x

    def _forward_coordinatewise(
        self,
        grad: torch.Tensor,
        per_coord: torch.Tensor,
        loss: Optional[torch.Tensor],
        step: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """``per_coord`` is (batch, dim, k) with k channels per coordinate before globals."""
        batch_size, dim, k = per_coord.shape
        assert self.dim == dim
        if self.hx1 is None:
            self.reset(batch_size, grad.device)
        device = grad.device
        log_loss, step_norm = self._log_loss_step_tensors(loss, step, batch_size, device)
        ll, sn = self._broadcast_global_cols(log_loss, step_norm, batch_size, dim)
        flat = per_coord.reshape(batch_size * dim, k)
        inp = torch.cat([flat, ll, sn], dim=-1)

        hx1 = self.rnn1(inp, self.hx1)
        self.hx1 = hx1
        hx2 = self.rnn2(hx1[0], self.hx2)
        self.hx2 = hx2
        output_flat = self.linear_out(hx2[0])
        raw = output_flat.view(batch_size, dim)
        raw = self.post_process(raw)
        return self._finalize_delta(raw, grad, loss, step)

    def forward(
        self,
        grad: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
        step: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, dim = grad.shape
        per_coord = grad.unsqueeze(-1)
        return self._forward_coordinatewise(grad, per_coord, loss, step)


@register_optimizer("gradenc")
class OptimizerNeuralCoordinatewiseGradEnc(OptimizerNeuralCoordinatewise):
    """Coordinate-wise net with ``log_encode(grad)`` plus global loss/step channels."""

    def __init__(self, dim: int, hidden_size: int, max_unroll_steps: float = 512.0):
        BaseOptimizer.__init__(self, dim)
        self.hidden_size = hidden_size
        self.max_unroll_steps = float(max_unroll_steps)
        self._init_neural_output_heads()
        # 2 (log_encode) + 2 global = 4
        self.rnn1 = nn.LSTMCell(4, hidden_size, bias=True)
        self.rnn2 = nn.LSTMCell(hidden_size, hidden_size, bias=True)
        self.linear_out = nn.Linear(hidden_size, 1)

    def pre_process(self, x):
        return log_encode(x)

    def forward(
        self,
        grad: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
        step: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, dim = grad.shape
        per_coord = self.pre_process(grad)
        if per_coord.shape != (batch_size, dim, 2):
            raise RuntimeError(f"Expected per_coord (B,D,2), got {per_coord.shape} for grad {grad.shape}")
        return self._forward_coordinatewise(grad, per_coord, loss, step)


@register_optimizer("gradenc_deep")
class OptimizerNeuralCoordinatewiseGradEncDeep(OptimizerNeuralCoordinatewiseGradEnc):
    """A deeper version with **three** stacked LSTM layers."""

    def __init__(self, dim: int, hidden_size: int, max_unroll_steps: float = 512.0):
        BaseOptimizer.__init__(self, dim)
        self.hidden_size = hidden_size
        self.max_unroll_steps = float(max_unroll_steps)
        self._init_neural_output_heads()
        self.rnn1 = nn.LSTMCell(4, hidden_size, bias=True)
        self.rnn2 = nn.LSTMCell(hidden_size, hidden_size, bias=True)
        self.rnn3 = nn.LSTMCell(hidden_size, hidden_size, bias=True)
        self.linear_out = nn.Linear(hidden_size, 1)

    def reset(self, batch_size=None, device="cpu"):
        if batch_size is None:
            self.hx1 = None
            self.hx2 = None
            self.hx3 = None
        else:
            z = torch.zeros(batch_size * self.dim, self.hidden_size).to(device)
            c = torch.zeros(batch_size * self.dim, self.hidden_size).to(device)
            self.hx1 = (z, c)
            self.hx2 = (z.clone(), c.clone())
            self.hx3 = (z.clone(), c.clone())

    def _forward_coordinatewise(
        self,
        grad: torch.Tensor,
        per_coord: torch.Tensor,
        loss: Optional[torch.Tensor],
        step: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, dim, k = per_coord.shape
        assert self.dim == dim
        if self.hx1 is None:
            self.reset(batch_size, grad.device)
        device = grad.device
        log_loss, step_norm = self._log_loss_step_tensors(loss, step, batch_size, device)
        ll, sn = self._broadcast_global_cols(log_loss, step_norm, batch_size, dim)
        flat = per_coord.reshape(batch_size * dim, k)
        inp = torch.cat([flat, ll, sn], dim=-1)

        hx1 = self.rnn1(inp, self.hx1)
        self.hx1 = hx1
        hx2 = self.rnn2(hx1[0], self.hx2)
        self.hx2 = hx2
        hx3 = self.rnn3(hx2[0], self.hx3)
        self.hx3 = hx3
        output_flat = self.linear_out(hx3[0])
        raw = output_flat.view(batch_size, dim)
        raw = self.post_process(raw)
        return self._finalize_delta(raw, grad, loss, step)


@register_optimizer("adamenc")
class OptimizerNeuralCoordinatewiseAdam(OptimizerNeuralCoordinatewise, OptimizerAdam):
    """Momentum + RMS-style 2-D features per coordinate, plus global loss/step (4-D RNN input)."""

    def __init__(
        self,
        dim: int,
        hidden_size: int,
        lr: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-6,
        max_unroll_steps: float = 512.0,
    ):
        # ``OptimizerAdam`` already calls ``BaseOptimizer.__init__``; do not call it twice.
        OptimizerAdam.__init__(self, dim, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        self.hidden_size = hidden_size
        self.max_unroll_steps = float(max_unroll_steps)
        self._init_neural_output_heads()
        self.rnn1 = nn.LSTMCell(4, hidden_size, bias=True)
        self.rnn2 = nn.LSTMCell(hidden_size, hidden_size, bias=True)
        self.linear_out = nn.Linear(hidden_size, 1)

    def reset(self, batch_size=None, device="cpu"):
        if (
            isinstance(device, str)
            and device == "cpu"
            and hasattr(self, "linear_out")
            and self.linear_out.weight.is_cuda
        ):
            device = self.linear_out.weight.device
        OptimizerAdam.reset(self, device)
        OptimizerNeuralCoordinatewise.reset(self, batch_size, device)

    def pre_process(self, grad):
        grad_sqr = self.forward_grad_sqr(grad)
        normalizer = torch.sqrt(grad_sqr + self.eps) + self.eps
        a = grad / normalizer
        b = self.forward_grad(grad) / normalizer
        return torch.stack([a, b], dim=-1)

    def forward(
        self,
        grad: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
        step: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, dim = grad.shape
        per_coord = self.pre_process(grad)
        if per_coord.shape != (batch_size, dim, 2):
            raise RuntimeError(f"Expected per_coord (B,D,2), got {per_coord.shape} for grad {grad.shape}")
        out = self._forward_coordinatewise(grad, per_coord, loss, step)
        # Match ``OptimizerAdam.forward``: bias-correction uses ``iter`` incremented once per step.
        self.iter += 1
        return out
