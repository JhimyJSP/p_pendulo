"""Modelos de referencia y arquitecturas para campos hamiltonianos."""

import torch
from torch import nn


def perceptron(out_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, out_dim),
    )


class MLPField(nn.Module):
    """Aprende directamente (q, p) -> (q_dot, p_dot)."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = perceptron(2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def vector_field(self, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        return self.forward(x)


class _AutogradHamiltonianField(nn.Module):
    def vector_field(self, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        # Incluso en evaluacion se necesita autograd para derivar H respecto a q,p.
        # create_graph=True en entrenamiento permite derivar el error respecto a pesos.
        with torch.enable_grad():
            states = x.detach().requires_grad_(True)
            h = self.hamiltonian(states)
            d_h = torch.autograd.grad(h.sum(), states, create_graph=create_graph)[0]
            return torch.stack((d_h[:, 1], -d_h[:, 0]), dim=1)


class HNN(_AutogradHamiltonianField):
    """Aprende H_theta(q,p), derivando el campo mediante Hamilton."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = perceptron(1, hidden_dim)

    def hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_model(kind: str, hidden_dim: int) -> nn.Module:
    if kind == "mlp":
        return MLPField(hidden_dim)
    if kind == "hnn":
        return HNN(hidden_dim)
    raise ValueError(f"Modelo desconocido: {kind}")
