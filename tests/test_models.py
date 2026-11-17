import numpy as np
import pytest
import torch
from torch import nn

from pendulum_project.models import HNN, MLPField, make_model
from pendulum_project.training import field_loss


class ExactEnergy(nn.Module):
    def forward(self, x):
        return (0.5 * x[:, 1] ** 2 + 9.81 * (1 - torch.cos(x[:, 0]))).unsqueeze(-1)


def test_hnn_recovers_analytical_field():
    model = HNN()
    model.net = ExactEnergy()
    x = torch.tensor([[0.6, 1.2], [-0.4, -1.3]], dtype=torch.float32)
    prediction = model.vector_field(x)
    true = torch.stack((x[:, 1], -9.81 * torch.sin(x[:, 0])), dim=1)
    torch.testing.assert_close(prediction, true)


def test_hnn_second_derivative_trains_weights():
    torch.manual_seed(0)
    hnn = HNN(8)
    inputs = torch.randn((10, 2))
    target = torch.stack((inputs[:, 1], -9.81 * torch.sin(inputs[:, 0])), dim=1)
    loss = field_loss(hnn.vector_field(inputs, create_graph=True), target, torch.ones(2))
    loss.backward()
    # La constante aditiva de H no cambia dH/dq ni dH/dp: su sesgo final no tiene gradiente.
    grads = [p.grad for name, p in hnn.named_parameters() if name != "net.4.bias"]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert hnn.net[-1].bias.grad is None
    assert np.isfinite(loss.item())


def test_mlp_output_shape():
    assert MLPField(16)(torch.ones((5, 2))).shape == (5, 2)


@pytest.mark.parametrize("kind", ("mlp", "hnn"))
def test_model_factory_supports_only_study_models(kind):
    assert isinstance(make_model(kind, 16), {"mlp": MLPField, "hnn": HNN}[kind])


def test_model_factory_rejects_unregistered_model():
    with pytest.raises(ValueError, match="Modelo desconocido"):
        make_model("unknown", 16)
