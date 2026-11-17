"""Entrenamiento determinista, perdida balanceada, early stopping y checkpoints."""

import copy
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .data import INPUTS, TARGETS
from .models import make_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA no esta disponible en este equipo")
    return torch.device(name)


class BatchedTensorDataset(TensorDataset):
    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, ...]:
        # Una lectura de tensor por lote. DataLoader conserva el muestreo original.
        return tuple(tensor[indices] for tensor in self.tensors)


def to_dataset(df: pd.DataFrame) -> BatchedTensorDataset:
    x = torch.from_numpy(df[INPUTS].to_numpy(dtype=np.float32, copy=True))
    y = torch.from_numpy(df[TARGETS].to_numpy(dtype=np.float32, copy=True))
    return BatchedTensorDataset(x, y)


def prebatched(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return batch


def field_loss(prediction: torch.Tensor, target: torch.Tensor,
               scale: torch.Tensor) -> torch.Tensor:
    """MSE de derivadas divididas por la desviacion estandar del ajuste."""
    return torch.mean(((prediction - target) / scale)**2)


def validation_loss(model, loader, scale: torch.Tensor, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # MLP no necesita gradientes, HNN necesita gradientes respecto a las ENTRADAS.
        prediction = model.vector_field(x, create_graph=False)
        loss = field_loss(prediction, y, scale)
        total += loss.item() * len(x)
        count += len(x)
    return total / count


def _validation_decisions(loss: float, best_loss: float,
                          patience_reference: float,
                          min_delta: float) -> tuple[bool, bool]:
    """Separa guardar el minimo de exigir una mejora relevante para la paciencia."""
    return loss < best_loss, loss < patience_reference - min_delta


def train_one(kind: str, fit: pd.DataFrame, val: pd.DataFrame,
              config: Config, device: torch.device):
    """Devuelve modelo con los mejores pesos, historial, metadatos y escalas."""
    set_seed(config.seed)
    model = make_model(kind, config.hidden_dim).to(device)
    fit_data, val_data = to_dataset(fit), to_dataset(val)
    targets = fit[TARGETS].to_numpy(dtype=np.float32)
    scale = torch.as_tensor(targets.std(axis=0).clip(min=1e-8), device=device)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(fit_data, batch_size=config.batch_size, shuffle=True,
                        generator=generator, num_workers=0, collate_fn=prebatched)
    validation = DataLoader(val_data, batch_size=4096, shuffle=False, collate_fn=prebatched)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_loss, patience_reference = float("inf"), float("inf")
    best_epoch, stale = 0, 0
    history: list[dict] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        sum_loss, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model.vector_field(x, create_graph=True)
            loss = field_loss(pred, y, scale)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * len(x)
            n += len(x)
        avg_train = sum_loss / n
        avg_val = validation_loss(model, validation, scale, device)
        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})
        # El checkpoint siempre corresponde al minimo observado. min_delta solo
        # controla cuando se reinicia la paciencia del early stopping.
        save_checkpoint, reset_patience = _validation_decisions(
            avg_val, best_loss, patience_reference, config.min_delta)
        if save_checkpoint:
            best_loss = avg_val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if reset_patience:
            patience_reference = avg_val
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % config.print_every == 0 or epoch == config.max_epochs:
            print(f"{kind.upper()} | epoca {epoch:4d} | fit={avg_train:.5e} | val={avg_val:.5e}", flush=True)
        if stale >= config.patience:
            print(f"{kind.upper()} | early stopping; mejor epoca {best_epoch}", flush=True)
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), {
        "best_epoch": best_epoch, "best_val_loss": best_loss,
        "epochs_completed": len(history), "optimizer": "adam",
    }, scale.detach().cpu().tolist()


def predict_field(model, states: np.ndarray, device: torch.device,
                  batch_size: int = 4096) -> np.ndarray:
    """Predice en minibatches: evita cargar 40 200 estados a la vez en GPU."""
    model.eval()
    z = np.asarray(states, dtype=np.float64)
    flat = z.reshape(-1, 2)
    chunks = []
    for start in range(0, len(flat), batch_size):
        array = np.array(flat[start:start + batch_size], dtype=np.float32, copy=True)
        x = torch.as_tensor(array, device=device)
        field = model.vector_field(x, create_graph=False)
        chunks.append(field.detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64).reshape(z.shape)


def predict_hamiltonian(model, states: np.ndarray, device: torch.device,
                        batch_size: int = 4096) -> np.ndarray:
    """Energia aprendida por HNN, distinta de la energia fisica verdadera."""
    z = np.asarray(states, dtype=np.float64)
    chunks = []
    model.eval()
    for part in np.array_split(z.reshape(-1, 2), max(1, (z.size // 2 + batch_size - 1) // batch_size)):
        with torch.no_grad():
            x = torch.as_tensor(part, dtype=torch.float32, device=device)
            chunks.append(model.hamiltonian(x).squeeze(-1).cpu().numpy())
    return np.concatenate(chunks).reshape(z.shape[:-1]).astype(np.float64)
