"""Parametros del experimento y directorios por defecto."""

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    g: float = 9.81
    mass: float = 1.0
    length: float = 1.0
    dt: float = 0.02
    n_steps: int = 200
    n_train: int = 200
    n_test: int = 200
    val_fraction: float = 0.10
    hidden_dim: int = 64
    batch_size: int = 2048
    learning_rate: float = 1e-3
    max_epochs: int = 1200
    patience: int = 120
    min_delta: float = 0.0
    seed: int = 2026
    print_every: int = 50
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.n_train not in (100, 150, 200):
            raise ValueError("n_train debe ser 100, 150 o 200")
        if self.n_test != 200:
            raise ValueError("Se usa siempre el mismo conjunto de prueba de 200 trayectorias")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction debe estar entre 0 y 1")
        if min(self.g, self.mass, self.length, self.dt, self.learning_rate) <= 0:
            raise ValueError("Los parametros fisicos y la tasa de aprendizaje deben ser positivos")
        if self.min_delta < 0:
            raise ValueError("min_delta no puede ser negativo")
        if min(self.n_steps, self.hidden_dim, self.batch_size, self.max_epochs, self.patience) < 1:
            raise ValueError("Los parametros discretos deben ser mayores que cero")

    def to_dict(self) -> dict:
        return asdict(self)
