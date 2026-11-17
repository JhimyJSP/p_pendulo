"""Pendulo ideal: energia, campo exacto e integracion RK4."""

import numpy as np


def hamiltonian(states: np.ndarray, g: float = 9.81,
                mass: float = 1.0, length: float = 1.0) -> np.ndarray:
    """H = p²/(2 m l²) + m g l (1 - cos(q)); devuelve energia en joules."""
    z = np.asarray(states, dtype=np.float64)
    q, p = z[..., 0], z[..., 1]
    return p**2 / (2 * mass * length**2) + mass * g * length * (1 - np.cos(q))


def exact_field(states: np.ndarray, g: float = 9.81,
                mass: float = 1.0, length: float = 1.0) -> np.ndarray:
    """Derivadas del estado x=(q,p) segun las ecuaciones de Hamilton."""
    z = np.asarray(states, dtype=np.float64)
    q, p = z[..., 0], z[..., 1]
    return np.stack((p / (mass * length**2), -mass * g * length * np.sin(q)), axis=-1)


def rk4_step(field, state: np.ndarray, dt: float) -> np.ndarray:
    """Un paso de Runge-Kutta de cuarto orden; acepta muchos estados en paralelo."""
    k1 = field(state)
    k2 = field(state + (dt / 2) * k1)
    k3 = field(state + (dt / 2) * k2)
    k4 = field(state + dt * k3)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(field, initial: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """Forma (trayectorias, tiempos, 2), incluso con una sola trayectoria."""
    z0 = np.asarray(initial, dtype=np.float64)
    if z0.shape[-1] != 2:
        raise ValueError("El estado debe ser (..., 2), con q y p")
    if z0.ndim == 1:
        z0 = z0[None, :]
    if z0.ndim != 2:
        raise ValueError("initial debe tener forma (N, 2)")
    path = np.empty((len(z0), n_steps + 1, 2), dtype=np.float64)
    path[:, 0] = z0
    for step in range(n_steps):
        path[:, step + 1] = rk4_step(field, path[:, step], dt)
    return path
