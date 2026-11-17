"""Metricas fisicas y numericas, siempre separadas por unidad."""

import numpy as np
import pandas as pd

from .physics import hamiltonian


def field_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Errores por componente; q_dot y p_dot no se agregan porque tienen unidades distintas."""
    error = np.asarray(pred) - np.asarray(true)
    mse = np.mean(error**2, axis=0)
    return {
        "field_mse_qdot": float(mse[0]),
        "field_mse_pdot": float(mse[1]),
        "field_rmse_qdot": float(np.sqrt(mse[0])),
        "field_rmse_pdot": float(np.sqrt(mse[1])),
    }


def drift_by_trajectory(energy: np.ndarray) -> np.ndarray:
    """Max_t |H(t)-H(0)| en joules, por trayectoria."""
    return np.max(np.abs(energy - energy[:, :1]), axis=1)


def trajectory_metrics(ids: np.ndarray, true: np.ndarray, pred: np.ndarray,
                       g: float = 9.81, mass: float = 1.0,
                       length: float = 1.0) -> tuple[dict[str, float | int], pd.DataFrame]:
    """RMSE global por variable + energia fisica del pendulo verdadero."""
    if true.shape != pred.shape or true.ndim != 3 or true.shape[2] != 2:
        raise ValueError("Las trayectorias deben tener la misma forma (N,T,2)")
    complete = np.isfinite(pred).all(axis=(1, 2))
    safe_pred = np.where(complete[:, None, None], pred, np.nan)
    errors = safe_pred - true
    sq = errors**2
    per_traj_rmse = np.sqrt(np.mean(sq, axis=1))
    global_rmse = np.sqrt(np.mean(sq, axis=(0, 1)))
    phys = dict(g=g, mass=mass, length=length)
    energy_pred = hamiltonian(safe_pred, **phys)
    energy_true = hamiltonian(true, **phys)
    drift_pred = drift_by_trajectory(energy_pred)
    drift_true = drift_by_trajectory(energy_true)
    frame = pd.DataFrame({
        "trajectory_id": ids,
        "rollout_complete": complete,
        "rmse_q_rad": per_traj_rmse[:, 0],
        "rmse_p_kg_m2_s": per_traj_rmse[:, 1],
        "max_abs_energy_drift_J": drift_pred,
        "reference_max_abs_energy_drift_J": drift_true,
        "initial_energy_J": energy_true[:, 0],
    })
    metrics = {
        "rollouts_completed": int(complete.sum()),
        "rollouts_failed": int((~complete).sum()),
        "rollout_rmse_q_rad": float(global_rmse[0]),
        "rollout_rmse_p_kg_m2_s": float(global_rmse[1]),
        "energy_drift_mean_J": float(np.mean(drift_pred)),
        "energy_drift_median_J": float(np.median(drift_pred)),
        "energy_drift_max_J": float(np.max(drift_pred)),
        "reference_energy_drift_mean_J": float(np.mean(drift_true)),
        "reference_energy_drift_max_J": float(np.max(drift_true)),
    }
    if not complete.all() and complete.any():
        completed_rmse = np.sqrt(np.mean(sq[complete], axis=(0, 1)))
        metrics.update({
            "completed_only_rollout_rmse_q_rad": float(completed_rmse[0]),
            "completed_only_rollout_rmse_p_kg_m2_s": float(completed_rmse[1]),
            "completed_only_energy_drift_mean_J": float(np.mean(drift_pred[complete])),
            "completed_only_energy_drift_max_J": float(np.max(drift_pred[complete])),
        })
    return metrics, frame
