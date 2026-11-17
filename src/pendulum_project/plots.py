"""Graficos listos para el informe del proyecto (Matplotlib no interactivo)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .physics import hamiltonian


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def learning_curves(histories: dict[str, pd.DataFrame], directory: Path) -> None:
    for kind, history in histories.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(history.epoch, history.train_loss, label="Ajuste")
        ax.plot(history.epoch, history.val_loss, label="Validacion")
        ax.set(xlabel="Epoca", ylabel="MSE balanceado", title=f"Aprendizaje — {kind.upper()}")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()
        _save(fig, directory / f"aprendizaje_{kind}.png")


def field_comparison(true_fields: np.ndarray, predictions: dict[str, np.ndarray],
                     directory: Path) -> None:
    # Solo graficamos una submuestra uniforme; las metricas usan TODOS los estados.
    sample = np.linspace(0, len(true_fields) - 1, min(2500, len(true_fields)), dtype=int)
    labels = ["q punto [rad/s]", "p punto [kg m^2/s^2]"]
    for col in range(2):
        fig, ax = plt.subplots(figsize=(6, 5.5))
        for kind, pred in predictions.items():
            ax.scatter(true_fields[sample, col], pred[sample, col], s=6,
                       alpha=0.3, label=kind.upper())
        lim = [float(min(true_fields[:, col].min(), *(p[:, col].min() for p in predictions.values()))),
               float(max(true_fields[:, col].max(), *(p[:, col].max() for p in predictions.values())))]
        ax.plot(lim, lim, "k--", lw=1, label="Prediccion exacta")
        ax.set(xlabel=f"Valor verdadero: {labels[col]}",
               ylabel=f"Valor predicho: {labels[col]}",
               title=f"Campo dinamico — componente {col + 1}")
        ax.legend()
        ax.grid(alpha=0.2)
        _save(fig, directory / f"campo_componente_{col + 1}.png")


def select_example_indices(true: np.ndarray, g: float = 9.81,
                           mass: float = 1.0, length: float = 1.0) -> list[int]:
    """Tres trayectorias representativas ordenadas por energia inicial."""
    e0 = hamiltonian(true[:, 0, :], g=g, mass=mass, length=length)
    order = np.argsort(e0)
    return sorted(set(map(int, (order[0], order[len(order) // 2], order[-1]))))


def rollout_plots(ids: np.ndarray, time: np.ndarray, true: np.ndarray,
                  paths: dict[str, np.ndarray], directory: Path,
                  g: float = 9.81, mass: float = 1.0,
                  length: float = 1.0) -> None:
    phys = dict(g=g, mass=mass, length=length)
    for index in select_example_indices(true, **phys):
        trajectory_id = int(ids[index])
        fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
        for k, (label, unit) in enumerate((("q", "rad"), ("p", "kg m²/s"))):
            ax = axes[k]
            ax.plot(time, true[index, :, k], "k", linewidth=2, label="Referencia RK4")
            for kind, z in paths.items():
                label = kind.upper() if np.isfinite(z[index]).all() else f"{kind.upper()} (diverge)"
                ax.plot(time, z[index, :, k], label=label, alpha=0.85)
            ax.set_ylabel(f"{label} [{unit}]")
            ax.grid(alpha=0.25)
            ax.legend()
        axes[-1].set_xlabel("Tiempo [s]")
        fig.suptitle(f"Trayectoria de prueba ID={trajectory_id}")
        _save(fig, directory / f"trayectoria_ID{trajectory_id}.png")

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(true[index, :, 0], true[index, :, 1], "k", lw=2, label="Referencia RK4")
        for kind, z in paths.items():
            ax.plot(z[index, :, 0], z[index, :, 1], label=kind.upper())
        ax.scatter([true[index, 0, 0]], [true[index, 0, 1]], s=35, marker="o", c="black")
        ax.set(xlabel="Angulo q [rad]", ylabel="Momento p [kg m²/s]",
               title=f"Espacio de fases ID={trajectory_id}")
        ax.legend()
        ax.grid(alpha=0.25)
        _save(fig, directory / f"fase_ID{trajectory_id}.png")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ref_e = hamiltonian(true[index], **phys)
        ax.plot(time, ref_e - ref_e[0], "k", lw=2, label="Referencia RK4")
        for kind, z in paths.items():
            e = hamiltonian(z[index], **phys)
            ax.plot(time, e - e[0], label=kind.upper())
        ax.set(xlabel="Tiempo [s]", ylabel="H fisica(t) - H fisica(0) [J]",
               title=f"Deriva de energia fisica ID={trajectory_id}")
        ax.legend()
        ax.grid(alpha=0.25)
        _save(fig, directory / f"energia_ID{trajectory_id}.png")


def summary_plot(metrics: dict[str, dict], directory: Path) -> None:
    """No se promedian q (radianes) y p (momento): se muestran por separado."""
    keys = ["field_mse_qdot", "field_mse_pdot", "rollout_rmse_q_rad",
            "rollout_rmse_p_kg_m2_s", "energy_drift_mean_J"]
    labels = ["MSE(q punto)", "MSE(p punto)", "RMSE(q)", "RMSE(p)", "Deriva media H [J]"]
    fig, axes = plt.subplots(1, len(keys), figsize=(17, 3.5))
    for ax, key, label in zip(axes, keys, labels):
        ax.bar([kind.upper() for kind in metrics], [values[key] for values in metrics.values()])
        ax.tick_params(axis="x", labelrotation=35)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    _save(fig, directory / "resumen_metricas.png")
