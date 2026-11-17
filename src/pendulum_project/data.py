"""Lectura, verificacion fisica y separacion de datos por trayectoria."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .physics import exact_field, hamiltonian, integrate

INPUTS = ["q_rad", "p_kg_m2_s"]
TARGETS = ["q_dot_rad_s", "p_dot_kg_m2_s2"]
BASE_COLUMNS = ["split", "trajectory_id", "step", "t_s", *INPUTS,
                *TARGETS, "H_J", "q0_rad", "p0_kg_m2_s", "in_N100", "in_N150", "in_N200"]


@dataclass
class DataSplit:
    fit: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    fit_ids: list[int]
    val_ids: list[int]


def read_datasets(data_dir: Path, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for split in ("train", "test"):
        path = data_dir / f"{split}_flat.csv"
        if not path.is_file():
            raise FileNotFoundError(f"No existe {path}. Ejecuta primero 'generate' o copia tus CSV.")
        df = pd.read_csv(path)
        missing = set(BASE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: faltan columnas {sorted(missing)}")
        if not df["split"].eq(split).all():
            raise ValueError(f"{path.name}: la columna split no corresponde a {split}")
        if df[BASE_COLUMNS[1:10]].isna().any().any():
            raise ValueError(f"{path.name}: hay valores faltantes")
        if df.duplicated(["trajectory_id", "step"]).any():
            raise ValueError(f"{path.name}: hay pasos duplicados dentro de una trayectoria")
        if df["trajectory_id"].nunique() != 200:
            raise ValueError(f"{path.name}: se esperaban 200 trayectorias")
        if not df.groupby("trajectory_id").size().eq(config.n_steps + 1).all():
            raise ValueError(f"{path.name}: numero de estados por trayectoria incorrecto")
        expected_steps = np.arange(config.n_steps + 1)
        for _, grp in df.groupby("trajectory_id", sort=False):
            ordered = grp.sort_values("step")
            if not np.array_equal(ordered["step"].to_numpy(), expected_steps):
                raise ValueError(f"{path.name}: los pasos deben ser 0..{config.n_steps}")
            if not np.allclose(ordered["t_s"].to_numpy(), expected_steps * config.dt,
                               atol=1e-8, rtol=0):
                raise ValueError(f"{path.name}: tiempos incompatibles con dt")
        x = df[INPUTS].to_numpy(dtype=np.float64)
        y = df[TARGETS].to_numpy(dtype=np.float64)
        phys = dict(g=config.g, mass=config.mass, length=config.length)
        if not np.allclose(y, exact_field(x, **phys), atol=1e-6, rtol=0):
            raise ValueError(f"{path.name}: el campo no satisface las ecuaciones del pendulo")
        if not np.allclose(df["H_J"], hamiltonian(x, **phys), atol=1e-6, rtol=0):
            raise ValueError(f"{path.name}: la energia es inconsistente")
        for n in (100, 150, 200):
            selected_ids = df.loc[df[f"in_N{n}"], "trajectory_id"].nunique()
            if selected_ids != n:
                raise ValueError(f"{path.name}: el subconjunto N{n} tiene {selected_ids} trayectorias")
            if df.groupby("trajectory_id")[f"in_N{n}"].nunique().max() != 1:
                raise ValueError(f"{path.name}: la bandera N{n} cambia dentro de una trayectoria")
        frames.append(df)
    return frames[0], frames[1]


def split_train_val(train: pd.DataFrame, test: pd.DataFrame, config: Config) -> DataSplit:
    """La validacion usa IDs completos y el test fijo son las 200 trayectorias."""
    selected = train.loc[train[f"in_N{config.n_train}"]].copy()
    test_selected = test.loc[test[f"in_N{config.n_test}"]].copy()
    ids = np.sort(selected.trajectory_id.unique()).copy()
    np.random.default_rng(config.seed).shuffle(ids)
    n_val = max(1, int(round(len(ids) * config.val_fraction)))
    val_ids = sorted(map(int, ids[:n_val]))
    fit_ids = sorted(map(int, ids[n_val:]))
    if not fit_ids:
        raise ValueError("La validacion dejo vacio el ajuste")
    fit = selected.loc[selected.trajectory_id.isin(fit_ids)].copy()
    val = selected.loc[selected.trajectory_id.isin(val_ids)].copy()
    return DataSplit(fit, val, test_selected, fit_ids, val_ids)


def ordered_trajectories(df: pd.DataFrame, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IDs, estados de referencia (N, T, 2) y tiempos, ordenados."""
    ordered = df.sort_values(["trajectory_id", "step"])
    ids = ordered.trajectory_id.unique().astype(int)
    states = ordered[INPUTS].to_numpy(dtype=np.float64).reshape(len(ids), config.n_steps + 1, 2)
    times = np.arange(config.n_steps + 1, dtype=np.float64) * config.dt
    return ids, states, times


def generate_data(input_dir: Path, output_dir: Path, config: Config, overwrite: bool = False) -> None:
    """Reconstruye train_flat/test_flat desde los CSV originales de condiciones iniciales.

    Los CSV planos se omiten del repositorio porque son artefactos regenerables.
    Las nuevas trayectorias no tienen por que ser bit a bit iguales a una
    exportacion previa por el redondeo decimal de las condiciones iniciales.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        source = input_dir / f"initial_conditions_{split}.csv"
        output = output_dir / f"{split}_flat.csv"
        if output.exists() and not overwrite:
            raise FileExistsError(f"{output} ya existe; usa --overwrite para reemplazarlo")
        ic = pd.read_csv(source)
        cols = ["ID", "q₀ [rad]", "p₀ [kg·m²/s]", "in_N100", "in_N150", "in_N200"]
        if set(cols) - set(ic.columns):
            raise ValueError(f"{source.name}: falta alguna columna de {cols}")
        if len(ic) != 200 or ic.ID.duplicated().any():
            raise ValueError(f"{source.name}: se requieren 200 condiciones iniciales distintas")
        ic = ic.sort_values("ID").reset_index(drop=True)
        z0 = ic[["q₀ [rad]", "p₀ [kg·m²/s]"]].to_numpy(dtype=np.float64)
        phys = dict(g=config.g, mass=config.mass, length=config.length)
        trajectories = integrate(lambda z: exact_field(z, **phys), z0, config.dt, config.n_steps)
        n, steps, _ = trajectories.shape
        flat = trajectories.reshape(-1, 2)
        field = exact_field(flat, **phys)
        energy = hamiltonian(flat, **phys)
        out = pd.DataFrame({
            "split": split,
            "trajectory_id": np.repeat(ic.ID.to_numpy(), steps),
            "step": np.tile(np.arange(steps), n),
            "t_s": np.tile(np.arange(steps) * config.dt, n),
            "q_rad": flat[:, 0],
            "p_kg_m2_s": flat[:, 1],
            "q_dot_rad_s": field[:, 0],
            "p_dot_kg_m2_s2": field[:, 1],
            "H_J": energy,
            "q0_rad": np.repeat(z0[:, 0], steps),
            "p0_kg_m2_s": np.repeat(z0[:, 1], steps),
            **{f"in_N{k}": np.repeat(ic[f"in_N{k}"].astype(bool).to_numpy(), steps)
               for k in (100, 150, 200)},
        }, columns=BASE_COLUMNS)
        out.to_csv(output, index=False, float_format="%.12g")
        print(f"Generado: {output} ({len(out)} estados)")
