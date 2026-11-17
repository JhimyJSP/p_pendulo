"""Pipeline reproducible train -> checkpoints -> test -> RK4 -> graficos."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data import TARGETS, ordered_trajectories, read_datasets, split_train_val
from .metrics import field_metrics, trajectory_metrics, drift_by_trajectory
from .models import make_model
from .physics import integrate
from .plots import field_comparison, learning_curves, rollout_plots, summary_plot
from .training import predict_field, predict_hamiltonian, resolve_device, train_one

MODELS = ("mlp", "hnn")


def fingerprint(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def subdir(output_dir: Path, n_train: int) -> Path:
    return output_dir / f"N{n_train}"


def run_training(data_dir: Path, output_dir: Path, config: Config,
                 kinds: tuple[str, ...] = MODELS) -> None:
    train, test = read_datasets(data_dir, config)
    splits = split_train_val(train, test, config)
    device = resolve_device(config.device)
    dest = subdir(output_dir, config.n_train)
    (dest / "models").mkdir(parents=True, exist_ok=True)
    (dest / "plots").mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config.to_dict(),
        "sha256": {name: fingerprint(data_dir / f"{name}_flat.csv") for name in ("train", "test")},
        "fit_ids": splits.fit_ids,
        "validation_ids": splits.val_ids,
        "fit_trajectories": len(splits.fit_ids),
        "validation_trajectories": len(splits.val_ids),
        "test_trajectories": splits.test.trajectory_id.nunique(),
        "models": list(kinds),
        "runtime": {"device": str(device), "torch": str(torch.__version__),
                    "torch_threads": torch.get_num_threads(), "numpy": np.__version__,
                    "pandas": pd.__version__},
        "training": {},
    }
    histories = {}
    for kind in kinds:
        print(f"\nEntrenando {kind.upper()} | N={config.n_train} | {device}", flush=True)
        model, history, info, scale = train_one(kind, splits.fit, splits.val, config, device)
        histories[kind] = history
        history.to_csv(dest / f"history_{kind}.csv", index=False)
        torch.save({
            "kind": kind,
            "hidden_dim": config.hidden_dim,
            "state_dict": model.cpu().state_dict(),
            "y_scale_fit": scale,
            "training_info": info,
            "config": config.to_dict(),
            "sha256": manifest["sha256"],
            "fit_ids": splits.fit_ids,
            "validation_ids": splits.val_ids,
        }, dest / "models" / f"{kind}.pt")
        manifest["training"][kind] = {**info, "y_scale_fit": scale,
                                      "parameters": sum(t.numel() for t in model.parameters())}
    save_json(manifest, dest / "manifest.json")
    learning_curves(histories, dest / "plots")
    print(f"\nModelos e historiales guardados en: {dest}")


def load_checkpoint(path: Path, kind: str, config: Config, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Falta {path}. Ejecuta primero 'train' o 'all'.")
    # weights_only=True evita ejecutar objetos Python arbitrarios al leer checkpoints.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["kind"] != kind:
        raise ValueError(f"Checkpoint con tipo incorrecto: {path}")
    params = checkpoint["config"]
    fields = ("g", "mass", "length", "dt", "n_steps", "n_train", "n_test", "hidden_dim", "seed")
    if any(params[key] != getattr(config, key) for key in fields):
        raise ValueError(f"Configuracion incompatible con {path}. Usa los mismos parametros de entrenamiento.")
    model = make_model(kind, checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def energy_domains(fit: pd.DataFrame, test: pd.DataFrame,
                   ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Clasifica trayectorias completas por H inicial frente al intervalo de ajuste."""
    fit_energy = fit.loc[fit.step.eq(0), "H_J"]
    test_energy = test.loc[test.step.eq(0)].set_index("trajectory_id").loc[ids, "H_J"]
    lo, hi = float(fit_energy.min()), float(fit_energy.max())
    below = test_energy.to_numpy() < lo
    above = test_energy.to_numpy() > hi
    masks = {"ID": ~(below | above), "OOD": below | above}
    definition = {"rule": "initial energy within closed fit interval => ID; otherwise OOD",
                  "fit_min_initial_energy_J": lo, "fit_max_initial_energy_J": hi,
                  "test_ids": {name: ids[mask].tolist() for name, mask in masks.items()},
                  "ood_below_ids": ids[below].tolist(), "ood_above_ids": ids[above].tolist()}
    return masks, definition


def run_evaluation(data_dir: Path, output_dir: Path, config: Config) -> dict[str, dict]:
    train, test = read_datasets(data_dir, config)
    selected = split_train_val(train, test, config)
    dest = subdir(output_dir, config.n_train)
    manifest_path = dest / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No existe {manifest_path}: entrena los modelos antes de evaluar")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config"] != config.to_dict():
        # device, print_every, epochs etc. pueden cambiar entre entrenamiento y evaluacion.
        core = ("g", "mass", "length", "dt", "n_steps", "n_train", "n_test", "hidden_dim", "seed")
        if any(manifest["config"][key] != getattr(config, key) for key in core):
            raise ValueError("La configuracion del modelo entrenado no coincide con la evaluacion")
    if manifest["fit_ids"] != selected.fit_ids or manifest["validation_ids"] != selected.val_ids:
        raise ValueError("La separacion fit/validacion ha cambiado")
    for name in ("train", "test"):
        if fingerprint(data_dir / f"{name}_flat.csv") != manifest["sha256"][name]:
            raise ValueError(f"Los datos {name} se modificaron desde el entrenamiento")
    ids, true, t = ordered_trajectories(selected.test, config)
    initial_states = true[:, 0, :]
    true_fields = selected.test.sort_values(["trajectory_id", "step"])[TARGETS].to_numpy()
    true_fields_by_trajectory = true_fields.reshape(true.shape)
    test_states = true.reshape(-1, 2)
    if not np.array_equal(ids, np.sort(selected.test.trajectory_id.unique())):
        raise AssertionError("Los IDs estan desordenados")
    device = resolve_device(config.device)
    phys = dict(g=config.g, mass=config.mass, length=config.length)
    paths, predicted_fields, metrics, per_trajectory = {}, {}, {}, []
    domains, definition = energy_domains(selected.fit, selected.test, ids)
    by_domain = []

    for kind in manifest.get("models", MODELS):
        model, checkpoint = load_checkpoint(dest / "models" / f"{kind}.pt", kind, config, device)
        if checkpoint["sha256"] != manifest["sha256"]:
            raise ValueError("El checkpoint no corresponde al conjunto de datos utilizado")
        if checkpoint["fit_ids"] != selected.fit_ids:
            raise ValueError("El checkpoint no corresponde al conjunto de ajuste")
        print(f"Evaluando {kind.upper()} en {len(ids)} trayectorias de prueba...", flush=True)
        yhat = predict_field(model, test_states, device)
        predicted_fields[kind] = yhat
        trajectory = integrate(lambda z: predict_field(model, z, device),
                               initial_states, config.dt, config.n_steps)
        roll_metrics, by_traj = trajectory_metrics(ids, true, trajectory, **phys)
        complete = by_traj["rollout_complete"].to_numpy()
        paths[kind] = np.where(complete[:, None, None], trajectory, np.nan)
        field_results = field_metrics(yhat, true_fields)
        metrics[kind] = {**field_results, **roll_metrics}
        h_drift = None
        if hasattr(model, "hamiltonian"):
            h_theta = predict_hamiltonian(model, paths[kind], device)
            h_drift = drift_by_trajectory(h_theta)
            metrics[kind]["learned_H_drift_mean_J"] = float(h_drift.mean())
            metrics[kind]["learned_H_drift_max_J"] = float(h_drift.max())
            by_traj["learned_H_max_abs_drift_J"] = h_drift
        yhat_by_trajectory = yhat.reshape(true.shape)
        for domain, mask in domains.items():
            group = {"model": kind, "domain": domain, "n_trajectories": int(mask.sum())}
            if mask.any():
                group.update(field_metrics(yhat_by_trajectory[mask].reshape(-1, 2),
                                           true_fields_by_trajectory[mask].reshape(-1, 2)))
                group.update(trajectory_metrics(ids[mask], true[mask], trajectory[mask], **phys)[0])
                if h_drift is not None:
                    group["learned_H_drift_mean_J"] = float(h_drift[mask].mean())
                    group["learned_H_drift_max_J"] = float(h_drift[mask].max())
            by_domain.append(group)
        by_traj.insert(0, "model", kind)
        by_traj.insert(1, "domain", np.where(domains["ID"], "ID", "OOD"))
        per_trajectory.append(by_traj)
        del model
    pd.concat(per_trajectory, ignore_index=True).to_csv(dest / "per_trajectory_test.csv", index=False)
    pd.DataFrame(metrics).T.to_csv(dest / "metrics_test.csv", index_label="model")
    # Las trayectorias divergentes dejan sus métricas globales indefinidas.
    # JSON estándar las representa como null y el CSV deja las celdas vacías.
    json_metrics = {kind: {key: value if np.isfinite(value) else None
                          for key, value in values.items()} for kind, values in metrics.items()}
    save_json(json_metrics, dest / "metrics_test.json")
    pd.DataFrame(by_domain).to_csv(dest / "metrics_test_domains.csv", index=False)
    save_json(definition, dest / "test_domains.json")
    field_comparison(true_fields, predicted_fields, dest / "plots")
    rollout_plots(ids, t, true, paths, dest / "plots", **phys)
    summary_plot(metrics, dest / "plots")
    print("\nMETRICAS DE PRUEBA (200 trayectorias; 4 s):", flush=True)
    print(pd.DataFrame(metrics).T.to_string(float_format=lambda x: f"{x:.6e}"), flush=True)
    print(f"\nGraficos y metricas guardados en: {dest}", flush=True)
    return metrics
