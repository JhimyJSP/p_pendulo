from pathlib import Path
import numpy as np
import pytest

from pendulum_project.config import Config, ROOT
from pendulum_project.data import generate_data, read_datasets, split_train_val, ordered_trajectories
from pendulum_project.metrics import field_metrics, trajectory_metrics


@pytest.fixture(scope="module")
def generated_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("generated-data")
    generate_data(ROOT / "data", directory, Config())
    return directory


def test_generated_csv_schema_and_kinematics(generated_data_dir: Path):
    config = Config()
    train, test = read_datasets(generated_data_dir, config)
    assert len(train) == len(test) == 40200
    ids, states, t = ordered_trajectories(test, config)
    assert len(ids) == 200 and states.shape == (200, 201, 2)
    assert np.isclose(t[-1], 4.0)


def test_validation_split_uses_disjoint_trajectories(generated_data_dir: Path):
    cfg = Config(n_train=100)
    train, test = read_datasets(generated_data_dir, cfg)
    split = split_train_val(train, test, cfg)
    assert len(split.fit_ids) == 90
    assert len(split.val_ids) == 10
    assert set(split.fit_ids).isdisjoint(split.val_ids)
    assert split.test.trajectory_id.nunique() == 200
    assert set(split.fit.trajectory_id.unique()) == set(split.fit_ids)


def test_exact_metrics_are_zero():
    z = np.zeros((2, 4, 2), dtype=np.float64)
    rmse, per = trajectory_metrics(np.array([1, 2]), z, z)
    assert rmse["rollout_rmse_q_rad"] == 0
    assert rmse["energy_drift_mean_J"] == 0
    assert len(per) == 2
    field = field_metrics(z.reshape(-1, 2), z.reshape(-1, 2))
    assert field["field_mse_qdot"] == 0
    assert field["field_mse_pdot"] == 0
