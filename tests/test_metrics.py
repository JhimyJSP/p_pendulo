import numpy as np
import pandas as pd

from pendulum_project.experiment import energy_domains
from pendulum_project.metrics import field_metrics, trajectory_metrics


def test_field_errors_remain_separate_by_physical_unit():
    metrics = field_metrics(np.array([[1.0, 3.0], [1.0, 3.0]]), np.zeros((2, 2)))
    assert metrics["field_rmse_qdot"] == 1.0
    assert metrics["field_rmse_pdot"] == 3.0
    assert "field_rmse_total" not in metrics
    assert "field_mse_mean_components" not in metrics


def test_energy_domains_use_initial_fit_energy_and_test_id_order():
    fit = pd.DataFrame({"step": [0, 0, 1], "H_J": [1.0, 3.0, 9.0]})
    test = pd.DataFrame({"step": [0, 0, 0, 0, 1], "trajectory_id": [1, 2, 3, 4, 4],
                         "H_J": [1.0, 0.5, 3.0, 4.0, 2.0]})
    masks, description = energy_domains(fit, test, np.array([4, 2, 1, 3]))
    assert masks["ID"].tolist() == [False, False, True, True]
    assert masks["OOD"].tolist() == [True, True, False, False]
    assert description["ood_below_ids"] == [2]
    assert description["ood_above_ids"] == [4]


def test_divergent_rollout_is_reported_and_kept_in_group():
    true = np.zeros((2, 3, 2))
    predicted = true.copy()
    predicted[1, -1] = np.inf
    metrics, rows = trajectory_metrics(np.array([11, 12]), true, predicted)
    assert metrics["rollouts_failed"] == 1
    assert metrics["rollouts_completed"] == 1
    assert np.isnan(metrics["rollout_rmse_q_rad"])
    assert metrics["completed_only_rollout_rmse_q_rad"] == 0.0
    assert rows.rollout_complete.tolist() == [True, False]
