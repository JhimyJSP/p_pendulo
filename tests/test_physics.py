import numpy as np

from pendulum_project.physics import exact_field, hamiltonian, integrate


def test_hamilton_equations():
    states = np.array([[0.0, 2.0], [np.pi / 2, 0.0]])
    field = exact_field(states)
    np.testing.assert_allclose(field[0], [2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(field[1], [0.0, -9.81], atol=1e-12)


def test_rk4_keeps_energy_to_numerical_precision():
    z = integrate(exact_field, np.array([[0.8, 1.2], [-1.1, -2.5]]), 0.02, 200)
    energy = hamiltonian(z)
    assert np.max(np.abs(energy - energy[:, :1])) < 3e-5


def test_rk4_fourth_order_convergence():
    initial = np.array([[0.5, 1.0]])
    fine = integrate(exact_field, initial, 0.005, 200)[:, -1]
    medium = integrate(exact_field, initial, 0.04, 25)[:, -1]
    coarse = integrate(exact_field, initial, 0.08, 12)  # 0.96 s
    coarse = integrate(exact_field, coarse[:, -1, :], 0.04, 1)[:, -1]  # 1.00 s
    error_medium = np.linalg.norm(medium - fine)
    error_coarse = np.linalg.norm(coarse - fine)
    assert error_medium < error_coarse / 5
