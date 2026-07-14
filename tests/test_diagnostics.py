from __future__ import annotations

import numpy as np

from core.coordinator import EnergyCoordinator
from experiments.ablate_pson_noise import QuadraticWell


def test_public_snapshot_and_noise_vector_expose_experiment_diagnostics() -> None:
    coord = EnergyCoordinator(
        modules=[
            QuadraticWell(target=0.2, curvature_value=2.0),
            QuadraticWell(target=0.8, curvature_value=4.0),
        ],
        couplings=[],
        constraints={},
        noise_mode="precision_orthogonal",
        noise_magnitude=0.02,
        precision_aware_noise_controller=True,
    )
    etas = [0.7, 0.3]
    snapshot = coord.inspect_state(etas)

    assert snapshot.etas == (0.7, 0.3)
    assert snapshot.energy > 0.0
    assert np.allclose(snapshot.gradient, (1.0, -2.0), rtol=0.0, atol=1e-12)
    assert np.allclose(snapshot.precision_diagonal, (2.0, 4.0), rtol=0.0, atol=1e-12)
    assert snapshot.lipschitz_bound >= 4.0

    noise = coord.build_noise_vector(np.array([0.3, 0.9]), np.asarray(snapshot.gradient))
    assert np.isclose(np.linalg.norm(noise), 0.02)
    assert abs(float(np.dot(noise, np.asarray(snapshot.gradient)))) <= 1e-12
