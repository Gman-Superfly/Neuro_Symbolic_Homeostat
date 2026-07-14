import numpy as np
import math

from core.energy import project_noise_orthogonal, project_noise_metric_orthogonal


def test_euclidean_projection_orthogonality():
    rng = np.random.default_rng(42)
    for _ in range(50):
        g = rng.normal(0.0, 1.0, size=(16,))
        # ensure non-degenerate gradient
        if np.linalg.norm(g) < 1e-6:
            continue
        z = rng.normal(0.0, 1.0, size=(16,))
        z_perp = project_noise_orthogonal(z, g)
        dot_val = float(np.dot(z_perp, g))
        assert math.isfinite(dot_val)
        assert abs(dot_val) <= 1e-9


def test_metric_projection_orthogonality():
    rng = np.random.default_rng(7)
    dimension = 8
    basis, _ = np.linalg.qr(rng.normal(0.0, 1.0, size=(dimension, dimension)))
    eigenvalues = np.geomspace(0.25, 8.0, num=dimension)
    M = basis @ np.diag(eigenvalues) @ basis.T
    for _ in range(50):
        g = rng.normal(0.0, 1.0, size=dimension)
        if np.linalg.norm(g) < 1e-6:
            continue
        z = rng.normal(0.0, 1.0, size=dimension)
        z_perp = project_noise_metric_orthogonal(z, g, M=M)
        metric_grad = np.linalg.solve(M, g)
        first_order_change = float(np.dot(z_perp, g))
        metric_inner_product = float(np.dot(z_perp, M @ metric_grad))
        assert math.isfinite(first_order_change)
        assert abs(first_order_change) <= 1e-9
        assert abs(metric_inner_product) <= 1e-9


def test_metric_projection_supports_matrix_free_solve():
    M = np.array(
        [
            [2.0, 0.4, 0.1, 0.0],
            [0.4, 1.5, 0.2, 0.1],
            [0.1, 0.2, 1.2, 0.3],
            [0.0, 0.1, 0.3, 1.0],
        ],
        dtype=float,
    )
    g = np.array([0.5, -0.7, 1.2, 0.3], dtype=float)
    z = np.array([1.0, 0.4, -0.2, 0.8], dtype=float)

    dense = project_noise_metric_orthogonal(z, g, M=M)
    matrix_free = project_noise_metric_orthogonal(z, g, metric_solve=lambda vector: np.linalg.solve(M, vector))

    assert np.allclose(matrix_free, dense, rtol=0.0, atol=1e-12)
    assert abs(float(np.dot(matrix_free, g))) <= 1e-12


