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
    # SPD metric with anisotropy
    diag = np.array([1.0, 3.0, 0.5, 2.5, 4.0, 0.8, 1.7, 2.2], dtype=float)
    M = np.diag(diag)
    for _ in range(50):
        g = rng.normal(0.0, 1.0, size=(len(diag),))
        if np.linalg.norm(g) < 1e-6:
            continue
        z = rng.normal(0.0, 1.0, size=(len(diag),))
        z_perp = project_noise_metric_orthogonal(z, g, M=M)
        Mg = M @ g
        dot_val = float(np.dot(z_perp, Mg))
        assert math.isfinite(dot_val)
        assert abs(dot_val) <= 1e-9


