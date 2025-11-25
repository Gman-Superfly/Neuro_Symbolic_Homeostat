from __future__ import annotations

import numpy as np

from core.noise_controller import PrecisionNoiseController


def test_precision_noise_weights_inverse_curvature() -> None:
    """Weights should be inversely proportional to curvature and ℓ2-normalized."""
    ctrl = PrecisionNoiseController(base_magnitude=0.1, decay=1.0, precision_epsilon=1e-8)
    curv = np.array([10.0, 1.0, 0.0], dtype=float)
    w = ctrl.weights_for_curvatures(curv)
    # Highest weight on smallest curvature (index 2), then mid (index 1), then smallest on stiff (index 0)
    assert w[2] > w[1] > w[0], f"unexpected ordering of weights: {w}"
    # ℓ2 norm ≈ 1
    norm = float(np.linalg.norm(w))
    assert abs(norm - 1.0) < 1e-6, f"weights not normalized, ||w||={norm}"

