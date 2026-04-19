"""
loss.py

Entropy-regularized loss for MIRAGE++.

The MIRAGE++ objective is:

    L(θ; X, y) = (1/m)||Xθ − y||²  −  λ H(θ)

where H(θ) = −Σ_i θ_i log θ_i ≥ 0 is the Shannon entropy (nats).

Subtracting entropy means the optimizer is *rewarded* for spreading weight
across dimensions: minimising L simultaneously fits the data (MSE↓) and
diversifies the weights (H(θ)↑).  This is the correct sign for a diversity
bonus — adding entropy would instead penalise spread and push toward
concentrated (low-entropy) solutions.

Gradient:
    ∇_θ L = (2/m) Xᵀ(Xθ−y)  +  λ (1 + log θ)

The entropy gradient term λ(1 + log θ) points toward higher-entropy
distributions; combined with mirror descent it produces the multiplicative
update θ_{t+1,i} ∝ θ_{t,i}^{1−ηλ} · exp(−η ∇_MSE_i), which shrinks large
components (1−ηλ < 1) and thus promotes diversification.
"""

import numpy as np
from typing import Tuple


def entropy(theta: np.ndarray, eps: float = 1e-8) -> float:
    """
    Shannon entropy  H(θ) = −Σ_i θ_i log θ_i  ≥ 0  (measured in nats).

    Maximised by the uniform distribution (H = log n),
    minimised at vertices of the simplex (H = 0).
    """
    theta = np.clip(theta, eps, 1.0)
    return -np.sum(theta * np.log(theta))


def loss(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    lam: float,
) -> float:
    """
    Entropy-regularized MSE loss:

        L(θ) = (1/m)||Xθ − y||²  −  λ H(θ)

    Minimising L rewards both data fit (low MSE) and diversity (high H).

    Args:
        X:   Feature matrix (m × n).
        y:   Target vector (m,).
        theta: Simplex weight vector (n,).
        lam: Entropy regularisation strength λ ≥ 0.

    Returns:
        Scalar loss L(θ).
    """
    preds = X @ theta
    mse = np.mean((preds - y) ** 2)
    ent = entropy(theta)
    return mse - lam * ent


def loss_components(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    lam: float,
) -> Tuple[float, float, float]:
    """
    Return (total_loss, mse, entropy) separately for diagnostics and plotting.

    Returns:
        (L(θ), MSE(θ), H(θ))
    """
    preds = X @ theta
    mse = float(np.mean((preds - y) ** 2))
    ent = float(entropy(theta))
    return mse - lam * ent, mse, ent
