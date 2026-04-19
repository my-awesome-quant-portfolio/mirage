"""
MIRAGE++: Mirror Descent Entropy-Regularized Linear Regression.

The model minimises the entropy-regularised objective:

    L(theta; X, y) = (1/m)||X theta - y||^2  -  lambda * H(theta)
    subject to  theta in Delta_{n-1}

where H(theta) = -sum_i theta_i log theta_i >= 0 is the Shannon entropy and
Delta_{n-1} = {theta >= 0 : sum theta_i = 1} is the probability simplex.

The entropy term is *subtracted*: minimising L rewards both data fit (MSE down)
and weight diversity (H up).  With lambda > 0 the solution is always in the
interior of the simplex (no sparsity), making weights interpretable as
portfolio allocations or signal importance scores.

Gradient:
    nabla_theta L = (2/m) X^T(X theta - y)  +  lambda * (1 + log theta)

The term lambda*(1 + log theta) grows large and negative for small theta_i,
acting as a barrier that prevents weights from collapsing to zero.

Supported optimisers
--------------------
'mirror_descent'    Exponentiated gradient with any BregmanDivergence (default KL).
                    Update: theta ∝ theta * exp(-eta * grad).
                    Regret: O(sqrt(T log n)) with KL.

'natural_gradient'  Fisher-Rao Riemannian NGD on the simplex.
                    Update: theta <- proj_Delta(theta - eta * diag(theta)*(grad - theta^T grad)).
                    First-order equivalent to KL mirror descent; differs at O(eta^2).

'mirror_prox'       Extragradient (Nemirovski 2004).
                    Achieves O(1/T) on smooth objectives vs O(1/sqrt(T)).

'ada_mirror'        Adaptive mirror descent with per-coordinate learning rates.
                    Useful when gradient magnitudes vary across features.

References:
  Kivinen & Warmuth (1997). Exponentiated gradient versus gradient descent.
  Amari (1998). Natural gradient works efficiently in learning.
  Nemirovski (2004). Prox-method with rate of convergence O(1/T).
  Duchi, Hazan & Singer (2011). Adaptive subgradient methods.
"""

import numpy as np
from typing import Literal, Optional

from .loss import loss, loss_components
from .optim import (
    mirror_descent_step,
    mirror_prox_step,
    natural_gradient_descent_step,
    AdaMirror,
)
from .bregman import BregmanDivergence, KLDivergence

OptimizerType = Literal["mirror_descent", "natural_gradient", "mirror_prox", "ada_mirror"]


class MirrorLinearRegression:
    """
    Entropy-regularised linear regression on the probability simplex,
    optimised via mirror descent and its variants.

    Parameters
    ----------
    learning_rate : float
        Step size eta for the mirror descent update.
    n_iters : int
        Maximum optimisation iterations.
    lam : float
        Entropy regularisation strength lambda >= 0.  Higher values push
        weights toward uniformity (more diversification).
    tol : float
        Early-stopping tolerance on absolute loss improvement.
    optimizer : str
        Solver: 'mirror_descent' | 'natural_gradient' | 'mirror_prox' | 'ada_mirror'.
    divergence : BregmanDivergence or None
        Bregman divergence for 'mirror_descent' and 'mirror_prox'.
        Ignored for 'natural_gradient' and 'ada_mirror'.
        Defaults to KLDivergence (exponentiated gradient).
    verbose : bool
        Print convergence diagnostics every 100 iterations.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iters: int = 1000,
        lam: float = 0.01,
        tol: float = 1e-6,
        optimizer: OptimizerType = "mirror_descent",
        divergence: Optional[BregmanDivergence] = None,
        verbose: bool = False,
    ):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lam = lam
        self.tol = tol
        self.optimizer = optimizer
        self.divergence = divergence if divergence is not None else KLDivergence()
        self.verbose = verbose

        self.theta: Optional[np.ndarray] = None
        self.loss_history: list = []
        self.mse_history: list = []
        self.entropy_history: list = []
        self.n_iters_run: int = 0

    def _gradient(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Gradient of L(theta) = MSE(theta) - lambda * H(theta):

            nabla L = (2/m) X^T(X theta - y)  +  lambda * (1 + log theta)

        The entropy gradient term lambda*(1 + log theta) is positive and
        grows as theta shrinks, preventing weights from reaching zero.
        """
        m = len(y)
        preds = x @ theta
        grad_mse = (2.0 / m) * x.T @ (preds - y)
        # nabla(-lambda * H) = lambda * (1 + log theta)
        grad_entropy = 1.0 + np.log(np.clip(theta, 1e-8, 1.0))
        return grad_mse + self.lam * grad_entropy

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MirrorLinearRegression":
        """
        Fit the model to (X, y) using the selected optimiser.

        Initialises theta to the uniform distribution (maximum entropy start),
        then iterates mirror descent until convergence or n_iters exhausted.

        Parameters
        ----------
        x : array-like, shape (m, n)
            Feature matrix.
        y : array-like, shape (m,)
            Target vector.

        Returns
        -------
        self
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m, n = x.shape

        # Uniform initialisation: maximum-entropy start on the simplex
        self.theta = np.ones(n) / n
        self.loss_history.clear()
        self.mse_history.clear()
        self.entropy_history.clear()
        self.n_iters_run = 0

        ada = AdaMirror(eta=self.learning_rate) if self.optimizer == "ada_mirror" else None
        prev_loss = float("inf")

        for i in range(self.n_iters):
            grad = self._gradient(x, y, self.theta)

            if self.optimizer == "mirror_descent":
                self.theta = mirror_descent_step(
                    grad, self.theta, self.learning_rate, self.divergence
                )
            elif self.optimizer == "natural_gradient":
                self.theta = natural_gradient_descent_step(
                    grad, self.theta, self.learning_rate
                )
            elif self.optimizer == "mirror_prox":
                def grad_fn(t, _x=x, _y=y):
                    return self._gradient(_x, _y, t)
                self.theta = mirror_prox_step(
                    grad_fn, self.theta, self.learning_rate, self.divergence
                )
            elif self.optimizer == "ada_mirror":
                self.theta = ada.step(grad, self.theta)
            else:
                raise ValueError(
                    f"Unknown optimizer '{self.optimizer}'. "
                    "Choose from: mirror_descent, natural_gradient, mirror_prox, ada_mirror."
                )

            total, mse, ent = loss_components(x, y, self.theta, self.lam)
            self.loss_history.append(total)
            self.mse_history.append(mse)
            self.entropy_history.append(ent)
            self.n_iters_run = i + 1

            if self.verbose and i % 100 == 0:
                print(
                    f"  [{self.optimizer}] iter {i:4d}: "
                    f"loss={total:.6f}  mse={mse:.6f}  H={ent:.4f}"
                )

            if abs(prev_loss - total) < self.tol:
                if self.verbose:
                    print(
                        f"  Early stopping at iter {i}: "
                        f"delta_loss={abs(prev_loss - total):.2e} < tol={self.tol:.2e}"
                    )
                break
            prev_loss = total

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Compute predictions  y_hat = X theta.

        Parameters
        ----------
        x : array-like, shape (m, n)

        Returns
        -------
        ndarray, shape (m,)
        """
        if self.theta is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return np.asarray(x, dtype=float) @ self.theta

    @property
    def weights(self) -> np.ndarray:
        """Learned weight vector theta on the probability simplex."""
        if self.theta is None:
            raise ValueError("Model is not fitted yet.")
        return self.theta.copy()

    def convergence_summary(self) -> dict:
        """
        Return a dictionary of convergence diagnostics.

        Keys: n_iters, final_loss, final_mse, final_entropy,
              convergence_rate (empirical alpha in L_t ~ t^{-alpha}),
              weight_entropy (H(theta) of the final weights).
        """
        from .convergence import convergence_rate_estimate

        return {
            "n_iters": self.n_iters_run,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "final_mse": self.mse_history[-1] if self.mse_history else None,
            "final_entropy": self.entropy_history[-1] if self.entropy_history else None,
            "convergence_rate": convergence_rate_estimate(self.loss_history),
            "weight_entropy": float(
                -np.sum(self.theta * np.log(np.clip(self.theta, 1e-12, 1.0)))
            ),
        }
