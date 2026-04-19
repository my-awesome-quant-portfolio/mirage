"""
Portfolio allocation via entropy-regularised mirror descent.

The PortfolioAllocator learns a simplex weight vector theta over N assets
by fitting a linear predictor of period returns.  The entropy regularisation
encourages diversification: the learned weights resist over-concentration in
any single asset, producing portfolios with higher effective number of bets.
"""

import numpy as np
from typing import Optional

from .core import MirrorLinearRegression
from .bregman import BregmanDivergence
from .utils_math import herfindahl_index, effective_number_of_bets, diversification_ratio


class PortfolioAllocator:
    """
    Entropy-regularised portfolio allocator using mirror descent.

    Fits simplex weights theta over N assets by minimising:

        L(theta) = (1/T)||R theta - r_bar||^2  -  lambda * H(theta)

    where R is the (T x N) return matrix and r_bar = R.mean(axis=1).

    Parameters
    ----------
    learning_rate : float
        Mirror descent step size.
    n_iters : int
        Maximum optimisation iterations.
    lam : float
        Entropy regularisation strength.  Higher values push toward equal
        weighting; lower values allow more concentrated allocations.
    optimizer : str
        Optimiser type (see MirrorLinearRegression).
    divergence : BregmanDivergence or None
        Bregman divergence (default KL).
    verbose : bool
        Print training progress.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iters: int = 1000,
        lam: float = 0.05,
        optimizer: str = "mirror_descent",
        divergence: Optional[BregmanDivergence] = None,
        verbose: bool = False,
    ):
        self.model = MirrorLinearRegression(
            learning_rate=learning_rate,
            n_iters=n_iters,
            lam=lam,
            optimizer=optimizer,
            divergence=divergence,
            verbose=verbose,
        )
        self.theta: Optional[np.ndarray] = None

    def fit(self, returns: np.ndarray) -> "PortfolioAllocator":
        """
        Fit portfolio weights to historical returns.

        Args:
            returns: (T x N) return matrix, T periods, N assets.

        Returns:
            self
        """
        returns = np.asarray(returns, dtype=float)
        target = returns.mean(axis=1)
        self.model.fit(returns, target)
        self.theta = self.model.theta
        return self

    def get_weights(self) -> np.ndarray:
        """Return the learned portfolio weights (sum to 1)."""
        if self.theta is None:
            raise ValueError("Allocator not fitted. Call fit() first.")
        return self.theta.copy()

    def portfolio_diagnostics(self) -> dict:
        """
        Return diversification diagnostics for the learned allocation.

        Returns:
            dict with keys: herfindahl_index, effective_bets, diversification_ratio,
            weight_entropy, convergence_summary.
        """
        w = self.get_weights()
        return {
            "herfindahl_index": herfindahl_index(w),
            "effective_bets": effective_number_of_bets(w),
            "diversification_ratio": diversification_ratio(w),
            "weight_entropy": float(-np.sum(w * np.log(np.clip(w, 1e-12, 1.0)))),
            "convergence": self.model.convergence_summary(),
        }
