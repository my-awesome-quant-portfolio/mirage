"""
Alpha signal combination via entropy-regularised mirror descent.

The AlphaSignalCombiner learns a simplex weight vector over N trading signals
by minimising the entropy-regularised prediction error against realised returns.
Entropy regularisation prevents the model from over-concentrating on a single
signal (a common failure mode in signal combination when signals are correlated).
"""

import numpy as np
from typing import Optional

from .core import MirrorLinearRegression
from .bregman import BregmanDivergence
from .utils_math import herfindahl_index, effective_number_of_bets, diversification_ratio


class AlphaSignalCombiner:
    """
    Entropy-regularised alpha signal combiner using mirror descent.

    Fits simplex weights theta over N signals by minimising:

        L(theta) = (1/T)||X theta - y||^2  -  lambda * H(theta)

    where X is the (T x N) signal matrix and y is the realised return vector.

    Parameters
    ----------
    learning_rate : float
        Mirror descent step size.
    n_iters : int
        Maximum optimisation iterations.
    lam : float
        Entropy regularisation strength.  Higher values produce more
        uniform signal weighting; lower values allow concentrated bets.
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

    def fit(self, x: np.ndarray, y: np.ndarray) -> "AlphaSignalCombiner":
        """
        Fit signal weights to historical data.

        Args:
            x: (T x N) signal matrix, T periods, N signals.
            y: (T,) realised returns or labels.

        Returns:
            self
        """
        self.model.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        self.theta = self.model.theta
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Compute combined signal prediction  y_hat = X theta.

        Args:
            x: (T x N) signal matrix.

        Returns:
            ndarray of shape (T,).
        """
        return self.model.predict(x)

    def get_signal_weights(self) -> np.ndarray:
        """Return the learned signal weights (sum to 1)."""
        if self.theta is None:
            raise ValueError("Combiner not fitted. Call fit() first.")
        return self.theta.copy()

    def signal_diagnostics(self) -> dict:
        """
        Return concentration and convergence diagnostics.

        Returns:
            dict with keys: herfindahl_index, effective_signals,
            diversification_ratio, weight_entropy, convergence.
        """
        w = self.get_signal_weights()
        return {
            "herfindahl_index": herfindahl_index(w),
            "effective_signals": effective_number_of_bets(w),
            "diversification_ratio": diversification_ratio(w),
            "weight_entropy": float(-np.sum(w * np.log(np.clip(w, 1e-12, 1.0)))),
            "convergence": self.model.convergence_summary(),
        }
