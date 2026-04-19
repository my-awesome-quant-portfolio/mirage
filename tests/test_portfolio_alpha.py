"""
Tests for PortfolioAllocator and AlphaSignalCombiner.

Coverage:
  - Simplex constraint for all 4 optimizers
  - portfolio_diagnostics / signal_diagnostics contract
  - Diversity increases with lambda (entropy regularization direction)
  - All BregmanDivergence variants
  - Multi-asset / high-dimensional inputs
  - Predict shape / dtype
"""

import numpy as np
import pytest

from mirror_linear_regression.portfolio_alloc import PortfolioAllocator
from mirror_linear_regression.finance_alpha import AlphaSignalCombiner
from mirror_linear_regression.bregman import (
    KLDivergence,
    SquaredEuclideanDivergence,
    ItakuraSaitoDivergence,
    BetaDivergence,
)
from mirror_linear_regression.utils_math import effective_number_of_bets

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def returns_data():
    rng = np.random.RandomState(42)
    T, N = 150, 6
    return rng.randn(T, N) * 0.02 + 0.001


@pytest.fixture
def alpha_data():
    rng = np.random.RandomState(7)
    T, N = 200, 5
    X = rng.randn(T, N)
    w_true = rng.dirichlet(np.ones(N))
    y = X @ w_true + 0.01 * rng.randn(T)
    return X, y, w_true


# ---------------------------------------------------------------------------
# PortfolioAllocator: simplex constraint
# ---------------------------------------------------------------------------

class TestPortfolioAllocatorSimplex:
    @pytest.mark.parametrize("optimizer", [
        "mirror_descent", "natural_gradient", "mirror_prox", "ada_mirror"
    ])
    def test_weights_on_simplex(self, returns_data, optimizer):
        lr = 0.05 if optimizer == "natural_gradient" else 0.1
        alloc = PortfolioAllocator(learning_rate=lr, n_iters=300,
                                   lam=0.05, optimizer=optimizer)
        alloc.fit(returns_data)
        w = alloc.get_weights()
        assert w.shape == (returns_data.shape[1],), "wrong weight shape"
        assert np.all(w >= -1e-10), "negative weights"
        assert abs(w.sum() - 1.0) < 1e-8, "weights don't sum to 1"

    @pytest.mark.parametrize("divergence", [
        KLDivergence(),
        SquaredEuclideanDivergence(),
        ItakuraSaitoDivergence(),
        BetaDivergence(beta=1.5),
    ])
    def test_all_divergences(self, returns_data, divergence):
        alloc = PortfolioAllocator(n_iters=200, lam=0.05, divergence=divergence)
        alloc.fit(returns_data)
        w = alloc.get_weights()
        assert abs(w.sum() - 1.0) < 1e-8
        assert np.all(w >= -1e-10)

    def test_higher_lambda_more_diversified(self):
        # Use AlphaSignalCombiner where one signal dominates (low lambda concentrates on it;
        # high lambda spreads across all signals)
        rng = np.random.RandomState(77)
        n, T = 8, 300
        X = rng.randn(T, n)
        # Strong single-signal ground truth: weight 0.9 on signal 0
        w = np.zeros(n); w[0] = 0.9; w[1:] = 0.1 / (n - 1)
        y = X @ w + 0.005 * rng.randn(T)
        c_lo = AlphaSignalCombiner(n_iters=600, lam=0.001, learning_rate=0.15)
        c_hi = AlphaSignalCombiner(n_iters=600, lam=0.5,   learning_rate=0.15)
        c_lo.fit(X, y); c_hi.fit(X, y)
        enb_lo = effective_number_of_bets(c_lo.get_signal_weights())
        enb_hi = effective_number_of_bets(c_hi.get_signal_weights())
        assert enb_hi > enb_lo, f"high lambda ENB={enb_hi:.2f} should exceed low lambda ENB={enb_lo:.2f}"


# ---------------------------------------------------------------------------
# PortfolioAllocator: diagnostics contract
# ---------------------------------------------------------------------------

class TestPortfolioAllocatorDiagnostics:
    def test_diagnostics_keys(self, returns_data):
        alloc = PortfolioAllocator(n_iters=200, lam=0.05)
        alloc.fit(returns_data)
        diag = alloc.portfolio_diagnostics()
        for key in ("herfindahl_index", "effective_bets",
                    "diversification_ratio", "weight_entropy"):
            assert key in diag, f"missing key: {key}"

    def test_hhi_in_range(self, returns_data):
        alloc = PortfolioAllocator(n_iters=300, lam=0.05)
        alloc.fit(returns_data)
        diag = alloc.portfolio_diagnostics()
        n = returns_data.shape[1]
        assert 1 / n - 1e-6 <= diag["herfindahl_index"] <= 1.0 + 1e-6

    def test_not_fitted_raises(self):
        alloc = PortfolioAllocator()
        with pytest.raises(ValueError, match="not fitted"):
            alloc.get_weights()


# ---------------------------------------------------------------------------
# AlphaSignalCombiner: simplex + weight recovery
# ---------------------------------------------------------------------------

class TestAlphaSignalCombiner:
    @pytest.mark.parametrize("optimizer", [
        "mirror_descent", "natural_gradient", "mirror_prox", "ada_mirror"
    ])
    def test_weights_on_simplex(self, alpha_data, optimizer):
        X, y, _ = alpha_data
        lr = 0.05 if optimizer == "natural_gradient" else 0.15
        combiner = AlphaSignalCombiner(learning_rate=lr, n_iters=300,
                                       lam=0.05, optimizer=optimizer)
        combiner.fit(X, y)
        w = combiner.get_signal_weights()
        assert w.shape == (X.shape[1],)
        assert abs(w.sum() - 1.0) < 1e-8
        assert np.all(w >= -1e-10)

    def test_weight_recovery_low_noise(self, alpha_data):
        X, y, w_true = alpha_data
        combiner = AlphaSignalCombiner(learning_rate=0.15, n_iters=600, lam=0.01)
        combiner.fit(X, y)
        w_hat = combiner.get_signal_weights()
        cos = float(np.dot(w_hat, w_true) /
                    (np.linalg.norm(w_hat) * np.linalg.norm(w_true)))
        assert cos > 0.80, f"cosine similarity {cos:.3f} too low"

    def test_predict_shape(self, alpha_data):
        X, y, _ = alpha_data
        combiner = AlphaSignalCombiner(n_iters=200, lam=0.05)
        combiner.fit(X, y)
        pred = combiner.predict(X)
        assert pred.shape == y.shape

    def test_diagnostics_keys(self, alpha_data):
        X, y, _ = alpha_data
        combiner = AlphaSignalCombiner(n_iters=200, lam=0.05)
        combiner.fit(X, y)
        diag = combiner.signal_diagnostics()
        for key in ("herfindahl_index", "effective_signals",
                    "diversification_ratio", "weight_entropy"):
            assert key in diag, f"missing key: {key}"

    def test_not_fitted_raises(self):
        combiner = AlphaSignalCombiner()
        with pytest.raises(ValueError, match="not fitted"):
            combiner.get_signal_weights()


# ---------------------------------------------------------------------------
# Scale / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_asset(self):
        rng = np.random.RandomState(0)
        returns = rng.randn(100, 1) * 0.02
        alloc = PortfolioAllocator(n_iters=200, lam=0.05)
        alloc.fit(returns)
        w = alloc.get_weights()
        assert abs(w[0] - 1.0) < 1e-6

    def test_many_assets(self):
        rng = np.random.RandomState(5)
        returns = rng.randn(200, 50) * 0.02
        alloc = PortfolioAllocator(n_iters=300, lam=0.05)
        alloc.fit(returns)
        w = alloc.get_weights()
        assert abs(w.sum() - 1.0) < 1e-7
        assert np.all(w >= -1e-10)

    def test_high_dimensional_signals(self):
        rng = np.random.RandomState(11)
        X = rng.randn(300, 40)
        w_true = rng.dirichlet(np.ones(40))
        y = X @ w_true + 0.05 * rng.randn(300)
        combiner = AlphaSignalCombiner(n_iters=400, lam=0.01)
        combiner.fit(X, y)
        w = combiner.get_signal_weights()
        assert abs(w.sum() - 1.0) < 1e-7

    def test_correlated_signals_stable(self):
        rng = np.random.RandomState(99)
        n, T = 10, 200
        rho = 0.9
        cov = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)
        L = np.linalg.cholesky(cov)
        X = rng.randn(T, n) @ L.T
        w_true = rng.dirichlet(np.ones(n))
        y = X @ w_true + 0.05 * rng.randn(T)
        combiner = AlphaSignalCombiner(n_iters=400, lam=0.05)
        combiner.fit(X, y)
        w = combiner.get_signal_weights()
        assert abs(w.sum() - 1.0) < 1e-7
        assert np.all(np.isfinite(w))
