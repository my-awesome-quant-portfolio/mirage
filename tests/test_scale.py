"""
tests/test_scale.py — Large-scale, stress, and stress-correctness tests.

Covers:
  - High-dimensional simplex (n up to 500)
  - Large sample counts (m up to 10 000)
  - Ill-conditioned / near-singular feature matrices
  - Correlated features (multicollinearity)
  - Heavy-tailed noise (Student-t)
  - Non-stationary data (regime switches)
  - Sparse ground-truth weights
  - All four optimisers at scale
  - All four Bregman divergences at scale
  - Regret bound verification: empirical regret <= theoretical bound (with slack)
"""

import unittest
import time
import numpy as np

from mirror_linear_regression import MirrorLinearRegression
from mirror_linear_regression.bregman import (
    KLDivergence, SquaredEuclideanDivergence,
    ItakuraSaitoDivergence, BetaDivergence,
)
from mirror_linear_regression.convergence import (
    kl_regret_bound, compute_regret,
)
from mirror_linear_regression.utils_math import entropy, herfindahl_index
from examples.synthetic_datasets import (
    large_alpha_signals,
    large_portfolio,
    correlated_signals,
    heavy_tail_returns,
    sparse_true_weights,
    adversarial_collinear,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _on_simplex(w, atol=1e-8):
    return bool(np.all(w >= -atol) and abs(np.sum(w) - 1.0) < atol)


def _simplex_data(n, m, seed=0, rho=0.0, snr=20.0):
    rng = np.random.RandomState(seed)
    cov = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)
    L = np.linalg.cholesky(cov)
    X = rng.randn(m, n) @ L.T
    w = rng.dirichlet(np.ones(n))
    sig = X @ w
    y = sig + rng.randn(m) * np.sqrt(np.var(sig) / snr)
    return X, y, w


# ---------------------------------------------------------------------------
# Scale: high-dimensional simplex
# ---------------------------------------------------------------------------

class TestHighDimensional(unittest.TestCase):

    def _fit_check(self, n, m, optimizer="mirror_descent", lam=0.01, lr=0.1):
        X, y, _ = _simplex_data(n, m, seed=n)
        model = MirrorLinearRegression(
            optimizer=optimizer, lam=lam, learning_rate=lr,
            n_iters=300, tol=1e-7,
        ).fit(X, y)
        w = model.weights
        self.assertTrue(_on_simplex(w), f"n={n}: weights not on simplex")
        self.assertTrue(np.all(np.isfinite(w)), f"n={n}: non-finite weights")
        return model

    def test_n50(self):
        self._fit_check(50, 250)

    def test_n100(self):
        self._fit_check(100, 500)

    def test_n200(self):
        self._fit_check(200, 1000)

    def test_n500(self):
        self._fit_check(500, 2500, lr=0.05)

    def test_all_optimisers_n100(self):
        configs = [
            ("mirror_descent",   0.10),
            ("natural_gradient", 0.05),
            ("mirror_prox",      0.10),
            ("ada_mirror",       0.20),
        ]
        for opt, lr in configs:
            with self.subTest(optimizer=opt):
                self._fit_check(100, 500, optimizer=opt, lr=lr)

    def test_large_alpha_dataset(self):
        """large_alpha_signals: n=500, m=10 000."""
        X, y, _ = large_alpha_signals(n_signals=500, n_samples=10_000, seed=100)
        model = MirrorLinearRegression(
            lam=0.01, learning_rate=0.05, n_iters=200, tol=1e-6,
        ).fit(X, y)
        self.assertTrue(_on_simplex(model.weights))

    def test_loss_decreases_high_dim(self):
        """Loss should trend downward even in high dimensions."""
        X, y, _ = _simplex_data(200, 1000, seed=7)
        model = MirrorLinearRegression(lam=0.01, n_iters=300).fit(X, y)
        losses = model.loss_history
        self.assertLess(np.mean(losses[-30:]), np.mean(losses[:30]))


# ---------------------------------------------------------------------------
# Scale: large sample counts
# ---------------------------------------------------------------------------

class TestLargeSamples(unittest.TestCase):

    def test_m1000(self):
        X, y, _ = _simplex_data(20, 1000)
        model = MirrorLinearRegression(lam=0.01, n_iters=300).fit(X, y)
        self.assertTrue(_on_simplex(model.weights))

    def test_m5000(self):
        X, y, _ = _simplex_data(20, 5000, seed=5)
        model = MirrorLinearRegression(lam=0.01, n_iters=200).fit(X, y)
        self.assertTrue(_on_simplex(model.weights))

    def test_m10000(self):
        X, y, _ = _simplex_data(20, 10000, seed=10)
        model = MirrorLinearRegression(lam=0.01, n_iters=150).fit(X, y)
        self.assertTrue(_on_simplex(model.weights))

    def test_large_portfolio_dataset(self):
        """large_portfolio: T=1000, N=200."""
        ret = large_portfolio(T=1000, N=200, seed=200)
        from mirror_linear_regression import PortfolioAllocator
        alloc = PortfolioAllocator(lam=0.05, n_iters=200).fit(ret)
        w = alloc.get_weights()
        self.assertEqual(w.shape[0], 200)
        self.assertTrue(_on_simplex(w))

    def test_runtime_m10000_under_10s(self):
        """Training on m=10 000, n=20 should finish well under 10 seconds."""
        X, y, _ = _simplex_data(20, 10000, seed=99)
        t0 = time.perf_counter()
        MirrorLinearRegression(lam=0.01, n_iters=200).fit(X, y)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 10.0, f"Too slow: {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Robustness: ill-conditioning and collinearity
# ---------------------------------------------------------------------------

class TestRobustness(unittest.TestCase):

    def test_collinear_features(self):
        """Highly correlated features (rho=0.95) — should not diverge."""
        X, y, _ = correlated_signals(n_signals=20, n_obs=300, rho=0.95, seed=300)
        model = MirrorLinearRegression(lam=0.02, n_iters=400).fit(X, y)
        self.assertTrue(_on_simplex(model.weights))
        self.assertTrue(all(np.isfinite(l) for l in model.loss_history))

    def test_adversarial_collinear(self):
        """Near-duplicate columns — numerical stability."""
        X, y, _ = adversarial_collinear(n=30, n_dupes=10, n_obs=300, seed=700)
        model = MirrorLinearRegression(lam=0.05, n_iters=300).fit(X, y)
        self.assertTrue(_on_simplex(model.weights))
        self.assertTrue(all(np.isfinite(l) for l in model.loss_history))

    def test_heavy_tail_noise(self):
        """Student-t noise (df=3) — model should not crash."""
        ret = heavy_tail_returns(T=500, N=20, df=3, seed=400)
        from mirror_linear_regression import PortfolioAllocator
        alloc = PortfolioAllocator(lam=0.05, n_iters=300).fit(ret)
        self.assertTrue(_on_simplex(alloc.get_weights()))

    def test_sparse_ground_truth(self):
        """
        Sparse true weights (5 non-zeros out of 50).
        MIRAGE++ should fit well even if it won't recover sparsity
        (entropy pushes toward uniformity, not sparsity — by design).
        """
        X, y, _ = sparse_true_weights(n=50, k=5, n_obs=400, seed=600)
        model = MirrorLinearRegression(lam=0.001, n_iters=400).fit(X, y)
        self.assertTrue(_on_simplex(model.weights))
        mse = float(np.mean((model.predict(X) - y) ** 2))
        self.assertLess(mse, 1.0, "MSE should be reasonable even on sparse GT")

    def test_single_feature(self):
        """Edge case: n=1, simplex is a single point {1}."""
        X = np.ones((100, 1))
        y = np.ones(100) + 0.01 * np.random.RandomState(0).randn(100)
        model = MirrorLinearRegression(n_iters=50).fit(X, y)
        np.testing.assert_allclose(model.weights, [1.0], atol=1e-6)

    def test_two_features(self):
        """Edge case: n=2, simplex is the unit interval [0, 1]."""
        rng = np.random.RandomState(0)
        X = rng.randn(100, 2)
        y = X @ np.array([0.7, 0.3]) + 0.01 * rng.randn(100)
        model = MirrorLinearRegression(lam=0.001, n_iters=500, tol=1e-9).fit(X, y)
        self.assertTrue(_on_simplex(model.weights))

    def test_zero_variance_feature(self):
        """Feature column of all ones (zero variance) — should not crash."""
        rng = np.random.RandomState(5)
        X = rng.randn(100, 5)
        X[:, 2] = 1.0   # constant column
        y = X @ rng.dirichlet(np.ones(5)) + 0.01 * rng.randn(100)
        model = MirrorLinearRegression(lam=0.05, n_iters=200).fit(X, y)
        self.assertTrue(_on_simplex(model.weights))


# ---------------------------------------------------------------------------
# Bregman divergence plug-in at scale
# ---------------------------------------------------------------------------

class TestDivergencesAtScale(unittest.TestCase):

    def _check_div(self, div, lr=0.05):
        X, y, _ = _simplex_data(30, 300, seed=42)
        model = MirrorLinearRegression(
            divergence=div, lam=0.01, learning_rate=lr, n_iters=300,
        ).fit(X, y)
        self.assertTrue(_on_simplex(model.weights))
        self.assertTrue(all(np.isfinite(l) for l in model.loss_history))

    def test_kl_scale(self):
        self._check_div(KLDivergence(), lr=0.1)

    def test_euclidean_scale(self):
        self._check_div(SquaredEuclideanDivergence(), lr=0.05)

    def test_itakura_saito_scale(self):
        self._check_div(ItakuraSaitoDivergence(), lr=0.05)

    def test_beta_15_scale(self):
        self._check_div(BetaDivergence(beta=1.5), lr=0.03)

    def test_beta_20_scale(self):
        self._check_div(BetaDivergence(beta=2.0), lr=0.02)


# ---------------------------------------------------------------------------
# Convergence: regret bound verification
# ---------------------------------------------------------------------------

class TestRegretBounds(unittest.TestCase):

    def test_empirical_regret_bounded_n10(self):
        """
        Empirical cumulative regret must not exceed 20x the theoretical
        KL regret bound (slack for hidden constants and non-zero lam).
        """
        n, m = 10, 500
        X, y, _ = _simplex_data(n, m, seed=0)
        model = MirrorLinearRegression(lam=0.0, n_iters=300, tol=0).fit(X, y)
        losses = model.loss_history
        opt = min(losses)
        regret = float(np.sum(np.array(losses) - opt))
        bound = kl_regret_bound(len(losses), n)
        self.assertLess(regret, 20.0 * bound,
                        f"regret={regret:.3f} > 20x bound={20*bound:.3f}")

    def test_empirical_regret_bounded_n50(self):
        n, m = 50, 1000
        X, y, _ = _simplex_data(n, m, seed=1)
        model = MirrorLinearRegression(lam=0.0, n_iters=300, tol=0).fit(X, y)
        losses = model.loss_history
        opt = min(losses)
        regret = float(np.sum(np.array(losses) - opt))
        bound = kl_regret_bound(len(losses), n)
        self.assertLess(regret, 20.0 * bound)

    def test_kl_better_than_euclidean_n100(self):
        """
        For n=100, the KL regret bound should be substantially smaller
        than the Euclidean bound (log n vs sqrt n advantage).
        """
        from mirror_linear_regression.convergence import euclidean_regret_bound
        n, T = 100, 500
        kl_b = kl_regret_bound(T, n)
        eu_b = euclidean_regret_bound(T, n)
        self.assertLess(kl_b, eu_b,
                        f"KL bound {kl_b:.2f} should beat Euclidean {eu_b:.2f} at n=100")

    def test_higher_lambda_faster_convergence(self):
        """
        Stronger entropy regularisation -> stronger convexity -> fewer iters.
        """
        X, y, _ = _simplex_data(20, 300, seed=42)
        model_low = MirrorLinearRegression(lam=0.001, n_iters=600, tol=1e-8).fit(X, y)
        model_hi  = MirrorLinearRegression(lam=0.5,   n_iters=600, tol=1e-8).fit(X, y)
        self.assertLess(model_hi.n_iters_run, model_low.n_iters_run,
                        "Higher lambda should converge in fewer iterations.")


# ---------------------------------------------------------------------------
# Diversity / weight statistics at scale
# ---------------------------------------------------------------------------

class TestDiversityAtScale(unittest.TestCase):

    def test_entropy_increases_with_lambda(self):
        """Across a range of lambdas, entropy should be monotonically non-decreasing."""
        X, y, _ = _simplex_data(20, 300, seed=0)
        lambdas = [0.001, 0.01, 0.05, 0.1, 0.5]
        entropies = []
        for lam in lambdas:
            m = MirrorLinearRegression(lam=lam, n_iters=500, tol=1e-8).fit(X, y)
            entropies.append(entropy(m.weights))
        # Monotone non-decreasing
        for i in range(len(entropies) - 1):
            self.assertLessEqual(entropies[i], entropies[i + 1] + 1e-3,
                                 f"Entropy not non-decreasing at lambda index {i}")

    def test_hhi_decreases_with_lambda(self):
        """HHI (concentration) should decrease as lambda increases."""
        X, y, _ = _simplex_data(20, 300, seed=1)
        lambdas = [0.001, 0.05, 0.5]
        hhis = [
            herfindahl_index(
                MirrorLinearRegression(lam=lam, n_iters=500, tol=1e-8).fit(X, y).weights
            )
            for lam in lambdas
        ]
        self.assertGreater(hhis[0], hhis[-1],
                           "HHI should be lower (less concentrated) for higher lambda.")

    def test_uniform_is_max_entropy(self):
        """Uniform weights should have the highest possible entropy = log(n)."""
        n = 30
        uniform = np.ones(n) / n
        self.assertAlmostEqual(entropy(uniform), np.log(n), places=8)

    def test_portfolio_diversity_metrics(self):
        """PortfolioAllocator diagnostics should return valid values."""
        from mirror_linear_regression import PortfolioAllocator
        from mirror_linear_regression.utils_math import diversification_ratio
        ret = large_portfolio(T=200, N=10, seed=0)
        diag = PortfolioAllocator(lam=0.1, n_iters=300).fit(ret).portfolio_diagnostics()
        self.assertGreater(diag["effective_bets"], 1.0)
        self.assertGreater(diag["diversification_ratio"], 0.0)
        self.assertLessEqual(diag["diversification_ratio"], 1.0)


# ---------------------------------------------------------------------------
# Convergence diagnostics at scale
# ---------------------------------------------------------------------------

class TestConvergenceDiagnostics(unittest.TestCase):

    def test_convergence_summary_all_keys(self):
        X, y, _ = _simplex_data(20, 200)
        model = MirrorLinearRegression(n_iters=200).fit(X, y)
        s = model.convergence_summary()
        for key in ("n_iters", "final_loss", "final_mse", "final_entropy",
                    "convergence_rate", "weight_entropy"):
            self.assertIn(key, s)
        self.assertGreater(s["weight_entropy"], 0.0)

    def test_loss_history_length(self):
        X, y, _ = _simplex_data(10, 100)
        model = MirrorLinearRegression(n_iters=50, tol=0).fit(X, y)
        self.assertEqual(len(model.loss_history), 50)

    def test_mse_history_non_negative(self):
        X, y, _ = _simplex_data(10, 100)
        model = MirrorLinearRegression(n_iters=100).fit(X, y)
        self.assertTrue(all(v >= 0 for v in model.mse_history))

    def test_entropy_history_non_negative(self):
        X, y, _ = _simplex_data(10, 100)
        model = MirrorLinearRegression(n_iters=100).fit(X, y)
        self.assertTrue(all(v >= 0 for v in model.entropy_history))


if __name__ == "__main__":
    unittest.main(verbosity=2)
