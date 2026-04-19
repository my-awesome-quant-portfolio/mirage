"""
Tests for convergence.py: regret bounds, diagnostics, and consistency
of theoretical guarantees with empirical behaviour.
"""

import unittest
import numpy as np

from mirror_linear_regression.convergence import (
    kl_regret_bound,
    euclidean_regret_bound,
    optimal_learning_rate,
    compute_regret,
    convergence_rate_estimate,
    strong_convexity_trajectory,
    print_convergence_report,
)
from mirror_linear_regression.core import MirrorLinearRegression


class TestRegretBounds(unittest.TestCase):

    def test_kl_bound_positive(self):
        self.assertGreater(kl_regret_bound(100, 10), 0.0)

    def test_euclidean_bound_positive(self):
        self.assertGreater(euclidean_regret_bound(100, 10), 0.0)

    def test_kl_better_than_euclidean_large_n(self):
        """KL regret bound should beat Euclidean for large n."""
        for n in [50, 200, 1000]:
            kl = kl_regret_bound(500, n)
            eu = euclidean_regret_bound(500, n)
            self.assertLess(
                kl, eu,
                f"KL should beat Euclidean for n={n}: {kl:.2f} vs {eu:.2f}."
            )

    def test_optimal_eta_positive(self):
        eta = optimal_learning_rate(200, 10)
        self.assertGreater(eta, 0.0)

    def test_optimal_eta_decreases_with_T(self):
        """Larger T => smaller optimal step size."""
        eta1 = optimal_learning_rate(100, 10)
        eta2 = optimal_learning_rate(1000, 10)
        self.assertGreater(eta1, eta2)

    def test_optimal_eta_decreases_with_G(self):
        """Larger gradient bound => smaller optimal step size."""
        eta1 = optimal_learning_rate(100, 10, gradient_bound=1.0)
        eta2 = optimal_learning_rate(100, 10, gradient_bound=2.0)
        self.assertGreater(eta1, eta2)

    def test_kl_bound_with_explicit_eta(self):
        eta = optimal_learning_rate(100, 10)
        bound_opt = kl_regret_bound(100, 10, eta=eta)
        bound_closed = kl_regret_bound(100, 10)
        self.assertAlmostEqual(bound_opt, bound_closed, places=6)


class TestComputeRegret(unittest.TestCase):

    def test_zero_regret_at_optimal(self):
        losses = [1.0, 1.0, 1.0, 1.0]
        regret = compute_regret(losses, optimal_loss=1.0)
        np.testing.assert_allclose(regret, np.zeros(4))

    def test_cumulative_shape(self):
        losses = [2.0, 1.5, 1.2, 1.0]
        regret = compute_regret(losses, optimal_loss=1.0)
        self.assertEqual(regret.shape, (4,))

    def test_non_decreasing_cumulative(self):
        losses = [3.0, 2.0, 1.5, 1.0]
        regret = compute_regret(losses, optimal_loss=0.5)
        # Each element >= previous
        self.assertTrue(np.all(np.diff(regret) >= 0))


class TestConvergenceRateEstimate(unittest.TestCase):

    def test_rate_fast_decay(self):
        """Exponentially decaying losses should yield high estimated rate."""
        t = np.arange(1, 201, dtype=float)
        losses = 10.0 / t  # L_t = 10/t => alpha = 1
        rate = convergence_rate_estimate(list(losses))
        self.assertGreater(rate, 0.7)

    def test_rate_slow_decay(self):
        """Slowly decaying losses should yield low estimated rate."""
        t = np.arange(1, 201, dtype=float)
        losses = 10.0 / np.sqrt(t)  # alpha = 0.5
        rate = convergence_rate_estimate(list(losses))
        self.assertLess(rate, 0.8)

    def test_rate_nan_for_short_history(self):
        rate = convergence_rate_estimate([1.0, 0.9, 0.8])
        self.assertTrue(np.isnan(rate))


class TestStrongConvexityTrajectory(unittest.TestCase):

    def test_monotone_decay(self):
        traj = strong_convexity_trajectory(
            initial_gap=10.0, mu=0.1, eta=0.1, num_steps=50
        )
        self.assertTrue(np.all(np.diff(traj) <= 1e-10))

    def test_starts_at_initial_gap(self):
        traj = strong_convexity_trajectory(
            initial_gap=5.0, mu=0.2, eta=0.2, num_steps=20
        )
        self.assertAlmostEqual(traj[0], 5.0, places=8)

    def test_exponential_rate(self):
        """Ratio of consecutive terms should equal exp(-mu*eta)."""
        mu, eta = 0.1, 0.2
        traj = strong_convexity_trajectory(1.0, mu, eta, 30)
        ratios = traj[1:] / traj[:-1]
        expected = np.exp(-mu * eta)
        np.testing.assert_allclose(ratios, expected, atol=1e-8)


class TestEmpiricalVsTheory(unittest.TestCase):
    """
    Empirical regret from a real training run should not exceed the
    theoretical KL regret bound (up to a generous constant factor).
    """

    def test_empirical_regret_bounded(self):
        rng = np.random.RandomState(0)
        n = 5
        x = rng.randn(200, n)
        true_w = np.ones(n) / n
        y = x @ true_w + 0.01 * rng.randn(200)

        model = MirrorLinearRegression(
            learning_rate=0.1, n_iters=300, lam=0.01
        ).fit(x, y)

        losses = model.loss_history
        opt = min(losses)
        regret = float(np.sum(np.array(losses) - opt))
        bound = kl_regret_bound(len(losses), n)

        # Allow 10x slack for constants not in the asymptotic bound
        self.assertLess(regret, 10.0 * bound)


class TestPrintConvergenceReport(unittest.TestCase):

    def test_runs_without_error(self):
        losses = list(10.0 / np.arange(1, 51))
        # Should not raise
        print_convergence_report(losses, dim=5, eta=0.1, lam=0.01)


if __name__ == "__main__":
    unittest.main()
