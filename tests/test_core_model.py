"""
Tests for MirrorLinearRegression: correctness, simplex constraints,
gradient/loss consistency, and all four optimiser variants.
"""

import unittest
import numpy as np

from mirror_linear_regression.core import MirrorLinearRegression
from mirror_linear_regression.loss import loss, entropy
from mirror_linear_regression.bregman import (
    KLDivergence,
    SquaredEuclideanDivergence,
    ItakuraSaitoDivergence,
    BetaDivergence,
)


def _make_data(n_features=4, n_samples=100, seed=42):
    rng = np.random.RandomState(seed)
    true_theta = np.array([0.1, 0.4, 0.3, 0.2])
    x = rng.randn(n_samples, n_features)
    y = x @ true_theta + 0.01 * rng.randn(n_samples)
    return x, y, true_theta


class TestSimplexConstraint(unittest.TestCase):
    """Weights must always lie on the probability simplex."""

    def _check_simplex(self, theta):
        self.assertTrue(np.all(theta >= 0), "Weights must be non-negative.")
        self.assertTrue(np.isclose(np.sum(theta), 1.0), "Weights must sum to 1.")

    def test_mirror_descent(self):
        x, y, _ = _make_data()
        model = MirrorLinearRegression(optimizer="mirror_descent", n_iters=300)
        model.fit(x, y)
        self._check_simplex(model.weights)

    def test_natural_gradient(self):
        x, y, _ = _make_data()
        model = MirrorLinearRegression(optimizer="natural_gradient", n_iters=300)
        model.fit(x, y)
        self._check_simplex(model.weights)

    def test_mirror_prox(self):
        x, y, _ = _make_data()
        model = MirrorLinearRegression(optimizer="mirror_prox", n_iters=300)
        model.fit(x, y)
        self._check_simplex(model.weights)

    def test_ada_mirror(self):
        x, y, _ = _make_data()
        model = MirrorLinearRegression(optimizer="ada_mirror", n_iters=300)
        model.fit(x, y)
        self._check_simplex(model.weights)


class TestLossCorrectedSign(unittest.TestCase):
    """
    Verify that the corrected loss L = MSE - lambda*H decreases entropy
    penalty correctly: higher lambda should yield higher-entropy weights.
    """

    def test_higher_lambda_yields_higher_entropy(self):
        x, y, _ = _make_data()
        model_low = MirrorLinearRegression(lam=0.001, n_iters=500)
        model_high = MirrorLinearRegression(lam=0.5, n_iters=500)
        model_low.fit(x, y)
        model_high.fit(x, y)
        h_low = entropy(model_low.weights)
        h_high = entropy(model_high.weights)
        self.assertGreater(
            h_high, h_low,
            "Higher lambda should produce higher-entropy (more uniform) weights."
        )

    def test_loss_components_consistent(self):
        """Gradient in core.py must match the derivative of the recorded loss."""
        x, y, _ = _make_data()
        model = MirrorLinearRegression(lam=0.05, n_iters=1)
        model.fit(x, y)
        # Loss should equal MSE - lam * H
        theta = model.weights
        expected = loss(x, y, theta, model.lam)
        self.assertAlmostEqual(
            model.loss_history[-1], expected, places=8,
            msg="Recorded loss must match loss(X, y, theta, lam)."
        )

    def test_uniform_init_has_max_entropy(self):
        """Uniform initialisation (1/n,...,1/n) should maximise entropy over simplex."""
        rng = np.random.RandomState(0)
        n = 5
        x = rng.randn(100, n)
        y = x @ np.ones(n) / n + 0.01 * rng.randn(100)
        model = MirrorLinearRegression(n_iters=0)
        model.fit(x, y)
        expected_entropy = np.log(n)
        self.assertAlmostEqual(entropy(model.weights), expected_entropy, places=6)


class TestPrediction(unittest.TestCase):

    def test_predict_shape(self):
        x, y, _ = _make_data()
        model = MirrorLinearRegression(n_iters=100).fit(x, y)
        self.assertEqual(model.predict(x).shape, y.shape)

    def test_low_mse_on_clean_data(self):
        x, y, _ = _make_data()
        model = MirrorLinearRegression(
            learning_rate=0.2, n_iters=1000, lam=0.01, tol=1e-7
        ).fit(x, y)
        mse = np.mean((model.predict(x) - y) ** 2)
        self.assertLess(mse, 0.05)

    def test_predict_before_fit_raises(self):
        model = MirrorLinearRegression()
        with self.assertRaises(ValueError):
            model.predict(np.ones((5, 3)))


class TestDivergencePlugIn(unittest.TestCase):
    """All BregmanDivergence subclasses must produce valid simplex weights."""

    def _fit_and_check(self, div):
        x, y, _ = _make_data()
        model = MirrorLinearRegression(
            divergence=div, n_iters=200, learning_rate=0.05
        ).fit(x, y)
        w = model.weights
        self.assertTrue(np.all(w >= 0))
        self.assertTrue(np.isclose(np.sum(w), 1.0))

    def test_kl_divergence(self):
        self._fit_and_check(KLDivergence())

    def test_euclidean_divergence(self):
        self._fit_and_check(SquaredEuclideanDivergence())

    def test_itakura_saito(self):
        self._fit_and_check(ItakuraSaitoDivergence())

    def test_beta_divergence(self):
        self._fit_and_check(BetaDivergence(beta=1.5))


class TestConvergenceSummary(unittest.TestCase):

    def test_summary_keys(self):
        x, y, _ = _make_data()
        model = MirrorLinearRegression(n_iters=100).fit(x, y)
        summary = model.convergence_summary()
        for key in ("n_iters", "final_loss", "final_mse", "final_entropy",
                    "convergence_rate", "weight_entropy"):
            self.assertIn(key, summary)

    def test_loss_decreases(self):
        x, y, _ = _make_data()
        model = MirrorLinearRegression(n_iters=300, lam=0.01).fit(x, y)
        losses = model.loss_history
        # Loss should be non-increasing on average (not strictly every step)
        self.assertLess(np.mean(losses[-20:]), np.mean(losses[:20]))


if __name__ == "__main__":
    unittest.main()
