"""
Tests for all BregmanDivergence subclasses:
- divergence axioms (non-negativity, zero at equality)
- mirror step preserves simplex constraint
- plug-in via get_divergence factory
- numerical properties of each divergence class
"""

import unittest
import numpy as np

from mirror_linear_regression.bregman import (
    BregmanDivergence,
    KLDivergence,
    SquaredEuclideanDivergence,
    ItakuraSaitoDivergence,
    BetaDivergence,
    get_divergence,
)


def _simplex_point(n, seed):
    rng = np.random.RandomState(seed)
    x = rng.dirichlet(np.ones(n))
    return x


class BregmanAxiomsMixin:
    """
    Reusable tests that must hold for every BregmanDivergence subclass.
    Subclasses set self.div and optionally self.n.
    """

    n = 5

    def _p(self, seed=0):
        return _simplex_point(self.n, seed)

    def test_divergence_non_negative(self):
        p, q = self._p(0), self._p(1)
        self.assertGreaterEqual(self.div.divergence(p, q), -1e-8)

    def test_divergence_zero_at_equality(self):
        p = self._p(0)
        d = self.div.divergence(p, p)
        self.assertAlmostEqual(d, 0.0, places=6)

    def test_mirror_step_on_simplex(self):
        theta = self._p(0)
        grad = np.random.RandomState(42).randn(self.n)
        theta_new = self.div.mirror_step(grad, theta, eta=0.1)
        self.assertTrue(np.all(theta_new >= -1e-10))
        self.assertAlmostEqual(float(np.sum(theta_new)), 1.0, places=8)

    def test_mirror_step_decreases_along_gradient(self):
        """A step in the negative gradient direction should reduce the linear term."""
        rng = np.random.RandomState(7)
        theta = self._p(3)
        grad = rng.randn(self.n)
        theta_new = self.div.mirror_step(grad, theta, eta=0.05)
        # Linear improvement: <grad, theta_new> should be < <grad, theta>
        self.assertLess(
            float(np.dot(grad, theta_new)),
            float(np.dot(grad, theta)) + 1e-6,
        )

    def test_generator_finite(self):
        p = self._p(0)
        val = self.div.generator(p)
        self.assertTrue(np.isfinite(val))

    def test_grad_generator_shape(self):
        p = self._p(0)
        g = self.div.grad_generator(p)
        self.assertEqual(g.shape, p.shape)

    def test_inverse_mirror_roundtrip(self):
        """(nabla phi)^-1(nabla phi(p)) should recover p up to projection."""
        p = self._p(0)
        z = self.div.grad_generator(p)
        p_recovered = self.div.inverse_mirror(z)
        # After projection, should match p
        p_proj = self.div.project(p_recovered)
        np.testing.assert_allclose(p_proj, p, atol=1e-6)


class TestKLDivergence(BregmanAxiomsMixin, unittest.TestCase):
    def setUp(self):
        self.div = KLDivergence()

    def test_kl_known_value(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        expected = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
        self.assertAlmostEqual(self.div.divergence(p, q), expected, places=8)

    def test_exponentiated_gradient_form(self):
        """KL mirror step should equal theta * exp(-eta * grad) / Z."""
        theta = np.array([0.4, 0.3, 0.3])
        grad = np.array([0.2, -0.1, 0.3])
        eta = 0.5
        expected = theta * np.exp(-eta * grad)
        expected /= expected.sum()
        result = self.div.mirror_step(grad, theta, eta)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestSquaredEuclideanDivergence(BregmanAxiomsMixin, unittest.TestCase):
    def setUp(self):
        self.div = SquaredEuclideanDivergence()

    def test_known_value(self):
        p = np.array([0.4, 0.4, 0.2])
        q = np.array([0.3, 0.5, 0.2])
        # Bregman divergence with phi = 0.5||p||^2 gives D(p||q) = 0.5||p-q||^2
        expected = 0.5 * float(np.sum((p - q) ** 2))
        self.assertAlmostEqual(self.div.divergence(p, q), expected, places=8)

    def test_projected_gradient_form(self):
        """Euclidean mirror step = project_simplex(theta - eta * grad)."""
        from mirror_linear_regression.utils_math import project_simplex
        theta = np.array([0.4, 0.3, 0.3])
        grad = np.array([0.2, -0.1, 0.3])
        eta = 0.1
        expected = project_simplex(theta - eta * grad)
        result = self.div.mirror_step(grad, theta, eta)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestItakuraSaitoDivergence(BregmanAxiomsMixin, unittest.TestCase):
    def setUp(self):
        self.div = ItakuraSaitoDivergence()

    def test_known_value(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        expected = float(np.sum(p / q - np.log(p / q) - 1.0))
        self.assertAlmostEqual(self.div.divergence(p, q), expected, places=8)

    def test_scale_property(self):
        """IS divergence is not scale-invariant (unlike KL), just non-negative."""
        p = np.array([0.5, 0.5])
        q = np.array([0.3, 0.7])
        self.assertGreater(self.div.divergence(p, q), 0.0)


class TestBetaDivergence(BregmanAxiomsMixin, unittest.TestCase):
    def setUp(self):
        self.div = BetaDivergence(beta=1.5)

    def test_beta_must_not_be_zero_or_one(self):
        with self.assertRaises(ValueError):
            BetaDivergence(beta=1.0)
        with self.assertRaises(ValueError):
            BetaDivergence(beta=0.0)

    def test_known_value(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        self.assertAlmostEqual(self.div.divergence(p, q), 0.0, places=8)


class TestGetDivergenceFactory(unittest.TestCase):

    def test_returns_kl(self):
        div = get_divergence("kl")
        self.assertIsInstance(div, KLDivergence)

    def test_returns_euclidean(self):
        div = get_divergence("euclidean")
        self.assertIsInstance(div, SquaredEuclideanDivergence)

    def test_returns_itakura_saito(self):
        div = get_divergence("itakura_saito")
        self.assertIsInstance(div, ItakuraSaitoDivergence)

    def test_returns_beta(self):
        div = get_divergence("beta", beta=2.0)
        self.assertIsInstance(div, BetaDivergence)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_divergence("unknown_divergence")


if __name__ == "__main__":
    unittest.main()
