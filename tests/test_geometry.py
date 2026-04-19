"""
Tests for Riemannian geometry operations on the probability simplex
and SPD manifold.
"""

import unittest
import numpy as np

from mirror_linear_regression.geometry import (
    natural_gradient,
    natural_gradient_step,
    fisher_rao_distance,
    geodesic,
    exponential_map,
    logarithmic_map,
    spd_riemannian_gradient,
    spd_retraction,
    spd_geodesic,
    affine_invariant_distance,
    project_to_spd,
    fisher_information,
)


# ---------------------------------------------------------------------------
# Simplex / Fisher-Rao geometry
# ---------------------------------------------------------------------------

class TestNaturalGradient(unittest.TestCase):

    def setUp(self):
        self.theta = np.array([0.4, 0.3, 0.2, 0.1])
        self.grad = np.array([0.5, -0.2, 0.1, 0.3])

    def test_natural_gradient_in_tangent_space(self):
        """Natural gradient must lie in the tangent space: sum_i ng_i = 0."""
        ng = natural_gradient(self.grad, self.theta)
        self.assertAlmostEqual(float(np.sum(ng)), 0.0, places=10)

    def test_natural_gradient_step_on_simplex(self):
        theta_new = natural_gradient_step(self.grad, self.theta, eta=0.05)
        self.assertTrue(np.all(theta_new >= 0))
        self.assertAlmostEqual(float(np.sum(theta_new)), 1.0, places=10)

    def test_natural_gradient_differs_from_mirror_descent_large_eta(self):
        """
        For large step sizes, natural gradient and mirror descent should diverge.
        (For small eta they are first-order equivalent.)
        """
        from mirror_linear_regression.bregman import KLDivergence
        div = KLDivergence()
        eta = 1.0
        theta_ng = natural_gradient_step(self.grad, self.theta, eta)
        theta_md = div.mirror_step(self.grad, self.theta, eta)
        # They should not be identical for large eta
        self.assertFalse(
            np.allclose(theta_ng, theta_md, atol=1e-6),
            "Natural gradient and KL mirror descent should differ at large eta."
        )

    def test_first_order_equivalence_small_eta(self):
        """
        For small eta, natural gradient step ~ KL mirror descent step to O(eta^2).
        """
        from mirror_linear_regression.bregman import KLDivergence
        div = KLDivergence()
        eta = 1e-4
        theta_ng = natural_gradient_step(self.grad, self.theta, eta)
        theta_md = div.mirror_step(self.grad, self.theta, eta)
        np.testing.assert_allclose(theta_ng, theta_md, atol=1e-6)


class TestFisherRaoDistance(unittest.TestCase):

    def test_distance_zero_to_self(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        self.assertAlmostEqual(fisher_rao_distance(p, p), 0.0, places=10)

    def test_distance_symmetric(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.1, 0.6, 0.3])
        self.assertAlmostEqual(
            fisher_rao_distance(p, q), fisher_rao_distance(q, p), places=10
        )

    def test_distance_non_negative(self):
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.2, 0.5, 0.3])
        self.assertGreater(fisher_rao_distance(p, q), 0.0)

    def test_distance_max_for_disjoint(self):
        """Distributions on disjoint supports: BC = 0, d_FR = pi."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        # With clipping, BC > 0 but should be close to pi
        self.assertGreater(fisher_rao_distance(p, q), 3.0)


class TestGeodesic(unittest.TestCase):

    def test_endpoints(self):
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.1, 0.2, 0.7])
        np.testing.assert_allclose(geodesic(p, q, 0.0), p, atol=1e-10)
        np.testing.assert_allclose(geodesic(p, q, 1.0), q, atol=1e-6)

    def test_midpoint_on_simplex(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.2, 0.4, 0.4])
        mid = geodesic(p, q, 0.5)
        self.assertTrue(np.all(mid >= 0))
        self.assertAlmostEqual(float(np.sum(mid)), 1.0, places=10)

    def test_monotone_distance(self):
        """Points along the geodesic should be ordered by distance from p."""
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.1, 0.2, 0.7])
        d1 = fisher_rao_distance(p, geodesic(p, q, 0.3))
        d2 = fisher_rao_distance(p, geodesic(p, q, 0.6))
        self.assertLess(d1, d2)


class TestExponentialLog(unittest.TestCase):

    def test_exp_map_on_simplex(self):
        theta = np.array([0.4, 0.35, 0.25])
        v = np.array([0.1, -0.05, -0.05])
        result = exponential_map(theta, v)
        self.assertTrue(np.all(result >= 0))
        self.assertAlmostEqual(float(np.sum(result)), 1.0, places=8)

    def test_log_map_zero_at_self(self):
        theta = np.array([0.4, 0.35, 0.25])
        log_v = logarithmic_map(theta, theta)
        np.testing.assert_allclose(log_v, np.zeros(3), atol=1e-8)


class TestFisherInformation(unittest.TestCase):

    def test_diagonal_equals_inv_theta(self):
        theta = np.array([0.5, 0.3, 0.2])
        fi = fisher_information(theta)
        expected_diag = 1.0 / theta
        np.testing.assert_allclose(np.diag(fi), expected_diag)

    def test_shape(self):
        theta = np.array([0.4, 0.4, 0.2])
        fi = fisher_information(theta)
        self.assertEqual(fi.shape, (3, 3))


# ---------------------------------------------------------------------------
# SPD manifold geometry
# ---------------------------------------------------------------------------

class TestSPDGeometry(unittest.TestCase):

    def setUp(self):
        self.sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
        self.egrad = np.array([[0.3, 0.1], [0.1, -0.2]])

    def test_riemannian_gradient_shape(self):
        rg = spd_riemannian_gradient(self.egrad, self.sigma)
        self.assertEqual(rg.shape, self.sigma.shape)

    def test_riemannian_gradient_symmetric(self):
        """Riemannian gradient should be symmetric (symmetric input)."""
        sym_egrad = (self.egrad + self.egrad.T) / 2
        rg = spd_riemannian_gradient(sym_egrad, self.sigma)
        np.testing.assert_allclose(rg, rg.T, atol=1e-10)

    def test_retraction_spd(self):
        rg = spd_riemannian_gradient(self.egrad, self.sigma)
        sigma_new = spd_retraction(self.sigma, rg, eta=0.1)
        # Must be symmetric
        np.testing.assert_allclose(sigma_new, sigma_new.T, atol=1e-8)
        # Must be positive definite
        eigvals = np.linalg.eigvalsh(sigma_new)
        self.assertTrue(np.all(eigvals > 0))

    def test_geodesic_endpoints(self):
        sigma1 = np.array([[3.0, 0.5], [0.5, 2.0]])
        g0 = spd_geodesic(self.sigma, sigma1, t=0.0)
        g1 = spd_geodesic(self.sigma, sigma1, t=1.0)
        np.testing.assert_allclose(g0, self.sigma, atol=1e-8)
        np.testing.assert_allclose(g1, sigma1, atol=1e-8)

    def test_affine_invariant_distance_zero_to_self(self):
        d = affine_invariant_distance(self.sigma, self.sigma)
        self.assertAlmostEqual(d, 0.0, places=6)

    def test_affine_invariant_distance_positive(self):
        sigma2 = np.array([[3.0, 0.0], [0.0, 2.0]])
        d = affine_invariant_distance(self.sigma, sigma2)
        self.assertGreater(d, 0.0)

    def test_affine_invariant_distance_symmetric(self):
        sigma2 = np.array([[3.0, 0.2], [0.2, 1.5]])
        d12 = affine_invariant_distance(self.sigma, sigma2)
        d21 = affine_invariant_distance(sigma2, self.sigma)
        self.assertAlmostEqual(d12, d21, places=6)

    def test_project_to_spd(self):
        a = np.array([[1.0, 2.0], [2.0, -1.0]])  # not PD
        spd = project_to_spd(a)
        np.testing.assert_allclose(spd, spd.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(spd)
        self.assertTrue(np.all(eigvals > 0))


if __name__ == "__main__":
    unittest.main()
