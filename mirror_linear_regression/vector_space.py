"""
Manifold and vector space abstractions for MIRAGE++.

Provides abstract base classes and concrete implementations for the two
parameter spaces used in the framework:

  EuclideanSpace           -- standard R^n with dot product and L2 norm
  ProbabilitySimplexSpace  -- Delta_{n-1} with Fisher-Rao metric
  SPDManifoldSpace         -- Sym+(n) with affine-invariant metric

Each manifold exposes:
  inner(u, v, base)    -- Riemannian inner product at a base point
  norm(v, base)        -- Riemannian norm
  distance(p, q)       -- geodesic distance
  project(p)           -- projection onto the feasible set
"""

import numpy as np
from abc import ABC, abstractmethod

from .utils_math import project_simplex
from .geometry import (
    fisher_rao_distance,
    natural_gradient,
    project_to_spd,
    affine_invariant_distance,
)


class VectorSpace(ABC):
    """Abstract base class for a (Riemannian) vector / manifold space."""

    @abstractmethod
    def zero(self, shape):
        """Return the zero element of the given shape."""

    @abstractmethod
    def inner(self, u: np.ndarray, v: np.ndarray, base: np.ndarray = None) -> float:
        """Riemannian inner product of tangent vectors u, v at base point."""

    @abstractmethod
    def norm(self, v: np.ndarray, base: np.ndarray = None) -> float:
        """Riemannian norm of tangent vector v at base point."""

    @abstractmethod
    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Geodesic distance between points p and q."""

    @abstractmethod
    def project(self, p: np.ndarray) -> np.ndarray:
        """Project p onto the feasible set."""


class EuclideanSpace(VectorSpace):
    """
    Standard Euclidean space R^n.

    Inner product:  <u, v> = u^T v
    Norm:           ||v|| = sqrt(v^T v)
    Distance:       ||p - q||_2
    Projection:     identity (no constraint)
    """

    def zero(self, shape):
        return np.zeros(shape)

    def inner(self, u: np.ndarray, v: np.ndarray, base: np.ndarray = None) -> float:
        return float(np.dot(u, v))

    def norm(self, v: np.ndarray, base: np.ndarray = None) -> float:
        return float(np.linalg.norm(v))

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        return float(np.linalg.norm(p - q))

    def project(self, p: np.ndarray) -> np.ndarray:
        return p.copy()


class ProbabilitySimplexSpace(VectorSpace):
    """
    Probability simplex Delta_{n-1} with the Fisher-Rao (information) metric.

    At a base point theta in Delta_{n-1}, the Fisher-Rao metric is:

        g_theta(u, v) = sum_i u_i v_i / theta_i

    This is the inner product of the inverse Fisher information matrix:
        g_theta(u, v) = u^T diag(1/theta) v

    Key properties:
      - Geodesic distance: d_FR(p,q) = 2 arccos(sum_i sqrt(p_i q_i))
      - Isometric to a spherical octant via theta -> 2*sqrt(theta)
      - Natural gradient = diag(theta)(grad - theta^T grad)
    """

    eps: float = 1e-12

    def zero(self, shape):
        n = shape if isinstance(shape, int) else shape[0]
        return np.ones(n) / n  # uniform = natural zero on the simplex

    def inner(self, u: np.ndarray, v: np.ndarray, base: np.ndarray = None) -> float:
        """
        Fisher-Rao inner product at base:
            g_base(u, v) = sum_i u_i v_i / base_i
        """
        if base is None:
            raise ValueError("Fisher-Rao inner product requires a base point.")
        base = np.clip(base, self.eps, None)
        return float(np.sum(u * v / base))

    def norm(self, v: np.ndarray, base: np.ndarray = None) -> float:
        """Fisher-Rao norm: sqrt(g_base(v, v))."""
        return float(np.sqrt(max(self.inner(v, v, base), 0.0)))

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Fisher-Rao geodesic distance: 2 arccos(sum_i sqrt(p_i q_i))."""
        return fisher_rao_distance(p, q)

    def project(self, p: np.ndarray) -> np.ndarray:
        """Project onto Delta_{n-1} (clip negatives, normalise)."""
        p = np.clip(p, self.eps, None)
        return p / np.sum(p)

    def natural_gradient(self, euclidean_grad: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Riemannian gradient under the Fisher-Rao metric:
            ng_i = theta_i * (grad_i - theta^T grad)
        """
        return natural_gradient(euclidean_grad, theta)


class SPDManifoldSpace(VectorSpace):
    """
    Manifold of symmetric positive definite (SPD) matrices Sym+(n)
    with the affine-invariant Riemannian metric.

    At Sigma in Sym+(n), the affine-invariant metric is:
        <U, V>_Sigma = tr(Sigma^-1 U Sigma^-1 V)

    Key properties:
      - Geodesic distance: ||logm(Sigma1^{-1/2} Sigma2 Sigma1^{-1/2})||_F
      - Invariant under congruence: d(A Sigma1 A^T, A Sigma2 A^T) = d(Sigma1, Sigma2)
      - Riemannian mean is the unique minimiser of sum of squared geodesic distances
    """

    def zero(self, shape):
        n = shape if isinstance(shape, int) else shape[0]
        return np.eye(n)  # identity = natural reference point

    def inner(self, u: np.ndarray, v: np.ndarray, base: np.ndarray = None) -> float:
        """
        Affine-invariant inner product:
            <U, V>_Sigma = tr(Sigma^-1 U Sigma^-1 V)
        """
        if base is None:
            raise ValueError("Affine-invariant inner product requires a base point (Sigma).")
        sigma_inv = np.linalg.inv(base)
        return float(np.trace(sigma_inv @ u @ sigma_inv @ v))

    def norm(self, v: np.ndarray, base: np.ndarray = None) -> float:
        """Affine-invariant norm: sqrt(<V,V>_Sigma)."""
        return float(np.sqrt(max(self.inner(v, v, base), 0.0)))

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Affine-invariant geodesic distance."""
        return affine_invariant_distance(p, q)

    def project(self, p: np.ndarray) -> np.ndarray:
        """Project onto Sym+(n) (symmetrise and threshold eigenvalues)."""
        return project_to_spd(p)
