"""
Riemannian geometry for optimization on structured parameter spaces.

Two manifolds are supported:

1.  Probability Simplex  Delta_{n-1}  with Fisher-Rao metric
    -----------------------------------------------------------
    The Fisher information metric at theta is  g_{ij}(theta) = delta_{ij}/theta_i.
    This makes Delta_{n-1} isometric to a spherical octant via  theta -> 2*sqrt(theta).

    Key operations:
      natural_gradient(grad, theta)          -- Riemannian gradient under Fisher-Rao metric
      natural_gradient_step(grad, theta, eta) -- NGD step (differs from mirror descent at O(eta^2))
      geodesic(theta0, theta1, t)            -- gamma(t) = normalise(theta0^(1-t) * theta1^t)
      fisher_rao_distance(p, q)              -- 2 arccos(sum_i sqrt(p_i q_i))
      exponential_map(theta, v)              -- Exp_theta(v) on the simplex
      logarithmic_map(theta, phi)            -- Log_theta(phi): tangent vector toward phi

2.  SPD Manifold  Sym+(n)  with affine-invariant metric
    ----------------------------------------------------
    The affine-invariant metric at Sigma is  <U,V>_Sigma = tr(Sigma^-1 U Sigma^-1 V).
    Geodesics are computed via symmetric matrix square roots.

    Key operations:
      spd_riemannian_gradient(egrad, Sigma)  -- Riemannian gradient from Euclidean gradient
      spd_retraction(Sigma, rgrad, eta)      -- Geodesic retraction (matrix exponential)
      spd_log_euclidean_step(Sigma, grad, eta) -- Cheaper Log-Euclidean approximation
      spd_geodesic(Sigma0, Sigma1, t)        -- Geodesic interpolation
      affine_invariant_distance(S1, S2)      -- ||logm(S1^-1/2 S2 S1^-1/2)||_F
      project_to_spd(A)                      -- Symmetrise and clip eigenvalues

Relationship between natural gradient and mirror descent
---------------------------------------------------------
Both the Fisher-Rao natural gradient step and the KL mirror descent step
produce the same first-order update:
    theta_{t+1,i} ~ theta_{t,i} - eta * theta_{t,i}*(grad_i - theta^T grad)  + O(eta^2)
They diverge at second order.  Mirror descent (exponentiated gradient) is
exact in the KL geometry; natural gradient is exact in the Fisher-Rao geometry.
For small eta the two are numerically indistinguishable.

References:
  Amari (1998). Natural gradient works efficiently in learning. Neural Computation.
  Sra & Hosseini (2015). Conic geometric optimization on the manifold of SPD matrices.
  Moakher (2005). A differential geometric approach to the geometric mean of SPD matrices.
  Nielsen & Barbaresco (2013). Geometric Science of Information.
"""

import numpy as np
from scipy.linalg import expm, logm, sqrtm

from .utils_math import project_simplex


# ===========================================================================
# Probability Simplex -- Fisher-Rao Geometry
# ===========================================================================

def fisher_information(theta: np.ndarray) -> np.ndarray:
    """
    Fisher information matrix  F(theta) = diag(1/theta)  for the categorical
    distribution.  This is the metric tensor of the Fisher-Rao geometry at theta.

    Args:
        theta: Probability vector on the simplex (n,).

    Returns:
        Diagonal matrix (n x n) -- in practice use the vector 1/theta directly
        to avoid O(n^2) storage.
    """
    return np.diag(1.0 / np.clip(theta, 1e-12, None))


def natural_gradient(euclidean_grad: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Riemannian (natural) gradient on the probability simplex under the Fisher-Rao metric.

    The natural gradient projects the Euclidean gradient into the tangent space
    T_theta Delta_{n-1} and rescales by F(theta)^-1 = diag(theta):

        ng_i = theta_i * (grad_i  -  sum_j theta_j * grad_j)

    The centering term  c = theta^T grad  ensures ng lies in the tangent space
    (satisfies  sum_i ng_i = 0), and diag(theta) is the inverse Fisher metric.

    For small step sizes, the natural gradient step:
        theta - eta * ng
    is first-order equivalent to the KL mirror descent update
        theta ∝ theta * exp(-eta * grad)
    but they differ at O(eta^2).

    Args:
        euclidean_grad: Euclidean gradient nabla L(theta) in R^n.
        theta: Current parameters on the simplex.

    Returns:
        Natural gradient in T_theta Delta_{n-1}.
    """
    theta = np.clip(theta, 1e-12, None)
    centering = float(np.dot(theta, euclidean_grad))
    return theta * (euclidean_grad - centering)


def natural_gradient_step(
    euclidean_grad: np.ndarray,
    theta: np.ndarray,
    eta: float,
) -> np.ndarray:
    """
    Riemannian gradient descent step on the probability simplex (Fisher-Rao metric):

        theta_{t+1} = proj_Delta( theta_t - eta * ng(theta_t) )

    where  ng_i = theta_i (grad_i - theta^T grad).

    Args:
        euclidean_grad: Euclidean gradient nabla L(theta).
        theta: Current simplex point.
        eta: Step size.

    Returns:
        Updated point on the simplex.
    """
    ng = natural_gradient(euclidean_grad, theta)
    theta_new = theta - eta * ng
    return project_simplex(theta_new)


def geodesic(theta0: np.ndarray, theta1: np.ndarray, t: float) -> np.ndarray:
    """
    Geodesic interpolation on Delta_{n-1} under the Fisher-Rao metric:

        gamma(t) = normalise( theta0^(1-t) ⊙ theta1^t ),   t in [0, 1]

    This is the geometric-mean path, natural in information geometry.
    At t=0 returns theta0; at t=1 returns theta1.

    Args:
        theta0: Start point on the simplex.
        theta1: End point on the simplex.
        t: Interpolation parameter in [0, 1].

    Returns:
        gamma(t) on the simplex.
    """
    theta0 = np.clip(theta0, 1e-12, None)
    theta1 = np.clip(theta1, 1e-12, None)
    gamma = (theta0 ** (1.0 - t)) * (theta1 ** t)
    return gamma / np.sum(gamma)


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Fisher-Rao geodesic distance on the probability simplex:

        d_FR(p, q) = 2 arccos( sum_i sqrt(p_i * q_i) )
                   = 2 arccos( BC(p, q) )

    where BC(p, q) = sum_i sqrt(p_i q_i) is the Bhattacharyya coefficient.
    This equals the spherical geodesic distance after the isometric embedding
    theta -> 2*sqrt(theta) onto the unit sphere.

    Args:
        p, q: Probability vectors on the simplex.

    Returns:
        Fisher-Rao geodesic distance (scalar >= 0).
    """
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    bc = float(np.sum(np.sqrt(p * q)))
    bc = np.clip(bc, -1.0, 1.0)
    return 2.0 * np.arccos(bc)


def exponential_map(theta: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Exponential map  Exp_theta(v)  on the probability simplex.

    Using the isometric embedding  f(theta)_i = 2*sqrt(theta_i) onto the sphere:

        Exp_theta(v)_i  proportional to  theta_i * exp(v_i / sqrt(theta_i))

    Maps tangent vector v in T_theta Delta to a new point on the simplex.

    Args:
        theta: Base point on the simplex.
        v: Tangent vector at theta.
        eps: Numerical floor.

    Returns:
        New point on the simplex.
    """
    theta = np.clip(theta, eps, None)
    result = theta * np.exp(v / (np.sqrt(theta) + eps))
    result = np.clip(result, eps, None)
    return result / np.sum(result)


def logarithmic_map(theta: np.ndarray, phi: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Logarithmic map  Log_theta(phi)  on the probability simplex.

    Returns the tangent vector at theta pointing toward phi:

        Log_theta(phi)_i = sqrt(theta_i) * log(phi_i / theta_i)

    This is the first-order inverse of exponential_map.

    Args:
        theta: Base point on the simplex.
        phi: Target point on the simplex.
        eps: Numerical floor.

    Returns:
        Tangent vector at theta.
    """
    theta = np.clip(theta, eps, None)
    phi = np.clip(phi, eps, None)
    return np.sqrt(theta) * np.log(phi / theta)


# ===========================================================================
# SPD Manifold -- Affine-Invariant Geometry
# ===========================================================================

def spd_riemannian_gradient(
    euclidean_grad: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Riemannian gradient on the SPD manifold under the affine-invariant metric.

    The affine-invariant metric at Sigma is  <U,V>_Sigma = tr(Sigma^-1 U Sigma^-1 V).
    The Riemannian gradient relates to the Euclidean gradient via:

        grad_R F(Sigma) = Sigma * sym(nabla_E F(Sigma)) * Sigma

    where sym(A) = (A + A^T)/2.  The Sigma factors transform from ambient-space
    gradients to intrinsic Riemannian gradients on the manifold.

    Args:
        euclidean_grad: Euclidean (ambient) gradient nabla_E F(Sigma).
        sigma: Current SPD matrix.

    Returns:
        Riemannian gradient at Sigma (same shape).
    """
    sym_grad = (euclidean_grad + euclidean_grad.T) / 2.0
    return sigma @ sym_grad @ sigma


def spd_retraction(
    sigma: np.ndarray,
    riemannian_grad: np.ndarray,
    eta: float,
) -> np.ndarray:
    """
    Geodesic retraction on the SPD manifold via the matrix exponential:

        Sigma_{t+1} = Sigma_t^{1/2}
                      * expm(-eta * Sigma_t^{-1/2} G_R Sigma_t^{-1/2})
                      * Sigma_t^{1/2}

    where G_R = spd_riemannian_gradient(nabla F, Sigma_t).
    This is the exact exponential map (geodesic retraction) on
    (Sym+(n), g_affine).

    Args:
        sigma: Current SPD matrix.
        riemannian_grad: Riemannian gradient G_R.
        eta: Step size.

    Returns:
        Updated SPD matrix.
    """
    s_sqrt = np.real(sqrtm(sigma))
    s_sqrt_inv = np.linalg.inv(s_sqrt)
    m = s_sqrt_inv @ riemannian_grad @ s_sqrt_inv
    m = (m + m.T) / 2.0
    sigma_new = s_sqrt @ expm(-eta * m) @ s_sqrt
    return project_to_spd(sigma_new)


def spd_log_euclidean_step(
    sigma: np.ndarray,
    euclidean_grad: np.ndarray,
    eta: float,
) -> np.ndarray:
    """
    Log-Euclidean gradient step on the SPD manifold.

    The Log-Euclidean metric treats SPD matrices via matrix logarithms,
    performing operations in the (flat) space of symmetric matrices:

        S_{t+1} = expm( logm(Sigma_t) - eta * sym(nabla F) )

    Cheaper than the full affine-invariant retraction (no matrix sqrt needed)
    while preserving SPD structure exactly.  Geodesics differ from
    affine-invariant geodesics, but convergence properties are similar.

    Reference: Arsigny et al. (2007). Geometric means in a novel vector space.

    Args:
        sigma: Current SPD matrix.
        euclidean_grad: Euclidean gradient nabla F(Sigma).
        eta: Step size.

    Returns:
        Updated SPD matrix.
    """
    log_sigma = np.real(logm(sigma))
    sym_grad = (euclidean_grad + euclidean_grad.T) / 2.0
    return project_to_spd(np.real(expm(log_sigma - eta * sym_grad)))


def spd_geodesic(sigma0: np.ndarray, sigma1: np.ndarray, t: float) -> np.ndarray:
    """
    Geodesic interpolation on the SPD manifold under the affine-invariant metric:

        Gamma(t) = Sigma0^{1/2}
                   * (Sigma0^{-1/2} Sigma1 Sigma0^{-1/2})^t
                   * Sigma0^{1/2}

    At t=0 returns Sigma0; at t=1 returns Sigma1.
    The midpoint t=1/2 is the Riemannian (geometric) mean of Sigma0 and Sigma1.

    Args:
        sigma0, sigma1: SPD matrices (n x n).
        t: Interpolation parameter in [0, 1].

    Returns:
        Gamma(t) on the geodesic.
    """
    s0_sqrt = np.real(sqrtm(sigma0))
    s0_sqrt_inv = np.linalg.inv(s0_sqrt)
    m = s0_sqrt_inv @ sigma1 @ s0_sqrt_inv
    m = (m + m.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(m)
    eigvals = np.clip(eigvals, 1e-12, None)
    mt = eigvecs @ np.diag(eigvals ** t) @ eigvecs.T
    return s0_sqrt @ mt @ s0_sqrt


def affine_invariant_distance(sigma1: np.ndarray, sigma2: np.ndarray) -> float:
    """
    Affine-invariant Riemannian distance on the SPD manifold:

        d_AI(Sigma1, Sigma2) = ||logm(Sigma1^{-1/2} Sigma2 Sigma1^{-1/2})||_F

    This is the geodesic distance under the affine-invariant metric.
    Invariant to congruence transformations: d(A Sigma1 A^T, A Sigma2 A^T) = d(Sigma1, Sigma2).

    Args:
        sigma1, sigma2: SPD matrices.

    Returns:
        Geodesic distance (scalar >= 0).
    """
    s1_sqrt_inv = np.linalg.inv(np.real(sqrtm(sigma1)))
    m = s1_sqrt_inv @ sigma2 @ s1_sqrt_inv
    m = (m + m.T) / 2.0
    m = project_to_spd(m)
    log_m = np.real(logm(m))
    return float(np.linalg.norm(log_m, "fro"))


def project_to_spd(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Project a matrix onto the SPD cone:
    symmetrise, then threshold eigenvalues from below by eps.

    Args:
        a: Input square matrix.
        eps: Minimum eigenvalue.

    Returns:
        Symmetric positive definite matrix.
    """
    a = (a + a.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(a)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
