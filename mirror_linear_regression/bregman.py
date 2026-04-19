"""
Plug-and-play Bregman divergence framework for mirror descent.

A Bregman divergence induced by a strictly convex, differentiable generator
phi is defined as:

    D_phi(p || q) = phi(p) - phi(q) - <grad_phi(q), p - q>

The choice of phi determines:
  - The geometry of the parameter space
  - The mirror map  nabla_phi   : primal -> dual
  - The inverse    (nabla_phi)^-1 : dual  -> primal
  - The mirror descent update rule (closed form when available)

Subclasses implement:
  generator(p)       -- scalar phi(p)
  grad_generator(p)  -- nabla phi(p)  (mirror map)
  inverse_mirror(z)  -- (nabla phi)^-1(z)
  project(p)         -- project onto the feasible set

References:
  Bregman (1967). The relaxation method of finding a common point of convex sets.
  Beck & Teboulle (2003). Mirror descent and nonlinear projected subgradient methods.
  Cichocki & Amari (2010). Families of Alpha- and Beta- and Gamma-Divergences.
  Fevotte & Idier (2011). Algorithms for NMF with the beta-divergence.
"""

from abc import ABC, abstractmethod

import numpy as np

from .utils_math import project_simplex


class BregmanDivergence(ABC):
    """Abstract base class for Bregman divergences."""

    @abstractmethod
    def generator(self, p: np.ndarray) -> float:
        """Compute the generator function phi(p)."""

    @abstractmethod
    def grad_generator(self, p: np.ndarray) -> np.ndarray:
        """Compute nabla phi(p) -- the mirror map from primal to dual space."""

    @abstractmethod
    def inverse_mirror(self, z: np.ndarray) -> np.ndarray:
        """Compute (nabla phi)^-1(z) -- inverse mirror map, dual -> primal."""

    @abstractmethod
    def project(self, p: np.ndarray) -> np.ndarray:
        """Project p onto the feasible set (e.g. probability simplex)."""

    def divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        D_phi(p || q) = phi(p) - phi(q) - <nabla phi(q), p - q>.
        """
        return (
            self.generator(p)
            - self.generator(q)
            - float(np.dot(self.grad_generator(q), p - q))
        )

    def mirror_step(self, grad: np.ndarray, theta: np.ndarray, eta: float) -> np.ndarray:
        """
        One mirror descent step via the dual update / inverse mirror map:

            z_{t+1}   = nabla phi(theta_t) - eta * grad L(theta_t)   [dual update]
            theta_{t+1} = (nabla phi)^-1(z_{t+1})                    [primal recovery]
            theta_{t+1} = project(theta_{t+1})                       [feasibility]

        This solves the proximal subproblem exactly:
            theta_{t+1} = argmin_theta { eta <grad L, theta> + D_phi(theta || theta_t) }
        """
        z = self.grad_generator(theta) - eta * grad
        theta_new = self.inverse_mirror(z)
        return self.project(theta_new)


class KLDivergence(BregmanDivergence):
    """
    KL divergence  D_KL(p || q) = sum_i p_i log(p_i / q_i).

    Generator:   phi(p) = sum_i p_i log p_i  (negative entropy)
    Mirror map:  nabla phi(p) = log p + 1
    Inverse:     (nabla phi)^-1(z) = exp(z - 1)  ~  exp(z)
    Update:      theta_{t+1,i}  proportional to  theta_{t,i} * exp(-eta * grad_i)
                 (exponentiated gradient / Hedge algorithm)

    The KL divergence is the natural geometry for probability simplices:
    its regret bound O(sqrt(T log n)) has log(n) vs O(sqrt(nT)) for
    Euclidean mirror descent -- exponentially better in high dimensions.
    """

    eps: float = 1e-12

    def generator(self, p: np.ndarray) -> float:
        p = np.clip(p, self.eps, 1.0)
        return float(np.sum(p * np.log(p)))

    def grad_generator(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, 1.0)
        return np.log(p) + 1.0

    def inverse_mirror(self, z: np.ndarray) -> np.ndarray:
        return np.exp(z - 1.0)

    def project(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, None)
        return p / np.sum(p)

    def mirror_step(self, grad: np.ndarray, theta: np.ndarray, eta: float) -> np.ndarray:
        # Closed form (numerically stable): theta * exp(-eta * grad), normalise
        theta_new = theta * np.exp(-eta * grad)
        theta_new = np.clip(theta_new, self.eps, None)
        return theta_new / np.sum(theta_new)

    def divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        p = np.clip(p, self.eps, 1.0)
        q = np.clip(q, self.eps, 1.0)
        return float(np.sum(p * np.log(p / q)))


class SquaredEuclideanDivergence(BregmanDivergence):
    """
    Squared Euclidean divergence  D(p || q) = 0.5 * ||p - q||^2.

    Generator:   phi(p) = 0.5 ||p||^2
    Mirror map:  nabla phi(p) = p
    Inverse:     (nabla phi)^-1(z) = z
    Update:      theta_{t+1} = project_simplex(theta_t - eta * grad)
                 (standard projected gradient descent)

    Regret bound O(sqrt(nT)) -- inferior to KL on large simplices.
    Useful as a baseline and for dense gradient problems.
    """

    def generator(self, p: np.ndarray) -> float:
        return 0.5 * float(np.dot(p, p))

    def grad_generator(self, p: np.ndarray) -> np.ndarray:
        return p.copy()

    def inverse_mirror(self, z: np.ndarray) -> np.ndarray:
        return z.copy()

    def project(self, p: np.ndarray) -> np.ndarray:
        return project_simplex(p)

    def mirror_step(self, grad: np.ndarray, theta: np.ndarray, eta: float) -> np.ndarray:
        return self.project(theta - eta * grad)


class ItakuraSaitoDivergence(BregmanDivergence):
    """
    Itakura-Saito divergence  D_IS(p || q) = sum_i (p_i/q_i - log(p_i/q_i) - 1).

    Generator:   phi(p) = -sum_i log p_i
    Mirror map:  nabla phi(p)_i = -1/p_i
    Inverse:     (nabla phi)^-1(z)_i = -1/z_i
    Update:      theta_{t+1,i}  proportional to  1 / (1/theta_{t,i} + eta * grad_i)

    Natural for strictly-positive vectors; scale-invariant (invariant to
    rescaling p and q by the same factor). Arises in spectral estimation and
    non-negative matrix factorisation.

    Reference: Fevotte & Idier (2011). Algorithms for NMF with beta-divergences.
    """

    eps: float = 1e-12

    def generator(self, p: np.ndarray) -> float:
        p = np.clip(p, self.eps, None)
        return float(-np.sum(np.log(p)))

    def grad_generator(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, None)
        return -1.0 / p

    def inverse_mirror(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -1.0 / self.eps, -self.eps)
        return -1.0 / z

    def project(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, None)
        return p / np.sum(p)

    def mirror_step(self, grad: np.ndarray, theta: np.ndarray, eta: float) -> np.ndarray:
        # Dual: z_i = -1/theta_i - eta*grad_i  =>  theta_new_i = -1/z_i = 1/(1/theta_i + eta*grad_i)
        inv_theta = 1.0 / np.clip(theta, self.eps, None)
        theta_new = 1.0 / np.clip(inv_theta + eta * grad, self.eps, None)
        theta_new = np.clip(theta_new, self.eps, None)
        return theta_new / np.sum(theta_new)

    def divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        p = np.clip(p, self.eps, None)
        q = np.clip(q, self.eps, None)
        return float(np.sum(p / q - np.log(p / q) - 1.0))


class BetaDivergence(BregmanDivergence):
    """
    Beta-divergence  D_beta(p || q) parametrises a family interpolating:
      beta -> 1 : KL divergence
      beta -> 0 : Itakura-Saito divergence
      beta  = 2 : squared Euclidean (up to scaling)

    For beta not in {0, 1}:
        D_beta(p||q) = sum_i [ p^beta/(beta*(beta-1))
                               - p*q^(beta-1)/(beta-1)
                               + q^beta/beta ]

    Generator:   phi_beta(p) = sum_i p^beta / (beta*(beta-1))
    Mirror map:  (nabla phi)_i = p^(beta-1) / (beta-1)
    Inverse:     ((beta-1)*z)^(1/(beta-1))

    Reference: Cichocki & Amari (2010). Families of Alpha- and Beta-Divergences.
    """

    eps: float = 1e-12

    def __init__(self, beta: float = 1.5):
        if beta in (0.0, 1.0):
            raise ValueError(
                "Use ItakuraSaitoDivergence for beta=0 and KLDivergence for beta=1."
            )
        self.beta = float(beta)

    def generator(self, p: np.ndarray) -> float:
        p = np.clip(p, self.eps, None)
        b = self.beta
        return float(np.sum(p ** b) / (b * (b - 1)))

    def grad_generator(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, None)
        b = self.beta
        return p ** (b - 1) / (b - 1)

    def inverse_mirror(self, z: np.ndarray) -> np.ndarray:
        b = self.beta
        return np.clip((b - 1) * z, self.eps, None) ** (1.0 / (b - 1))

    def project(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, None)
        return p / np.sum(p)

    def divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        p = np.clip(p, self.eps, None)
        q = np.clip(q, self.eps, None)
        b = self.beta
        return float(np.sum(
            p ** b / (b * (b - 1))
            - p * q ** (b - 1) / (b - 1)
            + q ** b / b
        ))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict = {
    "kl": KLDivergence,
    "euclidean": SquaredEuclideanDivergence,
    "itakura_saito": ItakuraSaitoDivergence,
    "beta": BetaDivergence,
}


def get_divergence(name: str, **kwargs) -> BregmanDivergence:
    """
    Instantiate a BregmanDivergence by name.

    Args:
        name: One of 'kl', 'euclidean', 'itakura_saito', 'beta'.
        **kwargs: Passed to the divergence constructor (e.g. beta=1.5).

    Returns:
        BregmanDivergence instance.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown divergence '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)
