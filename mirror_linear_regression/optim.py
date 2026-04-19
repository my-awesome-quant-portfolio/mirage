"""
Optimization primitives for mirror descent and its variants.

All stateless step functions share the signature:
    step(grad, theta, eta, ...) -> np.ndarray

Stateful optimizers (AdaMirror) are classes with a .step(grad, theta) method.

Algorithms
----------
mirror_descent_step        -- Exact mirror descent for any BregmanDivergence
mirror_prox_step           -- Extragradient (Mirror-Prox); O(1/T) on smooth problems
natural_gradient_step      -- Fisher-Rao Riemannian NGD on the simplex
AdaMirror                  -- Adaptive mirror descent with per-coordinate learning rates

Convergence comparison (convex loss, simplex constraint, dim n, T steps)
-----------------------------------------------------------------------
Algorithm            Regret bound          Notes
mirror_descent (KL)  O(sqrt(T log n))      Optimal for simplex
mirror_prox   (KL)   O(1/T) smooth obj.    Extragradient; better constant
natural_gradient     O(sqrt(T log n))      First-order equiv. to KL-MD
AdaMirror            O(sqrt(T) ||g||_1inf) Adapts to gradient sparsity

References:
  Nemirovski & Yudin (1983). Problem Complexity and Method Efficiency in Optimization.
  Beck & Teboulle (2003). Mirror descent and nonlinear projected subgradient methods.
  Nemirovski (2004). Prox-method with rate of convergence O(1/T) for VI problems.
  Duchi, Hazan & Singer (2011). Adaptive subgradient methods (AdaGrad).
"""

import numpy as np
from typing import Callable, Optional

from .bregman import BregmanDivergence, KLDivergence
from .geometry import natural_gradient_step as _fisher_rao_step

_DEFAULT_DIV = KLDivergence()


# ---------------------------------------------------------------------------
# Mirror Descent
# ---------------------------------------------------------------------------

def mirror_descent_step(
    grad: np.ndarray,
    theta: np.ndarray,
    eta: float,
    divergence: Optional[BregmanDivergence] = None,
) -> np.ndarray:
    """
    One step of mirror descent with the given Bregman divergence:

        theta_{t+1} = argmin_theta { eta <grad L, theta> + D_phi(theta || theta_t) }

    With KL divergence this reduces to the exponentiated gradient (Hedge):
        theta_{t+1,i}  proportional to  theta_{t,i} * exp(-eta * grad_i)

    Regret bound (KL, simplex):
        Regret_T <= log(n) / eta  +  eta * T * G^2 / 2
    Optimised over eta:  Regret_T <= G * sqrt(2 T log(n))

    Args:
        grad: Gradient nabla L(theta_t).
        theta: Current parameter vector on the simplex.
        eta: Learning rate.
        divergence: BregmanDivergence instance (default: KLDivergence).

    Returns:
        Updated parameter vector on the simplex.
    """
    div = divergence if divergence is not None else _DEFAULT_DIV
    return div.mirror_step(grad, theta, eta)


# ---------------------------------------------------------------------------
# Mirror-Prox (Extragradient)
# ---------------------------------------------------------------------------

def mirror_prox_step(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    eta: float,
    divergence: Optional[BregmanDivergence] = None,
) -> np.ndarray:
    """
    One step of Mirror-Prox (Nemirovski 2004) -- extragradient method.

    Mirror-Prox evaluates the gradient at an intermediate point, giving
    O(1/T) convergence for smooth convex objectives vs O(1/sqrt(T)) for
    standard mirror descent:

        theta_{t+1/2} = mirror_step( grad(theta_t),   theta_t, eta )   [half-step]
        theta_{t+1}   = mirror_step( grad(theta_{t+1/2}), theta_t, eta )   [full step]

    Note the full step uses theta_t (not theta_{t+1/2}) as the proximal centre.
    This is the key difference from the naive two-step method.

    Args:
        grad_fn: Callable theta -> gradient nabla L(theta).
        theta: Current parameter vector.
        eta: Learning rate.
        divergence: BregmanDivergence instance (default: KLDivergence).

    Returns:
        Updated parameter vector.
    """
    div = divergence if divergence is not None else _DEFAULT_DIV
    grad_t = grad_fn(theta)
    theta_half = div.mirror_step(grad_t, theta, eta)
    grad_half = grad_fn(theta_half)
    return div.mirror_step(grad_half, theta, eta)


# ---------------------------------------------------------------------------
# Natural Gradient Descent (Fisher-Rao)
# ---------------------------------------------------------------------------

def natural_gradient_descent_step(
    grad: np.ndarray,
    theta: np.ndarray,
    eta: float,
) -> np.ndarray:
    """
    Riemannian natural gradient descent step on the probability simplex
    under the Fisher-Rao metric (see geometry.py):

        ng_i      = theta_i * (grad_i - theta^T grad)
        theta_{t+1} = proj_Delta(theta_t - eta * ng)

    First-order equivalent to KL mirror descent; differs at O(eta^2).
    The Fisher-Rao metric makes this the intrinsically correct gradient
    for probability distributions (Amari 1998).

    Args:
        grad: Euclidean gradient nabla L(theta).
        theta: Current simplex point.
        eta: Step size.

    Returns:
        Updated simplex point.
    """
    return _fisher_rao_step(grad, theta, eta)


# ---------------------------------------------------------------------------
# Adaptive Mirror Descent (AdaMirror)
# ---------------------------------------------------------------------------

class AdaMirror:
    """
    Adaptive mirror descent with coordinate-wise learning rates.

    Combines AdaGrad's diagonal preconditioning with KL mirror descent:

        G_t     = G_{t-1} + grad_t^2              [accumulated squared gradients]
        eta_i_t = eta / sqrt(G_t_i)               [per-coordinate learning rate]
        theta_{t+1} proportional to theta_t * exp(-eta_i_t * grad_i)

    Automatically reduces the effective step size in directions with large
    historical gradients, which is beneficial when gradient magnitudes vary
    across coordinates (e.g. sparse alpha signals).

    Regret bound (Duchi et al. 2011):
        Regret_T <= (1/eta) * sum_i sqrt(T * sum_t grad_{t,i}^2)
    This is at most O(sqrt(T log n)) and can be much better when gradients
    are sparse or coordinate variances are unequal.

    Args:
        eta: Base learning rate.
        eps: Numerical floor for the denominator (prevents division by zero).
    """

    def __init__(self, eta: float = 0.1, eps: float = 1e-8):
        self.eta = eta
        self.eps = eps
        self._acc_sq_grad: Optional[np.ndarray] = None

    def step(self, grad: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Perform one adaptive mirror descent step.

        Args:
            grad: Gradient nabla L(theta).
            theta: Current simplex point.

        Returns:
            Updated simplex point.
        """
        if self._acc_sq_grad is None:
            self._acc_sq_grad = np.zeros_like(grad)
        self._acc_sq_grad += grad ** 2
        adaptive_eta = self.eta / (np.sqrt(self._acc_sq_grad) + self.eps)
        theta_new = theta * np.exp(-adaptive_eta * grad)
        theta_new = np.clip(theta_new, 1e-12, None)
        return theta_new / np.sum(theta_new)

    def reset(self) -> None:
        """Reset accumulated gradient statistics (e.g. between experiments)."""
        self._acc_sq_grad = None
