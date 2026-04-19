"""
Mathematical utility functions for MIRAGE++.

Covers probability simplex operations, information-theoretic measures,
and concentration/diversity indices used in portfolio and signal analysis.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Simplex operations
# ---------------------------------------------------------------------------

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize a non-negative vector to sum to 1 (L1 normalisation)."""
    x = np.asarray(x, dtype=float)
    total = np.sum(x)
    if total == 0.0:
        raise ValueError("Cannot normalise a zero vector.")
    return x / total


def project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project a vector v onto the probability simplex Delta_{n-1}.

    Implements the O(n log n) algorithm of Wang & Carreira-Perpinan (2013).
    Returns the unique point w* = argmin_{w in Delta} ||w - v||^2.

    Reference: Wang & Carreira-Perpinan (2013). Projection onto the probability simplex.
    """
    v = np.asarray(v, dtype=float)
    if np.sum(v) == 1.0 and np.all(v >= 0.0):
        return v
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = int(np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0][-1])
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax transformation."""
    x = np.asarray(x, dtype=float)
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


# ---------------------------------------------------------------------------
# Information-theoretic measures
# ---------------------------------------------------------------------------

def entropy(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    Shannon entropy  H(x) = -sum_i x_i log x_i  (nats).

    Args:
        x: Probability vector (need not be clipped; zeros are handled).
        eps: Numerical floor to avoid log(0).

    Returns:
        Entropy value >= 0.
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, eps, 1.0)
    return float(-np.sum(x * np.log(x)))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Kullback-Leibler divergence  D_KL(p || q) = sum_i p_i log(p_i / q_i).

    Args:
        p, q: Probability vectors.
        eps: Numerical floor.

    Returns:
        KL divergence (>= 0).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def hellinger_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Hellinger distance  H(p, q) = (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||_2.

    Satisfies  0 <= H(p,q) <= 1.  Related to the Bhattacharyya coefficient:
        H(p,q)^2 = 1 - BC(p,q)  where  BC = sum_i sqrt(p_i q_i).

    Equivalent to half the Fisher-Rao distance on the simplex (up to scaling).
    """
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    q = np.clip(np.asarray(q, dtype=float), eps, None)
    return float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2.0))


def bhattacharyya_coefficient(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Bhattacharyya coefficient  BC(p, q) = sum_i sqrt(p_i * q_i) in [0, 1].

    BC = 1 iff p = q; BC = 0 iff p and q have disjoint supports.
    Related to Fisher-Rao distance: d_FR(p,q) = 2 arccos(BC(p,q)).
    """
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    q = np.clip(np.asarray(q, dtype=float), eps, None)
    return float(np.sum(np.sqrt(p * q)))


# ---------------------------------------------------------------------------
# Diversity and concentration indices
# ---------------------------------------------------------------------------

def herfindahl_index(weights: np.ndarray) -> float:
    """
    Herfindahl-Hirschman Index (HHI):  HHI = sum_i w_i^2  in [1/n, 1].

    Measures weight concentration.  HHI = 1/n for uniform weights (maximum
    diversification); HHI = 1 for a single weight equal to 1 (full concentration).
    Widely used in portfolio analysis and market structure theory.
    """
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w ** 2))


def effective_number_of_bets(weights: np.ndarray, eps: float = 1e-12) -> float:
    """
    Effective Number of Bets (ENB):  ENB = exp(H(w)) = exp(-sum_i w_i log w_i).

    Measures the number of equally-weighted positions that would produce the
    same entropy as the actual allocation.  ENB in [1, n]:
      ENB = 1   for a single concentrated bet
      ENB = n   for a uniform portfolio (maximum diversification)

    Reference: Meucci (2009). Managing Diversification.
    """
    w = np.clip(np.asarray(weights, dtype=float), eps, None)
    return float(np.exp(-np.sum(w * np.log(w))))


def diversification_ratio(weights: np.ndarray, eps: float = 1e-12) -> float:
    """
    Diversification ratio relative to uniform weights:  DR = H(w) / log(n).

    DR in [0, 1]:
      DR = 0  for a single concentrated weight
      DR = 1  for the uniform distribution

    Equivalent to normalised entropy; invariant to the simplex dimension.
    """
    w = np.clip(np.asarray(weights, dtype=float), eps, None)
    n = len(w)
    if n <= 1:
        return 1.0
    h = float(-np.sum(w * np.log(w)))
    return h / np.log(n)
