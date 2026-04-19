# Mathematical Foundations of MIRAGE++

## 1. Problem Formulation

Let $X \in \mathbb{R}^{m \times n}$ be a design matrix of $m$ observations over $n$ features, and $y \in \mathbb{R}^m$ a target vector. Classical linear regression finds $\theta \in \mathbb{R}^n$ minimising the mean-squared error:

$$\min_\theta \frac{1}{m}\|X\theta - y\|^2$$

MIRAGE++ imposes a **probability-simplex constraint** and adds an **entropy regulariser**:

$$\min_{\theta \in \Delta_{n-1}} \mathcal{L}(\theta) = \underbrace{\frac{1}{m}\|X\theta - y\|^2}_{\text{MSE}} - \underbrace{\lambda H(\theta)}_{\text{diversity bonus}}$$

where:
- $\Delta_{n-1} = \{\theta \geq 0 : \sum_i \theta_i = 1\}$ is the probability simplex
- $H(\theta) = -\sum_i \theta_i \log \theta_i \geq 0$ is Shannon entropy
- $\lambda \geq 0$ controls the diversity–fit tradeoff

**Why subtract entropy?** Minimising $\mathcal{L}$ is equivalent to minimising MSE *while maximising* weight entropy. The entropy term acts as a **diversity bonus** that prevents any single feature from receiving all weight mass — a critical property for portfolio allocation and signal combination where over-concentration implies excessive idiosyncratic risk.

**The gradient** of $\mathcal{L}$ at $\theta$ is:

$$\nabla_\theta \mathcal{L} = \frac{2}{m} X^\top(X\theta - y) + \lambda(1 + \log\theta)$$

The term $\lambda(1 + \log\theta_i)$ becomes large and negative as $\theta_i \to 0$, acting as a **log-barrier** that keeps all weights strictly positive. The minimiser therefore always lies in the **interior** of $\Delta_{n-1}$.

---

## 2. The Probability Simplex as a Riemannian Manifold

The simplex $\Delta_{n-1}$ carries a natural Riemannian metric induced by the **Fisher information matrix**:

$$g_{ij}(\theta) = \frac{\delta_{ij}}{\theta_i}$$

Under this metric, the **inner product** of two tangent vectors $u, v \in T_\theta \Delta_{n-1}$ (i.e., $\sum_i u_i = \sum_i v_i = 0$) is:

$$\langle u, v \rangle_\theta = \sum_i \frac{u_i v_i}{\theta_i}$$

This is the **Fisher-Rao metric**, and it makes $\Delta_{n-1}$ isometric to the positive spherical octant $S^{n-1}_+ = \{x \geq 0 : \|x\|_2 = 1\}$ via the map $\phi(\theta)_i = \sqrt{\theta_i}$.

### 2.1 Geodesics

The geodesic from $p$ to $q$ on $\Delta_{n-1}$ under the Fisher-Rao metric is:

$$\gamma(t) = \frac{p^{1-t} \odot q^t}{\|p^{1-t} \odot q^t\|_1}, \quad t \in [0,1]$$

where the exponentiation and multiplication are element-wise. This is a **normalised geometric interpolation** that stays strictly on the simplex.

### 2.2 Fisher-Rao Distance

The geodesic distance between $p, q \in \Delta_{n-1}$ is:

$$d_{\text{FR}}(p, q) = 2 \arccos\!\left(\sum_i \sqrt{p_i q_i}\right) = 2 \arccos\!\left(\sum_i \sqrt{p_i q_i}\right)$$

The term $\sum_i \sqrt{p_i q_i}$ is the **Bhattacharyya coefficient**, measuring the geometric overlap of two distributions. When $p = q$, the cosine equals 1 and $d_\text{FR} = 0$.

### 2.3 Exponential and Logarithmic Maps

The exponential map at $\theta$ in direction $v$ (tangent vector):

$$\exp_\theta(v) = \frac{(\theta + v)}{\|\theta + v\|_1}$$

(approximation for small $v$; the exact formula involves normalisation along the geodesic direction).

The logarithmic map (inverse) is:

$$\log_\theta(q) = q - \theta$$

---

## 3. Bregman Divergence Framework

A Bregman divergence induced by a strictly convex, differentiable generator $\phi$ is:

$$D_\phi(p \| q) = \phi(p) - \phi(q) - \langle \nabla\phi(q),\, p - q \rangle$$

The **choice of $\phi$ determines the geometry** of the parameter space: it defines the mirror map $\nabla\phi: \text{primal} \to \text{dual}$, its inverse $(\nabla\phi)^{-1}: \text{dual} \to \text{primal}$, and thereby the mirror descent update rule.

### 3.1 KL Divergence (default)

Generator: $\phi(p) = \sum_i p_i \log p_i$ (negative entropy)

$$D_\text{KL}(p \| q) = \sum_i p_i \log\frac{p_i}{q_i}$$

Mirror map: $\nabla\phi(p)_i = \log p_i + 1$

Inverse: $(\nabla\phi)^{-1}(z)_i = e^{z_i - 1}$

Closed-form update (exponentiated gradient):
$$\theta_{t+1,i} \propto \theta_{t,i} \cdot \exp(-\eta \nabla_i \mathcal{L}(\theta_t))$$

**Regret bound:** $O(\sqrt{T \log n})$ — the $\log n$ vs $n$ factor is the key advantage over Euclidean mirror descent.

### 3.2 Squared Euclidean

Generator: $\phi(p) = \frac{1}{2}\|p\|^2$

$$D_\text{Euclid}(p \| q) = \frac{1}{2}\|p - q\|^2$$

Mirror map: $\nabla\phi(p) = p$ (identity)

Update: $\theta_{t+1} = \text{proj}_{\Delta}(\theta_t - \eta \nabla\mathcal{L})$

**Regret bound:** $O(\sqrt{nT})$ — linear in dimension.

### 3.3 Itakura-Saito Divergence

Generator: $\phi(p) = -\sum_i \log p_i$

$$D_\text{IS}(p \| q) = \sum_i \left(\frac{p_i}{q_i} - \log\frac{p_i}{q_i} - 1\right)$$

Scale-invariant: $D_\text{IS}(\alpha p \| \alpha q) = D_\text{IS}(p \| q)$. Natural for spectral estimation and NMF.

Update: $\theta_{t+1,i} \propto \dfrac{1}{1/\theta_{t,i} + \eta \nabla_i \mathcal{L}}$

### 3.4 Beta Divergence

Parametric family interpolating the above:

$$D_\beta(p \| q) = \sum_i \left[\frac{p^{\beta}}{\beta(\beta-1)} - \frac{p q^{\beta-1}}{\beta-1} + \frac{q^\beta}{\beta}\right]$$

- $\beta \to 1$: KL divergence
- $\beta \to 0$: Itakura-Saito
- $\beta = 2$: (scaled) squared Euclidean

---

## 4. Natural Gradient on the Simplex

The **Euclidean gradient** $\nabla\mathcal{L}$ is not the steepest ascent direction in the Fisher-Rao metric. The **Riemannian (natural) gradient** is obtained by applying the inverse Fisher information matrix:

$$\tilde{\nabla}_i \mathcal{L} = \theta_i \left(\nabla_i \mathcal{L} - \sum_j \theta_j \nabla_j \mathcal{L}\right)$$

This is the projection of the scaled gradient $(\theta_i \nabla_i \mathcal{L})$ onto the tangent space of $\Delta_{n-1}$ (which requires $\sum_i \tilde{\nabla}_i \mathcal{L} = 0$). The update becomes:

$$\theta_{t+1} = \text{proj}_\Delta\!\left(\theta_t - \eta\, \tilde{\nabla}\mathcal{L}(\theta_t)\right)$$

**First-order equivalence with KL mirror descent:** At small $\eta$, both updates agree up to $O(\eta^2)$ — they are equivalent in the limit. They differ in curvature (second-order) information used: KL-MD uses the Bregman divergence; NGD uses the Fisher information metric directly.

---

## 5. SPD Manifold Extension

For covariance matrix estimation problems, MIRAGE++ supports the **symmetric positive definite (SPD) manifold** $\text{Sym}^+(n)$ with the **affine-invariant metric**:

$$\langle U, V \rangle_\Sigma = \text{tr}(\Sigma^{-1} U \Sigma^{-1} V), \quad U, V \in T_\Sigma \text{Sym}^+(n)$$

The **Riemannian gradient** of a scalar function $f$ at $\Sigma$ is:

$$\text{grad} f(\Sigma) = \Sigma \cdot \text{sym}(\nabla_\Sigma f) \cdot \Sigma$$

where $\text{sym}(A) = (A + A^\top)/2$.

**Geodesic retraction** (exact):

$$R_\Sigma(\xi) = \Sigma^{1/2} \exp\!\left(\Sigma^{-1/2} \xi \Sigma^{-1/2}\right) \Sigma^{1/2}$$

**Affine-invariant distance:**

$$d_\text{AI}(\Sigma_1, \Sigma_2) = \|\log(\Sigma_1^{-1/2}\Sigma_2\Sigma_1^{-1/2})\|_F$$

---

## References

- Bregman, L. M. (1967). The relaxation method of finding the common point of convex sets. *USSR Computational Mathematics and Mathematical Physics*.
- Beck, A. & Teboulle, M. (2003). Mirror descent and nonlinear projected subgradient methods. *Operations Research Letters*.
- Amari, S. (1998). Natural gradient works efficiently in learning. *Neural Computation*.
- Cichocki, A. & Amari, S. (2010). Families of alpha- and beta-divergences. *Entropy*.
- Moakher, M. (2005). A differential geometric approach to the geometric mean of SPD matrices. *SIAM Journal on Matrix Analysis*.
