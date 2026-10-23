"""
geometry.py - The geometric / topological core of the Semantic Invariance Engine.

Implements (in pure NumPy, no extra dependencies):
  * Character n-gram hashing embeddings (robust to residual spelling noise)
  * Poincare-ball hyperbolic geometry   (Module 3 of the readme)
        exp map at origin, geodesic distance d_H
  * Entropic optimal transport (Sinkhorn) and
    Fused Gromov-Wasserstein distance   (Module 4 - length invariance)
  * 0-dimensional persistent homology + persistence landscapes
                                        (Module 5 - topological invariance)
"""
import zlib
import numpy as np

EPS = 1e-9
ROLE_INDEX = {"AGENT": 0, "VERB": 1, "PATIENT": 2, "OTHER": 3}


# ---------------------------------------------------------------------------
# Token embeddings
# ---------------------------------------------------------------------------
def hash_embed(token, dim=64):
    """Deterministic bag-of-char-trigrams embedding (unit norm)."""
    v = np.zeros(dim)
    t = f"<{token}>"
    for n in (2, 3):
        for i in range(len(t) - n + 1):
            h = zlib.crc32(t[i:i + n].encode())
            v[h % dim] += 1.0 if (h >> 16) & 1 else -1.0
    return v / (np.linalg.norm(v) + EPS)

def token_matrix(content, roles, dim=64, role_weight=1.0):
    """Tangent vectors = [char-gram embedding | role one-hot]."""
    X = np.zeros((len(content), dim + len(ROLE_INDEX)))
    for i, (c, r) in enumerate(zip(content, roles)):
        X[i, :dim] = hash_embed(c, dim)
        X[i, dim + ROLE_INDEX[r]] = role_weight
    return X


# ---------------------------------------------------------------------------
# Poincare ball  B^d = {x : ||x|| < 1}   (curvature -1)
# ---------------------------------------------------------------------------
def expmap0(v, scale=0.5):
    """Map Euclidean tangent vectors at the origin onto the Poincare ball."""
    v = np.atleast_2d(v) * scale
    n = np.linalg.norm(v, axis=1, keepdims=True) + EPS
    return np.tanh(n) * v / n

def poincare_dist(X, Y):
    """Pairwise geodesic distances d_H(x, y) = arcosh(1 + 2|x-y|^2 / ((1-|x|^2)(1-|y|^2)))."""
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    sq = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    nx = 1 - (X ** 2).sum(1)[:, None]
    ny = 1 - (Y ** 2).sum(1)[None, :]
    return np.arccosh(1 + 2 * sq / np.maximum(nx * ny, EPS))

def hyperbolic_centroid(X, w):
    """Weighted Einstein-midpoint-like centroid via the tangent space at 0."""
    n = np.linalg.norm(X, axis=1, keepdims=True) + EPS
    logs = np.arctanh(np.clip(n, 0, 1 - 1e-6)) * X / n        # log map at origin
    return expmap0((w[:, None] * logs).sum(0), scale=1.0)


# ---------------------------------------------------------------------------
# Optimal transport
# ---------------------------------------------------------------------------
def sinkhorn(p, q, C, eps=0.05, iters=200):
    Cn = (C - C.min()) / (C.max() - C.min() + EPS)        # rescale -> no underflow
    K = np.exp(-Cn / eps) + 1e-300
    u = np.ones_like(p)
    for _ in range(iters):
        v = q / (K.T @ u + EPS)
        u = p / (K @ v + EPS)
    return u[:, None] * K * v[None, :]

def fused_gromov_wasserstein(M, D1, D2, p, q, alpha=0.5, eps=0.05, outer=20):
    """
    Entropic FGW (Peyre et al. 2016; Vayer et al. 2019) with square loss:
        min_T  sum (1-a) M_ik T_ik + a sum |D1_ij - D2_kl|^2 T_ik T_jl
    Solved by projected (mirror) descent: linearise the quadratic term,
    then solve an entropic OT problem with Sinkhorn.
    """
    T = np.outer(p, q)
    const = (D1 ** 2) @ p[:, None] @ np.ones((1, len(q))) + \
            np.ones((len(p), 1)) @ q[None, :] @ (D2 ** 2).T
    for _ in range(outer):
        L = const - 2 * D1 @ T @ D2.T                     # GW gradient term
        T = sinkhorn(p, q, (1 - alpha) * M + alpha * L, eps)
    L = const - 2 * D1 @ T @ D2.T
    return float(((1 - alpha) * M * T).sum() + alpha * (L * T).sum()), T


# ---------------------------------------------------------------------------
# Topological data analysis - H0 persistence via single-linkage (Prim's MST)
# ---------------------------------------------------------------------------
def h0_persistence(Dm):
    """Death times of 0-dim homology classes of the Vietoris-Rips filtration."""
    n = len(Dm)
    if n < 2:
        return np.array([])
    in_tree = np.zeros(n, bool); in_tree[0] = True
    best = Dm[0].copy(); deaths = []
    for _ in range(n - 1):
        best[in_tree] = np.inf
        j = int(np.argmin(best)); deaths.append(best[j]); in_tree[j] = True
        best = np.minimum(best, Dm[j])
    return np.sort(np.array(deaths))

def persistence_landscape(deaths, grid, k=3):
    """lambda_1..k on a grid for pairs (birth=0, death=d): triangle peaks at d/2."""
    L = np.zeros((k, len(grid)))
    if len(deaths) == 0:
        return L
    tri = np.maximum(0, np.minimum(grid[None, :], deaths[:, None] - grid[None, :]))
    tri = -np.sort(-tri, axis=0)
    L[:min(k, len(deaths))] = tri[:k]
    return L

def landscape_distance(d1, d2, grid=np.linspace(0, 4, 80)):
    L1, L2 = persistence_landscape(d1, grid), persistence_landscape(d2, grid)
    return float(np.sqrt(((L1 - L2) ** 2).sum() * (grid[1] - grid[0])))
