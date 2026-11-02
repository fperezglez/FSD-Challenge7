"""Spread-spectrum watermark model following the lecture notation.

This is a small deterministic classroom implementation, not a production system.
The returned signature is the unweighted signed spreading template u. The perceptual
mask alpha is applied in embed(); in this challenge alpha=1 everywhere. The separate
binary validity mask indicates which observed pixels survive simulated damage.
"""
from __future__ import annotations
import hashlib
import numpy as np

# M=32 known antipodal symbols b_i in {-1,+1}.
KNOWN_B = np.array([
    -1, +1, +1, -1, +1, +1, +1, +1,
    +1, +1, -1, -1, -1, -1, -1, +1,
    +1, +1, +1, -1, -1, -1, -1, -1,
    +1, -1, -1, -1, -1, +1, +1, -1,
], dtype=np.float64)
DEFAULT_KEY = 1511506142  # 0x5A17C0DE


def _seed(key: int | str, purpose: str) -> int:
    raw = f"spread-spectrum-classroom:{key}:{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little", signed=False)


def make_signature(shape: tuple[int, int], key: int | str = DEFAULT_KEY,
                   b: np.ndarray = KNOWN_B) -> np.ndarray:
    """Generate the base template u[n] = sum_i b_i s[n] 1_{T_i}[n].

    The actual spread-spectrum watermark is w[n] = alpha[n] * u[n], where
    alpha is the perceptual amplitude mask. In this challenge alpha=1, so u=w.
    The key defines a balanced non-overlapping tiling T_i and a pseudorandom
    spreading sequence s with finite-array mean 0 and variance 1.
    """
    height, width = shape
    total = height * width
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    M = int(b.size)
    if M < 1 or total % M != 0:
        raise ValueError("Pixel count must be divisible by the number of symbols.")
    if not np.all(np.isin(b, [-1.0, 1.0])):
        raise ValueError("Every b_i must be -1 or +1.")
    perm = np.random.default_rng(_seed(key, "tiling")).permutation(total)
    tile_id = np.empty(total, dtype=np.int16)
    size = total // M
    for i in range(M):
        tile_id[perm[i*size:(i+1)*size]] = i
    s = np.random.default_rng(_seed(key, "spreading")).standard_normal(total)
    s = (s - s.mean()) / s.std(ddof=0)
    return (b[tile_id] * s).reshape(shape).astype(np.float64)


def embed(cover: np.ndarray, gamma: float, key: int | str = DEFAULT_KEY,
          b: np.ndarray = KNOWN_B, alpha: np.ndarray | None = None) -> np.ndarray:
    """Embed y=x+gamma*alpha*u and quantize to uint8.

    ``alpha`` is the perceptual amplitude mask, not the binary tear/validity
    mask. If omitted, alpha is identically one.
    """
    x = np.asarray(cover, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Expected a grayscale 2-D cover image.")
    u = make_signature(x.shape, key=key, b=b)
    if alpha is None:
        alpha_arr = np.ones_like(x, dtype=np.float64)
    else:
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        if alpha_arr.shape != x.shape or not np.all(np.isfinite(alpha_arr)) or np.any(alpha_arr < 0):
            raise ValueError("alpha must be a finite, nonnegative array matching the cover shape")
    w = alpha_arr * u
    return np.clip(np.rint(x + float(gamma)*w), 0, 255).astype(np.uint8)
