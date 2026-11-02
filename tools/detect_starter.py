"""Student starter for The Missing Pieces.

Run from this folder:
    python tools/detect_starter.py
Dependencies: NumPy and Pillow.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from watermark_model import DEFAULT_KEY, KNOWN_B, make_signature

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "evidence" / "images"
MASK_DIR = ROOT / "evidence" / "masks"
CANDIDATES = [f"IMG_{i:02d}" for i in range(1, 10)]
KAPPA = 3.0


def detector_statistics(y: np.ndarray, mask: np.ndarray, u: np.ndarray,
                        alpha: np.ndarray | None = None) -> tuple[float, float]:
    """Return raw r_Omega and estimated H0 standard deviation.

    `mask` is the binary validity mask: nonzero means observed/valid.
    `alpha` is the perceptual amplitude mask from the lecture; here it is 1
    everywhere. `u` is the key-dependent signed spreading template before
    applying alpha, so the actual watermark template is w = alpha * u.
    """
    # TODO 1: Convert y, mask, and u to NumPy arrays and check their shapes.
    # TODO 2: Set alpha to an all-ones array if it is None; otherwise validate it.
    # TODO 3: Use the binary validity mask to select only Omega. Do not use
    #         placeholder pixels in the mean, correlation, or variance.
    # TODO 4: Compute ybar_Omega from the valid observed samples.
    # TODO 5: Apply alpha to the template and compute the raw correlation:
    #   r_Omega = sum_n m[n]*(y[n]-ybar_Omega)*alpha[n]*u[n].
    #         Do NOT normalize it.
    # TODO 6: Estimate sigma0^2 = sum_n m[n]*(y[n]-ybar_Omega)^2*alpha[n]^2.
    #         Since alpha=1 in this challenge, the alpha factors equal one.
    # TODO 7: Reject a degenerate mask or zero variance.
    raise NotImplementedError("Complete detector_statistics().")


def main() -> None:
    u = make_signature((512, 512), key=DEFAULT_KEY, b=KNOWN_B)
    print("Candidate valid_px valid_% r_Omega sigma0 3*sigma0 decision")
    for candidate in CANDIDATES:
        y = np.asarray(Image.open(IMAGE_DIR/f"{candidate}.png").convert("L"), dtype=np.float64)
        mask = np.asarray(Image.open(MASK_DIR/f"{candidate}_mask.png").convert("L")) > 0
        alpha = np.ones(y.shape, dtype=np.float64)  # constant perceptual mask
        r, sigma0 = detector_statistics(y, mask, u, alpha)
        threshold = KAPPA*sigma0
        decision = "TARGET" if r > threshold else "not detected"
        print(f"{candidate:8s} {int(mask.sum()):7d} {mask.mean():6.3f} {r:10.1f} {sigma0:9.1f} {threshold:10.1f} {decision}")

if __name__ == "__main__":
    main()
