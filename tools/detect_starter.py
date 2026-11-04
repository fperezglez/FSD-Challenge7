"""Student starter: detect the known spread-spectrum watermark.

Run from the Student package directory:
    python tools/detect_starter.py

Only NumPy and Pillow are required.
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


def detector_statistics(y, mask, u, alpha=None):
    """Return the raw correlation r_Omega and estimated H0 standard deviation.

    Parameters
    ----------
    y : 2-D array
        Damaged grayscale image. Values in missing regions are placeholders.
    mask : 2-D array
        Validity mask: nonzero means that the pixel survived; zero means missing.
    u : 2-D array
        Key-dependent signed spreading template, including the known symbols b_i.
    alpha : 2-D array or None
        Perceptual mask. It is identically one in this challenge.

    The validity mask is used first: all calculations below use only surviving
    pixels. The perceptual mask alpha is a separate weighting of the watermark.
    """
    # Convert the inputs to arrays and make the validity mask Boolean.
    y = np.asarray(y, dtype=float)
    valid = np.asarray(mask) != 0
    u = np.asarray(u, dtype=float)

    # In this challenge alpha[n1,n2] = 1 for every pixel.
    if alpha is None:
        alpha = np.ones(y.shape, dtype=float)
    else:
        alpha = np.asarray(alpha, dtype=float)

    # All arrays must describe the same image.
    if y.shape != valid.shape or y.shape != u.shape or y.shape != alpha.shape:
        raise ValueError("y, mask, u and alpha must have the same shape")
    if not np.any(valid):
        raise ValueError("The validity mask contains no surviving pixels")

    # Keep only pixels that actually survived the tearing.
    # Placeholder values outside the mask are discarded here.
    y_valid = y[valid]
    u_valid = u[valid]
    alpha_valid = alpha[valid]

    # TODO 1: Compute the mean of the surviving image pixels.
    y_mean = ...

    # Centre the surviving samples around their mean.
    y_centered = y_valid - y_mean

    # TODO 2: Compute the raw (NOT normalized) correlation:
    # r_Omega = sum (y - y_mean) * alpha * u, over valid pixels only.
    r = ...

    # TODO 3: Estimate the standard deviation under H0:
    # sigma_0 = sqrt(sum ((y - y_mean) * alpha)^2), over valid pixels only.
    sigma0 = ...

    return float(r), float(sigma0)


def main():
    # The known symbols b_i and the key determine the reference template u.
    u = make_signature((512, 512), key=DEFAULT_KEY, b=KNOWN_B)
    alpha = np.ones((512, 512), dtype=float)  # constant perceptual mask

    print("Candidate  valid_%    r_Omega    sigma_0   3*sigma_0  decision")
    for candidate in CANDIDATES:
        image_path = IMAGE_DIR / f"{candidate}.png"
        mask_path = MASK_DIR / f"{candidate}_mask.png"

        y = np.asarray(Image.open(image_path).convert("L"), dtype=float)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0

        r, sigma0 = detector_statistics(y, mask, u, alpha)
        threshold = KAPPA * sigma0

        if r > threshold:
            decision = "TARGET"
        else:
            decision = "not detected"

        print(
            f"{candidate:8s} {mask.mean():7.3f} "
            f"{r:10.1f} {sigma0:9.1f} {threshold:10.1f}  {decision}"
        )


if __name__ == "__main__":
    main()
