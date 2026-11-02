# The Missing Pieces 

## Contents
- `evidence/damaged_colour/`: damaged colour previews for visual context only.
- `evidence/images/`: grayscale detector inputs (512×512). Missing regions contain placeholders, not observations.
- `evidence/masks/`: binary validity masks, where 255 = valid and 0 = missing.
- `evidence/contact_sheet.png`: all candidates and masks for convenient inspection.
- `tools/watermark_model.py`: deterministic key-dependent spreading-template generator and embedder using the lecture notation.
- `tools/detect_starter.py`: detector starter with TODOs.

## Requirements and use
Requires Python 3, NumPy and Pillow. From this directory, run:

```bash
python tools/detect_starter.py
```

Complete `detector_statistics()` before running. The binary validity mask must be applied in **both** the raw correlation and the null-variance estimate. Pixels outside the mask are placeholders and must not contribute to either statistic. The perceptual weighting `alpha` is a separate quantity: the challenge fixes `alpha[n1,n2] = 1` everywhere, but the detector code keeps it explicit in the watermark template and null variance, as in the lecture. Do not normalize the raw correlation.

The experiment uses grayscale arrays for embedding and detection. Colour previews are for visual context only.
