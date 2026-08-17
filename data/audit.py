"""
audit.py — Step 1, PRD Section 3

Audits the training image set to:
  1. Classify each image by apparent source type (Street-View-like vs dashcam/phone)
     — detects watermark presence, aspect ratio, and noise level
  2. Compute per-image noise statistics (variance of Laplacian = sharpness proxy)
  3. Flags the bottom-left watermark region dimensions to use for masking
  4. Outputs data/audit_report.csv and prints a summary

This is designed to run on a random sample (default 500 images) so it
finishes quickly on a laptop before committing to full-dataset preprocessing.

Usage:
  python data/audit.py
  python data/audit.py --n_sample 200 --seed 42
  python data/audit.py --full   # run on ALL training images (slow)
"""

import argparse
import os
import random
import sys
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[Warning] opencv-python not installed. Install with: pip install opencv-python-headless")

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "training_dataset" / "noised_dataset" / "images"
COORDS_CSV = ROOT / "training_dataset" / "noised_dataset" / "ground_truth_coordinates.csv"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Watermark region: bottom-left corner (Google logo on Street View images)
# Approximate region as fraction of image dimensions — will be confirmed by audit
WATERMARK_FRAC_X = (0.0, 0.15)  # left 15% horizontally
WATERMARK_FRAC_Y = (0.85, 1.0)  # bottom 15% vertically


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of Laplacian — high = sharp, low = blurry/noisy."""
    if not HAS_CV2:
        return float(np.var(gray.astype(float)))
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def noise_estimate(gray: np.ndarray) -> float:
    """Estimate image noise level via MAD of high-frequency residual."""
    # Simple: std of (image - gaussian_blur)
    if HAS_CV2:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = gray.astype(float) - blurred.astype(float)
    else:
        from scipy.ndimage import uniform_filter
        blurred = uniform_filter(gray.astype(float), size=5)
        residual = gray.astype(float) - blurred
    return float(np.std(residual))


def watermark_region_mean(gray: np.ndarray) -> float:
    """Mean pixel value in the bottom-left watermark region."""
    h, w = gray.shape
    y0 = int(WATERMARK_FRAC_Y[0] * h)
    y1 = int(WATERMARK_FRAC_Y[1] * h)
    x0 = int(WATERMARK_FRAC_X[0] * w)
    x1 = int(WATERMARK_FRAC_X[1] * w)
    region = gray[y0:y1, x0:x1]
    return float(region.mean()) if region.size > 0 else 0.0


def classify_source(img_path: Path, gray: np.ndarray) -> str:
    """
    Heuristic classification of image source type.
    Returns: 'streetview' | 'dashcam' | 'unknown'

    Heuristics:
    - Aspect ratio: Street View images tend to be wider or square-ish
    - File size relative to resolution: heavy JPEG artifacts -> Street View compression
    - Watermark region: notably bright/white region = likely Google watermark
    """
    h, w = gray.shape
    aspect = w / h

    wm_mean = watermark_region_mean(gray)
    file_size_kb = img_path.stat().st_size / 1024

    # Updated heuristics after audit: ALL images are 640x640 (aspect=1.0)
    # so aspect ratio is not discriminative.
    # Primary signal: watermark region brightness (bright white = Google logo)
    # Secondary: noise level (streetview-sourced images have more noise added)
    has_watermark_hint = wm_mean > 180  # bright region in bottom-left corner

    if has_watermark_hint:
        return "streetview"
    else:
        return "dashcam_or_other"


def audit_image(img_path: Path) -> dict:
    """Analyse one image and return a dict of metrics."""
    result = {
        "image_id": img_path.stem,
        "file_size_kb": img_path.stat().st_size / 1024,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "sharpness_lapvar": None,
        "noise_estimate": None,
        "watermark_region_mean": None,
        "source_type": "unknown",
        "error": None,
    }

    try:
        if HAS_CV2:
            img = cv2.imread(str(img_path))
            if img is None:
                result["error"] = "cv2 could not read"
                return result
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            from PIL import Image
            pil = Image.open(img_path).convert("L")
            gray = np.array(pil)
            h, w = gray.shape

        result["width"] = w
        result["height"] = h
        result["aspect_ratio"] = round(w / h, 3)
        result["sharpness_lapvar"] = round(laplacian_variance(gray), 2)
        result["noise_estimate"] = round(noise_estimate(gray), 4)
        result["watermark_region_mean"] = round(watermark_region_mean(gray), 2)
        result["source_type"] = classify_source(img_path, gray)

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Audit training images for source type and noise")
    parser.add_argument(
        "--n_sample",
        type=int,
        default=500,
        help="Number of random images to audit (ignored if --full)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Audit all training images (slow on full dataset)",
    )
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "audit_report.csv"))
    args = parser.parse_args()

    all_images = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Found {len(all_images)} training images in {IMAGES_DIR}")

    if args.full:
        sample = all_images
        print("Mode: FULL dataset audit")
    else:
        random.seed(args.seed)
        sample = random.sample(all_images, min(args.n_sample, len(all_images)))
        print(f"Mode: random sample of {len(sample)} images (seed={args.seed})")

    print("Auditing images …")
    records = []
    for i, img_path in enumerate(sample):
        rec = audit_image(img_path)
        records.append(rec)
        if (i + 1) % 50 == 0 or (i + 1) == len(sample):
            print(f"  {i+1}/{len(sample)}", end="\r")

    print()
    report = pd.DataFrame(records)
    report.to_csv(args.out, index=False)
    print(f"\nSaved audit report -> {args.out}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    valid = report[report["error"].isna()]
    print(f"\n=== Image Audit Summary ({len(valid)}/{len(report)} valid) ===")

    print("\nSource type distribution:")
    print(valid["source_type"].value_counts().to_string())

    print("\nAspect ratio (W/H):")
    print(valid["aspect_ratio"].describe().round(3).to_string())

    print("\nNoise estimate (std of HF residual):")
    for src, grp in valid.groupby("source_type"):
        print(f"  {src:12s}: mean={grp['noise_estimate'].mean():.4f}  "
              f"std={grp['noise_estimate'].std():.4f}  "
              f"p90={grp['noise_estimate'].quantile(0.9):.4f}")

    print("\nSharpness (Laplacian variance):")
    for src, grp in valid.groupby("source_type"):
        print(f"  {src:12s}: mean={grp['sharpness_lapvar'].mean():.1f}  "
              f"median={grp['sharpness_lapvar'].median():.1f}")

    print("\nWatermark region mean brightness (bottom-left):")
    for src, grp in valid.groupby("source_type"):
        print(f"  {src:12s}: mean={grp['watermark_region_mean'].mean():.1f}  "
              f"std={grp['watermark_region_mean'].std():.1f}")

    # --- Watermark mask recommendation ---
    sv = valid[valid["source_type"] == "streetview"]
    if not sv.empty and "width" in sv.columns:
        median_w = int(sv["width"].median())
        median_h = int(sv["height"].median())
        wm_x0 = int(WATERMARK_FRAC_X[0] * median_w)
        wm_x1 = int(WATERMARK_FRAC_X[1] * median_w)
        wm_y0 = int(WATERMARK_FRAC_Y[0] * median_h)
        wm_y1 = int(WATERMARK_FRAC_Y[1] * median_h)
        print(f"\nRecommended watermark mask (for {median_w}×{median_h} images):")
        print(f"  x: [{wm_x0}, {wm_x1}]  y: [{wm_y0}, {wm_y1}]")
        print("  (Apply to ALL images so mask absence isn't a spurious domain signal)")

    errors = report[report["error"].notna()]
    if not errors.empty:
        print(f"\n[Warning] {len(errors)} images had errors:")
        print(errors[["image_id", "error"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
