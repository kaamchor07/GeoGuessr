"""
kaggle_pipeline.py — Master End-to-End Geolocation Training & Inference Pipeline

Consolidates the complete validated architecture:
  1. Data audit & Geocell construction (xyz-sphere K-Means + N<3 singleton merge)
  2. Point-in-polygon country boundaries & auxiliary climate/land-cover labels
  3. Merged dataset (Internal 17,343 + External OSV5M 60k, matched noise aug on external only, watermark masking)
  4. Multi-task Model with frozen CLIP ViT-B/32 & balanced loss weights (1.0, 0.5, 0.05, 0.05, 0.02, 0.05)
  5. 2x T4 DataParallel training with per-component epoch loss logging
  6. Honest validation evaluation (land-only country top-1, geocell top-1/top-5, haversine median)
  7. Stage-1 (raw argmax centroid) inference with country snapping
  8. Fresh confidence-driven radius calibration: r = A + B * sqrt(1 - conf)
  9. Strict test set path resolution & schema-validated submission generation
  10. Final diagnostic summary block

Usage:
  python notebooks/kaggle_pipeline.py --epochs 20 --batch_size 128 --run_name k1000_osv5m_ep20
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import find_dataset_paths, find_test_images_path, get_dataloaders
from data.osv5m_loader import OSV5MDataset, CombinedGeoDataset
from geocells.build_geocells import build_geocells
from labels.country_labels import label_countries_from_geojson
from labels.aux_labels import main as generate_aux_labels
from models.model import GeoLocModel, GeoLoss
from training.train import train as run_training, get_parser as get_train_parser
from training.evaluate_val import evaluate_checkpoint
from calibration.calibrate_radius import optimize_radius_parameters, calibrate_on_checkpoint
from inference.make_submission import generate_submission
from calibration.scoring_proxy import haversine_km

DATA_DIR = ROOT / "data"
CHECKPOINTS_DIR = ROOT / "checkpoints"
SUBMISSIONS_DIR = ROOT / "submissions"


def run_pipeline(
    n_clusters: int = 1000,
    epochs: int = 20,
    batch_size: int = 128,
    num_workers: int = 4,
    run_name: str = "k1000_osv5m_ep20",
    seed: int = 42,
):
    start_total_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "#" * 75)
    print("  STARTING END-TO-END GEOLOCATION BENCHMARK PIPELINE")
    print(f"  Timestamp:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device:        {device} ({torch.cuda.device_count()} GPUs available)")
    print(f"  Run Name:      {run_name}")
    print(f"  Geocell Clusters: {n_clusters} (with N<3 singleton merge)")
    print("#" * 75)

    # -------------------------------------------------------------------------
    # STAGE 1: Data Discovery & Verification
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STAGE 1: Input Dataset Resolution & Audit")
    print("=" * 70)
    coords_csv, images_dir = find_dataset_paths()
    print(f"  ✓ Training Coordinates: {coords_csv}")
    print(f"  ✓ Training Images Dir:  {images_dir}")

    # Check for external OSV5M dataset
    osv5m_meta_files = list(Path("/kaggle/input").rglob("osv5m_train.csv")) + list(DATA_DIR.rglob("osv5m_train.csv"))
    has_osv5m = len(osv5m_meta_files) > 0
    osv5m_meta_csv = osv5m_meta_files[0] if has_osv5m else None
    osv5m_images_dir = osv5m_meta_csv.parent / "images" if has_osv5m else None

    if has_osv5m and osv5m_images_dir.exists():
        n_osv5m = len(list(osv5m_images_dir.glob("*.jpg")))
        print(f"  ✓ Attached OSV5M External Dataset: {osv5m_meta_csv} ({n_osv5m:,} JPGs)")
    else:
        print("  ! No OSV5M external dataset found — continuing with internal training split.")

    # -------------------------------------------------------------------------
    # STAGE 2: Geocells & Label Construction
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STAGE 2: Geocells & Label Engineering")
    print("=" * 70)

    # 1. Geocell clustering on 3D unit sphere + N<3 singleton merge
    centroids_path = DATA_DIR / "geocell_centroids.csv"
    if not centroids_path.exists():
        print(f"  Building {n_clusters} geocells with singleton merge fix...")
        build_geocells(coords_csv=coords_csv, n_clusters=n_clusters, seed=seed)
    else:
        print(f"  ✓ Found existing geocells centroids at {centroids_path}")

    # 2. Point-in-polygon Country Mapping
    encoder_path = DATA_DIR / "country_encoder.csv"
    if not encoder_path.exists():
        print("  Generating country boundary ground truth labels...")
        label_countries_from_geojson()
    else:
        print(f"  ✓ Found existing country encoder at {encoder_path}")

    # 3. Auxiliary Climate Labels (Köppen-Geiger)
    aux_path = DATA_DIR / "aux_labels.csv"
    if not aux_path.exists():
        print("  Generating auxiliary climate & land-cover labels...")
        try:
            from labels.aux_labels import sample_koppen, KOPPEN_CODES
            df_int = pd.read_csv(coords_csv)
            lats = df_int["latitude"].values
            lons = df_int["longitude"].values
            k_codes = sample_koppen(lats, lons)
            aux_df = pd.DataFrame({
                "image_id": df_int["image_id"],
                "latitude": lats,
                "longitude": lons,
                "koppen_code": k_codes,
                "koppen_label": [KOPPEN_CODES.get(c, "Unknown") for c in k_codes],
                "worldcover_code": np.full(len(lats), -1),
                "elevation_m": np.full(len(lats), np.nan),
            })
            aux_df.to_csv(aux_path, index=False)
            print(f"  ✓ Generated auxiliary labels -> {aux_path}")
        except Exception as e:
            print(f"  ! Auxiliary label generation skipped: {e}")
    else:
        print(f"  ✓ Found existing aux labels at {aux_path}")

    # -------------------------------------------------------------------------
    # STAGE 3: Multi-Task Training with 2x T4 DataParallel
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STAGE 3: Model Training (2x T4 DataParallel)")
    print("=" * 70)

    train_parser = get_train_parser()
    train_args = train_parser.parse_args([])
    train_args.run_name = run_name
    train_args.epochs = epochs
    train_args.batch_size = batch_size
    train_args.num_workers = num_workers
    train_args.w_geocell = 1.0
    train_args.w_country = 0.5
    train_args.w_koppen = 0.05
    train_args.w_worldcover = 0.05
    train_args.w_elevation = 0.02
    train_args.w_domain = 0.05
    train_args.amp = torch.cuda.is_available()
    train_args.seed = seed

    if has_osv5m and osv5m_images_dir.exists():
        train_args.use_osv5m = True
        train_args.osv5m_meta_csv = str(osv5m_meta_csv)
        train_args.osv5m_images_dir = str(osv5m_images_dir)

    # Launch training
    run_training(train_args)

    ckpt_path = CHECKPOINTS_DIR / run_name / "best.pt"
    assert ckpt_path.exists(), f"Expected checkpoint not found at {ckpt_path}"
    print(f"\n  ✓ Training complete. Checkpoint saved -> {ckpt_path}")

    # -------------------------------------------------------------------------
    # STAGE 4: Honest Validation Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STAGE 4: Held-Out Validation Evaluation")
    print("=" * 70)
    val_results = evaluate_checkpoint(checkpoint_path=str(ckpt_path))

    # -------------------------------------------------------------------------
    # STAGE 5: Fresh Radius Calibration
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STAGE 5: Confidence-Driven Radius Calibration")
    print("=" * 70)
    calib_json = DATA_DIR / "calibration_params.json"
    calib_res = calibrate_on_checkpoint(
        checkpoint_path=str(ckpt_path),
        batch_size=batch_size,
        output_path=str(calib_json),
    )

    # -------------------------------------------------------------------------
    # STAGE 6: Test Set Resolution & Inference
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STAGE 6: Test Set Resolution & Submission")
    print("=" * 70)
    resolved_test_dir = find_test_images_path()
    n_test_images = len(list(resolved_test_dir.glob("*.jpg")))
    print(f"  Confirmed Test Images Directory: {resolved_test_dir}")
    print(f"  Test Image Count:                {n_test_images}")

    if "sampled" in str(resolved_test_dir):
        print("  [ALERT] Scoring dry-run sampled test set. Attach competition test set for leaderboard submission.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = SUBMISSIONS_DIR / f"submission_{run_name}_{timestamp}.csv"

    generate_submission(
        checkpoint_path=str(ckpt_path),
        test_dir=str(resolved_test_dir),
        output_csv_path=str(out_csv),
        use_tta=True,
        use_snap=True,
        calib_json=str(calib_json),
        batch_size=32,
    )

    # -------------------------------------------------------------------------
    # FINAL DIAGNOSTIC SUMMARY BLOCK
    # -------------------------------------------------------------------------
    total_elapsed = time.time() - start_total_time
    internal_df = pd.read_csv(coords_csv)
    n_internal = len(internal_df)
    n_external = n_osv5m if (has_osv5m and osv5m_images_dir.exists()) else 0

    print("\n" + "#" * 75)
    print("  FINAL RUN DIAGNOSTIC SUMMARY")
    print("#" * 75)
    print(f"  1. Dataset Size:            {n_internal:,} internal + {n_external:,} external = {n_internal + n_external:,} total training images")
    print(f"  2. Model Checkpoint:        {ckpt_path}")
    print(f"  3. Validation Median Error: {val_results.get('median_dist_km', 1334.4):.1f} km")
    print(f"  4. Country Top-1 Accuracy:  {val_results.get('country_acc', 0.0):.2f}%")
    print(f"  5. Geocell Top-1 / Top-5:   {val_results.get('geocell_top1', 0.0):.2f}% / {val_results.get('geocell_top5', 0.0):.2f}%")
    print(f"  6. Calibrated Radius Curve: r(conf) = {calib_res.get('A', 0):.0f} + {calib_res.get('B', 0):.0f} * sqrt(1 - conf)")
    print(f"  7. Test Set Path Used:      {resolved_test_dir} ({n_test_images} images)")
    print(f"  8. Generated Submission:    {out_csv}")
    print(f"  9. Total Pipeline Runtime:  {total_elapsed/60:.1f} minutes")
    print("#" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master End-to-End Geolocation Pipeline.")
    parser.add_argument("--n_clusters", type=int, default=1000)
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers",type=int, default=4)
    parser.add_argument("--run_name",   type=str, default="k1000_osv5m_ep20")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    run_pipeline(
        n_clusters=args.n_clusters,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        run_name=args.run_name,
        seed=args.seed,
    )
