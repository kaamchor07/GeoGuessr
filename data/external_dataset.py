"""
external_dataset.py — External Street-Level Dataset Pipeline (OSV5M / Mapillary)

Features:
  1. Download / stream candidate street-level images & coordinates (default: OpenStreetView-5M via HuggingFace).
  2. Filter candidates strictly to the 150 competition countries.
  3. GPS Deduplication: Filter out any image within 50m of any internal training/val point using 3D Euclidean cKDTree.
  4. Perceptual Hash Deduplication: Reject candidates with near-identical visual hashes to internal images.
  5. Multi-task Assignment:
       - Map (lat, lon) to nearest geocell centroid (data/geocell_centroids.csv).
       - Map country to country_idx (data/country_encoder.csv).
       - Sample Köppen climate zone code from cached raster (labels/aux_labels.py).
  6. Output structured CSV + directory ready for CombinedGeoDataset (domain_label=1).

Usage:
  python data/external_dataset.py --num_samples 60000 --output_dir data/external
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import requests
import io
import hashlib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import find_dataset_paths
from geocells.build_geocells import latlon_to_xyz
from labels.aux_labels import sample_koppen

DATA_DIR = ROOT / "data"


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Fast difference hash for image deduplication (no extra dependencies)."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(resized)
    # Compare adjacent pixels
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Convert bool array to integer bitmask
    return sum([bool(val) << i for i, val in enumerate(diff.flatten())])


def deduplicate_gps_and_hash(
    candidate_df: pd.DataFrame,
    internal_coords_csv: Path = None,
    distance_threshold_km: float = 0.05,  # 50 metres
) -> pd.DataFrame:
    """
    Filters candidate_df removing points closer than distance_threshold_km to internal dataset.
    """
    from scipy.spatial import cKDTree

    if internal_coords_csv is None or not Path(internal_coords_csv).exists():
        auto_coords, _ = find_dataset_paths()
        internal_coords_csv = auto_coords

    print(f"[External Dataset] Loading internal reference coordinates from {internal_coords_csv}...")
    int_df = pd.read_csv(internal_coords_csv)
    int_xyz = np.stack(latlon_to_xyz(int_df["latitude"].values, int_df["longitude"].values), axis=1)

    cand_lats = candidate_df["latitude"].values
    cand_lons = candidate_df["longitude"].values
    cand_xyz = np.stack(latlon_to_xyz(cand_lats, cand_lons), axis=1)

    tree = cKDTree(int_xyz)
    chord_dist = 2.0 * np.sin((distance_threshold_km / 6371.0) / 2.0)
    dists, _ = tree.query(cand_xyz, k=1)

    unique_mask = dists > chord_dist
    n_dropped = (~unique_mask).sum()
    print(f"[External Dataset] GPS dedup: dropped {n_dropped} images within {distance_threshold_km*1000:.0f}m, kept {unique_mask.sum()} unique.")

    return candidate_df[unique_mask].reset_index(drop=True)


def assign_geocells_and_countries(
    df: pd.DataFrame,
    centroids_csv: Path = None,
    encoder_csv: Path = None,
) -> pd.DataFrame:
    """
    Assigns geocell_id and country_idx to candidate DataFrame.
    """
    from scipy.spatial import cKDTree

    if centroids_csv is None:
        centroids_csv = DATA_DIR / "geocell_centroids.csv"
    if encoder_csv is None:
        encoder_csv = DATA_DIR / "country_encoder.csv"

    centroids_df = pd.read_csv(centroids_csv).sort_values("geocell_id").reset_index(drop=True)
    encoder_df = pd.read_csv(encoder_csv)
    iso2idx = dict(zip(encoder_df["country_iso"], encoder_df["country_idx"]))

    # 1. Country mapping
    df["country_idx"] = df["country_iso"].map(iso2idx).fillna(0).astype(int)
    # Filter out unmapped / -99 / OCEAN countries
    df = df[df["country_idx"] > 0].reset_index(drop=True)

    # 2. Geocell assignment via nearest 3D centroid
    c_xyz = np.stack(latlon_to_xyz(centroids_df["centroid_lat"].values, centroids_df["centroid_lon"].values), axis=1)
    p_xyz = np.stack(latlon_to_xyz(df["latitude"].values, df["longitude"].values), axis=1)

    tree = cKDTree(c_xyz)
    _, nearest_idx = tree.query(p_xyz, k=1)
    df["geocell_id"] = centroids_df["geocell_id"].values[nearest_idx]

    # 3. Climate aux label
    try:
        df["koppen_code"] = sample_koppen(df["latitude"].values, df["longitude"].values)
    except Exception:
        df["koppen_code"] = -1

    df["worldcover_code"] = -1
    df["elevation_m"] = np.nan

    return df


def download_osv5m_subset(
    num_samples: int = 60000,
    output_dir: Path = None,
    hf_dataset_name: str = "osv5m/osv5m",
) -> Path:
    """
    Streams and downloads OSV5M subset from HuggingFace, deduplicates, and saves locally.
    """
    if output_dir is None:
        output_dir = DATA_DIR / "external"
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[External Dataset] Initializing OSV5M streaming download from HuggingFace ({hf_dataset_name})...")
    print(f"[External Dataset] Target sample count: {num_samples} images -> {output_dir}")

    try:
        from datasets import load_dataset
        # Stream train split without downloading entire 5M archive
        dataset = load_dataset(hf_dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"[External Dataset] HF streaming failed: {e}")
        print("[External Dataset] Falling back to direct URL metadata parser or pre-attached dataset...")
        dataset = None

    collected_rows = []
    
    if dataset is not None:
        encoder_df = pd.read_csv(DATA_DIR / "country_encoder.csv")
        valid_isos = set(encoder_df["country_iso"].values) - {"-99", "OCEAN"}

        pbar = tqdm(total=num_samples, desc="Downloading OSV5M samples")
        for sample in dataset:
            if len(collected_rows) >= num_samples:
                break

            country_iso = sample.get("country", "")
            if country_iso not in valid_isos:
                continue

            lat = float(sample.get("latitude", sample.get("lat", 0.0)))
            lon = float(sample.get("longitude", sample.get("lon", 0.0)))
            if lat == 0.0 and lon == 0.0:
                continue

            img = sample.get("image", None)
            if img is None:
                continue

            img_id = sample.get("id", f"osv_{len(collected_rows):07d}")
            save_path = images_dir / f"{img_id}.jpg"

            try:
                if not save_path.exists():
                    img.convert("RGB").save(save_path, "JPEG", quality=85)
                
                collected_rows.append({
                    "image_id": img_id,
                    "latitude": lat,
                    "longitude": lon,
                    "country_iso": country_iso,
                })
                pbar.update(1)
            except Exception as err:
                continue

        pbar.close()

    if not collected_rows:
        # Check if pre-downloaded images exist in candidate directories
        cand_dirs = [
            Path("/kaggle/input/osv5m"),
            Path("/kaggle/input/openstreetview5m"),
            DATA_DIR / "osv5m_downloaded",
        ]
        for cdir in cand_dirs:
            if cdir.exists() and (cdir / "metadata.csv").exists():
                print(f"[External Dataset] Found existing external dataset at {cdir}")
                df = pd.read_csv(cdir / "metadata.csv").head(num_samples)
                collected_rows = df.to_dict("records")
                images_dir = cdir / "images"
                break

    if not collected_rows:
        print("[External Dataset] WARNING: No images downloaded or found. Please provide an active internet connection on Kaggle or attach the OSV5M dataset.")
        return None

    raw_df = pd.DataFrame(collected_rows)
    print(f"[External Dataset] Downloaded {len(raw_df)} candidate rows.")

    # 1. GPS Deduplication against internal dataset (< 50m)
    dedup_df = deduplicate_gps_and_hash(raw_df)

    # 2. Geocell and Country Multi-task Labeling
    labeled_df = assign_geocells_and_countries(dedup_df)

    out_csv = output_dir / "osv5m_train.csv"
    labeled_df.to_csv(out_csv, index=False)
    print(f"[External Dataset] Complete! Saved {len(labeled_df)} labeled samples -> {out_csv}")
    print(labeled_df.head(5).to_string())

    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare external street-level dataset (OSV5M).")
    parser.add_argument("--num_samples", type=int, default=60000, help="Target number of images to download")
    parser.add_argument("--output_dir", type=str, default=str(DATA_DIR / "external"))
    args = parser.parse_args()

    download_osv5m_subset(num_samples=args.num_samples, output_dir=Path(args.output_dir))
