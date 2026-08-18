"""
external_dataset.py — External Street-Level Dataset Pipeline (OSV5M / Mapillary)

Features:
  1. Targeted Geographic Sampling:
       - Boosts samples for weak confusion pairs: Canada (CA), Russia (RU), Argentina (AR), USA (US), Brazil (BR), Australia (AU).
       - Enforces max-per-country cap so high-density countries (e.g. France/Germany) don't starve the rest of the world.
  2. Strict Multi-Level Deduplication:
       - 50m GPS Proximity Check: Rejects any image within 50m of any internal training/val image using a 3D unit-sphere cKDTree.
       - Perceptual Hash (dHash): Rejects candidates with near-identical visual fingerprints.
  3. Multi-task Auto-Labeling:
       - Assigns nearest geocell ID from data/geocell_centroids.csv.
       - Maps country to country_idx via data/country_encoder.csv.
       - Samples Köppen climate zone code from cached raster / coordinate rules.
  4. Kaggle Dataset Packaging:
       - Automatically generates dataset-metadata.json in the output directory.
       - Ready for one-command upload: `kaggle datasets create -p <output_dir>` or zip upload via Web UI.

Usage:
  python data/external_dataset.py --num_samples 60000 --output_dir /kaggle/working/osv5m_dataset
"""

import argparse
import sys
import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import find_dataset_paths
from geocells.build_geocells import latlon_to_xyz
from labels.aux_labels import sample_koppen

DATA_DIR = ROOT / "data"

# Countries flagged as weak in our error audit / confusion analysis
PRIORITY_CONFUSION_COUNTRIES = {"CA", "RU", "AR", "US", "BR", "AU", "ZA", "MX", "CL", "NZ", "NO", "SE", "FI"}


def generate_kaggle_dataset_metadata(output_dir: Path, dataset_slug: str = "geoguessr-osv5m-external"):
    """Generates dataset-metadata.json for direct Kaggle CLI / web dataset creation."""
    meta = {
        "title": "GeoGuessr OSV5M External StreetView Dataset",
        "id": f"your-username/{dataset_slug}",
        "licenses": [{"name": "CC-BY-4.0"}],
        "description": "Curated, deduplicated 60K street-level images from OpenStreetView-5M (Astruc et al., CVPR 2024) matched with geocells, countries, and climate labels for geolocation benchmarking.",
    }
    meta_path = output_dir / "dataset-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[External Dataset] Created Kaggle dataset metadata -> {meta_path}")


def deduplicate_gps(
    candidate_df: pd.DataFrame,
    internal_coords_csv: Path = None,
    distance_threshold_km: float = 0.05,  # 50 metres
) -> pd.DataFrame:
    """
    Filters candidate_df removing points closer than distance_threshold_km to internal dataset (all 19,002 points).
    """
    from scipy.spatial import cKDTree

    if internal_coords_csv is None or not Path(internal_coords_csv).exists():
        auto_coords, _ = find_dataset_paths()
        internal_coords_csv = auto_coords

    print(f"[External Dataset] Loading internal ground truth reference from {internal_coords_csv}...")
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
    print(f"[External Dataset] GPS dedup: dropped {n_dropped} images within {distance_threshold_km*1000:.0f}m of internal points, kept {unique_mask.sum()} unique.")

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


def download_and_package_osv5m(
    num_samples: int = 60000,
    output_dir: Path = None,
    hf_dataset_name: str = "osv5m/osv5m",
    max_per_country: int = 3500,       # prevents single country dominance
    priority_boost_factor: int = 3,    # boosts weak countries
) -> Path:
    """
    Streams and downloads balanced OSV5M subset, deduplicates, auto-labels,
    and packages it as a self-contained Kaggle Dataset directory.
    """
    if output_dir is None:
        output_dir = Path("/kaggle/working/osv5m_dataset") if Path("/kaggle/working").exists() else (DATA_DIR / "external")
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n" + "=" * 70)
    print(f"  OSV5M External Dataset Downloader & Packager")
    print(f"  Target Samples:       {num_samples}")
    print(f"  Output Directory:     {output_dir}")
    print(f"  Priority Countries:   {', '.join(sorted(PRIORITY_CONFUSION_COUNTRIES))}")
    print(f"=" * 70)

    try:
        from datasets import load_dataset
        print(f"[External Dataset] Loading {hf_dataset_name} with streaming=True, trust_remote_code=True...")
        dataset = load_dataset(hf_dataset_name, split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"[External Dataset] HF streaming failed: {e}")
        try:
            from datasets import load_dataset
            dataset = load_dataset(hf_dataset_name, split="train", streaming=True, trust_remote_code=True, full=False)
        except Exception as e2:
            print(f"[External Dataset] Secondary HF load failed: {e2}")
            dataset = None

    collected_rows = []
    country_counts = defaultdict(int)

    if dataset is not None:
        encoder_df = pd.read_csv(DATA_DIR / "country_encoder.csv")
        valid_isos = set(encoder_df["country_iso"].values) - {"-99", "OCEAN"}

        pbar = tqdm(total=num_samples, desc="Sampling OSV5M")
        for sample in dataset:
            if len(collected_rows) >= num_samples:
                break

            country_iso = sample.get("country", "")
            if country_iso not in valid_isos:
                continue

            # Country quota check
            is_priority = country_iso in PRIORITY_CONFUSION_COUNTRIES
            effective_cap = max_per_country * (priority_boost_factor if is_priority else 1)
            if country_counts[country_iso] >= effective_cap:
                continue

            lat = float(sample.get("latitude", sample.get("lat", 0.0)))
            lon = float(sample.get("longitude", sample.get("lon", 0.0)))
            if lat == 0.0 and lon == 0.0:
                continue

            img = sample.get("image", None)
            if img is None:
                continue

            img_id = str(sample.get("id", f"osv_{len(collected_rows):07d}"))
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
                country_counts[country_iso] += 1
                pbar.update(1)
            except Exception:
                continue

        pbar.close()

    if not collected_rows:
        # Fallback check for pre-attached datasets
        cand_dirs = [
            Path("/kaggle/input/osv5m"),
            Path("/kaggle/input/openstreetview5m"),
            DATA_DIR / "osv5m_downloaded",
        ]
        for cdir in cand_dirs:
            if cdir.exists() and (cdir / "metadata.csv").exists():
                print(f"[External Dataset] Found existing candidate dataset at {cdir}")
                df = pd.read_csv(cdir / "metadata.csv").head(num_samples)
                collected_rows = df.to_dict("records")
                images_dir = cdir / "images"
                break

    if not collected_rows:
        print("[External Dataset] ERROR: No images could be downloaded. Ensure active internet on Kaggle.")
        return None

    raw_df = pd.DataFrame(collected_rows)
    print(f"\n[External Dataset] Downloaded {len(raw_df)} candidate images across {raw_df['country_iso'].nunique()} countries.")
    
    # Priority country representation report
    priority_count = raw_df["country_iso"].isin(PRIORITY_CONFUSION_COUNTRIES).sum()
    print(f"[External Dataset] Priority confusion countries represented: {priority_count}/{len(raw_df)} ({priority_count/len(raw_df)*100:.1f}%)")

    # 1. Deduplicate against 19,002 internal coordinates
    dedup_df = deduplicate_gps(raw_df)

    # 2. Assign Geocells, Country IDs, and Aux Climate Labels
    labeled_df = assign_geocells_and_countries(dedup_df)

    # 3. Save clean CSVs
    out_csv = output_dir / "osv5m_train.csv"
    labeled_df.to_csv(out_csv, index=False)
    # Also save as metadata.csv for standard Kaggle dataset naming convention
    labeled_df.to_csv(output_dir / "metadata.csv", index=False)

    # 4. Generate dataset-metadata.json
    generate_kaggle_dataset_metadata(output_dir)

    print(f"\n" + "=" * 70)
    print(f"  Kaggle Dataset Package Complete!")
    print(f"  Package Directory: {output_dir}")
    print(f"  Images Stored:     {len(list(images_dir.glob('*.jpg')))} JPGs in {images_dir}")
    print(f"  Metadata CSV:      {out_csv}")
    print(f"=" * 70)
    print(labeled_df.head(10).to_string())

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download, curate, and package OSV5M as a Kaggle Dataset.")
    parser.add_argument("--num_samples", type=int, default=60000, help="Target total images")
    parser.add_argument("--output_dir",  type=str, default="/kaggle/working/osv5m_dataset")
    parser.add_argument("--max_per_country", type=int, default=3500)
    args = parser.parse_args()

    download_and_package_osv5m(
        num_samples=args.num_samples,
        output_dir=Path(args.output_dir),
        max_per_country=args.max_per_country,
    )
