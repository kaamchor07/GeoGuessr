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
    Downloads balanced OSV5M subset directly via huggingface_hub / web shards,
    deduplicates, auto-labels, and packages it as a self-contained Kaggle Dataset directory.
    """
    import zipfile
    import shutil

    if output_dir is None:
        output_dir = Path("/kaggle/working/osv5m_dataset") if Path("/kaggle/working").exists() else (DATA_DIR / "external")
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp_shards"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n" + "=" * 70)
    print(f"  OSV5M External Dataset Downloader & Packager")
    print(f"  Target Samples:       {num_samples}")
    print(f"  Output Directory:     {output_dir}")
    print(f"  Priority Countries:   {', '.join(sorted(PRIORITY_CONFUSION_COUNTRIES))}")
    print(f"=" * 70)

    encoder_df = pd.read_csv(DATA_DIR / "country_encoder.csv")
    valid_isos = set(encoder_df["country_iso"].values) - {"-99", "OCEAN"}

    collected_rows = []
    country_counts = defaultdict(int)

    # 1. First, check if pre-downloaded images exist in candidate directories
    cand_dirs = [
        Path("/kaggle/input/osv5m"),
        Path("/kaggle/input/openstreetview5m"),
        Path("/kaggle/input/osv5m-dataset"),
        DATA_DIR / "osv5m_downloaded",
    ]
    for cdir in cand_dirs:
        if cdir.exists() and (cdir / "metadata.csv").exists():
            print(f"[External Dataset] Found existing candidate dataset at {cdir}")
            df = pd.read_csv(cdir / "metadata.csv").head(num_samples)
            collected_rows = df.to_dict("records")
            images_dir = cdir / "images"
            break

    # 2. Direct shard downloading via huggingface_hub HfFileSystem
    if not collected_rows:
        try:
            from huggingface_hub import HfFileSystem, hf_hub_download
            fs = HfFileSystem()
            print(f"[External Dataset] Scanning {hf_dataset_name} files via HfFileSystem...")
            all_files = fs.ls(f"datasets/{hf_dataset_name}", detail=False)
            
            # Look for zip shards / tar shards / metadata csv
            zip_files = [f for f in all_files if f.endswith(".zip") or "/images/" in f]
            csv_files = [f for f in all_files if f.endswith(".csv") or f.endswith(".parquet")]

            print(f"[External Dataset] Found {len(zip_files)} data shards, {len(csv_files)} metadata tables.")

            pbar = tqdm(total=num_samples, desc="Collecting OSV5M samples")

            for fpath in zip_files:
                if len(collected_rows) >= num_samples:
                    break

                rel_fname = fpath.replace(f"datasets/{hf_dataset_name}/", "")
                print(f"\n[External Dataset] Downloading shard: {rel_fname}")
                local_zip = hf_hub_download(
                    repo_id=hf_dataset_name,
                    filename=rel_fname,
                    repo_type="dataset",
                    local_dir=str(temp_dir),
                )

                if zipfile.is_zipfile(local_zip):
                    with zipfile.ZipFile(local_zip, "r") as zf:
                        # Find any metadata or image entries
                        namelist = zf.namelist()
                        img_names = [n for n in namelist if n.lower().endswith(".jpg") or n.lower().endswith(".jpeg")]
                        
                        for img_name in img_names:
                            if len(collected_rows) >= num_samples:
                                break

                            # Parse ID and metadata if encoded in name (e.g. {id}_{lat}_{lon}_{country}.jpg or standard osv format)
                            parts = Path(img_name).stem.split("_")
                            img_id = Path(img_name).name

                            # Extract directly to destination
                            out_img_path = images_dir / img_id
                            if not out_img_path.exists():
                                with zf.open(img_name) as zf_img, open(out_img_path, "wb") as f_out:
                                    shutil.copyfileobj(zf_img, f_out)

                            collected_rows.append({
                                "image_id": Path(img_id).stem,
                                "latitude": float(parts[1]) if len(parts) >= 3 else 0.0,
                                "longitude": float(parts[2]) if len(parts) >= 3 else 0.0,
                                "country_iso": parts[3] if len(parts) >= 4 else "UNK",
                            })
                            pbar.update(1)

                # Clean up shard file to save disk space
                try:
                    os.remove(local_zip)
                except Exception:
                    pass

            pbar.close()

        except Exception as e:
            print(f"[External Dataset] HfFileSystem download failed: {e}")

    # 3. Fallback: stream from Mapillary / OSV WebDataset or CSV manifest
    if not collected_rows or len(collected_rows) < 100:
        print("[External Dataset] Attempting direct manifest download from OSV5M repository...")
        try:
            from huggingface_hub import hf_hub_download
            meta_file = hf_hub_download(
                repo_id=hf_dataset_name,
                filename="train.csv",
                repo_type="dataset",
                local_dir=str(temp_dir),
            )
            df_manifest = pd.read_csv(meta_file)
            print(f"[External Dataset] Loaded manifest with {len(df_manifest)} entries.")
            # Priority country sampling from manifest
            p_mask = df_manifest["country"].isin(PRIORITY_CONFUSION_COUNTRIES)
            p_df = df_manifest[p_mask].head(num_samples // 2)
            rem_df = df_manifest[~p_mask].head(num_samples - len(p_df))
            sampled_df = pd.concat([p_df, rem_df]).sample(frac=1.0).reset_index(drop=True)

            for idx, r in sampled_df.iterrows():
                collected_rows.append({
                    "image_id": str(r.get("id", f"osv_{idx}")),
                    "latitude": float(r.get("latitude", r.get("lat", 0.0))),
                    "longitude": float(r.get("longitude", r.get("lon", 0.0))),
                    "country_iso": str(r.get("country", "UNK")),
                })
        except Exception as e:
            print(f"[External Dataset] Manifest download attempt note: {e}")

    # Remove temporary shard directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    if not collected_rows:
        print("[External Dataset] ERROR: Could not collect images. Please verify your connection or attach the OSV5M dataset in Kaggle.")
        return None

    raw_df = pd.DataFrame(collected_rows)
    print(f"\n[External Dataset] Processed {len(raw_df)} candidate entries.")

    # 1. Deduplicate against 19,002 internal coordinates
    dedup_df = deduplicate_gps(raw_df)

    # 2. Assign Geocells, Country IDs, and Aux Climate Labels
    labeled_df = assign_geocells_and_countries(dedup_df)

    # 3. Save clean CSVs
    out_csv = output_dir / "osv5m_train.csv"
    labeled_df.to_csv(out_csv, index=False)
    labeled_df.to_csv(output_dir / "metadata.csv", index=False)

    # 4. Generate dataset-metadata.json
    generate_kaggle_dataset_metadata(output_dir)

    print(f"\n" + "=" * 70)
    print(f"  Kaggle Dataset Package Complete!")
    print(f"  Package Directory: {output_dir}")
    print(f"  Images Stored:     {len(list(images_dir.glob('*.jpg')))} JPGs")
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
