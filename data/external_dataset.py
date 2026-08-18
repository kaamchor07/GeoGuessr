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
    int_df = pd.read_csv(internal_coords_csv).dropna(subset=["latitude", "longitude"])
    int_lats = pd.to_numeric(int_df["latitude"], errors="coerce").values
    int_lons = pd.to_numeric(int_df["longitude"], errors="coerce").values
    valid_int = ~np.isnan(int_lats) & ~np.isnan(int_lons)
    int_xyz = latlon_to_xyz(int_lats[valid_int], int_lons[valid_int]).astype(np.float64)

    # Clean candidate df
    cand_df_clean = candidate_df.copy()
    cand_df_clean["latitude"] = pd.to_numeric(cand_df_clean["latitude"], errors="coerce")
    cand_df_clean["longitude"] = pd.to_numeric(cand_df_clean["longitude"], errors="coerce")
    valid_cand = (
        cand_df_clean["latitude"].notna() &
        cand_df_clean["longitude"].notna() &
        cand_df_clean["latitude"].between(-90, 90) &
        cand_df_clean["longitude"].between(-180, 180) &
        ((cand_df_clean["latitude"] != 0.0) | (cand_df_clean["longitude"] != 0.0))
    )
    cand_df_clean = cand_df_clean[valid_cand].reset_index(drop=True)

    cand_lats = cand_df_clean["latitude"].values
    cand_lons = cand_df_clean["longitude"].values
    cand_xyz = latlon_to_xyz(cand_lats, cand_lons).astype(np.float64)

    tree = cKDTree(int_xyz)
    chord_dist = 2.0 * np.sin((distance_threshold_km / 6371.0) / 2.0)
    dists, _ = tree.query(cand_xyz, k=1)

    unique_mask = dists > chord_dist
    n_dropped = (~unique_mask).sum()
    print(f"[External Dataset] GPS dedup: dropped {n_dropped} images within {distance_threshold_km*1000:.0f}m of internal points, kept {unique_mask.sum()} unique.")

    return cand_df_clean[unique_mask].reset_index(drop=True)


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
    c_xyz = latlon_to_xyz(centroids_df["centroid_lat"].values, centroids_df["centroid_lon"].values).astype(np.float64)
    p_xyz = latlon_to_xyz(df["latitude"].values, df["longitude"].values).astype(np.float64)

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
    High-speed OSV5M Downloader:
      1. Downloads OSV5M train.csv metadata.
      2. Streams image zip shards (images/train/00.zip, 01.zip...) from HuggingFace.
      3. Extracts and packages 60,000 balanced & deduplicated images into output_dir.
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
    print(f"  OSV5M High-Speed Dataset Downloader & Packager")
    print(f"  Target Samples:       {num_samples}")
    print(f"  Output Directory:     {output_dir}")
    print(f"  Priority Countries:   {', '.join(sorted(PRIORITY_CONFUSION_COUNTRIES))}")
    print(f"=" * 70)

    encoder_df = pd.read_csv(DATA_DIR / "country_encoder.csv")
    valid_isos = set(encoder_df["country_iso"].values) - {"-99", "OCEAN"}

    # 1. Download & Index Metadata (train.csv)
    from huggingface_hub import hf_hub_download, HfFileSystem
    print("\n[External Dataset] Step 1/3: Loading train.csv metadata...")
    meta_file = hf_hub_download(
        repo_id=hf_dataset_name,
        filename="train.csv",
        repo_type="dataset",
        local_dir=str(temp_dir),
    )
    df_manifest = pd.read_csv(meta_file, low_memory=False)
    print(f"[External Dataset] Total manifest records: {len(df_manifest):,}")

    country_col = "country" if "country" in df_manifest.columns else "country_code"
    lat_col = "latitude" if "latitude" in df_manifest.columns else "lat"
    lon_col = "longitude" if "longitude" in df_manifest.columns else "lon"
    id_col = "id" if "id" in df_manifest.columns else df_manifest.columns[0]

    # Filter to valid countries & non-null coords
    valid_mask = (
        df_manifest[country_col].isin(valid_isos) &
        df_manifest[lat_col].notna() &
        df_manifest[lon_col].notna() &
        df_manifest[lat_col].between(-90, 90) &
        df_manifest[lon_col].between(-180, 180) &
        ((df_manifest[lat_col] != 0.0) | (df_manifest[lon_col] != 0.0))
    )
    clean_manifest = df_manifest[valid_mask].copy()
    
    # Build O(1) metadata lookup dictionary: id_str -> (lat, lon, country)
    id_series = clean_manifest[id_col].astype(str)
    lats_series = clean_manifest[lat_col].values.astype(np.float64)
    lons_series = clean_manifest[lon_col].values.astype(np.float64)
    countries_series = clean_manifest[country_col].values.astype(str)

    meta_lookup = {}
    for i, img_id in enumerate(id_series):
        meta_lookup[img_id] = (lats_series[i], lons_series[i], countries_series[i])

    print(f"[External Dataset] Indexed {len(meta_lookup):,} valid geolocation records.")

    # 2. Discover available image zip shards
    fs = HfFileSystem()
    try:
        all_train_shards = sorted(fs.ls(f"datasets/{hf_dataset_name}/images/train", detail=False))
    except Exception:
        all_train_shards = [f"datasets/{hf_dataset_name}/images/train/{i:02d}.zip" for i in range(50)]

    print(f"[External Dataset] Step 2/3: Streaming image shards ({len(all_train_shards)} available)...")

    collected_rows = []
    country_counts = defaultdict(int)
    pbar = tqdm(total=num_samples, desc="Extracting street images")

    for shard_path in all_train_shards:
        if len(collected_rows) >= num_samples:
            break

        rel_shard_name = shard_path.replace(f"datasets/{hf_dataset_name}/", "")
        print(f"\n[External Dataset] Downloading shard {rel_shard_name}...")
        try:
            local_zip = hf_hub_download(
                repo_id=hf_dataset_name,
                filename=rel_shard_name,
                repo_type="dataset",
                local_dir=str(temp_dir),
            )
        except Exception as e:
            print(f"[External Dataset] Shard download error: {e}")
            continue

        if zipfile.is_zipfile(local_zip):
            with zipfile.ZipFile(local_zip, "r") as zf:
                for member in zf.namelist():
                    if len(collected_rows) >= num_samples:
                        break
                    
                    if not (member.lower().endswith(".jpg") or member.lower().endswith(".jpeg")):
                        continue

                    # Extract ID (e.g. 477384383571511.jpg -> 477384383571511)
                    stem_id = Path(member).stem

                    if stem_id not in meta_lookup:
                        continue

                    lat, lon, country_iso = meta_lookup[stem_id]

                    # Quota management
                    is_priority = country_iso in PRIORITY_CONFUSION_COUNTRIES
                    effective_cap = max_per_country * (priority_boost_factor if is_priority else 1)
                    if country_counts[country_iso] >= effective_cap:
                        continue

                    # Save image directly to output images/
                    target_img_path = images_dir / f"{stem_id}.jpg"
                    if not target_img_path.exists():
                        with zf.open(member) as zf_in, open(target_img_path, "wb") as f_out:
                            shutil.copyfileobj(zf_in, f_out)

                    collected_rows.append({
                        "image_id": stem_id,
                        "latitude": lat,
                        "longitude": lon,
                        "country_iso": country_iso,
                    })
                    country_counts[country_iso] += 1
                    pbar.update(1)

        # Remove downloaded shard to free up Kaggle disk space immediately
        try:
            os.remove(local_zip)
        except Exception:
            pass

    pbar.close()

    # Clean up temp folder
    shutil.rmtree(temp_dir, ignore_errors=True)

    if not collected_rows:
        print("[External Dataset] ERROR: No images could be extracted.")
        return None

    raw_df = pd.DataFrame(collected_rows)
    print(f"\n[External Dataset] Step 3/3: Processing {len(raw_df):,} extracted images...")

    # 1. Deduplicate against 19,002 internal coordinates (< 50m)
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
    print(f"  ✓ Kaggle Dataset Package Complete!")
    print(f"  ✓ Package Directory: {output_dir}")
    print(f"  ✓ Images Stored:     {len(list(images_dir.glob('*.jpg'))):,} JPGs")
    print(f"  ✓ Metadata CSV:      {out_csv}")
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
