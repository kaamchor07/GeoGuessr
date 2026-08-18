"""
country_labels.py — Step 1, Section 4 item 3

For every (lat, lon) in the training CSV, run a point-in-polygon test against
country_boundaries.geojson (using shapely + rtree spatial index for speed).
Outputs data/country_labels.csv with columns: image_id, country_iso, country_name.

Images that fall in no polygon (ocean / tiny gap) get country_iso = "OCEAN".

Usage:
  python labels/country_labels.py
  python labels/country_labels.py --coords_csv training_dataset/noised_dataset/ground_truth_coordinates.csv
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

COORDS_CSV = ROOT / "training_dataset" / "noised_dataset" / "ground_truth_coordinates.csv"
GEOJSON_PATH = ROOT / "country_boundaries.geojson"


def load_geojson(path: Path):
    """Load GeoJSON and return list of (iso, name, shapely_geometry) tuples."""
    from shapely.geometry import shape

    print(f"Loading GeoJSON from {path} …")
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    features = []
    for feat in gj["features"]:
        props = feat.get("properties", {})
        iso = props.get("ISO_A2") or props.get("iso_a2") or props.get("ISO") or props.get("ADM0_A3") or "UNK"
        name = props.get("ADMIN") or props.get("name") or props.get("NAME") or iso
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        features.append((iso.upper(), name, geom))

    print(f"  Loaded {len(features)} country polygons")
    return features


def build_spatial_index(features):
    """Build an R-tree spatial index over the country bounding boxes."""
    try:
        from rtree import index as rtree_index
        idx = rtree_index.Index()
        for i, (iso, name, geom) in enumerate(features):
            idx.insert(i, geom.bounds)
        print("  R-tree spatial index built (fast path)")
        return idx
    except ImportError:
        print("  [Warning] rtree not installed — falling back to brute-force (slow for large data)")
        print("  Install with: pip install rtree")
        return None


def point_in_country(lon, lat, features, rtree_idx):
    """Return (iso, name) for the country containing (lon, lat), or ('OCEAN', 'Ocean')."""
    from shapely.geometry import Point
    pt = Point(lon, lat)

    if rtree_idx is not None:
        candidates = list(rtree_idx.intersection((lon, lat, lon, lat)))
    else:
        candidates = range(len(features))

    for i in candidates:
        iso, name, geom = features[i]
        if geom.contains(pt):
            return iso, name

    # Fallback: nearest country by distance (handles coastal rounding)
    if rtree_idx is not None:
        # expand search box by ±0.5 degrees
        candidates = list(rtree_idx.intersection((lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5)))
        for i in candidates:
            iso, name, geom = features[i]
            if geom.distance(pt) < 0.1:  # within ~10 km
                return iso, name

    return "OCEAN", "Ocean"


def label_countries_from_geojson(
    coords_csv: Path = None,
    geojson_path: Path = None,
    out_csv: Path = None,
    batch_size: int = 500,
):
    if coords_csv is None:
        coords_csv = COORDS_CSV
    if geojson_path is None:
        geojson_path = GEOJSON_PATH
    if out_csv is None:
        out_csv = DATA_DIR / "country_labels.csv"

    print(f"Loading coordinates from {coords_csv}")
    df = pd.read_csv(coords_csv)
    print(f"  {len(df)} images to label")

    features = load_geojson(Path(geojson_path))
    rtree_idx = build_spatial_index(features)

    # Label each point
    print("Running point-in-polygon tests …")
    isos, names = [], []
    for i, row in enumerate(df.itertuples(index=False)):
        iso, name = point_in_country(row.longitude, row.latitude, features, rtree_idx)
        isos.append(iso)
        names.append(name)
        if (i + 1) % batch_size == 0 or (i + 1) == len(df):
            print(f"  {i+1}/{len(df)} done", end="\r")

    print()

    # Build output
    out_df = pd.DataFrame(
        {
            "image_id": df["image_id"].values,
            "latitude": df["latitude"].values,
            "longitude": df["longitude"].values,
            "country_iso": isos,
            "country_name": names,
        }
    )

    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")

    # Summary
    country_counts = out_df["country_iso"].value_counts()
    print(f"\n=== Country distribution ({len(country_counts)} unique) ===")
    print(country_counts.head(20).to_string())
    ocean = (out_df["country_iso"] == "OCEAN").sum()
    print(f"\nOcean/unmatched: {ocean} ({100*ocean/len(df):.1f}%)")

    # Save encoder mapping (iso -> integer index) for model training
    iso_list = sorted(out_df["country_iso"].unique())
    encoder = pd.DataFrame({"country_iso": iso_list, "country_idx": range(len(iso_list))})
    enc_path = DATA_DIR / "country_encoder.csv"
    encoder.to_csv(enc_path, index=False)
    print(f"Country encoder saved -> {enc_path}  ({len(iso_list)} classes)")
    return out_df, encoder


def main():
    parser = argparse.ArgumentParser(description="Generate country labels for training images")
    parser.add_argument("--coords_csv", type=str, default=str(COORDS_CSV))
    parser.add_argument("--geojson", type=str, default=str(GEOJSON_PATH))
    parser.add_argument(
        "--out",
        type=str,
        default=str(DATA_DIR / "country_labels.csv"),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Progress reporting interval",
    )
    args = parser.parse_args()

    label_countries_from_geojson(
        coords_csv=args.coords_csv,
        geojson_path=args.geojson,
        out_csv=args.out,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

