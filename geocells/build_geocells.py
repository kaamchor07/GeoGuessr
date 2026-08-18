"""
build_geocells.py — Step 1, Section 4 item 2

Cluster training coordinates into geocells using KMeans (fast, default)
or OPTICS (density-aware, no ocean-filler cells).

Outputs:
  data/geocell_assignments.csv   — original CSV + geocell_id column
  data/geocell_centroids.csv     — geocell_id, centroid_lat, centroid_lon, count

Usage:
  python geocells/build_geocells.py --method kmeans --n_clusters 1000
  python geocells/build_geocells.py --method optics --min_samples 5
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

COORDS_CSV = ROOT / "training_dataset" / "noised_dataset" / "ground_truth_coordinates.csv"


# ---------------------------------------------------------------------------
# Haversine distance (vectorised, km)
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------
# Cluster on 3-D unit-sphere coordinates (avoids lon wrap-around issues)
# ---------------------------------------------------------------------------
def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def xyz_to_latlon(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def spherical_weighted_average(lats_deg: np.ndarray, lons_deg: np.ndarray, weights: np.ndarray = None) -> tuple[float, float]:
    """
    Computes the true spherical weighted mean on the 3D unit sphere.
    Avoids antimeridian wrapping and high-latitude distortions.
    """
    lats_deg = np.asarray(lats_deg, dtype=np.float64)
    lons_deg = np.asarray(lons_deg, dtype=np.float64)

    if weights is None:
        weights = np.ones_like(lats_deg, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    weights = weights / (np.sum(weights) + 1e-12)

    xyz = latlon_to_xyz(lats_deg, lons_deg)  # [N, 3]
    avg_xyz = np.sum(xyz * weights[:, None], axis=0)  # [3]
    norm = np.linalg.norm(avg_xyz)
    if norm < 1e-6:
        # Opposite poles cancel out -> fallback to highest weight point
        best_idx = np.argmax(weights)
        return float(lats_deg[best_idx]), float(lons_deg[best_idx])

    avg_xyz /= norm
    lat_out, lon_out = xyz_to_latlon(avg_xyz[None, :])
    return float(lat_out[0]), float(lon_out[0])



# ---------------------------------------------------------------------------
# KMeans clustering
# ---------------------------------------------------------------------------
def cluster_kmeans(coords_xyz, n_clusters: int, min_cluster_size: int = 3, seed: int = 42):

    from sklearn.cluster import MiniBatchKMeans

    print(f"[KMeans] Fitting {n_clusters} clusters on {len(coords_xyz)} points …")
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=10_000,
        n_init=5,
        max_iter=300,
        verbose=0,
    )
    labels = km.fit_predict(coords_xyz)
    centroids_xyz = km.cluster_centers_
    centroids_xyz /= np.linalg.norm(centroids_xyz, axis=1, keepdims=True)  # re-project to sphere

    # Merge tiny clusters (< min_cluster_size) into nearest neighbor centroid
    counts = pd.Series(labels).value_counts()
    tiny_clusters = counts[counts < min_cluster_size].index.tolist()
    if tiny_clusters:
        print(f"[KMeans] Merging {len(tiny_clusters)} small clusters (<{min_cluster_size} points) into nearest neighbor...")
        valid_clusters = [c for c in range(n_clusters) if c not in tiny_clusters]
        valid_centroids = centroids_xyz[valid_clusters]

        for tc in tiny_clusters:
            mask = labels == tc
            tc_pts = coords_xyz[mask]
            # Find nearest valid centroid
            dists = np.linalg.norm(tc_pts[:, None] - valid_centroids[None, :], axis=2)
            nearest_valid_idx = np.argmin(dists, axis=1)
            labels[mask] = [valid_clusters[idx] for idx in nearest_valid_idx]

        # Re-index clusters to contiguous 0..N-1
        unique_labels = sorted(set(labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        # Recompute centroids
        centroids_xyz = np.array([coords_xyz[labels == l].mean(axis=0) for l in range(len(unique_labels))])
        centroids_xyz /= np.linalg.norm(centroids_xyz, axis=1, keepdims=True)
        print(f"[KMeans] Final cluster count after merging: {len(unique_labels)}")

    return labels, centroids_xyz



# ---------------------------------------------------------------------------
# OPTICS clustering
# ---------------------------------------------------------------------------
def cluster_optics(coords_xyz, min_samples: int, xi: float = 0.05, seed: int = 42):
    from sklearn.cluster import OPTICS

    print(f"[OPTICS] Fitting with min_samples={min_samples} on {len(coords_xyz)} points …")
    print("  (this may take a few minutes for >10k points — consider sampling first)")
    op = OPTICS(
        min_samples=min_samples,
        xi=xi,
        metric="euclidean",
        n_jobs=-1,
    )
    labels = op.fit_predict(coords_xyz)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    # Assign noise points to the nearest cluster centroid
    unique = [l for l in np.unique(labels) if l != -1]
    centroids_xyz = np.array([coords_xyz[labels == l].mean(axis=0) for l in unique])
    centroids_xyz /= np.linalg.norm(centroids_xyz, axis=1, keepdims=True)

    if n_noise > 0:
        noise_mask = labels == -1
        noise_xyz = coords_xyz[noise_mask]
        # brute-force nearest centroid for noise
        dists = np.linalg.norm(noise_xyz[:, None] - centroids_xyz[None, :], axis=2)
        nearest = unique[np.argmin(dists, axis=1)]  # type: ignore[index]
        labels[noise_mask] = nearest

    # Relabel to contiguous 0..N-1
    label_map = {old: new for new, old in enumerate(sorted(unique))}
    labels = np.array([label_map[l] for l in labels])

    return labels, centroids_xyz


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Main geocell construction entry point
# ---------------------------------------------------------------------------
def build_geocells(
    coords_csv: Path = None,
    n_clusters: int = 1000,
    method: str = "kmeans",
    min_cluster_size: int = 3,
    min_samples: int = 10,
    xi: float = 0.05,
    seed: int = 42,
    out_assignments: Path = None,
    out_centroids: Path = None,
):
    if coords_csv is None:
        coords_csv = COORDS_CSV
    if out_assignments is None:
        out_assignments = DATA_DIR / "geocell_assignments.csv"
    if out_centroids is None:
        out_centroids = DATA_DIR / "geocell_centroids.csv"

    np.random.seed(seed)

    print(f"Loading coordinates from {coords_csv}")
    df = pd.read_csv(coords_csv)
    assert "image_id" in df.columns and "latitude" in df.columns and "longitude" in df.columns, (
        "CSV must have columns: image_id, latitude, longitude"
    )
    print(f"  {len(df)} rows loaded")

    # Encode to unit-sphere XYZ (handles lon wrap-around cleanly)
    coords_xyz = latlon_to_xyz(df["latitude"].values, df["longitude"].values)

    # Cluster
    if method == "kmeans":
        labels, centroids_xyz = cluster_kmeans(
            coords_xyz,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
            seed=seed,
        )
    else:
        labels, centroids_xyz = cluster_optics(coords_xyz, min_samples, xi, seed)

    n_cells = len(np.unique(labels))
    print(f"Total geocells: {n_cells}")

    # Save assignments
    df_out = df.copy()
    df_out["geocell_id"] = labels
    df_out.to_csv(out_assignments, index=False)
    print(f"Saved assignments -> {out_assignments}")

    # Compute and save centroids
    centroid_lats, centroid_lons = xyz_to_latlon(centroids_xyz)
    counts = df_out.groupby("geocell_id").size().reset_index(name="count")
    centroids_df = pd.DataFrame(
        {
            "geocell_id": np.arange(n_cells),
            "centroid_lat": centroid_lats,
            "centroid_lon": centroid_lons,
        }
    ).merge(counts, on="geocell_id", how="left")
    centroids_df["count"] = centroids_df["count"].fillna(0).astype(int)

    # Compute median haversine radius of each cell (p90 error proxy)
    radii = []
    for gid, grp in df_out.groupby("geocell_id"):
        c_lat = centroid_lats[gid]
        c_lon = centroid_lons[gid]
        dist = haversine_km(grp["latitude"].values, grp["longitude"].values, c_lat, c_lon)
        radii.append(dist.max())
    centroids_df["max_radius_km"] = radii
    centroids_df.to_csv(out_centroids, index=False)
    print(f"Saved centroids  -> {out_centroids}")

    print(f"\nMedian cell max-radius: {np.median(radii):.1f} km")
    print(f"P90   cell max-radius: {np.percentile(radii, 90):.1f} km")
    print(f"Max   cell max-radius: {np.max(radii):.1f} km")
    return df_out, centroids_df


def main():
    parser = argparse.ArgumentParser(description="Cluster training coords into geocells")
    parser.add_argument("--coords_csv", type=str, default=str(COORDS_CSV))
    parser.add_argument("--method", choices=["kmeans", "optics"], default="kmeans")
    parser.add_argument("--n_clusters", type=int, default=1000, help="[kmeans] number of clusters")
    parser.add_argument(
        "--min_samples",
        type=int,
        default=10,
        help="[optics] min_samples parameter",
    )
    parser.add_argument("--min_cluster_size", type=int, default=3, help="[kmeans] merge clusters with fewer points")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_assignments",
        type=str,
        default=str(DATA_DIR / "geocell_assignments.csv"),
    )
    parser.add_argument(
        "--out_centroids",
        type=str,
        default=str(DATA_DIR / "geocell_centroids.csv"),
    )
    args = parser.parse_args()

    build_geocells(
        coords_csv=args.coords_csv,
        n_clusters=args.n_clusters,
        method=args.method,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        seed=args.seed,
        out_assignments=args.out_assignments,
        out_centroids=args.out_centroids,
    )


if __name__ == "__main__":
    main()

