"""
osv5m_loader.py — External Dataset Integration for OpenStreetView-5M (Astruc et al., CVPR 2024)

Rules Compliance:
  - External dataset explicitly disclosed: OpenStreetView-5M (osv5m/osv5m on HuggingFace).
  - Deduplicated against provided 19,002 images via spherical GPS proximity (< 50m threshold).
  - Synthetic noise/JPEG artifact matching applied via MatchedNoiseAugmentation.
  - Multi-task labels (geocell, country, aux climate) derived consistently with training data.
  - Assigned domain_label = 1 (vs domain_label = 0 for internal dataset) to train the GRL domain head.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import (
    BACKBONE_SIZE,
    WATERMARK_X0, WATERMARK_X1, WATERMARK_Y0, WATERMARK_Y1,
    find_dataset_paths,
    KOPPEN_MISSING, WORLDCOVER_MISSING, ELEVATION_MISSING
)
from data.noise_aug import MatchedNoiseAugmentation
from geocells.build_geocells import latlon_to_xyz

DATA_DIR = ROOT / "data"


def deduplicate_against_internal(
    candidate_lats: np.ndarray,
    candidate_lons: np.ndarray,
    internal_coords_csv: Path = None,
    threshold_km: float = 0.05,  # 50 metres
) -> np.ndarray:
    """
    Returns boolean mask of candidates that are >= threshold_km away from all internal images.
    Uses 3D Euclidean distance on unit sphere for fast vectorized check.
    """
    from scipy.spatial import cKDTree

    if internal_coords_csv is None or not Path(internal_coords_csv).exists():
        auto_coords, _ = find_dataset_paths()
        internal_coords_csv = auto_coords

    print(f"[OSV5M] Loading internal ground truth coordinates from: {internal_coords_csv}")
    internal_df = pd.read_csv(internal_coords_csv)
    int_lats = internal_df["latitude"].values
    int_lons = internal_df["longitude"].values

    # Convert to 3D unit sphere coords
    int_xyz = np.stack(latlon_to_xyz(int_lats, int_lons), axis=1)
    cand_xyz = np.stack(latlon_to_xyz(candidate_lats, candidate_lons), axis=1)

    tree = cKDTree(int_xyz)
    # Chord length for threshold_km on Earth of radius 6371 km
    chord_dist = 2.0 * np.sin((threshold_km / 6371.0) / 2.0)

    # Query nearest internal neighbor for each candidate
    dists, _ = tree.query(cand_xyz, k=1)
    is_unique = dists > chord_dist

    n_dupes = (~is_unique).sum()
    print(f"[OSV5M] Deduplication: {len(candidate_lats)} candidates checked -> "
          f"{n_dupes} within {threshold_km*1000:.0f}m dropped, {is_unique.sum()} unique kept.")
    return is_unique


def map_to_nearest_geocell(
    lats: np.ndarray,
    lons: np.ndarray,
    centroids_csv: Path = None,
) -> np.ndarray:
    """Assigns each (lat, lon) to its nearest geocell centroid on the 3D unit sphere."""
    from scipy.spatial import cKDTree

    if centroids_csv is None or not Path(centroids_csv).exists():
        centroids_csv = DATA_DIR / "geocell_centroids.csv"

    centroids_df = pd.read_csv(centroids_csv).sort_values("geocell_id").reset_index(drop=True)
    c_lats = centroids_df["centroid_lat"].values
    c_lons = centroids_df["centroid_lon"].values
    c_xyz = np.stack(latlon_to_xyz(c_lats, c_lons), axis=1)

    pts_xyz = np.stack(latlon_to_xyz(lats, lons), axis=1)
    tree = cKDTree(c_xyz)
    _, nearest_indices = tree.query(pts_xyz, k=1)
    return centroids_df["geocell_id"].values[nearest_indices]


class OSV5MDataset(Dataset):
    """
    PyTorch Dataset for OpenStreetView-5M external samples with matched noise augmentation.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        images_dir: Path,
        augment: bool = True,
    ):
        super().__init__()
        self.df = metadata_df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.augment = augment

        # Noise augmentation matching internal distribution
        self.noise_aug = MatchedNoiseAugmentation(target_noise_std=18.8, p_apply=0.7)

        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std  = [0.26862954, 0.26130258, 0.27577711]

        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(BACKBONE_SIZE, scale=(0.7, 1.0)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(clip_mean, clip_std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(BACKBONE_SIZE),
                transforms.CenterCrop(BACKBONE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(clip_mean, clip_std),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = str(row["image_id"])
        img_path = self.images_dir / f"{img_id}.jpg"

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (640, 640), 0)

        # Apply noise augmentation to match target domain characteristics
        if self.augment:
            img = self.noise_aug(img)

        img_tensor = self.transform(img)

        return {
            "image":           img_tensor,
            "geocell_id":      int(row["geocell_id"]),
            "country_idx":     int(row["country_idx"]),
            "koppen_code":     int(row.get("koppen_code", KOPPEN_MISSING)),
            "worldcover_code": int(row.get("worldcover_code", WORLDCOVER_MISSING)),
            "elevation_m":     float(row.get("elevation_m", ELEVATION_MISSING)),
            "latitude":        float(row["latitude"]),
            "longitude":       float(row["longitude"]),
            "image_id":        f"osv5m_{img_id}",
            "domain_label":    1,  # 1 = external domain (OSV5M)
        }


class CombinedGeoDataset(Dataset):
    """
    Combines internal GeoDataset (domain_label=0) with external OSV5MDataset (domain_label=1).
    Enables multi-task training with domain adversarial loss (GRL).
    """

    def __init__(self, internal_dataset, external_dataset=None):
        self.internal_ds = internal_dataset
        self.external_ds = external_dataset
        self.n_internal = len(internal_dataset)
        self.n_external = len(external_dataset) if external_dataset is not None else 0
        self.total_len = self.n_internal + self.n_external

        # Propagate dataset metadata
        self.n_geocells = getattr(internal_dataset, "n_geocells", 1000)
        self.n_countries = getattr(internal_dataset, "n_countries", 150)
        print(f"[CombinedGeoDataset] Total: {self.total_len} samples "
              f"({self.n_internal} internal [domain 0], {self.n_external} external [domain 1])")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx: int):
        if idx < self.n_internal:
            item = self.internal_ds[idx]
            item["domain_label"] = 0  # 0 = internal domain
            return item
        else:
            return self.external_ds[idx - self.n_internal]
