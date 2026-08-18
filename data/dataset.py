"""
dataset.py — Step 2, PRD Section 4 items 1-6

PyTorch Dataset that:
  - Loads images from training_dataset/noised_dataset/images/
  - Masks the bottom-left watermark region (applied to ALL images)
  - Resizes to backbone input size (224x224 for CLIP ViT-B)
  - Returns: image tensor + geocell_id + country_idx + aux labels + domain label

Usage:
  from data.dataset import GeoDataset, get_dataloaders
  train_loader, val_loader = get_dataloaders(batch_size=32)
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "training_dataset" / "noised_dataset" / "images"
DATA_DIR   = ROOT / "data"

# --- Watermark mask config (confirmed by audit: all images 640x640) ---
WATERMARK_X0, WATERMARK_X1 = 0, 96     # left 15%
WATERMARK_Y0, WATERMARK_Y1 = 544, 640  # bottom 15%

# --- Backbone input size ---
BACKBONE_SIZE = 224  # CLIP ViT-B/32 and SigLIP ViT-B/16

# --- Aux label fill values for missing data ---
KOPPEN_MISSING    = -1   # will be masked in loss
WORLDCOVER_MISSING = -1
ELEVATION_MISSING  = 0.0


def find_dataset_paths():
    """Locates training images dir and ground_truth_coordinates.csv across local and Kaggle environments."""
    # 1. Local workspace default
    local_coords = ROOT / "training_dataset" / "noised_dataset" / "ground_truth_coordinates.csv"
    local_images = ROOT / "training_dataset" / "noised_dataset" / "images"
    if local_coords.exists() and local_images.exists():
        return local_coords, local_images

    # 2. Check Kaggle input directory
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # Search recursively for ground_truth_coordinates.csv
        found_csvs = list(kaggle_input.rglob("ground_truth_coordinates.csv"))
        if found_csvs:
            coords_path = found_csvs[0]
            # images folder is usually next to CSV or under noised_dataset/images
            cand_img_dirs = [
                coords_path.parent / "images",
                coords_path.parent / "noised_dataset" / "images",
            ]
            for cid in cand_img_dirs:
                if cid.exists():
                    return coords_path, cid
            # Search any images folder in kaggle input
            for img_dir in kaggle_input.rglob("images"):
                if img_dir.is_dir() and len(list(img_dir.glob("*.jpg"))) > 10:
                    return coords_path, img_dir

    # Fallback to local default path
    return local_coords, local_images


def find_test_images_path():
    """Locates test images directory across local and Kaggle environments.

    Priority:
      1. /kaggle/input/<competition-slug>/test_images  (real competition test set)
      2. /kaggle/input/**/*test*  (any attached dataset with 'test' in path)
      3. /kaggle/working/**/*test* (working dir copy)
      4. Local workspace test_images_sampled/ (dry-run fallback — LAST RESORT)
    """
    # Resolve sample_id from sample_submission.csv for rglob matching
    sample_sub = ROOT / "sample_submission.csv"
    sample_id = "34f65e00cc3df67d.jpg"
    if sample_sub.exists():
        try:
            df_s = pd.read_csv(sample_sub)
            if not df_s.empty and "image_id" in df_s.columns:
                sample_id = str(df_s["image_id"].iloc[0])
        except Exception:
            pass

    # 1. Known competition dataset path patterns (highest priority)
    known_patterns = [
        Path("/kaggle/input/geolocation-hackathon/test_images"),
        Path("/kaggle/input/geolocation-hackathon/test"),
        Path("/kaggle/input/datasets/harshsolanki07/geolocation-data/test_images"),
        Path("/kaggle/input/datasets/harshsolanki07/geolocation-data/test"),
    ]
    for p in known_patterns:
        if p.exists() and len(list(p.glob("*.jpg"))) > 0:
            print(f"[find_test_images_path] Found real test set at: {p} ({len(list(p.glob('*.jpg')))} images)")
            return p

    # 2. Search /kaggle/input for the sample image ID (catches any dataset layout)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        found = list(kaggle_input.rglob(sample_id))
        if found:
            p = found[0].parent
            print(f"[find_test_images_path] Found real test set via sample_id '{sample_id}': {p}")
            return p
        # Wildcard: any dir with 'test' in name containing .jpg files
        for cand in sorted(kaggle_input.rglob("*test*")):
            if cand.is_dir() and len(list(cand.glob("*.jpg"))) > 0:
                print(f"[find_test_images_path] Found test dir via wildcard in /kaggle/input: {cand}")
                return cand

    # 3. /kaggle/working fallback
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        found = list(kaggle_working.rglob(sample_id))
        if found:
            p = found[0].parent
            print(f"[find_test_images_path] Found test set in /kaggle/working: {p}")
            return p

    # 4. LOCAL FALLBACK — dry-run sampled set (NOT the real competition test set)
    local_test = ROOT / "test_images_sampled"
    print(
        f"[find_test_images_path] WARNING: Real competition test images NOT found. "
        f"Falling back to local dry-run sample: {local_test}. "
        f"Attach the competition dataset on Kaggle to get real test predictions."
    )
    return local_test



class GeoDataset(Dataset):
    """
    Geolocation dataset with multi-task labels.

    Args:
        split: 'train' or 'val'
        val_frac: fraction of data held out for validation (stratified by geocell)
        seed: random seed for reproducible splits
        augment: whether to apply training augmentations
        max_samples: if set, cap the dataset (for dry runs)
    """

    def __init__(
        self,
        split: str = "train",
        val_frac: float = 0.1,
        seed: int = 42,
        augment: bool = True,
        max_samples: int = None,
        images_dir: Path = None,
        coords_csv: Path = None,
        data_dir: Path = DATA_DIR,
    ):
        super().__init__()
        self.split = split
        self.augment = augment and (split == "train")

        auto_coords, auto_images = find_dataset_paths()
        self.images_dir = Path(images_dir) if images_dir is not None else auto_images
        coords_path = Path(coords_csv) if coords_csv is not None else auto_coords

        print(f"[GeoDataset] Using coords: {coords_path}")
        print(f"[GeoDataset] Using images: {self.images_dir}")

        if not coords_path.exists():
            raise FileNotFoundError(
                f"Ground truth coordinates CSV not found at '{coords_path}'. "
                "If running on Kaggle, please ensure the competition dataset is attached under /kaggle/input/."
            )

        # --- Load and merge all label tables ---
        coords     = pd.read_csv(coords_path)
        geocells   = pd.read_csv(data_dir / "geocell_assignments.csv")[["image_id", "geocell_id"]]
        countries  = pd.read_csv(data_dir / "country_labels.csv")[["image_id", "country_iso"]]
        encoder    = pd.read_csv(data_dir / "country_encoder.csv")


        # Merge
        df = coords.merge(geocells, on="image_id").merge(countries, on="image_id")

        # Encode country ISO -> integer
        iso2idx = dict(zip(encoder["country_iso"], encoder["country_idx"]))
        df["country_idx"] = df["country_iso"].map(iso2idx).fillna(0).astype(int)

        # Aux labels (optional — fill with missing if not generated yet)
        aux_path = data_dir / "aux_labels.csv"
        if aux_path.exists():
            aux = pd.read_csv(aux_path)[["image_id", "koppen_code", "worldcover_code", "elevation_m"]]
            df = df.merge(aux, on="image_id", how="left")
        else:
            df["koppen_code"]    = KOPPEN_MISSING
            df["worldcover_code"] = WORLDCOVER_MISSING
            df["elevation_m"]    = ELEVATION_MISSING

        df["koppen_code"]    = df["koppen_code"].fillna(KOPPEN_MISSING).astype(int)
        df["worldcover_code"] = df["worldcover_code"].fillna(WORLDCOVER_MISSING).astype(int)
        df["elevation_m"]    = df["elevation_m"].fillna(ELEVATION_MISSING).astype(float)

        # --- Stratified train/val split by geocell (matching original 1000-cluster run) ---
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        val_idx = (
            df.groupby("geocell_id", group_keys=False)
              .apply(lambda g: g.head(max(1, int(len(g) * val_frac))))
              .index
        )
        val_mask = df.index.isin(val_idx)
        df["_split"] = "train"
        df.loc[val_mask, "_split"] = "val"



        self.df = df[df["_split"] == split].reset_index(drop=True)

        if max_samples is not None:
            self.df = self.df.head(max_samples).reset_index(drop=True)

        self.n_geocells  = int(df["geocell_id"].max()) + 1
        self.n_countries = len(encoder)

        # --- Transforms ---
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std  = [0.26862954, 0.26130258, 0.27577711]

        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(BACKBONE_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.0),   # NO flips — invert driving side
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

        print(f"[GeoDataset] {split}: {len(self.df)} images | "
              f"{self.n_geocells} geocells | {self.n_countries} countries")

    def __len__(self):
        return len(self.df)

    def _apply_watermark_mask(self, img: Image.Image) -> Image.Image:
        """Zero-fill the bottom-left watermark region (on all images)."""
        arr = np.array(img)
        arr[WATERMARK_Y0:WATERMARK_Y1, WATERMARK_X0:WATERMARK_X1] = 0
        return Image.fromarray(arr)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.images_dir / f"{row['image_id']}.jpg"

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (640, 640), 0)

        # Mask watermark (applied to all images)
        img = self._apply_watermark_mask(img)

        # Augment / resize
        img_tensor = self.transform(img)

        # Domain label: 1 = street view, 0 = dashcam/other
        # We don't have per-image domain labels from audit, so use brightness proxy
        # (will be refined once full audit is run — for now set to -1 = unknown)
        domain_label = -1

        return {
            "image":         img_tensor,                        # [3, 224, 224]
            "geocell_id":    int(row["geocell_id"]),
            "country_idx":   int(row["country_idx"]),
            "koppen_code":   int(row["koppen_code"]),
            "worldcover_code": int(row["worldcover_code"]),
            "elevation_m":   float(row["elevation_m"]),
            "latitude":      float(row["latitude"]),
            "longitude":     float(row["longitude"]),
            "image_id":      str(row["image_id"]),
            "domain_label":  domain_label,
        }


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 2,
    val_frac: float = 0.1,
    seed: int = 42,
    max_train_samples: int = None,
    max_val_samples: int = None,
    images_dir: Path = None,
    coords_csv: Path = None,
    use_osv5m: bool = False,
    osv5m_meta_csv: Path = None,
    osv5m_images_dir: Path = None,
):
    """
    Returns (train_loader, val_loader, dataset_meta).
    dataset_meta: dict with n_geocells, n_countries.
    """
    train_ds = GeoDataset("train", val_frac=val_frac, seed=seed, augment=True,
                          max_samples=max_train_samples, images_dir=images_dir, coords_csv=coords_csv)
    val_ds   = GeoDataset("val",   val_frac=val_frac, seed=seed, augment=False,
                          max_samples=max_val_samples, images_dir=images_dir, coords_csv=coords_csv)

    if use_osv5m:
        from data.osv5m_loader import OSV5MDataset, CombinedGeoDataset
        meta_path = osv5m_meta_csv or (DATA_DIR / "osv5m_train.csv")
        img_dir = osv5m_images_dir or (DATA_DIR / "osv5m_images")
        if Path(meta_path).exists() and Path(img_dir).exists():
            osv_meta = pd.read_csv(meta_path)
            osv_ds = OSV5MDataset(osv_meta, img_dir, augment=True)
            train_ds = CombinedGeoDataset(train_ds, osv_ds)
            print(f"[get_dataloaders] Integrated OSV5M external dataset ({len(osv_ds)} images)")
        else:
            print(f"[get_dataloaders] WARNING: OSV5M data not found at {meta_path} or {img_dir}. Proceeding with internal data only.")

    g = torch.Generator()
    g.manual_seed(seed)

    _pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=_pin,
        worker_init_fn=lambda wid: random.seed(seed + wid),
        generator=g, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin,
    )

    meta = {
        "n_geocells":  getattr(train_ds, "n_geocells", 1000),
        "n_countries": getattr(train_ds, "n_countries", 150),
    }
    return train_loader, val_loader, meta




class TestDataset(Dataset):
    """Dataset for inference on the test set (no labels)."""

    def __init__(self, test_dir: Path = None):
        if test_dir is None:
            test_dir = find_test_images_path()
        self.test_dir = Path(test_dir)
        self.paths = sorted(self.test_dir.glob("*.jpg"))
        print(f"[TestDataset] Found {len(self.paths)} test images in {self.test_dir}")
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std  = [0.26862954, 0.26130258, 0.27577711]
        self.transform = transforms.Compose([
            transforms.Resize(BACKBONE_SIZE),
            transforms.CenterCrop(BACKBONE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(clip_mean, clip_std),
        ])

    def __len__(self):
        return len(self.paths)

    def _apply_watermark_mask(self, img):
        arr = np.array(img)
        arr[WATERMARK_Y0:WATERMARK_Y1, WATERMARK_X0:WATERMARK_X1] = 0
        return Image.fromarray(arr)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (640, 640), 0)
        img = self._apply_watermark_mask(img)
        return {
            "image":    self.transform(img),
            "image_id": path.name,
        }

