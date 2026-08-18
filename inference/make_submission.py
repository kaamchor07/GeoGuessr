"""
make_submission.py — Step 7, PRD Section 4 items 9-11 & Section 11

Stage-1-only inference pipeline (raw argmax geocell centroid).
Soft blending and kNN refinement are intentionally disabled — both
degraded haversine median on the held-out val split. Radius is
calibrated by the grid-searched alpha/min_r params from calibration_params.json,
NOT a hardcoded confidence-bucket heuristic.

Usage:
  python inference/make_submission.py \
    --checkpoint checkpoints/k1000_original/best.pt \
    --out_csv submissions/submission_stage1.csv

Flags:
  --use_tta        Enable 5-crop TTA (default: on, pass --no_tta to disable)
  --use_snap       Enable country-boundary snapping (default: on)
  --calib_json     Path to calibration_params.json (default: data/calibration_params.json)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.model import GeoLocModel
from data.dataset import (
    TestDataset, BACKBONE_SIZE, find_test_images_path
)

DATA_DIR      = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)


class TTATestDataset(TestDataset):
    """
    Test dataset returning 5 multi-scale crops for Test-Time Augmentation (TTA).
    No horizontal flips — preserves driving-side and text cues.
    """

    def __init__(self, test_dir: Path = None):
        super().__init__(test_dir)
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std  = [0.26862954, 0.26130258, 0.27577711]
        self.norm = transforms.Normalize(clip_mean, clip_std)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (640, 640), 0)
        img = self._apply_watermark_mask(img)

        # 5 crops: Center + 4 corners (scaled to 256 first)
        crops = [
            transforms.functional.center_crop(transforms.functional.resize(img, 256), BACKBONE_SIZE),
            transforms.functional.crop(transforms.functional.resize(img, 256), 0, 0, BACKBONE_SIZE, BACKBONE_SIZE),
            transforms.functional.crop(transforms.functional.resize(img, 256), 0, 256 - BACKBONE_SIZE, BACKBONE_SIZE, BACKBONE_SIZE),
            transforms.functional.crop(transforms.functional.resize(img, 256), 256 - BACKBONE_SIZE, 0, BACKBONE_SIZE, BACKBONE_SIZE),
            transforms.functional.crop(transforms.functional.resize(img, 256), 256 - BACKBONE_SIZE, 256 - BACKBONE_SIZE, BACKBONE_SIZE, BACKBONE_SIZE),
        ]
        tensors = torch.stack([self.norm(transforms.functional.to_tensor(c)) for c in crops])
        return {"images": tensors, "image_id": path.name}


def generate_submission(
    checkpoint_path: str = None,
    test_dir: str = None,
    sample_sub_path: str = str(ROOT / "sample_submission.csv"),
    output_csv_path: str = None,
    use_tta: bool = True,
    use_snap: bool = True,
    calib_json: str = str(DATA_DIR / "calibration_params.json"),
    batch_size: int = 16,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    print(f"[Inference] Pipeline: Stage 1 ONLY (raw argmax centroid) — soft blend & kNN disabled")
    print(f"[Inference] Running on {device}")

    if test_dir is None:
        test_dir = find_test_images_path()
    test_dir = Path(test_dir)
    print(f"[Inference] Test images: {test_dir}")

    # Load geocell and country lookups
    centroids_df = pd.read_csv(DATA_DIR / "geocell_centroids.csv").sort_values("geocell_id").reset_index(drop=True)
    encoder_df   = pd.read_csv(DATA_DIR / "country_encoder.csv")
    idx2country  = dict(zip(encoder_df["country_idx"], encoder_df["country_iso"]))

    n_geocells  = len(centroids_df)
    n_countries = len(encoder_df)

    # Build model
    model = GeoLocModel(
        n_geocells=n_geocells,
        n_countries=n_countries,
        clip_model_name="ViT-B-32",
        clip_pretrained="openai",
    )
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[Inference] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt))
    else:
        print("[Inference] WARNING: No checkpoint — using base model weights")

    model = model.to(device)
    model.eval()

    # Load test data
    if use_tta:
        print("[Inference] TTA: 5 crops per image")
        ds = TTATestDataset(Path(test_dir))
    else:
        print("[Inference] TTA: disabled (single center crop)")
        ds = TestDataset(Path(test_dir))

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_image_ids    = []
    all_pred_geocells = []
    all_pred_countries = []
    all_country_confs  = []

    print("[Inference] Running forward pass...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if use_tta:
                b, n_crops, c, h, w = batch["images"].shape
                flat_imgs = batch["images"].view(-1, c, h, w).to(device)
                out = model(flat_imgs)
                # Average logits across crops
                gc_logits = out["geocell_logits"].view(b, n_crops, -1).mean(dim=1)
                ct_logits = out["country_logits"].view(b, n_crops, -1).mean(dim=1)
            else:
                imgs = batch["image"].to(device)
                out  = model(imgs)
                gc_logits = out["geocell_logits"]
                ct_logits = out["country_logits"]

            # Stage 1: raw argmax geocell — no blending, no kNN
            pred_gc = gc_logits.argmax(dim=-1).cpu().numpy()
            ct_probs = F.softmax(ct_logits, dim=-1)
            pred_ct_conf, pred_ct = ct_probs.max(dim=-1)

            all_image_ids.extend(batch["image_id"])
            all_pred_geocells.extend(pred_gc)
            all_pred_countries.extend([idx2country.get(int(i), "UNK") for i in pred_ct.cpu().numpy()])
            all_country_confs.extend(pred_ct_conf.cpu().numpy())

    all_pred_geocells  = np.array(all_pred_geocells)
    all_country_confs  = np.array(all_country_confs)

    # Stage 1 coordinates: raw argmax centroid lookup
    pred_lats  = centroids_df["centroid_lat"].values[all_pred_geocells].copy()
    pred_lons  = centroids_df["centroid_lon"].values[all_pred_geocells].copy()
    base_radii = centroids_df["max_radius_km"].values[all_pred_geocells].copy()

    print(f"[Inference] Stage 1 complete. {len(pred_lats)} predictions generated.")

    # Optional: country-boundary snapping (keeps coords, doesn't change lat/lon much)
    if use_snap and (ROOT / "country_boundaries.geojson").exists():
        print("[Inference] Applying country boundary snapping...")
        try:
            from calibration.country_snap import CountrySnapper
            snapper = CountrySnapper()
            pred_lats, pred_lons = snapper.snap_coordinates(
                lats=pred_lats,
                lons=pred_lons,
                country_isos=all_pred_countries,
                country_confidences=all_country_confs,
                min_confidence=0.45,
            )
        except Exception as e:
            print(f"[Inference] Country snapping skipped: {e}")

    # Radius: load calibration params and apply per-image confidence-driven curve
    # Formula: r(conf) = A + B * sqrt(1 - conf)
    calib_path = Path(calib_json)
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)
        A = calib.get("A", calib.get("min_radius_km", 1500.0))
        B = calib.get("B", 0.0)
        formula = calib.get("formula", "A + B * sqrt(1 - conf)")
        print(f"[Inference] Radius formula: {formula}")
        print(f"[Inference] A={A:.0f} km, B={B:.0f} km")
        print(f"[Inference] r(conf=1.0)={A:.0f} km | r(conf=0.5)={A + B*(0.5**0.5):.0f} km | r(conf=0.0)={A+B:.0f} km")
        sqrt_uncertainty = np.sqrt(np.clip(1.0 - all_country_confs, 0.0, 1.0))
        pred_radii = np.clip(A + B * sqrt_uncertainty, 50.0, 3000.0)
    else:
        print(f"[Inference] WARNING: calibration_params.json not found at {calib_path}")
        print(f"[Inference] Run: python calibration/calibrate_radius.py --checkpoint <ckpt>")
        print(f"[Inference] Using fallback: r = 1500 + 700*sqrt(1-conf)")
        sqrt_uncertainty = np.sqrt(np.clip(1.0 - all_country_confs, 0.0, 1.0))
        pred_radii = np.clip(1500.0 + 700.0 * sqrt_uncertainty, 50.0, 3000.0)

    print(f"[Inference] Radii: min={pred_radii.min():.1f} km | median={np.median(pred_radii):.1f} km | max={pred_radii.max():.1f} km")

    # Build output DataFrame
    pred_df = pd.DataFrame({
        "image_id":       all_image_ids,
        "pred_lat":       np.round(pred_lats, 6),
        "pred_lon":       np.round(pred_lons, 6),
        "pred_radius_km": np.round(pred_radii, 2),
    })

    # Align to sample_submission.csv row order (fills missing with 0/1000 defaults)
    sample_sub_path = Path(sample_sub_path)
    if sample_sub_path.exists():
        sample_sub = pd.read_csv(sample_sub_path)
        pred_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
        pred_df["pred_lat"]       = pred_df["pred_lat"].fillna(0.0)
        pred_df["pred_lon"]       = pred_df["pred_lon"].fillna(0.0)
        pred_df["pred_radius_km"] = pred_df["pred_radius_km"].fillna(2000.0)
        missing = sample_sub[~sample_sub["image_id"].isin(all_image_ids)]
        if len(missing):
            print(f"[Inference] WARNING: {len(missing)} test images had no prediction — filled with defaults")
    else:
        print(f"[Inference] WARNING: sample_submission.csv not found at {sample_sub_path} — using raw row order")

    # Write CSV
    if output_csv_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_path = str(SUBMISSIONS_DIR / f"submission_stage1_{timestamp}.csv")

    pred_df.to_csv(output_csv_path, index=False)

    # Schema validation
    print(f"\n[Inference] Saved -> {output_csv_path}  ({len(pred_df)} rows)")
    assert list(pred_df.columns) == ["image_id", "pred_lat", "pred_lon", "pred_radius_km"], \
        f"Schema mismatch: columns are {list(pred_df.columns)}"
    assert pred_df["pred_lat"].between(-90, 90).all(),  "pred_lat out of range [-90, 90]"
    assert pred_df["pred_lon"].between(-180, 180).all(), "pred_lon out of range [-180, 180]"
    assert (pred_df["pred_radius_km"] > 0).all(),        "pred_radius_km must be > 0"
    print("[Inference] Schema validation PASSED: columns, lat/lon ranges, radius > 0")
    print()
    print(pred_df.head(10).to_string(index=False))

    return output_csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, default=None)
    parser.add_argument("--test_dir",    type=str, default=None)
    parser.add_argument("--out_csv",     type=str, default=None)
    parser.add_argument("--no_tta",      action="store_true")
    parser.add_argument("--no_snap",     action="store_true")
    parser.add_argument("--calib_json",  type=str, default=str(DATA_DIR / "calibration_params.json"))
    parser.add_argument("--batch_size",  type=int, default=16)
    args = parser.parse_args()

    generate_submission(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        output_csv_path=args.out_csv,
        use_tta=not args.no_tta,
        use_snap=not args.no_snap,
        calib_json=args.calib_json,
        batch_size=args.batch_size,
    )
