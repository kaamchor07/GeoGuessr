"""
make_submission.py — Step 7, PRD Section 4 items 9-11 & Section 11

End-to-end inference pipeline producing competitive, schema-validated submission CSVs:
  1. Loads test images with universal watermark masking
  2. Applies Test-Time Augmentation (TTA: 5-crop / multi-scale, no flips)
  3. Predicts geocell logits, country logits, and extracted embeddings
  4. Applies kNN FAISS embedding refinement (if index exists)
  5. Applies calibrated radius scaling
  6. Applies country-snap boundary post-processing
  7. Formats & validates schema against sample_submission.csv

Usage:
  python inference/make_submission.py --checkpoint checkpoints/best.pt --out_csv submissions/submission_v1.csv --use_tta --use_knn --use_snap
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
from data.dataset import TestDataset, BACKBONE_SIZE, WATERMARK_X0, WATERMARK_X1, WATERMARK_Y0, WATERMARK_Y1

DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)


class TTATestDataset(TestDataset):
    """
    Test dataset returning 5 multi-scale crops for Test-Time Augmentation (TTA).
    No horizontal flips to preserve driving-side and text cues.
    """

    def __init__(self, test_dir: Path = ROOT / "test_images_sampled"):
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

        w, h = img.size
        # 5 crops: Center, Top-Left, Top-Right, Bottom-Left, Bottom-Right (scaled)
        crops = [
            transforms.functional.center_crop(transforms.functional.resize(img, 256), BACKBONE_SIZE),
            transforms.functional.crop(transforms.functional.resize(img, 256), 0, 0, BACKBONE_SIZE, BACKBONE_SIZE),
            transforms.functional.crop(transforms.functional.resize(img, 256), 0, 256 - BACKBONE_SIZE, BACKBONE_SIZE, BACKBONE_SIZE),
            transforms.functional.crop(transforms.functional.resize(img, 256), 256 - BACKBONE_SIZE, 0, BACKBONE_SIZE, BACKBONE_SIZE),
            transforms.functional.crop(transforms.functional.resize(img, 256), 256 - BACKBONE_SIZE, 256 - BACKBONE_SIZE, BACKBONE_SIZE, BACKBONE_SIZE),
        ]

        tensors = torch.stack([self.norm(transforms.functional.to_tensor(c)) for c in crops])  # [5, 3, 224, 224]

        return {
            "images": tensors,
            "image_id": path.name,
        }


def generate_submission(
    checkpoint_path: str = None,
    test_dir: str = str(ROOT / "test_images_sampled"),
    sample_sub_path: str = str(ROOT / "sample_submission.csv"),
    output_csv_path: str = None,
    use_tta: bool = True,
    use_knn: bool = True,
    use_snap: bool = True,
    batch_size: int = 16,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    print(f"[Inference] Running on {device}")

    # 1. Load Geocell and Country lookups
    centroids_df = pd.read_csv(DATA_DIR / "geocell_centroids.csv").sort_values("geocell_id").reset_index(drop=True)
    encoder_df = pd.read_csv(DATA_DIR / "country_encoder.csv")
    idx2country = dict(zip(encoder_df["country_idx"], encoder_df["country_iso"]))

    n_geocells = len(centroids_df)
    n_countries = len(encoder_df)

    # 2. Build Model
    model = GeoLocModel(
        n_geocells=n_geocells,
        n_countries=n_countries,
        clip_model_name="ViT-B-32",
        clip_pretrained="openai",
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[Inference] Loading model checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt))
    else:
        print("[Inference] WARNING: No checkpoint provided — running inference with base model")

    model = model.to(device)
    model.eval()

    # 3. Load Test Data
    if use_tta:
        print("[Inference] Using TTA (5 crops per image)")
        ds = TTATestDataset(Path(test_dir))
    else:
        ds = TestDataset(Path(test_dir))

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_image_ids = []
    all_pred_geocells = []
    all_pred_countries = []
    all_country_confs = []
    all_embeddings = []

    print("[Inference] Generating model predictions...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if use_tta:
                # batch['images']: [B, 5, 3, 224, 224]
                b, n_crops, c, h, w = batch["images"].shape
                flat_imgs = batch["images"].view(-1, c, h, w).to(device)
                out = model(flat_imgs)
                
                # Average logits across crops
                gc_logits = out["geocell_logits"].view(b, n_crops, -1).mean(dim=1)
                ct_logits = out["country_logits"].view(b, n_crops, -1).mean(dim=1)
                embeds = out["embeddings"].view(b, n_crops, -1).mean(dim=1)
                embeds = F.normalize(embeds, p=2, dim=-1)
            else:
                imgs = batch["image"].to(device)
                out = model(imgs)
                gc_logits = out["geocell_logits"]
                ct_logits = out["country_logits"]
                embeds = F.normalize(out["embeddings"], p=2, dim=-1)

            # Predictions
            pred_gc = gc_logits.argmax(dim=-1).cpu().numpy()
            pred_ct_probs = F.softmax(ct_logits, dim=-1)
            pred_ct_conf, pred_ct = pred_ct_probs.max(dim=-1)

            all_image_ids.extend(batch["image_id"])
            all_pred_geocells.extend(pred_gc)
            all_pred_countries.extend([idx2country.get(i, "UNK") for i in pred_ct.cpu().numpy()])
            all_country_confs.extend(pred_ct_conf.cpu().numpy())
            all_embeddings.append(embeds.cpu().numpy())

    all_pred_geocells = np.array(all_pred_geocells)
    all_country_confs = np.array(all_country_confs)
    all_embeddings = np.vstack(all_embeddings)

    # Initial coordinates from geocell centroids
    pred_lats = centroids_df["centroid_lat"].values[all_pred_geocells]
    pred_lons = centroids_df["centroid_lon"].values[all_pred_geocells]
    base_radii = centroids_df["max_radius_km"].values[all_pred_geocells]

    # 4. kNN Refinement
    if use_knn and (DATA_DIR / "faiss_index.bin").exists():
        print("[Inference] Applying FAISS kNN refinement...")
        try:
            from refinement.knn_refine import KNNRefiner
            refiner = KNNRefiner()
            pred_lats, pred_lons = refiner.refine(
                query_embeddings=all_embeddings,
                centroid_lats=pred_lats,
                centroid_lons=pred_lons,
                top_k=5,
                blend_weight=0.35,
            )
        except Exception as e:
            print(f"[Inference] kNN refinement skipped due to: {e}")

    # 5. Radius Calibration
    calib_params_path = DATA_DIR / "calibration_params.json"
    alpha = 1.15
    min_r = 100.0
    if calib_params_path.exists():
        with open(calib_params_path, "r") as f:
            cp = json.load(f)
            alpha = cp.get("alpha", alpha)
            min_r = cp.get("min_radius_km", min_r)
        print(f"[Inference] Loaded calibrated radius params: alpha={alpha}, min_r={min_r}")

    pred_radii = np.clip(alpha * base_radii, min_r, 2500.0)

    # 6. Country Snapping
    if use_snap and (ROOT / "country_boundaries.geojson").exists():
        print("[Inference] Applying Country Boundary snapping post-processing...")
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
            print(f"[Inference] Country snapping skipped due to: {e}")

    # 7. Format Output & Match Schema
    pred_df = pd.DataFrame({
        "image_id": all_image_ids,
        "pred_lat": np.round(pred_lats, 6),
        "pred_lon": np.round(pred_lons, 6),
        "pred_radius_km": np.round(pred_radii, 2),
    })

    # Ensure ordering and IDs match sample_submission.csv
    if Path(sample_sub_path).exists():
        sample_sub = pd.read_csv(sample_sub_path)
        pred_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
        # Fill any missing with defaults
        pred_df["pred_lat"] = pred_df["pred_lat"].fillna(0.0)
        pred_df["pred_lon"] = pred_df["pred_lon"].fillna(0.0)
        pred_df["pred_radius_km"] = pred_df["pred_radius_km"].fillna(1000.0)

    if output_csv_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_path = str(SUBMISSIONS_DIR / f"submission_{timestamp}.csv")

    pred_df.to_csv(output_csv_path, index=False)
    print(f"\n[Inference] Saved final submission -> {output_csv_path}")
    print(f"[Inference] Total rows: {len(pred_df)}")
    print(pred_df.head(10).to_string(index=False))

    # Schema assertions
    assert list(pred_df.columns) == ["image_id", "pred_lat", "pred_lon", "pred_radius_km"]
    assert pred_df["pred_lat"].between(-90, 90).all()
    assert pred_df["pred_lon"].between(-180, 180).all()
    assert (pred_df["pred_radius_km"] > 0).all()
    print("\n[Inference] SUCCESS: Submission CSV passed all schema and range validations!")

    return output_csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=str(ROOT / "test_images_sampled"))
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--no_knn", action="store_true")
    parser.add_argument("--no_snap", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    generate_submission(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        output_csv_path=args.out_csv,
        use_tta=not args.no_tta,
        use_knn=not args.no_knn,
        use_snap=not args.no_snap,
        batch_size=args.batch_size,
    )
