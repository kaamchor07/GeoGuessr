"""
calibrate_radius.py — Step 6, PRD Section 4 item 8

Grid-searches the optimal radius scaling factor and offset to maximize the
competition scoring metric on the held-out validation set.

Outputs:
  data/calibration_params.json
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calibration.scoring_proxy import compute_competition_score, haversine_km

DATA_DIR = ROOT / "data"


def optimize_radius_parameters(
    val_true_lats: np.ndarray,
    val_true_lons: np.ndarray,
    val_pred_lats: np.ndarray,
    val_pred_lons: np.ndarray,
    val_cell_base_radii: np.ndarray,
    val_true_isos: list[str] = None,
    val_pred_isos: list[str] = None,
    output_path: str = str(DATA_DIR / "calibration_params.json"),
) -> dict:
    """
    Grid-searches multiplier alpha and floor min_r to maximize median competition score.
    Radius = clip(alpha * base_cell_radius + offset, min_r, max_r)
    """
    print("[Radius Calibrator] Starting grid search against competition proxy...")

    alpha_grid = np.linspace(0.5, 2.5, 21)
    min_r_grid = [25.0, 50.0, 100.0, 150.0, 200.0, 300.0, 500.0]

    best_score = -float("inf")
    best_params = {}

    for alpha in alpha_grid:
        for min_r in min_r_grid:
            cand_radii = np.clip(alpha * val_cell_base_radii, min_r, 2500.0)
            res = compute_competition_score(
                true_lats=val_true_lats,
                true_lons=val_true_lons,
                pred_lats=val_pred_lats,
                pred_lons=val_pred_lons,
                pred_radii_km=cand_radii,
                true_country_isos=val_true_isos,
                pred_country_isos=val_pred_isos,
            )
            score = res["median_score"]
            if score > best_score:
                best_score = score
                best_params = {
                    "alpha": float(alpha),
                    "min_radius_km": float(min_r),
                    "max_radius_km": 2500.0,
                    "val_median_score": float(res["median_score"]),
                    "val_mean_score": float(res["mean_score"]),
                    "val_coverage_rate": float(res["coverage_rate"]),
                    "val_median_dist_km": float(res["median_haversine_km"]),
                }

    print(f"[Radius Calibrator] Best params found:")
    for k, v in best_params.items():
        print(f"  {k:20s}: {v}")

    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"[Radius Calibrator] Saved calibration params -> {output_path}")

    return best_params


def calibrate_on_checkpoint(
    checkpoint_path: str = str(ROOT / "checkpoints" / "best.pt"),
    batch_size: int = 64,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_path: str = str(DATA_DIR / "calibration_params.json"),
):
    device = torch.device(device_str)
    print(f"[Radius Calibrator] Running on {device}")

    # Load validation split
    from data.dataset import GeoDataset
    from models.model import GeoLocModel
    from torch.utils.data import DataLoader

    val_ds = GeoDataset("val", val_frac=0.1, augment=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2 if device.type == "cuda" else 0)

    centroids_df = pd.read_csv(DATA_DIR / "geocell_centroids.csv").sort_values("geocell_id").reset_index(drop=True)
    encoder_df = pd.read_csv(DATA_DIR / "country_encoder.csv")
    idx2country = dict(zip(encoder_df["country_idx"], encoder_df["country_iso"]))

    # Load model
    model = GeoLocModel(
        n_geocells=val_ds.n_geocells,
        n_countries=val_ds.n_countries,
        clip_model_name="ViT-B-32",
        clip_pretrained="openai",
    )

    if Path(checkpoint_path).exists():
        print(f"[Radius Calibrator] Loading {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt))

    model = model.to(device)
    model.eval()

    true_lats, true_lons = [], []
    pred_lats, pred_lons = [], []
    base_radii = []
    true_isos, pred_isos = [], []

    print("[Radius Calibrator] Evaluating validation set...")
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            out = model(imgs)

            pred_gc = out["geocell_logits"].argmax(dim=-1).cpu().numpy()
            pred_ct = out["country_logits"].argmax(dim=-1).cpu().numpy()

            true_lats.extend(batch["latitude"].numpy())
            true_lons.extend(batch["longitude"].numpy())
            true_isos.extend([idx2country.get(i, "UNK") for i in batch["country_idx"].numpy()])

            pred_lats.extend(centroids_df["centroid_lat"].values[pred_gc])
            pred_lons.extend(centroids_df["centroid_lon"].values[pred_gc])
            base_radii.extend(centroids_df["max_radius_km"].values[pred_gc])
            pred_isos.extend([idx2country.get(i, "UNK") for i in pred_ct])

    return optimize_radius_parameters(
        val_true_lats=np.array(true_lats),
        val_true_lons=np.array(true_lons),
        val_pred_lats=np.array(pred_lats),
        val_pred_lons=np.array(pred_lons),
        val_cell_base_radii=np.array(base_radii),
        val_true_isos=true_isos,
        val_pred_isos=pred_isos,
        output_path=output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "best.pt"))
    args = parser.parse_args()

    if Path(args.checkpoint).exists():
        calibrate_on_checkpoint(args.checkpoint)
    else:
        # Fallback synthetic test
        n = 200
        t_lats = np.random.uniform(-50, 60, n)
        t_lons = np.random.uniform(-150, 150, n)
        p_lats = t_lats + np.random.normal(0, 1.5, n)
        p_lons = t_lons + np.random.normal(0, 1.5, n)
        base_r = np.random.uniform(50, 300, n)
        optimize_radius_parameters(t_lats, t_lons, p_lats, p_lons, base_r)

