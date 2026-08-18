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
    val_country_confs: np.ndarray,         # per-image confidence ∈ [0, 1]
    val_true_isos: list = None,
    val_pred_isos: list = None,
    output_path: str = str(DATA_DIR / "calibration_params.json"),
) -> dict:
    """
    Grid-searches the optimal per-image confidence-driven radius curve:

        r(conf) = clip(A + B * sqrt(1 - conf), min_r, max_r)

    where:
      A = base floor radius (km) — radius even at perfect confidence
      B = uncertainty scaling (km) — how much radius grows as confidence drops

    At conf=1.0 → r = A         (tight, high certainty)
    At conf=0.0 → r = A + B     (wide, complete uncertainty)
    At conf=0.5 → r = A + B*0.707

    This is a strict improvement over a flat global radius because:
      - High-confidence correct predictions can use tight radii → radius bonus
      - Low-confidence predictions stay wide → avoid large penalties
    """
    print("[Radius Calibrator] Grid-searching confidence-driven curve: r = A + B * sqrt(1 - conf)")
    print(f"[Radius Calibrator] Val samples: {len(val_true_lats)}")

    A_grid = np.arange(200.0, 1600.0, 100.0)    # floor: 200–1500 km
    B_grid = np.arange(200.0, 2200.0, 100.0)    # slope: 200–2100 km

    best_score = -float("inf")
    best_params = {}

    sqrt_uncertainty = np.sqrt(np.clip(1.0 - val_country_confs, 0.0, 1.0))

    for A in A_grid:
        for B in B_grid:
            cand_radii = np.clip(A + B * sqrt_uncertainty, 50.0, 3000.0)
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
                    "formula": "A + B * sqrt(1 - conf)",
                    "A": float(A),
                    "B": float(B),
                    # Keep these for back-compat with any code still reading them
                    "alpha": None,
                    "min_radius_km": float(A),
                    "max_radius_km": float(A + B),
                    "val_median_score":    float(res["median_score"]),
                    "val_mean_score":      float(res["mean_score"]),
                    "val_coverage_rate":   float(res["coverage_rate"]),
                    "val_median_dist_km":  float(res["median_haversine_km"]),
                    # Confidence percentile at which r = A (conf=1) and r = A+B (conf=0)
                    "r_at_conf1": float(A),
                    "r_at_conf0": float(A + B),
                    "r_at_conf05": float(A + B * (0.5 ** 0.5)),
                }

    print(f"[Radius Calibrator] Best params found:")
    for k, v in best_params.items():
        if v is not None:
            print(f"  {k:25s}: {v}")

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
    country_confs = []
    true_isos, pred_isos = [], []

    print("[Radius Calibrator] Running Stage-1 forward pass on val split...")
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            out  = model(imgs)

            # Stage 1: raw argmax — no blending, no kNN
            pred_gc   = out["geocell_logits"].argmax(dim=-1).cpu().numpy()
            ct_probs  = torch.nn.functional.softmax(out["country_logits"], dim=-1)
            ct_conf, pred_ct = ct_probs.max(dim=-1)

            true_lats.extend(batch["latitude"].numpy())
            true_lons.extend(batch["longitude"].numpy())
            true_isos.extend([idx2country.get(int(i), "UNK") for i in batch["country_idx"].numpy()])

            pred_lats.extend(centroids_df["centroid_lat"].values[pred_gc])
            pred_lons.extend(centroids_df["centroid_lon"].values[pred_gc])
            country_confs.extend(ct_conf.cpu().numpy())
            pred_isos.extend([idx2country.get(int(i), "UNK") for i in pred_ct.cpu().numpy()])

    return optimize_radius_parameters(
        val_true_lats=np.array(true_lats),
        val_true_lons=np.array(true_lons),
        val_pred_lats=np.array(pred_lats),
        val_pred_lons=np.array(pred_lons),
        val_country_confs=np.array(country_confs),
        val_true_isos=true_isos,
        val_pred_isos=pred_isos,
        output_path=output_path,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate radius params against Stage-1 val predictions."
    )
    parser.add_argument("--checkpoint",  type=str, default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--val_frac",    type=float, default=0.1)
    parser.add_argument("--output",      type=str, default=str(DATA_DIR / "calibration_params.json"))
    args = parser.parse_args()

    if Path(args.checkpoint).exists():
        calibrate_on_checkpoint(
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            output_path=args.output,
        )
    else:
        # Fallback synthetic test
        n = 200
        t_lats = np.random.uniform(-50, 60, n)
        t_lons = np.random.uniform(-150, 150, n)
        p_lats = t_lats + np.random.normal(0, 1.5, n)
        p_lons = t_lons + np.random.normal(0, 1.5, n)
        base_r = np.random.uniform(50, 300, n)
        optimize_radius_parameters(t_lats, t_lons, p_lats, p_lons, base_r)

