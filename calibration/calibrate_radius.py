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


if __name__ == "__main__":
    # Test stub with synthetic validation predictions
    print("Testing radius calibration module...")
    n = 200
    t_lats = np.random.uniform(-50, 60, n)
    t_lons = np.random.uniform(-150, 150, n)
    p_lats = t_lats + np.random.normal(0, 1.5, n)
    p_lons = t_lons + np.random.normal(0, 1.5, n)
    base_r = np.random.uniform(50, 300, n)

    optimize_radius_parameters(t_lats, t_lons, p_lats, p_lons, base_r)
