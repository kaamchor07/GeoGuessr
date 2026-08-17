"""
scoring_proxy.py — Step 6, PRD Section 1 & Section 10

Implements a local proxy of the official competition scoring function:
  Score per image combines:
    1. Haversine distance decay score: S_dist = exp(-d / decay_km)
    2. Radius calibration term:
         - If true location is INSIDE claimed radius (d <= r):
             Bonus for tightness: S_rad = (1.0 - (r / r_max)**0.5) * bonus_scale
         - If true location is OUTSIDE claimed radius (d > r):
             Heavy penalty: S_rad = -penalty_scale * ((d - r) / r_ref)**1.2
    3. Country match bonus:
         - S_country = +bonus if predicted point is in correct country and r < country_r_thresh
    4. Total image score = max(0.0, S_dist + S_rad + S_country)
    
Overall evaluation metric: Median per-image score across test set.
"""

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def compute_competition_score(
    true_lats: np.ndarray,
    true_lons: np.ndarray,
    pred_lats: np.ndarray,
    pred_lons: np.ndarray,
    pred_radii_km: np.ndarray,
    true_country_isos: list[str] = None,
    pred_country_isos: list[str] = None,
    dist_decay_km: float = 750.0,
    r_max_km: float = 2500.0,
    r_penalty_scale: float = 1.5,
    country_bonus: float = 0.25,
) -> dict:
    """
    Computes per-image and summary median scores using the PRD formula proxy.
    """
    distances_km = haversine_km(true_lats, true_lons, pred_lats, pred_lons)
    
    # 1. Distance score (smooth exponential decay)
    s_dist = np.exp(-distances_km / dist_decay_km)

    # 2. Radius calibration bonus / penalty
    s_rad = np.zeros_like(distances_km)
    inside_mask = distances_km <= pred_radii_km
    outside_mask = ~inside_mask

    # Inside claimed radius: reward tight radius
    clipped_r = np.clip(pred_radii_km[inside_mask], 10.0, r_max_km)
    s_rad[inside_mask] = 0.5 * (1.0 - (clipped_r / r_max_km) ** 0.5)

    # Outside claimed radius: penalty proportional to miss distance
    overshoot = distances_km[outside_mask] - pred_radii_km[outside_mask]
    s_rad[outside_mask] = -r_penalty_scale * (overshoot / 500.0) ** 0.8

    # 3. Country match bonus
    s_country = np.zeros_like(distances_km)
    if true_country_isos is not None and pred_country_isos is not None:
        country_match = np.array([t == p and t not in ["OCEAN", "-99", "UNK"] for t, p in zip(true_country_isos, pred_country_isos)])
        tight_enough = pred_radii_km < 1000.0
        s_country[country_match & tight_enough] = country_bonus

    # Per image composite score
    per_image_score = np.clip(s_dist + s_rad + s_country, 0.0, 2.0)

    return {
        "median_score": float(np.median(per_image_score)),
        "mean_score": float(np.mean(per_image_score)),
        "median_haversine_km": float(np.median(distances_km)),
        "mean_haversine_km": float(np.mean(distances_km)),
        "coverage_rate": float(np.mean(inside_mask)),
        "within_25km": float(np.mean(distances_km < 25.0)),
        "within_200km": float(np.mean(distances_km < 200.0)),
        "within_750km": float(np.mean(distances_km < 750.0)),
        "per_image_scores": per_image_score,
        "distances_km": distances_km,
    }
