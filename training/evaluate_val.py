"""
evaluate_val.py — Comprehensive validation evaluation script

Evaluates a trained model checkpoint on the held-out validation set across 4 stages:
  Stage 1: Raw Argmax Geocell Centroid (Baseline in train.py)
  Stage 2: Top-K Soft Probability Centroid Blending
  Stage 3: FAISS kNN Visual Embedding Refinement
  Stage 4: Country Boundary Snapping Post-Processing

Reports:
  - Country Top-1 Accuracy
  - Geocell Top-1 and Top-5 Accuracy
  - Haversine Distance (Median, Mean, p25, p75)
  - Within 25km, 200km, 750km
  - Competition Proxy Median & Mean Score
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.model import GeoLocModel
from data.dataset import GeoDataset
from calibration.scoring_proxy import haversine_km, compute_competition_score

DATA_DIR = ROOT / "data"


from geocells.build_geocells import spherical_weighted_average


def evaluate_checkpoint(
    checkpoint_path: str = "checkpoints/best.pt",
    batch_size: int = 64,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_knn: bool = True,
    use_snap: bool = True,
):
    device = torch.device(device_str)
    print(f"[Eval] Device: {device}")
    print(f"[Eval] Checkpoint: {checkpoint_path}")

    val_ds = GeoDataset(split="val", val_frac=0.1, augment=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2 if device.type == "cuda" else 0)

    centroids_df = pd.read_csv(DATA_DIR / "geocell_centroids.csv").sort_values("geocell_id").reset_index(drop=True)
    encoder_df = pd.read_csv(DATA_DIR / "country_encoder.csv")
    idx2country = dict(zip(encoder_df["country_idx"], encoder_df["country_iso"]))

    n_geocells = len(centroids_df)
    n_countries = len(encoder_df)

    model = GeoLocModel(
        n_geocells=n_geocells,
        n_countries=n_countries,
        clip_model_name="ViT-B-32",
        clip_pretrained="openai",
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    model = model.to(device)
    model.eval()

    all_true_lats, all_true_lons = [], []
    all_true_cells, all_true_countries = [], []
    all_raw_pred_lats, all_raw_pred_lons = [], []
    all_soft_pred_lats, all_soft_pred_lons = [], []
    all_pred_cells, all_top5_cells = [], []
    all_pred_countries, all_country_confs = [], []
    all_pred_country_idxs  = []   # top-1 predicted country index (int)
    all_top2_country_idxs  = []   # top-2 predicted country index (int)
    all_top3_country_idxs  = []   # top-3 predicted country index (int)
    all_image_ids = []
    all_embeddings = []
    base_radii = []

    print("[Eval] Running model forward pass over validation split...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            imgs = batch["image"].to(device)
            out = model(imgs)

            gc_logits = out["geocell_logits"]
            ct_logits = out["country_logits"]
            embeds = F.normalize(out["embeddings"], p=2, dim=-1)

            # 1. Raw argmax
            pred_gc = gc_logits.argmax(dim=-1).cpu().numpy()
            top5_gc = torch.topk(gc_logits, k=min(5, gc_logits.shape[-1]), dim=-1).indices.cpu().numpy()

            # 2. Spherical Top-K centroid blending on 3D unit sphere
            top5_probs = F.softmax(torch.topk(gc_logits, k=5, dim=-1).values, dim=-1).cpu().numpy()
            top5_indices = top5_gc
            batch_soft_lats = np.zeros(len(imgs))
            batch_soft_lons = np.zeros(len(imgs))
            for i in range(len(imgs)):
                c_lats = centroids_df["centroid_lat"].values[top5_indices[i]]
                c_lons = centroids_df["centroid_lon"].values[top5_indices[i]]
                s_lat, s_lon = spherical_weighted_average(c_lats, c_lons, top5_probs[i])
                batch_soft_lats[i] = s_lat
                batch_soft_lons[i] = s_lon

            # Country predictions — top-1, top-2, top-3 for masked retrieval
            ct_probs = F.softmax(ct_logits, dim=-1)
            ct_top3 = torch.topk(ct_probs, k=min(3, ct_probs.shape[-1]), dim=-1)
            ct_top3_idxs  = ct_top3.indices.cpu().numpy()   # [B, 3]
            ct_top3_confs = ct_top3.values.cpu().numpy()    # [B, 3]
            ct_conf = ct_top3_confs[:, 0]
            ct_idx  = ct_top3_idxs[:, 0]

            all_image_ids.extend(batch["image_id"])
            all_true_lats.extend(batch["latitude"].numpy())
            all_true_lons.extend(batch["longitude"].numpy())
            all_true_cells.extend(batch["geocell_id"].numpy())
            all_true_countries.extend([idx2country.get(i, "UNK") for i in batch["country_idx"].numpy()])

            all_raw_pred_lats.extend(centroids_df["centroid_lat"].values[pred_gc])
            all_raw_pred_lons.extend(centroids_df["centroid_lon"].values[pred_gc])
            all_soft_pred_lats.extend(batch_soft_lats)
            all_soft_pred_lons.extend(batch_soft_lons)
            base_radii.extend(centroids_df["max_radius_km"].values[pred_gc])

            all_pred_cells.extend(pred_gc)
            all_top5_cells.extend(top5_gc)
            all_pred_countries.extend([idx2country.get(i, "UNK") for i in ct_idx])
            all_country_confs.extend(ct_conf)
            all_pred_country_idxs.extend(ct_top3_idxs[:, 0].tolist())
            all_top2_country_idxs.extend(ct_top3_idxs[:, 1].tolist() if ct_top3_idxs.shape[1] > 1 else ct_top3_idxs[:, 0].tolist())
            all_top3_country_idxs.extend(ct_top3_idxs[:, 2].tolist() if ct_top3_idxs.shape[1] > 2 else ct_top3_idxs[:, 0].tolist())
            all_embeddings.append(embeds.cpu().numpy())


    true_lats = np.array(all_true_lats)
    true_lons = np.array(all_true_lons)
    all_true_cells       = np.array(all_true_cells)
    all_pred_cells       = np.array(all_pred_cells)
    all_top5_cells       = np.array(all_top5_cells)
    all_country_confs    = np.array(all_country_confs)
    all_pred_country_idxs = np.array(all_pred_country_idxs, dtype=np.int32)
    all_top2_country_idxs = np.array(all_top2_country_idxs, dtype=np.int32)
    all_top3_country_idxs = np.array(all_top3_country_idxs, dtype=np.int32)
    all_embeddings       = np.vstack(all_embeddings)
    base_radii           = np.array(base_radii)

    # Base classification accuracies
    geocell_top1 = float((all_true_cells == all_pred_cells).mean())
    geocell_top5 = float(np.mean([all_true_cells[i] in all_top5_cells[i] for i in range(len(all_true_cells))]))
    country_top1 = float(np.mean([t == p for t, p in zip(all_true_countries, all_pred_countries)]))

    print("\n" + "=" * 65)
    print(f"  CLASSIFICATION METRICS (Val N = {len(true_lats)})")
    print("=" * 65)
    print(f"  Country Top-1 Accuracy:  {country_top1 * 100:.2f}%")
    print(f"  Geocell Top-1 Accuracy:  {geocell_top1 * 100:.2f}%")
    print(f"  Geocell Top-5 Accuracy:  {geocell_top5 * 100:.2f}%")

    # Stage 1: Raw Argmax Centroid
    dists_raw = haversine_km(true_lats, true_lons, np.array(all_raw_pred_lats), np.array(all_raw_pred_lons))
    
    # Stage 2: Soft Top-5 Blending
    dists_soft = haversine_km(true_lats, true_lons, np.array(all_soft_pred_lats), np.array(all_soft_pred_lons))

    # Stage 3: Country-Masked FAISS kNN Refinement
    # NOTE: We refine from the RAW argmax centroid (not soft blend) so the two
    # stages are independent and their contributions are clearly measurable.
    refined_lats = np.array(all_raw_pred_lats).copy()
    refined_lons = np.array(all_raw_pred_lons).copy()
    if use_knn and (DATA_DIR / "faiss_index.bin").exists():
        try:
            from refinement.knn_refine import KNNRefiner
            refiner = KNNRefiner()
            refined_lats, refined_lons = refiner.refine(
                query_embeddings=all_embeddings,
                centroid_lats=refined_lats,
                centroid_lons=refined_lons,
                pred_country_idxs=all_pred_country_idxs,
                top2_country_idxs=all_top2_country_idxs,
                top3_country_idxs=all_top3_country_idxs,
                top_k=10,
                oversample=60,
                blend_weight=0.30,
                min_similarity_threshold=0.65,
                min_candidates=2,
            )
            dists_knn = haversine_km(true_lats, true_lons, refined_lats, refined_lons)
        except Exception as e:
            print(f"kNN refinement skipped: {e}")
            dists_knn = dists_soft
    else:
        dists_knn = dists_soft

    # Stage 4: Country Snapping
    snapped_lats, snapped_lons = refined_lats.copy(), refined_lons.copy()
    if use_snap and (ROOT / "country_boundaries.geojson").exists():
        try:
            from calibration.country_snap import CountrySnapper
            snapper = CountrySnapper()
            snapped_lats, snapped_lons = snapper.snap_coordinates(
                lats=snapped_lats,
                lons=snapped_lons,
                country_isos=all_pred_countries,
                country_confidences=all_country_confs,
                min_confidence=0.45,
            )
            dists_final = haversine_km(true_lats, true_lons, snapped_lats, snapped_lons)
        except Exception as e:
            print(f"Country snapping skipped: {e}")
            dists_final = dists_knn
    else:
        dists_final = dists_knn

    # -------------------------------------------------------------------------
    # Radius Calibration: run grid search on THIS set of predictions to find
    # the optimal radius curve, instead of relying on a stale hardcoded 3-bucket
    # -------------------------------------------------------------------------
    from calibration.calibrate_radius import optimize_radius_parameters

    print("\n[Eval] Running radius grid search on Stage 4 predictions...")
    calib_params = optimize_radius_parameters(
        val_true_lats=true_lats,
        val_true_lons=true_lons,
        val_pred_lats=snapped_lats,
        val_pred_lons=snapped_lons,
        val_cell_base_radii=base_radii,
        val_true_isos=all_true_countries,
        val_pred_isos=all_pred_countries,
    )
    alpha     = calib_params["alpha"]
    min_r     = calib_params["min_radius_km"]
    max_r     = calib_params.get("max_radius_km", 2500.0)
    pred_radii = np.clip(alpha * base_radii, min_r, max_r)

    # Also compute per-stage scores so we can see the refinement impact on the proxy
    def stage_score(pred_lats_s, pred_lons_s, radii):
        return compute_competition_score(
            true_lats=true_lats, true_lons=true_lons,
            pred_lats=pred_lats_s, pred_lons=pred_lons_s,
            pred_radii_km=radii,
            true_country_isos=all_true_countries,
            pred_country_isos=all_pred_countries,
        )

    score_raw  = stage_score(np.array(all_raw_pred_lats), np.array(all_raw_pred_lons), pred_radii)
    score_soft = stage_score(np.array(all_soft_pred_lats), np.array(all_soft_pred_lons), pred_radii)
    score_knn  = stage_score(refined_lats, refined_lons, pred_radii)
    score_snap = stage_score(snapped_lats, snapped_lons, pred_radii)

    print("\n" + "=" * 65)
    print("  SAMPLE VALIDATION PREDICTIONS (First 5 Images)")
    print("=" * 65)
    for i in range(min(5, len(true_lats))):
        img_id = all_image_ids[i]
        t_lat, t_lon = true_lats[i], true_lons[i]
        r_lat, r_lon = all_raw_pred_lats[i], all_raw_pred_lons[i]
        s_lat, s_lon = all_soft_pred_lats[i], all_soft_pred_lons[i]
        k_lat, k_lon = refined_lats[i], refined_lons[i]

        d_raw = haversine_km(np.array([t_lat]), np.array([t_lon]), np.array([r_lat]), np.array([r_lon]))[0]
        d_soft = haversine_km(np.array([t_lat]), np.array([t_lon]), np.array([s_lat]), np.array([s_lon]))[0]
        d_knn = haversine_km(np.array([t_lat]), np.array([t_lon]), np.array([k_lat]), np.array([k_lon]))[0]

        print(f"\n[Sample #{i+1}] Image: {img_id}")
        print(f"  True Coords:         ({t_lat:8.4f}, {t_lon:9.4f}) | Country: {all_true_countries[i]}")
        print(f"  Stage 1 (Raw):       ({r_lat:8.4f}, {r_lon:9.4f}) | Error: {d_raw:7.1f} km")
        print(f"  Stage 2 (Soft 3D):   ({s_lat:8.4f}, {s_lon:9.4f}) | Error: {d_soft:7.1f} km (delta: {d_soft - d_raw:+6.1f} km)")
        print(f"  Stage 3 (kNN 3D):    ({k_lat:8.4f}, {k_lon:9.4f}) | Error: {d_knn:7.1f} km (delta: {d_knn - d_soft:+6.1f} km)")

    print("\n" + "=" * 65)
    print("  HAVERSINE DISTANCE PROGRESSION BY STAGE")
    print("=" * 65)
    print(f"  Stage 1 (Raw Argmax Centroid):     Median = {np.median(dists_raw):.1f} km | Mean = {np.mean(dists_raw):.1f} km | <750km = {(dists_raw < 750).mean()*100:.1f}%")
    print(f"  Stage 2 (Soft Top-5 Blending):     Median = {np.median(dists_soft):.1f} km | Mean = {np.mean(dists_soft):.1f} km | <750km = {(dists_soft < 750).mean()*100:.1f}%")
    print(f"  Stage 3 (FAISS kNN Refinement):    Median = {np.median(dists_knn):.1f} km | Mean = {np.mean(dists_knn):.1f} km | <750km = {(dists_knn < 750).mean()*100:.1f}%")
    print(f"  Stage 4 (Final + Country Snapped): Median = {np.median(dists_final):.1f} km | Mean = {np.mean(dists_final):.1f} km | <750km = {(dists_final < 750).mean()*100:.1f}%")

    print()
    print(f"  Calibrated Radius: alpha={alpha:.2f} * base_radius, floor={min_r:.0f} km, max={max_r:.0f} km")
    print(f"  Calibrated pred_radii: p25={np.percentile(pred_radii,25):.0f} median={np.median(pred_radii):.0f} p75={np.percentile(pred_radii,75):.0f} km")
    print()
    print(f"  {'Stage':<36}  {'MedianScore':>12}  {'Coverage':>10}  {'MeanScore':>10}")
    print(f"  {'-'*36}  {'-'*12}  {'-'*10}  {'-'*10}")
    print(f"  {'Stage 1 (Raw Centroid)':<36}  {score_raw['median_score']:>12.4f}  {score_raw['coverage_rate']*100:>9.1f}%  {score_raw['mean_score']:>10.4f}")
    print(f"  {'Stage 2 (Soft 3D Blend)':<36}  {score_soft['median_score']:>12.4f}  {score_soft['coverage_rate']*100:>9.1f}%  {score_soft['mean_score']:>10.4f}")
    print(f"  {'Stage 3 (Country-Masked kNN)':<36}  {score_knn['median_score']:>12.4f}  {score_knn['coverage_rate']*100:>9.1f}%  {score_knn['mean_score']:>10.4f}")
    print(f"  {'Stage 4 (+ Country Snapped)':<36}  {score_snap['median_score']:>12.4f}  {score_snap['coverage_rate']*100:>9.1f}%  {score_snap['mean_score']:>10.4f}")
    print("=" * 65)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    args = parser.parse_args()
    evaluate_checkpoint(checkpoint_path=args.checkpoint)
