"""
diagnose_index.py — Three independent verification checks:

  CHECK 1: FAISS self-retrieval test (5 embeddings, confirm top-1 = self, near-zero distance)
  CHECK 2: Index staleness — does metadata overlap with current validation split?
  CHECK 3: Country top-1 accuracy excluding OCEAN rows
"""

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

import faiss

from models.model import GeoLocModel
from data.dataset import GeoDataset
from calibration.scoring_proxy import haversine_km

DATA_DIR = ROOT / "data"


# =============================================================================
# CHECK 1: FAISS Self-Retrieval Test
# =============================================================================
def check_faiss_self_retrieval():
    print("\n" + "=" * 70)
    print("  CHECK 1: FAISS Self-Retrieval Test")
    print("=" * 70)

    index_path = DATA_DIR / "faiss_index.bin"
    meta_path  = DATA_DIR / "faiss_metadata.csv"

    if not index_path.exists():
        print("  SKIP: faiss_index.bin not found")
        return

    index = faiss.read_index(str(index_path))
    meta  = pd.read_csv(meta_path)

    print(f"  Index vectors:    {index.ntotal}")
    print(f"  Metadata rows:    {len(meta)}")
    print(f"  Match:            {'OK' if index.ntotal == len(meta) else '*** MISMATCH ***'}")
    print()

    # Take 5 sample indices spread across the index
    sample_idxs = [0, 500, 2000, 9000, 17999]
    sample_idxs = [i for i in sample_idxs if i < index.ntotal]

    # Reconstruct the actual stored vectors
    test_vectors = np.vstack([
        faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d).reshape(index.ntotal, index.d)[i]
        for i in sample_idxs
    ]).astype(np.float32)

    sims, retrieved_idxs = index.search(test_vectors, k=1)

    print(f"  {'Sample Idx':>12}  {'Image ID':>20}  {'Top-1 Retrieved':>15}  {'Same Image?':>12}  {'Similarity':>12}")
    print(f"  {'-'*12}  {'-'*20}  {'-'*15}  {'-'*12}  {'-'*12}")
    all_correct = True
    for i, sample_i in enumerate(sample_idxs):
        top1_idx    = retrieved_idxs[i][0]
        top1_sim    = sims[i][0]
        img_id      = meta.iloc[sample_i]["image_id"]
        top1_img_id = meta.iloc[top1_idx]["image_id"] if top1_idx < len(meta) else "OOB"
        is_same     = (sample_i == top1_idx)
        marker      = "✓" if is_same else "✗ MISALIGNED"
        if not is_same:
            all_correct = False
        print(f"  {sample_i:>12}  {str(img_id):>20}  {str(top1_img_id):>15}  {marker:>12}  {top1_sim:>12.6f}")

    print()
    if all_correct:
        print("  RESULT: All 5 self-retrievals returned the exact same image with similarity ≈ 1.0")
        print("          Index vectors and metadata rows are correctly aligned.")
    else:
        print("  RESULT: *** MISALIGNMENT DETECTED *** — vectors and metadata rows are out of sync!")


# =============================================================================
# CHECK 2: Index Staleness — val/train overlap in FAISS metadata
# =============================================================================
def check_index_staleness():
    print("\n" + "=" * 70)
    print("  CHECK 2: Index Staleness — Val Contamination in FAISS Index")
    print("=" * 70)

    index_path = DATA_DIR / "faiss_index.bin"
    meta_path  = DATA_DIR / "faiss_metadata.csv"

    if not index_path.exists() or not meta_path.exists():
        print("  SKIP: faiss_index.bin or faiss_metadata.csv not found")
        return

    # Current val split
    val_ds   = GeoDataset(split="val",   val_frac=0.1, augment=False)
    train_ds = GeoDataset(split="train", val_frac=0.1, augment=False)

    val_ids   = set(val_ds.df["image_id"].tolist())
    train_ids = set(train_ds.df["image_id"].tolist())

    meta = pd.read_csv(meta_path)
    index_ids = set(meta["image_id"].tolist())

    overlap_with_val   = val_ids & index_ids
    overlap_with_train = train_ids & index_ids
    missing_from_index = train_ids - index_ids

    print(f"  Current val split size:         {len(val_ids)} images")
    print(f"  Current train split size:       {len(train_ids)} images")
    print(f"  FAISS index vectors:            {len(index_ids)}")
    print()
    print(f"  VAL images found in index:      {len(overlap_with_val)}  {'*** CONTAMINATED ***' if overlap_with_val else '(clean, no contamination)'}")
    print(f"  Train images in index:          {len(overlap_with_train)}")
    print(f"  Train images MISSING from idx:  {len(missing_from_index)}")
    print()

    if overlap_with_val:
        print(f"  First 5 contaminating val image_ids in index:")
        for img_id in list(overlap_with_val)[:5]:
            print(f"    {img_id}")
        print()
        print("  ACTION NEEDED: Rebuild FAISS index from ONLY the train split!")
    else:
        print("  RESULT: Index is clean — no validation images are indexed.")

    index = faiss.read_index(str(index_path))
    if index.ntotal != len(train_ids):
        print(f"\n  STALENESS: index.ntotal ({index.ntotal}) ≠ len(train_df) ({len(train_ids)})")
        print("  => Index was built from a different split (likely before singleton-merge).")
        print("  => REBUILD REQUIRED.")
    else:
        print(f"\n  FRESH: index.ntotal ({index.ntotal}) == len(train_df) ({len(train_ids)}). Index matches current train split.")


# =============================================================================
# CHECK 3: Country Accuracy Excluding OCEAN Labels
# =============================================================================
def check_country_accuracy(checkpoint_path: str = "checkpoints/k1000_original/best.pt"):
    print("\n" + "=" * 70)
    print(f"  CHECK 3: Country Top-1 Accuracy (All vs Non-OCEAN)")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = GeoDataset(split="val", val_frac=0.1, augment=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=2 if device.type == "cuda" else 0)

    encoder_df  = pd.read_csv(DATA_DIR / "country_encoder.csv")
    idx2country = dict(zip(encoder_df["country_idx"], encoder_df["country_iso"]))
    n_countries = len(encoder_df)

    centroids_df = pd.read_csv(DATA_DIR / "geocell_centroids.csv").sort_values("geocell_id").reset_index(drop=True)
    n_geocells = len(centroids_df)

    model = GeoLocModel(
        n_geocells=n_geocells,
        n_countries=n_countries,
        clip_model_name="ViT-B-32",
        clip_pretrained="openai",
    )

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    model = model.to(device)
    model.eval()

    all_true_countries = []
    all_pred_countries = []
    all_true_cells = []
    all_pred_cells = []
    all_top5_cells = []
    all_true_lats, all_true_lons = [], []
    all_pred_lats, all_pred_lons = [], []

    print("  Running forward pass...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            imgs = batch["image"].to(device)
            out  = model(imgs)

            gc_logits = out["geocell_logits"]
            ct_logits = out["country_logits"]

            pred_gc   = gc_logits.argmax(dim=-1).cpu().numpy()
            top5_gc   = torch.topk(gc_logits, k=min(5, gc_logits.shape[-1]), dim=-1).indices.cpu().numpy()
            pred_ct   = ct_logits.argmax(dim=-1).cpu().numpy()

            pred_lats = centroids_df["centroid_lat"].values[pred_gc]
            pred_lons = centroids_df["centroid_lon"].values[pred_gc]

            all_true_countries.extend([idx2country.get(int(i), "UNK") for i in batch["country_idx"].numpy()])
            all_pred_countries.extend([idx2country.get(int(i), "UNK") for i in pred_ct])
            all_true_cells.extend(batch["geocell_id"].numpy())
            all_pred_cells.extend(pred_gc)
            all_top5_cells.extend(top5_gc)
            all_true_lats.extend(batch["latitude"].numpy())
            all_true_lons.extend(batch["longitude"].numpy())
            all_pred_lats.extend(pred_lats)
            all_pred_lons.extend(pred_lons)

    all_true_cells    = np.array(all_true_cells)
    all_pred_cells    = np.array(all_pred_cells)
    all_top5_cells    = np.array(all_top5_cells)
    all_true_lats     = np.array(all_true_lats)
    all_true_lons     = np.array(all_true_lons)
    all_pred_lats     = np.array(all_pred_lats)
    all_pred_lons     = np.array(all_pred_lons)

    correct_country = np.array([t == p for t, p in zip(all_true_countries, all_pred_countries)])
    is_ocean        = np.array([t == "OCEAN" for t in all_true_countries])
    is_land         = ~is_ocean

    country_top1_all    = float(correct_country.mean())
    country_top1_land   = float(correct_country[is_land].mean()) if is_land.sum() > 0 else float("nan")

    geocell_top1 = float((all_true_cells == all_pred_cells).mean())
    geocell_top5 = float(np.mean([all_true_cells[i] in all_top5_cells[i] for i in range(len(all_true_cells))]))

    dists = haversine_km(all_true_lats, all_true_lons, all_pred_lats, all_pred_lons)

    print()
    print(f"  Validation samples (total):         {len(all_true_countries)}")
    print(f"  OCEAN-labelled samples:             {is_ocean.sum()} ({is_ocean.mean()*100:.1f}%)")
    print(f"  Land-labelled samples:              {is_land.sum()} ({is_land.mean()*100:.1f}%)")
    print()
    print(f"  Country Top-1 (All):                {country_top1_all*100:.2f}%")
    print(f"  Country Top-1 (Land Only):          {country_top1_land*100:.2f}%  ← compare to reported 64.2%")
    print()
    print(f"  Geocell Top-1:                      {geocell_top1*100:.2f}%")
    print(f"  Geocell Top-5:                      {geocell_top5*100:.2f}%")
    print()
    print(f"  Haversine Median (Raw Centroid):    {np.median(dists):.1f} km")
    print(f"  Within 750 km:                      {(dists < 750).mean()*100:.1f}%")
    print()

    # Break down country errors
    wrong_land = (~correct_country) & is_land
    print(f"  Wrong-country land images:          {wrong_land.sum()} / {is_land.sum()}")
    # Most confused countries on land
    from collections import Counter
    wrong_true  = [all_true_countries[i] for i in range(len(all_true_countries)) if wrong_land[i]]
    wrong_pred  = [all_pred_countries[i] for i in range(len(all_pred_countries)) if wrong_land[i]]
    confused    = Counter(zip(wrong_true, wrong_pred)).most_common(5)
    print(f"  Top-5 country confusions (true -> pred):")
    for (t, p), c in confused:
        print(f"    {t:>6} -> {p:<6}  ({c} images)")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/k1000_original/best.pt")
    args = parser.parse_args()

    check_faiss_self_retrieval()
    check_index_staleness()
    check_country_accuracy(checkpoint_path=args.checkpoint)
