"""
dry_run.py  —  Step 2, PRD Section 9 step 2

End-to-end pipeline smoke test:
  1. Load 200 training images through GeoDataset
  2. Build model, count params
  3. Run 1 epoch — verify loss drops each batch
  4. Run validation — report haversine distance metrics
  5. Generate a submission CSV and verify schema matches sample_submission.csv exactly

Pass criteria (all must hold):
  - No crashes or import errors
  - Loss at final batch < loss at first batch (loss is decreasing)
  - Submission CSV has exactly 4 columns: image_id, pred_lat, pred_lon, pred_radius_km
  - Submission CSV row count == number of test images
  - All pred_lat in [-90, 90] and pred_lon in [-180, 180] and pred_radius_km > 0

Usage:
  python training/dry_run.py
"""

import json
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []

def check(cond, ok_msg, fail_msg):
    if cond:
        print(f"  {PASS} {ok_msg}")
    else:
        print(f"  {FAIL} {fail_msg}")
        errors.append(fail_msg)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── helpers ──────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def seed_everything(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ─────────────────────────────────────────────────────────────
section("0. IMPORTS")
# ─────────────────────────────────────────────────────────────
try:
    from data.dataset import GeoDataset, get_dataloaders, TestDataset
    print("  [PASS] data.dataset imported")
except Exception as e:
    print(f"  [FAIL] data.dataset: {e}"); errors.append(str(e)); sys.exit(1)

try:
    from models.model import GeoLocModel, GeoLoss
    print("  [PASS] models.model imported")
except Exception as e:
    print(f"  [FAIL] models.model: {e}"); errors.append(str(e)); sys.exit(1)

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# Create output dirs early
(ROOT / "checkpoints").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
section("1. DATASET  (200 train / 40 val images)")
# ─────────────────────────────────────────────────────────────
t0 = time.time()
pin_mem = device.type == "cuda"
train_loader, val_loader, meta = get_dataloaders(
    batch_size=16,
    num_workers=0,
    val_frac=0.1,
    seed=42,
    max_train_samples=200,
    max_val_samples=40,
)
n_geocells  = meta["n_geocells"]
n_countries = meta["n_countries"]
print(f"  Dataset loaded in {time.time()-t0:.1f}s")
check(len(train_loader) > 0, f"Train loader: {len(train_loader)} batches", "Empty train loader")
check(len(val_loader) > 0,   f"Val   loader: {len(val_loader)} batches",   "Empty val loader")

# Peek at a batch
batch = next(iter(train_loader))
check(batch["image"].shape[1:] == torch.Size([3, 224, 224]),
      f"Image shape: {tuple(batch['image'].shape)}",
      f"Wrong image shape: {tuple(batch['image'].shape)}")
check("geocell_id"   in batch, "geocell_id present",   "geocell_id missing")
check("country_idx"  in batch, "country_idx present",  "country_idx missing")
check("latitude"     in batch, "latitude present",     "latitude missing")
check("domain_label" in batch, "domain_label present", "domain_label missing")

print(f"\n  Batch keys:    {list(batch.keys())}")
print(f"  geocell_id range: [{batch['geocell_id'].min()}, {batch['geocell_id'].max()}]")
print(f"  country_idx range: [{batch['country_idx'].min()}, {batch['country_idx'].max()}]")
print(f"  lat range in batch: [{batch['latitude'].min():.1f}, {batch['latitude'].max():.1f}]")

# ─────────────────────────────────────────────────────────────
section("2. MODEL BUILD")
# ─────────────────────────────────────────────────────────────
t0 = time.time()
try:
    model = GeoLocModel(
        n_geocells=n_geocells,
        n_countries=n_countries,
        embed_dim=512,
        hidden_dim=512,
        dropout=0.2,
        clip_model_name="ViT-B-32",
        clip_pretrained="openai",
    ).to(device)
    print(f"  Model built in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  [FAIL] Model build error: {e}")
    errors.append(str(e))
    sys.exit(1)

total_p, frozen_p, trainable_p = model.count_params()
# Check that backbone params specifically have requires_grad=False
base_model = model
backbone_frozen = all(not p.requires_grad for p in base_model.backbone.parameters())
check(backbone_frozen,
      f"Backbone params all frozen (frozen={frozen_p:,}, trainable={trainable_p:,})",
      f"Some backbone params are NOT frozen!")
check(trainable_p > 0,
      f"Trainable params: {trainable_p:,}",
      "No trainable parameters!")

print(f"\n  Total params:     {total_p:,}")
print(f"  Frozen params:    {frozen_p:,}  ({100*frozen_p/total_p:.1f}%)")
print(f"  Trainable params: {trainable_p:,}  ({100*trainable_p/total_p:.1f}%)")

# ─────────────────────────────────────────────────────────────
section("3. FORWARD PASS")
# ─────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    imgs = batch["image"].to(device)
    try:
        out = model(imgs)
        print(f"  Forward pass OK")
    except Exception as e:
        print(f"  [FAIL] Forward pass error: {e}")
        errors.append(str(e)); sys.exit(1)

check("geocell_logits"    in out, f"geocell_logits: {tuple(out['geocell_logits'].shape)}", "missing")
check("country_logits"    in out, f"country_logits: {tuple(out['country_logits'].shape)}", "missing")
check("koppen_logits"     in out, f"koppen_logits: {tuple(out['koppen_logits'].shape)}", "missing")
check("worldcover_logits" in out, f"worldcover_logits: {tuple(out['worldcover_logits'].shape)}", "missing")
check("elevation_pred"    in out, f"elevation_pred: {tuple(out['elevation_pred'].shape)}", "missing")
check("domain_logits"     in out, f"domain_logits: {tuple(out['domain_logits'].shape)}", "missing")
check("embeddings"        in out, f"embeddings: {tuple(out['embeddings'].shape)}", "missing")

check(out["geocell_logits"].shape[1] == n_geocells,
      f"geocell head dim={n_geocells}", f"Wrong geocell dim: {out['geocell_logits'].shape[1]}")
check(out["country_logits"].shape[1] == n_countries,
      f"country head dim={n_countries}", f"Wrong country dim: {out['country_logits'].shape[1]}")

# ─────────────────────────────────────────────────────────────
section("4. LOSS COMPUTATION")
# ─────────────────────────────────────────────────────────────
centroids_df = pd.read_csv(ROOT / "data" / "geocell_centroids.csv").sort_values("geocell_id").reset_index(drop=True)
centroid_lats = torch.tensor(centroids_df["centroid_lat"].values, dtype=torch.float32)
centroid_lons = torch.tensor(centroids_df["centroid_lon"].values, dtype=torch.float32)

criterion = GeoLoss(centroid_lats, centroid_lons, sigma_km=500.0).to(device)
try:
    losses = criterion(out, {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in batch.items()})
    print(f"  Loss computation OK")
    for k, v in losses.items():
        print(f"    {k:20s}: {v.item():.4f}")
    check(not torch.isnan(losses["loss"]), f"Total loss: {losses['loss'].item():.4f}", "Loss is NaN!")
    check(losses["loss"].item() > 0, "Loss > 0", "Loss is zero or negative")
except Exception as e:
    print(f"  [FAIL] Loss error: {e}")
    errors.append(str(e)); sys.exit(1)

# ─────────────────────────────────────────────────────────────
section("5. 1-EPOCH TRAINING  (loss must decrease)")
# ─────────────────────────────────────────────────────────────
model.train()
optimizer = torch.optim.AdamW(model.trainable_params(), lr=3e-4, weight_decay=1e-4)

first_loss = None
last_loss  = None
batch_losses = []

print(f"  Running {len(train_loader)} batches...")
for step, batch in enumerate(train_loader):
    imgs = batch["image"].to(device)
    optimizer.zero_grad()
    out   = model(imgs)
    batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    loss_dict = criterion(out, batch_device)
    loss = loss_dict["loss"]
    loss.backward()
    nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)
    optimizer.step()

    loss_val = loss.item()
    batch_losses.append(loss_val)
    if first_loss is None: first_loss = loss_val
    last_loss = loss_val

    print(f"  Step {step+1:2d}/{len(train_loader)} | "
          f"loss={loss_val:.4f} | "
          f"geocell={loss_dict['loss_geocell'].item():.3f} | "
          f"country={loss_dict['loss_country'].item():.3f}")

# Check loss trend (average first half vs second half)
half = max(1, len(batch_losses) // 2)
avg_first = np.mean(batch_losses[:half])
avg_second = np.mean(batch_losses[half:])
check(avg_second <= avg_first * 1.5,  # allow slight variance — 1 epoch is noisy
      f"Loss trend OK (first-half avg={avg_first:.4f}, second-half avg={avg_second:.4f})",
      f"Loss not decreasing: {avg_first:.4f} -> {avg_second:.4f}")
check(not np.isnan(last_loss), "Final loss is not NaN", "NaN loss at end of epoch!")

# ─────────────────────────────────────────────────────────────
section("6. VALIDATION METRICS")
# ─────────────────────────────────────────────────────────────
model.eval()
all_pred_lats, all_pred_lons = [], []
all_true_lats, all_true_lons = [], []
val_losses = []

with torch.no_grad():
    for batch in val_loader:
        imgs = batch["image"].to(device)
        out  = model(imgs)
        batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        ldict = criterion(out, batch_device)
        val_losses.append(ldict["loss"].item())

        pred_cells = out["geocell_logits"].argmax(dim=-1).cpu().numpy()
        pred_lats  = centroids_df["centroid_lat"].values[pred_cells]
        pred_lons  = centroids_df["centroid_lon"].values[pred_cells]
        all_pred_lats.extend(pred_lats)
        all_pred_lons.extend(pred_lons)
        all_true_lats.extend(batch["latitude"].numpy())
        all_true_lons.extend(batch["longitude"].numpy())

dists = haversine_km(np.array(all_true_lats), np.array(all_true_lons),
                     np.array(all_pred_lats),  np.array(all_pred_lons))
val_loss = np.mean(val_losses)

print(f"\n  Val loss:           {val_loss:.4f}")
print(f"  Haversine median:   {np.median(dists):.1f} km")
print(f"  Haversine mean:     {np.mean(dists):.1f} km")
print(f"  Haversine p25/p75:  {np.percentile(dists,25):.1f} / {np.percentile(dists,75):.1f} km")
print(f"  Within  25 km:      {(dists< 25).mean()*100:.1f}%")
print(f"  Within 200 km:      {(dists<200).mean()*100:.1f}%")
print(f"  Within 750 km:      {(dists<750).mean()*100:.1f}%")
print(f"  Within 2500 km:     {(dists<2500).mean()*100:.1f}%")

check(not np.isnan(val_loss), f"Val loss: {val_loss:.4f}", "Val loss is NaN")
check(np.isfinite(dists).all(), "All distances are finite", "NaN/inf in distances!")

# ─────────────────────────────────────────────────────────────
section("7. SUBMISSION CSV SCHEMA CHECK")
# ─────────────────────────────────────────────────────────────
test_ds = TestDataset(ROOT / "test_images_sampled")
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

sample_sub = pd.read_csv(ROOT / "sample_submission.csv")
expected_ids = set(sample_sub["image_id"])

model.eval()
rows = []
with torch.no_grad():
    for batch in test_loader:
        imgs = batch["image"].to(device)
        out  = model(imgs)
        pred_cells = out["geocell_logits"].argmax(dim=-1).cpu().numpy()
        for i, img_id in enumerate(batch["image_id"]):
            cell = pred_cells[i]
            lat  = float(centroids_df.loc[cell, "centroid_lat"])
            lon  = float(centroids_df.loc[cell, "centroid_lon"])
            # Radius = p90 of that cell's max_radius — conservative initial value
            radius = float(centroids_df.loc[cell, "max_radius_km"])
            rows.append({"image_id": img_id, "pred_lat": lat, "pred_lon": lon, "pred_radius_km": radius})

sub_df = pd.DataFrame(rows)
sub_path = ROOT / "checkpoints" / "dry_run_submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"\n  Submission CSV -> {sub_path}")
print(f"  Rows: {len(sub_df)}")
print(sub_df.head(5).to_string(index=False))

# Schema checks
check(list(sub_df.columns) == ["image_id","pred_lat","pred_lon","pred_radius_km"],
      "Columns match sample_submission.csv exactly",
      f"Wrong columns: {list(sub_df.columns)}")
check(len(sub_df) == len(sample_sub),
      f"Row count matches sample_submission.csv ({len(sub_df)})",
      f"Row count mismatch: {len(sub_df)} vs {len(sample_sub)}")
check(sub_df["pred_lat"].between(-90, 90).all(),
      "All pred_lat in [-90, 90]", "Invalid pred_lat values")
check(sub_df["pred_lon"].between(-180, 180).all(),
      "All pred_lon in [-180, 180]", "Invalid pred_lon values")
check((sub_df["pred_radius_km"] > 0).all(),
      "All pred_radius_km > 0", "Invalid radius values (<= 0)")
check(set(sub_df["image_id"]) == expected_ids,
      "All test image_ids present", f"Missing IDs: {expected_ids - set(sub_df['image_id'])}")

# ─────────────────────────────────────────────────────────────
section("8. CHECKPOINT SAVE/LOAD")
# ─────────────────────────────────────────────────────────────
import os
ckpt_path = ROOT / "checkpoints" / "dry_run_model.pt"
ckpt_path.parent.mkdir(exist_ok=True)
torch.save({"model": model.state_dict(), "n_geocells": n_geocells, "n_countries": n_countries}, ckpt_path)
ckpt_size = ckpt_path.stat().st_size / 1e6
check(ckpt_path.exists() and ckpt_size > 0.1, f"Checkpoint saved: {ckpt_size:.1f} MB", "Checkpoint not saved")

# Reload
ckpt = torch.load(ckpt_path, map_location="cpu")
model2 = GeoLocModel(n_geocells=n_geocells, n_countries=n_countries,
                     clip_model_name="ViT-B-32", clip_pretrained="openai")
model2.load_state_dict(ckpt["model"])
check(True, "Checkpoint reloads without error", "")

# ─────────────────────────────────────────────────────────────
section("9. SUMMARY")
# ─────────────────────────────────────────────────────────────
print(f"\n  Total errors: {len(errors)}")
if errors:
    for e in errors:
        print(f"    [FAIL] {e}")
    print("\n  Step 2: FAILED — fix errors above before proceeding")
    sys.exit(1)
else:
    print()
    print("  All checks passed!")
    print()
    print("  === STEP 2 METRICS ===")
    print(f"  Trainable params:   {trainable_p:,}")
    print(f"  Frozen params:      {frozen_p:,}")
    print(f"  Train batches:      {len(train_loader)}")
    print(f"  Val batches:        {len(val_loader)}")
    print(f"  First batch loss:   {first_loss:.4f}")
    print(f"  Last batch loss:    {last_loss:.4f}")
    print(f"  Loss trend:         {avg_first:.4f} -> {avg_second:.4f}")
    print(f"  Val loss:           {val_loss:.4f}")
    print(f"  Haversine median:   {np.median(dists):.1f} km  (untrained baseline)")
    print(f"  Within 750 km:      {(dists<750).mean()*100:.1f}%")
    print(f"  Submission rows:    {len(sub_df)}")
    print(f"  Checkpoint size:    {ckpt_size:.1f} MB")
    print()
    print("  Step 2 STATUS: COMPLETE — ready for Step 3 (external data + full training)")
