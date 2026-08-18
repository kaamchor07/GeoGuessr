"""
train.py — Step 2+4, PRD Section 4 + Section 7

Full training loop with:
  - DataParallel for 2x T4 (Kaggle) or single GPU
  - Fixed seeds everywhere
  - Checkpoint saving every N steps + best model tracking
  - GRL alpha annealing schedule
  - SWA (Stochastic Weight Averaging) for snapshot ensembling
  - Validation loop computing haversine accuracy metrics

Usage:
  # Dry run (200 images, 1 epoch, CPU/small GPU):
  python training/train.py --dry_run --max_samples 200 --epochs 1

  # Full training (Kaggle 2x T4):
  python training/train.py --epochs 20 --batch_size 64 --num_workers 4

  # Resume from checkpoint:
  python training/train.py --resume checkpoints/last.pt
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import get_dataloaders
from models.model import GeoLocModel, GeoLoss

CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Seed everything
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# GRL alpha annealing (p = step / total_steps, 0→1)
# ---------------------------------------------------------------------------
def grl_alpha(p: float, gamma: float = 10.0) -> float:
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


# ---------------------------------------------------------------------------
# Haversine distance (km) — for validation metrics
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, val_loader, criterion, device, centroids_df):
    model.eval()
    total_loss = 0.0
    all_pred_lats, all_pred_lons = [], []
    all_true_lats, all_true_lons = [], []
    n_batches = 0

    all_true_cells, all_pred_cells = [], []
    all_top5_cells = []
    all_true_countries, all_pred_countries = [], []

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        outputs = model(images) if not isinstance(model, nn.DataParallel) else model(images)

        losses = criterion(outputs, batch)
        total_loss += losses["loss"].item()
        n_batches += 1

        # Geocell predictions (Top-1 and Top-5)
        gc_logits = outputs["geocell_logits"]
        pred_cells = gc_logits.argmax(dim=-1).cpu().numpy()
        top5_cells = torch.topk(gc_logits, k=min(5, gc_logits.shape[-1]), dim=-1).indices.cpu().numpy()

        # Country predictions
        ct_logits = outputs["country_logits"]
        pred_countries = ct_logits.argmax(dim=-1).cpu().numpy()

        pred_lats = centroids_df["centroid_lat"].values[pred_cells]
        pred_lons = centroids_df["centroid_lon"].values[pred_cells]

        all_pred_lats.extend(pred_lats)
        all_pred_lons.extend(pred_lons)
        all_true_lats.extend(batch["latitude"].numpy())
        all_true_lons.extend(batch["longitude"].numpy())

        all_true_cells.extend(batch["geocell_id"].numpy())
        all_pred_cells.extend(pred_cells)
        all_top5_cells.extend(top5_cells)
        all_true_countries.extend(batch["country_idx"].numpy())
        all_pred_countries.extend(pred_countries)

    # Accuracy metrics
    all_true_cells = np.array(all_true_cells)
    all_pred_cells = np.array(all_pred_cells)
    all_top5_cells = np.array(all_top5_cells)
    all_true_countries = np.array(all_true_countries)
    all_pred_countries = np.array(all_pred_countries)

    geocell_top1 = float((all_true_cells == all_pred_cells).mean())
    geocell_top5 = float(np.mean([all_true_cells[i] in all_top5_cells[i] for i in range(len(all_true_cells))]))
    country_top1 = float((all_true_countries == all_pred_countries).mean())

    # Haversine distances
    all_pred_lats = np.array(all_pred_lats)
    all_pred_lons = np.array(all_pred_lons)
    all_true_lats = np.array(all_true_lats)
    all_true_lons = np.array(all_true_lons)
    dists = haversine_km(all_true_lats, all_true_lons, all_pred_lats, all_pred_lons)

    metrics = {
        "val_loss":          total_loss / max(n_batches, 1),
        "country_top1":      country_top1,
        "geocell_top1":      geocell_top1,
        "geocell_top5":      geocell_top5,
        "haversine_median":  float(np.median(dists)),
        "haversine_mean":    float(np.mean(dists)),
        "haversine_p25":     float(np.percentile(dists, 25)),
        "haversine_p75":     float(np.percentile(dists, 75)),
        "within_25km":       float((dists < 25).mean()),
        "within_200km":      float((dists < 200).mean()),
        "within_750km":      float((dists < 750).mean()),
    }
    model.train()
    return metrics



# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Checkpoint output directory
    ckpt_dir = CKPT_DIR if not args.run_name else (CKPT_DIR / args.run_name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints directory: {ckpt_dir}")


    # --- Data ---
    print("\nLoading datasets...")
    max_train = args.max_samples if args.dry_run else None
    max_val   = max(args.max_samples // 5, 20) if args.dry_run else None

    train_loader, val_loader, meta = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        seed=args.seed,
        max_train_samples=max_train,
        max_val_samples=max_val,
        images_dir=args.images_dir,
        coords_csv=args.coords_csv,
        use_osv5m=args.use_osv5m,
        osv5m_meta_csv=args.osv5m_meta_csv,
        osv5m_images_dir=args.osv5m_images_dir,
    )
    n_geocells  = meta["n_geocells"]
    n_countries = meta["n_countries"]

    # Load geocell centroids (needed for val metrics + loss)
    centroids_df = pd.read_csv(ROOT / "data" / "geocell_centroids.csv").sort_values("geocell_id").reset_index(drop=True)
    centroid_lats = torch.tensor(centroids_df["centroid_lat"].values, dtype=torch.float32)
    centroid_lons = torch.tensor(centroids_df["centroid_lon"].values, dtype=torch.float32)

    # --- Model ---
    print("\nBuilding model...")
    model = GeoLocModel(
        n_geocells=n_geocells,
        n_countries=n_countries,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )

    total_p, frozen_p, trainable_p = model.count_params()
    print(f"Parameters — total: {total_p:,}  frozen: {frozen_p:,}  trainable: {trainable_p:,}")

    # DataParallel on multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # --- Loss ---
    criterion = GeoLoss(
        centroid_lats=centroid_lats,
        centroid_lons=centroid_lons,
        sigma_km=args.sigma_km,
        w_geocell=args.w_geocell,
        w_country=args.w_country,
        w_koppen=args.w_koppen,
        w_worldcover=args.w_worldcover,
        w_elevation=args.w_elevation,
        w_domain=args.w_domain,
    ).to(device)

    # --- Optimizer (only trainable params) ---
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = torch.optim.AdamW(
        base_model.trainable_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=args.lr * 0.01
    )

    # --- Mixed precision ---
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    history = []

    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        global_step  = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history      = ckpt.get("history", [])
        print(f"  Resumed at epoch {start_epoch}, step {global_step}")

    # --- Training ---
    total_steps = args.epochs * len(train_loader)
    print(f"\nTraining: {args.epochs} epochs x {len(train_loader)} batches = {total_steps} steps")
    print(f"Validation: {len(val_loader)} batches")
    print()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            p = global_step / max(total_steps, 1)
            alpha = grl_alpha(p)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images, grl_alpha=alpha)
                    losses  = criterion(outputs, batch)
                    loss    = losses["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(base_model.trainable_params(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images, grl_alpha=alpha)
                losses  = criterion(outputs, batch)
                loss    = losses["loss"]
                loss.backward()
                nn.utils.clip_grad_norm_(base_model.trainable_params(), 1.0)
                optimizer.step()

            scheduler.step()
            global_step += 1
            epoch_losses.append(loss.item())

            # Log every 10 steps
            if (step + 1) % max(1, min(10, len(train_loader) // 5)) == 0 or step == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  Ep {epoch+1}/{args.epochs} | Step {step+1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} "
                    f"(gc={losses['loss_geocell'].item():.3f} "
                    f"co={losses['loss_country'].item():.3f}) | "
                    f"LR: {lr_now:.2e} | GRL-alpha: {alpha:.3f}"
                )

            # Checkpoint every N steps
            if args.ckpt_every_steps > 0 and global_step % args.ckpt_every_steps == 0:
                _save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                 best_val_loss, history, ckpt_dir / "last.pt")

        # --- End of epoch ---
        epoch_train_loss = np.mean(epoch_losses)
        val_metrics = validate(model, val_loader, criterion, device, centroids_df)
        epoch_time = time.time() - epoch_start

        print(f"\n[Epoch {epoch+1}] "
              f"train_loss={epoch_train_loss:.4f} | "
              f"val_loss={val_metrics['val_loss']:.4f} | "
              f"country_acc={val_metrics['country_top1']*100:.1f}% | "
              f"geocell_top1={val_metrics['geocell_top1']*100:.1f}% | "
              f"geocell_top5={val_metrics['geocell_top5']*100:.1f}% | "
              f"median_dist={val_metrics['haversine_median']:.1f}km | "
              f"within_200km={val_metrics['within_200km']*100:.1f}% | "
              f"time={epoch_time:.0f}s\n")


        record = {"epoch": epoch + 1, "train_loss": epoch_train_loss, **val_metrics}
        history.append(record)

        is_best = val_metrics["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["val_loss"]
            _save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                             best_val_loss, history, ckpt_dir / "best.pt")
            # Also keep root checkpoints/best.pt updated as latest
            if ckpt_dir != CKPT_DIR:
                _save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                 best_val_loss, history, CKPT_DIR / "best.pt")
            print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")

        _save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                         best_val_loss, history, ckpt_dir / "last.pt")

    # Save training history
    hist_path = ckpt_dir / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to {hist_path}")

    # Plot metrics
    plot_training_curves(history, ckpt_dir / "training_metrics.png")


    # --- Final summary ---
    if history:
        best = min(history, key=lambda r: r["val_loss"])
        print(f"\n=== Best epoch: {best['epoch']} ===")
        for k, v in best.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")


def plot_training_curves(history: list, save_path: Path):
    """Generates visual plots for Loss, Haversine Distance, and Accuracy."""
    try:
        import matplotlib.pyplot as plt
        epochs = [r["epoch"] for r in history]
        train_losses = [r["train_loss"] for r in history]
        val_losses = [r["val_loss"] for r in history]
        median_dists = [r["haversine_median"] for r in history]
        acc_200 = [r["within_200km"] * 100 for r in history]
        acc_750 = [r["within_750km"] * 100 for r in history]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Loss
        axes[0].plot(epochs, train_losses, label="Train Loss", marker="o", color="#3b82f6")
        axes[0].plot(epochs, val_losses, label="Val Loss", marker="s", color="#ef4444")
        axes[0].set_title("Multi-Task Loss Progression", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend()

        # 2. Haversine Median Distance
        axes[1].plot(epochs, median_dists, label="Median Error (km)", marker="^", color="#10b981")
        axes[1].set_title("Validation Haversine Error (Median km)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Distance (km)")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend()

        # 3. Accuracy thresholds
        axes[2].plot(epochs, acc_200, label="Within 200 km (%)", marker="o", color="#8b5cf6")
        axes[2].plot(epochs, acc_750, label="Within 750 km (%)", marker="d", color="#f59e0b")
        axes[2].set_title("Prediction Accuracy @ Radius", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Accuracy (%)")
        axes[2].grid(True, linestyle="--", alpha=0.6)
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[Visualizer] Training curves plot saved -> {save_path}")
    except Exception as e:
        print(f"[Visualizer] Plotting skipped: {e}")



def _save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_val_loss, history, path):
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "model":         base_model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "epoch":         epoch,
        "global_step":   global_step,
        "best_val_loss": best_val_loss,
        "history":       history,
    }, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_parser():
    p = argparse.ArgumentParser()
    # Mode
    p.add_argument("--run_name",   type=str, default=None, help="Distinct subfolder name under checkpoints/")
    p.add_argument("--dry_run",    action="store_true", help="Quick 1-epoch test on 200 images")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--resume",     type=str, default=None)

    # Data
    p.add_argument("--val_frac",   type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers",type=int, default=0)
    p.add_argument("--coords_csv", type=str, default=None, help="Path to ground_truth_coordinates.csv")
    p.add_argument("--images_dir", type=str, default=None, help="Path to images directory")
    p.add_argument("--use_osv5m",  action="store_true", help="Merge OSV5M external dataset into training")
    p.add_argument("--osv5m_meta_csv", type=str, default=None, help="Path to osv5m_train.csv")
    p.add_argument("--osv5m_images_dir", type=str, default=None, help="Path to osv5m images directory")

    # Model
    p.add_argument("--clip_model",     type=str, default="ViT-B-32")
    p.add_argument("--clip_pretrained",type=str, default="openai")
    p.add_argument("--embed_dim",  type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout",    type=float, default=0.2)

    # Loss
    p.add_argument("--sigma_km",   type=float, default=500.0)
    p.add_argument("--w_geocell",  type=float, default=1.0)
    p.add_argument("--w_country",  type=float, default=0.5)
    p.add_argument("--w_koppen",   type=float, default=0.2)
    p.add_argument("--w_worldcover", type=float, default=0.2)
    p.add_argument("--w_elevation", type=float, default=0.1)
    p.add_argument("--w_domain",   type=float, default=0.3)
    # Training
    p.add_argument("--epochs",     type=int, default=1)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--amp",        action="store_true", help="Mixed precision (CUDA only)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--ckpt_every_steps", type=int, default=100)
    return p



if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.dry_run:
        print("=" * 60)
        print("  DRY RUN MODE: 200 images, 1 epoch")
        print("=" * 60)
    train(args)
