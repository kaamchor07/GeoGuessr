"""
build_index.py — Step 5, PRD Section 4 item 7

Extracts embeddings for all training images using the frozen backbone
and stores a FAISS index with coordinates for kNN spatial refinement at inference time.

Outputs:
  data/faiss_index.bin
  data/faiss_metadata.parquet (or .csv)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import GeoDataset
from models.model import GeoLocModel
import faiss

DATA_DIR = ROOT / "data"


def build_faiss_index(
    model_checkpoint: str = None,
    batch_size: int = 64,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_index_path: str = str(DATA_DIR / "faiss_index.bin"),
    output_meta_path: str = str(DATA_DIR / "faiss_metadata.csv"),
):
    device = torch.device(device_str)
    print(f"[FAISS Indexer] Device: {device}")

    # Load dataset (all training samples without augmentation)
    ds = GeoDataset(split="train", val_frac=0.0, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2 if device.type == "cuda" else 0)

    # Initialize model
    model = GeoLocModel(
        n_geocells=ds.n_geocells,
        n_countries=ds.n_countries,
        clip_model_name="ViT-B-32",
        clip_pretrained="openai",
    )

    if model_checkpoint and Path(model_checkpoint).exists():
        print(f"[FAISS Indexer] Loading weights from {model_checkpoint}")
        ckpt = torch.load(model_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt))

    model = model.to(device)
    model.eval()

    all_embeddings = []
    all_image_ids = []
    all_lats = []
    all_lons = []
    all_geocells = []
    all_countries = []

    print("[FAISS Indexer] Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(loader):
            imgs = batch["image"].to(device)
            # Use the frozen backbone embedding
            embeds = model.encode_image(imgs)
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=-1)
            
            all_embeddings.append(embeds.cpu().numpy())
            all_image_ids.extend(batch["image_id"])
            all_lats.extend(batch["latitude"].numpy())
            all_lons.extend(batch["longitude"].numpy())
            all_geocells.extend(batch["geocell_id"].numpy())
            all_countries.extend(batch["country_idx"].numpy())

    embeddings_np = np.vstack(all_embeddings).astype(np.float32)
    dim = embeddings_np.shape[1]
    print(f"[FAISS Indexer] Embeddings shape: {embeddings_np.shape} (dim={dim})")

    # Build IndexFlatIP (Inner Product = Cosine Similarity since normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_np)
    print(f"[FAISS Indexer] Total indexed vectors: {index.ntotal}")

    # Save FAISS index
    faiss.write_index(index, output_index_path)
    print(f"[FAISS Indexer] Saved FAISS index -> {output_index_path}")

    # Save metadata
    meta_df = pd.DataFrame({
        "image_id": all_image_ids,
        "latitude": all_lats,
        "longitude": all_lons,
        "geocell_id": all_geocells,
        "country_idx": all_countries,
    })
    meta_df.to_csv(output_meta_path, index=False)
    print(f"[FAISS Indexer] Saved metadata -> {output_meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    build_faiss_index(model_checkpoint=args.checkpoint, batch_size=args.batch_size)
