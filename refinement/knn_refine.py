"""
knn_refine.py — Step 5, PRD Section 4 item 7

Refines coarse geocell predictions using nearest neighbor visual retrieval:
  1. Retrieve top-K most visually similar training images from FAISS index.
  2. Filter or weight neighbors belonging to the top-N predicted geocells / countries.
  3. Compute weighted spatial average of neighbor coordinates to produce refined (lat, lon).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import faiss

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class KNNRefiner:
    """
    Spatial refinement engine using visual embedding retrieval.
    """

    def __init__(
        self,
        index_path: str = str(DATA_DIR / "faiss_index.bin"),
        metadata_path: str = str(DATA_DIR / "faiss_metadata.csv"),
    ):
        self.index = faiss.read_index(index_path)
        self.meta = pd.read_csv(metadata_path)
        print(f"[KNNRefiner] Loaded index with {self.index.ntotal} vectors and {len(self.meta)} metadata rows")

    def refine(
        self,
        query_embeddings: np.ndarray,      # [B, dim] (L2-normalized)
        centroid_lats: np.ndarray,         # [B] coarse model prediction
        centroid_lons: np.ndarray,         # [B] coarse model prediction
        top_k: int = 5,
        blend_weight: float = 0.4,          # weight of kNN vs model centroid (0=centroid only, 1=kNN only)
        min_similarity_threshold: float = 0.65,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Refines (lat, lon) coordinates by blending with top-K visual neighbors.
        """
        query_embeddings = query_embeddings.astype(np.float32)
        # Query FAISS
        similarities, indices = self.index.search(query_embeddings, top_k)

        refined_lats = np.zeros(len(query_embeddings))
        refined_lons = np.zeros(len(query_embeddings))

        for i in range(len(query_embeddings)):
            c_lat = centroid_lats[i]
            c_lon = centroid_lons[i]

            sims = similarities[i]
            idxs = indices[i]

            # Filter by similarity threshold
            valid_mask = (sims >= min_similarity_threshold) & (idxs >= 0)
            if not np.any(valid_mask):
                # Fallback to coarse centroid
                refined_lats[i] = c_lat
                refined_lons[i] = c_lon
                continue

            valid_idxs = idxs[valid_mask]
            valid_sims = sims[valid_mask]

            neighbor_lats = self.meta.iloc[valid_idxs]["latitude"].values
            neighbor_lons = self.meta.iloc[valid_idxs]["longitude"].values

            # Softmax weights over similarity scores
            weights = np.exp((valid_sims - np.max(valid_sims)) * 10.0)
            weights /= weights.sum()

            knn_lat = np.sum(neighbor_lats * weights)
            knn_lon = np.sum(neighbor_lons * weights)

            # Blend
            refined_lats[i] = (1.0 - blend_weight) * c_lat + blend_weight * knn_lat
            refined_lons[i] = (1.0 - blend_weight) * c_lon + blend_weight * knn_lon

        return refined_lats, refined_lons
