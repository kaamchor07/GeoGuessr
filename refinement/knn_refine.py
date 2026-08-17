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


from geocells.build_geocells import spherical_weighted_average, latlon_to_xyz, xyz_to_latlon


class KNNRefiner:
    """
    Spatial refinement engine using visual embedding retrieval with spherical geodesic averaging.
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
        blend_weight: float = 0.25,         # conservative blend weight on sphere
        min_similarity_threshold: float = 0.70,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Refines (lat, lon) coordinates by spherical geodesic blending with top-K visual neighbors.
        """
        query_embeddings = query_embeddings.astype(np.float32)
        similarities, indices = self.index.search(query_embeddings, top_k)

        refined_lats = np.zeros(len(query_embeddings))
        refined_lons = np.zeros(len(query_embeddings))

        for i in range(len(query_embeddings)):
            c_lat = float(centroid_lats[i])
            c_lon = float(centroid_lons[i])

            sims = similarities[i]
            idxs = indices[i]

            # Filter by similarity threshold
            valid_mask = (sims >= min_similarity_threshold) & (idxs >= 0)
            if not np.any(valid_mask):
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

            # Spherical mean of nearest neighbors
            knn_lat, knn_lon = spherical_weighted_average(neighbor_lats, neighbor_lons, weights)

            # Spherical blend between model centroid and kNN point
            blend_lat, blend_lon = spherical_weighted_average(
                np.array([c_lat, knn_lat]),
                np.array([c_lon, knn_lon]),
                np.array([1.0 - blend_weight, blend_weight]),
            )

            refined_lats[i] = blend_lat
            refined_lons[i] = blend_lon

        return refined_lats, refined_lons

