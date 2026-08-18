"""
knn_refine.py — Step 5, PRD Section 4 item 7

Refines coarse geocell predictions using nearest neighbor visual retrieval with
COUNTRY-MASKED candidate filtering — critical to prevent global retrieval
from pulling visually similar images from wrong continents.

Algorithm:
  1. Over-retrieve top-K * oversample_factor candidates from the full FAISS index.
  2. Filter candidates to those whose country_idx matches the model's top-1 or top-2
     predicted country (border-case tolerance), falling back to top-3 if < min_candidates found.
  3. Among filtered candidates, apply similarity threshold.
  4. Compute spherical geodesic weighted average of surviving neighbor coordinates.
  5. Spherically blend the kNN-derived point with the model's raw argmax centroid.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"

from geocells.build_geocells import spherical_weighted_average


class KNNRefiner:
    """
    Spatial refinement engine using country-masked visual embedding retrieval
    with spherical geodesic averaging.
    """

    def __init__(
        self,
        index_path: str = str(DATA_DIR / "faiss_index.bin"),
        metadata_path: str = str(DATA_DIR / "faiss_metadata.csv"),
    ):
        self.index = faiss.read_index(index_path)
        self.meta  = pd.read_csv(metadata_path)
        # Pre-build country -> row index mapping for fast masked lookup
        self._country_to_rows = (
            self.meta
            .reset_index()
            .groupby("country_idx")["index"]
            .apply(list)
            .to_dict()
        )
        print(
            f"[KNNRefiner] Loaded index with {self.index.ntotal} vectors "
            f"and {len(self.meta)} metadata rows "
            f"({len(self._country_to_rows)} distinct countries indexed)"
        )

    def refine(
        self,
        query_embeddings: np.ndarray,      # [B, dim] (L2-normalized)
        centroid_lats: np.ndarray,         # [B] coarse model prediction
        centroid_lons: np.ndarray,         # [B] coarse model prediction
        pred_country_idxs: np.ndarray,     # [B] top-1 predicted country index
        top2_country_idxs: np.ndarray = None,  # [B] top-2 predicted country index (optional)
        top3_country_idxs: np.ndarray = None,  # [B] top-3 predicted country index (optional)
        top_k: int = 10,
        oversample: int = 60,              # retrieve oversample candidates before country masking
        blend_weight: float = 0.30,        # spherical blend weight toward kNN (0=centroid only)
        min_similarity_threshold: float = 0.65,
        min_candidates: int = 2,           # minimum masked candidates to apply refinement
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Refines (lat, lon) using country-masked spherical geodesic kNN blending.

        Country masking prevents the core failure mode where a globally queried index
        pulls visually similar images from the wrong continent (e.g., forest roads in
        Germany matching forest roads in Turkey), causing error spikes of +2000 km.
        """
        query_embeddings = query_embeddings.astype(np.float32)

        # Over-retrieve for post-hoc country masking
        fetch_k = min(oversample, self.index.ntotal)
        similarities, indices = self.index.search(query_embeddings, fetch_k)

        refined_lats = centroid_lats.copy().astype(np.float64)
        refined_lons = centroid_lons.copy().astype(np.float64)

        n_refined   = 0  # count how many points were actually refined
        n_fallback  = 0  # count how many fell back to centroid

        for i in range(len(query_embeddings)):
            c_lat = float(centroid_lats[i])
            c_lon = float(centroid_lons[i])

            sims = similarities[i]
            idxs = indices[i]

            # Step 1: similarity threshold
            sim_mask = (sims >= min_similarity_threshold) & (idxs >= 0)
            if not np.any(sim_mask):
                n_fallback += 1
                continue

            valid_idxs = idxs[sim_mask]
            valid_sims = sims[sim_mask]

            # Step 2: Country mask — try progressively wider sets until we have
            # enough candidates (top1 -> top2 -> top3 -> no mask)
            allowed_countries = {int(pred_country_idxs[i])}
            if top2_country_idxs is not None:
                allowed_countries.add(int(top2_country_idxs[i]))

            # Filter by country
            meta_countries = self.meta.iloc[valid_idxs]["country_idx"].values
            country_mask = np.array([c in allowed_countries for c in meta_countries])

            if country_mask.sum() < min_candidates:
                # Widen to top-3
                if top3_country_idxs is not None:
                    allowed_countries.add(int(top3_country_idxs[i]))
                    country_mask = np.array([c in allowed_countries for c in meta_countries])

            if country_mask.sum() < min_candidates:
                # No country mask produced enough candidates — fall back to centroid
                # Do NOT blend with globally-retrieved neighbors (that's what broke things)
                n_fallback += 1
                continue

            masked_idxs = valid_idxs[country_mask]
            masked_sims = valid_sims[country_mask]

            # Take top-K from masked candidates
            if len(masked_idxs) > top_k:
                top_order = np.argsort(-masked_sims)[:top_k]
                masked_idxs = masked_idxs[top_order]
                masked_sims = masked_sims[top_order]

            neighbor_lats = self.meta.iloc[masked_idxs]["latitude"].values
            neighbor_lons = self.meta.iloc[masked_idxs]["longitude"].values

            # Softmax weights over similarity scores (sharply peaked)
            weights = np.exp((masked_sims - np.max(masked_sims)) * 10.0)
            weights /= weights.sum()

            # Spherical mean of country-masked nearest neighbors
            knn_lat, knn_lon = spherical_weighted_average(neighbor_lats, neighbor_lons, weights)

            # Spherical blend between model centroid and kNN point
            blend_lat, blend_lon = spherical_weighted_average(
                np.array([c_lat, knn_lat]),
                np.array([c_lon, knn_lon]),
                np.array([1.0 - blend_weight, blend_weight]),
            )

            refined_lats[i] = blend_lat
            refined_lons[i] = blend_lon
            n_refined += 1

        total = len(query_embeddings)
        print(
            f"[KNNRefiner] Refined {n_refined}/{total} points "
            f"({n_refined/total*100:.1f}% country-masked, "
            f"{n_fallback} fell back to raw centroid)"
        )
        return refined_lats, refined_lons
