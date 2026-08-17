"""
country_snap.py — Step 6, PRD Section 4 item 9

Post-processing module:
If the model's country head is confident in a country ISO, but the predicted
(lat, lon) coordinates fall slightly outside that country's polygon (e.g. coastal edge or near border),
nudge the point toward or inside the boundary to secure the country-match bonus.
"""

import json
from pathlib import Path
import numpy as np
from shapely.geometry import Point, shape
from shapely.ops import nearest_points

ROOT = Path(__file__).resolve().parent.parent
GEOJSON_PATH = ROOT / "country_boundaries.geojson"


class CountrySnapper:
    """
    Snaps predicted coordinates to nearest valid country territory if confident.
    """

    def __init__(self, geojson_path: Path = GEOJSON_PATH):
        print(f"[CountrySnapper] Loading country boundaries from {geojson_path}...")
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)

        self.polygons = {}
        for feat in gj["features"]:
            props = feat.get("properties", {})
            iso = props.get("ISO_A2") or props.get("iso_a2") or props.get("ISO") or props.get("ADM0_A3") or "UNK"
            iso = iso.upper()
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            self.polygons[iso] = geom

        print(f"[CountrySnapper] Loaded {len(self.polygons)} country geometries")

    def snap_coordinates(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        country_isos: list[str],
        country_confidences: np.ndarray,
        min_confidence: float = 0.50,
        max_snap_dist_deg: float = 1.0,  # ~110 km max nudge
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Nudges coordinates into matching country polygon if point is outside but nearby.
        """
        out_lats = lats.copy()
        out_lons = lons.copy()
        snapped_count = 0

        for i in range(len(lats)):
            iso = country_isos[i]
            conf = country_confidences[i]

            if iso not in self.polygons or iso in ["OCEAN", "-99", "UNK"]:
                continue

            if conf < min_confidence:
                continue

            poly = self.polygons[iso]
            pt = Point(lons[i], lats[i])

            if poly.contains(pt):
                # Already inside
                continue

            # Check distance to boundary
            dist_deg = poly.distance(pt)
            if dist_deg <= max_snap_dist_deg:
                # Find nearest point on/in the polygon
                nearest_geom_pt, _ = nearest_points(poly, pt)
                out_lons[i] = nearest_geom_pt.x
                out_lats[i] = nearest_geom_pt.y
                snapped_count += 1

        print(f"[CountrySnapper] Snapped {snapped_count}/{len(lats)} points into country boundaries")
        return out_lats, out_lons
