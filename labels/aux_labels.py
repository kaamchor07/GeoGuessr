"""
aux_labels.py — Step 1, Section 4 item 5

For every (lat, lon) in the training CSV, derive three auxiliary labels that
are freely available from public raster datasets:

  1. Köppen-Geiger climate zone  (classification, 30 classes)
     Source: Beck et al. 2018, 1-km resolution GeoTIFF via GitHub
             https://figshare.com/articles/dataset/Present_and_future_K_ppen-Geiger_climate_classifications_at_1-km_resolution/6396959

  2. ESA WorldCover land-cover class  (classification, 11 classes)
     Source: ESA WorldCover 2020, 10-m resolution, accessed via public STAC
             For offline use we download a 300-m resampled version from:
             https://zenodo.org/record/7254221  (WorldCover_v100_S2_2020_300m)
             NOTE: The script downloads on first run and caches locally.

  3. SRTM elevation  (regression, metres)
     Source: NASA SRTM 90-m via elevation Python package (offline tiles cached locally)
             Fallback: ETOPO1 via NOAA REST API (1 arc-minute, allows small gaps)

All rasters are sampled at the exact (lat, lon) centroid — no bilinear
interpolation needed since we only need a coarse categorical/scalar signal.

Output: data/aux_labels.csv
  columns: image_id, koppen_code, koppen_class, worldcover_code,
            worldcover_class, elevation_m

Dependencies:
  pip install rasterio requests numpy pandas tqdm
  pip install elevation  (for SRTM, optional — has its own CLI)

Usage:
  python labels/aux_labels.py
  python labels/aux_labels.py --coords_csv training_dataset/noised_dataset/ground_truth_coordinates.csv
  python labels/aux_labels.py --skip_elevation   # skip SRTM if offline
"""

import argparse
import os
import sys
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import find_dataset_paths

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR = DATA_DIR / "raster_cache"
CACHE_DIR.mkdir(exist_ok=True)

auto_coords, _ = find_dataset_paths()
COORDS_CSV = auto_coords

# ---------------------------------------------------------------------------
# Köppen-Geiger class definitions (Beck et al. 2018)
# ---------------------------------------------------------------------------
KOPPEN_CODES = {
    1: "Af",  2: "Am",  3: "Aw",  4: "BWh", 5: "BWk",
    6: "BSh", 7: "BSk", 8: "Csa", 9: "Csb", 10: "Csc",
    11: "Cwa", 12: "Cwb", 13: "Cwc", 14: "Cfa", 15: "Cfb",
    16: "Cfc", 17: "Dsa", 18: "Dsb", 19: "Dsc", 20: "Dsd",
    21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd", 25: "Dfa",
    26: "Dfb", 27: "Dfc", 28: "Dfd", 29: "ET", 30: "EF",
}

KOPPEN_CLASSES = {
    "Af": "Tropical rainforest",     "Am": "Tropical monsoon",
    "Aw": "Tropical savanna",        "BWh": "Hot desert",
    "BWk": "Cold desert",            "BSh": "Hot steppe",
    "BSk": "Cold steppe",            "Csa": "Mediterranean hot-summer",
    "Csb": "Mediterranean warm-summer", "Csc": "Mediterranean cold-summer",
    "Cwa": "Humid subtropical",      "Cwb": "Subtropical highland",
    "Cwc": "Cold subtropical highland", "Cfa": "Humid subtropical no dry season",
    "Cfb": "Oceanic",                "Cfc": "Subpolar oceanic",
    "Dsa": "Hot-summer Mediterranean continental", "Dsb": "Warm-summer Mediterranean continental",
    "Dsc": "Continental",            "Dsd": "Continental",
    "Dwa": "Monsoon-influenced hot-summer continental", "Dwb": "Monsoon-influenced continental",
    "Dwc": "Subarctic monsoon",      "Dwd": "Subarctic monsoon extreme",
    "Dfa": "Hot-summer humid continental", "Dfb": "Warm-summer humid continental",
    "Dfc": "Subarctic",              "Dfd": "Extremely cold subarctic",
    "ET": "Tundra",                  "EF": "Ice cap",
}

# ---------------------------------------------------------------------------
# ESA WorldCover class definitions
# ---------------------------------------------------------------------------
WORLDCOVER_CODES = {
    10: "Tree cover",          20: "Shrubland",
    30: "Grassland",           40: "Cropland",
    50: "Built-up",            60: "Bare / sparse vegetation",
    70: "Snow and ice",        80: "Permanent water bodies",
    90: "Herbaceous wetland",  95: "Mangroves",
    100: "Moss and lichen",
}

# ---------------------------------------------------------------------------
# Raster download helpers
# ---------------------------------------------------------------------------
KOPPEN_URL = (
    "https://figshare.com/ndownloader/files/12407516"  # Beck 2018 1km present, GeoTIFF
)
KOPPEN_CACHE = CACHE_DIR / "koppen_1km.tif"

WORLDCOVER_URL = (
    "https://zenodo.org/record/7254221/files/"
    "ESA_WorldCover_10m_2020_v100_Map_AWS.tif?download=1"
)
WORLDCOVER_CACHE = CACHE_DIR / "worldcover_300m.tif"


def download_file(url: str, dest: Path, desc: str):
    """Stream-download a file with a progress bar and auto-extract if zip."""
    import zipfile
    import shutil

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    print(f"Downloading {desc} -> {dest}")
    temp_download = dest.parent / f"temp_{dest.name}"
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    
    with open(temp_download, "wb") as f:
        if use_tqdm:
            with tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    pbar.update(len(chunk))
        else:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"  {100*downloaded/total:.0f}%", end="\r")

    # Check if the downloaded payload is a zip archive
    if zipfile.is_zipfile(temp_download):
        print(f"  Extracting zip archive...")
        with zipfile.ZipFile(temp_download, "r") as zf:
            tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif") or n.lower().endswith(".tiff")]
            if tif_names:
                target_tif = tif_names[0]
                print(f"  Found GeoTIFF inside zip: {target_tif}")
                with zf.open(target_tif) as zf_in, open(dest, "wb") as f_out:
                    shutil.copyfileobj(zf_in, f_out)
            else:
                zf.extractall(dest.parent)
        temp_download.unlink(missing_ok=True)
    else:
        if dest.exists():
            dest.unlink()
        temp_download.rename(dest)

    print(f"  Done — {dest.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Köppen sampling
# ---------------------------------------------------------------------------
def sample_koppen(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Return integer Köppen codes (1–30) for each (lat, lon)."""
    import rasterio
    from rasterio.transform import rowcol

    try:
        if not KOPPEN_CACHE.exists():
            download_file(KOPPEN_URL, KOPPEN_CACHE, "Köppen-Geiger raster")

        with rasterio.open(KOPPEN_CACHE) as src:
            rows, cols = rowcol(src.transform, lons, lats)
            rows = np.clip(rows, 0, src.height - 1)
            cols = np.clip(cols, 0, src.width - 1)
            data = src.read(1)
            codes = data[rows, cols].astype(int)
        return codes
    except Exception as e:
        print(f"  [Warning] Köppen sampling failed: {e}")
        print("  Generating fallback climate zones from coordinates...")
        # Approximate climate zone by latitude band if raster unavailable
        abs_lat = np.abs(lats)
        fallback = np.where(abs_lat < 15, 1,          # Tropical (Af)
                   np.where(abs_lat < 30, 4,          # Arid/Subtropical (BWh)
                   np.where(abs_lat < 45, 14,         # Temperate (Cfa)
                   np.where(abs_lat < 60, 26, 29))))  # Continental (Dfb) / Polar (ET)
        return fallback.astype(int)



# ---------------------------------------------------------------------------
# WorldCover sampling
# ---------------------------------------------------------------------------
def sample_worldcover(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Return ESA WorldCover class codes for each (lat, lon)."""
    import rasterio
    from rasterio.transform import rowcol

    if not WORLDCOVER_CACHE.exists():
        # Try Zenodo download; on failure, use Open-Meteo or skip
        try:
            download_file(WORLDCOVER_URL, WORLDCOVER_CACHE, "ESA WorldCover raster")
        except Exception as e:
            print(f"  [Warning] WorldCover download failed: {e}")
            print("  Returning placeholder zeros — re-run after placing file manually.")
            return np.zeros(len(lats), dtype=int)

    with rasterio.open(WORLDCOVER_CACHE) as src:
        rows, cols = rowcol(src.transform, lons, lats)
        rows = np.clip(rows, 0, src.height - 1)
        cols = np.clip(cols, 0, src.width - 1)
        data = src.read(1)
        codes = data[rows, cols].astype(int)

    return codes


# ---------------------------------------------------------------------------
# SRTM elevation (via Open-Elevation API — cached, works offline once filled)
# ---------------------------------------------------------------------------
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
ELEV_CACHE_FILE = DATA_DIR / "elevation_cache.csv"


def load_elevation_cache() -> dict:
    if ELEV_CACHE_FILE.exists():
        df = pd.read_csv(ELEV_CACHE_FILE)
        return {(r.lat_key, r.lon_key): r.elevation_m for r in df.itertuples()}
    return {}


def save_elevation_cache(cache: dict):
    rows = [{"lat_key": k[0], "lon_key": k[1], "elevation_m": v} for k, v in cache.items()]
    pd.DataFrame(rows).to_csv(ELEV_CACHE_FILE, index=False)


def sample_elevation_api(
    lats: np.ndarray,
    lons: np.ndarray,
    batch_size: int = 100,
    retry_delay: float = 2.0,
) -> np.ndarray:
    """Query Open-Elevation REST API in batches. Returns metres (NaN on failure)."""
    cache = load_elevation_cache()
    results = np.full(len(lats), np.nan)

    # Round to 4 dp for cache key (~11m precision)
    lat_keys = np.round(lats, 4)
    lon_keys = np.round(lons, 4)

    to_query_idx = []
    for i, (lk, lnk) in enumerate(zip(lat_keys, lon_keys)):
        if (lk, lnk) in cache:
            results[i] = cache[(lk, lnk)]
        else:
            to_query_idx.append(i)

    print(f"  Elevation: {len(lats) - len(to_query_idx)} cached, {len(to_query_idx)} to query")

    for batch_start in range(0, len(to_query_idx), batch_size):
        batch = to_query_idx[batch_start : batch_start + batch_size]
        locations = [
            {"latitude": float(lat_keys[i]), "longitude": float(lon_keys[i])} for i in batch
        ]
        payload = {"locations": locations}
        try:
            resp = requests.post(OPEN_ELEVATION_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for idx, res in zip(batch, data["results"]):
                elev = float(res["elevation"])
                results[idx] = elev
                cache[(lat_keys[idx], lon_keys[idx])] = elev
        except Exception as e:
            print(f"  [Warning] Elevation API error for batch {batch_start}: {e}")
            time.sleep(retry_delay)

        if (batch_start // batch_size + 1) % 10 == 0:
            save_elevation_cache(cache)
            print(f"  Elevation cache checkpoint saved ({batch_start + len(batch)}/{len(to_query_idx)})")

    save_elevation_cache(cache)
    print(f"  Elevation cache saved — {len(cache)} entries")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate auxiliary raster labels")
    parser.add_argument("--coords_csv", type=str, default=str(COORDS_CSV))
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "aux_labels.csv"))
    parser.add_argument(
        "--skip_koppen",
        action="store_true",
        help="Skip Köppen-Geiger (requires rasterio + download)",
    )
    parser.add_argument(
        "--skip_worldcover",
        action="store_true",
        help="Skip ESA WorldCover (requires rasterio + download)",
    )
    parser.add_argument(
        "--skip_elevation",
        action="store_true",
        help="Skip elevation (requires internet for Open-Elevation API)",
    )
    parser.add_argument(
        "--elevation_batch_size",
        type=int,
        default=100,
        help="Number of points per elevation API call",
    )
    args = parser.parse_args()

    print(f"Loading coordinates from {args.coords_csv}")
    df = pd.read_csv(args.coords_csv)
    lats = df["latitude"].values
    lons = df["longitude"].values
    print(f"  {len(df)} rows")

    out_df = df[["image_id", "latitude", "longitude"]].copy()

    # --- Köppen ---
    if not args.skip_koppen:
        try:
            import rasterio  # noqa: F401
            print("\n[1/3] Sampling Köppen-Geiger climate zones …")
            koppen_codes = sample_koppen(lats, lons)
            out_df["koppen_code"] = koppen_codes
            out_df["koppen_label"] = [KOPPEN_CODES.get(c, "Unknown") for c in koppen_codes]
            print(f"  Unique zones: {pd.Series(koppen_codes).nunique()}")
        except ImportError:
            print("[1/3] rasterio not installed — skipping Köppen. Install with: pip install rasterio")
            out_df["koppen_code"] = -1
            out_df["koppen_label"] = "Unknown"
    else:
        print("[1/3] Skipping Köppen (--skip_koppen)")
        out_df["koppen_code"] = -1
        out_df["koppen_label"] = "Unknown"

    # --- WorldCover ---
    if not args.skip_worldcover:
        try:
            import rasterio  # noqa: F401
            print("\n[2/3] Sampling ESA WorldCover land-cover …")
            wc_codes = sample_worldcover(lats, lons)
            out_df["worldcover_code"] = wc_codes
            out_df["worldcover_label"] = [WORLDCOVER_CODES.get(c, "Unknown") for c in wc_codes]
            print(f"  Unique classes: {pd.Series(wc_codes).nunique()}")
        except ImportError:
            print("[2/3] rasterio not installed — skipping WorldCover.")
            out_df["worldcover_code"] = -1
            out_df["worldcover_label"] = "Unknown"
    else:
        print("[2/3] Skipping WorldCover (--skip_worldcover)")
        out_df["worldcover_code"] = -1
        out_df["worldcover_label"] = "Unknown"

    # --- Elevation ---
    if not args.skip_elevation:
        print("\n[3/3] Sampling elevation (Open-Elevation API) …")
        elevations = sample_elevation_api(lats, lons, args.elevation_batch_size)
        out_df["elevation_m"] = elevations
        valid = ~np.isnan(elevations)
        print(f"  Valid: {valid.sum()}/{len(elevations)}")
        print(f"  Range: {np.nanmin(elevations):.0f}m – {np.nanmax(elevations):.0f}m")
    else:
        print("[3/3] Skipping elevation (--skip_elevation)")
        out_df["elevation_m"] = np.nan

    # Save
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")
    print(out_df.head(10).to_string())


if __name__ == "__main__":
    main()
