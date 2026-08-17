"""
test_step1.py — Full validation of all Step 1 outputs
Run: python test_step1.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

errors = []

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check(condition, msg_pass, msg_fail):
    if condition:
        print(f"  {PASS} {msg_pass}")
    else:
        print(f"  {FAIL} {msg_fail}")
        errors.append(msg_fail)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# ============================================================
section("1. FILE EXISTENCE")
# ============================================================
required_files = {
    "ground_truth_coordinates.csv": ROOT/"training_dataset"/"noised_dataset"/"ground_truth_coordinates.csv",
    "country_labels.csv":           DATA_DIR/"country_labels.csv",
    "country_encoder.csv":          DATA_DIR/"country_encoder.csv",
    "geocell_assignments.csv":      DATA_DIR/"geocell_assignments.csv",
    "geocell_centroids.csv":        DATA_DIR/"geocell_centroids.csv",
    "audit_report.csv":             DATA_DIR/"audit_report_v2.csv",
}
for name, path in required_files.items():
    check(path.exists(), f"{name} exists ({path.stat().st_size/1024:.1f} KB)" if path.exists() else "", f"{name} MISSING at {path}")

# ============================================================
section("2. GROUND TRUTH COORDINATES")
# ============================================================
coords = pd.read_csv(ROOT/"training_dataset"/"noised_dataset"/"ground_truth_coordinates.csv")
check(len(coords) > 0, f"Loaded {len(coords)} rows", "Empty CSV")
check(set(coords.columns) >= {"image_id","latitude","longitude"},
      "Has required columns", f"Missing columns: {set(coords.columns)}")
check(coords["latitude"].between(-90, 90).all(), "All latitudes in [-90, 90]", "Invalid latitudes")
check(coords["longitude"].between(-180, 180).all(), "All longitudes in [-180, 180]", "Invalid longitudes")
check(coords["image_id"].nunique() == len(coords), "No duplicate image IDs", "Duplicate image IDs found")
print(f"\n  Coordinate ranges:")
print(f"    Lat:  [{coords.latitude.min():.2f}, {coords.latitude.max():.2f}]")
print(f"    Lon:  [{coords.longitude.min():.2f}, {coords.longitude.max():.2f}]")

# ============================================================
section("3. COUNTRY LABELS")
# ============================================================
cl = pd.read_csv(DATA_DIR/"country_labels.csv")
check(len(cl) == len(coords), f"Row count matches coords ({len(cl)})", f"Row count mismatch: {len(cl)} vs {len(coords)}")
check(set(cl.columns) >= {"image_id","country_iso"}, "Has required columns", f"Missing columns")
check(cl["image_id"].nunique() == len(cl), "No duplicate image IDs", "Duplicates found")

n_countries = cl["country_iso"].nunique()
n_ocean     = (cl["country_iso"] == "OCEAN").sum()
n_unknown   = (cl["country_iso"] == "-99").sum()
n_labeled   = len(cl) - n_ocean - n_unknown

check(n_countries >= 100, f"{n_countries} unique country codes", f"Too few countries: {n_countries}")
print(f"\n  Labeled (land):  {n_labeled:>6} ({100*n_labeled/len(cl):.1f}%)")
print(f"  Ocean:           {n_ocean:>6} ({100*n_ocean/len(cl):.1f}%)")
print(f"  No-ISO (-99):    {n_unknown:>6} ({100*n_unknown/len(cl):.1f}%)")
print(f"  Unique countries: {n_countries}")
print(f"\n  Top 15 countries:")
top = cl["country_iso"].value_counts().head(15)
for iso, cnt in top.items():
    name_row = cl[cl["country_iso"]==iso]["country_name"].mode()
    name = name_row.iloc[0] if len(name_row) else iso
    print(f"    {iso:6s} {cnt:>5}  {name}")

# ============================================================
section("4. COUNTRY ENCODER")
# ============================================================
enc = pd.read_csv(DATA_DIR/"country_encoder.csv")
check(set(enc.columns) >= {"country_iso","country_idx"}, "Has required columns", "Missing columns")
n_classes = len(enc)
check(n_classes >= 100, f"{n_classes} classes in encoder", f"Too few classes: {n_classes}")
check((enc["country_idx"] == range(n_classes)).all(), "Indices are 0-contiguous", "Non-contiguous indices")
all_isos = set(cl["country_iso"].dropna().astype(str).unique())
enc_isos = set(enc["country_iso"].astype(str).unique())
check(all_isos == enc_isos, "Encoder covers all country ISOs in labels", 
      f"Missing from encoder: {all_isos - enc_isos}")
print(f"  Country classes: {n_classes}")

# ============================================================
section("5. GEOCELL ASSIGNMENTS")
# ============================================================
ga = pd.read_csv(DATA_DIR/"geocell_assignments.csv")
check(len(ga) == len(coords), f"Row count matches coords ({len(ga)})", f"Row mismatch: {len(ga)}")
check("geocell_id" in ga.columns, "Has geocell_id column", "Missing geocell_id")
n_geocells = ga["geocell_id"].nunique()
check(n_geocells >= 500, f"{n_geocells} unique geocells", f"Too few geocells: {n_geocells}")

gcounts = ga["geocell_id"].value_counts()
print(f"\n  Geocell count:   {n_geocells}")
print(f"  Images/cell:     min={gcounts.min()}  median={gcounts.median():.0f}  mean={gcounts.mean():.1f}  max={gcounts.max()}")
singleton_cells = (gcounts == 1).sum()
print(f"  Singleton cells: {singleton_cells}")
check(singleton_cells < n_geocells * 0.05, 
      f"<5% singleton cells ({singleton_cells})", 
      f"Too many singletons: {singleton_cells}")

# ============================================================
section("6. GEOCELL CENTROIDS")
# ============================================================
gc = pd.read_csv(DATA_DIR/"geocell_centroids.csv")
check(len(gc) == n_geocells, f"Centroid count matches geocells ({len(gc)})", f"Mismatch: {len(gc)} vs {n_geocells}")
check(set(gc.columns) >= {"geocell_id","centroid_lat","centroid_lon","count","max_radius_km"},
      "Has all required columns", f"Columns: {list(gc.columns)}")
check(gc["centroid_lat"].between(-90, 90).all(), "All centroid lats valid", "Invalid centroid lats")
check(gc["centroid_lon"].between(-180, 180).all(), "All centroid lons valid", "Invalid centroid lons")

r = gc["max_radius_km"]
print(f"\n  Max-radius per cell (km):")
print(f"    min={r.min():.1f}  p25={r.quantile(.25):.1f}  median={r.median():.1f}  "
      f"p75={r.quantile(.75):.1f}  p90={r.quantile(.90):.1f}  max={r.max():.1f}")

# Cross-check: verify centroid-to-member haversine
print("\n  Cross-checking centroid accuracy (sample 50 cells)...")
sample_cells = ga["geocell_id"].unique()[:50]
max_centroid_err = 0
for gid in sample_cells:
    members = ga[ga["geocell_id"] == gid]
    c_row = gc[gc["geocell_id"] == gid].iloc[0]
    dists = haversine_km(members["latitude"].values, members["longitude"].values,
                         c_row["centroid_lat"], c_row["centroid_lon"])
    max_centroid_err = max(max_centroid_err, dists.max())
check(max_centroid_err < 2000, 
      f"Centroid max-error across 50 cells: {max_centroid_err:.1f} km (reasonable)", 
      f"Centroid error too large: {max_centroid_err:.1f} km")

# ============================================================
section("7. AUDIT REPORT")
# ============================================================
audit = pd.read_csv(DATA_DIR/"audit_report_v2.csv")
check(len(audit) >= 100, f"{len(audit)} images audited", "Too few audited images")
check("source_type" in audit.columns, "Has source_type column", "Missing source_type")
check("noise_estimate" in audit.columns, "Has noise_estimate column", "Missing noise_estimate")

valid = audit[audit["error"].isna()]
print(f"\n  Images audited:  {len(audit)}")
print(f"  Parse errors:    {audit['error'].notna().sum()}")
print(f"\n  Source type split:")
for src, grp in valid.groupby("source_type"):
    pct = 100*len(grp)/len(valid)
    print(f"    {src:20s}: {len(grp):>4} ({pct:.1f}%)")
print(f"\n  Image dimensions: all {int(valid['width'].mode()[0])}x{int(valid['height'].mode()[0])}")
print(f"\n  Noise by source:")
for src, grp in valid.groupby("source_type"):
    print(f"    {src:20s}: mean={grp['noise_estimate'].mean():.3f}  "
          f"std={grp['noise_estimate'].std():.3f}  "
          f"p90={grp['noise_estimate'].quantile(0.9):.3f}")
print(f"\n  Sharpness (Laplacian var) by source:")
for src, grp in valid.groupby("source_type"):
    print(f"    {src:20s}: mean={grp['sharpness_lapvar'].mean():.0f}  "
          f"median={grp['sharpness_lapvar'].median():.0f}")
print(f"\n  Watermark brightness by source:")
for src, grp in valid.groupby("source_type"):
    print(f"    {src:20s}: mean={grp['watermark_region_mean'].mean():.1f}  "
          f"std={grp['watermark_region_mean'].std():.1f}")

# ============================================================
section("8. JOIN CONSISTENCY CHECK")
# ============================================================
# All three label CSVs should cover the same image IDs
coords_ids = set(coords["image_id"])
cl_ids = set(cl["image_id"])
ga_ids = set(ga["image_id"])
check(coords_ids == cl_ids, "country_labels.csv IDs match coords", 
      f"Mismatched IDs: {len(coords_ids ^ cl_ids)} diff")
check(coords_ids == ga_ids, "geocell_assignments.csv IDs match coords", 
      f"Mismatched IDs: {len(coords_ids ^ ga_ids)} diff")

# Verify geocell_id in assignments is within centroid range
ga_ids_cell = set(ga["geocell_id"].unique())
gc_ids_cell = set(gc["geocell_id"].unique())
check(ga_ids_cell == gc_ids_cell, "All geocell_ids have centroid entries", 
      f"Missing centroids for: {ga_ids_cell - gc_ids_cell}")

# ============================================================
section("9. QUICK GEOGRAPHIC DISTRIBUTION CHECK")
# ============================================================
# Merge all labels for one row check
merged = coords.merge(cl[["image_id","country_iso"]], on="image_id") \
               .merge(ga[["image_id","geocell_id"]], on="image_id")
check(len(merged) == len(coords), f"Full merge: {len(merged)} rows", f"Merge lost rows")

# Northern vs Southern hemisphere
nh = (merged["latitude"] > 0).sum()
sh = (merged["latitude"] <= 0).sum()
print(f"\n  Hemisphere split: N={nh} ({100*nh/len(merged):.1f}%)  S={sh} ({100*sh/len(merged):.1f}%)")

# Continent proxy via longitude + latitude regions
eu_asia = ((merged["longitude"] > -10) & (merged["longitude"] < 180) & (merged["latitude"] > 0)).sum()
americas = (merged["longitude"] < -30).sum()
print(f"  E.Hemisphere (land):  {eu_asia}")
print(f"  W.Hemisphere:         {americas}")

# ============================================================
section("10. SUMMARY")
# ============================================================
print(f"\n  Total errors: {len(errors)}")
if errors:
    print("  Errors:")
    for e in errors:
        print(f"    - {e}")
    print(f"\n  Step 1: PARTIAL ({len(errors)} issues to fix)")
    sys.exit(1)
else:
    print("\n  All checks passed!")
    print("\n  Step 1 STATUS: COMPLETE")
    print(f"  - {len(coords):>6} training images with coordinates")
    print(f"  - {n_countries:>6} unique country labels")
    print(f"  - {n_classes:>6} country classes in encoder")
    print(f"  - {n_geocells:>6} geocells (median {gcounts.median():.0f} images/cell)")
    print(f"  - {len(audit):>6} images audited for source type")
    print(f"\n  Ready for Step 2: Pipeline dry run")
