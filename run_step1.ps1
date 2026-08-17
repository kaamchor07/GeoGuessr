# run_step1.ps1 — Step 1 runner script
# Run from the project root: .\run_step1.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== STEP 1: Data Audit + Label Generation + Geocell Clustering ===" -ForegroundColor Cyan
Write-Host ""

# 1a. Data audit (sample 500 images)
Write-Host "[1a] Running data audit on 500 sampled images..." -ForegroundColor Yellow
python data/audit.py --n_sample 500 --seed 42
if ($LASTEXITCODE -ne 0) { Write-Host "Audit failed!" -ForegroundColor Red; exit 1 }
Write-Host ""

# 1b. Country labels
Write-Host "[1b] Generating country labels via point-in-polygon..." -ForegroundColor Yellow
python labels/country_labels.py
if ($LASTEXITCODE -ne 0) { Write-Host "Country labels failed!" -ForegroundColor Red; exit 1 }
Write-Host ""

# 1c. Auxiliary labels (skip elevation for speed; run separately with internet)
Write-Host "[1c] Generating auxiliary labels (Köppen + WorldCover, skipping elevation)..." -ForegroundColor Yellow
python labels/aux_labels.py --skip_elevation
if ($LASTEXITCODE -ne 0) { Write-Host "Aux labels failed!" -ForegroundColor Red; exit 1 }
Write-Host ""

# 1d. Geocell clustering (KMeans, 1000 cells)
Write-Host "[1d] Building geocells (KMeans, 1000 clusters)..." -ForegroundColor Yellow
python geocells/build_geocells.py --method kmeans --n_clusters 1000
if ($LASTEXITCODE -ne 0) { Write-Host "Geocell clustering failed!" -ForegroundColor Red; exit 1 }
Write-Host ""

Write-Host "=== Step 1 complete! Outputs in /data/ ===" -ForegroundColor Green
Write-Host "  data/audit_report.csv"
Write-Host "  data/country_labels.csv"
Write-Host "  data/country_encoder.csv"
Write-Host "  data/aux_labels.csv"
Write-Host "  data/geocell_assignments.csv"
Write-Host "  data/geocell_centroids.csv"
