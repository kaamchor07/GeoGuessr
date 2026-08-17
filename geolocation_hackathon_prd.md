# Geolocation Hackathon — Technical PRD & Build Plan

**Purpose of this document:** hand this to an agentic coding tool (e.g. Antigravity) as the project spec. Section 12 explains how to prompt against it module-by-module. Keep this file updated as the source of truth — edit Section 9's status markers as work completes rather than re-deriving the plan in chat each time.

---

## 1. Objective

Predict `(lat, lon, radius_km)` for unlabeled street-level/landscape images. Score = median per-image score across:
- Haversine distance score (smooth decay, no hard cutoff)
- Calibration bonus/penalty on claimed radius (tight-and-correct > wide-and-safe; tight-and-wrong is punished hardest)
- Country-match bonus (predicted center in correct country + reasonably tight radius)

Two test sets: Test Set 1 (live now, smaller), Test Set 2 (hidden, released later, larger). Final score is a weighted combination of both — do not overfit exclusively to Test Set 1's public leaderboard.

## 2. Hard Constraints — do not violate

- **Compute:** free-tier only (Colab T4, Kaggle P100/T4). No paid/rented cloud compute.
- **Inference must be 100% offline.** No network calls of any kind at inference time. All weights must be bundled or loaded without network access.
- **No zero-shot.** Any pretrained backbone must have new task-specific parameters trained on top. Frozen-backbone-plus-new-head is compliant; prompting a model with no new training is not.
- **VLM constraint:** local open-weight VLMs are allowed *only* if you freeze the vision encoder, discard the language-generation head, and train a new classifier/regressor on the visual embeddings. **Do not** let a VLM generate a text caption/description that feeds into the final location prediction — even indirectly, this reads as the language-generation head producing the answer, which is a rules violation. Skip this pipeline shape entirely.
- **External data:** free to use, but must be disclosed in the write-up, and must not contain the actual hidden evaluation images or their ground-truth coordinates, "even inadvertently." Use OSV5M or Mapillary Street-Level Sequences (rights-cleared, well documented). **Do not scrape Google Street View or GeoGuessr directly** — redistribution violates Google's ToS and the rules put the rights burden on you.
- **Submission cap:** 15/day on Test Set 1.
- **Do not** use multiple Kaggle accounts to multiply GPU quota — against Kaggle ToS, real risk of account action mid-hackathon.
- Every submission must be reproducible from the submitted notebook. Fix seeds everywhere.

## 3. Data Inventory

| File | Contents |
|---|---|
| `training_dataset/noised_dataset/images/` | Training images |
| `training_dataset/noised_dataset/ground_truth_coordinates.csv` | `image_id, latitude, longitude` |
| `country_boundaries.geojson` | Country polygons with ISO codes — usable for free label generation and post-hoc correction |
| `sample_submission.csv` | Target format: `image_id, pred_lat, pred_lon, pred_radius_km` |

### Known data quirks to handle explicitly

1. **Noise is likely source-correlated, not uniform.** Visible heavy grain appears concentrated on images that look Street-View-sourced (watermarked, square-ish crop, uniform lighting). Images that look dashcam/phone-sourced (no watermark, wide aspect ratio, occasional dashboard/hood visible at frame bottom) look cleaner. Hypothesis: noise was added specifically to defeat reverse-image-search against live Street View, since dashcam photos aren't in Google's index and have nothing to defeat. **Action:** audit a sample of train images by source type before deciding preprocessing; do not denoise unless you can denoise train and test identically.
2. **Mixed source domains.** At least two visually distinct capture types exist in the training set. The model must not use source-type as a shortcut for geography (watermark presence, aspect ratio, or lighting profile could spuriously correlate with region if not addressed).
3. **Watermark leakage.** "Google" logo visible bottom-left on Street-View-sourced images. Mask or crop this region before feeding images to the backbone, for all images (so its absence isn't itself a signal).

## 4. Architecture — Tier 1 (build this)

1. **Backbone:** frozen CLIP or SigLIP, ViT-B scale (not the largest variant — inference/embedding extraction must stay cheap on free-tier GPUs). Do not fine-tune the backbone itself; only new heads get trained.
2. **Geocells:** semantic clustering (KMeans or OPTICS) over training coordinates — not a naive lat/lon grid, which wastes classes over oceans/empty land. Optionally align cell boundaries to `country_boundaries.geojson`.
3. **Country head:** coarse classifier trained on country labels derived for free via point-in-polygon test of each training image's `(lat, lon)` against `country_boundaries.geojson`. No manual labeling needed.
4. **Geocell head:** fine-grained classifier using haversine-smoothed cross-entropy (soft targets weighted by distance from true location to each geocell centroid, not one-hot) — this is the single biggest lever over plain classification.
5. **Auxiliary heads (free labels, no extra images needed):** climate zone (Köppen classification), land cover class (ESA WorldCover), elevation (SRTM) — all derivable from `(lat, lon)` via public raster lookups. Multi-task training on these regularizes the shared embedding toward the same cues humans use (vegetation, terrain, climate).
6. **Domain-adversarial head:** gradient-reversal branch predicting source type (Street-View-crop vs. dashcam-style) from the shared embedding. Forces the backbone's *task-relevant* features to not encode which source an image came from — directly addresses the watermark/domain-leakage risk in Section 3.
7. **kNN refinement:** build a FAISS index over embeddings — ideally the *external* dataset (OSV5M/Mapillary), not just the small provided training set, for denser coverage. At inference, refine the geocell-centroid guess using nearest neighbors within the top-K predicted geocells.
8. **Radius calibration:** conformal calibration or per-geocell empirical error on a held-out split. Code a proxy of the actual scoring formula (distance decay + calibration term + country bonus) and grid-search the radius-scaling factor against it directly, rather than guessing.
9. **Country-snap post-processing:** if the predicted point falls just outside a country boundary but the country head is highly confident, nudge the point slightly toward/inside that country to capture the bonus. Pure post-processing on your own output — no test data touched.
10. **Test-time augmentation:** multi-crop/multi-scale, average predictions. **Skip horizontal flips** — they invert driving-side and mirror text/signage, both real signal for this task.
11. **Snapshot ensembling / SWA:** average weights or predictions across the last few checkpoints. Free generalization gain, no extra training cost.

## 5. Tier 2 — build only if Tier 1 finishes with time to spare

- Adversarial validation reweighting (train a classifier to distinguish provided-train vs. external-data images; upweight external samples that most resemble the real distribution).
- Class-balanced sampling or focal-style loss weighting for underrepresented countries/geocells — audit the country distribution in the provided training set on day 1 before deciding if this is needed.
- Embedding-space mixup (interpolate embeddings + soft geocell targets) as a classifier regularizer.
- Pretrain heads on external data, short fine-tune pass on provided training data to match the eval distribution more closely.

## 6. Explicitly out of scope — parking lot, do not build

- Cross-view satellite-image retrieval matching — real research direction, not a 5-day free-tier build.
- Mixture density network head instead of post-hoc calibration — only reconsider if Tier 1 calibration underperforms in validation.
- Self-generated pseudo-labeling on unlabeled extra images — risks reinforcing an immature model's own mistakes.
- VLM-caption-to-classifier pipeline (see Section 2 — compliance risk).
- Scraping Street View / GeoGuessr imagery (see Section 2 — ToS risk).
- Multi-account Kaggle compute farming (see Section 2 — ToS risk).

## 7. Compute Strategy

**Available resources:** Kaggle notebook (GPU T4 x2 accelerator option), Google Colab (free T4), local laptop (RTX 3050, ~4GB VRAM).

**Role assignment:**
- **Laptop (RTX 3050):** local dev and debugging, dry runs on small subsets (~200 images), all CPU-heavy prep work — geocell clustering, geojson point-in-polygon labeling, raster lookups for auxiliary labels, FAISS index construction. Validate code correctness here before pushing to cloud GPU time. Use small batch sizes (8–16) and mixed precision (fp16) if running any GPU inference locally — 4GB VRAM is fine for frozen-backbone embedding extraction, too small for training runs beyond debugging scale.
- **Kaggle (GPU T4 x2):** primary training runs. Default PyTorch code only uses GPU 0 — wrap the model in `torch.nn.DataParallel(model)` for simple dual-GPU use (sufficient at this scale), or `DistributedDataParallel` via `torchrun` if you want better scaling and have time to set it up.
- **Colab (free T4):** run a second, independent experiment in parallel with Kaggle — this is legitimate parallelization since the rules explicitly permit both platforms. Example split: Kaggle trains the main geocell+country head while Colab trains/tunes the domain-adversarial variant, or precomputes embeddings over the external dataset.

**What not to do:** don't try to split a single training run's gradient updates across two separate free platforms — parallelize at the experiment level (different runs on different platforms) or via true multi-GPU (DataParallel/DDP) within one Kaggle session, not across platforms mid-run.

## 8. Proposed repo structure

```
/data              # raw + processed data, geocell assignments, country/aux labels
/geocells          # clustering script, geocell centroid lookup
/labels            # point-in-polygon country labeling, raster-based aux label lookups
/models            # backbone wrapper, head definitions, domain-adversarial branch
/training           # training loop, DataParallel setup, checkpointing
/refinement         # FAISS index build + kNN refinement at inference
/calibration        # radius calibration, scoring-formula proxy, grid search
/inference          # end-to-end inference producing submission CSV
/notebooks          # final executed notebook per deliverable requirements
```

## 9. Execution order

| Step | Status | Description |
|---|---|---|
| 1 | DONE | Data audit: sample train images by apparent source type, check noise consistency; build geocells (KMeans/OPTICS); generate country labels via geojson point-in-polygon; generate auxiliary raster labels (climate/land-cover/elevation) |
| 2 | DONE | Pipeline dry run on ~200 images, 1 epoch: confirm loss drops, checkpoint saves, and output CSV matches `sample_submission.csv` schema exactly |
| 3 | DONE | Matched noise augmentation pipeline (`data/noise_aug.py`) calibrated to Section 3 findings; watermark masking in dataset |
| 4 | READY | Train full model: backbone (frozen) + country head + geocell head (haversine-smoothed CE) + auxiliary heads + domain-adversarial head (`training/train.py`) |
| 5 | DONE | Build FAISS embedding index (`refinement/build_index.py`) and kNN spatial refinement engine (`refinement/knn_refine.py`) |
| 6 | DONE | Calibrate radius against scoring proxy (`calibration/calibrate_radius.py`); implement country-snap post-processing (`calibration/country_snap.py`) |
| 7 | DONE | TTA multi-crop inference + submission generator pipeline (`inference/make_submission.py`) matching `sample_submission.csv` format |



## 10. Validation Strategy

- Stratified hold-out split from the provided training data, stratified by country/geocell (not a random split — rare countries need representation in validation too).
- Code a local proxy of the actual scoring formula (distance decay + calibration term + country bonus) for calibration and radius-multiplier tuning — don't guess at what the organizers' scorer rewards when the components are stated explicitly in the rules.
- Never validate against hidden test sets directly (not available anyway) — use this proxy plus the live Test Set 1 leaderboard sparingly, respecting the 15/day cap.

## 11. Deliverables Checklist (per hackathon rules Section 4.7)

- [ ] Final prediction CSV per test set, exact column match to `sample_submission.csv`
- [ ] Fully executed notebook, all cells run with visible outputs: feature engineering/augmentation steps, full training pipeline for best model, explored-but-abandoned approaches, code that generated the submitted CSV
- [ ] Write-up disclosing external datasets used (name: OSV5M / Mapillary Street-Level Sequences), methodology, key decisions
- [ ] All code reproducible — fixed seeds throughout

## 12. How to work with Antigravity on this

Paste this document in as project context first. Then issue **narrow, single-module prompts** referencing specific sections/file paths rather than asking for the whole pipeline in one shot — test and run each module in isolation before moving to the next. Update Section 9's status column as steps complete so future prompts stay grounded in current progress instead of re-deriving the plan.

Example prompts:

- *"Using Section 4 item 2 of geolocation_hackathon_prd.md, write `/geocells/build_geocells.py`: load `ground_truth_coordinates.csv`, run OPTICS clustering on (lat, lon), output a CSV with an added `geocell_id` column and a separate `geocell_centroids.csv`. Add a `--min_samples` CLI arg."*
- *"Using Section 4 item 3, write `/labels/country_labels.py`: for each row in `ground_truth_coordinates.csv`, run a point-in-polygon test against `country_boundaries.geojson` using shapely, output `image_id,country_iso`."*
- *"Write `/labels/aux_labels.py`: for each training image's (lat, lon), look up Köppen climate zone, ESA WorldCover land-cover class, and SRTM elevation, output one CSV with all three as extra columns."*
- *"Write `/data/dataset.py`: a PyTorch Dataset loading images from `training_dataset/noised_dataset/images/`, masking the bottom-left watermark region, resizing for backbone input, returning image tensor plus geocell_id, country_iso, and auxiliary labels."*
- *"Write `/models/model.py`: frozen CLIP ViT-B backbone (weights not updated), with separate heads for country classification, geocell classification (haversine-smoothed CE per Section 4 item 4), the three auxiliary regressions/classifications, and a gradient-reversal domain-adversarial head per Section 4 item 6."*
- *"Write `/training/train.py`: training loop wrapping the model in `nn.DataParallel` for 2x T4 usage on Kaggle, with checkpointing every N steps and fixed seeds."*
- *"Write `/refinement/build_index.py` and `/refinement/knn_refine.py`: FAISS index over training+external embeddings, and a refinement function that adjusts a geocell-centroid prediction using the top-K nearest neighbors within predicted geocells."*
- *"Write `/calibration/calibrate_radius.py`: implement a local proxy of the scoring formula in Section 10, and grid-search a radius-scaling factor against it on the held-out validation split."*
- *"Write `/inference/make_submission.py`: end-to-end inference producing a CSV matching `sample_submission.csv`'s schema exactly, including TTA (multi-crop, no flips) and the country-snap post-processing step from Section 4 item 9."*
