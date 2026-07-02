# Crop Type and Cropping Pattern Mapping using Multi-season Satellite Data and Machine Learning

**Project Code:** P7
**Study Area:** Central Punjab rice–wheat belt, India
**Platform:** Google Earth Engine (Python API) + scikit-learn
**Classifier:** Random Forest

---

## 1. Objective

To classify crop types and analyse cropping patterns across an agricultural region using multi-season Sentinel-2 imagery, and to apply a Random Forest classifier to improve classification accuracy and spatial interpretation. A secondary, and ultimately central, objective that emerged during the work was to correctly characterise **cropping intensity** (single vs. double cropping) in a landscape dominated by rice–wheat rotation.

---

## 2. Study Area

A rectangular Area of Interest (AOI) was defined over the central Punjab agricultural belt (approximately Ludhiana–Moga–Sangrur–Patiala corridor):

- **Extent:** 75.0°–76.3° E, 30.2°–31.0° N
- **Approximate area:** ~11,000 km²

This region was deliberately chosen after an initial attempt over a much larger AOI (73.8°–76.5° E, 29.5°–32.0° N, ~72,000 km²) proved unworkable. The larger extent spanned multiple agro-climatic zones — the Himalayan foothills in the north and the semi-arid south-western cotton belt — which could not be adequately represented by a small training sample. Restricting the AOI to the spectrally coherent central rice–wheat belt was the first major corrective decision of the project.

---

## 3. Data Used

- **Satellite data:** Sentinel-2 Level-2A Surface Reflectance (`COPERNICUS/S2_SR_HARMONIZED`), accessed via Google Earth Engine.
- **Reference land cover:** Google Dynamic World V1 (`GOOGLE/DYNAMICWORLD/V1`) — used both as a labelling aid and as a cropland mask.
- **Seasons captured:**
  - **Kharif** (monsoon-sown, rice-dominant): 2023-06-01 to 2023-10-31
  - **Rabi** (winter-sown, wheat-dominant): 2023-11-01 to 2024-03-31

### 3.1 Note on data source migration

The original project brief referenced the Copernicus Open Access Hub (`scihub.copernicus.eu`) for Sentinel data. This portal has been decommissioned; imagery is now served through the Copernicus Data Space Ecosystem (`dataspace.copernicus.eu`). In this implementation, imagery was accessed server-side through Google Earth Engine rather than downloaded, avoiding local storage and preprocessing entirely.

---

## 4. Methodology

### 4.1 Feature construction

For each season, a cloud-masked median composite was produced. From each composite the following were derived, giving a **16-band feature stack**:

| Feature group | Bands | Count |
|---|---|---|
| Spectral bands (per season) | B2, B3, B4, B8, B11, B12 | 12 |
| NDVI (per season) | (B8 − B4) / (B8 + B4) | 2 |
| MNDWI (per season) | (B3 − B11) / (B3 + B11) | 2 |

NDVI captures vegetation vigour; MNDWI (Modified Normalized Difference Water Index) was added specifically to address a water-classification problem discussed below.

### 4.2 Cloud masking — a corrected approach

The initial cloud-masking function used the Sentinel-2 **QA60** band. This was found to be ineffective for the study period: ESA stopped populating QA60 with cloud-mask polygons after 25 January 2022 and did not reconstruct the band until 28 February 2024. Both the Kharif and Rabi windows fall almost entirely inside this gap, meaning the QA60-based mask was silently passing all pixels — clouds included — into the composites. This is especially damaging in the Kharif (monsoon) season.

The mask was rewritten to use the **Scene Classification Layer (SCL)**, retaining only vegetation (4), bare soil (5), water (6), and unclassified (7) pixels, which is the current standard approach for Level-2A data.

### 4.3 Training data collection

Training polygons were digitised interactively on a `geemap` viewer, cross-referenced against true-colour and NDVI layers for both seasons. Four classes were defined:

- 0 = Rice, 1 = Wheat, 2 = Fallow/Barren, 3 = Water

Class identification relied on **multi-temporal reasoning** rather than single-image interpretation: rice appears green (high NDVI) in Kharif and bare in Rabi; wheat shows the inverse; fallow is bare in both; water is spectrally stable across seasons. Google Dynamic World was used to highlight candidate locations for classes that were hard to locate by eye (particularly Fallow and Water). Genuine fallow land proved rare in this intensively farmed belt — a finding consistent with the region's very high cropping intensity.

Each polygon was assigned a unique `poly_id` in addition to its class label; this identifier is essential for the accuracy assessment (Section 4.5).

### 4.4 Sample extraction and class balancing

Pixels were extracted from within each polygon using `sampleRegions` at 10 m resolution. Because polygons vary in size, the raw pixel count per class was highly uneven, and the total exceeded Earth Engine's 5,000-element synchronous query limit. To resolve both issues, a per-class random cap of **400 samples** was applied (using a `randomColumn` shuffle), producing a balanced training set of 1,600 samples (400 per class).

### 4.5 Classifier training and — critically — honest accuracy assessment

A Random Forest (100 trees, max depth 10) was trained. Two classifiers were used in parallel: scikit-learn locally (for evaluation and feature-importance analysis) and Earth Engine's `smileRandomForest` (for wall-to-wall map production).

**The accuracy assessment required a fundamental correction.** An initial 70/30 random `train_test_split` at the pixel level produced an implausible **99.79% overall accuracy (kappa 0.997)**. This figure is a classic symptom of **spatial autocorrelation leakage**: pixels from the same field are near-identical, and a random pixel-level split places neighbouring pixels of the same polygon into both training and test sets. The model was effectively being tested on data it had already seen.

The split was replaced with `GroupShuffleSplit`, grouping on `poly_id` so that all pixels from any given polygon fall entirely into either the training or the test set, never both. This produced an honest and defensible accuracy (Section 5).

---

## 5. Results

### 5.1 Classification accuracy

Under the corrected, group-aware split:

| Metric | Value |
|---|---|
| Overall Accuracy | **95.28%** |
| Kappa Coefficient | **0.93** |

Per-class performance:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Rice | 1.00 | 0.90 | 0.95 |
| Wheat | 0.96 | 1.00 | 0.98 |
| Fallow | 1.00 | 1.00 | 1.00 |
| Water | 0.81 | 1.00 | 0.89 |

The only meaningful residual confusion is **Rice → Water**: of 262 test Rice pixels, 23 were predicted as Water. This is the well-documented rice-paddy-versus-open-water confusion — early in the Kharif season, transplanted paddies stand under water for several weeks before the canopy closes, producing a spectral signature close to open water. MNDWI reduced but did not fully eliminate this effect.

### 5.2 The role of MNDWI, and a note on feature importance

MNDWI was added specifically to separate flooded paddies from open water. Its effect on the final map is clear (Section 5.3): the classified Water area fell from 1,850 km² to 319 km² across the AOI as MNDWI (and the other fixes) were applied.

However, the **Mean Decrease in Impurity (MDI)** feature-importance metric ranked MNDWI near the bottom. This is a known artefact: MDI is biased when features are correlated, and MNDWI is derived from B3 and B11, which are also present as raw bands in the stack. The tree can approximate the MNDWI decision boundary using the raw bands, so credit is spread across the three correlated features.

**Permutation importance** — which measures the actual drop in accuracy when a feature is shuffled, and is immune to this bias — told the correct story: **MNDWI_kharif ranked 3rd overall**, confirming its genuine contribution to the Rice/Water separation. MNDWI_rabi ranked near zero, which is physically correct, since there is no flooding in the winter wheat season for it to detect.

### 5.3 Area statistics (4-class map)

Areas were computed after masking the classification to cropland using a Dynamic World crops mask (flagged as crops in either Kharif or Rabi, to avoid excluding single-season fields):

| Class | Area (km²) | % of classified area |
|---|---|---|
| Rice | 3,056.08 | 31.7% |
| Wheat | 6,186.61 | 64.1% |
| Fallow/Barren | 87.70 | 0.9% |
| Water | 318.76 | 3.3% |
| **Total** | **9,649.15** | 100% |

**Evolution of the Water estimate** as corrective measures were applied:

| Stage | Water area (km²) |
|---|---|
| Initial (QA60 mask, no MNDWI, large AOI) | ~36% of AOI (implausible) |
| After AOI restriction + SCL mask | 1,850.27 |
| After adding MNDWI | 660.17 |
| After cropland masking | 318.76 |

The final ~3% water footprint is physically reasonable for a canal-irrigated farmland region.

### 5.4 The cropping-intensity finding

The 4-class area statistics show Wheat area at roughly **twice** Rice area (≈1:2). In a landscape known for intensive rice–wheat rotation — where most fields grow both crops in successive seasons — a ratio close to 1:1 would be expected instead. This discrepancy prompted a direct investigation.

Using NDVI thresholds (> 0.5 = actively growing) to test whether classified pixels were green in the *other* season:

| Check | Area (km²) | Interpretation |
|---|---|---|
| Wheat-labelled land also green in Kharif | 5,213.94 | 84.3% of "Wheat" is actually double-cropped |
| Rice-labelled land also green in Rabi | 2,501.35 | 81.8% of "Rice" is actually double-cropped |

Recomputing the landscape on this basis:

| Category | Area (km²) | % of classified area |
|---|---|---|
| Double-cropped (rice–wheat rotation) | 7,715.29 | ~80% |
| Single-cropped Rice only | 554.73 | ~6% |
| Single-cropped Wheat only | 972.67 | ~10% |
| Fallow | 87.70 | ~1% |
| Water | 318.76 | ~3% |
| **Total** | **9,649.15** | 100% |

**Interpretation.** Roughly 80% of the classified cropland is double-cropped rice–wheat land. The apparent 1:2 Rice:Wheat ratio was therefore misleading: "Rice" and "Wheat" were never two disjoint populations of fields — they are, for the most part, the *same* land at different times of year. When a pixel was green in both seasons, the single-label classifier was forced to assign it to one class or the other, and it happened to default the majority of ambiguous pixels to Wheat. This is not a classifier error so much as a limitation of imposing four mutually exclusive labels on a landscape where "both" is the dominant reality. The finding is consistent with published characterisations of central Punjab as one of the most intensively double-cropped regions in the world.

This motivated building a dedicated, spatially explicit cropping-pattern map, presented next.

### 5.5 The 5-class cropping-pattern map

Rather than relying only on the arithmetic estimate above, a pixel-level cropping-pattern layer was produced directly. Each cropland pixel was classified as **Double-cropped**, **Rice-only**, or **Wheat-only** using the same season-crossing NDVI logic (threshold 0.5), with the trained Random Forest classifier's own Rice/Wheat label used as a fallback for the small number of pixels not clearly resolved by the NDVI test alone; Fallow and Water were carried over directly from the validated 4-class classifier output.

| Category | Area (km²) | % of classified area |
|---|---|---|
| Double-cropped (rice–wheat rotation) | 7,392.01 | 76.6% |
| Single-cropped Rice only | 876.01 | 9.1% |
| Single-cropped Wheat only | 974.66 | 10.1% |
| Fallow | 87.70 | 0.9% |
| Water | 318.76 | 3.3% |
| **Total** | **9,649.14** | 100% |

The total matches the 4-class map's total (9,649.15 km²) to within 0.01 km², confirming that every classified cropland pixel is accounted for and none are silently dropped between the two representations.

These figures are close to, but not identical to, the arithmetic estimate in Section 5.4 (7,715.29 / 554.73 / 972.67 km²). The difference stems from a methodological distinction between the two approaches: the arithmetic estimate starts from the classifier's Rice/Wheat *label* and asks whether the pixel was also green in the other season, whereas the pixel-level map applies the NDVI double-season test first and falls back to the classifier label only for the residual pixels the NDVI test cannot resolve. The two methods agree on the overall picture — the map is treated as the authoritative figure, since it is a complete spatial product rather than a post-hoc recombination of two separate area sums.

**Interpretation.** Roughly 77% of the classified cropland is double-cropped rice–wheat land, confirming the arithmetic finding from Section 5.4. The apparent 1:2 Rice:Wheat ratio in the original 4-class map was therefore misleading: "Rice" and "Wheat" were never two disjoint populations of fields — they are, for the most part, the *same* land at different times of year. When a pixel was green in both seasons, the single-label 4-class classifier was forced to assign it to one class or the other, and it happened to default the majority of ambiguous pixels to Wheat. This is not a classifier error so much as a limitation of imposing four mutually exclusive labels on a landscape where "both" is the dominant reality. The finding is consistent with published characterisations of central Punjab as one of the most intensively double-cropped regions in the world, and directly fulfils the cropping-pattern classification called for in the original project brief.

---

## 6. Summary of Corrective Decisions

For transparency, the key methodological corrections made during the project are summarised below, since each materially affected the result:

1. **AOI restriction** — from ~72,000 km² spanning multiple agro-climatic zones to a ~11,000 km² spectrally coherent belt.
2. **Cloud mask** — QA60 (non-functional for the study period) replaced with the SCL band.
3. **Training data** — from 25 unverified single points to hundreds of pixels drawn from verified multi-season polygons, class-balanced at 400 samples each.
4. **MNDWI features added** — to separate flooded paddies from open water.
5. **Accuracy assessment** — pixel-level random split (leaky, fake 99.79%) replaced with polygon-grouped split (honest 95.28%).
6. **Feature importance** — MDI interpreted with caution; permutation importance used to confirm MNDWI's real contribution.
7. **Cropland masking** — Dynamic World used to exclude non-crop land from area statistics.
8. **Cropping-intensity analysis** — NDVI cross-season check revealed ~77-80% double-cropping, reframing the entire Rice:Wheat interpretation.
9. **5-class cropping-pattern map** — built as a spatially explicit product rather than relying solely on aggregate area arithmetic, confirming the double-cropping finding pixel-by-pixel and satisfying the brief's original cropping-pattern requirement directly.

---

## 7. Conclusion

A multi-season Sentinel-2 Random Forest classifier achieved 95.28% overall accuracy (kappa 0.93) for crop-type mapping over central Punjab, once spatial-autocorrelation leakage in the accuracy assessment was corrected. The addition of MNDWI substantially resolved the rice–water confusion characteristic of flooded-paddy landscapes, as confirmed by permutation importance.

The most significant finding is that a simple four-class crop-type map is an inadequate representation of this landscape: a dedicated 5-class cropping-pattern map shows approximately 77% of the classified cropland is double-cropped rice–wheat rotation, and the original four mutually exclusive classes obscured this by forcing double-cropped pixels into a single label. Cropping *intensity*, derived from multi-season NDVI and mapped pixel-by-pixel, is therefore a more faithful description of the region than crop *type* alone, and directly answers the cropping-pattern objective set out in the original project brief.

### Limitations

- A single median composite per season cannot capture within-season phenological transitions; multi-date compositing would likely further reduce residual Rice–Water confusion.
- Training data, though verified, is limited in spatial spread; a larger, more distributed sample would improve generalisation.
- The double-cropping analysis uses a fixed NDVI threshold (0.5); a sensitivity analysis on this threshold would strengthen the conclusion.
- Rice area is modestly underestimated in the 4-class map because ~9% of true rice leaks into the Water class.

### Skills demonstrated

Multi-temporal satellite compositing, cloud masking, vegetation and water spectral indices, supervised classification with Random Forest, rigorous accuracy assessment (including avoidance of spatial leakage), feature-importance interpretation, and cropping-pattern analysis — applied to a real agricultural monitoring problem with direct relevance to food security and resource management.
