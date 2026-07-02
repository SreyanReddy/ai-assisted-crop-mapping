# Crop Type & Cropping Pattern Mapping - Punjab Rice–Wheat Belt

Multi-season **Sentinel-2 + Random Forest** classification over central Punjab, India. The project maps crop types and cropping *intensity* from satellite imagery, achieving **95.3% accuracy** — and finds that roughly **77% of the classified farmland is double-cropped rice–wheat rotation**, a pattern a simple crop-type map completely hides.

---

## Key Results

| Metric | Value |
|---|---|
| Overall Accuracy | **95.28%** |
| Kappa Coefficient | **0.93** |
| Classes (crop type) | Rice, Wheat, Fallow, Water |
| Classes (cropping pattern) | Double-cropped, Rice-only, Wheat-only, Fallow, Water |
| Study area | ~11,000 km², central Punjab |

**Headline finding:** Nearly 77% of classified cropland is double-cropped rice–wheat land. The apparent 1:2 Rice:Wheat area ratio in the 4-class map was misleading - "Rice" and "Wheat" are largely the *same* fields at different times of year, not two separate populations.

---

## What This Project Does

1. Builds cloud-masked median composites for two crop seasons - **Kharif** (monsoon, rice) and **Rabi** (winter, wheat).
2. Derives a **16-band multi-temporal feature stack**: 6 spectral bands × 2 seasons, plus NDVI and MNDWI per season.
3. Trains a **Random Forest** classifier on manually verified training polygons.
4. Produces a **4-class crop-type map** and a **5-class cropping-pattern map**.
5. Computes per-class area statistics and cropping intensity.

---

## Tech Stack

- **Google Earth Engine** (Python API) - imagery access, compositing, wall-to-wall classification
- **geemap** - interactive map + training-polygon digitization
- **scikit-learn**- Random Forest, accuracy assessment, feature importance
- **Google Dynamic World** - cropland masking + training-location guidance
- pandas / numpy / matplotlib / seaborn

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── Crop_Mapping.ipynb            # main annotated notebook
├── Crop_Mapping.py               # exported script (clean git diffs)
├── Crop_Mapping_Report.md        # full technical report
├── figures/                      # exported charts & maps (GitHub can't render live GEE layers)
│   ├── confusion_matrix.png
│   ├── feature_importance_mdi.png
│   ├── feature_importance_permutation.png
│   ├── crop_map_4class.png
│   └── cropping_pattern_5class.png
└── training_data/
    └── training_polygons.geojson # verified training polygons (reproducibility)
```

---

## Setup & Running

### Prerequisites
- A free **Google Earth Engine** account: https://earthengine.google.com (sign up + create a cloud project)
- Python 3.9+

### Install

```bash
git clone https://github.com/<your-username>/crop-mapping-p7.git
cd crop-mapping-p7
pip install -r requirements.txt
```

### Authenticate with Earth Engine (first run only)

```python
import ee
ee.Authenticate()
ee.Initialize(project='your-gee-project-id')   # <-- replace with your own project ID
```

### Run

Open the notebook and run cells top to bottom:

```bash
jupyter notebook Crop_Mapping.ipynb
```

> **Note:** the notebook uses `geemap` to *interactively draw* training polygons. This repo ships the polygons already frozen into the notebook (and in `training_data/`), so you can run end-to-end without redrawing them. To use your own study area, replace the AOI in the "Study Area" cell and re-collect training polygons.

---

## Method Highlights

A few decisions in here are worth calling out, because they're the difference between a result that looks good and one that *is* good:

- **SCL cloud masking, not QA60.** The QA60 band was not populated by ESA for most of this study's date range (Jan 2022 – Feb 2024), so a QA60 mask silently passes clouds through. The Scene Classification Layer (SCL) is used instead.
- **MNDWI to separate rice from water.** Flooded paddies early in Kharif spectrally resemble open water; adding MNDWI (Green/SWIR water index) resolved most of this confusion. Its contribution was confirmed via **permutation importance** (MDI importance under-ranks it due to correlation with its source bands).
- **Polygon-grouped train/test split.** A naive pixel-level random split gave a *fake* 99.8% accuracy — pixels from the same field are near-identical and leak across the split. `GroupShuffleSplit` on a per-polygon ID gives the honest 95.3%.
- **Cropland masking via Dynamic World** (crops in *either* season) so area stats reflect farmland only.

Full reasoning, including the debugging journey, is in [`Crop_Mapping_Report.md`](Crop_Mapping_Report.md).

---

## Limitations

- Single median composite per season can't capture within-season phenology; multi-date compositing would likely reduce residual rice–water confusion further.
- Training polygons are limited in spatial spread — a more distributed sample would improve generalization.
- The double-cropping split uses a fixed NDVI threshold (0.5); a sensitivity analysis would strengthen the conclusion.

---

## License

Released under the MIT License — see `LICENSE`. Sentinel-2 imagery courtesy of ESA/Copernicus; land-cover reference from Google Dynamic World.
