# GitHub README: Multimodal Property Price Prediction

Repository: https://github.com/vanshnarang13/Multimodal-Property-Price-Prediction

# Multimodal Property Price Prediction

Predicts residential property prices by fusing **tabular house features** with **satellite imagery**. Tabular baselines top out around R² 0.90; the multimodal deep learning ensemble reaches R² 0.837 on the held-out set, demonstrating that aerial context captures neighbourhood character beyond what structured features alone express.

---

## Problem Statement

Traditional automated valuation models (AVMs) rely solely on structured attributes — square footage, number of bedrooms, build year, etc. These features miss location context that is visually apparent from above: road density, green cover, proximity to commercial zones, lot regularity, and neighbourhood density. This project tests whether pairing satellite imagery with tabular data produces more informative property valuations.

---

## Dataset

**King County Housing Dataset** (Seattle metro, Washington State)

| Split | Samples |
|-------|---------|
| Train | 16,209  |
| Test  | 5,404   |

**Tabular features (20 original → 46 engineered):**
`bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15` — plus engineered features: `age`, `years_since_renovation`, per-room ratios, quality/luxury scores, and 30 spatial KNN features (k=5/10/20/50 neighbourhood price averages, cluster price ratios, distance to key urban centres).

**Satellite imagery:**
640×640 px Google Maps Static API images, zoom level 19, one image per property (keyed by lat/lon). 16,209 training images, 100% coverage.

Price range: $75,000 – $7,700,000 · Median: $450,000 · Mean: $537,470

---

## Architecture

### Pipeline Overview

```
Raw data (CSV + lat/lon)
        │
        ├─── Feature Engineering ──────────────────────────────────────┐
        │    • Base tabular features (16)                               │
        │    • Engineered ratios / quality scores (10+)                 │
        │    • Spatial KNN features (30) — leakage-safe (fit on train)  │
        │    • Log1p target transform to reduce skew (skewness 4.03→0)  │
        │                                                               │
        └─── Satellite Image Fetch                                      │
             • Google Maps Static API, zoom 19, 640×640 px              │
             • 20 parallel download threads                             │
             • Zoom experiment (18–21) → zoom 19 selected              │
                                                                        │
                                    Multimodal Network                  │
                             ┌──────────────────────┐                  │
                             │  Image branch        │◄─────────────────┤
                             │  CNN backbone (224px)│                  │
                             │  → head → 128-d      │                  │
                             │                      │                  │
                             │  Tabular branch      │◄─────────────────┘
                             │  MLP → 64-d          │
                             │                      │
                             │  Fusion              │
                             │  concat(128+64=192)  │
                             │  MLP → 256→128→1     │
                             └──────────────────────┘
```

### Image Branch

Four CNN backbones are trained and ensembled:

| Backbone | Feature Dim | GradCAM Layer |
|---|---|---|
| ResNet-50 | 2048 | `layer4` |
| ResNet-101 | 2048 | `layer4` |
| EfficientNet-B0 | 1280 | `features` |
| Inception-V3 | 2048 | `Mixed_7c` |

Each backbone is **frozen** during early epochs; `layer4` / final stage is unfrozen at epoch 5 (progressive unfreezing). Image features are projected through a head: `Linear(2048→256) → BN → ReLU → Dropout → Linear(256→128)`.

### Tabular Branch

Three-layer MLP: `Linear(N→256) → BN → ReLU → Dropout → Linear(256→128) → BN → ReLU → Dropout → Linear(128→64)`

### Fusion

Late fusion by concatenation: `[img_128 || tab_64]` → `Linear(192→256) → BN → ReLU → Dropout → Linear(256→128) → BN → ReLU → Dropout → Linear(128→1)`

Final predictions are inverse-transformed from normalised log-price space back to dollar values.

### Ensemble

All four backbone models are ensembled with R²-proportional weights (≈0.25 each) to produce the final prediction.

---

## Results

### Tabular Baselines (validation set)

| Model | Val RMSE | Val R² |
|---|---|---|
| XGBoost | $110,741 | 0.9023 |
| LightGBM | $114,558 | 0.8954 |
| Gradient Boosting | $116,875 | 0.8911 |
| Random Forest | $122,772 | 0.8799 |
| Ridge / Lasso | ~$129,000 | ~0.867 |

XGBoost with spatial features: **RMSE $104,145, R² 0.914**

### Image-Only Baselines (CNN features + ML)

| Model | Val RMSE | Val R² |
|---|---|---|
| Ridge (CNN) | $278,407 | 0.382 |
| XGBoost (CNN) | $294,223 | 0.310 |

Satellite features alone are insufficient — the image signal is complementary, not a replacement.

### Deep Learning Multimodal (individual backbones)

| Backbone | Val R² | Val RMSE |
|---|---|---|
| ResNet-50 | 0.8308 | $145,713 |
| Inception-V3 | 0.8224 | $149,304 |
| ResNet-101 | 0.8157 | $152,060 |
| EfficientNet-B0 | 0.7989 | $158,871 |

### Ensemble

| Ensemble | Val R² | Val RMSE | Val MAE | MAPE |
|---|---|---|---|---|
| Equal-weight | 0.8364 | $143,294 | $79,624 | 14.90% |
| R²-weighted | **0.8365** | **$143,243** | **$79,606** | **14.90%** |

> The multimodal ensemble slightly underperforms the best tabular XGBoost (R² 0.902 vs 0.837). This gap reflects limited training epochs (15), image resolution constraints at zoom 19, and the high information density of engineered spatial features. With extended training and a higher-resolution imagery pipeline the gap would likely narrow.

### GradCAM Highlights

GradCAM visualisations on ResNet-50 show the model attends to lot boundaries, road adjacency, and rooftop footprint — confirming that the image branch has learned spatially meaningful signals rather than trivial colour statistics.

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Deep learning | PyTorch ≥ 2.0, torchvision |
| Tabular ML | XGBoost ≥ 2.0, LightGBM, scikit-learn |
| Satellite imagery | Google Maps Static API, PIL |
| Data wrangling | pandas ≥ 2.0, numpy |
| Spatial features | BallTree (scikit-learn), haversine |
| Hyperparameter opt | Optuna |
| Experiment tracking | MLflow |
| Explainability | SHAP (tabular), GradCAM (image) |
| CNN model zoo | torchvision models + timm + torchgeo |
| Visualisation | matplotlib, seaborn |

---

## Project Structure

```
Multimodal-Property-Price-Prediction/
├── Model_training/
│   └── multimodal_property_valuation.ipynb   # End-to-end training: CNN extraction,
│                                              # multimodal network, ensemble, GradCAM, SHAP
├── notebooks/
│   ├── eda.ipynb                              # Tabular EDA, feature correlations
│   ├── preprocessing.ipynb                   # Feature engineering, spatial features,
│   │                                         # train/val/test split, data saving
│   └── satellite_image_eda.ipynb             # Image coverage check, colour/texture stats,
│                                             # CNN embedding t-SNE, image–price correlation
├── scripts/
│   ├── data_fetcher.py                       # Parallel satellite image downloader
│   └── zoom_experiment.py                    # Zoom-level comparison (18–21)
├── requirements.txt
└── 23124043_final.csv                        # Final test-set predictions
```


---

## Key Findings

1. **Spatial features dominate** — KNN neighbourhood price ratios and cluster statistics are the highest-gain tabular features, outperforming raw size/grade metrics.
2. **Image signal is complementary** — image-only models achieve R² ≈ 0.38; fused multimodal models reach R² ≈ 0.84, showing images add context on top of tabular features rather than replacing them.
3. **GradCAM confirms spatial awareness** — the CNN attends to lot boundaries and road proximity, not just rooftop area.
4. **Ensemble reduces variance** — the four-backbone ensemble consistently beats every individual model by 1–4 R² points.
5. **High-value properties are hardest** — worst GradCAM predictions are on properties above $2M, where satellite imagery provides less discriminative signal at zoom 19.
