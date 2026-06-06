# Project: Multimodal Property Price Prediction

## Overview

Built an end-to-end automated valuation model (AVM) for King County (Seattle metro) residential properties that fuses structured house attributes with Google Maps satellite imagery. The multimodal deep learning ensemble achieves R² 0.837 / RMSE $143,243 on the held-out validation set. The best tabular-only XGBoost baseline reaches R² 0.902 / RMSE $104,145, so the project demonstrates both the power of engineered spatial features and the complementary (but not yet dominant) signal that overhead imagery adds.

## Tech Stack

- **Deep learning**: PyTorch 2.x, torchvision (ResNet-50, ResNet-101, EfficientNet-B0, Inception-V3)
- **Tabular ML**: XGBoost, LightGBM, scikit-learn (Ridge, Lasso, GradientBoosting, RandomForest)
- **Satellite imagery**: Google Maps Static API, PIL/Pillow, 20-thread parallel downloader
- **Data processing**: pandas, numpy, BallTree (spatial KNN), haversine distances
- **Explainability**: SHAP (tabular branch), GradCAM via pytorch-grad-cam (image branch)
- **Experiment tracking**: MLflow
- **Hyperparameter optimisation**: Optuna

## Dataset

King County Housing Dataset — 16,209 training samples and 5,404 test samples covering Seattle-area residential sales (2014–2015). Price range $75,000–$7,700,000, median $450,000. Each record has 20 tabular features (bedrooms, bathrooms, sqft, grade, condition, waterfront, view, lat/lon, etc.) plus one 640×640 px satellite image fetched at zoom level 19 from the Google Maps Static API (100% image coverage achieved). Zoom level was selected empirically by testing levels 18–21 on sample locations.

## Approach & Architecture

**Feature engineering (preprocessing.ipynb)**

Raw tabular features are expanded to 46 features:
- Original 16 structural features
- Engineered ratios: sqft_per_bedroom, sqft_per_bathroom, bath_to_bed_ratio, basement_ratio
- Quality composite: grade + condition score, luxury_index (grade ≥ 10)
- Age / renovation: property_age, years_since_renovation
- Spatial KNN features (leakage-safe — fit on training set only): for k in {5, 10, 20, 50}, compute log-price mean/std of k nearest neighbours using BallTree on lat/lon; also 50-cluster spatial price ratios and zipcode-level price statistics (30 spatial features total)

Target is log1p-transformed (price skewness 4.03 → near-normal) before training; inverse-transformed for evaluation.

**Multimodal fusion network (MultimodalPropertyNetwork)**

Two branches, fused by late concatenation:

```
Image branch:
  CNN backbone (frozen) → GlobalAvgPool → 2048-d (or 1280-d for EfficientNet)
  → Linear(→256) → BN → ReLU → Dropout(0.3) → Linear(→128)   [128-d]

Tabular branch:
  Linear(46→256) → BN → ReLU → Dropout → Linear(→128) → BN → ReLU → Dropout → Linear(→64)   [64-d]

Fusion:
  concat([128, 64]) = 192-d
  → Linear(→256) → BN → ReLU → Dropout → Linear(→128) → BN → ReLU → Dropout(0.15) → Linear(→1)
```

Progressive unfreezing: backbone frozen for first 5 epochs, then layer4 (or equivalent final stage) unfrozen with a lower learning rate to prevent feature destruction.

Four backbone variants trained independently: ResNet-50, ResNet-101, EfficientNet-B0, Inception-V3. Final prediction is a weighted ensemble using each model's validation R² as its weight (weights ≈ 0.254 / 0.250 / 0.244 / 0.252).

## Key Techniques

- **Leakage-safe spatial features**: SpatialFeatureTransformer is fit exclusively on training rows, then applied to validation and test sets to prevent data leakage from target-encoded neighbourhood statistics.
- **Progressive backbone unfreezing**: avoids catastrophic forgetting of ImageNet weights while still adapting to satellite imagery.
- **Log-price normalisation**: target is standardised (mean/std of log-price on training set) inside the DL pipeline, with inverse transform for metric reporting.
- **GradCAM interpretability**: a GradCAMModelWrapper feeds fixed tabular features alongside the image branch to produce attention maps; the model attends to lot edges, road adjacency, and rooftop footprints.
- **SHAP analysis**: computed for both the best tabular XGBoost model and the tabular branch of the multimodal DL model; top features are `grade`, `luxury_index`, `sqft_living`, and neighbourhood price ratio features.
- **Zoom level selection**: zoom_experiment.py tested levels 18–21 on three sample locations; zoom 19 was chosen as the best balance between ground resolution and context coverage.

## Results & Findings

| Model | Val R² | Val RMSE |
|---|---|---|
| XGBoost (tabular only, 16 features) | 0.8960 | $114,260 |
| XGBoost (tabular + 30 spatial features) | 0.9136 | $104,145 |
| Image-only (ResNet-50 CNN features → Ridge) | 0.382 | $278,407 |
| Multimodal ResNet-50 (end-to-end DL) | 0.8308 | $145,713 |
| Multimodal ResNet-101 | 0.8157 | $152,060 |
| Multimodal EfficientNet-B0 | 0.7989 | $158,871 |
| Multimodal Inception-V3 | 0.8224 | $149,304 |
| **Weighted Ensemble (4 backbones)** | **0.8365** | **$143,243** |

Key insight: satellite features alone (image-only R² 0.38) are insufficient, but they add meaningful signal when fused — the ensemble closes roughly half the gap to tabular-only performance while providing additional robustness on edge cases where structured features are ambiguous.

## Challenges & Solutions

1. **Data leakage in spatial features**: KNN price features are derived from the target variable. Solved by building a `SpatialFeatureTransformer` that is fit only on training rows and applied to val/test without re-fitting.
2. **Memory pressure during CNN feature extraction**: Extracting 2048-d features for 16K images across four backbones is expensive. Solved with aggressive `gc.collect()` + `torch.cuda.empty_cache()` checkpoints between extraction passes, and chunked processing.
3. **Satellite image path portability**: Image paths were stored as absolute paths tied to a local machine. Preprocessing notebooks include path-rewriting helpers to make the CSV portable.
4. **High-value property underprediction**: Properties above $2M are systematically underpredicted (worst GradCAM error: predicted $1.49M vs actual $2.9M). At zoom 19, visual differences between a $2M and $5M home are subtle; higher zoom or aerial oblique views would help.

## What I'd Do Differently

- **Higher-resolution or multi-zoom imagery**: Fetching images at two zoom levels (19 for detail, 17 for neighbourhood context) and fusing both would give the CNN richer spatial context without losing local detail.
- **Cross-attention fusion instead of late concat**: A cross-attention module between tabular embeddings and spatial image tokens (ViT patch tokens) would let the model learn which image regions are most relevant for each tabular feature combination.
- **Contrastive pre-training on property pairs**: Pre-train the image encoder on pairs of same-neighbourhood/different-price properties using a contrastive loss before fine-tuning end-to-end.
- **Optuna tuning for all backbones**: Only basic hyperparameter configurations were tested; systematic Optuna sweeps over learning rate schedules, dropout rates, and unfreeze timing could meaningfully improve results.
- **Confidence intervals / uncertainty quantification**: An ensemble naturally allows MC Dropout or deep ensembles to produce prediction intervals — important for any real AVM use case.

## Links

GitHub: https://github.com/vanshnarang13/Multimodal-Property-Price-Prediction
