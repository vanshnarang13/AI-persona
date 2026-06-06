# Project: Amazon ML Challenge 2025 — Product Price Prediction

## Overview
Built a multimodal product price prediction pipeline for the Amazon ML Challenge 2025. The task was to predict product prices from noisy product catalog text and product image URLs, with SMAPE (Symmetric Mean Absolute Percentage Error) as the competition metric. The implementation focuses on turning raw catalog listings into cleaner multimodal representations rather than fine-tuning a large vision-language model end-to-end.

The core idea is: clean and condense the product text, pair it with the product image, extract a frozen LLaVA multimodal embedding, then train a compact PyTorch MLP regressor on top in log-price space.

## Tech Stack
Python, PyTorch, Hugging Face Transformers, LLaVA 1.5 7B (`llava-hf/llava-1.5-7b-hf`), FLAN-T5 Base (`google/flan-t5-base`), pandas, NumPy, scikit-learn, PIL/Pillow, requests, tqdm, AdamW, cosine annealing warm restarts

## Repository Structure

```
amazon-ml-challenge-2025/
├── config.py               # Central paths, model IDs, hyperparameters
├── text_preprocessing.py   # Stage A: rule-based catalog text cleaning
├── llm_condenser.py        # Stage B: FLAN-T5 summarization/condensing
├── embedding_extractor.py  # LLaVA image+text embedding extraction
├── losses.py               # Smooth SMAPE and hybrid loss
├── model.py                # MLPRegressor and PricePredictor wrapper
├── train.py                # End-to-end training and submission generation
└── requirements.txt
```

## Data Layout

The expected local data layout is:

```
dataset/
├── train.csv   # id, catalog_content, image_link, price
└── test.csv    # id, catalog_content, image_link
```

The training script reads both CSVs, extracts or loads cached embeddings, splits the training set into train/validation with a 15% validation split, trains the regressor, and writes `test_out.csv` with predicted prices.

## Pipeline Architecture

```
Raw catalog_content + image_link
        │
        ├─ Stage A: rule-based text preprocessing
        │    ├─ HTML/unicode cleanup
        │    ├─ URL and emoji removal
        │    ├─ marketing-fluff filtering
        │    ├─ whitespace/casing normalization
        │    └─ light signal extraction: IPQ / pack quantity, bullet points
        │
        ├─ Stage B: FLAN-T5 condensing
        │    └─ short 1-2 bullet summary preserving product name, value, units
        │
        ├─ Image download layer
        │    ├─ 64-worker parallel image fetching
        │    └─ gray placeholder image fallback for broken URLs
        │
        ├─ Frozen LLaVA multimodal encoder
        │    └─ mean-pool final hidden states into 4096-d embedding
        │
        └─ PyTorch MLP regressor
             └─ log-price prediction → inverse transform → clipped nonnegative price
```

## Stage A: Rule-Based Text Preprocessing

`TextPreprocessor` cleans the raw `catalog_content` before any LLM step. It:
- Unescapes HTML entities
- Normalizes unicode with NFKD
- Removes URLs and emoji artifacts
- Removes common marketing filler like "buy now", "shop now", "limited time offer", and similar phrases
- Normalizes whitespace and lowercases the text
- Extracts item pack quantity patterns such as `IPQ`, `Item Pack Quantity`, `Pack of`, and `Quantity`
- Extracts bullet point fragments from catalog text

This stage handles the low-cost deterministic cleanup first so the model is not spending context on junk tokens.

## Stage B: FLAN-T5 Text Condensing

The repo uses `google/flan-t5-base` as a text condenser. The prompt asks the model to summarize product catalog data into 1-2 concise bullet points, preserving product names, values, and units when available. Inputs are truncated to 512 tokens for generation, with up to 150 new tokens, two-beam decoding, and deterministic generation (`do_sample=False`).

This is not used as the final predictor. It is a preprocessing step that compresses long, noisy product descriptions into a smaller text representation before LLaVA embedding extraction.

## Multimodal Embedding Extraction

`EmbeddingExtractor` loads `llava-hf/llava-1.5-7b-hf` through `LlavaForConditionalGeneration` and `AutoProcessor`. For each product, it builds a prompt:

```
<image>
Describe this product: {processed_text}
```

The product image and condensed text are passed through frozen LLaVA. The final hidden states are mean-pooled across the sequence dimension to produce a 4096-dimensional multimodal embedding per product. Embeddings are cached to `.npy` files in `embeddings/`, so expensive LLaVA extraction only needs to run once per split.

The image pipeline is robust to bad URLs: if an image request fails or cannot be decoded, the extractor uses a gray 224x224 placeholder image instead of crashing the run.

## Regressor Model

The predictor is a feed-forward PyTorch MLP:

```
4096 → 2048 → 1024 → 512 → 256 → 128 → 1
```

Each hidden layer uses Linear + ReLU + Dropout. Dropout is configured at 0.3. The final scalar output is a transformed price. By default, the model trains in log-price space using:

```
log(price + 1.0)
```

At inference, predictions are inverse-transformed with `exp(y) - 1.0` and clipped to a minimum price of `0.01`.

## Loss Function and Metric

The repo defines:
- `SMAPELoss`: differentiable smooth SMAPE using a smooth absolute value
- `HybridLoss`: `0.7 × smooth SMAPE + 0.3 × normalized MSE`
- `calculate_smape`: evaluation helper that reports SMAPE in original price space

The default training objective is the hybrid loss. That keeps the training signal aligned with the competition's SMAPE metric while using normalized MSE to stabilize regression in log-price space.

## Training Loop

`train.py` runs the end-to-end flow:
1. Load `train.csv` and `test.csv`
2. Extract or load cached train/test embeddings
3. Transform training prices into log space
4. Split train/validation with `VAL_SPLIT = 0.15` and `random_state = 42`
5. Train with AdamW (`lr=1e-4`, `weight_decay=1e-4`)
6. Use cosine annealing warm restarts (`T_0=10`, `T_mult=2`)
7. Apply gradient clipping at norm `1.0`
8. Track validation SMAPE in original price space
9. Save `checkpoints/best_model.pt` whenever validation SMAPE improves by at least `0.001`
10. Stop early after 15 non-improving epochs
11. Generate `test_out.csv`

## Key Techniques

**Two-stage text cleanup**: deterministic text normalization handles formatting noise, while FLAN-T5 compresses high-variance catalog descriptions into dense feature text.

**Frozen multimodal encoder**: LLaVA is used as a general-purpose image+text representation model. Training only the MLP regressor avoids the cost and instability of fine-tuning a 7B vision-language model for a competition pipeline.

**Embedding cache**: `.npy` embedding caching separates expensive feature extraction from fast regressor iteration.

**Log-price training**: product prices are skewed, so the model learns in log space and evaluates after inverse transform.

**SMAPE-aligned optimization**: the hybrid loss combines smooth SMAPE with normalized MSE, matching the competition metric while avoiding purely percentage-based gradient instability.

**Fault-tolerant image ingestion**: broken product image URLs are handled through a placeholder image fallback, keeping the batch pipeline from failing on noisy marketplace data.

## Results & Findings

The repository is structured to report validation SMAPE during training and to save the best checkpoint based on validation SMAPE. The code does not include a checked-in final leaderboard result or saved training log, so the documented result should be treated as implementation-focused rather than a fixed benchmark claim.

The main engineering finding is that product price prediction benefits from compressing messy catalog text before multimodal encoding. The pipeline explicitly avoids feeding long, noisy product descriptions directly into the multimodal encoder.

## Challenges & Solutions

**Noisy catalog text**: Marketplace descriptions contain inconsistent formatting, repeated marketing phrases, URLs, symbols, and variable-length product details. The solution was a two-stage text pipeline: rule-based cleanup first, then FLAN-T5 condensation.

**Expensive multimodal inference**: LLaVA 1.5 7B embedding extraction is the slowest step. The solution was to cache train/test embeddings to disk and iterate only on the MLP after extraction.

**Broken image URLs**: Product image links can fail. The extractor catches failures and substitutes a placeholder image so the pipeline remains batch-safe.

**Skewed regression target**: Raw prices have high variance. Training in log-price space makes optimization more stable while still reporting SMAPE on original prices.

**Competition metric mismatch**: Standard MSE is not aligned with SMAPE. The repo implements a differentiable SMAPE-style loss and combines it with normalized MSE.

## What I'd Do Differently

1. Add reproducible experiment logging with saved validation SMAPE curves, final checkpoint metadata, and exact run configs.
2. Add ablations comparing raw text, Stage A only, Stage A + FLAN-T5, image-only, text-only, and full image+text LLaVA embeddings.
3. Replace mean pooling over LLaVA hidden states with a more deliberate pooling strategy, such as CLS-like token pooling or attention pooling over image/text token subsets.
4. Add category-aware modeling if category labels or reliable category extraction are available, because price dynamics vary heavily across product types.
5. Use quantization or a smaller vision-language encoder for faster embedding extraction when GPU memory is limited.

## Links
GitHub: https://github.com/oyaah/amazon-ml-challenge-2025
