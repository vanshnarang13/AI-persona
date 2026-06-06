# GitHub README: Stock Sentiment Analysis

Repository: https://github.com/vanshnarang13/Stock-Sentiment-Analysis

# Stock Sentiment Analysis & Algorithmic Trading Strategy

> NLP-driven stock movement prediction for Apple and Tesla using VADER + TextBlob sentiment analysis on news headlines, classified with Linear Discriminant Analysis, and validated through a backtested algorithmic trading strategy with stop-loss and take-profit controls.

---

## Overview

Stock price movements are influenced heavily by news sentiment. This project builds an end-to-end pipeline that: (1) processes news headline datasets for Apple and Tesla, (2) extracts sentiment signals using VADER and TextBlob, (3) trains a Linear Discriminant Analysis (LDA) classifier to predict next-day price movement direction (up/down), and (4) back-tests a complete trading strategy using those predictions on 10 years of historical OHLC data from Yahoo Finance.

The key deliverable is not just a classifier — it is a full trading simulation with stop-loss (5%), take-profit (10%), win ratio, Sharpe ratio, and portfolio growth metrics, demonstrating whether sentiment signals translate into real alpha.

---

## Dataset

| Dataset | Description |
|---|---|
| `apple_dataset.csv` | Apple news headlines with binary stock movement labels (1 = up, 0 = down) |
| `tesla_dataset.csv` | Tesla news headlines with binary stock movement labels (1 = up, 0 = down) |
| Yahoo Finance (via `yfinance`) | 10 years of daily OHLC data for AAPL and TSLA (2014–2024) |

**Temporal split:**
- Train: headlines before 2020-01-01
- Test: headlines from 2020-01-01 onwards

---

## Approach / Methodology

### 1. Data Loading & Merging
News headline datasets are loaded for both companies and merged with historical OHLC data from Yahoo Finance on the `Date` column, giving each headline a corresponding set of market prices.

### 2. Text Preprocessing
- Lowercasing
- Punctuation removal with regex (`re.sub`)
- Lemmatization using `WordNetLemmatizer` (NLTK)
- Combined all four splits into a single DataFrame for efficient batch preprocessing, then re-split

### 3. Sentiment Feature Extraction
Six sentiment features are extracted per headline:

| Feature | Source | Description |
|---|---|---|
| `Polarity` | TextBlob | Sentiment polarity (-1.0 to +1.0) |
| `Subjectivity` | TextBlob | Subjectivity score (0.0 to 1.0) |
| `Compound` | VADER | Overall normalized sentiment score |
| `Positive` | VADER | Proportion of positive tokens |
| `Negative` | VADER | Proportion of negative tokens |
| `Neutral` | VADER | Proportion of neutral tokens |

### 4. Classification
A **Linear Discriminant Analysis (LDA)** model is trained on the combined Apple + Tesla training set (pre-2020) using all six sentiment features. LDA was chosen for its interpretability and efficiency on low-dimensional feature sets.

### 5. Backtesting & Trading Strategy
Predicted buy/sell signals are merged with historical OHLC data. A rule-based trading simulator applies:
- **Buy signal (label = 1):** Purchase 100 shares at next open price
- **Sell signal (label = 0):** Sell all held shares at next open price
- **Stop-loss:** Auto-sell if position drops 5% below buy price
- **Take-profit:** Auto-sell if position gains 10% above buy price
- Initial capital: $10,000

---

## Tech Stack

| Category | Libraries / Tools |
|---|---|
| Sentiment analysis | `vaderSentiment` (VADER SentimentIntensityAnalyzer), `TextBlob` |
| NLP preprocessing | `nltk` (WordNetLemmatizer), `re` |
| Classification | `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` |
| Market data | `yfinance` |
| Evaluation metrics | `sklearn.metrics` (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay) |
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `matplotlib.dates` |

---

## Results

### Classification Performance (Test Set: 2020 onwards)

| Stock | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Apple (AAPL) | 97.0% | 1.00 | 0.95 | 0.97 | 0.97 |
| Tesla (TSLA) | 97.1% | 1.00 | 0.94 | 0.97 | 0.97 |

### Backtesting Results (Initial capital: $10,000)

| Metric | Apple | Tesla |
|---|---|---|
| Final Portfolio Value | $165,400 | $534,122 |
| Total Return | 1,537% | 5,221% |
| Annualized Return | 103.7% | 178.9% |
| Sharpe Ratio | 5.92 | 5.54 |
| Max Drawdown | -9.59% | -7.25% |
| Win Ratio | 0.81 | 0.86 |
| Total Trades | 1,034 | 994 |

---
Historical market data is fetched live from Yahoo Finance via `yfinance`.
