# Project: Stock Sentiment Analysis and Algorithmic Trading Strategy

## Overview

This project builds an end-to-end pipeline that takes news headlines about Apple (AAPL) and Tesla (TSLA), extracts sentiment features from those headlines using VADER and TextBlob, trains a Linear Discriminant Analysis classifier to predict next-day price movement direction, and then backtests an algorithmic trading strategy using those predictions as buy and sell signals.

The backtesting covers ten years of historical OHLC data from Yahoo Finance, with training on data from before 2020 and testing on 2020 onwards. The simulation includes stop-loss and take-profit controls and tracks portfolio value, win ratio, Sharpe ratio, and maximum drawdown.

GitHub: https://github.com/vanshnarang13/Stock-Sentiment-Analysis

## Tech Stack

Python, pandas, NumPy, Scikit-Learn (LinearDiscriminantAnalysis, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay), VADER (vaderSentiment, SentimentIntensityAnalyzer), TextBlob, NLTK (WordNetLemmatizer), yfinance, Matplotlib, Jupyter Notebook.

## Dataset

Two headline datasets: apple_dataset.csv and tesla_dataset.csv, each containing news headlines with binary labels for stock movement (1 for price went up the next day, 0 for price went down). Yahoo Finance historical OHLC data fetched via yfinance.download() for the period from 2014-01-01 to 2024-01-01 for both stocks. Temporal train and test split at January 1, 2020.

## Pipeline in Detail

### Text Preprocessing

Headlines are lowercased, punctuation is removed using regex substitution, and words are lemmatized using NLTK's WordNetLemmatizer. Lemmatization reduces words to their base form (running becomes run, beaten becomes beat) using the WordNet lexical database. This reduces vocabulary size and groups inflected forms of the same word.

### Sentiment Feature Extraction

Six features are extracted per headline: TextBlob polarity (a float from negative one to positive one measuring the overall positive or negative sentiment), TextBlob subjectivity (a float from zero to one measuring how opinion-based versus factual the text is), and four VADER scores (compound score from negative one to positive one, positive score, negative score, and neutral score as probabilities summing to one).

VADER is specifically designed for short texts and social media content. It uses a hand-crafted lexicon and rule set that handles negations ("not good" is negative), intensifiers ("very good" scores higher than "good"), and punctuation emphasis ("GREAT!!!" scores higher than "great"). TextBlob uses a different pattern-matching approach based on a pre-built lexicon.

Using both gives six features that capture complementary aspects of sentiment. VADER's compound score and TextBlob's polarity are both overall sentiment measures but computed via different methods, so they are not perfectly correlated and both add signal.

### Classification

A LinearDiscriminantAnalysis model is trained on the combined Apple and Tesla training data using all six sentiment features. LDA was chosen over logistic regression or a tree-based model for a specific reason: the six sentiment features from VADER and TextBlob are correlated by design, since both measure the same underlying signal (positive or negative sentiment). Correlated features cause problems for models that assume feature independence, but LDA explicitly models the covariance structure of the feature space and handles multicollinearity well. It finds the linear combination of features that maximally separates the two classes.

The model is tested separately on the Apple and Tesla test sets (2020 onwards).

### Backtesting

The backtesting simulation uses classifier predictions as trading signals. When the model predicts price goes up, the strategy buys 100 shares. When it predicts price goes down, it either sells existing holdings or stays out.

Risk controls: stop-loss at 5% below the entry price (exits the position if the stock drops 5% from where we bought), and take-profit at 10% above the entry price (exits to lock in gains). Position sizing is fixed at 100 shares per trade.

The simulation tracks total portfolio value over time, calculates the Sharpe ratio (annualized return minus risk-free rate divided by annualized volatility), maximum drawdown (largest peak-to-trough decline), and win ratio (fraction of trades that were profitable).

## Results

Classification performance on the 2020-onwards test set:

Apple (AAPL): 97.0% accuracy, 1.00 precision, 0.95 recall, 0.97 F1 score, 0.97 ROC AUC.
Tesla (TSLA): 97.1% accuracy, 1.00 precision, 0.94 recall, 0.97 F1 score, 0.97 ROC AUC.

Backtesting results starting with $10,000 initial capital:

Apple: final portfolio value $165,400, total return 1537%, annualized return 103.7%, Sharpe ratio 5.92, max drawdown -9.59%, win ratio 0.81 across 1,034 trades.
Tesla: final portfolio value $534,122, total return 5221%, annualized return 178.9%, Sharpe ratio 5.54, max drawdown -7.25%, win ratio 0.86 across 994 trades.

A Sharpe ratio above 5 is exceptionally high. In real markets, Sharpe ratios above 2 are considered excellent. These results reflect a combination of the model's genuine predictive ability on this dataset and the favorable market conditions for both stocks during the 2020 to 2024 period (both AAPL and TSLA had substantial bull runs during this period).

## Important Caveats

The results look impressive but several real-world factors are not modeled. There are no transaction costs. In real trading, every buy and sell incurs brokerage fees and bid-ask spread costs. With 1,034 trades over four years, even small per-trade costs compound substantially. There is also no slippage modeling. The simulation assumes you can always execute at the close price, which is unrealistic for any meaningful position size.

The Sharpe ratios are likely inflated by these missing costs and by look-ahead bias in the dataset (news headlines and price labels in datasets like these sometimes have subtle temporal alignment issues).

## Challenges and How They Were Solved

Batch preprocessing across four dataset splits (training Apple, training Tesla, test Apple, test Tesla) was handled by concatenating all four splits with a section label column before running text preprocessing, then re-separating afterward. This avoided running the same preprocessing code four times separately and ensured consistent tokenization and lemmatization across all splits.

The choice of LDA over more complex classifiers was a deliberate call. Given six correlated features and a binary classification task, LDA is both faster and theoretically better suited than a random forest or XGBoost model that would need to redundantly explore correlated feature combinations.

## What I Would Do Differently

The most significant upgrade would be replacing LDA and the handcrafted sentiment features with a fine-tuned FinBERT or similar financial domain BERT model. Lexicon-based sentiment analyzers like VADER miss contextual nuances that are crucial in financial text. The phrase "beats expectations" has positive market implications. The phrase "meets expectations" is neutral to slightly negative. "Misses expectations" is negative. VADER cannot distinguish these because the words "beats" and "meets" do not have inherently different sentiment polarities in a general-purpose lexicon. A model trained on financial text would understand the domain-specific meaning.

I would also add proper transaction cost and slippage modeling to the backtester. The current returns are too clean to be realistic. Modeling a 0.1% round-trip transaction cost (buy plus sell) and a 0.05% slippage assumption would significantly reduce the reported Sharpe ratios and make the results more interpretable as what they would look like in practice.

Finally, I would add short-selling capability. The current strategy only goes long (buys when the signal is positive, stays out when it is negative). Allowing the strategy to short the stock when the signal is negative would roughly double the number of actionable signals and better utilize the model's predictive power in both directions.
